#!/usr/bin/env python3
"""
Screening-rigor study harness.

For every (candidate, condition, run) cell:
  - build the prompt from frozen files
  - call the model with a fresh instance
  - save the full raw response as its own JSON file
  - parse the verdict from the required FINAL: line

Resumable: cells whose output file already exists are skipped.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

VERDICT_RE = re.compile(r"^\s*FINAL:\s*(ADVANCE|REJECT)\s*$")


# ---------------------------------------------------------------- loading ---

def load_config():
    with open(HERE / "config.json") as f:
        return json.load(f)


def load_text(name):
    return (HERE / name).read_text().strip()


def load_conditions(ids):
    return {i: load_text(f"conditions/condition_{i}.txt") for i in ids}


def load_labels():
    labels = {}
    with open(HERE / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            cid = row["candidate_id"].strip()
            lab = row["label"].strip()
            if not lab:
                sys.exit(f"labels.csv: candidate {cid} has no label. Fill it in first.")
            labels[cid] = lab
    return labels


def load_candidates():
    """Split candidate-profiles.md on '### Cxx' headers. Returns {id: text}."""
    text = load_text("candidate-profiles.md")
    parts = re.split(r"^### (C\d{2}) — ", text, flags=re.MULTILINE)
    # parts = [preamble, id, body, id, body, ...]
    candidates = {}
    for i in range(1, len(parts), 2):
        cid = parts[i]
        body = parts[i + 1].split("\n---")[0].strip()
        candidates[cid] = body
    if not candidates:
        sys.exit("No candidates parsed from candidate-profiles.md")
    return candidates


# ---------------------------------------------------------------- prompt ----

def build_user_prompt(jd, resume, instruction):
    return (
        "JOB DESCRIPTION\n"
        "===============\n"
        f"{jd}\n\n"
        "CANDIDATE RÉSUMÉ\n"
        "================\n"
        f"{resume}\n\n"
        "SCREENING INSTRUCTION\n"
        "=====================\n"
        f"{instruction}\n"
    )


def parse_verdict(text):
    """Return 'ADVANCE', 'REJECT', or None. Checks the last 3 non-empty lines."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines[-3:]):
        m = VERDICT_RE.match(line)
        if m:
            return m.group(1)
    return None


# ------------------------------------------------------------ model call ----

def call_model(cfg, system, user):
    """
    The single seam for swapping providers. Returns (text, usage_dict).
    Fresh client per call is deliberate: nothing carries between runs.
    """
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return text, usage


def call_with_retry(cfg, system, user, attempts=5):
    delay = 2.0
    for i in range(attempts):
        try:
            return call_model(cfg, system, user)
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1} after error: {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2


# ----------------------------------------------------------------- cells ----

def cell_path(out_raw, cid, cond, run):
    return out_raw / f"{cid}_cond{cond}_run{run:03d}.json"


def run_cell(cfg, system, jd, cid, resume, label, cond, instruction, run, out_raw):
    path = cell_path(out_raw, cid, cond, run)
    if path.exists():
        return "skip"

    user = build_user_prompt(jd, resume, instruction)
    started = datetime.now(timezone.utc).isoformat()
    text, usage = call_with_retry(cfg, system, user)
    verdict = parse_verdict(text)

    record = {
        "candidate_id": cid,
        "label": label,
        "condition": cond,
        "run": run,
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "parseable": verdict is not None,
        "usage": usage,
        "response": text,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    tmp.rename(path)
    return verdict or "UNPARSEABLE"


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="condition 1 only, pilot_runs runs, one candidate per label")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one full prompt and exit; no API calls")
    args = ap.parse_args()

    cfg = load_config()
    system = load_text("system_prompt.txt")
    jd = load_text("job_description.md")
    labels = load_labels()
    candidates = load_candidates()
    conditions = load_conditions(cfg["conditions"])

    missing = set(candidates) - set(labels)
    if missing:
        sys.exit(f"Candidates without labels: {sorted(missing)}")

    if args.pilot:
        # one candidate per label, first seen in file order
        seen, keep = set(), []
        for cid in candidates:
            if labels[cid] not in seen:
                seen.add(labels[cid])
                keep.append(cid)
        candidates = {c: candidates[c] for c in keep}
        conditions = {1: conditions[1]}
        runs = cfg["pilot_runs"]
    else:
        runs = cfg["runs_per_cell"]

    if args.dry_run:
        cid = next(iter(candidates))
        cond = next(iter(conditions))
        print("=== SYSTEM ===\n" + system)
        print("\n=== USER ===\n" + build_user_prompt(jd, candidates[cid], conditions[cond]))
        return

    out_raw = HERE / cfg["out_dir"] / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    jobs = [
        (cid, cond, run)
        for cid in candidates
        for cond in conditions
        for run in range(1, runs + 1)
    ]
    total = len(jobs)
    print(f"{total} cells ({len(candidates)} candidates × "
          f"{len(conditions)} conditions × {runs} runs)")

    counts = {"ADVANCE": 0, "REJECT": 0, "UNPARSEABLE": 0, "skip": 0}
    done = 0

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {
            pool.submit(
                run_cell, cfg, system, jd, cid, candidates[cid], labels[cid],
                cond, conditions[cond], run, out_raw,
            ): (cid, cond, run)
            for cid, cond, run in jobs
        }
        for fut in as_completed(futures):
            cid, cond, run = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {cid} cond{cond} run{run}: {e}", file=sys.stderr)
                continue
            counts[result] += 1
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total}  {counts}")

    print("done:", counts)


if __name__ == "__main__":
    main()
