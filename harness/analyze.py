#!/usr/bin/env python3
"""
Analysis for the screening-rigor study.

Reads every raw response file, computes:
  - rejection rate per (condition, label) with 95% Wilson interval
  - rejection rate per (condition, candidate) for the CLEARLY_CAPABLE group
  - unparseable count per condition

The primary preregistered comparison is the CLEARLY_CAPABLE row across
conditions. Everything else is context.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / json.load(open(HERE / "config.json"))["out_dir"] / "raw"

LABEL_ORDER = ["CLEARLY_CAPABLE", "BORDERLINE_CAPABLE", "NOT_CLEARLY_CAPABLE"]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load():
    rows = []
    for path in sorted(RAW.glob("*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(f"No raw files found in {RAW}")
    return rows


def fmt(k, n):
    if n == 0:
        return "   —   "
    lo, hi = wilson(k, n)
    return f"{k / n:5.1%} [{lo:4.0%}–{hi:4.0%}] n={n}"


def main():
    rows = load()
    conditions = sorted({r["condition"] for r in rows})

    # --- unparseable per condition ---------------------------------------
    print("\nUnparseable responses per condition")
    for c in conditions:
        sub = [r for r in rows if r["condition"] == c]
        bad = sum(1 for r in sub if not r["parseable"])
        print(f"  condition {c}: {bad}/{len(sub)}")

    parsed = [r for r in rows if r["parseable"]]

    # --- rejection rate per (condition, label) ----------------------------
    print("\nRejection rate by condition × label  (95% Wilson CI)")
    header = "  label".ljust(24) + "".join(f"cond {c}".ljust(28) for c in conditions)
    print(header)
    for lab in LABEL_ORDER:
        line = f"  {lab}".ljust(24)
        for c in conditions:
            sub = [r for r in parsed if r["condition"] == c and r["label"] == lab]
            k = sum(1 for r in sub if r["verdict"] == "REJECT")
            line += fmt(k, len(sub)).ljust(28)
        print(line)

    # --- per-candidate breakdown for the primary group --------------------
    print("\nCLEARLY_CAPABLE — rejection rate per candidate")
    by_cand = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in parsed:
        if r["label"] != "CLEARLY_CAPABLE":
            continue
        cell = by_cand[r["candidate_id"]][r["condition"]]
        cell[1] += 1
        if r["verdict"] == "REJECT":
            cell[0] += 1
    print("  candidate".ljust(14) + "".join(f"cond {c}".ljust(14) for c in conditions))
    for cid in sorted(by_cand):
        line = f"  {cid}".ljust(14)
        for c in conditions:
            k, n = by_cand[cid][c]
            line += (f"{k}/{n}" if n else "—").ljust(14)
        print(line)

    # --- primary comparison: each condition vs condition 1 ----------------
    print("\nPrimary comparison — CLEARLY_CAPABLE rejection, condition vs. condition 1")
    base = [r for r in parsed if r["condition"] == conditions[0]
            and r["label"] == "CLEARLY_CAPABLE"]
    bk, bn = sum(r["verdict"] == "REJECT" for r in base), len(base)
    bp = bk / bn if bn else float("nan")
    for c in conditions[1:]:
        sub = [r for r in parsed if r["condition"] == c
               and r["label"] == "CLEARLY_CAPABLE"]
        k, n = sum(r["verdict"] == "REJECT" for r in sub), len(sub)
        p = k / n if n else float("nan")
        print(f"  cond {c} vs cond {conditions[0]}: "
              f"{p:.1%} vs {bp:.1%}  (Δ = {p - bp:+.1%})")
    print()


if __name__ == "__main__":
    main()
