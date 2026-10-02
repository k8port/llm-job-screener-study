# LLM Screener Study — Harness

Runs the preregistered study: 12 candidate profiles × 5 conditions × N runs,
each a fresh model call, each verdict parsed from a required `FINAL:` line.

## Layout

```
harness/
  run_study.py          # sends prompts, stores raw responses, parses verdicts
  analyze.py            # rejection rates, CIs, unparseable counts
  config.json           # model, temperature, runs per cell, workers
  labels.csv            # candidate_id,label  — YOU fill this in
  job_description.md    # the frozen JD text
  candidate-profiles.md # the frozen 12 profiles
  conditions/
    condition_1.txt ... condition_5.txt
  system_prompt.txt     # constant wrapper around every call
  out/
    raw/                # one JSON per run — source of truth
```

## Setup

```
pip install anthropic
cp harness/.env.example harness/.env
```

Fill in the values in `harness/.env`. `ANTHROPIC_WORKSPACE_ID` is required when
the API key is not scoped to a specific workspace. Find the workspace ID in the
Anthropic Console settings. The harness loads this file automatically.

Fill `labels.csv` from your registered labels. Verify every file under
`conditions/` and `system_prompt.txt` matches your OSF registration
word-for-word — the harness sends exactly what's in those files.

## Pilot

Ten runs, condition 1 only, one candidate per label:

```
python run_study.py --pilot
```

Check `out/raw/` by hand. Confirm the model emits `FINAL:` lines, picks
ADVANCE on the strong profile, and nothing looks broken. Then pilot again.

## Full run

```
python run_study.py
```

Resumable — rerun the same command after a crash and it skips completed cells.

## Analyze

```
python analyze.py
```

Prints rejection rate per condition per label with 95% Wilson intervals,
and unparseable counts per condition.

## Protocol notes

- `system_prompt.txt` is constant across all conditions. It is part of the
  stimulus and should be treated as frozen.
- Temperature is a protocol choice. Record it in the writeup.
- Model name is recorded in every raw file. Pin a dated model version.

See OSF registration documents at https://osf.io/832yx
