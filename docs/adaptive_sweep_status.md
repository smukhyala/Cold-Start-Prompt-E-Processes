# Adaptive Sweep Status

This file tracks the 12-arm adaptive WebArena Gmail run. Unlike the uniform
paired sweep, this run has a single 60-task stream and uses the e-process-aware
SPRUCE policy to choose which prompt arm to pull at each task.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| adaptive_spruce | done | 60/60 | logs/adaptive_sweep/webarena_gmail_adaptive_spruce_60_trial0_2026-06-19T21-46-42Z_FULL.jsonl | reports/adaptive_sweep/spruce_60 |

## Headline

- Durable tasks: 60/60
- Total successes: 38/60 (63.3%)
- Total cost: $3.2436
- Total wall time: 65.8 min
- Final global log-e: 0.152
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| baseline | 8 | 6 | 75.0% |
| planner | 8 | 7 | 87.5% |
| exploratory | 7 | 5 | 71.4% |
| algorithmic | 5 | 3 | 60.0% |
| rapid | 5 | 3 | 60.0% |
| balanced | 4 | 2 | 50.0% |
| cautious | 4 | 3 | 75.0% |
| domain_expert | 4 | 2 | 50.0% |
| explorer | 4 | 2 | 50.0% |
| overthinker | 4 | 1 | 25.0% |
| verifier | 4 | 2 | 50.0% |
| junior_reactive | 3 | 2 | 66.7% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 8 | 6 | 0.750 | 0.827 | 0.262 | 0.985 | False |
| planner | 8 | 7 | 0.875 | 1.740 | 0.323 | 0.985 | False |
| exploratory | 7 | 5 | 0.714 | 0.236 | 0.169 | 0.985 | False |
| algorithmic | 5 | 3 | 0.600 | -0.575 | 0.015 | 0.985 | False |
| rapid | 5 | 3 | 0.600 | -0.687 | 0.015 | 0.985 | False |
| balanced | 4 | 2 | 0.500 | -0.706 | 0.015 | 0.985 | False |
| cautious | 4 | 3 | 0.750 | -0.389 | 0.015 | 0.985 | False |
| domain_expert | 4 | 2 | 0.500 | -0.706 | 0.015 | 0.985 | False |
| explorer | 4 | 2 | 0.500 | -0.706 | 0.015 | 0.985 | False |
| overthinker | 4 | 1 | 0.250 | -0.693 | 0.015 | 0.985 | False |
| verifier | 4 | 2 | 0.500 | -0.706 | 0.015 | 0.985 | False |
| junior_reactive | 3 | 2 | 0.667 | -0.693 | 0.015 | 0.985 | False |

## Policy

- Arms: all 12 prompt arms from `configs/arms_initial.yaml`.
- Sampler: `spruce`.
- Evidence signal: per-arm upward e-process log-wealth against `m0 = 0.5`.
- Global evidence: convex linear mixture over all per-arm e-processes.
- Warm start: one pull per arm, then adaptive allocation.
- Tie behavior: deterministic first-arm tie break, so resume paths are stable.

## Metrics To Compare Against Uniform

- Successes and success rate over the 60 adaptive tasks.
- Pull allocation by arm.
- First global rejection time, if any.
- Per-arm first rejection times.
- Total token cost and total wall-clock time.
- Token cost and wall-clock time to first rejection.
- Best-arm identification path over time.

## Live Commands

```bash
tail -f logs/adaptive_sweep/adaptive_stdout.log
tail -f logs/adaptive_sweep/watchdog_adaptive_spruce.log
screen -r csp_adaptive_spruce
```
