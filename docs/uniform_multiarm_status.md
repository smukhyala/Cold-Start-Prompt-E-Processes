# Uniform Multi-Arm Status

This file tracks the 12-arm, 60-task non-adaptive WebArena Gmail baseline. It
uses the same total task budget as the adaptive run, but allocates round-robin
across all arms.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| uniform_multiarm | done | 60/60 | logs/uniform_multiarm/webarena_gmail_uniform_multiarm_60_trial0_2026-06-19T19-27-58Z_FULL.jsonl | reports/uniform_multiarm/round_robin_60 |

## Headline

- Durable tasks: 60/60
- Total successes: 39/60 (65.0%)
- Total cost: $3.3710
- Total wall time: 71.5 min
- Final global log-e: 0.475
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| algorithmic | 5 | 4 | 80.0% |
| balanced | 5 | 4 | 80.0% |
| baseline | 5 | 5 | 100.0% |
| cautious | 5 | 2 | 40.0% |
| domain_expert | 5 | 3 | 60.0% |
| exploratory | 5 | 4 | 80.0% |
| explorer | 5 | 3 | 60.0% |
| junior_reactive | 5 | 2 | 40.0% |
| overthinker | 5 | 3 | 60.0% |
| planner | 5 | 5 | 100.0% |
| rapid | 5 | 3 | 60.0% |
| verifier | 5 | 1 | 20.0% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| algorithmic | 5 | 4 | 0.800 | 0.523 | 0.138 | 0.985 | False |
| balanced | 5 | 4 | 0.800 | 0.523 | 0.138 | 0.985 | False |
| baseline | 5 | 5 | 1.000 | 1.622 | 0.262 | 0.985 | False |
| cautious | 5 | 2 | 0.400 | -0.693 | 0.015 | 0.985 | False |
| domain_expert | 5 | 3 | 0.600 | -0.575 | 0.015 | 0.985 | False |
| exploratory | 5 | 4 | 0.800 | 0.523 | 0.138 | 0.985 | False |
| explorer | 5 | 3 | 0.600 | -0.575 | 0.015 | 0.985 | False |
| junior_reactive | 5 | 2 | 0.400 | -0.693 | 0.015 | 0.985 | False |
| overthinker | 5 | 3 | 0.600 | -1.082 | 0.015 | 0.985 | False |
| planner | 5 | 5 | 1.000 | 1.622 | 0.262 | 0.985 | False |
| rapid | 5 | 3 | 0.600 | -0.439 | 0.015 | 0.985 | False |
| verifier | 5 | 1 | 0.200 | 0.000 | 0.015 | 0.908 | False |

## Policy

- Arms: all 12 prompt arms from `configs/arms_initial.yaml`.
- Sampler: `uniform`, `round_robin`.
- Budget: 60 total tasks, 5 pulls per arm.
- Evidence: same upward per-arm e-process and global linear mixture as adaptive.

## Live Commands

```bash
tail -f logs/uniform_multiarm/uniform_stdout.log
tail -f logs/uniform_multiarm/watchdog_uniform_multiarm.log
screen -r csp_uniform_multiarm
```
