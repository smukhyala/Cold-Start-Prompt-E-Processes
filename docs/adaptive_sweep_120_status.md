# Adaptive SPRUCE 120-Task Status

This file tracks `adaptive_spruce_120`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| adaptive_spruce_120 | done | 120/120 | logs/adaptive_sweep_120/webarena_gmail_adaptive_spruce_120_trial0_2026-06-20T23-35-23Z_FULL.jsonl | reports/adaptive_sweep/spruce_120 |

## Headline

- Durable tasks: 120/120
- Total successes: 72/120 (60.0%)
- Total cost: $6.0562
- Total wall time: 116.2 min
- Final global log-e: -0.184
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| balanced | 15 | 11 | 73.3% |
| planner | 12 | 8 | 66.7% |
| domain_expert | 11 | 7 | 63.6% |
| verifier | 11 | 5 | 45.5% |
| baseline | 10 | 6 | 60.0% |
| cautious | 10 | 6 | 60.0% |
| rapid | 10 | 6 | 60.0% |
| algorithmic | 9 | 5 | 55.6% |
| exploratory | 8 | 5 | 62.5% |
| explorer | 8 | 4 | 50.0% |
| junior_reactive | 8 | 4 | 50.0% |
| overthinker | 8 | 5 | 62.5% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| balanced | 15 | 11 | 0.733 | 1.282 | 0.369 | 0.985 | False |
| planner | 12 | 8 | 0.667 | -0.351 | 0.246 | 0.985 | False |
| domain_expert | 11 | 7 | 0.636 | -0.309 | 0.215 | 0.985 | False |
| verifier | 11 | 5 | 0.455 | 0.000 | 0.077 | 0.877 | False |
| baseline | 10 | 6 | 0.600 | -0.813 | 0.185 | 0.985 | False |
| cautious | 10 | 6 | 0.600 | -0.632 | 0.169 | 0.985 | False |
| rapid | 10 | 6 | 0.600 | -0.120 | 0.169 | 0.985 | False |
| algorithmic | 9 | 5 | 0.556 | -0.869 | 0.123 | 0.985 | False |
| exploratory | 8 | 5 | 0.625 | -0.874 | 0.108 | 0.985 | False |
| explorer | 8 | 4 | 0.500 | -0.800 | 0.092 | 0.969 | False |
| junior_reactive | 8 | 4 | 0.500 | -1.118 | 0.015 | 0.938 | False |
| overthinker | 8 | 5 | 0.625 | -0.755 | 0.123 | 0.985 | False |

## Policy

- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 120 task executions over two cycles of the 60-task Gmail bank.
