# Adaptive SPRUCE 120-Task rep3 Status

This file tracks `adaptive_spruce_120_rep3`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| adaptive_spruce_120_rep3 | done | 120/120 | logs/adaptive_sweep_120_rep3/webarena_gmail_adaptive_spruce_120_rep3_trial0_2026-06-22T02-26-11Z_FULL.jsonl | reports/adaptive_sweep/spruce_120_rep3 |

## Headline

- Durable tasks: 120/120
- Total successes: 68/120 (56.7%)
- Total cost: $5.0521
- Total wall time: 155.1 min
- Final global log-e: -0.026
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| explorer | 16 | 12 | 75.0% |
| baseline | 13 | 9 | 69.2% |
| algorithmic | 11 | 7 | 63.6% |
| rapid | 11 | 5 | 45.5% |
| verifier | 11 | 6 | 54.5% |
| exploratory | 10 | 6 | 60.0% |
| junior_reactive | 9 | 5 | 55.6% |
| balanced | 8 | 3 | 37.5% |
| domain_expert | 8 | 4 | 50.0% |
| overthinker | 8 | 5 | 62.5% |
| planner | 8 | 3 | 37.5% |
| cautious | 7 | 3 | 42.9% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| explorer | 16 | 12 | 0.750 | 1.688 | 0.400 | 0.985 | False |
| baseline | 13 | 9 | 0.692 | 0.382 | 0.292 | 0.985 | False |
| algorithmic | 11 | 7 | 0.636 | -0.385 | 0.215 | 0.985 | False |
| rapid | 11 | 5 | 0.455 | 0.000 | 0.077 | 0.877 | False |
| verifier | 11 | 6 | 0.545 | -0.377 | 0.154 | 0.938 | False |
| exploratory | 10 | 6 | 0.600 | -0.958 | 0.154 | 0.985 | False |
| junior_reactive | 9 | 5 | 0.556 | -1.193 | 0.077 | 0.954 | False |
| balanced | 8 | 3 | 0.375 | -0.950 | 0.015 | 0.877 | False |
| domain_expert | 8 | 4 | 0.500 | -1.030 | 0.062 | 0.969 | False |
| overthinker | 8 | 5 | 0.625 | -0.964 | 0.123 | 0.985 | False |
| planner | 8 | 3 | 0.375 | -0.929 | 0.015 | 0.877 | False |
| cautious | 7 | 3 | 0.429 | -1.383 | 0.015 | 0.938 | False |

## Policy

- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 120 task executions over two cycles of the 60-task Gmail bank.
