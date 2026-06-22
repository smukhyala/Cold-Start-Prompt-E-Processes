# Adaptive SPRUCE 120-Task rep2 Status

This file tracks `adaptive_spruce_120_rep2`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| adaptive_spruce_120_rep2 | done | 120/120 | logs/adaptive_sweep_120_rep2/webarena_gmail_adaptive_spruce_120_rep2_trial0_2026-06-21T07-50-52Z_FULL.jsonl | reports/adaptive_sweep/spruce_120_rep2 |

## Headline

- Durable tasks: 120/120
- Total successes: 73/120 (60.8%)
- Total cost: $6.3226
- Total wall time: 258.3 min
- Final global log-e: 0.721
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| domain_expert | 19 | 15 | 78.9% |
| balanced | 12 | 8 | 66.7% |
| algorithmic | 11 | 7 | 63.6% |
| cautious | 10 | 7 | 70.0% |
| rapid | 10 | 6 | 60.0% |
| baseline | 9 | 4 | 44.4% |
| explorer | 9 | 6 | 66.7% |
| overthinker | 9 | 6 | 66.7% |
| junior_reactive | 8 | 3 | 37.5% |
| planner | 8 | 3 | 37.5% |
| verifier | 8 | 4 | 50.0% |
| exploratory | 7 | 4 | 57.1% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| domain_expert | 19 | 15 | 0.789 | 2.904 | 0.462 | 0.985 | False |
| balanced | 12 | 8 | 0.667 | 0.066 | 0.262 | 0.985 | False |
| algorithmic | 11 | 7 | 0.636 | -0.391 | 0.215 | 0.985 | False |
| cautious | 10 | 7 | 0.700 | -0.153 | 0.231 | 0.985 | False |
| rapid | 10 | 6 | 0.600 | -0.710 | 0.185 | 0.985 | False |
| baseline | 9 | 4 | 0.444 | -0.613 | 0.062 | 0.938 | False |
| explorer | 9 | 6 | 0.667 | -0.884 | 0.154 | 0.985 | False |
| overthinker | 9 | 6 | 0.667 | -0.349 | 0.185 | 0.985 | False |
| junior_reactive | 8 | 3 | 0.375 | -0.693 | 0.015 | 0.862 | False |
| planner | 8 | 3 | 0.375 | -0.929 | 0.015 | 0.877 | False |
| verifier | 8 | 4 | 0.500 | -0.867 | 0.077 | 0.985 | False |
| exploratory | 7 | 4 | 0.571 | -0.983 | 0.046 | 0.985 | False |

## Policy

- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 120 task executions over two cycles of the 60-task Gmail bank.
