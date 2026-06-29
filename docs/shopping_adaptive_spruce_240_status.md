# Shopping Adaptive SPRUCE 240-Task rep1 Status

This file tracks `shopping_adaptive_spruce_240`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| shopping_adaptive_spruce_240 | done | 240/240 | logs/shopping_adaptive_sweep_240/webarena_shopping_adaptive_spruce_240_trial0_2026-06-25T18-25-28Z_FULL.jsonl | reports/shopping_adaptive_sweep/spruce_240 |

## Headline

- Durable tasks: 240/240
- Total successes: 98/240 (40.8%)
- Total cost: $7.1262
- Total wall time: 407.5 min
- Final global log-e: -0.642
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| exploratory | 23 | 6 | 26.1% |
| verifier | 23 | 7 | 30.4% |
| balanced | 22 | 10 | 45.5% |
| cautious | 20 | 12 | 60.0% |
| junior_reactive | 20 | 12 | 60.0% |
| overthinker | 20 | 6 | 30.0% |
| planner | 20 | 5 | 25.0% |
| baseline | 19 | 8 | 42.1% |
| explorer | 19 | 11 | 57.9% |
| rapid | 19 | 7 | 36.8% |
| domain_expert | 18 | 10 | 55.6% |
| algorithmic | 17 | 4 | 23.5% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| exploratory | 23 | 6 | 0.261 | 0.000 | 0.046 | 0.569 | False |
| verifier | 23 | 7 | 0.304 | 0.000 | 0.077 | 0.615 | False |
| balanced | 22 | 10 | 0.455 | -0.059 | 0.169 | 0.785 | False |
| cautious | 20 | 12 | 0.600 | -0.969 | 0.277 | 0.892 | False |
| junior_reactive | 20 | 12 | 0.600 | -0.957 | 0.277 | 0.892 | False |
| overthinker | 20 | 6 | 0.300 | -0.693 | 0.031 | 0.600 | False |
| planner | 20 | 5 | 0.250 | -0.706 | 0.031 | 0.554 | False |
| baseline | 19 | 8 | 0.421 | -0.909 | 0.154 | 0.738 | False |
| explorer | 19 | 11 | 0.579 | -1.228 | 0.262 | 0.877 | False |
| rapid | 19 | 7 | 0.368 | -0.909 | 0.123 | 0.677 | False |
| domain_expert | 18 | 10 | 0.556 | -1.373 | 0.231 | 0.877 | False |
| algorithmic | 17 | 4 | 0.235 | -1.331 | 0.015 | 0.569 | False |

## Policy

- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 240 task executions over three cycles of the 80-task shopping bank.
