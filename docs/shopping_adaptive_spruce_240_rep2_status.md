# Shopping Adaptive SPRUCE 240-Task rep2 Status

This file tracks `shopping_adaptive_spruce_240_rep2`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| shopping_adaptive_spruce_240_rep2 | done | 240/240 | logs/shopping_adaptive_sweep_240_rep2/webarena_shopping_adaptive_spruce_240_rep2_trial0_2026-06-26T17-36-04Z_FULL.jsonl | reports/shopping_adaptive_sweep/spruce_240_rep2 |

## Headline

- Durable tasks: 240/240
- Total successes: 100/240 (41.7%)
- Total cost: $7.4744
- Total wall time: 380.8 min
- Final global log-e: -0.787
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| explorer | 25 | 16 | 64.0% |
| exploratory | 24 | 9 | 37.5% |
| balanced | 21 | 7 | 33.3% |
| overthinker | 21 | 5 | 23.8% |
| domain_expert | 20 | 9 | 45.0% |
| algorithmic | 19 | 5 | 26.3% |
| baseline | 19 | 8 | 42.1% |
| planner | 19 | 7 | 36.8% |
| verifier | 19 | 5 | 26.3% |
| junior_reactive | 18 | 10 | 55.6% |
| rapid | 18 | 9 | 50.0% |
| cautious | 17 | 10 | 58.8% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| explorer | 25 | 16 | 0.640 | -0.054 | 0.354 | 0.892 | False |
| exploratory | 24 | 9 | 0.375 | 0.000 | 0.123 | 0.677 | False |
| balanced | 21 | 7 | 0.333 | -0.706 | 0.077 | 0.631 | False |
| overthinker | 21 | 5 | 0.238 | -0.693 | 0.015 | 0.538 | False |
| domain_expert | 20 | 9 | 0.450 | -0.836 | 0.138 | 0.754 | False |
| algorithmic | 19 | 5 | 0.263 | -1.133 | 0.031 | 0.585 | False |
| baseline | 19 | 8 | 0.421 | -1.118 | 0.123 | 0.754 | False |
| planner | 19 | 7 | 0.368 | -0.945 | 0.123 | 0.692 | False |
| verifier | 19 | 5 | 0.263 | -1.133 | 0.015 | 0.585 | False |
| junior_reactive | 18 | 10 | 0.556 | -1.438 | 0.215 | 0.862 | False |
| rapid | 18 | 9 | 0.500 | -1.294 | 0.185 | 0.815 | False |
| cautious | 17 | 10 | 0.588 | -1.530 | 0.215 | 0.877 | False |

## Policy

- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 240 task executions over three cycles of the 80-task shopping bank.
