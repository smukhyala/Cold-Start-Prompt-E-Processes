# Shopping Adaptive SPRUCE 240-Task rep3 Status

This file tracks `shopping_adaptive_spruce_240_rep3`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| shopping_adaptive_spruce_240_rep3 | done | 240/240 | logs/shopping_adaptive_sweep_240_rep3/webarena_shopping_adaptive_spruce_240_rep3_trial0_2026-06-28T05-15-19Z_FULL.jsonl | reports/shopping_adaptive_sweep/spruce_240_rep3 |

## Headline

- Durable tasks: 240/240
- Total successes: 112/240 (46.7%)
- Total cost: $8.0009
- Total wall time: 375.7 min
- Final global log-e: -0.869
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| exploratory | 24 | 11 | 45.8% |
| domain_expert | 21 | 6 | 28.6% |
| overthinker | 21 | 6 | 28.6% |
| verifier | 21 | 11 | 52.4% |
| baseline | 20 | 9 | 45.0% |
| cautious | 20 | 11 | 55.0% |
| junior_reactive | 20 | 9 | 45.0% |
| rapid | 20 | 12 | 60.0% |
| explorer | 19 | 8 | 42.1% |
| algorithmic | 18 | 11 | 61.1% |
| balanced | 18 | 10 | 55.6% |
| planner | 18 | 8 | 44.4% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| exploratory | 24 | 11 | 0.458 | -0.112 | 0.185 | 0.754 | False |
| domain_expert | 21 | 6 | 0.286 | -0.706 | 0.046 | 0.585 | False |
| overthinker | 21 | 6 | 0.286 | -0.706 | 0.062 | 0.585 | False |
| verifier | 21 | 11 | 0.524 | -0.629 | 0.231 | 0.831 | False |
| baseline | 20 | 9 | 0.450 | -0.836 | 0.138 | 0.754 | False |
| cautious | 20 | 11 | 0.550 | -0.933 | 0.246 | 0.846 | False |
| junior_reactive | 20 | 9 | 0.450 | -0.812 | 0.154 | 0.754 | False |
| rapid | 20 | 12 | 0.600 | -1.281 | 0.262 | 0.877 | False |
| explorer | 19 | 8 | 0.421 | -1.179 | 0.154 | 0.738 | False |
| algorithmic | 18 | 11 | 0.611 | -1.294 | 0.262 | 0.908 | False |
| balanced | 18 | 10 | 0.556 | -1.544 | 0.215 | 0.862 | False |
| planner | 18 | 8 | 0.444 | -1.289 | 0.138 | 0.769 | False |

## Policy

- Sampler: SPRUCE over all 12 arms. Warm start: 3 pulls per arm, then e-process-aware adaptive allocation. Budget: 240 task executions over three cycles of the 80-task shopping bank.
