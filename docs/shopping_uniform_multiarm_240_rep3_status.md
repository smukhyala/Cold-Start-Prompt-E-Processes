# Shopping Uniform Multi-Arm 240-Task rep3 Status

This file tracks `shopping_uniform_multiarm_240_rep3`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| shopping_uniform_multiarm_240_rep3 | done | 240/240 | logs/shopping_uniform_multiarm_240_rep3/webarena_shopping_uniform_multiarm_240_rep3_trial0_2026-06-27T02-13-39Z_FULL.jsonl | reports/shopping_uniform_multiarm/round_robin_240_rep3 |

## Headline

- Durable tasks: 240/240
- Total successes: 108/240 (45.0%)
- Total cost: $7.9996
- Total wall time: 366.5 min
- Final global log-e: -0.644
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| algorithmic | 20 | 7 | 35.0% |
| balanced | 20 | 7 | 35.0% |
| baseline | 20 | 7 | 35.0% |
| cautious | 20 | 11 | 55.0% |
| domain_expert | 20 | 7 | 35.0% |
| exploratory | 20 | 10 | 50.0% |
| explorer | 20 | 10 | 50.0% |
| junior_reactive | 20 | 14 | 70.0% |
| overthinker | 20 | 9 | 45.0% |
| planner | 20 | 9 | 45.0% |
| rapid | 20 | 10 | 50.0% |
| verifier | 20 | 7 | 35.0% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| algorithmic | 20 | 7 | 0.350 | -1.133 | 0.077 | 0.677 | False |
| balanced | 20 | 7 | 0.350 | -0.812 | 0.108 | 0.662 | False |
| baseline | 20 | 7 | 0.350 | -1.174 | 0.092 | 0.662 | False |
| cautious | 20 | 11 | 0.550 | -1.046 | 0.215 | 0.831 | False |
| domain_expert | 20 | 7 | 0.350 | -1.121 | 0.092 | 0.662 | False |
| exploratory | 20 | 10 | 0.500 | -1.806 | 0.185 | 0.815 | False |
| explorer | 20 | 10 | 0.500 | -1.215 | 0.185 | 0.800 | False |
| junior_reactive | 20 | 14 | 0.700 | 0.779 | 0.385 | 0.938 | False |
| overthinker | 20 | 9 | 0.450 | -0.952 | 0.154 | 0.754 | False |
| planner | 20 | 9 | 0.450 | -1.453 | 0.169 | 0.769 | False |
| rapid | 20 | 10 | 0.500 | -1.252 | 0.185 | 0.800 | False |
| verifier | 20 | 7 | 0.350 | 0.000 | 0.092 | 0.692 | False |

## Policy

- Sampler: uniform round-robin across all 12 arms. Budget: 240 task executions, giving 20 pulls per arm over three cycles of the 80-task shopping bank.
