# Shopping Uniform Multi-Arm 240-Task rep2 Status

This file tracks `shopping_uniform_multiarm_240_rep2`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| shopping_uniform_multiarm_240_rep2 | done | 240/240 | logs/shopping_uniform_multiarm_240_rep2/webarena_shopping_uniform_multiarm_240_rep2_trial0_2026-06-26T07-57-31Z_FULL.jsonl | reports/shopping_uniform_multiarm/round_robin_240_rep2 |

## Headline

- Durable tasks: 240/240
- Total successes: 91/240 (37.9%)
- Total cost: $7.2970
- Total wall time: 420.1 min
- Final global log-e: -0.849
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| algorithmic | 20 | 8 | 40.0% |
| balanced | 20 | 8 | 40.0% |
| baseline | 20 | 6 | 30.0% |
| cautious | 20 | 9 | 45.0% |
| domain_expert | 20 | 5 | 25.0% |
| exploratory | 20 | 7 | 35.0% |
| explorer | 20 | 10 | 50.0% |
| junior_reactive | 20 | 8 | 40.0% |
| overthinker | 20 | 7 | 35.0% |
| planner | 20 | 6 | 30.0% |
| rapid | 20 | 10 | 50.0% |
| verifier | 20 | 7 | 35.0% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| algorithmic | 20 | 8 | 0.400 | -1.315 | 0.108 | 0.738 | False |
| balanced | 20 | 8 | 0.400 | -1.030 | 0.138 | 0.708 | False |
| baseline | 20 | 6 | 0.300 | -0.706 | 0.046 | 0.600 | False |
| cautious | 20 | 9 | 0.450 | -2.175 | 0.154 | 0.785 | False |
| domain_expert | 20 | 5 | 0.250 | -0.706 | 0.031 | 0.554 | False |
| exploratory | 20 | 7 | 0.350 | 0.000 | 0.092 | 0.677 | False |
| explorer | 20 | 10 | 0.500 | -1.664 | 0.200 | 0.815 | False |
| junior_reactive | 20 | 8 | 0.400 | -1.133 | 0.108 | 0.723 | False |
| overthinker | 20 | 7 | 0.350 | -0.706 | 0.077 | 0.662 | False |
| planner | 20 | 6 | 0.300 | -0.929 | 0.062 | 0.600 | False |
| rapid | 20 | 10 | 0.500 | -1.138 | 0.185 | 0.800 | False |
| verifier | 20 | 7 | 0.350 | -0.374 | 0.108 | 0.708 | False |

## Policy

- Sampler: uniform round-robin across all 12 arms. Budget: 240 task executions, giving 20 pulls per arm over three cycles of the 80-task shopping bank.
