# Shopping Uniform Multi-Arm 240-Task rep1 Status

This file tracks `shopping_uniform_multiarm_240`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| shopping_uniform_multiarm_240 | done | 240/240 | logs/shopping_uniform_multiarm_240/webarena_shopping_uniform_multiarm_240_trial0_2026-06-25T02-01-10Z_FULL.jsonl | reports/shopping_uniform_multiarm/round_robin_240 |

## Headline

- Durable tasks: 240/240
- Total successes: 109/240 (45.4%)
- Total cost: $7.8008
- Total wall time: 387.6 min
- Final global log-e: -0.929
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| algorithmic | 20 | 8 | 40.0% |
| balanced | 20 | 9 | 45.0% |
| baseline | 20 | 9 | 45.0% |
| cautious | 20 | 12 | 60.0% |
| domain_expert | 20 | 9 | 45.0% |
| exploratory | 20 | 8 | 40.0% |
| explorer | 20 | 9 | 45.0% |
| junior_reactive | 20 | 10 | 50.0% |
| overthinker | 20 | 9 | 45.0% |
| planner | 20 | 9 | 45.0% |
| rapid | 20 | 8 | 40.0% |
| verifier | 20 | 9 | 45.0% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| algorithmic | 20 | 8 | 0.400 | -1.239 | 0.108 | 0.738 | False |
| balanced | 20 | 9 | 0.450 | -1.289 | 0.154 | 0.769 | False |
| baseline | 20 | 9 | 0.450 | -1.337 | 0.154 | 0.769 | False |
| cautious | 20 | 12 | 0.600 | -0.754 | 0.277 | 0.892 | False |
| domain_expert | 20 | 9 | 0.450 | -0.706 | 0.138 | 0.754 | False |
| exploratory | 20 | 8 | 0.400 | 0.000 | 0.138 | 0.754 | False |
| explorer | 20 | 9 | 0.450 | -1.133 | 0.123 | 0.769 | False |
| junior_reactive | 20 | 10 | 0.500 | -1.401 | 0.185 | 0.800 | False |
| overthinker | 20 | 9 | 0.450 | -0.812 | 0.154 | 0.754 | False |
| planner | 20 | 9 | 0.450 | -1.163 | 0.154 | 0.769 | False |
| rapid | 20 | 8 | 0.400 | -0.812 | 0.123 | 0.708 | False |
| verifier | 20 | 9 | 0.450 | -1.732 | 0.154 | 0.769 | False |

## Policy

- Sampler: uniform round-robin across all 12 arms. Budget: 240 task executions, giving 20 pulls per arm over three cycles of the 80-task shopping bank.
