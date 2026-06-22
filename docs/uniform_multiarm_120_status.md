# Uniform Multi-Arm 120-Task Status

This file tracks `uniform_multiarm_120`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| uniform_multiarm_120 | done | 120/120 | logs/uniform_multiarm_120/webarena_gmail_uniform_multiarm_120_trial0_2026-06-20T19-46-07Z_FULL.jsonl | reports/uniform_multiarm/round_robin_120 |

## Headline

- Durable tasks: 120/120
- Total successes: 74/120 (61.7%)
- Total cost: $7.1489
- Total wall time: 139.5 min
- Final global log-e: 0.832
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| algorithmic | 10 | 5 | 50.0% |
| balanced | 10 | 6 | 60.0% |
| baseline | 10 | 9 | 90.0% |
| cautious | 10 | 4 | 40.0% |
| domain_expert | 10 | 6 | 60.0% |
| exploratory | 10 | 8 | 80.0% |
| explorer | 10 | 6 | 60.0% |
| junior_reactive | 10 | 5 | 50.0% |
| overthinker | 10 | 6 | 60.0% |
| planner | 10 | 8 | 80.0% |
| rapid | 10 | 7 | 70.0% |
| verifier | 10 | 4 | 40.0% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| algorithmic | 10 | 5 | 0.500 | -1.460 | 0.062 | 0.908 | False |
| balanced | 10 | 6 | 0.600 | -1.115 | 0.138 | 0.969 | False |
| baseline | 10 | 9 | 0.900 | 2.737 | 0.446 | 0.985 | False |
| cautious | 10 | 4 | 0.400 | -0.693 | 0.015 | 0.831 | False |
| domain_expert | 10 | 6 | 0.600 | -0.848 | 0.154 | 0.985 | False |
| exploratory | 10 | 8 | 0.800 | 1.452 | 0.323 | 0.985 | False |
| explorer | 10 | 6 | 0.600 | -0.703 | 0.169 | 0.985 | False |
| junior_reactive | 10 | 5 | 0.500 | -1.175 | 0.062 | 0.908 | False |
| overthinker | 10 | 6 | 0.600 | -1.506 | 0.123 | 0.969 | False |
| planner | 10 | 8 | 0.800 | 1.341 | 0.323 | 0.985 | False |
| rapid | 10 | 7 | 0.700 | 0.149 | 0.262 | 0.985 | False |
| verifier | 10 | 4 | 0.400 | -0.979 | 0.046 | 0.846 | False |

## Policy

- Sampler: uniform round-robin across all 12 arms. Budget: 120 task executions, giving 10 pulls per arm over two cycles of the 60-task Gmail bank.
