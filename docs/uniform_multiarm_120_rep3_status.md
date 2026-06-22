# Uniform Multi-Arm 120-Task rep3 Status

This file tracks `uniform_multiarm_120_rep3`.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| uniform_multiarm_120_rep3 | done | 120/120 | logs/uniform_multiarm_120_rep3/webarena_gmail_uniform_multiarm_120_rep3_trial0_2026-06-21T21-04-04Z_FULL.jsonl | reports/uniform_multiarm/round_robin_120_rep3 |

## Headline

- Durable tasks: 120/120
- Total successes: 64/120 (53.3%)
- Total cost: $5.4233
- Total wall time: 152.0 min
- Final global log-e: -0.627
- Global null rejected: False
- First global rejection: not yet

## Allocation By Arm

| arm | pulls | successes | rate |
|---|---:|---:|---:|
| algorithmic | 10 | 6 | 60.0% |
| balanced | 10 | 6 | 60.0% |
| baseline | 10 | 7 | 70.0% |
| cautious | 10 | 5 | 50.0% |
| domain_expert | 10 | 6 | 60.0% |
| exploratory | 10 | 6 | 60.0% |
| explorer | 10 | 4 | 40.0% |
| junior_reactive | 10 | 5 | 50.0% |
| overthinker | 10 | 4 | 40.0% |
| planner | 10 | 6 | 60.0% |
| rapid | 10 | 6 | 60.0% |
| verifier | 10 | 3 | 30.0% |

## End-State E-Process By Arm

| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| algorithmic | 10 | 6 | 0.600 | -0.848 | 0.154 | 0.985 | False |
| balanced | 10 | 6 | 0.600 | -1.115 | 0.138 | 0.969 | False |
| baseline | 10 | 7 | 0.700 | 0.193 | 0.246 | 0.985 | False |
| cautious | 10 | 5 | 0.500 | -0.306 | 0.108 | 0.938 | False |
| domain_expert | 10 | 6 | 0.600 | -0.848 | 0.154 | 0.985 | False |
| exploratory | 10 | 6 | 0.600 | -0.848 | 0.154 | 0.985 | False |
| explorer | 10 | 4 | 0.400 | -0.867 | 0.015 | 0.831 | False |
| junior_reactive | 10 | 5 | 0.500 | -1.175 | 0.062 | 0.908 | False |
| overthinker | 10 | 4 | 0.400 | -1.133 | 0.015 | 0.831 | False |
| planner | 10 | 6 | 0.600 | -1.095 | 0.138 | 0.969 | False |
| rapid | 10 | 6 | 0.600 | -0.769 | 0.169 | 0.985 | False |
| verifier | 10 | 3 | 0.300 | 0.000 | 0.015 | 0.769 | False |

## Policy

- Sampler: uniform round-robin across all 12 arms. Budget: 120 task executions, giving 10 pulls per arm over two cycles of the 60-task Gmail bank.
