# Initial WebArena Gmail Prompt-Arm Experiment: Methods and Results

## Purpose

This initial experiment evaluates whether an e-process-aware adaptive allocation policy can provide a more efficient prompt-arm search procedure than uniform allocation. The prompt arms are fixed behavioral prompt vectors for a browser-use Gmail agent. The task environment is the WebArena-style local Gmail app, and each task is scored by an automated verifier as binary success or failure.

The experiment has three connected components:

1. **Full paired arm benchmark:** each of the 12 prompt arms is run on the same 60-task Gmail task set. This is the clean prompt-quality baseline because task assignment is paired across arms.
2. **Equal-budget uniform policy run:** a 60-task non-adaptive policy allocates exactly five pulls to each of the 12 arms. This is the fair policy baseline for the adaptive run because both use the same total task budget.
3. **Equal-budget adaptive SPRUCE run:** a 60-task online policy uses e-process state and exploration bonuses to select arms adaptively.

All three components used `gpt-5.4-mini` with low reasoning effort through browser-use. Logs record token usage, wall-clock duration, selected arm, task metadata, policy scores, per-arm e-process state, and global e-process state at every completed task.

## Experimental Units

The task set contains 60 Gmail tasks: 20 easy, 20 medium, and 20 hard. The full paired benchmark runs every prompt arm on all 60 tasks, for 720 task executions. The policy comparison runs use only 60 total task executions each.

The 12 prompt arms are:

`baseline`, `planner`, `cautious`, `explorer`, `balanced`, `overthinker`, `rapid`, `verifier`, `exploratory`, `algorithmic`, `junior_reactive`, and `domain_expert`.

The reward is binary verifier success. The null used by the e-process summaries is per-arm mean reward `m_0 = 0.5` with `alpha = 0.05`. Global evidence is tracked by a linear mixture over per-arm e-processes.

## Allocation Policies

### Full Paired Benchmark

The full paired benchmark is not a policy-selection result. It is a prompt-quality benchmark: every arm receives the same 60 tasks. This makes the arm success rates directly comparable and avoids confounding prompt quality with task difficulty.

### Equal-Budget Uniform Policy

The uniform policy uses a 60-task budget and allocates round-robin across all 12 arms, producing five pulls per arm. This is intentionally noisy at the per-arm level, but it is the correct equal-budget baseline for the adaptive policy because it uses the same number of total task executions.

### Adaptive SPRUCE Policy

The adaptive run uses SPRUCE with one warm-start pull per arm, then chooses arms online using the observed reward history, e-process log-wealth, and an exploration bonus. The run used deterministic tie-breaking so resume paths are stable after watchdog restarts. Each JSONL row stores both the policy scores and the resulting per-arm and global e-process state, which makes the allocation path auditable after the fact.

## Logging and Recovery

The runner flushes one JSONL record per completed task. During the long browser-use runs, some tasks stalled after browser interaction or during structured-output parsing. A watchdog monitored durable log growth and restarted the runner from the last completed task when needed. Resume runs were merged by task index `t`, and final `_FULL.jsonl` logs are the source of truth.

Final source artifacts:

- Full 12-arm paired benchmark status: `docs/paired_sweep_status.md`
- Equal-budget uniform log: `logs/uniform_multiarm/webarena_gmail_uniform_multiarm_60_trial0_2026-06-19T19-27-58Z_FULL.jsonl`
- Equal-budget uniform report: `reports/uniform_multiarm/round_robin_60/`
- Adaptive SPRUCE log: `logs/adaptive_sweep/webarena_gmail_adaptive_spruce_60_trial0_2026-06-19T21-46-42Z_FULL.jsonl`
- Adaptive SPRUCE report: `reports/adaptive_sweep/spruce_60/`
- Adaptive row-by-row e-process timeline: `reports/comparison_sweeps/adaptive_eprocess_timeline.csv`

## Results: Full Paired Benchmark

The full paired benchmark completed all 12 arms on the same 60 tasks. The best arm was `junior_reactive`, with 42/60 successes, followed by `exploratory` with 39/60.

| rank | arm | successes | rate | cost | wall time |
|---:|---|---:|---:|---:|---:|
| 1 | `junior_reactive` | 42/60 | 70.0% | $3.6692 | 66.0 min |
| 2 | `exploratory` | 39/60 | 65.0% | $3.6455 | 84.4 min |
| 3 | `domain_expert` | 36/60 | 60.0% | $3.6282 | 82.5 min |
| 4 | `baseline` | 35/60 | 58.3% | $2.6262 | 58.6 min |
| 5 | `rapid` | 35/60 | 58.3% | $3.4957 | 72.8 min |
| 6 | `balanced` | 34/60 | 56.7% | $2.9575 | 76.2 min |
| 7 | `verifier` | 34/60 | 56.7% | $3.7239 | 86.3 min |
| 8 | `cautious` | 33/60 | 55.0% | $2.6031 | 74.2 min |
| 9 | `overthinker` | 33/60 | 55.0% | $2.8748 | 80.7 min |
| 10 | `explorer` | 33/60 | 55.0% | $2.9363 | 71.8 min |
| 11 | `algorithmic` | 30/60 | 50.0% | $2.5731 | 85.6 min |
| 12 | `planner` | 27/60 | 45.0% | $2.6365 | 96.9 min |

This paired table is the best estimate of prompt-arm quality because every arm saw the same tasks. It also reveals an important caution for the adaptive run: early outcomes can be misleading. In the full paired benchmark, `planner` was the weakest arm at 27/60, yet it appeared very strong in the 60-task adaptive stream because of favorable early assignments.

## Results: Equal-Budget Uniform Policy

The equal-budget uniform run completed 60/60 tasks, allocating five pulls to each arm.

- Successes: **39/60 (65.0%)**
- Total cost: **$3.3710**
- Total wall time: **71.5 min**
- Final global log-e: **0.475**
- Global null rejected: **False**
- Input tokens: **6,626,341**
- Output tokens: **200,208**
- Cache-read tokens: **3,703,296**

| arm | pulls | successes | rate | cost | wall min |
|---|---:|---:|---:|---:|---:|
| `baseline` | 5 | 5 | 100.0% | $0.3077 | 6.0 |
| `planner` | 5 | 5 | 100.0% | $0.1327 | 2.8 |
| `algorithmic` | 5 | 4 | 80.0% | $0.2210 | 5.9 |
| `balanced` | 5 | 4 | 80.0% | $0.2253 | 5.2 |
| `exploratory` | 5 | 4 | 80.0% | $0.2334 | 3.4 |
| `domain_expert` | 5 | 3 | 60.0% | $0.2068 | 6.2 |
| `explorer` | 5 | 3 | 60.0% | $0.3718 | 7.2 |
| `overthinker` | 5 | 3 | 60.0% | $0.1660 | 5.7 |
| `rapid` | 5 | 3 | 60.0% | $0.3649 | 5.9 |
| `cautious` | 5 | 2 | 40.0% | $0.4779 | 7.4 |
| `junior_reactive` | 5 | 2 | 40.0% | $0.3195 | 7.8 |
| `verifier` | 5 | 1 | 20.0% | $0.3438 | 7.9 |

Uniform matched the full paired `exploratory` arm's 65.0% success rate, but it did so as a policy mixture, not as a single prompt. The five-pull per-arm estimates are too noisy to rank arms reliably.

## Results: Adaptive SPRUCE Policy

The adaptive SPRUCE run completed 60/60 tasks.

- Successes: **38/60 (63.3%)**
- Total cost: **$3.2436**
- Total wall time: **65.8 min**
- Final global log-e: **0.152**
- Global null rejected: **False**
- Input tokens: **6,320,719**
- Output tokens: **191,155**
- Cache-read tokens: **3,492,096**

| arm | pulls | successes | rate | cost | wall min |
|---|---:|---:|---:|---:|---:|
| `planner` | 8 | 7 | 87.5% | $0.3512 | 7.0 |
| `baseline` | 8 | 6 | 75.0% | $0.3923 | 4.9 |
| `exploratory` | 7 | 5 | 71.4% | $0.2441 | 5.3 |
| `algorithmic` | 5 | 3 | 60.0% | $0.2403 | 3.8 |
| `rapid` | 5 | 3 | 60.0% | $0.2313 | 5.5 |
| `cautious` | 4 | 3 | 75.0% | $0.3029 | 5.1 |
| `balanced` | 4 | 2 | 50.0% | $0.2711 | 6.4 |
| `domain_expert` | 4 | 2 | 50.0% | $0.2765 | 7.4 |
| `explorer` | 4 | 2 | 50.0% | $0.1832 | 3.2 |
| `verifier` | 4 | 2 | 50.0% | $0.3278 | 6.7 |
| `overthinker` | 4 | 1 | 25.0% | $0.1727 | 6.4 |
| `junior_reactive` | 3 | 2 | 66.7% | $0.2503 | 4.0 |

The adaptive policy finished at 38/60, slightly below the equal-budget uniform policy's 39/60 and below the best full paired arm, `junior_reactive`, at 42/60. It concentrated most heavily on `planner` and `baseline`; in this adaptive stream `planner` achieved 7/8, even though the full paired benchmark showed `planner` was the weakest arm overall. This is useful negative evidence for the methodology: a single 60-task adaptive run can overfit early favorable outcomes and should be repeated or evaluated with regret against the paired benchmark.

## E-Process Evidence Accumulation During Adaptive Run

The row-by-row e-process trajectory is saved in `reports/comparison_sweeps/adaptive_eprocess_timeline.csv`. The table below shows the warm-start endpoint, leader changes, global log-e maxima, and regular checkpoints.

| t | selected arm | reward | cumulative success | global log-e | global e | leading arm by log-e | leader log-e | cumulative cost | cumulative wall |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | `baseline` | 0 | 0/1 (0.0%) | 0.000 | 1.00 | `baseline` | 0.000 | $0.040 | 0.3 min |
| 12 | `domain_expert` | 1 | 11/12 (91.7%) | 0.000 | 1.00 | `baseline` | 0.000 | $0.303 | 3.0 min |
| 14 | `planner` | 1 | 13/14 (92.9%) | 0.041 | 1.04 | `planner` | 0.405 | $0.346 | 3.5 min |
| 17 | `balanced` | 1 | 15/17 (88.2%) | 0.080 | 1.08 | `planner` | 0.405 | $0.536 | 5.4 min |
| 20 | `verifier` | 1 | 17/20 (85.0%) | 0.118 | 1.12 | `planner` | 0.405 | $0.643 | 7.6 min |
| 21 | `exploratory` | 1 | 18/21 (85.7%) | 0.154 | 1.17 | `planner` | 0.405 | $0.664 | 7.9 min |
| 22 | `algorithmic` | 1 | 19/22 (86.4%) | 0.189 | 1.21 | `planner` | 0.405 | $0.757 | 9.0 min |
| 25 | `planner` | 1 | 21/25 (84.0%) | 0.240 | 1.27 | `planner` | 0.811 | $1.071 | 13.7 min |
| 30 | `exploratory` | 1 | 22/30 (73.3%) | 0.080 | 1.08 | `planner` | 0.811 | $1.532 | 23.9 min |
| 37 | `planner` | 1 | 27/37 (73.0%) | 0.276 | 1.32 | `planner` | 1.622 | $1.811 | 31.0 min |
| 38 | `exploratory` | 1 | 28/38 (73.7%) | 0.377 | 1.46 | `planner` | 1.622 | $1.862 | 31.9 min |
| 39 | `planner` | 1 | 29/39 (74.4%) | 0.512 | 1.67 | `planner` | 2.027 | $1.887 | 32.4 min |
| 40 | `exploratory` | 0 | 29/40 (72.5%) | 0.377 | 1.46 | `planner` | 2.027 | $1.950 | 34.2 min |
| 41 | `planner` | 1 | 30/41 (73.2%) | 0.574 | 1.77 | `planner` | 2.433 | $2.006 | 34.8 min |
| 42 | `baseline` | 1 | 31/42 (73.8%) | 0.590 | 1.80 | `planner` | 2.433 | $2.043 | 35.6 min |
| 50 | `rapid` | 1 | 35/50 (70.0%) | 0.311 | 1.36 | `planner` | 1.740 | $2.558 | 48.4 min |
| 60 | `overthinker` | 0 | 38/60 (63.3%) | 0.152 | 1.16 | `planner` | 1.740 | $3.244 | 65.8 min |

The global e-process never crossed the `alpha = 0.05` rejection threshold `log(1/alpha) = 2.996`. The strongest per-arm final evidence was for `planner` with log-e 1.740, but that evidence was based on only eight pulls and conflicts with the full paired benchmark, where `planner` was the worst arm. The global log-e peaked around the mid-run and ended at 0.152, corresponding to an e-value of 1.16, which is not strong evidence against the global null.

## Interpretation

The initial result does not support the claim that this specific adaptive SPRUCE run was more successful than equal-budget uniform allocation. Uniform achieved 65.0%; adaptive achieved 63.3%. Adaptive used slightly less cost and wall time, but the difference is small and came with a lower success count.

The run still validates the infrastructure and the measurement strategy:

- Both policies were run under equal 60-task budgets.
- Every task row logs token cost, wall time, selected arm, reward, policy scores, per-arm e-process state, and global e-process state.
- The adaptive trajectory is auditable step by step.
- The full paired benchmark gives a clean reference for prompt quality and for post hoc regret analysis.

For a paper, the strongest honest framing is that this is a pilot experiment. It demonstrates a complete methodology for comparing full paired prompt benchmarks, equal-budget uniform policy allocation, and adaptive e-process-guided allocation. The pilot also exposes a real risk: adaptive allocation can be misled by early wins on a weak arm when the budget is small and the task stream is heterogeneous.

## Recommended Next Experiment

A stronger follow-up would run multiple random task orders or multiple seeds for both equal-budget uniform and adaptive policies. The primary estimands should be:

- Mean success rate under a 60-task policy budget
- Cost per success
- Wall-clock time per success
- Regret relative to the best paired arm, `junior_reactive`
- Frequency with which adaptive allocation concentrates on the paired best arm
- Time to first meaningful e-process evidence threshold, if any

The current pilot should be reported as evidence that the logging and anytime-valid evidence machinery works, but not as evidence that adaptive allocation is already superior.
