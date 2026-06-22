# Uniform vs Adaptive 120-Task Replicates

## Purpose

This experiment extends the initial 60-task pilot by comparing equal-budget uniform allocation against e-process-aware adaptive allocation across three 120-task replicates. Each 120-task run cycles through the same 60-task WebArena Gmail task bank twice, so the budget is 120 executions rather than 120 unique tasks.

The goal is to test whether adaptive allocation can identify stronger prompt arms and improve efficiency relative to uniform round-robin sampling while logging enough evidence to support paper-ready analysis.

## Design

- Environment: local WebArena-style Gmail task suite.
- Model: `gpt-5.4-mini`, low reasoning effort.
- Arms: the 12 prompt arms in `configs/arms_initial.yaml`.
- Uniform policy: round-robin across all arms, giving 10 pulls per arm in each 120-task run.
- Adaptive policy: SPRUCE over all arms, with 3 warm-start pulls per arm followed by e-process-aware adaptive allocation.
- Reward: binary verifier success.
- Evidence process: per-arm and global e-process summaries with null mean reward `m0 = 0.5` and `alpha = 0.05`.
- Logged metrics: selected arm, task index, reward, steps, token input/output/cache usage, token cost, wall-clock time, per-arm allocation, global log-e, and final per-arm confidence sequence summaries.

The paired full-arm benchmark remains the cleanest prompt-quality baseline because every arm sees the same 60 tasks. These 120-task policy runs instead compare allocation policies under equal execution budgets.

## Final Results

| replicate | uniform successes | uniform rate | uniform cost | uniform wall | adaptive successes | adaptive rate | adaptive cost | adaptive wall | adaptive minus uniform |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74/120 | 61.7% | $7.1489 | 139.5 min | 72/120 | 60.0% | $6.0562 | 116.2 min | -2 |
| 2 | 64/120 | 53.3% | $6.5545 | 136.6 min | 73/120 | 60.8% | $6.3226 | 258.3 min | +9 |
| 3 | 64/120 | 53.3% | $5.4233 | 152.0 min | 68/120 | 56.7% | $5.0521 | 155.1 min | +4 |

Aggregate:

| policy | tasks | successes | success rate | total cost | cost per success | total wall | wall per success |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 360 | 202 | 56.1% | $19.1267 | $0.0947 | 428.2 min | 2.12 min |
| adaptive | 360 | 213 | 59.2% | $17.4309 | $0.0818 | 529.6 min | 2.49 min |

Adaptive finished with 11 more successes over 360 task executions and used $1.6958 less token cost than uniform. Wall-clock time was worse for adaptive because replicate 2 had substantial browser-use stalls and watchdog restarts; this should be reported as operational overhead, not purely model inference time.

## Adaptive Allocation

Across the three adaptive replicates, allocation concentrated most heavily on the arms below.

| arm | adaptive pulls | successes | rate |
|---|---:|---:|---:|
| `domain_expert` | 38 | 26 | 68.4% |
| `balanced` | 35 | 22 | 62.9% |
| `explorer` | 33 | 22 | 66.7% |
| `baseline` | 32 | 19 | 59.4% |
| `algorithmic` | 31 | 19 | 61.3% |
| `rapid` | 31 | 17 | 54.8% |
| `verifier` | 30 | 15 | 50.0% |
| `planner` | 28 | 14 | 50.0% |
| `cautious` | 27 | 16 | 59.3% |
| `exploratory` | 25 | 15 | 60.0% |
| `junior_reactive` | 25 | 12 | 48.0% |
| `overthinker` | 25 | 16 | 64.0% |

The adaptive policy did not simply lock onto the best full paired benchmark arm (`junior_reactive`). Instead, it allocated more often to `domain_expert`, `balanced`, and `explorer`. This is an important methodological point: the adaptive stream measures online policy utility under the observed task ordering, not intrinsic prompt quality in the paired benchmark sense.

## E-Process Timeline

The adaptive e-process checkpoint timeline is saved at:

`reports/comparison_sweeps/adaptive_120_eprocess_timeline.csv`

Key checkpoints:

| replicate | t | event | cumulative success | global log-e | cumulative cost | cumulative wall |
|---|---:|---|---:|---:|---:|---:|
| 1 | 36 | checkpoint | 27/36 (75.0%) | 0.433 | $1.7060 | 23.2 min |
| 1 | 60 | checkpoint | 36/60 (60.0%) | 0.124 | $2.8455 | 52.5 min |
| 1 | 96 | max global log-e | 61/96 (63.5%) | 0.654 | $4.6972 | 83.8 min |
| 1 | 120 | final | 72/120 (60.0%) | -0.184 | $6.0562 | 116.2 min |
| 2 | 36 | checkpoint | 27/36 (75.0%) | 0.223 | $1.6288 | 144.4 min |
| 2 | 60 | checkpoint | 37/60 (61.7%) | -0.306 | $3.0000 | 185.5 min |
| 2 | 102 | max global log-e | 69/102 (67.6%) | 1.922 | $5.2328 | 224.6 min |
| 2 | 120 | final | 73/120 (60.8%) | 0.721 | $6.3226 | 258.3 min |
| 3 | 36 | checkpoint | 25/36 (69.4%) | 0.206 | $1.3440 | 31.2 min |
| 3 | 60 | checkpoint | 35/60 (58.3%) | -0.162 | $2.4320 | 67.1 min |
| 3 | 91 | max global log-e | 58/91 (63.7%) | 0.650 | $3.6803 | 104.2 min |
| 3 | 120 | final | 68/120 (56.7%) | -0.026 | $5.0521 | 155.1 min |

No adaptive replicate crossed the global rejection threshold `log(1 / 0.05) = 2.996`. Replicate 2 produced the strongest global evidence, peaking at log-e 1.922 around task 102, but even that remained below the anytime-valid rejection boundary.

## Interpretation

The 120-task replicate study gives stronger support for adaptive allocation than the initial 60-task pilot. Across three equal-budget comparisons, adaptive won two of three replicates and led by 11 successes overall. Adaptive also reduced token cost and cost per success.

The wall-clock result is mixed. Adaptive took longer overall because of stalls and watchdog restarts, especially in replicate 2. For the paper, runtime should be reported separately as operational wall-clock time, with a caveat that browser-use stalls were a major contributor. Token cost and success count are cleaner efficiency metrics for this run.

The e-process evidence should be framed carefully. The e-process made the allocation path auditable and provided anytime-valid evidence traces, but it did not produce a formal global rejection. The practical result is therefore a policy-performance improvement in this finite experiment, not a statistically decisive e-process rejection of the null.

## Source Artifacts

- Uniform reports:
  - `reports/uniform_multiarm/round_robin_120/`
  - `reports/uniform_multiarm/round_robin_120_rep2/`
  - `reports/uniform_multiarm/round_robin_120_rep3/`
- Adaptive reports:
  - `reports/adaptive_sweep/spruce_120/`
  - `reports/adaptive_sweep/spruce_120_rep2/`
  - `reports/adaptive_sweep/spruce_120_rep3/`
- Status docs:
  - `docs/uniform_multiarm_120_status.md`
  - `docs/adaptive_sweep_120_status.md`
  - `docs/uniform_multiarm_120_rep2_status.md`
  - `docs/adaptive_spruce_120_rep2_status.md`
  - `docs/uniform_multiarm_120_rep3_status.md`
  - `docs/adaptive_spruce_120_rep3_status.md`
