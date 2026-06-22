# Uniform vs Adaptive Comparison Summary

## 120-Task Replicate Study

The 120-task follow-up completed three uniform/adaptive pairs. Each run used a 120-execution budget over two cycles of the 60-task Gmail task bank.

| policy | tasks | successes | rate | cost | cost per success | wall time |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 360 | 202 | 56.1% | $19.1267 | $0.0947 | 428.2 min |
| adaptive SPRUCE | 360 | 213 | 59.2% | $17.4309 | $0.0818 | 529.6 min |

Adaptive won two of three 120-task replicates and finished +11 successes overall with lower token cost. Wall-clock was worse because of browser-use stalls and watchdog restarts, especially in adaptive replicate 2.

Detailed 120-task write-up: `reports/comparison_sweeps/uniform_vs_adaptive_120_replicates.md`
Adaptive 120-task e-process timeline: `reports/comparison_sweeps/adaptive_120_eprocess_timeline.csv`

## 60-Task Pilot

## Completed Runs

| run | budget | successes | rate | cost | wall time | final global log-e | rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| equal-budget uniform | 60 | 39/60 | 65.0% | $3.3710 | 71.5 min | 0.475 | False |
| adaptive SPRUCE | 60 | 38/60 | 63.3% | $3.2436 | 65.8 min | 0.152 | False |
| best full paired arm (`junior_reactive`) | 60 | 42/60 | 70.0% | $3.6692 | 66.0 min | 4.334 | true in per-arm report |

## Bottom Line

The equal-budget uniform policy beat adaptive SPRUCE by one task: 39/60 vs 38/60. Adaptive was slightly cheaper and faster, but it did not improve success rate in this pilot.

The adaptive run concentrated on `planner` and `baseline`. `planner` looked strong inside the adaptive stream at 7/8, but the full paired benchmark found `planner` was the weakest arm at 27/60. Treat this as a pilot showing that the measurement stack works and that single-stream adaptive allocation can be misled by early task-order luck.

Detailed methods/results write-up: `reports/comparison_sweeps/initial_uniform_vs_adaptive_methodology.md`
Row-by-row adaptive e-process timeline: `reports/comparison_sweeps/adaptive_eprocess_timeline.csv`
