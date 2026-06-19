# Uniform vs Adaptive Comparison Summary

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
