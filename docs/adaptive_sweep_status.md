# Adaptive Sweep Status

This file tracks the 12-arm adaptive WebArena Gmail run. Unlike the uniform
paired sweep, this run has a single 60-task stream and uses the e-process-aware
SPRUCE policy to choose which prompt arm to pull at each task.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| adaptive_spruce | not started | 0/60 |  |  |

## Policy

- Arms: all 12 prompt arms from `configs/arms_initial.yaml`.
- Sampler: `spruce`.
- Evidence signal: per-arm upward e-process log-wealth against `m0 = 0.5`.
- Global evidence: convex linear mixture over all per-arm e-processes.
- Warm start: one pull per arm, then adaptive allocation.
- Tie behavior: deterministic first-arm tie break, so resume paths are stable.

## Metrics To Compare Against Uniform

- Successes and success rate over the 60 adaptive tasks.
- Pull allocation by arm.
- First global rejection time, if any.
- Per-arm first rejection times.
- Total token cost and total wall-clock time.
- Token cost and wall-clock time to first rejection.
- Best-arm identification path over time.

## Live Commands

```bash
tail -f logs/adaptive_sweep/adaptive_stdout.log
tail -f logs/adaptive_sweep/watchdog_adaptive_spruce.log
screen -r csp_adaptive_spruce
```
