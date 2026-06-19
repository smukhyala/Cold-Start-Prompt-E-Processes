# Uniform Multi-Arm Status

This file tracks the 12-arm, 60-task non-adaptive WebArena Gmail baseline. It
uses the same total task budget as the adaptive run, but allocates round-robin
across all arms.

## Current Run

| run | status | durable progress | raw log | report |
|---|---|---:|---|---|
| uniform_multiarm | not started | 0/60 |  |  |

## Policy

- Arms: all 12 prompt arms from `configs/arms_initial.yaml`.
- Sampler: `uniform`, `round_robin`.
- Budget: 60 total tasks, 5 pulls per arm.
- Evidence: same upward per-arm e-process and global linear mixture as adaptive.

## Live Commands

```bash
tail -f logs/uniform_multiarm/uniform_stdout.log
tail -f logs/uniform_multiarm/watchdog_uniform_multiarm.log
screen -r csp_uniform_multiarm
```
