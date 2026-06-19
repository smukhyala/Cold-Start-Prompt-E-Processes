# Paired Sweep Status

This file tracks the 12-arm paired WebArena Gmail sweep. Each arm should
eventually receive the same 60-task cycle.

## Read This First

The uniform sweep runs each prompt arm on the same 60 Gmail tasks: 20 easy,
20 medium, and 20 hard. A completed arm is comparable to every other completed
arm because it saw the same task set.

Current headline: the first five arms are complete. `baseline` is still the
best completed arm by raw success rate, with `balanced` very close behind.
The `overthinker` arm is currently running.

## Completed Results

| rank | arm | successes | rate | easy | medium | hard | cost | wall time | avg min/task | log-e | CS low | CS high |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 35/60 | 58.3% | 15/20 | 13/20 | 7/20 | $2.6262 | 58.6 min | 1.0 | -0.582 | 0.400 | 0.800 |
| 2 | balanced | 34/60 | 56.7% | 14/20 | 13/20 | 7/20 | $2.9575 | 76.2 min | 1.3 | -1.434 | 0.385 | 0.769 |
| 3 | cautious | 33/60 | 55.0% | 15/20 | 13/20 | 5/20 | $2.6031 | 74.2 min | 1.2 | -1.626 | 0.369 | 0.769 |
| 3 | explorer | 33/60 | 55.0% | 16/20 | 13/20 | 4/20 | $2.9363 | 71.8 min | 1.2 | -1.117 | 0.369 | 0.754 |
| 5 | planner | 27/60 | 45.0% | 15/20 | 8/20 | 4/20 | $2.6365 | 96.9 min | 1.6 | -1.311 | 0.292 | 0.646 |

Interpretation:

- The completed arms are clustered except for `planner`, which is clearly
  weaker in this first uniform batch.
- `baseline` leads by one task over `balanced`, so the current lead is small.
- `balanced` tied `baseline` on hard tasks but lost one easy task.
- `planner` was the slowest completed arm and also had the lowest accuracy.
- None of these single-arm 60-task runs rejected the null at alpha 0.05 on
  the final e-process value, so treat this as empirical ranking rather than
  strong sequential evidence.

## Current Live Run

| arm | status | durable progress | notes |
|---|---|---:|---|
| overthinker | running | 40/60 as of last manual check | Recovered once from a post-task stall at row 20; live count may already be higher. |

Live commands:

```bash
tail -f logs/paired_sweep/overthinker_stdout.log
tail -f logs/paired_sweep/watchdog_paired_arms_5_6.log
wc -l logs/paired_sweep/webarena_gmail_pair_overthinker_60_trial0_*.jsonl
```

## Current Segment

Use this table for file locations and completion state. The JSONL is the raw
source of truth; the report directory contains `run.csv`, `per_arm.csv`, and
the generated report for that arm.

| order | arm | status | raw log | report |
|---:|---|---|---|---|
| 1 | baseline | done | logs/paired_sweep/webarena_gmail_pair_baseline_60_trial0_2026-06-16T18-47-31Z_FULL.jsonl | reports/paired_sweep/baseline |
| 2 | planner | done | logs/paired_sweep/webarena_gmail_pair_planner_60_trial0_2026-06-16T20-03-52Z_FULL.jsonl | reports/paired_sweep/planner |
| 3 | cautious | done | logs/paired_sweep/webarena_gmail_pair_cautious_60_trial0_2026-06-18T17-26-29Z_FULL.jsonl | reports/paired_sweep/cautious |
| 4 | explorer | done | logs/paired_sweep/webarena_gmail_pair_explorer_60_trial0_2026-06-18T19-02-05Z_FULL.jsonl | reports/paired_sweep/explorer |
| 5 | balanced | done | logs/paired_sweep/webarena_gmail_pair_balanced_60_trial0_2026-06-18T21-31-55Z_FULL.jsonl | reports/paired_sweep/balanced |
| 6 | overthinker | resuming: 40/60; next t=41 | logs/paired_sweep/webarena_gmail_pair_overthinker_60_trial0_2026-06-18T23-09-57Z_MERGED_SO_FAR.jsonl |  |
| 7 | rapid | pending |  |  |
| 8 | verifier | pending |  |  |
| 9 | exploratory | pending |  |  |
| 10 | algorithmic | pending |  |  |
| 11 | junior_reactive | pending |  |  |
| 12 | domain_expert | pending |  |  |

## Prompt Arms

These labels are shorthand for different prompt vectors. `baseline` is not
no-prompt; it is the simplest direct-action prompt in the catalog.

- `baseline`: mid-level, direct action, no upfront planning, no explicit verification.
- `planner`: stronger planning instruction before action.
- `cautious`: stronger verification/checking behavior.
- `explorer`: more willingness to explore alternatives.
- `balanced`: moderate planning plus moderate verification.
- `overthinker`: maximum planning/verification/expertise/structuredness; currently running.

## Logging Contract

For each completed task, the JSONL log must include:

- `arm_id`
- `task_id`
- `task_meta.difficulty`
- `success`
- `reward`
- `steps`
- `wallclock_s`
- `tokens.input`
- `tokens.output`
- `tokens.cache_read`
- `tokens.cache_write`
- `tokens.cost_usd`
- `policy`
- `per_arm_state`
- `global_e`

The runner flushes each JSONL record immediately after writing it, so completed
task data remains available if a later task fails.

## Invalid Attempts

- `logs/paired_sweep/INVALID_connection_webarena_gmail_pair_baseline_60_trial0_2026-06-16T18-12-11Z.jsonl`
  contains 2 baseline records from a workplace network where Anthropic/Claude
  connections were blocked. These records should not be used as research data.
