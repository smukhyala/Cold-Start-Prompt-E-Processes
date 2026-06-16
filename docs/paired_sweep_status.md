# Paired Sweep Status

This file tracks the 12-arm paired WebArena Gmail sweep. Each arm should
eventually receive the same 60-task cycle.

## Current Segment

| order | arm | status | log | report |
|---:|---|---|---|---|
| 1 | baseline | pending: GPT infra ready |  |  |
| 2 | planner | pending |  |  |
| 3 | cautious | pending |  |  |
| 4 | explorer | pending |  |  |
| 5 | balanced | pending |  |  |
| 6 | overthinker | pending |  |  |
| 7 | rapid | pending |  |  |
| 8 | verifier | pending |  |  |
| 9 | exploratory | pending |  |  |
| 10 | algorithmic | pending |  |  |
| 11 | junior_reactive | pending |  |  |
| 12 | domain_expert | pending |  |  |

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
