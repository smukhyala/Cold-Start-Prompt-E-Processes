# Paired Sweep Status

This file tracks the 12-arm paired WebArena Gmail sweep. Each arm should
eventually receive the same 60-task cycle.

## Read This First

The uniform sweep runs each prompt arm on the same 60 Gmail tasks: 20 easy,
20 medium, and 20 hard. A completed arm is comparable to every other completed
arm because it saw the same task set.

Current headline: 12/12 arms are complete. The current best
completed arm is `junior_reactive` with 42/60 successes
(70.0%).

## Completed Results

| rank | arm | successes | rate | easy | medium | hard | cost | wall time | avg min/task | log-e | CS low | CS high |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | junior_reactive | 42/60 | 70.0% | 18/20 | 16/20 | 8/20 | $3.6692 | 66.0 min | 1.1 | 4.334 | 0.523 | 0.892 |
| 2 | exploratory | 39/60 | 65.0% | 16/20 | 14/20 | 9/20 | $3.6455 | 84.4 min | 1.4 | 1.385 | 0.462 | 0.815 |
| 3 | domain_expert | 36/60 | 60.0% | 16/20 | 14/20 | 6/20 | $3.6282 | 82.5 min | 1.4 | -0.141 | 0.415 | 0.800 |
| 4 | baseline | 35/60 | 58.3% | 15/20 | 13/20 | 7/20 | $2.6262 | 58.6 min | 1.0 | -0.582 | 0.400 | 0.800 |
| 5 | rapid | 35/60 | 58.3% | 17/20 | 12/20 | 6/20 | $3.4957 | 72.8 min | 1.2 | -0.646 | 0.400 | 0.754 |
| 6 | balanced | 34/60 | 56.7% | 14/20 | 13/20 | 7/20 | $2.9575 | 76.2 min | 1.3 | -1.434 | 0.385 | 0.769 |
| 7 | verifier | 34/60 | 56.7% | 15/20 | 13/20 | 6/20 | $3.7239 | 86.3 min | 1.4 | -1.327 | 0.385 | 0.785 |
| 8 | cautious | 33/60 | 55.0% | 15/20 | 13/20 | 5/20 | $2.6031 | 74.2 min | 1.2 | -1.626 | 0.369 | 0.769 |
| 9 | overthinker | 33/60 | 55.0% | 16/20 | 11/20 | 6/20 | $2.8748 | 80.7 min | 1.3 | -1.132 | 0.369 | 0.754 |
| 10 | explorer | 33/60 | 55.0% | 16/20 | 13/20 | 4/20 | $2.9363 | 71.8 min | 1.2 | -1.117 | 0.369 | 0.754 |
| 11 | algorithmic | 30/60 | 50.0% | 14/20 | 9/20 | 7/20 | $2.5731 | 85.6 min | 1.4 | -1.733 | 0.323 | 0.708 |
| 12 | planner | 27/60 | 45.0% | 15/20 | 8/20 | 4/20 | $2.6365 | 96.9 min | 1.6 | -1.311 | 0.292 | 0.646 |

Interpretation:

- Rankings are empirical success-rate rankings over the paired 60-task set.
- `cost` and `wall time` are measured from the run logs and should be used
  when comparing efficiency across arms.
- The e-process columns are per-arm evidence against m0 = 0.5; final
  rejection status should be checked in each generated arm report.

## Current Live Run

All 12 arms are complete. No live runner should remain.

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
| 6 | overthinker | done | logs/paired_sweep/webarena_gmail_pair_overthinker_60_trial0_2026-06-18T23-09-57Z_FULL.jsonl | reports/paired_sweep/overthinker |
| 7 | rapid | done | logs/paired_sweep/webarena_gmail_pair_rapid_60_trial0_2026-06-19T00-52-23Z_FULL.jsonl | reports/paired_sweep/rapid |
| 8 | verifier | done | logs/paired_sweep/webarena_gmail_pair_verifier_60_trial0_2026-06-19T02-40-25Z_FULL.jsonl | reports/paired_sweep/verifier |
| 9 | exploratory | done | logs/paired_sweep/webarena_gmail_pair_exploratory_60_trial0_2026-06-19T05-17-59Z_FULL.jsonl | reports/paired_sweep/exploratory |
| 10 | algorithmic | done | logs/paired_sweep/webarena_gmail_pair_algorithmic_60_trial0_2026-06-19T07-26-07Z_FULL.jsonl | reports/paired_sweep/algorithmic |
| 11 | junior_reactive | done | logs/paired_sweep/webarena_gmail_pair_junior_reactive_60_trial0_2026-06-19T15-56-01Z_FULL.jsonl | reports/paired_sweep/junior_reactive |
| 12 | domain_expert | done | logs/paired_sweep/webarena_gmail_pair_domain_expert_60_trial0_2026-06-19T17-14-55Z_FULL.jsonl | reports/paired_sweep/domain_expert |

## Prompt Arms

These labels are shorthand for different prompt vectors. `baseline` is not
no-prompt; it is the simplest direct-action prompt in the catalog.

- `baseline`: mid-level, direct action, no upfront planning, no explicit verification.
- `planner`: stronger planning instruction before action.
- `cautious`: stronger verification/checking behavior.
- `explorer`: more willingness to explore alternatives.
- `balanced`: moderate planning plus moderate verification.
- `overthinker`: maximum planning/verification/expertise/structuredness.
- `rapid`: fast, low-overhead, proactive, outcome-focused.
- `verifier`: light planning with every-step verification pressure.
- `exploratory`: proactive exploration with moderate planning.
- `algorithmic`: domain-expert, plan-then-act, algorithmic format.
- `junior_reactive`: low-expertise, cautious-reactive anchor.
- `domain_expert`: expert-driven, proactive, constraint-aware.

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
