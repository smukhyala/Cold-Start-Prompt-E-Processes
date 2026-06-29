# Shopping Paired Sweep 80-Task Extension Status

This tracks the per-arm Shopping baseline extension from the original 60-task paired prefix to the 80-task bank used by the 240-task uniform/adaptive comparison runs.

## Finding

- The original paired Shopping arm configs were `num_tasks: 60` and all completed exactly `t=1..60`.
- The 240-task Shopping comparison configs describe three cycles of an 80-task bank, so the paired per-arm baseline needs `t=61..80` for direct 80-task coverage.
- Extension runs resume from each arm-specific 60-task FULL log and execute only the remaining 20 tasks.

Current headline: 12/12 arms complete at 80 tasks. Current empirical leader is `rapid` with 37/80 successes (46.2%).

## Completed Results

| rank | arm | successes | rate | tail rows | cost | wall time | log-e |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | rapid | 37/80 | 46.2% | 20 | $2.5794 | 112.8 min | -1.681 |
| 2 | baseline | 37/80 | 46.2% | 20 | $2.9495 | 125.8 min | -1.713 |
| 3 | explorer | 36/80 | 45.0% | 20 | $2.6992 | 124.8 min | -1.633 |
| 4 | exploratory | 35/80 | 43.8% | 20 | $2.4833 | 127.9 min | -1.633 |
| 5 | cautious | 34/80 | 42.5% | 20 | $2.7083 | 141.2 min | -1.824 |
| 6 | planner | 32/80 | 40.0% | 20 | $2.3121 | 136.0 min | -1.711 |
| 7 | junior_reactive | 31/80 | 38.8% | 20 | $2.2843 | 132.3 min | -2.267 |
| 8 | verifier | 30/80 | 37.5% | 20 | $2.2894 | 133.7 min | -1.494 |
| 9 | balanced | 30/80 | 37.5% | 20 | $2.4633 | 135.0 min | -1.725 |
| 10 | domain_expert | 26/80 | 32.5% | 20 | $1.9408 | 152.0 min | -1.460 |
| 11 | overthinker | 25/80 | 31.2% | 20 | $1.9326 | 138.2 min | -1.667 |
| 12 | algorithmic | 23/80 | 28.7% | 20 | $1.4768 | 131.5 min | -1.278 |

## Current Segment

| order | arm | status | raw log | report |
|---:|---|---|---|---|
| 1 | baseline | done | logs/paired_sweep_shopping/webarena_shopping_pair_baseline_60_trial0_2026-06-23T00-29-51Z_FULL80.jsonl | reports/paired_sweep_shopping_80/baseline |
| 2 | planner | done | logs/paired_sweep_shopping/webarena_shopping_pair_planner_60_trial0_2026-06-23T02-39-21Z_FULL80.jsonl | reports/paired_sweep_shopping_80/planner |
| 3 | cautious | done | logs/paired_sweep_shopping/webarena_shopping_pair_cautious_60_trial0_2026-06-23T05-00-12Z_FULL80.jsonl | reports/paired_sweep_shopping_80/cautious |
| 4 | explorer | done | logs/paired_sweep_shopping/webarena_shopping_pair_explorer_60_trial0_2026-06-23T07-46-51Z_FULL80.jsonl | reports/paired_sweep_shopping_80/explorer |
| 5 | balanced | done | logs/paired_sweep_shopping/webarena_shopping_pair_balanced_60_trial0_2026-06-23T17-13-43Z_FULL80.jsonl | reports/paired_sweep_shopping_80/balanced |
| 6 | overthinker | done | logs/paired_sweep_shopping/webarena_shopping_pair_overthinker_60_trial0_2026-06-23T19-09-56Z_FULL80.jsonl | reports/paired_sweep_shopping_80/overthinker |
| 7 | rapid | done | logs/paired_sweep_shopping/webarena_shopping_pair_rapid_60_trial0_2026-06-23T21-19-59Z_FULL80.jsonl | reports/paired_sweep_shopping_80/rapid |
| 8 | verifier | done | logs/paired_sweep_shopping/webarena_shopping_pair_verifier_60_trial0_2026-06-23T23-01-39Z_FULL80.jsonl | reports/paired_sweep_shopping_80/verifier |
| 9 | exploratory | done | logs/paired_sweep_shopping/webarena_shopping_pair_exploratory_60_trial0_2026-06-24T04-06-47Z_FULL80.jsonl | reports/paired_sweep_shopping_80/exploratory |
| 10 | algorithmic | done | logs/paired_sweep_shopping/webarena_shopping_pair_algorithmic_60_trial0_2026-06-24T07-42-34Z_FULL80.jsonl | reports/paired_sweep_shopping_80/algorithmic |
| 11 | junior_reactive | done | logs/paired_sweep_shopping/webarena_shopping_pair_junior_reactive_60_trial0_2026-06-24T17-28-22Z_FULL80.jsonl | reports/paired_sweep_shopping_80/junior_reactive |
| 12 | domain_expert | done | logs/paired_sweep_shopping/webarena_shopping_pair_domain_expert_60_trial0_2026-06-24T19-27-26Z_FULL80.jsonl | reports/paired_sweep_shopping_80/domain_expert |

## Commands

```bash
# one arm
bash scripts/run_paired_shopping_extend_to_80.sh baseline
# all arms
bash scripts/run_paired_shopping_extend_to_80.sh all
# refresh this status file
bash scripts/finalize_paired_sweep_shopping_80.sh
```
