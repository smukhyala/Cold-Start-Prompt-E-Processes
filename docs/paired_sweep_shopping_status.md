# Shopping Paired Sweep Status

This file tracks the 12-arm paired WebArena Shopping sweep. Each arm should receive the same deterministic Shopping task order, mirroring `docs/paired_sweep_status.md` for Gmail.

## Design

- Domain: WebArena Shopping (`apps/shopping`, `real-tasks`).
- Arms: same prompt catalog as Gmail; single-arm configs point to the corresponding `configs/arms_<arm>_only.yaml`.
- Task order: `sample_mode: cycle`; every arm starts at t=1, so each arm sees the same bank prefix/order.
- Budget: 60 tasks per arm unless the Shopping bank inspection shows fewer than 60 available tasks; if fewer, set `SHOPPING_EXPECTED_TASKS` and config `trial.num_tasks` to that count before running.
- Reward: binary verifier success.
- Future placeholders: `shopping_uniform_multiarm_120`, `shopping_adaptive_spruce_120`, and randomized task-order Shopping replicates should consume these paired logs as the intrinsic-quality baseline.

## Task Bank Validation

Run:

```bash
python scripts/inspect_shopping_task_bank.py
```

This repository does not vendor `webarena-infinity`; the command reports the live count, task IDs, and available metadata when `WEBARENA_INFINITY_ROOT` (or `../webarena-infinity`) is present.

## Cloud Smoke-Test Prerequisites

Yes, the Shopping smoke test is intended to run in the same cloud/remote WebArena environment used for Gmail. The code in this repository is already present once this branch is checked out; the cloud host still needs the runtime pieces that are intentionally not vendored here:

1. A sibling `webarena-infinity` checkout, or `WEBARENA_INFINITY_ROOT` pointing at that checkout.
2. The WebArena Shopping app directory at `$WEBARENA_INFINITY_ROOT/apps/shopping` with the `real-tasks` suite.
3. The Python environment for this repo, including the `webarena` extra/browser-use dependencies.
4. Browser dependencies installed by the WebArena/browser-use setup.
5. The `.env`/secret values needed by the configured browser LLM provider. The Shopping paired configs use OpenAI `gpt-5.4-mini`; the smoke config currently mirrors the Gmail smoke adapter setting and uses Claude unless changed.
6. Writable `logs/paired_sweep_shopping/` and `reports/paired_sweep_shopping/` directories, or a persistent volume if the cloud runner is ephemeral.

Minimal cloud sequence after checkout:

```bash
python scripts/inspect_shopping_task_bank.py
bash scripts/run_paired_shopping_smoke.sh
```

If `python scripts/inspect_shopping_task_bank.py` fails with a missing app path, deploy or mount `webarena-infinity` first rather than starting the smoke run.

Current headline: 12/12 arms complete. Current empirical leader is `baseline` with 35/60 successes (58.3%).

## Completed Results

| rank | arm | successes | rate | difficulty breakdown | cost | wall time | avg min/task | log-e | CS low | CS high |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 35/60 | 58.3% | unknown:35/60 | $2.0148 | 83.9 min | 1.4 | -0.728 | 0.400 | 0.754 |
| 2 | explorer | 33/60 | 55.0% | unknown:33/60 | $1.7886 | 84.9 min | 1.4 | -1.237 | 0.369 | 0.723 |
| 3 | rapid | 33/60 | 55.0% | unknown:33/60 | $1.8677 | 78.6 min | 1.3 | -1.270 | 0.369 | 0.723 |
| 4 | exploratory | 32/60 | 53.3% | unknown:32/60 | $1.5379 | 82.5 min | 1.4 | -1.427 | 0.354 | 0.708 |
| 5 | cautious | 31/60 | 51.7% | unknown:31/60 | $1.8694 | 97.3 min | 1.6 | -1.724 | 0.338 | 0.692 |
| 6 | planner | 30/60 | 50.0% | unknown:30/60 | $1.6974 | 94.8 min | 1.6 | -1.694 | 0.323 | 0.677 |
| 7 | junior_reactive | 29/60 | 48.3% | unknown:29/60 | $1.3581 | 84.0 min | 1.4 | -2.267 | 0.308 | 0.662 |
| 8 | balanced | 29/60 | 48.3% | unknown:29/60 | $1.7065 | 92.8 min | 1.5 | -1.725 | 0.308 | 0.662 |
| 9 | verifier | 27/60 | 45.0% | unknown:27/60 | $1.3482 | 88.6 min | 1.5 | -1.494 | 0.292 | 0.631 |
| 10 | overthinker | 25/60 | 41.7% | unknown:25/60 | $1.3501 | 94.1 min | 1.6 | -1.667 | 0.262 | 0.600 |
| 11 | domain_expert | 24/60 | 40.0% | unknown:24/60 | $1.1533 | 104.8 min | 1.7 | -1.460 | 0.262 | 0.569 |
| 12 | algorithmic | 20/60 | 33.3% | unknown:20/60 | $0.9843 | 96.5 min | 1.6 | -1.278 | 0.200 | 0.508 |

## Current Segment

| order | arm | status | raw log | report |
|---:|---|---|---|---|
| 1 | baseline | done | logs/paired_sweep_shopping/webarena_shopping_pair_baseline_60_trial0_2026-06-23T00-29-51Z_FULL.jsonl | reports/paired_sweep_shopping/baseline |
| 2 | planner | done | logs/paired_sweep_shopping/webarena_shopping_pair_planner_60_trial0_2026-06-23T02-39-21Z_FULL.jsonl | reports/paired_sweep_shopping/planner |
| 3 | cautious | done | logs/paired_sweep_shopping/webarena_shopping_pair_cautious_60_trial0_2026-06-23T05-00-12Z_FULL.jsonl | reports/paired_sweep_shopping/cautious |
| 4 | explorer | done | logs/paired_sweep_shopping/webarena_shopping_pair_explorer_60_trial0_2026-06-23T07-46-51Z_FULL.jsonl | reports/paired_sweep_shopping/explorer |
| 5 | balanced | done | logs/paired_sweep_shopping/webarena_shopping_pair_balanced_60_trial0_2026-06-23T17-13-43Z_FULL.jsonl | reports/paired_sweep_shopping/balanced |
| 6 | overthinker | done | logs/paired_sweep_shopping/webarena_shopping_pair_overthinker_60_trial0_2026-06-23T19-09-56Z_FULL.jsonl | reports/paired_sweep_shopping/overthinker |
| 7 | rapid | done | logs/paired_sweep_shopping/webarena_shopping_pair_rapid_60_trial0_2026-06-23T21-19-59Z_FULL.jsonl | reports/paired_sweep_shopping/rapid |
| 8 | verifier | done | logs/paired_sweep_shopping/webarena_shopping_pair_verifier_60_trial0_2026-06-23T23-01-39Z_FULL.jsonl | reports/paired_sweep_shopping/verifier |
| 9 | exploratory | done | logs/paired_sweep_shopping/webarena_shopping_pair_exploratory_60_trial0_2026-06-24T04-06-47Z_FULL.jsonl | reports/paired_sweep_shopping/exploratory |
| 10 | algorithmic | done | logs/paired_sweep_shopping/webarena_shopping_pair_algorithmic_60_trial0_2026-06-24T07-42-34Z_FULL.jsonl | reports/paired_sweep_shopping/algorithmic |
| 11 | junior_reactive | done | logs/paired_sweep_shopping/webarena_shopping_pair_junior_reactive_60_trial0_2026-06-24T17-28-22Z_FULL.jsonl | reports/paired_sweep_shopping/junior_reactive |
| 12 | domain_expert | done | logs/paired_sweep_shopping/webarena_shopping_pair_domain_expert_60_trial0_2026-06-24T19-27-26Z_FULL.jsonl | reports/paired_sweep_shopping/domain_expert |

## Logging Contract

- `arm_id`
- `task_id`
- `task_meta.difficulty` if available
- `success`
- `reward`
- `steps`
- `wallclock_s`
- token fields (`tokens.input`, `tokens.output`, `tokens.cache_read`, `tokens.cache_write`, `tokens.cost_usd`)
- `policy`
- `per_arm_state`
- `global_e`

## Commands

```bash
# smoke
bash scripts/run_paired_shopping_smoke.sh
# one full arm
bash scripts/run_paired_shopping_arm.sh baseline
# full 12-arm paired sweep
bash scripts/run_paired_shopping_all_arms.sh
# report/status refresh
bash scripts/finalize_paired_sweep_shopping.sh
```
