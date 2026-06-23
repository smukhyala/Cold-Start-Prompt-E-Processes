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

Current headline: 1/12 arms complete. Current empirical leader is `baseline` with 35/60 successes (58.3%).

## Completed Results

| rank | arm | successes | rate | difficulty breakdown | cost | wall time | avg min/task | log-e | CS low | CS high |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 35/60 | 58.3% | unknown:35/60 | $2.0148 | 83.9 min | 1.4 | -0.728 | 0.400 | 0.754 |

## Current Segment

| order | arm | status | raw log | report |
|---:|---|---|---|---|
| 1 | baseline | done | logs/paired_sweep_shopping/webarena_shopping_pair_baseline_60_trial0_2026-06-23T00-29-51Z_FULL.jsonl | reports/paired_sweep_shopping/baseline |
| 2 | planner | resuming: 46/60; next t=47 | logs/paired_sweep_shopping/webarena_shopping_pair_planner_60_trial0_2026-06-23T02-39-21Z_MERGED_SO_FAR.jsonl |  |
| 3 | cautious | pending/running |  |  |
| 4 | explorer | pending/running |  |  |
| 5 | balanced | pending/running |  |  |
| 6 | overthinker | pending/running |  |  |
| 7 | rapid | pending/running |  |  |
| 8 | verifier | pending/running |  |  |
| 9 | exploratory | pending/running |  |  |
| 10 | algorithmic | pending/running |  |  |
| 11 | junior_reactive | pending/running |  |  |
| 12 | domain_expert | pending/running |  |  |

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
