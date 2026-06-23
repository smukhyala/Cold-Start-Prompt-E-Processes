#!/usr/bin/env bash
# Rebuild the Shopping paired sweep status document.
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_DIR"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
python3 - <<'PY'
import csv,json
from pathlib import Path
arms=['baseline','planner','cautious','explorer','balanced','overthinker','rapid','verifier','exploratory','algorithmic','junior_reactive','domain_expert']
log_dir=Path('logs/paired_sweep_shopping'); rep_root=Path('reports/paired_sweep_shopping')
rows=[]; locations={}
for idx,arm in enumerate(arms,1):
    per=rep_root/arm/'per_arm.csv'; run=rep_root/arm/'run.csv'
    full=sorted(log_dir.glob(f'webarena_shopping_pair_{arm}_60_trial0_*_FULL.jsonl'))
    seg=[p for p in log_dir.glob(f'webarena_shopping_pair_{arm}_60_trial0_*.jsonl') if not p.name.startswith('INVALID_') and not p.name.endswith('_FULL.jsonl') and not p.name.endswith('_MERGED_SO_FAR.jsonl') and p.stat().st_size>0]
    lp=str(full[-1]) if full else (str(max(seg,key=lambda p:p.stat().st_mtime)) if seg else '')
    rd=str(rep_root/arm) if per.exists() else ''
    locations[arm]=(lp,rd)
    if per.exists() and run.exists():
        s=next(csv.DictReader(per.open())); tasks=list(csv.DictReader(run.open()))
        def breakdown():
            out={}
            for r in tasks:
                d=r.get('difficulty','') or 'unknown'; out.setdefault(d,[0,0]); out[d][0]+=int(float(r.get('success',0))); out[d][1]+=1
            return ', '.join(f'{k}:{v[0]}/{v[1]}' for k,v in sorted(out.items())) or 'n/a'
        rows.append({'order':idx,'arm':arm,'successes':int(float(s['successes'])),'pulls':int(float(s['pulls'])),'rate':100*float(s['mu_hat']),'cost':float(s['total_cost_usd']),'wall_min':float(s['total_wallclock_s'])/60,'avg_min':float(s['total_wallclock_s'])/60/max(float(s['pulls']),1),'log_e':float(s['log_e']),'cs_low':float(s['cs_low']),'cs_high':float(s['cs_high']),'diff':breakdown()})
ranked=sorted(rows,key=lambda r:(-r['successes'],r['cost'],r['arm']))
completed={r['arm'] for r in rows if r['pulls']==60}
lines=['# Shopping Paired Sweep Status','', 'This file tracks the 12-arm paired WebArena Shopping sweep. Each arm should receive the same deterministic Shopping task order, mirroring `docs/paired_sweep_status.md` for Gmail.','', '## Design','', '- Domain: WebArena Shopping (`apps/shopping`, `real-tasks`).','- Arms: same prompt catalog as Gmail; single-arm configs point to the corresponding `configs/arms_<arm>_only.yaml`.','- Task order: `sample_mode: cycle`; every arm starts at t=1, so each arm sees the same bank prefix/order.','- Budget: 60 tasks per arm unless the Shopping bank inspection shows fewer than 60 available tasks; if fewer, set `SHOPPING_EXPECTED_TASKS` and config `trial.num_tasks` to that count before running.','- Reward: binary verifier success.','- Future placeholders: `shopping_uniform_multiarm_120`, `shopping_adaptive_spruce_120`, and randomized task-order Shopping replicates should consume these paired logs as the intrinsic-quality baseline.','']
lines += ['## Task Bank Validation','', 'Run:', '', '```bash','python scripts/inspect_shopping_task_bank.py','```','', 'This repository does not vendor `webarena-infinity`; the command reports the live count, task IDs, and available metadata when `WEBARENA_INFINITY_ROOT` (or `../webarena-infinity`) is present.','', '## Cloud Smoke-Test Prerequisites','', 'Yes, the Shopping smoke test is intended to run in the same cloud/remote WebArena environment used for Gmail. The code in this repository is already present once this branch is checked out; the cloud host still needs the runtime pieces that are intentionally not vendored here:', '', '1. A sibling `webarena-infinity` checkout, or `WEBARENA_INFINITY_ROOT` pointing at that checkout.', '2. The WebArena Shopping app directory at `$WEBARENA_INFINITY_ROOT/apps/shopping` with the `real-tasks` suite.', '3. The Python environment for this repo, including the `webarena` extra/browser-use dependencies.', '4. Browser dependencies installed by the WebArena/browser-use setup.', '5. The `.env`/secret values needed by the configured browser LLM provider. The Shopping paired configs use OpenAI `gpt-5.4-mini`; the smoke config currently mirrors the Gmail smoke adapter setting and uses Claude unless changed.', '6. Writable `logs/paired_sweep_shopping/` and `reports/paired_sweep_shopping/` directories, or a persistent volume if the cloud runner is ephemeral.', '', 'Minimal cloud sequence after checkout:', '', '```bash', 'python scripts/inspect_shopping_task_bank.py', 'bash scripts/run_paired_shopping_smoke.sh', '```', '', 'If `python scripts/inspect_shopping_task_bank.py` fails with a missing app path, deploy or mount `webarena-infinity` first rather than starting the smoke run.', '']
if rows:
    best=ranked[0]; lines += [f'Current headline: {len(completed)}/12 arms complete. Current empirical leader is `{best["arm"]}` with {best["successes"]}/60 successes ({best["rate"]:.1f}%).','', '## Completed Results','', '| rank | arm | successes | rate | difficulty breakdown | cost | wall time | avg min/task | log-e | CS low | CS high |','|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for i,r in enumerate(ranked,1): lines.append(f'| {i} | {r["arm"]} | {r["successes"]}/{r["pulls"]} | {r["rate"]:.1f}% | {r["diff"]} | ${r["cost"]:.4f} | {r["wall_min"]:.1f} min | {r["avg_min"]:.1f} | {r["log_e"]:.3f} | {r["cs_low"]:.3f} | {r["cs_high"]:.3f} |')
    lines.append('')
else: lines += ['Current headline: no completed Shopping arm reports were found.','']
lines += ['## Current Segment','','| order | arm | status | raw log | report |','|---:|---|---|---|---|']
for i,arm in enumerate(arms,1):
    lp,rd=locations[arm]; status='done' if arm in completed else 'pending/running'
    lines.append(f'| {i} | {arm} | {status} | {lp} | {rd} |')
lines += ['', '## Logging Contract','', '- `arm_id`','- `task_id`','- `task_meta.difficulty` if available','- `success`','- `reward`','- `steps`','- `wallclock_s`','- token fields (`tokens.input`, `tokens.output`, `tokens.cache_read`, `tokens.cache_write`, `tokens.cost_usd`)','- `policy`','- `per_arm_state`','- `global_e`','', '## Commands','', '```bash','# smoke','bash scripts/run_paired_shopping_smoke.sh','# one full arm','bash scripts/run_paired_shopping_arm.sh baseline','# full 12-arm paired sweep','bash scripts/run_paired_shopping_all_arms.sh','# report/status refresh','bash scripts/finalize_paired_sweep_shopping.sh','```','']
Path('docs/paired_sweep_shopping_status.md').write_text('\n'.join(lines))
PY
if [[ "${COMMIT_FINAL:-0}" == "1" ]]; then
  git add docs/paired_sweep_shopping_status.md configs/webarena_shopping_*.yaml configs/arms_smoke_shopping.yaml scripts/*shopping* || true
  git commit -m "Add Shopping paired sweep infrastructure" || true
fi
