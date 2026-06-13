# Cloud-Agent Runbook — Cold-Start Prompts on WebArena

This is the operational guide for executing the bootstrap H3 experiment
("does adaptive allocation reach `τ_α` faster than uniform?") inside a
Cursor cloud agent or similar unattended environment. The science questions
are answered in `README.md` and the paper; this document is about getting
the run to *finish*.

## What this run does

`cold-start-compare --base configs/webarena_gmail.yaml --policies uniform,ucb,spruce`
runs three policies on the same Gmail-easy task stream with Haiku 4.5,
producing `reports/<base_name>_comparison/comparison.md` with τ_α per
policy, cumulative regret proxy, and overlaid global-log-e trajectories.

## One-time setup on a fresh box

```bash
# 1. Clone both repos as siblings.
git clone <coldStartPrompts repo URL> coldStartPrompts
git clone https://github.com/web-arena-x/webarena-infinity.git
cd coldStartPrompts

# 2. Python deps. Webarena needs uv; cold-start uses pip.
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,webarena]"

# 3. Sibling repo's own setup (installs Chromium for browser-use).
cd ../webarena-infinity && bash setup.sh && cd ../coldStartPrompts

# 4. Env vars.
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

## Pre-flight before any long run

The CLI runs these automatically. They should pass cleanly:

- `ANTHROPIC_API_KEY` set in env (webarena always needs it; orchestrator's
  Anthropic model client also needs it).
- `logs/` writable.
- Port 8001 free (default; a leftover server from a prior aborted run
  is the usual reason it's busy).
- Sibling `../webarena-infinity` repo present (or `WEBARENA_INFINITY_ROOT`
  pointing at one).
- `apps/gmail/real-tasks.json` exists (default app + suite).
- `browser_use` installable from `.[webarena]` extras.

If any fails, `cold-start-run` exits with code 2 and a message naming the
specific check. Fix and rerun. `--skip-preflight` disables the lot — only
use when you know what you're doing.

## Launching the experiment

```bash
source .venv/bin/activate
source .env  # so ANTHROPIC_API_KEY is exported

# Primary command. Logs go to logs/<run_id>.jsonl; comparison artifacts
# to reports/webarena_gmail_comparison/.
nohup python -m cold_start.cli.compare \
    --base configs/webarena_gmail.yaml \
    --policies uniform,ucb,spruce \
    --out reports/webarena_gmail_comparison \
    > /tmp/cold_start.log 2>&1 &
echo $! > /tmp/cold_start.pid
```

## Monitoring while it runs

The orchestrator emits one heartbeat line per task:

```
t=14/60 arm=planner r=1 steps=4 wall=22.3s cum=10/14
  log_e(arm)=0.412 log_e(global)=0.301 ETA=0:25:17
```

**A healthy run produces a new line every ~25-60s.**

Quick checks:

```bash
# Is anything happening?
tail -f /tmp/cold_start.log | grep --line-buffered "t="

# How many tasks have we completed?
grep -c "^.*t=[0-9]" /tmp/cold_start.log

# Is the process alive?
ps -p $(cat /tmp/cold_start.pid)

# Current cumulative cost (USD), per-task average:
python3 -c "
import json, pathlib
runs = list(pathlib.Path('logs').glob('webarena_*comparison*.jsonl'))
total = 0.0; n = 0
for p in sorted(runs):
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        rec = json.loads(line)
        total += rec.get('tokens', {}).get('cost_usd', 0.0)
        n += 1
print(f'tasks: {n}, cumulative cost: \${total:.2f}, per-task: \${total/max(n,1):.3f}')
"
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| No new heartbeat for >5 minutes | Chromium hung; browser-use's `BrowserStartEvent` timed out | Kill via PID; the env teardown is wrapped in `try/finally` so port should release. Then resume (below). |
| `port 8001 already in use` on startup | Prior aborted run | `lsof -ti :8001 \| xargs kill` then rerun. |
| `ANTHROPIC_API_KEY not set` | `.env` not sourced, or key absent | `source .env` then re-launch. |
| `webarena-infinity not found` | Sibling repo missing or env var wrong | `git clone` the sibling next to this repo or set `WEBARENA_INFINITY_ROOT`. |
| `app dir not found: apps/X` | Bad `web_app` in config | Edit YAML to one of `apps/{gmail, linear-account-settings, figma-slides, ...}`. |
| Long-running task burns budget | Likely an arm with no plan looping the same page | Per-task `timeout_s` (currently 180) caps it; the per-arm e-process suppresses the arm on subsequent draws. |
| Anthropic 429 / 529 | Rate limit / capacity | Backoff is handled by the Anthropic SDK; if it persists, kill and resume after a few minutes. |

## Resuming an interrupted run

The framework replays prior records into per-arm e-processes, confidence
sequences, and policy state before continuing.

```bash
# Find where you left off — last "t=" line in the JSONL.
LAST_LOG=$(ls -t logs/webarena_*.jsonl | head -1)
LAST_T=$(tail -1 "$LAST_LOG" | jq -r .t)
NEXT_T=$((LAST_T + 1))

# Resume from the next task.
python -m cold_start.cli.run \
    --config configs/webarena_gmail.yaml \
    --resume-from "$LAST_LOG" \
    --start-at "$NEXT_T"
```

Caveats:

- Only `num_trials=1` is supported for resume; per-policy slices of
  `cold-start-compare` are individually resumable.
- Resume drops the *would-have-been-made* arm choices from the prior tail,
  so policy RNG state diverges. For uniform / memoryless ε-greedy this is
  harmless; for Thompson and SPRUCE the resumed trajectory is a fresh
  sample from the same posterior — still valid, just not byte-identical
  to the original.
- The `_resume_from_N` suffix on the new run_id keeps the resumed log
  separate but joinable (same `run_id` base) to the original.

## Artifacts produced

For each policy in the comparison:

```
reports/webarena_gmail_comparison/
├── comparison.md              ← τ_α, cum. regret, pulls per arm, ETA
├── global_log_e_overlay.png   ← log E_t per policy
├── pulls_by_policy.png        ← per-arm bars per policy
├── uniform/
│   └── <run_id>.jsonl         ← full per-step log
├── ucb/<run_id>.jsonl
└── spruce/<run_id>.jsonl

logs/webarena/<arm_id>/<task_id>/
├── conversations/             ← browser-use's per-step LLM transcripts
├── history.json               ← full agent history
├── result.json                ← webarena verifier output
└── screenshots/               ← step-by-step PNGs
```

Per-task artifact size: ~340 KB `history.json` + a few hundred KB of
screenshots. Plan for ~250 MB across a 5-policy × 60-task run.

## Cost monitoring

Each JSONL record carries `tokens.cost_usd` populated by browser-use's
`TokenCost` service. To get a cumulative figure:

```bash
jq -s 'map(.tokens.cost_usd // 0) | add' logs/<run_id>.jsonl
```

Target: ~$0.10/task on Haiku 4.5 with caching. If you see >$1/task, the
prompt caching is broken (compare `tokens.cache_read` vs `tokens.input` —
should be 60%+ on the second task onwards).

## When the run completes

```bash
# Generate the final report (the comparison harness already does this,
# but rerunnable on the JSONL if you tweak the analysis).
cold-start-report --log <path>.jsonl

# Inspect:
cat reports/webarena_gmail_comparison/comparison.md
open reports/webarena_gmail_comparison/global_log_e_overlay.png
```

The thing to look at first: the `τ_α` table. If `spruce` and `ucb` τ_α
are materially less than `uniform`'s τ_α, **H3 is supported**: adaptive
allocation reaches the same statistical conclusion with fewer samples.
That's the headline result for the paper.

## Stopping cleanly

**Known limitation:** while a task is executing inside `agent.run()`,
browser-use is in `asyncio.wait_for(...)`. SIGINT does not propagate through
the asyncio loop in time for our try/finally to run env.close() reliably.
In practice, expect to force-kill:

```bash
# 1. Try graceful stop first (works if the run is between tasks).
kill -INT $(cat /tmp/cold_start.pid)
sleep 10

# 2. If the process is still alive (it usually is mid-task), force-kill.
kill -9 $(cat /tmp/cold_start.pid)
# Sweep up the orphaned webarena server.
lsof -ti :8001 | xargs -r kill -9
```

Then resume from the last completed `t` in the JSONL (see "Resuming an
interrupted run" above).
