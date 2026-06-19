#!/usr/bin/env bash
# Rebuild the adaptive sweep status and optionally commit compact artifacts.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

python3 - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

status = Path("docs/adaptive_sweep_status.md")
log_dir = Path("logs/adaptive_sweep")
report_dir = Path("reports/adaptive_sweep/spruce_60")
full_logs = sorted(log_dir.glob("webarena_gmail_adaptive_spruce_60_trial0_*_FULL.jsonl"))
latest_segments = [
    p for p in log_dir.glob("webarena_gmail_adaptive_spruce_60_trial0_*.jsonl")
    if not p.name.startswith("INVALID_")
    and not p.name.endswith("_FULL.jsonl")
    and not p.name.endswith("_MERGED_SO_FAR.jsonl")
    and p.stat().st_size > 0
]
raw_log = full_logs[-1] if full_logs else (max(latest_segments, key=lambda p: p.stat().st_mtime) if latest_segments else None)

rows = []
if raw_log and raw_log.exists() and raw_log.stat().st_size > 0:
    rows = [json.loads(line) for line in raw_log.read_text().splitlines() if line.strip()]

progress = f"{len(rows)}/60"
done = bool(full_logs and len(rows) == 60)
status_text = "done" if done else ("running/resumable" if rows else "not started")
raw_log_s = str(raw_log) if raw_log else ""
report_s = str(report_dir) if (report_dir / "run.csv").exists() else ""

lines = [
    "# Adaptive Sweep Status",
    "",
    "This file tracks the 12-arm adaptive WebArena Gmail run. Unlike the uniform",
    "paired sweep, this run has a single 60-task stream and uses the e-process-aware",
    "SPRUCE policy to choose which prompt arm to pull at each task.",
    "",
    "## Current Run",
    "",
    "| run | status | durable progress | raw log | report |",
    "|---|---|---:|---|---|",
    f"| adaptive_spruce | {status_text} | {progress} | {raw_log_s} | {report_s} |",
    "",
]

if rows:
    last = rows[-1]
    arm_counts = Counter(r["arm_id"] for r in rows)
    arm_successes = Counter(r["arm_id"] for r in rows if float(r["reward"]) >= 0.5)
    total_success = sum(1 for r in rows if float(r["reward"]) >= 0.5)
    total_cost = sum(float((r.get("tokens") or {}).get("cost_usd") or 0.0) for r in rows)
    total_wall = sum(float(r.get("wallclock_s") or 0.0) for r in rows)
    global_log_e = float((last.get("global_e") or {}).get("log_e", 0.0))
    global_rejected = bool((last.get("global_e") or {}).get("rejected", False))

    first_reject = None
    for r in rows:
        if bool((r.get("global_e") or {}).get("rejected", False)):
            first_reject = int(r["t"])
            break

    lines.extend([
        "## Headline",
        "",
        f"- Durable tasks: {len(rows)}/60",
        f"- Total successes: {total_success}/{len(rows)} ({100.0 * total_success / len(rows):.1f}%)",
        f"- Total cost: ${total_cost:.4f}",
        f"- Total wall time: {total_wall / 60.0:.1f} min",
        f"- Final global log-e: {global_log_e:.3f}",
        f"- Global null rejected: {global_rejected}",
        f"- First global rejection: {first_reject if first_reject is not None else 'not yet'}",
        "",
        "## Allocation By Arm",
        "",
        "| arm | pulls | successes | rate |",
        "|---|---:|---:|---:|",
    ])
    for arm in sorted(arm_counts, key=lambda a: (-arm_counts[a], a)):
        pulls = arm_counts[arm]
        succ = arm_successes[arm]
        lines.append(f"| {arm} | {pulls} | {succ} | {100.0 * succ / pulls:.1f}% |")
    lines.append("")

    if (report_dir / "per_arm.csv").exists():
        with (report_dir / "per_arm.csv").open() as f:
            report_rows = list(csv.DictReader(f))
        lines.extend([
            "## End-State E-Process By Arm",
            "",
            "| arm | pulls | successes | mu_hat | log-e | CS low | CS high | rejected |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for r in sorted(report_rows, key=lambda x: (-int(float(x["pulls"])), x["arm_id"])):
            lines.append(
                f"| {r['arm_id']} | {int(float(r['pulls']))} | {int(float(r['successes']))} | "
                f"{float(r['mu_hat']):.3f} | {float(r['log_e']):.3f} | "
                f"{float(r['cs_low']):.3f} | {float(r['cs_high']):.3f} | {r['rejected']} |"
            )
        lines.append("")

lines.extend([
    "## Policy",
    "",
    "- Arms: all 12 prompt arms from `configs/arms_initial.yaml`.",
    "- Sampler: `spruce`.",
    "- Evidence signal: per-arm upward e-process log-wealth against `m0 = 0.5`.",
    "- Global evidence: convex linear mixture over all per-arm e-processes.",
    "- Warm start: one pull per arm, then adaptive allocation.",
    "- Tie behavior: deterministic first-arm tie break, so resume paths are stable.",
    "",
    "## Metrics To Compare Against Uniform",
    "",
    "- Successes and success rate over the 60 adaptive tasks.",
    "- Pull allocation by arm.",
    "- First global rejection time, if any.",
    "- Per-arm first rejection times.",
    "- Total token cost and total wall-clock time.",
    "- Token cost and wall-clock time to first rejection.",
    "- Best-arm identification path over time.",
    "",
    "## Live Commands",
    "",
    "```bash",
    "tail -f logs/adaptive_sweep/adaptive_stdout.log",
    "tail -f logs/adaptive_sweep/watchdog_adaptive_spruce.log",
    "screen -r csp_adaptive_spruce",
    "```",
    "",
])

status.write_text("\n".join(lines))
PY

if [[ "${COMMIT_FINAL:-0}" == "1" ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        git add docs/adaptive_sweep_status.md configs/webarena_gmail_adaptive_spruce_60.yaml scripts
        git add -f logs/adaptive_sweep/webarena_gmail_adaptive_spruce_60_trial0_*_FULL.jsonl 2>/dev/null || true
        git add -f reports/adaptive_sweep/spruce_60/run.csv \
            reports/adaptive_sweep/spruce_60/per_arm.csv \
            reports/adaptive_sweep/spruce_60/summary.md 2>/dev/null || true
        if ! git diff --cached --quiet; then
            git commit -m "Add adaptive Gmail SPRUCE sweep"
        fi
    fi
fi
