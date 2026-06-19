#!/usr/bin/env bash
# Rebuild the uniform multi-arm status and optionally commit compact artifacts.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

python3 - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

status = Path("docs/uniform_multiarm_status.md")
log_dir = Path("logs/uniform_multiarm")
report_dir = Path("reports/uniform_multiarm/round_robin_60")
full_logs = sorted(log_dir.glob("webarena_gmail_uniform_multiarm_60_trial0_*_FULL.jsonl"))
latest_segments = [
    p for p in log_dir.glob("webarena_gmail_uniform_multiarm_60_trial0_*.jsonl")
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
    "# Uniform Multi-Arm Status",
    "",
    "This file tracks the 12-arm, 60-task non-adaptive WebArena Gmail baseline. It",
    "uses the same total task budget as the adaptive run, but allocates round-robin",
    "across all arms.",
    "",
    "## Current Run",
    "",
    "| run | status | durable progress | raw log | report |",
    "|---|---|---:|---|---|",
    f"| uniform_multiarm | {status_text} | {progress} | {raw_log_s} | {report_s} |",
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
    for arm in sorted(arm_counts, key=lambda a: a):
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
        for r in sorted(report_rows, key=lambda x: x["arm_id"]):
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
    "- Sampler: `uniform`, `round_robin`.",
    "- Budget: 60 total tasks, 5 pulls per arm.",
    "- Evidence: same upward per-arm e-process and global linear mixture as adaptive.",
    "",
    "## Live Commands",
    "",
    "```bash",
    "tail -f logs/uniform_multiarm/uniform_stdout.log",
    "tail -f logs/uniform_multiarm/watchdog_uniform_multiarm.log",
    "screen -r csp_uniform_multiarm",
    "```",
    "",
])

status.write_text("\n".join(lines))
PY

if [[ "${COMMIT_FINAL:-0}" == "1" ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        git add docs/uniform_multiarm_status.md configs/webarena_gmail_uniform_multiarm_60.yaml scripts
        git add -f logs/uniform_multiarm/webarena_gmail_uniform_multiarm_60_trial0_*_FULL.jsonl 2>/dev/null || true
        git add -f reports/uniform_multiarm/round_robin_60/run.csv \
            reports/uniform_multiarm/round_robin_60/per_arm.csv \
            reports/uniform_multiarm/round_robin_60/summary.md 2>/dev/null || true
        if ! git diff --cached --quiet; then
            git commit -m "Add uniform multi-arm Gmail sweep"
        fi
    fi
fi
