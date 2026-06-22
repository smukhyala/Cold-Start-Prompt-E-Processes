#!/usr/bin/env bash
# Rebuild a compact status markdown for a policy run.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

: "${STATUS_PATH:?}"
: "${LOG_DIR:?}"
: "${TRIAL_NAME:?}"
: "${REPORT_DIR:?}"
: "${TARGET_T:?}"
: "${RUN_LABEL:?}"
: "${RUN_TITLE:?}"
: "${POLICY_DESCRIPTION:?}"

python3 - <<'PY'
import csv
import json
import os
from collections import Counter
from pathlib import Path

status = Path(os.environ["STATUS_PATH"])
log_dir = Path(os.environ["LOG_DIR"])
trial_name = os.environ["TRIAL_NAME"]
report_dir = Path(os.environ["REPORT_DIR"])
target_t = int(os.environ["TARGET_T"])
run_label = os.environ["RUN_LABEL"]
run_title = os.environ["RUN_TITLE"]
policy_description = os.environ["POLICY_DESCRIPTION"]

full_logs = sorted(log_dir.glob(f"{trial_name}_trial0_*_FULL.jsonl"))
segments = [
    p for p in log_dir.glob(f"{trial_name}_trial0_*.jsonl")
    if not p.name.startswith("INVALID_")
    and not p.name.endswith("_FULL.jsonl")
    and not p.name.endswith("_MERGED_SO_FAR.jsonl")
    and p.stat().st_size > 0
]

rows = []
raw_log = None
if full_logs:
    raw_log = full_logs[-1]
    rows = [json.loads(line) for line in raw_log.read_text().splitlines() if line.strip()]
elif segments:
    latest = max(segments, key=lambda p: p.stat().st_mtime)
    first = next(json.loads(line) for line in latest.read_text().splitlines() if line.strip())
    base = first["run_id"].split("_resume_from_")[0]
    by_t = {}
    for p in log_dir.glob(f"{base}*.jsonl"):
        name = p.name
        if name.startswith("INVALID_"):
            continue
        if name.endswith("_FULL.jsonl") or name.endswith("_MERGED_SO_FAR.jsonl"):
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_t[int(rec["t"])] = rec
    rows = [by_t[t] for t in sorted(by_t)]
    if rows:
        raw_log = log_dir / f"{base}_MERGED_SO_FAR.jsonl"
        raw_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

done = bool(full_logs and len(rows) == target_t)
status_text = "done" if done else ("running/resumable" if rows else "not started")
report_s = str(report_dir) if (report_dir / "run.csv").exists() else ""
raw_log_s = str(raw_log) if raw_log else ""

lines = [
    f"# {run_title}",
    "",
    f"This file tracks `{run_label}`.",
    "",
    "## Current Run",
    "",
    "| run | status | durable progress | raw log | report |",
    "|---|---|---:|---|---|",
    f"| {run_label} | {status_text} | {len(rows)}/{target_t} | {raw_log_s} | {report_s} |",
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
        f"- Durable tasks: {len(rows)}/{target_t}",
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

    per_arm = report_dir / "per_arm.csv"
    if per_arm.exists():
        with per_arm.open() as f:
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
    policy_description,
    "",
])

status.write_text("\n".join(lines))
PY
