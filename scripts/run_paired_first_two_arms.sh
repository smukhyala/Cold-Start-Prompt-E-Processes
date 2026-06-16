#!/usr/bin/env bash
# Run the first two arms of the paired WebArena Gmail sweep.
#
# Each arm gets the full 60-task cycle in a single-arm config:
#   1. baseline
#   2. planner
#
# The script writes:
#   - per-task JSONL logs under logs/paired_sweep/
#   - stdout/stderr logs under logs/paired_sweep/
#   - reports/CSVs under reports/paired_sweep/<arm>/
#   - arm completion status to docs/paired_sweep_status.md

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
set -a
source .env
set +a

mkdir -p logs/paired_sweep reports/paired_sweep

STATUS="docs/paired_sweep_status.md"
PIDFILE="/tmp/cold_start_paired_first_two.pid"
echo $$ > "$PIDFILE"
TRACE_LOG="logs/paired_sweep/driver_trace.log"
trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed at line ${LINENO}: ${BASH_COMMAND}" >> "$TRACE_LOG"' ERR
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wrapper started pid=$$" >> "$TRACE_LOG"

update_status() {
    local arm="$1"
    local status="$2"
    local log_path="${3:-}"
    local report_path="${4:-}"
    python3 -c '
import sys
from pathlib import Path

path = Path(sys.argv[1])
arm, status, log_path, report_path = sys.argv[2:6]
out = []
for line in path.read_text().splitlines():
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) == 5 and parts[1] == arm:
        out.append(f"| {parts[0]} | {arm} | {status} | {log_path} | {report_path} |")
    else:
        out.append(line)
path.write_text("\n".join(out) + "\n")
' "$STATUS" "$arm" "$status" "$log_path" "$report_path"
}

latest_log_for() {
    local trial_name="$1"
    ls -t "logs/paired_sweep/${trial_name}"_trial0_*.jsonl 2>/dev/null | head -1
}

run_one_arm() {
    local arm="$1"
    local config="$2"
    local trial_name="$3"

    local stdout_log="logs/paired_sweep/${arm}_stdout.log"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${arm}: marking running" >> "$TRACE_LOG"
    update_status "$arm" "running" "" ""
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${arm}: launching cold-start-run" >> "$TRACE_LOG"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting $arm with $config" | tee -a "$stdout_log"
    cold-start-run --config "$config" 2>&1 | tee -a "$stdout_log"

    local log_path
    log_path="$(latest_log_for "$trial_name")"
    if [[ -z "$log_path" ]]; then
        update_status "$arm" "failed: no JSONL log found" "" ""
        echo "ERROR: no log found for $trial_name" >&2
        exit 1
    fi

    local count
    count="$(wc -l < "$log_path" | tr -d ' ')"
    if [[ "$count" != "60" ]]; then
        update_status "$arm" "partial: ${count}/60" "$log_path" ""
        echo "ERROR: $arm produced $count records, expected 60: $log_path" >&2
        exit 1
    fi

    local report_dir="reports/paired_sweep/${arm}"
    mkdir -p "$report_dir"
    cold-start-export-csv --log "$log_path" --out-dir "$report_dir" 2>&1 | tee -a "$stdout_log"
    cold-start-report --log "$log_path" --out-dir "$report_dir" 2>&1 | tee -a "$stdout_log"

    update_status "$arm" "done" "$log_path" "$report_dir"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] finished $arm: $log_path" | tee -a "$stdout_log"
}

run_one_arm \
    "baseline" \
    "configs/webarena_gmail_pair_baseline_60.yaml" \
    "webarena_gmail_pair_baseline_60"

run_one_arm \
    "planner" \
    "configs/webarena_gmail_pair_planner_60.yaml" \
    "webarena_gmail_pair_planner_60"

echo "paired first-two run complete"
