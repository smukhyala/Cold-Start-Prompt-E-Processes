#!/usr/bin/env bash
# Resume/run arms 3 and 4 of the paired WebArena Gmail sweep:
#   3. cautious
#   4. explorer

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
set -a
source .env
set +a

mkdir -p logs/paired_sweep reports/paired_sweep

STATUS="docs/paired_sweep_status.md"
TRACE_LOG="logs/paired_sweep/resume_driver_arms_3_4_trace.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] arms 3-4 resume wrapper started pid=$$" >> "$TRACE_LOG"

trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed at line ${LINENO}: ${BASH_COMMAND}" >> "$TRACE_LOG"' ERR

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

latest_segment_for() {
    local trial_name="$1"
    python3 - "$trial_name" <<'PY'
import pathlib
import sys

trial_name = sys.argv[1]
paths = []
for p in pathlib.Path("logs/paired_sweep").glob(f"{trial_name}_trial0_*.jsonl"):
    name = p.name
    if name.startswith("INVALID_"):
        continue
    if name.endswith("_FULL.jsonl") or name.endswith("_MERGED_SO_FAR.jsonl"):
        continue
    if p.stat().st_size == 0:
        continue
    paths.append(p)
if paths:
    print(max(paths, key=lambda p: p.stat().st_mtime))
PY
}

base_run_id_for() {
    local log_path="$1"
    python3 - "$log_path" <<'PY'
import json
import sys
from pathlib import Path

line = next(l for l in Path(sys.argv[1]).read_text().splitlines() if l.strip())
print(json.loads(line)["run_id"].split("_resume_from_")[0])
PY
}

last_t_for() {
    local log_path="$1"
    python3 - "$log_path" <<'PY'
import json
import sys
from pathlib import Path

lines = [l for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
print(json.loads(lines[-1])["t"] if lines else 0)
PY
}

merge_segments() {
    local base_run_id="$1"
    local suffix="$2"
    python3 - "$base_run_id" "$suffix" <<'PY'
import json
import pathlib
import sys

base, suffix = sys.argv[1:3]
log_dir = pathlib.Path("logs/paired_sweep")
segments = []
for p in log_dir.glob(f"{base}*.jsonl"):
    name = p.name
    if name.startswith("INVALID_"):
        continue
    if name.endswith("_FULL.jsonl") or name.endswith("_MERGED_SO_FAR.jsonl"):
        continue
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if not lines:
        continue
    first_t = json.loads(lines[0])["t"]
    last_t = json.loads(lines[-1])["t"]
    segments.append((first_t, last_t, p, lines))

if not segments:
    raise SystemExit(f"ERROR: no segments found for {base}")

segments.sort(key=lambda item: (item[0], item[1], item[2].name))
seen = set()
merged = []
for _, _, p, lines in segments:
    for line in lines:
        t = json.loads(line)["t"]
        if t in seen:
            raise SystemExit(f"ERROR: duplicate t={t} while merging {base}; check {p}")
        seen.add(t)
        merged.append((t, line))

merged.sort(key=lambda item: item[0])
expected = list(range(1, max(seen) + 1))
if sorted(seen) != expected:
    missing = sorted(set(expected) - seen)
    raise SystemExit(f"ERROR: non-contiguous task rows for {base}; missing {missing[:10]}")

out_path = log_dir / f"{base}_{suffix}.jsonl"
out_path.write_text("\n".join(line for _, line in merged) + "\n")
print(out_path)
PY
}

export_arm() {
    local arm="$1"
    local full_log="$2"
    local stdout_log="$3"
    local report_dir="reports/paired_sweep/${arm}"

    mkdir -p "$report_dir"
    cold-start-export-csv --log "$full_log" --out-dir "$report_dir" 2>&1 | tee -a "$stdout_log"
    cold-start-report --log "$full_log" --out-dir "$report_dir" 2>&1 | tee -a "$stdout_log"
    update_status "$arm" "done" "$full_log" "$report_dir"
}

run_or_resume_arm() {
    local arm="$1"
    local config="$2"
    local trial_name="$3"
    local stdout_log="logs/paired_sweep/${arm}_stdout.log"

    local latest_log
    latest_log="$(latest_segment_for "$trial_name")"

    if [[ -n "$latest_log" ]]; then
        local base_run_id merged_so_far last_t next_t count
        base_run_id="$(base_run_id_for "$latest_log")"
        merged_so_far="$(merge_segments "$base_run_id" "MERGED_SO_FAR")"
        count="$(wc -l < "$merged_so_far" | tr -d ' ')"
        last_t="$(last_t_for "$merged_so_far")"

        if [[ "$count" == "60" ]]; then
            local full_log
            full_log="$(merge_segments "$base_run_id" "FULL")"
            export_arm "$arm" "$full_log" "$stdout_log"
            return
        fi

        next_t=$((last_t + 1))
        update_status "$arm" "resuming: ${count}/60; next t=${next_t}" "$merged_so_far" ""
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] resuming $arm from $merged_so_far at t=$next_t" | tee -a "$stdout_log"
        cold-start-run --config "$config" --resume-from "$merged_so_far" --start-at "$next_t" 2>&1 | tee -a "$stdout_log"
    else
        update_status "$arm" "running: 0/60" "" ""
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting fresh $arm with $config" | tee -a "$stdout_log"
        cold-start-run --config "$config" 2>&1 | tee -a "$stdout_log"
    fi

    latest_log="$(latest_segment_for "$trial_name")"
    if [[ -z "$latest_log" ]]; then
        update_status "$arm" "failed: no JSONL log found" "" ""
        echo "ERROR: no log found for $trial_name" >&2
        exit 1
    fi

    local base_run_id full_log final_count
    base_run_id="$(base_run_id_for "$latest_log")"
    full_log="$(merge_segments "$base_run_id" "FULL")"
    final_count="$(wc -l < "$full_log" | tr -d ' ')"
    if [[ "$final_count" != "60" ]]; then
        update_status "$arm" "partial: ${final_count}/60" "$full_log" ""
        echo "ERROR: $arm has $final_count records after merge, expected 60: $full_log" >&2
        exit 1
    fi

    export_arm "$arm" "$full_log" "$stdout_log"
}

run_or_resume_arm \
    "cautious" \
    "configs/webarena_gmail_pair_cautious_60.yaml" \
    "webarena_gmail_pair_cautious_60"

run_or_resume_arm \
    "explorer" \
    "configs/webarena_gmail_pair_explorer_60.yaml" \
    "webarena_gmail_pair_explorer_60"

echo "paired arms 3-4 resume/run complete"
