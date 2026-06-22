#!/usr/bin/env bash
# Generic resume/run wrapper for one WebArena policy config.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

: "${CONFIG:?}"
: "${TRIAL_NAME:?}"
: "${LOG_DIR:?}"
: "${REPORT_DIR:?}"
: "${STDOUT_LOG:?}"
: "${TARGET_T:?}"
: "${FINALIZE_SCRIPT:?}"

source .venv/bin/activate
set -a
source .env
set +a

mkdir -p "$LOG_DIR" "$REPORT_DIR"

TRACE_LOG="${LOG_DIR}/resume_trace.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] resume wrapper started trial=${TRIAL_NAME} pid=$$" >> "$TRACE_LOG"
trap 'echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed at line ${LINENO}: ${BASH_COMMAND}" >> "$TRACE_LOG"' ERR

latest_segment() {
    python3 - "$LOG_DIR" "$TRIAL_NAME" <<'PY'
import pathlib
import sys

log_dir = pathlib.Path(sys.argv[1])
trial_name = sys.argv[2]
paths = []
for p in log_dir.glob(f"{trial_name}_trial0_*.jsonl"):
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
    python3 - "$LOG_DIR" "$base_run_id" "$suffix" <<'PY'
import json
import pathlib
import sys

log_dir = pathlib.Path(sys.argv[1])
base, suffix = sys.argv[2:4]
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

export_run() {
    local full_log="$1"
    mkdir -p "$REPORT_DIR"
    cold-start-export-csv --log "$full_log" --out-dir "$REPORT_DIR" 2>&1 | tee -a "$STDOUT_LOG"
    cold-start-report --log "$full_log" --out-dir "$REPORT_DIR" 2>&1 | tee -a "$STDOUT_LOG"
    bash "$FINALIZE_SCRIPT"
}

latest_log="$(latest_segment)"
if [[ -n "$latest_log" ]]; then
    base_run_id="$(base_run_id_for "$latest_log")"
    merged_so_far="$(merge_segments "$base_run_id" "MERGED_SO_FAR")"
    count="$(wc -l < "$merged_so_far" | tr -d ' ')"
    last_t="$(last_t_for "$merged_so_far")"

    if [[ "$count" == "$TARGET_T" ]]; then
        full_log="$(merge_segments "$base_run_id" "FULL")"
        export_run "$full_log"
        echo "${TRIAL_NAME} resume/run complete"
        exit 0
    fi

    next_t=$((last_t + 1))
    bash "$FINALIZE_SCRIPT"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] resuming ${TRIAL_NAME} from ${merged_so_far} at t=${next_t}" | tee -a "$STDOUT_LOG"
    cold-start-run --config "$CONFIG" --resume-from "$merged_so_far" --start-at "$next_t" 2>&1 | tee -a "$STDOUT_LOG"
else
    bash "$FINALIZE_SCRIPT"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${TRIAL_NAME} with ${CONFIG}" | tee -a "$STDOUT_LOG"
    cold-start-run --config "$CONFIG" 2>&1 | tee -a "$STDOUT_LOG"
fi

latest_log="$(latest_segment)"
if [[ -z "$latest_log" ]]; then
    echo "ERROR: no JSONL log found for ${TRIAL_NAME}" >&2
    exit 1
fi

base_run_id="$(base_run_id_for "$latest_log")"
full_log="$(merge_segments "$base_run_id" "FULL")"
final_count="$(wc -l < "$full_log" | tr -d ' ')"
if [[ "$final_count" != "$TARGET_T" ]]; then
    bash "$FINALIZE_SCRIPT"
    echo "ERROR: ${TRIAL_NAME} has ${final_count} records after merge, expected ${TARGET_T}: ${full_log}" >&2
    exit 1
fi

export_run "$full_log"
echo "${TRIAL_NAME} resume/run complete"
