#!/usr/bin/env bash
# Shared helpers for WebArena Shopping paired sweep runners.
set -Eeuo pipefail

SHOPPING_LOG_DIR="${SHOPPING_LOG_DIR:-logs/paired_sweep_shopping}"
SHOPPING_REPORT_DIR="${SHOPPING_REPORT_DIR:-reports/paired_sweep_shopping}"
SHOPPING_STATUS="${SHOPPING_STATUS:-docs/paired_sweep_shopping_status.md}"
SHOPPING_EXPECTED_TASKS="${SHOPPING_EXPECTED_TASKS:-60}"
SHOPPING_FINALIZE_SCRIPT="${SHOPPING_FINALIZE_SCRIPT:-scripts/finalize_paired_sweep_shopping.sh}"
SHOPPING_ARMS=(baseline planner cautious explorer balanced overthinker rapid verifier exploratory algorithmic junior_reactive domain_expert)

update_shopping_status_row() {
    local arm="$1" status="$2" log_path="${3:-}" report_path="${4:-}"
    [[ -f "$SHOPPING_STATUS" ]] || return 0
    python3 - "$SHOPPING_STATUS" "$arm" "$status" "$log_path" "$report_path" <<'PY'
import sys
from pathlib import Path
path=Path(sys.argv[1]); arm,status,log_path,report_path=sys.argv[2:6]
out=[]
for line in path.read_text().splitlines():
    parts=[p.strip() for p in line.strip().strip('|').split('|')]
    if len(parts)==5 and parts[1]==arm:
        out.append(f"| {parts[0]} | {arm} | {status} | {log_path} | {report_path} |")
    else:
        out.append(line)
path.write_text('\n'.join(out)+'\n')
PY
}

shopping_latest_segment_for() {
    local trial_name="$1"
    python3 - "$SHOPPING_LOG_DIR" "$trial_name" <<'PY'
import pathlib, sys
log_dir=pathlib.Path(sys.argv[1]); trial_name=sys.argv[2]
paths=[]
for p in log_dir.glob(f"{trial_name}_trial0_*.jsonl"):
    n=p.name
    if n.startswith('INVALID_') or n.endswith('_FULL.jsonl') or n.endswith('_MERGED_SO_FAR.jsonl') or p.stat().st_size==0:
        continue
    paths.append(p)
if paths: print(max(paths, key=lambda p:p.stat().st_mtime))
PY
}

shopping_base_run_id_for() { python3 - "$1" <<'PY'
import json,sys
from pathlib import Path
print(json.loads(next(l for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()))['run_id'].split('_resume_from_')[0])
PY
}

shopping_last_t_for() { python3 - "$1" <<'PY'
import json,sys
from pathlib import Path
lines=[l for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
print(json.loads(lines[-1])['t'] if lines else 0)
PY
}

shopping_merge_segments() {
    local base_run_id="$1" suffix="$2"
    python3 - "$SHOPPING_LOG_DIR" "$base_run_id" "$suffix" <<'PY'
import json,pathlib,sys
log_dir=pathlib.Path(sys.argv[1]); base=sys.argv[2]; suffix=sys.argv[3]
segments=[]
for p in log_dir.glob(f"{base}*.jsonl"):
    n=p.name
    if n.startswith('INVALID_') or n.endswith('_FULL.jsonl') or n.endswith('_FULL80.jsonl') or n.endswith('_MERGED_SO_FAR.jsonl') or n.endswith('_MERGED80_SO_FAR.jsonl'): continue
    lines=[l for l in p.read_text().splitlines() if l.strip()]
    if not lines: continue
    segments.append((json.loads(lines[0])['t'], json.loads(lines[-1])['t'], p, lines))
if not segments: raise SystemExit(f'ERROR: no segments found for {base}')
seen=set(); merged=[]
for _,_,p,lines in sorted(segments):
    for line in lines:
        rec=json.loads(line); key=rec['t']
        if key in seen: raise SystemExit(f'ERROR: duplicate t={key} while merging {base}; check {p}')
        seen.add(key); merged.append((key,line))
if sorted(seen)!=list(range(1,max(seen)+1)):
    raise SystemExit(f'ERROR: non-contiguous task rows for {base}')
out=log_dir/f"{base}_{suffix}.jsonl"; out.write_text('\n'.join(line for _,line in sorted(merged))+'\n'); print(out)
PY
}

shopping_export_arm() {
    local arm="$1" full_log="$2" stdout_log="$3" report_dir="$SHOPPING_REPORT_DIR/$arm"
    mkdir -p "$report_dir"
    cold-start-export-csv --log "$full_log" --out-dir "$report_dir" 2>&1 | tee -a "$stdout_log"
    cold-start-report --log "$full_log" --out-dir "$report_dir" 2>&1 | tee -a "$stdout_log"
    update_shopping_status_row "$arm" done "$full_log" "$report_dir"
    bash "$SHOPPING_FINALIZE_SCRIPT"
}

shopping_run_or_resume_arm() {
    local arm="$1" config="$2" trial_name="$3" stdout_log="$SHOPPING_LOG_DIR/${arm}_stdout.log"
    mkdir -p "$SHOPPING_LOG_DIR" "$SHOPPING_REPORT_DIR"
    local latest_log; latest_log="$(shopping_latest_segment_for "$trial_name")"
    if [[ -n "$latest_log" ]]; then
        local base merged count last_t next_t
        base="$(shopping_base_run_id_for "$latest_log")"; merged="$(shopping_merge_segments "$base" MERGED_SO_FAR)"
        count="$(wc -l < "$merged" | tr -d ' ')"; last_t="$(shopping_last_t_for "$merged")"
        if [[ "$count" == "$SHOPPING_EXPECTED_TASKS" ]]; then shopping_export_arm "$arm" "$(shopping_merge_segments "$base" FULL)" "$stdout_log"; return; fi
        next_t=$((last_t+1)); update_shopping_status_row "$arm" "resuming: ${count}/${SHOPPING_EXPECTED_TASKS}; next t=${next_t}" "$merged" ""
        cold-start-run --config "$config" --resume-from "$merged" --start-at "$next_t" 2>&1 | tee -a "$stdout_log"
    else
        update_shopping_status_row "$arm" "running: 0/${SHOPPING_EXPECTED_TASKS}" "" ""
        cold-start-run --config "$config" 2>&1 | tee -a "$stdout_log"
    fi
    latest_log="$(shopping_latest_segment_for "$trial_name")"
    [[ -n "$latest_log" ]] || { update_shopping_status_row "$arm" "failed: no JSONL log found" "" ""; exit 1; }
    local base full final_count
    base="$(shopping_base_run_id_for "$latest_log")"; full="$(shopping_merge_segments "$base" FULL)"; final_count="$(wc -l < "$full" | tr -d ' ')"
    [[ "$final_count" == "$SHOPPING_EXPECTED_TASKS" ]] || { update_shopping_status_row "$arm" "partial: ${final_count}/${SHOPPING_EXPECTED_TASKS}" "$full" ""; exit 1; }
    shopping_export_arm "$arm" "$full" "$stdout_log"
}
