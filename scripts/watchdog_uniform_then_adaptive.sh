#!/usr/bin/env bash
# Watchdog for sequential uniform-multiarm then adaptive-SPRUCE Gmail runs.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

POLL_SECS="${POLL_SECS:-120}"
STALL_SECS="${STALL_SECS:-600}"
NO_LOG_STALL_SECS="${NO_LOG_STALL_SECS:-300}"
STARTUP_GRACE_SECS="${STARTUP_GRACE_SECS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-80}"
SCREEN_NAME="${SCREEN_NAME:-csp_uniform_then_adaptive}"
WATCHDOG_LOG="${WATCHDOG_LOG:-logs/comparison_sweeps/watchdog_uniform_then_adaptive.log}"
WRAPPER_STDOUT="${WRAPPER_STDOUT:-logs/comparison_sweeps/uniform_then_adaptive_stdout.log}"
WRAPPER_SCRIPT="${WRAPPER_SCRIPT:-scripts/resume_uniform_then_adaptive.sh}"

mkdir -p logs/comparison_sweeps

log() { printf '[watchdog-comparison %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$WATCHDOG_LOG"; }

screen_alive() {
    screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${SCREEN_NAME}$"
}

runner_alive() {
    pgrep -f 'cold-start-run --config configs/webarena_gmail_(uniform_multiarm|adaptive_spruce)_60[.]yaml' >/dev/null 2>&1
}

wrapper_alive() {
    screen_alive || pgrep -f '[r]esume_uniform_then_adaptive[.]sh' >/dev/null 2>&1 || runner_alive
}

both_done() {
    python3 - <<'PY'
from pathlib import Path

checks = [
    (Path("docs/uniform_multiarm_status.md"), "uniform_multiarm"),
    (Path("docs/adaptive_sweep_status.md"), "adaptive_spruce"),
]
for path, run in checks:
    if not path.exists():
        raise SystemExit(1)
    ok = False
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 5 and parts[0] == run and parts[1] == "done":
            ok = True
            break
    if not ok:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

active_log() {
    python3 - <<'PY'
import pathlib

specs = [
    (pathlib.Path("logs/adaptive_sweep"), "webarena_gmail_adaptive_spruce_60"),
    (pathlib.Path("logs/uniform_multiarm"), "webarena_gmail_uniform_multiarm_60"),
]
paths = []
for log_dir, prefix in specs:
    for p in log_dir.glob(f"{prefix}_trial0_*.jsonl"):
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

last_t() {
    local log_path="$1"
    [[ -s "$log_path" ]] || { echo 0; return; }
    tail -1 "$log_path" | python3 -c '
import json, sys
try:
    print(json.loads(sys.stdin.read())["t"])
except Exception:
    print(0)
' 2>/dev/null
}

row_count() {
    local log_path="$1"
    [[ -f "$log_path" ]] || { echo 0; return; }
    wc -l < "$log_path" | tr -d " "
}

restart_wrapper() {
    log "stopping screen=${SCREEN_NAME}, comparison runners, and port 8001"
    screen -ls 2>/dev/null | awk -v name="$SCREEN_NAME" '$1 ~ "^[0-9]+[.]" name "$" {print $1}' | while read -r session; do
        screen -S "$session" -X quit 2>/dev/null || true
    done
    pgrep -f 'cold-start-run --config configs/webarena_gmail_(uniform_multiarm|adaptive_spruce)_60[.]yaml' 2>/dev/null | while read -r pid; do
        [[ "$pid" == "$$" ]] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
    pgrep -f '[r]esume_(uniform_multiarm|adaptive_spruce|uniform_then_adaptive)[.]sh' 2>/dev/null | while read -r pid; do
        [[ "$pid" == "$$" ]] && continue
        kill "$pid" 2>/dev/null || true
    done
    kill -9 $(lsof -ti :8001 2>/dev/null) 2>/dev/null || true
    sleep 3

    log "relaunching ${WRAPPER_SCRIPT} in screen=${SCREEN_NAME}"
    screen -dmS "$SCREEN_NAME" bash -lc "cd '$REPO_DIR' && bash '$WRAPPER_SCRIPT' > '$WRAPPER_STDOUT' 2>&1"
    grace_until=$(( $(date +%s) + STARTUP_GRACE_SECS ))
    if wrapper_alive; then
        log "wrapper relaunch confirmed"
    else
        log "WARNING: wrapper relaunch was not visible after screen start"
    fi
}

log "starting; poll=${POLL_SECS}s stall=${STALL_SECS}s no_log_stall=${NO_LOG_STALL_SECS}s startup_grace=${STARTUP_GRACE_SECS}s max_restarts=${MAX_RESTARTS}"

restarts=0
no_log_since=""
grace_until=$(( $(date +%s) + STARTUP_GRACE_SECS ))
while true; do
    if both_done; then
        log "uniform and adaptive runs are done; exiting"
        exit 0
    fi

    log_path="$(active_log)"
    if [[ -z "$log_path" ]]; then
        now="$(date +%s)"
        if [[ -z "$no_log_since" ]]; then
            no_log_since="$now"
        fi
        no_log_gap=$((now - no_log_since))
        log "no active JSONL found yet"
        if ! wrapper_alive || (( no_log_gap > NO_LOG_STALL_SECS )); then
            if (( restarts >= MAX_RESTARTS )); then
                log "GIVE UP: no active JSONL and max restarts reached"
                exit 2
            fi
            restart_wrapper
            restarts=$((restarts + 1))
            no_log_since="$(date +%s)"
        fi
        sleep "$POLL_SECS"
        continue
    fi
    no_log_since=""

    now="$(date +%s)"
    mtime="$(stat -f %m "$log_path" 2>/dev/null || echo "$now")"
    gap=$((now - mtime))
    lt="$(last_t "$log_path")"
    rows="$(row_count "$log_path")"

    if wrapper_alive; then
        if (( now < grace_until && gap > STALL_SECS )); then
            log "grace: ${log_path} rows=${rows} last_t=${lt} stale ${gap}s, but runner is inside startup grace"
        elif (( gap > STALL_SECS )); then
            log "STALL: ${log_path} rows=${rows} last_t=${lt} last grew ${gap}s ago"
            if (( restarts >= MAX_RESTARTS )); then
                log "GIVE UP: hit MAX_RESTARTS=${MAX_RESTARTS}"
                exit 2
            fi
            restart_wrapper
            restarts=$((restarts + 1))
            log "restart #${restarts}/${MAX_RESTARTS} complete"
        else
            log "alive: ${log_path} rows=${rows} last_t=${lt} last grew ${gap}s ago"
        fi
    else
        log "wrapper screen missing while run incomplete; restarting"
        if (( restarts >= MAX_RESTARTS )); then
            log "GIVE UP: hit MAX_RESTARTS=${MAX_RESTARTS}"
            exit 2
        fi
        restart_wrapper
        restarts=$((restarts + 1))
        log "restart #${restarts}/${MAX_RESTARTS} complete"
    fi

    sleep "$POLL_SECS"
done
