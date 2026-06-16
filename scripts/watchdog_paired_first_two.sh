#!/usr/bin/env bash
# Watchdog for the paired first-two WebArena Gmail sweep.
#
# It monitors the active baseline/planner JSONL under logs/paired_sweep. If the
# wrapper is alive but the active JSONL has not grown for STALL_SECS, it kills
# the detached screen run, frees the WebArena port, and relaunches the
# resume-aware driver. The driver then resumes from the last durable JSONL row.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

POLL_SECS="${POLL_SECS:-120}"
STALL_SECS="${STALL_SECS:-720}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
SCREEN_NAME="${SCREEN_NAME:-csp_paired_first_two_resume}"
WATCHDOG_PIDFILE="${WATCHDOG_PIDFILE:-/tmp/cold_start_paired_watchdog.pid}"
WATCHDOG_LOG="${WATCHDOG_LOG:-logs/paired_sweep/watchdog_paired.log}"
WRAPPER_STDOUT="${WRAPPER_STDOUT:-logs/paired_sweep/resume_driver_stdout.log}"
WRAPPER_SCRIPT="${WRAPPER_SCRIPT:-scripts/resume_paired_first_two_arms.sh}"

mkdir -p logs/paired_sweep
echo $$ > "$WATCHDOG_PIDFILE"

log() { printf '[watchdog %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$WATCHDOG_LOG"; }

screen_alive() {
    screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${SCREEN_NAME}$"
}

runner_alive() {
    pgrep -f 'cold-start-run --config configs/webarena_gmail_pair_.*_60[.]yaml' >/dev/null 2>&1
}

wrapper_alive() {
    screen_alive || pgrep -f '[r]esume_paired_first_two_arms[.]sh' >/dev/null 2>&1 || runner_alive
}

status_done() {
    python3 - <<'PY'
from pathlib import Path

path = Path("docs/paired_sweep_status.md")
if not path.exists():
    raise SystemExit(1)
statuses = {}
for line in path.read_text().splitlines():
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) == 5 and parts[1] in {"baseline", "planner"}:
        statuses[parts[1]] = parts[2]
raise SystemExit(0 if statuses.get("baseline") == "done" and statuses.get("planner") == "done" else 1)
PY
}

active_log() {
    python3 - <<'PY'
import pathlib

def candidates(prefix):
    out = []
    for p in pathlib.Path("logs/paired_sweep").glob(f"{prefix}_trial0_*.jsonl"):
        name = p.name
        if name.startswith("INVALID_"):
            continue
        if name.endswith("_FULL.jsonl") or name.endswith("_MERGED_SO_FAR.jsonl"):
            continue
        out.append(p)
    return out

planner = candidates("webarena_gmail_pair_planner_60")
baseline = candidates("webarena_gmail_pair_baseline_60")
if planner:
    print(max(planner, key=lambda p: p.stat().st_mtime))
elif baseline:
    print(max(baseline, key=lambda p: p.stat().st_mtime))
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
    log "stopping screen=${SCREEN_NAME}, cold-start runners, and port 8001"
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
    pgrep -f 'cold-start-run --config configs/webarena_gmail_pair_.*_60[.]yaml' 2>/dev/null | while read -r pid; do
        [[ "$pid" == "$$" ]] && continue
        kill -9 "$pid" 2>/dev/null || true
    done
    # shellcheck disable=SC2046
    kill -9 $(lsof -ti :8001 2>/dev/null) 2>/dev/null || true
    sleep 3

    log "relaunching ${WRAPPER_SCRIPT} in screen=${SCREEN_NAME}"
    screen -dmS "$SCREEN_NAME" bash -lc "cd '$REPO_DIR' && bash '$WRAPPER_SCRIPT' > '$WRAPPER_STDOUT' 2>&1"
    if wrapper_alive; then
        log "wrapper relaunch confirmed"
    else
        log "WARNING: wrapper relaunch was not visible after screen start"
    fi
}

log "starting; poll=${POLL_SECS}s stall=${STALL_SECS}s max_restarts=${MAX_RESTARTS}"

restarts=0
while true; do
    if status_done; then
        log "baseline and planner are both done; exiting"
        exit 0
    fi

    log_path="$(active_log)"
    if [[ -z "$log_path" ]]; then
        log "no active JSONL found yet"
        if ! wrapper_alive; then
            if (( restarts >= MAX_RESTARTS )); then
                log "GIVE UP: wrapper missing and max restarts reached"
                exit 2
            fi
            restart_wrapper
            restarts=$((restarts + 1))
        fi
        sleep "$POLL_SECS"
        continue
    fi

    now="$(date +%s)"
    mtime="$(stat -f %m "$log_path" 2>/dev/null || echo "$now")"
    gap=$((now - mtime))
    lt="$(last_t "$log_path")"
    rows="$(row_count "$log_path")"

    if wrapper_alive; then
        if (( gap > STALL_SECS )); then
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
