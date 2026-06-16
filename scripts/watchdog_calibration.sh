#!/usr/bin/env bash
# Self-healing watchdog for the calibration run. Mirrors watchdog_webarena_gmail.sh
# but targets webarena_gmail_calibration_trial0_*.jsonl and target_t=25.
#
# Launched via:
#   nohup bash scripts/watchdog_calibration.sh \
#       > /tmp/cold_start_calibration_watchdog.log 2>&1 &
#   echo $! > /tmp/cold_start_calibration_watchdog.pid

set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

POLL_SECS="${POLL_SECS:-120}"
STALL_SECS="${STALL_SECS:-720}"
TARGET_T="${TARGET_T:-25}"
MAX_RESTARTS="${MAX_RESTARTS:-5}"
WRAPPER_PIDFILE="${WRAPPER_PIDFILE:-/tmp/cold_start_calibration.pid}"
WRAPPER_LOG="${WRAPPER_LOG:-/tmp/cold_start_calibration.log}"
WRAPPER_SCRIPT="${WRAPPER_SCRIPT:-scripts/run_calibration.sh}"
TODAY_PREFIX=$(date -u +%Y-%m-%d)

log() { printf '[watchdog %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
newest_log() { ls -t logs/webarena_gmail_calibration_trial0_${TODAY_PREFIX}*.jsonl 2>/dev/null | head -1; }

last_t() {
    local log="$1"
    [[ -s "$log" ]] || { echo 0; return; }
    tail -1 "$log" | python3 -c \
        'import sys,json
try:
    print(json.loads(sys.stdin.read())["t"])
except Exception:
    print(0)' 2>/dev/null
}

is_alive() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] || return 1
    ps -p "$(cat "$pidfile")" > /dev/null 2>&1
}

relaunch_wrapper() {
    log "killing leftovers before relaunch"
    if [[ -f "$WRAPPER_PIDFILE" ]]; then
        kill -9 "$(cat "$WRAPPER_PIDFILE")" 2>/dev/null || true
    fi
    pkill -9 -f "cold-start-run --config configs/webarena_gmail_calibration.yaml" 2>/dev/null || true
    # shellcheck disable=SC2046
    kill -9 $(lsof -ti :8001 2>/dev/null) 2>/dev/null || true
    sleep 3
    log "relaunching $WRAPPER_SCRIPT"
    nohup bash "$WRAPPER_SCRIPT" > "$WRAPPER_LOG" 2>&1 &
    echo $! > "$WRAPPER_PIDFILE"
    log "new wrapper PID: $(cat "$WRAPPER_PIDFILE")"
}

log "starting; poll=${POLL_SECS}s stall=${STALL_SECS}s target_t=$TARGET_T max_restarts=$MAX_RESTARTS"

restarts=0
while true; do
    LOG="$(newest_log)"
    if [[ -z "$LOG" ]]; then
        log "no calibration JSONL yet; sleeping"
        sleep "$POLL_SECS"
        continue
    fi

    LT=$(last_t "$LOG")
    if (( LT >= TARGET_T )); then
        log "SUCCESS: $LOG reached t=$LT (target=$TARGET_T). Exiting."
        exit 0
    fi

    NOW=$(date +%s)
    MTIME=$(stat -f %m "$LOG")
    GAP=$((NOW - MTIME))

    if is_alive "$WRAPPER_PIDFILE"; then
        if (( GAP > STALL_SECS )); then
            log "STALL: $LOG last grew ${GAP}s ago (t=$LT/$TARGET_T). Restarting."
            if (( restarts >= MAX_RESTARTS )); then
                log "GIVE UP: hit MAX_RESTARTS=$MAX_RESTARTS."
                exit 2
            fi
            relaunch_wrapper
            restarts=$((restarts + 1))
            log "restart #$restarts of $MAX_RESTARTS done"
        else
            log "alive: t=$LT/$TARGET_T, last task ${GAP}s ago"
        fi
    else
        log "wrapper not alive but t=$LT < $TARGET_T. Restarting."
        if (( restarts >= MAX_RESTARTS )); then
            log "GIVE UP: hit MAX_RESTARTS=$MAX_RESTARTS."
            exit 2
        fi
        relaunch_wrapper
        restarts=$((restarts + 1))
        log "restart #$restarts of $MAX_RESTARTS done"
    fi

    sleep "$POLL_SECS"
done
