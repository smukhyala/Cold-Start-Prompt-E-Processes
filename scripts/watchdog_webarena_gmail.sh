#!/usr/bin/env bash
# Self-healing watchdog for the webarena_gmail full run.
#
# Polls the newest webarena_gmail JSONL log every $POLL_SECS. If the log
# hasn't grown in $STALL_SECS (meaning no task has completed in that long),
# force-kills the resume wrapper + any orphan on port 8001 and relaunches
# scripts/resume_webarena_gmail.sh. The resume script always picks up from
# the latest JSONL's last `t`, so back-to-back restarts compose cleanly.
#
# Exits cleanly when the run is complete (last `t` == TARGET_T) or after
# MAX_RESTARTS automatic restarts (so a pathological infinite loop doesn't
# burn the API budget unattended).
#
# Launched via:
#   nohup bash scripts/watchdog_webarena_gmail.sh \
#       > /tmp/cold_start_webarena_watchdog.log 2>&1 &
#   echo $! > /tmp/cold_start_webarena_watchdog.pid

set -uo pipefail
# Deliberately NOT set -e: we want to handle transient failures (e.g. missing
# JSONL during the first few seconds) without exiting the watchdog.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

POLL_SECS="${POLL_SECS:-120}"           # check every 2 min
STALL_SECS="${STALL_SECS:-720}"         # 12 min without a new task = stall
TARGET_T="${TARGET_T:-60}"              # full run target (matches webarena_gmail.yaml)
MAX_RESTARTS="${MAX_RESTARTS:-5}"       # cap auto-restarts
RESUME_PIDFILE="${RESUME_PIDFILE:-/tmp/cold_start_webarena_resume.pid}"
RESUME_LOG="${RESUME_LOG:-/tmp/cold_start_webarena_resume.log}"
RESUME_SCRIPT="${RESUME_SCRIPT:-scripts/resume_webarena_gmail.sh}"

log() { printf '[watchdog %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

newest_log() { ls -t logs/webarena_gmail_trial0_*.jsonl 2>/dev/null | head -1; }

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

relaunch_resume() {
    log "killing any leftover children before relaunch"
    if [[ -f "$RESUME_PIDFILE" ]]; then
        kill -9 "$(cat "$RESUME_PIDFILE")" 2>/dev/null || true
    fi
    # Sweep any cold-start-run python that might still be limping
    pkill -9 -f 'cold-start-run --config configs/webarena_gmail.yaml' 2>/dev/null || true
    # Free port 8001 (orphaned webarena server)
    # shellcheck disable=SC2046
    kill -9 $(lsof -ti :8001 2>/dev/null) 2>/dev/null || true
    sleep 3

    log "relaunching $RESUME_SCRIPT"
    nohup bash "$RESUME_SCRIPT" > "$RESUME_LOG" 2>&1 &
    echo $! > "$RESUME_PIDFILE"
    log "new resume PID: $(cat "$RESUME_PIDFILE")"
}

log "starting; poll=${POLL_SECS}s stall=${STALL_SECS}s target_t=$TARGET_T max_restarts=$MAX_RESTARTS"
log "monitoring pidfile=$RESUME_PIDFILE"

restarts=0
while true; do
    LOG="$(newest_log)"
    if [[ -z "$LOG" ]]; then
        log "no JSONL yet (resume script still spinning up); sleeping"
        sleep "$POLL_SECS"
        continue
    fi

    LT=$(last_t "$LOG")
    if (( LT >= TARGET_T )); then
        log "SUCCESS: $LOG reached t=$LT (target=$TARGET_T). Exiting watchdog."
        exit 0
    fi

    NOW=$(date +%s)
    MTIME=$(stat -f %m "$LOG")
    GAP=$((NOW - MTIME))

    if is_alive "$RESUME_PIDFILE"; then
        if (( GAP > STALL_SECS )); then
            log "STALL detected: $LOG last grew ${GAP}s ago (t=$LT/$TARGET_T). Restarting."
            if (( restarts >= MAX_RESTARTS )); then
                log "GIVE UP: hit MAX_RESTARTS=$MAX_RESTARTS. Manual intervention needed."
                exit 2
            fi
            relaunch_resume
            restarts=$((restarts + 1))
            log "restart #$restarts of $MAX_RESTARTS done; sleeping ${POLL_SECS}s before next check"
        else
            log "alive: t=$LT/$TARGET_T, last task ${GAP}s ago"
        fi
    else
        # Resume wrapper not alive but run incomplete. Two reasons:
        #   (a) wrapper crashed mid-run — restart it.
        #   (b) wrapper completed phases cleanly but our completion check
        #       hasn't fired yet — should be impossible because phase 1
        #       only ends after cold-start-run reaches t=TARGET_T.
        log "wrapper not alive but t=$LT < $TARGET_T. Treating as crash; restarting."
        if (( restarts >= MAX_RESTARTS )); then
            log "GIVE UP: hit MAX_RESTARTS=$MAX_RESTARTS. Manual intervention needed."
            exit 2
        fi
        relaunch_resume
        restarts=$((restarts + 1))
        log "restart #$restarts of $MAX_RESTARTS done"
    fi

    sleep "$POLL_SECS"
done
