#!/usr/bin/env bash
# Honest liveness check: shows the newest JSONL log's record count, its
# mtime (= time of the last completed task), and how long ago that was.
# If the gap > 5 min, the run is stalled (see runbook § "Stopping cleanly").
# Reports DONE when the newest log's last `t` has reached TARGET_T.

# Deliberately NOT set -e: ps -p against a missing wrapper pidfile is
# expected when the run has finished and should not fail the whole check.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET_T="${TARGET_T:-60}"

LOG="$(ls -t logs/webarena_gmail_trial0_*.jsonl 2>/dev/null | head -1)"
if [[ -z "$LOG" ]]; then
    echo "no webarena_gmail JSONL logs in logs/"
    exit 1
fi

NOW_EPOCH=$(date +%s)
MTIME_EPOCH=$(stat -f %m "$LOG")
GAP=$((NOW_EPOCH - MTIME_EPOCH))
GAP_MIN=$((GAP / 60))

RECS=$(wc -l < "$LOG" | tr -d ' ')
LAST_LINE=$(tail -1 "$LOG")
LAST_T=$(echo "$LAST_LINE" | python3 -c 'import sys,json;print(json.loads(sys.stdin.read())["t"])' 2>/dev/null || echo "?")
LAST_TASK=$(echo "$LAST_LINE" | python3 -c '
import sys, json
r = json.loads(sys.stdin.read())
print("task=" + str(r["task_id"]) + " arm=" + str(r["arm_id"]) + " success=" + str(r["success"]))
' 2>/dev/null || echo "?")

echo "log:          $LOG"
echo "records:      $RECS (last t=$LAST_T)"
echo "last task:    $LAST_TASK"
echo "log mtime:    $(date -r $MTIME_EPOCH '+%Y-%m-%d %H:%M:%S %Z')"
echo "gap to now:   ${GAP_MIN} min ago"

if   [[ "$LAST_T" =~ ^[0-9]+$ ]] && (( LAST_T >= TARGET_T )); then
    echo "STATUS:       🎉 DONE (reached t=$LAST_T / target=$TARGET_T)"
elif (( GAP > 600 )); then echo "STATUS:       ❌ STALLED (no new task in >10 min, not at target)"
elif (( GAP > 300 )); then echo "STATUS:       ⚠️  SLOW (no new task in >5 min — may be a long task)"
else                       echo "STATUS:       ✅ active"
fi

echo ""
echo "running wrappers:"
found=0
for pidf in /tmp/cold_start_webarena.pid /tmp/cold_start_webarena_resume.pid /tmp/cold_start_webarena_watchdog.pid; do
    if [[ -f "$pidf" ]]; then
        pid=$(cat "$pidf")
        if ps -p "$pid" -o pid=,etime=,command= 2>/dev/null; then
            echo "  pid file: $pidf"
            found=1
        fi
    fi
done
if (( found == 0 )); then
    echo "  (none running)"
fi
exit 0
