#!/usr/bin/env bash
# Start a queued screen that waits for the current 120 pair, then runs two more pairs.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/comparison_sweeps_120_replicates logs/comparison_sweeps_120

QUEUE_SCREEN="${QUEUE_SCREEN:-csp_120_replicate_queue}"
WATCHDOG_SCREEN="${WATCHDOG_SCREEN:-csp_120_replicate_queue_watchdog}"
CAFFEINE_SCREEN="${CAFFEINE_SCREEN:-csp_comparison_120_caffeinate}"

if ! screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${CAFFEINE_SCREEN}$"; then
    screen -dmS "$CAFFEINE_SCREEN" bash -lc "cd '$REPO_DIR' && while true; do date -u '+[caffeinate-120 %Y-%m-%dT%H:%M:%SZ] renewing 1h assertion' >> logs/comparison_sweeps_120/caffeinate.log; caffeinate -i -t 3600 >> logs/comparison_sweeps_120/caffeinate.log 2>&1; sleep 5; done"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${QUEUE_SCREEN}$"; then
    echo "replicate queue screen already exists: ${QUEUE_SCREEN}" >&2
else
    screen -dmS "$QUEUE_SCREEN" bash -lc "cd '$REPO_DIR' && bash scripts/resume_uniform_then_adaptive_120_replicates.sh > logs/comparison_sweeps_120_replicates/stdout.log 2>&1"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${WATCHDOG_SCREEN}$"; then
    echo "replicate queue watchdog screen already exists: ${WATCHDOG_SCREEN}" >&2
else
    screen -dmS "$WATCHDOG_SCREEN" bash -lc "cd '$REPO_DIR' && SCREEN_NAME='$QUEUE_SCREEN' bash scripts/watchdog_120_followup_replicates.sh >> logs/comparison_sweeps_120_replicates/watchdog_screen.log 2>&1"
fi

cat <<EOF
120-task follow-up replicate queue launched.

It waits for the current 120-task uniform/adaptive pair to finish, then runs:
  1. uniform_multiarm_120_rep2
  2. adaptive_spruce_120_rep2
  3. uniform_multiarm_120_rep3
  4. adaptive_spruce_120_rep3

Tail:
  tail -f logs/comparison_sweeps_120_replicates/stdout.log
  tail -f logs/comparison_sweeps_120_replicates/resume_trace.log
  tail -f logs/comparison_sweeps_120_replicates/watchdog.log

Attach:
  screen -r ${QUEUE_SCREEN}
  screen -r ${WATCHDOG_SCREEN}
  screen -r ${CAFFEINE_SCREEN}
EOF
