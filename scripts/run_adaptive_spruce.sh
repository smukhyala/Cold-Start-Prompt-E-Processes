#!/usr/bin/env bash
# Start the adaptive SPRUCE Gmail sweep plus watchdog in detached screens.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p logs/adaptive_sweep

RUN_SCREEN="${RUN_SCREEN:-csp_adaptive_spruce}"
WATCHDOG_SCREEN="${WATCHDOG_SCREEN:-csp_adaptive_spruce_watchdog}"

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${RUN_SCREEN}$"; then
    echo "run screen already exists: ${RUN_SCREEN}" >&2
else
    screen -dmS "$RUN_SCREEN" bash -lc "cd '$REPO_DIR' && bash scripts/resume_adaptive_spruce.sh > logs/adaptive_sweep/adaptive_stdout.log 2>&1"
fi

if screen -ls 2>/dev/null | awk '{print $1}' | grep -Eq "^[0-9]+[.]${WATCHDOG_SCREEN}$"; then
    echo "watchdog screen already exists: ${WATCHDOG_SCREEN}" >&2
else
    screen -dmS "$WATCHDOG_SCREEN" bash -lc "cd '$REPO_DIR' && SCREEN_NAME='$RUN_SCREEN' bash scripts/watchdog_adaptive_spruce.sh >> logs/adaptive_sweep/watchdog_adaptive_spruce_screen.log 2>&1"
fi

cat <<EOF
Adaptive SPRUCE run launched.

Tail:
  tail -f logs/adaptive_sweep/adaptive_stdout.log
  tail -f logs/adaptive_sweep/watchdog_adaptive_spruce.log

Attach:
  screen -r ${RUN_SCREEN}
  screen -r ${WATCHDOG_SCREEN}
EOF
