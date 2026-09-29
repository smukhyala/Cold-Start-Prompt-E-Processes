#!/usr/bin/env bash
# Launch the empirical-pool collector under its watchdog, detached and awake.
#   scripts/run_empirical_pool.sh --pilot --budget 40   # the 660-episode pilot (gate G2), capped at $40
#   scripts/run_empirical_pool.sh                       # everything left in the queue (resumes the pilot)
#   scripts/run_empirical_pool.sh --profile gitlab --pilot --budget 100   # a heterogeneity profile
# Every argument (--profile NAME, --pilot, --workers N, --budget USD) goes to the watchdog unchanged,
# which forwards it to collect.py. The log dir (logs/empirical_pool for prereg9,
# logs/heterogeneity/<profile> otherwise) is the watchdog's own answer for these arguments, so
# watchdog.out lands beside STATUS; bad arguments fail here, before anything is detached.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="$(.venv/bin/python scripts/watchdog_empirical_pool.py --print-log-dir "$@")"
mkdir -p "$LOG_DIR"
nohup caffeinate -dimsu .venv/bin/python scripts/watchdog_empirical_pool.py "$@" \
  >> "$LOG_DIR/watchdog.out" 2>&1 &
echo "watchdog pid $! -- tail -f $LOG_DIR/watchdog.out $LOG_DIR/collect.out"
