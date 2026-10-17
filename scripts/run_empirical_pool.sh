#!/usr/bin/env bash
# Launch the empirical-pool collector under its watchdog, detached and awake.
#   scripts/run_empirical_pool.sh --pilot --budget 40   # the 660-episode pilot (gate G2), capped at $40
#   scripts/run_empirical_pool.sh                       # everything left in the queue (resumes the pilot)
#   scripts/run_empirical_pool.sh --profile gitlab --pilot --budget 100   # a heterogeneity profile
#   scripts/run_empirical_pool.sh --profile gitlab --budget 400 --through-index N   # a registered stage
#   scripts/run_empirical_pool.sh --profile gitlab --budget 260 --through-index 3059 --pools GLK   # one pool of it
# Heterogeneity staging (spec 4.5 as amended): gitlab --pilot; gitlab --through-index N where N is
# `make_het_pools.py --print-stages`' block_a_with_replicates; gmail; gitlab (the rest); bridge -- one
# heterogeneity profile at a time (collect.py's shared logs/heterogeneity/het.lock). A --through-index
# run ends with STATUS `through`, which the watchdog treats as finished (never relaunched); so does a
# --pools A,B run (only those pools' queue items, composable with --through-index).
# Every argument (--profile NAME, --pilot, --workers N, --budget USD, --through-index N, --pools A,B) goes to the
# watchdog unchanged, which forwards it to collect.py. The log dir (logs/empirical_pool for prereg9,
# logs/heterogeneity/<profile> otherwise) is the watchdog's own answer for these arguments, so
# watchdog.out lands beside STATUS; bad arguments fail here, before anything is detached.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG_DIR="$(.venv/bin/python scripts/watchdog_empirical_pool.py --print-log-dir "$@")"
mkdir -p "$LOG_DIR"
nohup caffeinate -dimsu .venv/bin/python scripts/watchdog_empirical_pool.py "$@" \
  >> "$LOG_DIR/watchdog.out" 2>&1 &
echo "watchdog pid $! -- tail -f $LOG_DIR/watchdog.out $LOG_DIR/collect.out"
