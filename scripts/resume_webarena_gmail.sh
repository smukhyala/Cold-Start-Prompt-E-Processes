#!/usr/bin/env bash
# Resume the full webarena_gmail run from the next task after the latest
# JSONL log's last record, then run CSV export and markdown report when
# the resumed run finishes.
#
# Launched via:
#   nohup bash scripts/resume_webarena_gmail.sh \
#       > /tmp/cold_start_webarena_resume.log 2>&1 &
#   echo $! > /tmp/cold_start_webarena_resume.pid

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate
set -a
# shellcheck disable=SC1091
source .env
set +a

CONFIG="configs/webarena_gmail.yaml"

banner() {
    printf '\n========================================\n'
    printf '== %s\n' "$1"
    printf '== %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '========================================\n\n'
}

LAST_LOG="$(ls -t logs/webarena_gmail_trial0_*.jsonl 2>/dev/null | head -1)"
if [[ -z "$LAST_LOG" ]]; then
    echo "ERROR: no logs/webarena_gmail_trial0_*.jsonl to resume from." >&2
    exit 1
fi
LAST_T="$(tail -1 "$LAST_LOG" | python3 -c 'import sys,json;print(json.loads(sys.stdin.read())["t"])')"
NEXT_T=$((LAST_T + 1))

banner "Resume — base log: $LAST_LOG  last_t=$LAST_T  start_at=$NEXT_T"
cold-start-run --config "$CONFIG" --resume-from "$LAST_LOG" --start-at "$NEXT_T"

# Resume writes a new JSONL with _resume_from_N suffix in the run_id. To get
# the full 60-task picture we merge ALL segments in this run's chain (the
# original + every resume hop). Done entirely in python so this works on the
# macOS system bash 3.2 (no `mapfile` builtin).
BASE_RUN_ID="$(tail -1 "$LAST_LOG" | python3 -c \
    'import sys,json
r = json.loads(sys.stdin.read())
print(r["run_id"].split("_resume_from_")[0])')"
echo "base run_id: $BASE_RUN_ID"

MERGED="$(python3 - <<PY
import json, pathlib, sys

base = "${BASE_RUN_ID}"
segs = []
for p in pathlib.Path("logs").glob(f"{base}*.jsonl"):
    if p.name.endswith("_FULL.jsonl"):
        continue
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if not lines:
        continue
    first_t = json.loads(lines[0])["t"]
    segs.append((first_t, p, len(lines)))

if not segs:
    sys.exit(f"ERROR: no segments found for base run_id {base}")

segs.sort(key=lambda s: s[0])
merged_path = pathlib.Path(f"logs/{base}_FULL.jsonl")
with merged_path.open("w") as out:
    for _, p, _ in segs:
        text = p.read_text()
        out.write(text if text.endswith("\n") else text + "\n")
total = sum(s[2] for s in segs)
sys.stderr.write(f"merged {total} records from {len(segs)} segment(s)\n")
print(merged_path)
PY
)"
echo "merged log: $MERGED"

REPORT_DIR="reports/${BASE_RUN_ID}_FULL"
mkdir -p "$REPORT_DIR"

banner "CSV export -> $REPORT_DIR"
cold-start-export-csv --log "$MERGED" --out-dir "$REPORT_DIR"

banner "Markdown report + plots -> $REPORT_DIR"
cold-start-report --log "$MERGED" --out-dir "$REPORT_DIR"

banner "Done. Artifacts:"
ls -la "$REPORT_DIR"
