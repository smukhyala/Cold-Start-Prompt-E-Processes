#!/usr/bin/env bash
# Run the m_0 calibration end-to-end:
#   1. cold-start-run configs/webarena_gmail_calibration.yaml (25 baseline-only tasks).
#   2. cold-start-export-csv + cold-start-report on the resulting log.
#   3. Print the empirical baseline mean μ̂_baseline so it can be patched into
#      webarena_gmail.yaml (replacing the placeholder m_0 = 0.5).
#
# Designed for resume safety: if a prior calibration JSONL exists with t < 25,
# the run continues from the next task instead of starting over.
#
# Launched via:
#   nohup bash scripts/run_calibration.sh \
#       > /tmp/cold_start_calibration.log 2>&1 &
#   echo $! > /tmp/cold_start_calibration.pid

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate
set -a
# shellcheck disable=SC1091
source .env
set +a

CONFIG="configs/webarena_gmail_calibration.yaml"
TARGET_T=25

banner() {
    printf '\n========================================\n'
    printf '== %s\n' "$1"
    printf '== %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '========================================\n\n'
}

newest_calibration_log() {
    ls -t logs/webarena_gmail_calibration_trial0_*.jsonl 2>/dev/null | head -1
}

last_t_in() {
    local log="$1"
    [[ -s "$log" ]] || { echo 0; return; }
    tail -1 "$log" | python3 -c 'import sys,json;print(json.loads(sys.stdin.read()).get("t",0))' 2>/dev/null || echo 0
}

# --- Phase 1: run (fresh or resume from latest partial in this same session) -

banner "Phase 1/3: calibration ($CONFIG, target_t=$TARGET_T)"

# We deliberately do NOT resume from the 3-day-old logs/webarena_gmail_calibration_*Jun-10*.jsonl
# (infra has changed). But within today's session, if a partial exists from a
# prior crash, resume from it.
TODAY_PREFIX=$(date -u +%Y-%m-%d)
RESUME_FROM=""
NEXT_T=1
for f in $(ls -t logs/webarena_gmail_calibration_trial0_${TODAY_PREFIX}*.jsonl 2>/dev/null); do
    lt=$(last_t_in "$f")
    if (( lt > 0 && lt < TARGET_T )); then
        RESUME_FROM="$f"
        NEXT_T=$((lt + 1))
        break
    fi
done

if [[ -n "$RESUME_FROM" ]]; then
    echo "Resuming from $RESUME_FROM (start_at=$NEXT_T)"
    cold-start-run --config "$CONFIG" --resume-from "$RESUME_FROM" --start-at "$NEXT_T"
else
    echo "Starting fresh calibration."
    cold-start-run --config "$CONFIG"
fi

# --- Phase 2: merge all today's segments into one calibration log -----------

banner "Phase 2/3: merge segments + CSV + report"

# Merge all today's segments into one calibration log. Done entirely in
# python (single subprocess, no bash arrays) so this works on the macOS
# system bash 3.2 which lacks `mapfile`.
MERGED="$(python3 - <<PY
import json, pathlib, sys

today = "${TODAY_PREFIX}"
candidates = sorted(pathlib.Path("logs").glob(
    f"webarena_gmail_calibration_trial0_{today}*.jsonl"))
if not candidates:
    sys.exit("no calibration segments found for today")

# Group by base run_id (strip _resume_from_N) and order by first t.
segs = []
for p in candidates:
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if not lines:
        continue
    first_t = json.loads(lines[0])["t"]
    run_id = json.loads(lines[0])["run_id"].split("_resume_from_")[0]
    segs.append((run_id, first_t, p, len(lines)))

# Use the newest base run_id (in case there are multiple unrelated runs today).
latest_base = sorted({s[0] for s in segs})[-1]
chain = sorted([s for s in segs if s[0] == latest_base], key=lambda s: s[1])

merged_path = pathlib.Path(f"logs/{latest_base}_FULL.jsonl")
with merged_path.open("w") as out:
    for _, _, p, _ in chain:
        out.write(p.read_text())
        if not p.read_text().endswith("\n"):
            out.write("\n")
total = sum(s[3] for s in chain)
sys.stderr.write(f"merged {total} records from {len(chain)} segment(s)\n")
print(merged_path)
PY
)"
echo "merged log: $MERGED"
BASE_RUN_ID="$(basename "$MERGED" _FULL.jsonl)"

REPORT_DIR="reports/${BASE_RUN_ID}_FULL"
mkdir -p "$REPORT_DIR"
cold-start-export-csv --log "$MERGED" --out-dir "$REPORT_DIR"
cold-start-report --log "$MERGED" --out-dir "$REPORT_DIR"

# --- Phase 3: compute empirical m_0 -----------------------------------------

banner "Phase 3/3: empirical m_0"

python3 <<PY
import json, math
recs = [json.loads(l) for l in open("$MERGED").read().splitlines() if l.strip()]
wins = sum(int(r["success"]) for r in recs)
n = len(recs)
mu = wins / n
se = math.sqrt(mu * (1 - mu) / n)
print(f"baseline calibration: {wins}/{n} = {mu:.4f}")
print(f"SE = {se:.4f}, 95% CI = [{mu - 1.96*se:.3f}, {mu + 1.96*se:.3f}]")
print()
print("To patch webarena_gmail.yaml:")
print(f"  sed -i.bak 's/m0: 0.5/m0: {mu:.3f}/' configs/webarena_gmail.yaml")
PY

banner "Done."
echo "Calibration log: $MERGED"
echo "Report dir:      $REPORT_DIR"
