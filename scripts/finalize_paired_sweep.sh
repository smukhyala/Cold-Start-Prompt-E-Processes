#!/usr/bin/env bash
# Rebuild the paired sweep summary and optionally commit compact final artifacts.

set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

STATUS="docs/paired_sweep_status.md"

python3 - <<'PY'
import csv
import json
from pathlib import Path

arms = [
    ("1", "baseline"),
    ("2", "planner"),
    ("3", "cautious"),
    ("4", "explorer"),
    ("5", "balanced"),
    ("6", "overthinker"),
    ("7", "rapid"),
    ("8", "verifier"),
    ("9", "exploratory"),
    ("10", "algorithmic"),
    ("11", "junior_reactive"),
    ("12", "domain_expert"),
]

prompt_notes = {
    "baseline": "mid-level, direct action, no upfront planning, no explicit verification.",
    "planner": "stronger planning instruction before action.",
    "cautious": "stronger verification/checking behavior.",
    "explorer": "more willingness to explore alternatives.",
    "balanced": "moderate planning plus moderate verification.",
    "overthinker": "maximum planning/verification/expertise/structuredness.",
    "rapid": "fast, low-overhead, proactive, outcome-focused.",
    "verifier": "light planning with every-step verification pressure.",
    "exploratory": "proactive exploration with moderate planning.",
    "algorithmic": "domain-expert, plan-then-act, algorithmic format.",
    "junior_reactive": "low-expertise, cautious-reactive anchor.",
    "domain_expert": "expert-driven, proactive, constraint-aware.",
}

rows = []
locations = {}
for order, arm in arms:
    per_arm = Path(f"reports/paired_sweep/{arm}/per_arm.csv")
    run_csv = Path(f"reports/paired_sweep/{arm}/run.csv")
    full_logs = sorted(Path("logs/paired_sweep").glob(f"webarena_gmail_pair_{arm}_60_trial0_*_FULL.jsonl"))
    latest_segments = [
        p for p in Path("logs/paired_sweep").glob(f"webarena_gmail_pair_{arm}_60_trial0_*.jsonl")
        if not p.name.startswith("INVALID_")
        and not p.name.endswith("_FULL.jsonl")
        and not p.name.endswith("_MERGED_SO_FAR.jsonl")
        and p.stat().st_size > 0
    ]
    log_path = str(full_logs[-1]) if full_logs else (str(max(latest_segments, key=lambda p: p.stat().st_mtime)) if latest_segments else "")
    report_dir = f"reports/paired_sweep/{arm}" if per_arm.exists() else ""
    locations[arm] = (log_path, report_dir)
    if not per_arm.exists() or not run_csv.exists():
        continue
    with per_arm.open() as f:
        summary = next(csv.DictReader(f))
    with run_csv.open() as f:
        tasks = list(csv.DictReader(f))
    def count(prefix):
        subset = [r for r in tasks if r["task_id"].startswith(prefix)]
        return sum(int(r["success"]) for r in subset), len(subset)
    easy = count("task_e")
    medium = count("task_m")
    hard = count("task_h")
    rows.append({
        "order": order,
        "arm": arm,
        "successes": int(float(summary["successes"])),
        "pulls": int(float(summary["pulls"])),
        "rate": 100.0 * float(summary["mu_hat"]),
        "easy": easy,
        "medium": medium,
        "hard": hard,
        "cost": float(summary["total_cost_usd"]),
        "wall_min": float(summary["total_wallclock_s"]) / 60.0,
        "avg_min": float(summary["total_wallclock_s"]) / 60.0 / max(float(summary["pulls"]), 1.0),
        "log_e": float(summary["log_e"]),
        "cs_low": float(summary["cs_low"]),
        "cs_high": float(summary["cs_high"]),
        "rejected": summary["rejected"],
    })

ranked = sorted(rows, key=lambda r: (-r["successes"], r["cost"], r["arm"]))
completed = {r["arm"] for r in rows if r["pulls"] == 60}
all_done = len(completed) == len(arms)

lines = [
    "# Paired Sweep Status",
    "",
    "This file tracks the 12-arm paired WebArena Gmail sweep. Each arm should",
    "eventually receive the same 60-task cycle.",
    "",
    "## Read This First",
    "",
    "The uniform sweep runs each prompt arm on the same 60 Gmail tasks: 20 easy,",
    "20 medium, and 20 hard. A completed arm is comparable to every other completed",
    "arm because it saw the same task set.",
    "",
]

if rows:
    best = ranked[0]
    lines.extend([
        f"Current headline: {len(completed)}/12 arms are complete. The current best",
        f"completed arm is `{best['arm']}` with {best['successes']}/60 successes",
        f"({best['rate']:.1f}%).",
        "",
        "## Completed Results",
        "",
        "| rank | arm | successes | rate | easy | medium | hard | cost | wall time | avg min/task | log-e | CS low | CS high |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, r in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {r['arm']} | {r['successes']}/{r['pulls']} | {r['rate']:.1f}% | "
            f"{r['easy'][0]}/{r['easy'][1]} | {r['medium'][0]}/{r['medium'][1]} | "
            f"{r['hard'][0]}/{r['hard'][1]} | ${r['cost']:.4f} | {r['wall_min']:.1f} min | "
            f"{r['avg_min']:.1f} | {r['log_e']:.3f} | {r['cs_low']:.3f} | {r['cs_high']:.3f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- Rankings are empirical success-rate rankings over the paired 60-task set.",
        "- `cost` and `wall time` are measured from the run logs and should be used",
        "  when comparing efficiency across arms.",
        "- The e-process columns are per-arm evidence against m0 = 0.5; final",
        "  rejection status should be checked in each generated arm report.",
        "",
    ])
else:
    lines.extend(["Current headline: no completed arm reports were found.", ""])

incomplete = [(order, arm) for order, arm in arms if arm not in completed]
if incomplete:
    lines.extend([
        "## Current Live Run",
        "",
        "| arm | status | durable progress | notes |",
        "|---|---|---:|---|",
    ])
    for _, arm in incomplete:
        log_path, _ = locations.get(arm, ("", ""))
        progress = "0/60"
        if log_path:
            path = Path(log_path)
            if path.exists() and path.stat().st_size > 0:
                try:
                    last = json.loads(path.read_text().splitlines()[-1])
                    progress = f"{last['t']}/60"
                except Exception:
                    progress = f"{sum(1 for _ in path.open())}/60"
        lines.append(f"| {arm} | pending/running | {progress} | Live count may already be higher. |")
    lines.extend([
        "",
        "Live commands:",
        "",
        "```bash",
        "tail -f logs/paired_sweep/remaining_stdout.log",
        "tail -f logs/paired_sweep/watchdog_paired_remaining.log",
        "screen -r csp_paired_remaining",
        "```",
        "",
    ])
else:
    lines.extend([
        "## Current Live Run",
        "",
        "All 12 arms are complete. No live runner should remain.",
        "",
    ])

lines.extend([
    "## Current Segment",
    "",
    "Use this table for file locations and completion state. The JSONL is the raw",
    "source of truth; the report directory contains `run.csv`, `per_arm.csv`, and",
    "the generated report for that arm.",
    "",
    "| order | arm | status | raw log | report |",
    "|---:|---|---|---|---|",
])
for order, arm in arms:
    log_path, report_dir = locations.get(arm, ("", ""))
    status = "done" if arm in completed else "pending/running"
    lines.append(f"| {order} | {arm} | {status} | {log_path} | {report_dir} |")

lines.extend([
    "",
    "## Prompt Arms",
    "",
    "These labels are shorthand for different prompt vectors. `baseline` is not",
    "no-prompt; it is the simplest direct-action prompt in the catalog.",
    "",
])
for _, arm in arms:
    lines.append(f"- `{arm}`: {prompt_notes[arm]}")

lines.extend([
    "",
    "## Logging Contract",
    "",
    "For each completed task, the JSONL log must include:",
    "",
    "- `arm_id`",
    "- `task_id`",
    "- `task_meta.difficulty`",
    "- `success`",
    "- `reward`",
    "- `steps`",
    "- `wallclock_s`",
    "- `tokens.input`",
    "- `tokens.output`",
    "- `tokens.cache_read`",
    "- `tokens.cache_write`",
    "- `tokens.cost_usd`",
    "- `policy`",
    "- `per_arm_state`",
    "- `global_e`",
    "",
    "The runner flushes each JSONL record immediately after writing it, so completed",
    "task data remains available if a later task fails.",
    "",
    "## Invalid Attempts",
    "",
    "- `logs/paired_sweep/INVALID_connection_webarena_gmail_pair_baseline_60_trial0_2026-06-16T18-12-11Z.jsonl`",
    "  contains 2 baseline records from a workplace network where Anthropic/Claude",
    "  connections were blocked. These records should not be used as research data.",
    "",
])

Path("docs/paired_sweep_status.md").write_text("\n".join(lines))
PY

if [[ "${COMMIT_FINAL:-0}" == "1" ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
        git add docs/paired_sweep_status.md docs/research_process.md configs scripts
        for arm in baseline planner cautious explorer balanced overthinker rapid verifier exploratory algorithmic junior_reactive domain_expert; do
            git add -f "reports/paired_sweep/${arm}/run.csv" \
                "reports/paired_sweep/${arm}/per_arm.csv" \
                "reports/paired_sweep/${arm}/summary.md" 2>/dev/null || true
            git add -f logs/paired_sweep/webarena_gmail_pair_${arm}_60_trial0_*_FULL.jsonl 2>/dev/null || true
        done
        if ! git diff --cached --quiet; then
            git commit -m "Finalize paired Gmail prompt sweep"
        fi
    fi
fi
