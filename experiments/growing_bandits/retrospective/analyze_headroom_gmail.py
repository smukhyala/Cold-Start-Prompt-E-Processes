"""Pre-registration 15: the Gmail answer-key headroom (frozen analyzer).

    PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_headroom_gmail.py
    PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_headroom_gmail.py --synthetic

Arms: O = ``GM_oracle_full`` (60-task answer key), D = ``GM_anchor_baseline`` (the Gmail default). Every ``ok``
record is scored at the 30-step rescore (``study.rescore``: success and not timed_out and steps <= 30); an arm's
per-task score is the mean of its 2 replicates.

PRIMARY: H = mean over the 60 tasks of O_t - D_t, paired t, two-sided 90% CI (t_{0.95, 59}).
    large       iff the CI lower bound >= 0.10
    small       iff the CI upper bound <  0.10
    unresolved  otherwise

SECONDARY (reported whatever the outcome):
    1. task-cluster bootstrap 90% interval of H;
    2. capture: Pre-reg 13 Leg M's held-out lead of the search's winner G_14 over the default (0.000, on its 30
       tasks) set against H on the same 30 tasks;
    3. default drift: D now minus Leg M's default (same prompt) on those 30 tasks, mean with 90% CI;
    4. per-arm operations; rerun disagreement (replicate 0 vs 1) per arm.

Refuses (``AnalysisRefused``) unless every queued episode has a terminal ``ok`` record; never analyses records
carrying the synthetic marker except in ``--synthetic`` mode, which writes to a temp dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import analyze_winner_check as awc  # noqa: E402
import collect  # noqa: E402
import make_pools  # noqa: E402
import study as st  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402

PROFILE = "gmail"
DATA = ROOT / "data" / "headroom_gmail"
LOGS = ROOT / "logs" / "headroom_gmail"
RESULTS = ROOT / "results" / "headroom_gmail"
LEG_M_DATA = ROOT / "data" / "winner_check_gmail"
LEG_M_LOGS = ROOT / "logs" / "winner_check_gmail"
LEG_M_DEFAULT_ARM = "anchor_baseline"
LEG_M_LEAD = 0.0  # PREREG13_RESULTS.md, Leg M primary: G_14 - default = 0.000 on its 30 tasks (30-step rescore)
SEED = 20261013
N_BOOT = 10_000
ALPHA = 0.05
THRESHOLD = 0.10
SCORE_CAP = 30
SYNTHETIC_KEY = awc.SYNTHETIC_KEY
READING_LARGE = "large"
READING_SMALL = "small"
READING_UNRESOLVED = "unresolved"


class AnalysisRefused(RuntimeError):
    """The log is not a complete, scored, genuine record of the frozen queue."""


def reading(lower: float, upper: float) -> str:
    if lower >= THRESHOLD:
        return READING_LARGE
    if upper < THRESHOLD:
        return READING_SMALL
    return READING_UNRESOLVED


def cluster_bootstrap(diff, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    v = np.asarray(diff, dtype=float)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo = float(np.quantile(means, ALPHA))
    hi = float(np.quantile(means, 1.0 - ALPHA))
    return {"n_boot": n_boot, "seed": seed, "mean": float(means.mean()), "interval_two_sided_90": [lo, hi],
            "reading": reading(lo, hi), "share_of_resamples_ge_threshold": float((means >= THRESHOLD).mean())}


def load_scored(data_dir: Path, log_dir: Path, allow_synthetic: bool = False):
    """The complete, 30-step-rescored terminal frame of the frozen queue, or `AnalysisRefused`."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    queue = make_pools.read_queue(data_dir / collect.profile_spec(PROFILE)["queue"])
    if not Path(log_dir).is_dir():
        raise AnalysisRefused(f"{log_dir} does not exist")
    attempts = collect.load_all(Path(log_dir))
    if attempts.empty:
        raise AnalysisRefused(f"{log_dir} holds no records")
    marked = 0
    if SYNTHETIC_KEY in attempts.columns:
        marked = int(attempts[SYNTHETIC_KEY].fillna(False).astype(bool).sum())
    if marked > 0 and not allow_synthetic:
        raise AnalysisRefused(f"{marked} records carry the {SYNTHETIC_KEY} marker: fabricated data is never analysed")
    if marked == 0 and allow_synthetic:
        raise AnalysisRefused("synthetic mode requires records carrying the synthetic marker")
    try:
        terminal = emp.terminal_outcomes(attempts)
    except ValueError as err:
        raise AnalysisRefused(str(err)) from err
    prompts = collect.load_prompts(data_dir, PROFILE)
    try:
        collect.check_existing_records(Path(log_dir), prompts)
    except RuntimeError as err:
        raise AnalysisRefused(str(err)) from err
    queue_keys = set()
    for item in queue:
        queue_keys.add((item.arm_id, item.task_id, item.replicate))
    have = {}
    for row in terminal.itertuples():
        have[(str(row.arm_id), str(row.task_id), int(row.replicate))] = str(row.status)
    absent = 0
    not_ok = 0
    outside = 0
    for key in queue_keys:
        if key not in have:
            absent += 1
        elif have[key] != emp.STATUS_OK:
            not_ok += 1
    for key in have:
        if key not in queue_keys:
            outside += 1
    if absent or not_ok or outside:
        raise AnalysisRefused(f"incomplete: {absent} queued episodes without a terminal record, {not_ok} not ok, "
                              f"{outside} outside the queue; the primary result is not reported until every queued "
                              "episode is scored")
    scored = st.rescore(terminal, SCORE_CAP)
    changed = int((scored["success"].astype(int) != terminal["success"].astype(int)).sum())
    return manifest, queue, scored, changed


def build_table(manifest: dict, scored) -> list[dict]:
    o_arm = manifest["arms"]["O"]["arm_id"]
    d_arm = manifest["arms"]["D"]["arm_id"]
    cells = {}
    for row in scored.itertuples():
        cells[(str(row.arm_id), str(row.task_id), int(row.replicate))] = int(row.success)
    table = []
    pos = 0
    for task in manifest["task_ids"]:
        rec = {"queue_pos": pos, "task_id": task}
        for role, arm in (("O", o_arm), ("D", d_arm)):
            r0 = cells[(arm, task, 0)]
            r1 = cells[(arm, task, 1)]
            rec[f"{role}_r0"] = r0
            rec[f"{role}_r1"] = r1
            rec[role] = (r0 + r1) / 2.0
        rec["O_minus_D"] = rec["O"] - rec["D"]
        table.append(rec)
        pos += 1
    return table


def column(table: list[dict], key: str) -> np.ndarray:
    out = []
    for rec in table:
        out.append(float(rec[key]))
    return np.asarray(out, dtype=float)


def leg_m_default_scores(leg_m_data: Path = LEG_M_DATA, leg_m_logs: Path = LEG_M_LOGS) -> dict | None:
    """Leg M's default per-task score (mean of its 2 replicates, 30-step rescored), or None if unavailable."""
    log = Path(leg_m_logs) / "worker_0.jsonl"
    manifest_path = Path(leg_m_data) / "manifest.json"
    if not log.exists() or not manifest_path.exists():
        return None
    tasks = json.loads(manifest_path.read_text())["task_ids"]
    last = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("arm_id") != LEG_M_DEFAULT_ARM or rec.get("status") != emp.STATUS_OK:
            continue
        ok = int(bool(rec["success"]) and not bool(rec["timed_out"]) and int(rec["steps"]) <= SCORE_CAP)
        last[(str(rec["task_id"]), int(rec["replicate"]))] = ok
    out = {}
    for task in tasks:
        if (task, 0) not in last or (task, 1) not in last:
            return None
        out[task] = (last[(task, 0)] + last[(task, 1)]) / 2.0
    return out


def per_arm_operations(manifest: dict, scored, table: list[dict]) -> dict:
    out = {}
    for role in ("O", "D"):
        arm = manifest["arms"][role]["arm_id"]
        sub = scored[scored["arm_id"] == arm]
        disagree = 0
        for rec in table:
            if rec[f"{role}_r0"] != rec[f"{role}_r1"]:
                disagree += 1
        timed_out = int(sub["timed_out"].astype(bool).sum())
        out[role] = {"arm_id": arm, "episodes": int(len(sub)),
                     "mean_cost_usd": float(sub["cost_usd"].astype(float).mean()),
                     "total_cost_usd": float(sub["cost_usd"].astype(float).sum()),
                     "mean_steps": float(sub["steps"].astype(float).mean()),
                     "mean_wallclock_s": float(sub["wallclock_s"].astype(float).mean()),
                     "fraction_timed_out": timed_out / max(len(sub), 1),
                     "rerun_disagreement": disagree / len(table)}
    return out


def analyze(data_dir: Path = DATA, log_dir: Path = LOGS, out_dir: Path = RESULTS, allow_synthetic: bool = False,
            leg_m_data: Path = LEG_M_DATA, leg_m_logs: Path = LEG_M_LOGS) -> dict:
    manifest, queue, scored, changed = load_scored(data_dir, log_dir, allow_synthetic)
    table = build_table(manifest, scored)
    o = column(table, "O")
    d = column(table, "D")
    h = o - d
    t = awc.t_summary(h, ALPHA)
    lo, hi = t["interval_two_sided_90"]
    primary = {
        "study": manifest["study"], "synthetic": bool(allow_synthetic),
        "n_tasks": len(table), "n_episodes": len(queue),
        "arm_means": {"O": float(o.mean()), "D": float(d.mean())},
        "primary": {"quantity": "H = mean over tasks of O_t - D_t (arm score = mean of 2 replicates, 30-step rescored)",
                    "mean": t["mean"], "sd": t["sd"], "se": t["se"], "t_crit": t["t_crit"],
                    "interval_two_sided_90": [lo, hi], "threshold": THRESHOLD},
        "reading": reading(lo, hi),
        "rescored_changes_vs_recorded_success": changed,
    }
    if allow_synthetic:
        primary["WARNING"] = "SYNTHETIC DRY RUN: fabricated outcomes, not a result"

    leg_m = leg_m_default_scores(leg_m_data, leg_m_logs)
    leg_m_tasks = json.loads((Path(leg_m_data) / "manifest.json").read_text())["task_ids"]
    by_task = {}
    for rec in table:
        by_task[rec["task_id"]] = rec
    h30 = []
    for task in leg_m_tasks:
        h30.append(by_task[task]["O_minus_D"])
    t30 = awc.t_summary(np.asarray(h30), ALPHA)
    capture = {"quantity": "Leg M held-out lead of G_14 over the default (registered result) vs H on Leg M's 30 tasks",
               "leg_m_lead": LEG_M_LEAD, "H_on_leg_m_tasks": t30["mean"],
               "H_on_leg_m_tasks_interval_two_sided_90": t30["interval_two_sided_90"],
               "captured_fraction": (LEG_M_LEAD / t30["mean"]) if t30["mean"] > 0 else None}
    if leg_m is None:
        drift = {"available": False}
    else:
        diffs = []
        for task in leg_m_tasks:
            diffs.append(by_task[task]["D"] - leg_m[task])
        td = awc.t_summary(np.asarray(diffs), ALPHA)
        old = []
        new = []
        for task in leg_m_tasks:
            old.append(leg_m[task])
            new.append(by_task[task]["D"])
        drift = {"available": True, "quantity": "D now - Leg M default (same prompt) on Leg M's 30 tasks",
                 "mean": td["mean"], "se": td["se"], "interval_two_sided_90": td["interval_two_sided_90"],
                 "D_leg_m_mean": float(np.mean(old)), "D_now_mean": float(np.mean(new))}
    secondary = {"synthetic": bool(allow_synthetic),
                 "task_cluster_bootstrap_H": cluster_bootstrap(h),
                 "capture": capture,
                 "default_drift_vs_leg_m": drift,
                 "operations_by_arm": per_arm_operations(manifest, scored, table)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primary.json").write_text(json.dumps(primary, indent=2) + "\n")
    (out_dir / "secondary.json").write_text(json.dumps(secondary, indent=2) + "\n")
    fields = ["queue_pos", "task_id", "O_r0", "O_r1", "D_r0", "D_r1", "O", "D", "O_minus_D"]
    with open(out_dir / "task_table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in table:
            writer.writerow(rec)
    return {"primary": primary, "secondary": secondary}


def make_synthetic_log(data_dir: Path, log_dir: Path, seed: int = 0, p_o: float = 0.85, p_d: float = 0.60) -> None:
    """A complete FAKE worker log for the frozen queue, every record marked ``SYNTHETIC_DO_NOT_USE``."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    queue = make_pools.read_queue(Path(data_dir) / collect.profile_spec(PROFILE)["queue"])
    prompts = collect.load_prompts(Path(data_dir), PROFILE)
    rng = np.random.default_rng(seed)
    base = {manifest["arms"]["O"]["arm_id"]: p_o, manifest["arms"]["D"]["arm_id"]: p_d}
    difficulty = {}
    for task in manifest["task_ids"]:
        difficulty[task] = float(rng.normal(0.0, 0.12))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    lines = []
    for item in queue:
        p = min(max(base[item.arm_id] + difficulty[item.task_id], 0.02), 0.98)
        success = int(rng.random() < p)
        lines.append(json.dumps({
            "schema": emp.SCHEMA, "pool": item.pool, "arm_id": item.arm_id, "task_id": item.task_id,
            "replicate": item.replicate, "attempt": 1, "status": "ok", "success": success,
            "cost_usd": float(rng.uniform(0.05, 0.12)), "steps": int(rng.integers(3, 25)),
            "wallclock_s": float(rng.uniform(20, 170)), "timed_out": False, "is_done": True, "ended_by": "agent",
            "queue_index": item.index, "pilot": False, "worker": 0, "profile": PROFILE, "timeout_s": 180,
            "prompt_sha256": prompts[item.arm_id].sha256, "task_dir": "SYNTHETIC", SYNTHETIC_KEY: True,
        }))
    (Path(log_dir) / "worker_0.jsonl").write_text("\n".join(lines) + "\n")


def synthetic_dry_run(data_dir: Path = DATA, out_dir: Path | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="headroom_gmail_synthetic_"))
    make_synthetic_log(data_dir, tmp / "logs")
    target = out_dir if out_dir is not None else tmp / "results"
    result = analyze(data_dir, tmp / "logs", target, allow_synthetic=True)
    result["primary"]["_synthetic_workdir"] = str(tmp)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="fabricated outcomes in a temp dir (pipeline check)")
    args = ap.parse_args(argv)
    if args.synthetic:
        result = synthetic_dry_run()
    else:
        result = analyze()
    print(json.dumps(result["primary"], indent=2))
    print(json.dumps(result["secondary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
