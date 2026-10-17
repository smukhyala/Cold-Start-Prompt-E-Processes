"""Pre-registration 13, Gmail arm: the REGISTERED analysis of the winner's-curse test (G_14 vs the default).

    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_winner_check_gmail.py
    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_winner_check_gmail.py --synthetic

Offline and read-only (no model call, no server). Reads ``data/winner_check_gmail`` (frozen manifest + queue),
the collector's ``logs/winner_check_gmail/worker_*.jsonl`` (the default arm, 60 episodes) and the first-run
pilot's ``logs/first_run_pilot_gmail/worker_*.jsonl`` (G_14, 60 episodes). REFUSES unless all 60 new episodes
are terminal with status ``ok`` and all 60 pilot G_14 episodes exist and are ``ok`` (no duplicates, nothing
outside the queue, prompt shas equal to the frozen pools'), and refuses logs carrying the synthetic marker.

Registered quantities (G = G_14, D = default arm; arm score per task = mean of its 2 replicates):

* PRIMARY: both arms scored at the 30-step rescore (study.rescore: success and not timed_out and steps <= 30);
  d_t = G_t - D_t. Mean(d) with the one-sided 95% UPPER (and LOWER) bound, paired t over the 30 tasks. Reading:
  LB >= 0.10 -> "the +10 in-sample lead replicated at full size" (reproduced); UB < 0.10 -> "the lead shrank
  below its in-sample size"; otherwise inconclusive. Also the two-sided 90% interval and a 10,000-sample task
  bootstrap (seed 20261012).
* SECONDARY (confounded): the same on as-collected scores (uncapped pilot G_14 vs capped default); the September
  in-sample advantage vs the held-out estimate (shrinkage); per-arm rerun disagreement; cost / steps per arm.

Writes ``results/winner_check_gmail/{primary.json, task_table.csv}`` (``*_synthetic.*`` with --synthetic).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import collect  # noqa: E402
import make_pools  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from experiments.growing_bandits.empirical import study as st  # noqa: E402

DATA = ROOT / "data" / "winner_check_gmail"
LOGS = ROOT / "logs" / "winner_check_gmail"
PILOT_LOGS = ROOT / "logs" / "first_run_pilot_gmail"
RESULTS = ROOT / "results" / "winner_check_gmail"
SEED = 20261012
N_BOOT = 10_000
ALPHA = 0.05
MARGIN = 0.10
SCORE_CAP = 30
WINNER_ARM = "G_14"
SYNTHETIC_KEY = "SYNTHETIC_DO_NOT_USE"


class AnalysisRefused(RuntimeError):
    """The logs are not a complete, scored, genuine record of the frozen design."""


def load_frozen(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    queue = make_pools.read_queue(data_dir / "queue.jsonl")
    return manifest, queue


def _terminal(log_dir: Path, allow_synthetic: bool, prompts):
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
        collect.check_existing_records(Path(log_dir), prompts)
    except (ValueError, RuntimeError) as err:
        raise AnalysisRefused(str(err)) from err
    return terminal


def load_scored(data_dir: Path, log_dir: Path, pilot_log_dir: Path, allow_synthetic: bool = False):
    """(manifest, queue, as_collected, rescored): one frame of 120 episodes (60 default + 60 G_14), or refusal."""
    manifest, queue = load_frozen(data_dir)
    prompts = collect.load_prompts(data_dir)
    new = _terminal(log_dir, allow_synthetic, prompts)
    pilot = _terminal(pilot_log_dir, allow_synthetic, prompts)
    default_arm = manifest["default_arm"]
    want_new = {(q.arm_id, q.task_id, q.replicate) for q in queue}
    have_new = {}
    for row in new.itertuples():
        have_new[(str(row.arm_id), str(row.task_id), int(row.replicate))] = str(row.status)
    outside = [k for k in have_new if k not in want_new]
    absent = [k for k in want_new if k not in have_new]
    not_ok = [(k, s) for k, s in have_new.items() if k in want_new and s != emp.STATUS_OK]
    if outside or absent or not_ok:
        raise AnalysisRefused(
            f"incomplete: {len(absent)} of {len(queue)} queued default episodes have no terminal record, "
            f"{len(not_ok)} are not status ok (e.g. {not_ok[:2]}), {len(outside)} records are outside the queue; "
            "the primary result is not reported until every queued episode is scored")
    want_pilot = set()
    for task in manifest["task_ids"]:
        for rep in (0, 1):
            want_pilot.add((WINNER_ARM, task, rep))
    pil = pilot[pilot["arm_id"] == WINNER_ARM]
    have_pilot = {}
    for row in pil.itertuples():
        have_pilot[(str(row.arm_id), str(row.task_id), int(row.replicate))] = str(row.status)
    p_absent = [k for k in want_pilot if k not in have_pilot]
    p_bad = [(k, s) for k, s in have_pilot.items() if k in want_pilot and s != emp.STATUS_OK]
    if p_absent or p_bad:
        raise AnalysisRefused(f"pilot incomplete: {len(p_absent)} of 60 {WINNER_ARM} pilot episodes missing, "
                              f"{len(p_bad)} not status ok")
    keep = []
    for row in pil.itertuples():
        keep.append((str(row.arm_id), str(row.task_id), int(row.replicate)) in want_pilot)
    pil = pil[np.asarray(keep, dtype=bool)]
    combined = pd.concat([new[new["arm_id"] == default_arm], pil], ignore_index=True)
    if len(combined) != 120:
        raise AnalysisRefused(f"expected 120 episodes, found {len(combined)}")
    rescored = st.rescore(combined, SCORE_CAP)
    return manifest, queue, combined, rescored


def t_summary(d: np.ndarray) -> dict:
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    t_crit = float(stats.t.ppf(1 - ALPHA, n - 1))
    return {"mean": mean, "sd": sd, "se": float(se), "t_crit": t_crit,
            "upper_bound_one_sided": mean + t_crit * float(se),
            "lower_bound_one_sided": mean - t_crit * float(se),
            "interval_two_sided_90": [mean - t_crit * float(se), mean + t_crit * float(se)]}


def reading(summary: dict) -> str:
    if summary["lower_bound_one_sided"] >= MARGIN:
        return "reproduced: the +10 in-sample lead replicated at full size"
    if summary["upper_bound_one_sided"] < MARGIN:
        return "shrank: the lead shrank below its in-sample size"
    return "inconclusive"


def bootstrap_mean(d: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(d)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        means[b] = d[rng.integers(0, n, size=n)].mean()
    return {"n_boot": N_BOOT, "seed": SEED, "mean": float(means.mean()),
            "interval_two_sided_90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
            "lower_5pct": float(np.quantile(means, 0.05)), "upper_95pct": float(np.quantile(means, 0.95)),
            "fraction_ge_margin": float((means >= MARGIN).mean())}


def build_table(manifest: dict, frame_rescored, frame_raw) -> list[dict]:
    default_arm = manifest["default_arm"]
    cell = {}
    for tag, frame in (("cap", frame_rescored), ("raw", frame_raw)):
        for row in frame.itertuples():
            cell[(tag, str(row.arm_id), str(row.task_id), int(row.replicate))] = float(row.success)
    table = []
    for task in manifest["task_ids"]:
        rec = {"task_id": task}
        for tag in ("cap", "raw"):
            for name, arm in (("G14", WINNER_ARM), ("D", default_arm)):
                r0 = cell[(tag, arm, task, 0)]
                r1 = cell[(tag, arm, task, 1)]
                rec[f"{name}_{tag}_r0"] = r0
                rec[f"{name}_{tag}_r1"] = r1
                rec[f"{name}_{tag}"] = (r0 + r1) / 2.0
            rec[f"d_{tag}"] = rec[f"G14_{tag}"] - rec[f"D_{tag}"]
        table.append(rec)
    return table


def column(table: list[dict], key: str) -> np.ndarray:
    out = []
    for rec in table:
        out.append(rec[key])
    return np.asarray(out, dtype=float)


def operations(frame, arm: str) -> dict:
    sub = frame[frame["arm_id"] == arm]
    n = len(sub)
    ended = {"agent": 0, "steps": 0, "clock": 0, "unknown": 0}
    for row in sub.itertuples():
        kind = getattr(row, "ended_by", None)
        if kind is None or kind != kind:
            if bool(row.timed_out):
                kind = "clock"
            elif int(row.steps) >= SCORE_CAP and not bool(row.is_done):
                kind = "steps"
            else:
                kind = "agent"
        if kind not in ended:
            kind = "unknown"
        ended[kind] += 1
    return {"arm_id": arm, "episodes": n,
            "mean_cost_usd": float(sub["cost_usd"].astype(float).mean()),
            "total_cost_usd": float(sub["cost_usd"].astype(float).sum()),
            "mean_steps": float(sub["steps"].astype(float).mean()),
            "max_steps": int(sub["steps"].astype(int).max()),
            "episodes_over_30_steps": int((sub["steps"].astype(int) > SCORE_CAP).sum()),
            "fraction_ended_by_agent": ended["agent"] / n,
            "fraction_ended_by_step_cap": ended["steps"] / n,
            "fraction_ended_by_clock": ended["clock"] / n}


def analyze(data_dir: Path = DATA, log_dir: Path = LOGS, pilot_log_dir: Path = PILOT_LOGS,
            out_dir: Path = RESULTS, allow_synthetic: bool = False) -> dict:
    manifest, queue, raw, scored = load_scored(data_dir, log_dir, pilot_log_dir, allow_synthetic)
    table = build_table(manifest, scored, raw)
    n = len(table)
    d_cap = column(table, "d_cap")
    d_raw = column(table, "d_raw")
    prim = t_summary(d_cap)
    conf = t_summary(d_raw)
    insample = manifest["september_in_sample"]["difference"]
    disagreement = {}
    for name in ("G14", "D"):
        r0 = column(table, f"{name}_cap_r0")
        r1 = column(table, f"{name}_cap_r1")
        disagreement[name] = {"rate_rescored": float((r0 != r1).mean()), "n_tasks": n}
    for name in ("G14", "D"):
        r0 = column(table, f"{name}_raw_r0")
        r1 = column(table, f"{name}_raw_r1")
        disagreement[name]["rate_as_collected"] = float((r0 != r1).mean())
    changed = int((scored["success"].astype(int) != raw["success"].astype(int)).sum())
    primary = {
        "study": manifest["study"],
        "synthetic": bool(allow_synthetic),
        "n_tasks": n, "n_episodes_default": len(queue), "n_episodes_g14_pilot": 60,
        "default_arm": manifest["default_arm"], "winner_arm": WINNER_ARM,
        "arm_means_30step": {"G_14": float(column(table, "G14_cap").mean()),
                             "default": float(column(table, "D_cap").mean())},
        "primary": {
            "quantity": "mean over tasks of d_t = G14_t - default_t, both arms 30-step rescored, "
                        "arm score per task = mean of 2 replicates",
            "mean": prim["mean"], "sd": prim["sd"], "se": prim["se"], "t_crit": prim["t_crit"],
            "upper_bound_one_sided_95": prim["upper_bound_one_sided"],
            "lower_bound_one_sided_95": prim["lower_bound_one_sided"],
            "interval_two_sided_90": prim["interval_two_sided_90"],
            "margin": MARGIN,
            "reproduced_lower_bound_ge_margin": bool(prim["lower_bound_one_sided"] >= MARGIN),
            "shrank_upper_bound_lt_margin": bool(prim["upper_bound_one_sided"] < MARGIN),
            "registered_reading": reading(prim),
            "task_bootstrap": bootstrap_mean(d_cap),
        },
        "secondary": {
            "as_collected_CONFOUNDED": {
                "note": "uncapped pilot G_14 vs capped default; the cap difference is a confound",
                "arm_means": {"G_14": float(column(table, "G14_raw").mean()),
                              "default": float(column(table, "D_raw").mean())},
                "mean": conf["mean"], "se": conf["se"],
                "upper_bound_one_sided_95": conf["upper_bound_one_sided"],
                "lower_bound_one_sided_95": conf["lower_bound_one_sided"],
                "interval_two_sided_90": conf["interval_two_sided_90"],
                "task_bootstrap": bootstrap_mean(d_raw),
            },
            "winners_curse": {
                "september_in_sample_G14_minus_default": insample,
                "heldout_G14_minus_default_30step": prim["mean"],
                "shrinkage_points": insample - prim["mean"],
                "heldout_as_collected_confounded": conf["mean"],
                "note": "September: 30 Pre-reg 9 tasks, replicate 0, G_14 chosen as the best G prompt on them",
            },
            "rerun_disagreement": disagreement,
            "operations_by_arm": {"G_14": operations(raw, WINNER_ARM),
                                  "default": operations(raw, manifest["default_arm"])},
            "rescored_changes_vs_recorded_success": changed,
        },
        "disclosure": manifest["disclosure"],
    }
    if allow_synthetic:
        primary["WARNING"] = "SYNTHETIC DRY RUN: fabricated outcomes, not a result"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_synthetic" if allow_synthetic else ""
    (out_dir / f"primary{suffix}.json").write_text(json.dumps(primary, indent=2) + "\n")
    fields = list(table[0].keys())
    with open(out_dir / f"task_table{suffix}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in table:
            writer.writerow(rec)
    return primary


# ---- the synthetic dry run -----------------------------------------------------------------------


def _fake_rows(items, prompts, rng, base, difficulty, profile="prereg9"):
    lines = []
    for index, (pool, arm, task, rep) in enumerate(items):
        p = min(max(base + difficulty[task], 0.02), 0.98)
        success = int(rng.random() < p)
        over = rng.random() < 0.08
        steps = int(rng.integers(31, 40)) if over else int(rng.integers(3, 30))
        lines.append(json.dumps({
            "schema": emp.SCHEMA, "pool": pool, "arm_id": arm, "task_id": task, "replicate": rep,
            "attempt": 1, "status": "ok", "success": success, "cost_usd": float(rng.uniform(0.03, 0.15)),
            "steps": steps, "wallclock_s": float(rng.uniform(10, 150)), "timed_out": False,
            "is_done": True, "ended_by": "agent", "queue_index": index, "pilot": False, "worker": 0,
            "profile": profile, "timeout_s": 180, "prompt_sha256": prompts[arm].sha256,
            "task_dir": "SYNTHETIC", SYNTHETIC_KEY: True}))
    return lines


def make_synthetic_logs(data_dir: Path, log_dir: Path, pilot_log_dir: Path, seed: int = 0) -> None:
    """COMPLETE FAKE logs (new default + pilot G_14), every record marked ``SYNTHETIC_DO_NOT_USE``."""
    manifest, queue = load_frozen(data_dir)
    prompts = collect.load_prompts(data_dir)
    rng = np.random.default_rng(seed)
    difficulty = {}
    for task in manifest["task_ids"]:
        difficulty[task] = float(rng.normal(0.0, 0.12))
    items = []
    for q in queue:
        items.append((q.pool, q.arm_id, q.task_id, q.replicate))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "worker_0.jsonl").write_text("\n".join(_fake_rows(items, prompts, rng, 0.55, difficulty)) + "\n")
    pitems = []
    for task in manifest["task_ids"]:
        for rep in (0, 1):
            pitems.append(("G", WINNER_ARM, task, rep))
    pilot_log_dir.mkdir(parents=True, exist_ok=True)
    (pilot_log_dir / "worker_0.jsonl").write_text("\n".join(_fake_rows(pitems, prompts, rng, 0.65, difficulty)) + "\n")


def synthetic_dry_run(data_dir: Path = DATA, out_dir: Path | None = None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="winner_check_gmail_synthetic_"))
    make_synthetic_logs(data_dir, tmp / "logs", tmp / "pilot_logs")
    target = out_dir if out_dir is not None else tmp / "results"
    primary = analyze(data_dir, tmp / "logs", tmp / "pilot_logs", target, allow_synthetic=True)
    primary["_synthetic_workdir"] = str(tmp)
    return primary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--log-dir", type=Path, default=LOGS)
    ap.add_argument("--pilot-log-dir", type=Path, default=PILOT_LOGS)
    ap.add_argument("--out", type=Path, default=None, help=f"default: {RESULTS}")
    ap.add_argument("--synthetic", action="store_true",
                    help="dry run on fabricated logs in a temp dir; writes only *_synthetic.* files")
    args = ap.parse_args()
    if args.synthetic:
        primary = synthetic_dry_run(args.data, args.out if args.out is not None else RESULTS)
    else:
        primary = analyze(args.data, args.log_dir, args.pilot_log_dir, args.out or RESULTS)
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
