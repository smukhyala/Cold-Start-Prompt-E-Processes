"""Pre-registration 13 analyzer: the REGISTERED analysis of the GitLab winner check.

    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_winner_check.py
    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_winner_check.py --synthetic

Offline and read-only: it reads ``data/winner_check_gitlab`` (frozen manifest + queue) and the collector's
``logs/winner_check_gitlab/worker_*.jsonl``, makes no model call and starts no server, and nothing the
collector does depends on it -- no decision during collection may use its output. It REFUSES to report the
primary result unless every queued episode has a terminal record with status ``ok`` (no ``missing``, no
duplicates, no record outside the queue, every prompt sha equal to the frozen pool's), and refuses logs that
carry the synthetic marker.

Registered quantities (W = the old-data GLK winner, D = default prompt, O = planted positive control):

* per task, an arm's score is the mean of its 2 replicates (each scored at the 30-step budget: success and not
  timed out and steps <= 30); d_t = W_t - D_t.
* PRIMARY: one-sided 95% upper confidence bound on mean(d), paired t over tasks;
  "certified: W is not >= 10 points better than D" iff UB < 0.10. Also the mean and the 90% two-sided interval.
* POSITIVE CONTROL (the study is informative only if it holds): one-sided 95% lower bound on mean(O_t - D_t) > 0.
* SECONDARY: task-cluster bootstrap (10,000, seed 20261010); anytime-valid paths for reference (E5 futility,
  delta 0.10 and 0.15, K = 1, on d_t; E1b efficacy for O vs D on binary replicate outcomes), replayed in queue
  task order; rerun disagreement per arm; cost / steps / what ended each episode per arm; the old-data in-sample
  advantage of W over D on block A against the new held-out estimate (the winner's-curse shrinkage).

Writes ``results/winner_check_gitlab/{primary.json, task_table.csv, secondary.json}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import collect  # noqa: E402
import make_pools  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from experiments.growing_bandits.audit import flatness_eprocess as flat  # noqa: E402
from experiments.growing_bandits.audit import futility_eprocess as fut  # noqa: E402
from experiments.growing_bandits.empirical import study as st  # noqa: E402

DATA = ROOT / "data" / "winner_check_gitlab"
LOGS = ROOT / "logs" / "winner_check_gitlab"
RESULTS = ROOT / "results" / "winner_check_gitlab"
SEED = 20261010
N_BOOT = 10_000
ALPHA = 0.05
MARGIN = 0.10
FUTILITY_DELTAS = (0.10, 0.15)
SCORE_CAP = 30
SYNTHETIC_KEY = "SYNTHETIC_DO_NOT_USE"


class AnalysisRefused(RuntimeError):
    """The log is not a complete, scored, genuine record of the frozen queue."""


# ---- statistics ------------------------------------------------------------------------------


def t_summary(x, alpha: float = ALPHA) -> dict:
    """Mean, sd, se and the paired-t bounds of a sample: one-sided (1 - alpha) upper/lower bound and the
    two-sided (1 - 2 alpha) interval (the same critical value t_{1-alpha, n-1})."""
    v = np.asarray(x, dtype=float)
    n = len(v)
    if n < 2:
        raise ValueError("a paired t interval needs at least 2 tasks")
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    crit = float(stats.t.ppf(1.0 - alpha, n - 1))
    return {"n": n, "mean": mean, "sd": sd, "se": se, "t_crit": crit,
            "upper_bound_one_sided": mean + crit * se, "lower_bound_one_sided": mean - crit * se,
            "interval_two_sided_90": [mean - crit * se, mean + crit * se]}


def bootstrap_means(d, o_minus_d, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Task-cluster bootstrap of mean(d) and mean(o - d) (same resampled tasks for both)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    dm = np.asarray(d, dtype=float)[idx].mean(axis=1)
    om = np.asarray(o_minus_d, dtype=float)[idx].mean(axis=1)
    ub = float(np.quantile(dm, 1.0 - ALPHA))
    lb = float(np.quantile(om, ALPHA))
    return {
        "n_boot": n_boot, "seed": seed,
        "W_minus_D": {"mean": float(dm.mean()), "upper_bound_one_sided_95": ub,
                      "interval_two_sided_90": [float(np.quantile(dm, ALPHA)), float(np.quantile(dm, 1.0 - ALPHA))],
                      "certified_not_10pts_better": ub < MARGIN, "share_of_resamples_ge_margin": float((dm >= MARGIN).mean())},
        "O_minus_D": {"mean": float(om.mean()), "lower_bound_one_sided_95": lb,
                      "interval_two_sided_90": [float(np.quantile(om, ALPHA)), float(np.quantile(om, 1.0 - ALPHA))],
                      "positive_control_holds": lb > 0.0, "share_of_resamples_le_zero": float((om <= 0.0).mean())},
    }


def eprocess_summary(log_path: np.ndarray, alpha: float = ALPHA) -> dict:
    """Final / max log e-value and the first (1-based) task at which the path reaches 1/alpha."""
    path = np.asarray(log_path, dtype=float)
    threshold = math.log(1.0 / alpha)
    hit = np.nonzero(path >= threshold)[0]
    first = int(hit[0]) + 1 if len(hit) > 0 else None
    return {"final_log_e": float(path[-1]), "max_log_e": float(path.max()), "threshold_log": threshold,
            "crossed": first is not None, "first_crossing_task": first}


# ---- loading and the refusal gate ------------------------------------------------------------------


def load_frozen(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    queue = make_pools.read_queue(data_dir / "gitlab" / "queue.jsonl")
    return manifest, queue


def load_scored(data_dir: Path, log_dir: Path, allow_synthetic: bool = False):
    """The complete, scored terminal frame, or `AnalysisRefused`."""
    manifest, queue = load_frozen(data_dir)
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
    prompts = collect.load_prompts(data_dir, "gitlab")
    try:
        collect.check_existing_records(Path(log_dir), prompts)
    except RuntimeError as err:
        raise AnalysisRefused(str(err)) from err

    queue_keys = {}
    for item in queue:
        queue_keys[(item.arm_id, item.task_id, item.replicate)] = item.index
    have = {}
    for row in terminal.itertuples():
        have[(str(row.arm_id), str(row.task_id), int(row.replicate))] = str(row.status)
    outside = []
    for key in have:
        if key not in queue_keys:
            outside.append(key)
    absent = []
    not_ok = []
    for key in queue_keys:
        if key not in have:
            absent.append(key)
        elif have[key] != emp.STATUS_OK:
            not_ok.append((key, have[key]))
    if outside or absent or not_ok:
        raise AnalysisRefused(
            f"incomplete: {len(absent)} of {len(queue)} queued episodes have no terminal record, "
            f"{len(not_ok)} are not status ok (e.g. {not_ok[:2]}), {len(outside)} records are outside the queue; "
            "the primary result is not reported until every queued episode is scored")
    scored = st.rescore(terminal, SCORE_CAP)
    changed = int((scored["success"].astype(int) != terminal["success"].astype(int)).sum())
    return manifest, queue, scored, changed


# ---- the analysis -----------------------------------------------------------------------------


def _arm_role(manifest: dict) -> dict:
    roles = {}
    for role in ("W", "D", "O"):
        roles[manifest["arms"][role]["arm_id"]] = role
    return roles


def build_table(manifest: dict, scored) -> list[dict]:
    roles = _arm_role(manifest)
    cell = {}
    for row in scored.itertuples():
        role = roles[str(row.arm_id)]
        cell[(str(row.task_id), role, int(row.replicate))] = float(row.success)
    table = []
    pos = 0
    for task in manifest["task_ids"]:
        pos += 1
        rec = {"queue_pos": pos, "task_id": task}
        for role in ("W", "D", "O"):
            r0 = cell[(task, role, 0)]
            r1 = cell[(task, role, 1)]
            rec[f"{role}_r0"] = r0
            rec[f"{role}_r1"] = r1
            rec[role] = (r0 + r1) / 2.0
        rec["W_minus_D"] = rec["W"] - rec["D"]
        rec["O_minus_D"] = rec["O"] - rec["D"]
        table.append(rec)
    return table


def _column(table: list[dict], key: str) -> np.ndarray:
    out = []
    for rec in table:
        out.append(rec[key])
    return np.asarray(out, dtype=float)


def per_arm_operations(manifest: dict, scored) -> dict:
    roles = _arm_role(manifest)
    out = {}
    for arm_id in roles:
        sub = scored[scored["arm_id"] == arm_id]
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
        out[roles[arm_id]] = {
            "arm_id": arm_id, "episodes": n,
            "mean_cost_usd": float(sub["cost_usd"].astype(float).mean()),
            "total_cost_usd": float(sub["cost_usd"].astype(float).sum()),
            "mean_steps": float(sub["steps"].astype(float).mean()),
            "mean_wallclock_s": float(sub["wallclock_s"].astype(float).mean()) if "wallclock_s" in sub.columns else None,
            "fraction_ended_by_agent": ended["agent"] / n,
            "fraction_ended_by_step_cap": ended["steps"] / n,
            "fraction_ended_by_clock": ended["clock"] / n,
            "fraction_unknown_end": ended["unknown"] / n,
        }
    return out


def analyze(data_dir: Path = DATA, log_dir: Path = LOGS, out_dir: Path = RESULTS,
            allow_synthetic: bool = False) -> dict:
    manifest, queue, scored, rescored_changes = load_scored(data_dir, log_dir, allow_synthetic)
    table = build_table(manifest, scored)
    n = len(table)
    w = _column(table, "W")
    d = _column(table, "D")
    o = _column(table, "O")
    diff = w - d
    od = o - d

    primary_t = t_summary(diff)
    control_t = t_summary(od)
    certified = primary_t["upper_bound_one_sided"] < MARGIN
    control_holds = control_t["lower_bound_one_sided"] > 0.0
    primary = {
        "study": manifest["study"],
        "synthetic": bool(allow_synthetic),
        "eligibility": manifest["eligibility"],
        "n_tasks": n, "n_episodes": len(queue),
        "W_arm": manifest["arms"]["W"]["arm_id"],
        "arm_means": {"W": float(w.mean()), "D": float(d.mean()), "O": float(o.mean())},
        "primary": {
            "quantity": "mean over tasks of d_t = W_t - D_t (arm score = mean of its 2 replicates)",
            "mean": primary_t["mean"], "sd": primary_t["sd"], "se": primary_t["se"], "t_crit": primary_t["t_crit"],
            "upper_bound_one_sided_95": primary_t["upper_bound_one_sided"],
            "interval_two_sided_90": primary_t["interval_two_sided_90"],
            "margin": MARGIN,
            "certified_W_not_10pts_better_than_D": bool(certified),
        },
        "positive_control": {
            "quantity": "mean over tasks of O_t - D_t",
            "mean": control_t["mean"], "se": control_t["se"],
            "lower_bound_one_sided_95": control_t["lower_bound_one_sided"],
            "interval_two_sided_90": control_t["interval_two_sided_90"],
            "holds": bool(control_holds),
        },
        "study_informative": bool(control_holds),
        "certification_valid": bool(certified and control_holds),
        "note": ("positive control failed: the study could not have detected a real effect, so a certification "
                 "is not informative" if not control_holds else "positive control holds"),
        "rescored_changes_vs_recorded_success": rescored_changes,
    }
    if allow_synthetic:
        primary["WARNING"] = "SYNTHETIC DRY RUN: fabricated outcomes, not a result"

    # ---- secondary
    boot = bootstrap_means(diff, od)

    ep = {}
    D_fut = diff.reshape(1, 1, n)
    for delta in FUTILITY_DELTAS:
        path = fut.futility_log_paths(D_fut, delta)[0, 0]
        ep[f"E5_futility_delta_{delta:.2f}"] = eprocess_summary(path)
    for rep in (0, 1):
        o_bin = _column(table, f"O_r{rep}")
        d_bin = _column(table, f"D_r{rep}")
        Y = np.stack([o_bin, d_bin]).reshape(1, 2, n)
        path = flat.e1b_log_paths(Y, np.array([1]), np.array([[True, False]]))[0]
        ep[f"E1b_efficacy_O_vs_D_replicate{rep}"] = eprocess_summary(path)

    disagreement = {}
    for role in ("W", "D", "O"):
        r0 = _column(table, f"{role}_r0")
        r1 = _column(table, f"{role}_r1")
        disagreement[role] = {"rate": float((r0 != r1).mean()), "n_tasks": n,
                              "replicate0_rate": float(r0.mean()), "replicate1_rate": float(r1.mean())}

    old = manifest["old_data_block_a"]
    curse = {
        "old_in_sample_W_minus_D_glk30": old["W_glk30_rate"] - old["D_glk30_rate"],
        "old_in_sample_W_minus_D_uncapped": old["W_glk_rate"] - old["D_glk_rate"],
        "old_in_sample_O_minus_D_glk30": old["O_glk30_rate"] - old["D_glk30_rate"],
        "new_heldout_W_minus_D": primary_t["mean"],
        "new_heldout_O_minus_D": control_t["mean"],
        "shrinkage_W_minus_D_glk30": (old["W_glk30_rate"] - old["D_glk30_rate"]) - primary_t["mean"],
        "note": "old = 30 block-A tasks, replicate 0, W chosen as the best of 40 on these same tasks",
    }
    secondary = {
        "synthetic": bool(allow_synthetic),
        "task_cluster_bootstrap": boot,
        "anytime_valid_reference": {
            "order": "queue task order (manifest.task_ids)",
            "E5_futility_note": "D_t = W_t - D_t with replicate-mean scores in [-1, 1]; H0: E[d] >= delta; K = 1",
            "E1b_note": "binary replicate outcomes of O vs D, one replicate at a time (E1b needs 0/1 outcomes)",
            "paths": ep,
        },
        "rerun_disagreement_replicate0_vs_1": disagreement,
        "operations_by_arm": per_arm_operations(manifest, scored),
        "winners_curse": curse,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primary.json").write_text(json.dumps(primary, indent=2) + "\n")
    (out_dir / "secondary.json").write_text(json.dumps(secondary, indent=2) + "\n")
    fields = ["queue_pos", "task_id", "W_r0", "W_r1", "D_r0", "D_r1", "O_r0", "O_r1", "W", "D", "O",
              "W_minus_D", "O_minus_D"]
    with open(out_dir / "task_table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in table:
            writer.writerow(rec)
    return primary


# ---- the synthetic dry run -----------------------------------------------------------------------


def make_synthetic_log(data_dir: Path, log_dir: Path, seed: int = 0) -> None:
    """A complete FAKE worker log for the frozen queue, every record marked ``SYNTHETIC_DO_NOT_USE``."""
    manifest, queue = load_frozen(data_dir)
    prompts = collect.load_prompts(data_dir, "gitlab")
    roles = _arm_role(manifest)
    rng = np.random.default_rng(seed)
    base = {"W": 0.62, "D": 0.60, "O": 0.82}
    difficulty = {}
    for task in manifest["task_ids"]:
        difficulty[task] = float(rng.normal(0.0, 0.12))
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in queue:
        p = min(max(base[roles[item.arm_id]] + difficulty[item.task_id], 0.02), 0.98)
        success = int(rng.random() < p)
        capped = rng.random() < 0.08
        steps = 30 if capped else int(rng.integers(4, 30))
        if capped:
            success = 0
        lines.append(json.dumps({
            "schema": emp.SCHEMA, "pool": item.pool, "arm_id": item.arm_id, "task_id": item.task_id,
            "replicate": item.replicate, "attempt": 1, "status": "ok", "success": success,
            "cost_usd": float(rng.uniform(0.05, 0.15)), "steps": steps, "wallclock_s": float(rng.uniform(20, 300)),
            "timed_out": False, "is_done": not capped, "ended_by": "steps" if capped else "agent",
            "queue_index": item.index, "pilot": False, "worker": 0, "profile": "gitlab", "timeout_s": 600,
            "prompt_sha256": prompts[item.arm_id].sha256, "task_dir": "SYNTHETIC", SYNTHETIC_KEY: True,
        }))
    (log_dir / "worker_0.jsonl").write_text("\n".join(lines) + "\n")


def synthetic_dry_run(data_dir: Path = DATA, out_dir: Path | None = None, keep: bool = False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="winner_check_synthetic_"))
    make_synthetic_log(data_dir, tmp / "logs")
    target = out_dir if out_dir is not None else tmp / "results"
    primary = analyze(data_dir, tmp / "logs", target, allow_synthetic=True)
    primary["_synthetic_workdir"] = str(tmp)
    return primary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--log-dir", type=Path, default=LOGS)
    ap.add_argument("--out", type=Path, default=None, help=f"default: {RESULTS} (a temp dir with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="exercise the analyzer end to end on a fabricated, clearly marked worker log in a temp dir")
    args = ap.parse_args(argv)
    if args.synthetic:
        if args.out is not None and Path(args.out).resolve() == RESULTS.resolve():
            ap.error("--synthetic never writes to the real results dir")
        primary = synthetic_dry_run(args.data, args.out)
        print(json.dumps(primary, indent=2))
        return 0
    try:
        primary = analyze(args.data, args.log_dir, args.out if args.out is not None else RESULTS)
    except AnalysisRefused as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 2
    print(json.dumps(primary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
