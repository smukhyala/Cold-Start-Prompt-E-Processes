"""Control-check analyzer: does the full-coverage oracle (O2) beat the default prompt (D) on the 110 Leg G tasks?

    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_control_check.py
    ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_control_check.py --synthetic

Offline and read-only: it reads ``data/control_check_gitlab`` (frozen manifest + queue) and the collector's
``logs/control_check_gitlab/worker_*.jsonl`` (plus ``logs/winner_check_gitlab`` for the Leg G comparison), makes no
model call and starts no server. Like ``analyze_winner_check`` it REFUSES to report anything unless every queued
episode has exactly one terminal record with status ``ok`` (no missing, no duplicate, nothing outside the queue,
every prompt sha equal to the frozen pool's) and refuses logs that carry the synthetic marker (except with
``--synthetic``, which runs on fabricated, clearly marked logs in a temp dir). The refusal gate is
``analyze_winner_check.load_scored``, reused as is.

Quantities (O2 = ``GL_oracle_full``, Dnew = ``GL_anchor_baseline`` in this study, Dold = the same arm in Leg G;
an arm's score on a task is the mean of its 2 replicates, each scored at the 30-step budget: success and not
timed out and steps <= 30):

* PRIMARY: mean over the 110 tasks of O2_t - Dnew_t, paired t, one-sided 95% lower bound;
  ``control_holds`` iff the bound > 0.
* SECONDARY: (a) task-cluster bootstrap (10,000, seed 20261012) of the same; (b) day-to-day drift of the default,
  Dnew_t - Dold_t (mean and two-sided 90% interval); (c) the Leg G primary recomputed from
  ``logs/winner_check_gitlab`` (W - Dold, upper bound) and the combined reading; (d) exploratory split of O2 - Dnew
  by whether the task was in the OLD oracle's 40-task cookbook (= the tasks of ``results/audit/matrices/gitlab18.csv``);
  (e) per-arm operations (cost, steps, what ended each episode).

Writes ``results/control_check_gitlab/{primary.json, secondary.json, task_table.csv}``.
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

from cold_start.growing import empirical as emp  # noqa: E402

DATA = ROOT / "data" / "control_check_gitlab"
LOGS = ROOT / "logs" / "control_check_gitlab"
RESULTS = ROOT / "results" / "control_check_gitlab"
LEG_G_DATA = ROOT / "data" / "winner_check_gitlab"
LEG_G_LOGS = ROOT / "logs" / "winner_check_gitlab"
OLD_COOKBOOK_CSV = ROOT / "results" / "audit" / "matrices" / "gitlab18.csv"
SEED = 20261012
N_BOOT = 10_000
ALPHA = awc.ALPHA
MARGIN = awc.MARGIN
SCORE_CAP = awc.SCORE_CAP
SYNTHETIC_KEY = awc.SYNTHETIC_KEY
AnalysisRefused = awc.AnalysisRefused

READING_INFORMATIVE = "Leg G certification is informative with a next-day control"
READING_UNINFORMATIVE = "still uninformative"
READING_CONTROL_HOLDS_LEG_G_OPEN = ("control holds, but the Leg G upper bound is not below 0.10, so Leg G does not "
                                    "certify")
READING_NO_LEG_G = "Leg G logs unavailable: no combined reading"


# ---- statistics ------------------------------------------------------------------------------


def cluster_bootstrap(diff, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Task-cluster bootstrap of the mean of per-task differences (one resample of tasks per draw)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(diff, dtype=float)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lb = float(np.quantile(means, ALPHA))
    return {"n_boot": n_boot, "seed": seed, "mean": float(means.mean()),
            "lower_bound_one_sided_95": lb,
            "interval_two_sided_90": [float(np.quantile(means, ALPHA)), float(np.quantile(means, 1.0 - ALPHA))],
            "control_holds": lb > 0.0, "share_of_resamples_le_zero": float((means <= 0.0).mean())}


def combined_reading(control_holds: bool, leg_g_upper_bound: float | None) -> str:
    if leg_g_upper_bound is None:
        return READING_NO_LEG_G
    if not control_holds:
        return READING_UNINFORMATIVE
    if leg_g_upper_bound < MARGIN:
        return READING_INFORMATIVE
    return READING_CONTROL_HOLDS_LEG_G_OPEN


# ---- loading ---------------------------------------------------------------------------------------


def old_cookbook_tasks(path: Path = OLD_COOKBOOK_CSV) -> set:
    """Task ids the OLD oracle's cookbook covered (the tasks present in the exported gitlab18 matrix)."""
    out = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.add(str(row["task_id"]))
    return out


def _roles(manifest: dict) -> dict:
    roles = {}
    for role in manifest["arms"]:
        roles[manifest["arms"][role]["arm_id"]] = role
    return roles


def build_table(manifest: dict, scored, leg_g_table: dict | None, cookbook: set) -> list[dict]:
    roles = _roles(manifest)
    cell = {}
    for row in scored.itertuples():
        role = roles[str(row.arm_id)]
        cell[(str(row.task_id), role, int(row.replicate))] = float(row.success)
    table = []
    pos = 0
    for task in manifest["task_ids"]:
        pos += 1
        rec = {"queue_pos": pos, "task_id": task}
        for role in ("O2", "D"):
            r0 = cell[(task, role, 0)]
            r1 = cell[(task, role, 1)]
            rec[f"{role}_r0"] = r0
            rec[f"{role}_r1"] = r1
            rec[role] = (r0 + r1) / 2.0
        rec["O2_minus_D"] = rec["O2"] - rec["D"]
        if leg_g_table is not None:
            old = leg_g_table[task]
            rec["Dold"] = old["D"]
            rec["Wold"] = old["W"]
            rec["D_minus_Dold"] = rec["D"] - old["D"]
            rec["W_minus_Dold"] = old["W"] - old["D"]
        else:
            rec["Dold"] = ""
            rec["Wold"] = ""
            rec["D_minus_Dold"] = ""
            rec["W_minus_Dold"] = ""
        rec["in_old_cookbook"] = 1 if task in cookbook else 0
        table.append(rec)
    return table


def _column(table: list[dict], key: str) -> np.ndarray:
    out = []
    for rec in table:
        out.append(rec[key])
    return np.asarray(out, dtype=float)


def per_arm_operations(manifest: dict, scored) -> dict:
    roles = _roles(manifest)
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


def _group_summary(values: list[float]) -> dict:
    n = len(values)
    if n < 2:
        mean = float(np.mean(values)) if n == 1 else None
        return {"n_tasks": n, "mean": mean}
    t = awc.t_summary(values)
    return {"n_tasks": n, "mean": t["mean"], "se": t["se"], "interval_two_sided_90": t["interval_two_sided_90"]}


def cookbook_split(table: list[dict]) -> dict:
    inside = []
    outside = []
    for rec in table:
        if rec["in_old_cookbook"] == 1:
            inside.append(rec["O2_minus_D"])
        else:
            outside.append(rec["O2_minus_D"])
    return {
        "note": "exploratory: O2 - Dnew by whether the task was in the OLD oracle's 40-task cookbook",
        "in_old_cookbook": _group_summary(inside),
        "not_in_old_cookbook": _group_summary(outside),
    }


# ---- the analysis -----------------------------------------------------------------------------


def load_leg_g(leg_g_data: Path, leg_g_logs: Path, allow_synthetic: bool) -> tuple[dict | None, str | None]:
    """``({task: {W, D, O}}, None)`` from the Leg G logs, or ``(None, reason)`` if they cannot be scored."""
    try:
        manifest, _queue, scored, _changed = awc.load_scored(leg_g_data, leg_g_logs, allow_synthetic)
    except AnalysisRefused as err:
        return None, str(err)
    table = awc.build_table(manifest, scored)
    out = {}
    for rec in table:
        out[rec["task_id"]] = {"W": rec["W"], "D": rec["D"], "O": rec["O"]}
    return out, None


def analyze(data_dir: Path = DATA, log_dir: Path = LOGS, out_dir: Path = RESULTS, allow_synthetic: bool = False,
            leg_g_data: Path = LEG_G_DATA, leg_g_logs: Path = LEG_G_LOGS, cookbook_csv: Path = OLD_COOKBOOK_CSV
            ) -> dict:
    manifest, queue, scored, rescored_changes = awc.load_scored(data_dir, log_dir, allow_synthetic)
    leg_g, leg_g_reason = load_leg_g(leg_g_data, leg_g_logs, allow_synthetic)
    if leg_g is not None:
        for task in manifest["task_ids"]:
            if task not in leg_g:
                raise AnalysisRefused(f"task {task} of the control check is not in the Leg G data")
    cookbook = old_cookbook_tasks(cookbook_csv)
    table = build_table(manifest, scored, leg_g, cookbook)
    n = len(table)
    o2 = _column(table, "O2")
    d = _column(table, "D")
    od = o2 - d

    control_t = awc.t_summary(od)
    control_holds = bool(control_t["lower_bound_one_sided"] > 0.0)
    primary = {
        "study": manifest["study"],
        "synthetic": bool(allow_synthetic),
        "n_tasks": n, "n_episodes": len(queue),
        "O2_arm": manifest["arms"]["O2"]["arm_id"], "D_arm": manifest["arms"]["D"]["arm_id"],
        "arm_means": {"O2": float(o2.mean()), "Dnew": float(d.mean())},
        "primary": {
            "quantity": "mean over tasks of O2_t - Dnew_t (arm score = mean of its 2 replicates, 30-step rescored)",
            "mean": control_t["mean"], "sd": control_t["sd"], "se": control_t["se"], "t_crit": control_t["t_crit"],
            "lower_bound_one_sided_95": control_t["lower_bound_one_sided"],
            "interval_two_sided_90": control_t["interval_two_sided_90"],
            "control_holds": control_holds,
        },
        "control_holds": control_holds,
        "note": "positive control holds" if control_holds else
                "positive control failed: the oracle could not be shown better than the default",
        "rescored_changes_vs_recorded_success": rescored_changes,
    }
    if allow_synthetic:
        primary["WARNING"] = "SYNTHETIC DRY RUN: fabricated outcomes, not a result"

    # ---- secondary
    boot = cluster_bootstrap(od)
    leg_g_block = {"available": leg_g is not None}
    leg_g_ub = None
    drift = None
    if leg_g is not None:
        dold = _column(table, "Dold")
        wold = _column(table, "Wold")
        drift_t = awc.t_summary(d - dold)
        drift = {"quantity": "mean over tasks of Dnew_t - Dold_t (same arm GL_anchor_baseline, Leg G vs now)",
                 "mean": drift_t["mean"], "se": drift_t["se"], "interval_two_sided_90": drift_t["interval_two_sided_90"],
                 "Dold_mean": float(dold.mean()), "Dnew_mean": float(d.mean())}
        leg_t = awc.t_summary(wold - dold)
        leg_g_ub = leg_t["upper_bound_one_sided"]
        leg_g_block.update({
            "quantity": "mean over tasks of W_t - Dold_t from logs/winner_check_gitlab (30-step rescored)",
            "mean": leg_t["mean"], "upper_bound_one_sided_95": leg_g_ub,
            "interval_two_sided_90": leg_t["interval_two_sided_90"], "margin": MARGIN,
            "certified_W_not_10pts_better_than_D": bool(leg_g_ub < MARGIN),
        })
    else:
        leg_g_block["reason"] = leg_g_reason
    secondary = {
        "synthetic": bool(allow_synthetic),
        "task_cluster_bootstrap_O2_minus_D": boot,
        "default_drift": drift if drift is not None else {"available": False, "reason": leg_g_reason},
        "leg_g_recomputed": leg_g_block,
        "combined_reading": combined_reading(control_holds, leg_g_ub),
        "old_cookbook_split": cookbook_split(table),
        "operations_by_arm": per_arm_operations(manifest, scored),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "primary.json").write_text(json.dumps(primary, indent=2) + "\n")
    (out_dir / "secondary.json").write_text(json.dumps(secondary, indent=2) + "\n")
    fields = ["queue_pos", "task_id", "O2_r0", "O2_r1", "D_r0", "D_r1", "O2", "D", "O2_minus_D", "Dold", "Wold",
              "D_minus_Dold", "W_minus_Dold", "in_old_cookbook"]
    with open(out_dir / "task_table.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in table:
            writer.writerow(rec)
    return {"primary": primary, "secondary": secondary}


# ---- the synthetic dry run -----------------------------------------------------------------------


def make_synthetic_log(data_dir: Path, log_dir: Path, seed: int = 0, p_o2: float = 0.85, p_d: float = 0.60) -> None:
    """A complete FAKE worker log for the frozen queue, every record marked ``SYNTHETIC_DO_NOT_USE``."""
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    queue = make_pools.read_queue(Path(data_dir) / "gitlab" / "queue.jsonl")
    prompts = collect.load_prompts(Path(data_dir), "gitlab")
    roles = _roles(manifest)
    rng = np.random.default_rng(seed)
    base = {"O2": p_o2, "D": p_d}
    difficulty = {}
    for task in manifest["task_ids"]:
        difficulty[task] = float(rng.normal(0.0, 0.12))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
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
    (Path(log_dir) / "worker_0.jsonl").write_text("\n".join(lines) + "\n")


def synthetic_dry_run(data_dir: Path = DATA, out_dir: Path | None = None, leg_g_data: Path = LEG_G_DATA) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="control_check_synthetic_"))
    make_synthetic_log(data_dir, tmp / "logs")
    awc.make_synthetic_log(leg_g_data, tmp / "leg_g_logs")
    target = out_dir if out_dir is not None else tmp / "results"
    result = analyze(data_dir, tmp / "logs", target, allow_synthetic=True, leg_g_data=leg_g_data,
                     leg_g_logs=tmp / "leg_g_logs")
    result["primary"]["_synthetic_workdir"] = str(tmp)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--log-dir", type=Path, default=LOGS)
    ap.add_argument("--out", type=Path, default=None, help=f"default: {RESULTS} (a temp dir with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="exercise the analyzer end to end on fabricated, clearly marked worker logs in a temp dir")
    args = ap.parse_args(argv)
    if args.synthetic:
        if args.out is not None and Path(args.out).resolve() == RESULTS.resolve():
            ap.error("--synthetic never writes to the real results dir")
        result = synthetic_dry_run(args.data, args.out)
        print(json.dumps(result["primary"], indent=2))
        print(json.dumps(result["secondary"], indent=2))
        return 0
    try:
        result = analyze(args.data, args.log_dir, args.out if args.out is not None else RESULTS)
    except AnalysisRefused as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 2
    print(json.dumps(result["primary"], indent=2))
    print(json.dumps(result["secondary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
