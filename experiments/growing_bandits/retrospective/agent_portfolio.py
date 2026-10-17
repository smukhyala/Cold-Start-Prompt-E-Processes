"""Offline agent-portfolio analysis of frozen prompt/task trials.

No agent episodes or API calls. Every task split reuses an observed task bank and
is exploratory, even when the selector cannot see its held-out task outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from task_transfer import EMPIRICAL, GITLAB, OUTPUT, PAIRED, Pool, load_pools, split_tasks

ROOT = Path(__file__).resolve().parents[3]
LOGS = {"G": ROOT / "logs/empirical_pool", "F": ROOT / "logs/empirical_pool",
        "GLK30": ROOT / "logs/heterogeneity/gitlab"}
SEED = 20261006


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_repeats(source: Path, pool_name: str) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for line in source.open():
        row = json.loads(line)
        if row["pool"] != pool_name or int(row["replicate"]) != 1:
            continue
        if row["status"] != "ok" or int(row["success"]) not in (0, 1):
            raise ValueError(f"invalid repeat: {row}")
        key = row["arm_id"], row["task_id"]
        if key in result:
            raise ValueError(f"duplicate repeat: {key}")
        result[key] = int(row["success"])
    return result


def check_frozen_snapshot(source: Path) -> None:
    manifest = json.loads((source.parent / "manifest.json").read_text())
    stamp = manifest["outcomes_snapshot"]
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if stamp["file"] != source.name or stamp["sha256"] != actual:
        raise ValueError(f"snapshot differs from its frozen manifest: {source}")


def coverage(y: np.ndarray, indices: tuple[int, ...], tasks: np.ndarray) -> np.ndarray:
    return np.max(y[np.ix_(indices, tasks)], axis=0)


def select_team(y: np.ndarray, train: np.ndarray, candidates: np.ndarray,
                size: int, method: str) -> tuple[int, ...]:
    """Choose agents from identical observed training cells for every method."""
    if size > len(candidates):
        raise ValueError("team exceeds candidate set")
    rates = y[np.ix_(candidates, train)].mean(axis=1)
    ordered = sorted(range(len(candidates)), key=lambda k: (-rates[k], int(candidates[k])))
    if method == "top_marginal":
        return tuple(int(candidates[k]) for k in ordered[:size])
    if method not in ("greedy_coverage", "shrunken_coverage"):
        raise ValueError(method)
    selected = [ordered[0]]
    while len(selected) < size:
        options = []
        for k in range(len(candidates)):
            if k in selected:
                continue
            next_ids = tuple(int(candidates[z]) for z in selected + [k])
            empirical = float(coverage(y, next_ids, train).mean())
            if method == "shrunken_coverage":
                independence = 1.0 - float(np.prod(1.0 - rates[selected + [k]]))
                score = (len(train) * empirical + 6.0 * independence) / (len(train) + 6.0)
            else:
                score = empirical
            options.append((-score, -rates[k], int(candidates[k]), k))
        selected.append(min(options)[-1])
    return tuple(int(candidates[k]) for k in selected)


def mean_random_team_coverage(y: np.ndarray, candidates: np.ndarray, tasks: np.ndarray,
                              size: int) -> float:
    """Exact random-team expectation from successes per task, without pair enumeration."""
    n = len(candidates)
    failures = n - y[np.ix_(candidates, tasks)].sum(axis=0)
    none = np.ones(len(tasks), dtype=float)
    for step in range(size):
        none *= np.maximum(failures - step, 0) / (n - step)
    return float(np.mean(1.0 - none))


def repeated_attempts(pool: Pool, repeats: dict[tuple[str, str], int],
                      other_pool: Pool | None, n_boot: int, seed: int) -> tuple[list[dict], list[dict]]:
    if other_pool is not None and other_pool.tasks != pool.tasks:
        raise ValueError("cross-family task banks differ")
    rows = []
    task_to_j = {task: j for j, task in enumerate(pool.tasks)}
    arm_to_i = {arm: i for i, arm in enumerate(pool.arms)}
    for (arm, task), repeat in sorted(repeats.items()):
        if arm not in arm_to_i or task not in task_to_j:
            raise ValueError(f"repeat outside main grid: {arm}, {task}")
        i, j = arm_to_i[arm], task_to_j[task]
        first = int(pool.y[i, j])
        within = float((pool.y[:, j].sum() - first) / (len(pool.arms) - 1))
        cross = float(other_pool.y[:, j].mean()) if other_pool is not None else np.nan
        rows.append({"pool": pool.name, "arm_id": arm, "task_id": task, "first": first,
                     "repeat": repeat, "within_second_rate": within,
                     "cross_second_rate": cross, "same_coverage": max(first, repeat),
                     "within_coverage": first + (1 - first) * within,
                     "cross_coverage": first + (1 - first) * cross if other_pool is not None else np.nan,
                     "within_delta": (1 - first) * (within - repeat),
                     "second_quality_delta": within - repeat,
                     "excess_disagreement": 2 * (1 - first) * (within - repeat) - (within - repeat)})
    if not rows:
        raise ValueError(f"{pool.name}: no repeats")
    a = np.asarray([[r["same_coverage"], r["within_coverage"], r["within_delta"],
                     r["second_quality_delta"], r["excess_disagreement"],
                     r["cross_coverage"], r["cross_coverage"] - r["within_coverage"]]
                    for r in rows], dtype=float)
    labels = ("same_coverage", "within_coverage", "within_delta", "second_quality_delta",
              "excess_disagreement", "cross_coverage", "cross_minus_within")
    valid_cols = np.flatnonzero(np.isfinite(a).any(axis=0))
    task_idx = np.asarray([task_to_j[r["task_id"]] for r in rows])
    observed_tasks = np.unique(task_idx)
    task_rows = {j: np.flatnonzero(task_idx == j) for j in observed_tasks}
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, a.shape[1]), np.nan)
    for b in range(n_boot):
        sample = rng.choice(observed_tasks, len(observed_tasks), replace=True)
        idx = np.concatenate([task_rows[j] for j in sample])
        boots[b, valid_cols] = np.mean(a[np.ix_(idx, valid_cols)], axis=0)
    return [{"pool": pool.name, "metric": label, "estimate": float(np.nanmean(a[:, k])),
             "task_boot_lo": float(np.nanquantile(boots[:, k], .025)),
             "task_boot_hi": float(np.nanquantile(boots[:, k], .975)),
             "n_repeat_cells": len(rows), "n_repeat_tasks": len(observed_tasks)}
            for k, label in enumerate(labels) if k in valid_cols], rows


def heldout_portfolios(pool: Pool, n_splits: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    all_candidates = np.arange(len(pool.arms))
    for split in range(n_splits):
        train, test = split_tasks(pool.tasks, rng)
        for size in (1, 2, 3):
            random_rate = float(pool.y[:, test].mean()) if size == 1 else mean_random_team_coverage(
                pool.y, all_candidates, test, size)
            methods = ("top_marginal",) if size == 1 else (
                "top_marginal", "greedy_coverage", "shrunken_coverage")
            for method in methods:
                team = select_team(pool.y, train, all_candidates, size, method)
                rows.append({"pool": pool.name, "split": split, "team_size": size,
                             "method": method, "train_episodes": len(all_candidates) * len(train),
                             "train_coverage": float(coverage(pool.y, team, train).mean()),
                             "test_coverage": float(coverage(pool.y, team, test).mean()),
                             "random_team_test_coverage": random_rate,
                             "team": "|".join(pool.arms[i] for i in team)})
    return rows


def nested_training_subsets(tasks: tuple[str, ...], train: np.ndarray,
                            rng: np.random.Generator) -> dict[int, np.ndarray]:
    """One shuffled order per tier, so a smaller audit is contained in a larger one."""
    groups: dict[str, list[int]] = defaultdict(list)
    for j in train:
        groups[tasks[int(j)].split("_")[1][0]].append(int(j))
    ordered = {key: rng.permutation(value) for key, value in groups.items()}
    return {n: np.asarray([int(j) for key in sorted(ordered)
                            for j in ordered[key][:n // len(ordered)]])
            for n in (3, 6, 12) if n % len(ordered) == 0
            and all(len(group) >= n // len(ordered) for group in ordered.values())}


def fixed_budget_breadth(pool: Pool, n_splits: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    schedules = ((48, ((4, 12), (8, 6), (16, 3))),
                 (96, ((8, 12), (16, 6), (32, 3))))
    n_strata = len({task.split("_")[1][0] for task in pool.tasks})
    for split in range(n_splits):
        train, test = split_tasks(pool.tasks, rng)
        arm_order = rng.permutation(len(pool.arms))
        train_subsets = nested_training_subsets(pool.tasks, train, rng)
        for budget, choices in schedules:
            for k, n_tasks in choices:
                if k > len(pool.arms) or n_tasks % n_strata:
                    continue
                candidates = arm_order[:k]
                observed = train_subsets[n_tasks]
                single = select_team(pool.y, observed, candidates, 1, "top_marginal")
                pair = select_team(pool.y, observed, candidates, 2, "top_marginal")
                rows.append({"pool": pool.name, "split": split, "budget": budget,
                             "n_candidates": k, "n_train_tasks": n_tasks,
                             "single_test": float(coverage(pool.y, single, test).mean()),
                             "pair_test": float(coverage(pool.y, pair, test).mean()),
                             "random_candidate_test": float(pool.y[np.ix_(candidates, test)].mean()),
                             "random_pair_test": mean_random_team_coverage(pool.y, candidates, test, 2)})
    return rows


def summarize_portfolios(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["pool"], row["team_size"], row["method"]].append(row)
    output = []
    by_split = {(r["pool"], r["split"], r["team_size"], r["method"]): r for r in rows}
    for (pool, size, method), subset in grouped.items():
        val = np.asarray([r["test_coverage"] for r in subset])
        random = np.asarray([r["random_team_test_coverage"] for r in subset])
        top = np.asarray([by_split[pool, r["split"], size, "top_marginal"]["test_coverage"]
                          for r in subset])
        gain = val - top
        output.append({"pool": pool, "team_size": size, "method": method,
                       "n_reused_splits": len(subset), "mean_test_coverage": float(val.mean()),
                       "minus_random": float((val-random).mean()),
                       "minus_top_marginal": float(gain.mean()),
                       "split_p05_minus_top": float(np.quantile(gain, .05)),
                       "split_p95_minus_top": float(np.quantile(gain, .95)),
                       "share_splits_beating_top": float(np.mean(gain > 0))})
    return output


def summarize_breadth(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["pool"], row["budget"], row["n_candidates"]].append(row)
    output = []
    for (pool, budget, k), subset in grouped.items():
        output.append({"pool": pool, "budget": budget, "n_candidates": k,
                       "n_train_tasks": subset[0]["n_train_tasks"],
                       "n_reused_splits": len(subset),
                       "selected_single_test": float(np.mean([r["single_test"] for r in subset])),
                       "selected_pair_test": float(np.mean([r["pair_test"] for r in subset])),
                       "random_candidate_test": float(np.mean([r["random_candidate_test"] for r in subset])),
                       "random_pair_test": float(np.mean([r["random_pair_test"] for r in subset]))})
    return output


def loo_portfolios(pool: Pool) -> list[dict]:
    """Each task once as test; overlapping training sets make this sensitivity only."""
    rows = []
    candidates = np.arange(len(pool.arms))
    for heldout in range(len(pool.tasks)):
        train = np.asarray([j for j in range(len(pool.tasks)) if j != heldout])
        test = np.asarray([heldout])
        for size in (1, 2, 3):
            random_rate = mean_random_team_coverage(pool.y, candidates, test, size)
            methods = ("top_marginal",) if size == 1 else (
                "top_marginal", "greedy_coverage", "shrunken_coverage")
            for method in methods:
                team = select_team(pool.y, train, candidates, size, method)
                rows.append({"pool": pool.name, "heldout_task": pool.tasks[heldout],
                             "team_size": size, "method": method,
                             "test_coverage": int(coverage(pool.y, team, test)[0]),
                             "random_team_test_coverage": random_rate,
                             "team": "|".join(pool.arms[i] for i in team)})
    return rows


def summarize_loo(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["pool"], row["team_size"], row["method"]].append(row)
    return [{"pool": pool, "team_size": size, "method": method,
             "n_tasks": len(group),
             "coverage": float(np.mean([r["test_coverage"] for r in group])),
             "minus_random": float(np.mean([r["test_coverage"] - r["random_team_test_coverage"]
                                            for r in group]))}
            for (pool, size, method), group in grouped.items()]


def load_log_metrics(log_dir: Path, pool_name: str) -> dict[tuple[str, str, int], dict]:
    result: dict[tuple[str, str, int], dict] = {}
    for path in sorted(log_dir.glob("worker_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row["pool"] != pool_name or row["status"] != "ok":
                continue
            key = row["arm_id"], row["task_id"], int(row["replicate"])
            if key in result:
                raise ValueError(f"duplicate ok log row {key}")
            result[key] = {"cost": float(row["cost_usd"]), "steps": int(row["steps"]),
                           "timed_out": bool(row["timed_out"]),
                           "raw_success": int(row["success"]),
                           "trace_path": row.get("task_dir")}
    return result


def trajectory_summary(pool: Pool, repeats: dict[tuple[str, str], int]) -> list[dict]:
    raw = load_log_metrics(LOGS[pool.name], "GLK" if pool.name == "GLK30" else pool.name)
    joined = []
    for i, arm in enumerate(pool.arms):
        for j, task in enumerate(pool.tasks):
            key = arm, task, 0
            if key not in raw:
                raise ValueError(f"missing log row: {key}")
            item = raw[key]
            if pool.name != "GLK30" and item["raw_success"] != int(pool.y[i, j]):
                raise ValueError(f"main snapshot/log mismatch: {key}")
            if pool.name == "GLK30" and int(pool.y[i, j]) > item["raw_success"]:
                raise ValueError(f"30-step score exceeds uncapped score: {key}")
            joined.append({"success": int(pool.y[i, j]), **item})
    success = [r for r in joined if r["success"]]
    failure = [r for r in joined if not r["success"]]
    rows = []
    for label, group in (("all", joined), ("success", success), ("failure", failure)):
        rows.append({"pool": pool.name, "group": label, "n": len(group),
                     "success_rate": float(np.mean([r["success"] for r in group])),
                     "mean_cost_usd": float(np.mean([r["cost"] for r in group])),
                     "median_steps": float(np.median([r["steps"] for r in group])),
                     "timeout_rate": float(np.mean([r["timed_out"] for r in group])),
                     "trace_history_present": float(np.mean([
                         bool(r["trace_path"]) and (Path(r["trace_path"]) / "history.json").exists()
                         for r in group])),
                     "raw_success_rate": float(np.mean([r["raw_success"] for r in group]))})
    for (arm, task), repeat in repeats.items():
        if (arm, task, 1) not in raw:
            raise ValueError(f"missing repeat log row: {arm}, {task}")
        if pool.name != "GLK30" and raw[arm, task, 1]["raw_success"] != repeat:
            raise ValueError("snapshot/log repeat mismatch")
    return rows


def repeat_mechanisms(pool: Pool, cells: list[dict]) -> tuple[list[dict], list[dict]]:
    """Failure-conditioned success and saved, as-collected second-attempt costs."""
    raw = load_log_metrics(LOGS[pool.name], "GLK" if pool.name == "GLK30" else pool.name)
    by_tier: dict[str, list[dict]] = defaultdict(list)
    costs = []
    for row in cells:
        tier = row["task_id"].split("_")[1][0]
        by_tier[tier].append(row)
        arm, task = row["arm_id"], row["task_id"]
        same_cost = raw[arm, task, 1]["cost"]
        alternatives = [raw[b, task, 0]["cost"] for b in pool.arms if b != arm]
        costs.append({"same_second_cost": same_cost,
                      "different_second_cost": float(np.mean(alternatives)),
                      "first_failed": int(row["first"] == 0)})
    tier_rows = []
    for tier, subset in sorted(by_tier.items()):
        failed = [r for r in subset if r["first"] == 0]
        tier_rows.append({"pool": pool.name, "tier": tier, "n_repeats": len(subset),
                          "n_first_failures": len(failed),
                          "repeat_success_after_failure": float(np.mean([r["repeat"] for r in failed]))
                          if failed else np.nan,
                          "different_success_after_failure": float(np.mean(
                              [r["within_second_rate"] for r in failed])) if failed else np.nan})
    cost_rows = []
    for group, subset in (("all", costs), ("after_first_failure",
                                         [r for r in costs if r["first_failed"]])):
        cost_rows.append({"pool": pool.name, "group": group, "n": len(subset),
                          "mean_same_second_cost_usd": float(np.mean([
                              r["same_second_cost"] for r in subset])),
                          "mean_different_second_cost_usd": float(np.mean([
                              r["different_second_cost"] for r in subset]))})
    return tier_rows, cost_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=500)
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT / "agent_portfolio")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check_frozen_snapshot(EMPIRICAL)
    check_frozen_snapshot(GITLAB)
    pools = load_pools()
    named = {p.name: p for p in pools}
    primary = [named[n] for n in ("G", "F", "GLK30", "GitLab17")]
    repeat_summaries, repeat_cells, trajectories, repeat_tiers, repeat_costs = [], [], [], [], []
    for k, pool in enumerate(primary[:3]):
        source = EMPIRICAL if pool.name in ("G", "F") else GITLAB
        repeats = load_repeats(source, "GLK" if pool.name == "GLK30" else pool.name)
        other = named["F" if pool.name == "G" else "G"] if pool.name in ("G", "F") else None
        summary, cells = repeated_attempts(pool, repeats, other, args.boot, args.seed + k)
        repeat_summaries.extend(summary)
        repeat_cells.extend(cells)
        trajectories.extend(trajectory_summary(pool, repeats))
        tiers, costs = repeat_mechanisms(pool, cells)
        repeat_tiers.extend(tiers)
        repeat_costs.extend(costs)
    write_csv(args.output / "repeat_summary.csv", repeat_summaries)
    write_csv(args.output / "repeat_cells.csv", repeat_cells)
    write_csv(args.output / "trajectory_summary.csv", trajectories)
    write_csv(args.output / "repeat_by_tier.csv", repeat_tiers)
    write_csv(args.output / "repeat_cost_summary.csv", repeat_costs)
    portfolio_rows, breadth_rows, loo_rows = [], [], []
    for k, pool in enumerate(primary):
        portfolio_rows.extend(heldout_portfolios(pool, args.splits, args.seed + 1000 * k))
        breadth_rows.extend(fixed_budget_breadth(pool, args.splits, args.seed + 1000 * k + 500))
        loo_rows.extend(loo_portfolios(pool))
    write_csv(args.output / "portfolio_splits.csv", portfolio_rows)
    write_csv(args.output / "portfolio_summary.csv", summarize_portfolios(portfolio_rows))
    write_csv(args.output / "breadth_splits.csv", breadth_rows)
    write_csv(args.output / "breadth_summary.csv", summarize_breadth(breadth_rows))
    write_csv(args.output / "loo_tasks.csv", loo_rows)
    write_csv(args.output / "loo_summary.csv", summarize_loo(loo_rows))
    sources = [EMPIRICAL, GITLAB, PAIRED]
    log_sources = sorted(set(LOGS.values()))
    log_files = sorted(path for directory in log_sources for path in directory.glob("worker_*.jsonl"))
    provenance = {"seed": args.seed, "n_reused_splits": args.splits,
                  "n_task_bootstrap": args.boot,
                  "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources + log_files},
                  "note": "Exploratory reanalysis; no new agent episodes; splits reuse the same task banks."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
