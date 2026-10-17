"""Retrospective, task-held-out prompt selection on existing outcome matrices.

No model/API calls are made. Splits are repeated views of the same finite task bank,
not independent replications or a prospective validation set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results" / "growing_bandits" / "retrospective"
EMPIRICAL = ROOT / "data" / "empirical_pool" / "reservoirs" / "outcomes_snapshot.jsonl"
GITLAB = ROOT / "data" / "heterogeneity" / "reservoirs_glk30" / "outcomes_snapshot.jsonl"
PAIRED = ROOT / "results" / "gitlab_strong_arm" / "paired" / "paired_results.csv"


@dataclass(frozen=True)
class Pool:
    name: str
    arms: tuple[str, ...]
    tasks: tuple[str, ...]
    y: np.ndarray
    source: Path
    excluded: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(name: str, records: list[tuple[str, str, int]], source: Path,
            excluded: tuple[str, ...] = ()) -> Pool:
    records = [r for r in records if r[0] not in excluded]
    arms = tuple(sorted({a for a, _, _ in records}))
    tasks = tuple(sorted({t for _, t, _ in records}))
    if not arms or not tasks:
        raise ValueError(f"{name}: empty arm or task set")
    arm_index = {a: i for i, a in enumerate(arms)}
    task_index = {t: j for j, t in enumerate(tasks)}
    y = np.full((len(arms), len(tasks)), -1, dtype=np.int8)
    for arm, task, outcome in records:
        if outcome not in (0, 1):
            raise ValueError(f"{name}: nonbinary outcome")
        i, j = arm_index[arm], task_index[task]
        if y[i, j] != -1:
            raise ValueError(f"{name}: duplicate arm-task cell {arm}, {task}")
        y[i, j] = outcome
    if np.any(y < 0):
        raise ValueError(f"{name}: incomplete arm-task matrix")
    return Pool(name, arms, tasks, y, source, excluded)


def load_pools() -> list[Pool]:
    empirical = [json.loads(line) for line in EMPIRICAL.open()]
    gitlab = [json.loads(line) for line in GITLAB.open()]
    paired = list(csv.DictReader(PAIRED.open()))

    def rows(source: list[dict], pool: str) -> list[tuple[str, str, int]]:
        selected = [r for r in source if r["pool"] == pool and r["replicate"] == 0]
        if any(r["status"] != "ok" for r in selected):
            raise ValueError(f"{pool}: non-ok primary records")
        return [(r["arm_id"], r["task_id"], int(r["success"])) for r in selected]

    old_rows = [(r["arm_id"], r["task"], int(r["success"])) for r in paired]
    return [
        _matrix("G", rows(empirical, "G"), EMPIRICAL),
        _matrix("F", rows(empirical, "F"), EMPIRICAL),
        _matrix("GLK30", rows(gitlab, "GLK"), GITLAB),
        _matrix("GitLab17", old_rows, PAIRED, ("gitlab_oracle_operator",)),
        _matrix("GitLab18_cookbook", old_rows, PAIRED),
    ]


def _strata(tasks: tuple[str, ...]) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {}
    for j, task in enumerate(tasks):
        pieces = task.split("_")
        if len(pieces) < 2 or not pieces[1]:
            raise ValueError(f"unrecognized task stratum: {task}")
        result.setdefault(pieces[1][0], []).append(j)
    return {key: np.asarray(value) for key, value in sorted(result.items())}


def split_tasks(tasks: tuple[str, ...], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    groups = _strata(tasks)
    train: list[int] = []
    test: list[int] = []
    for indices in groups.values():
        shuffled = rng.permutation(indices)
        cut = len(indices) // 2
        train.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return np.asarray(train), np.asarray(test)


def _winner(scores: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.flatnonzero(scores == scores.max())
    return int(rng.choice(candidates))


def _pilot_signal(y: np.ndarray, train: np.ndarray, tasks: tuple[str, ...],
                  pilot_arms: np.ndarray, rng: np.random.Generator) -> float:
    # Six training tasks on each side, balanced over difficulty strata.
    train_set = set(map(int, train))
    groups = _strata(tasks)
    left: list[int] = []
    right: list[int] = []
    for indices in groups.values():
        eligible = np.asarray([j for j in indices if int(j) in train_set])
        if len(eligible) < 4:
            raise ValueError("pilot requires at least four training tasks per stratum")
        shuffled = rng.permutation(eligible)
        take = 2 if len(groups) == 3 else 3
        left.extend(shuffled[:take])
        right.extend(shuffled[take:2 * take])
    left_rate = y[np.ix_(pilot_arms, left)].mean(axis=1)
    right_rate = y[np.ix_(pilot_arms, right)].mean(axis=1)
    return float(np.cov(left_rate, right_rate, ddof=1)[0, 1])


def evaluate(pool: Pool, *, n_splits: int, seed: int, pilot_size: int = 8) -> list[dict]:
    if len(pool.arms) < pilot_size:
        raise ValueError("pilot_size exceeds number of arms")
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for split in range(n_splits):
        train, test = split_tasks(pool.tasks, rng)
        assert not set(train) & set(test)
        train_rate = pool.y[:, train].mean(axis=1)
        test_rate = pool.y[:, test].mean(axis=1)
        winner = _winner(train_rate, rng)
        pilot_arms = rng.choice(len(pool.arms), size=pilot_size, replace=False)
        pilot_winner = int(pilot_arms[_winner(train_rate[pilot_arms], rng)])
        pilot_signal = _pilot_signal(pool.y, train, pool.tasks, pilot_arms, rng)
        rows.append({
            "pool": pool.name,
            "split": split,
            "n_train_tasks": len(train),
            "n_test_tasks": len(test),
            "selected_arm": pool.arms[winner],
            "pilot_selected_arm": pool.arms[pilot_winner],
            "train_optimism": float(train_rate[winner] - train_rate.mean()),
            "heldout_gain": float(test_rate[winner] - test_rate.mean()),
            "pilot_heldout_gain": float(test_rate[pilot_winner] - test_rate.mean()),
            "incremental_gain": float(test_rate[winner] - test_rate[pilot_winner]),
            "pilot_covariance": pilot_signal,
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_pool: dict[str, list[dict]] = {}
    for row in rows:
        by_pool.setdefault(row["pool"], []).append(row)
    output = []
    for name, subset in by_pool.items():
        gain = np.asarray([r["heldout_gain"] for r in subset], dtype=float)
        pilot = np.asarray([r["pilot_covariance"] for r in subset], dtype=float)
        incremental = np.asarray([r["incremental_gain"] for r in subset], dtype=float)
        output.append({
            "pool": name,
            "n_splits": len(subset),
            "mean_train_optimism": np.mean([r["train_optimism"] for r in subset]),
            "mean_heldout_gain": gain.mean(),
            "median_heldout_gain": np.median(gain),
            "split_p05_heldout_gain": np.quantile(gain, .05),
            "split_p95_heldout_gain": np.quantile(gain, .95),
            "share_splits_gain_positive": np.mean(gain > 0),
            "mean_pilot_heldout_gain": np.mean([r["pilot_heldout_gain"] for r in subset]),
            "mean_incremental_gain": incremental.mean(),
            "mean_pilot_covariance": pilot.mean(),
            "share_pilot_covariance_positive": np.mean(pilot > 0),
            "winner_mode": Counter(r["selected_arm"] for r in subset).most_common(1)[0][0],
            "winner_mode_share": Counter(r["selected_arm"] for r in subset).most_common(1)[0][1] / len(subset),
        })
    return output


def leave_one_task_out(pool: Pool, *, n_splits: int, seed: int) -> list[dict]:
    """Sensitivity of the split-averaged gain to dropping each task once.

    The resulting range is a robustness check, not a confidence interval.
    """
    rows = []
    for omitted, task in enumerate(pool.tasks):
        keep = [j for j in range(len(pool.tasks)) if j != omitted]
        reduced = Pool(pool.name, pool.arms, tuple(pool.tasks[j] for j in keep),
                       pool.y[:, keep], pool.source, pool.excluded)
        trials = evaluate(reduced, n_splits=n_splits, seed=seed + omitted)
        rows.append({"pool": pool.name, "omitted_task": task,
                     "mean_heldout_gain": np.mean([r["heldout_gain"] for r in trials])})
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=int, default=2000)
    parser.add_argument("--leave-one-out-splits", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pools = load_pools()
    all_rows: list[dict] = []
    for offset, pool in enumerate(pools):
        all_rows.extend(evaluate(pool, n_splits=args.splits, seed=args.seed + offset))
    _write_csv(args.output / "task_transfer_splits.csv", all_rows)
    _write_csv(args.output / "task_transfer_summary.csv", summarize(all_rows))
    if args.leave_one_out_splits:
        jackknife_rows = []
        for offset, pool in enumerate(pools):
            jackknife_rows.extend(leave_one_task_out(
                pool, n_splits=args.leave_one_out_splits,
                seed=args.seed + 10_000 * (offset + 1)))
        _write_csv(args.output / "task_transfer_leave_one_task_out.csv", jackknife_rows)
    (args.output / "provenance.json").write_text(json.dumps({
        "analysis": "retrospective task-held-out prompt selection and pilot signal",
        "seed": args.seed,
        "splits_per_pool": args.splits,
        "leave_one_out_splits_per_task": args.leave_one_out_splits,
        "pilot_prompts": 8,
        "source_sha256": {str(p.source.relative_to(ROOT)): _sha256(p.source) for p in pools},
        "pools": [{"name": p.name, "arms": len(p.arms), "tasks": len(p.tasks),
                   "excluded": p.excluded} for p in pools],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
