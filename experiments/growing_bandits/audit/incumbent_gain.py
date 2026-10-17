"""EXPLORATORY: held-out gain of the training winner over the incumbent ``baseline`` prompt.

For each pool, split the tasks in half 1000 times. Pick the arm with the best training-half success rate
(ties broken at random) and score winner minus incumbent on the held-out half. The incumbent is the
pool's ``baseline`` arm, or the Gmail ``anchor_baseline`` run on the same 30 tasks for G and F. GLK has
no incumbent run on its tasks and is skipped.

Unlike the winner-minus-pool-mean gain in ``cross_pool.py``, this gain does not reward avoiding bad prompts.

    ./.venv/bin/python experiments/growing_bandits/audit/incumbent_gain.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MATRICES = ROOT / "results" / "audit" / "matrices"
OUT = ROOT / "results" / "audit" / "e3" / "incumbent_gain.csv"

ORACLE = "gitlab_oracle_operator"  # synthetic positive control: hard-codes task-specific issue ids
EXPLORER = "explorer"

# (label, matrix file, arms dropped)
VARIANTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("G", "G", ()),
    ("F", "F", ()),
    ("gitlab18", "gitlab18", ()),
    ("gitlab17_no_oracle", "gitlab18", (ORACLE,)),
    ("gitlab16_no_oracle_no_explorer", "gitlab18", (ORACLE, EXPLORER)),
    ("gmail_paired12", "gmail_paired12", ()),
    ("shopping_paired12", "shopping_paired12", ()),
)


def primary_matrix(name: str, drop: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(MATRICES / f"{name}.csv")
    frame = frame[frame["replicate"] == 0]
    frame = frame[~frame["arm_id"].isin(list(drop))]
    return frame.pivot(index="arm_id", columns="task_id", values="success")


def split_arms_and_incumbent(name: str, Y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if name in ("G", "F"):
        anchor = pd.read_csv(MATRICES / "gmail_anchor.csv").set_index("task_id")["success"]
        return Y.to_numpy(dtype=float), anchor.reindex(Y.columns).to_numpy(dtype=float)
    return Y.drop(index="baseline").to_numpy(dtype=float), Y.loc["baseline"].to_numpy(dtype=float)


def held_out_gains(arms: np.ndarray, incumbent: np.ndarray, *, n_splits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_tasks = arms.shape[1]
    gains = np.empty(n_splits)
    for s in range(n_splits):
        order = rng.permutation(n_tasks)
        train = order[: n_tasks // 2]
        test = order[n_tasks // 2:]
        scores = arms[:, train].mean(axis=1)
        best = np.flatnonzero(scores == scores.max())
        winner = rng.choice(best)
        gains[s] = arms[winner, test].mean() - incumbent[test].mean()
    return gains


def run(*, n_splits: int = 1000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for label, name, drop in VARIANTS:
        Y = primary_matrix(name, drop)
        arms, incumbent = split_arms_and_incumbent(name, Y)
        gains = held_out_gains(arms, incumbent, n_splits=n_splits, seed=seed)
        rows.append({
            "label": "EXPLORATORY",
            "pool": label,
            "n_challengers": arms.shape[0],
            "n_tasks": arms.shape[1],
            "incumbent_rate": float(incumbent.mean()),
            "best_arm_rate": float(arms.mean(axis=1).max()),
            "gain_pp_mean": 100.0 * float(gains.mean()),
            "gain_pp_p05": 100.0 * float(np.percentile(gains, 5)),
            "gain_pp_p95": 100.0 * float(np.percentile(gains, 95)),
            "p_gain_positive": float(np.mean(gains > 0)),
            "n_splits": n_splits,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-splits", type=int, default=1000)
    args = parser.parse_args()
    table = run(n_splits=args.n_splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
