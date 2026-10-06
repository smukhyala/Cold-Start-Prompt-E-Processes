"""Checks that the offline portfolio counterfactuals preserve their intended units."""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/retrospective"))

from agent_portfolio import (  # noqa: E402
    coverage,
    mean_random_team_coverage,
    nested_training_subsets,
    select_team,
)


def test_random_team_coverage_matches_exact_pair_and_triple_enumeration() -> None:
    y = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=np.int8)
    candidates = np.arange(len(y))
    tasks = np.arange(y.shape[1])
    for size in (1, 2, 3):
        exact = np.mean([coverage(y, pair, tasks).mean()
                         for pair in combinations(candidates, size)])
        assert np.isclose(mean_random_team_coverage(y, candidates, tasks, size), exact)


def test_greedy_coverage_selects_complementary_agent() -> None:
    y = np.asarray([
        [1, 1, 0, 0, 1, 1],  # best first agent
        [1, 1, 0, 0, 1, 0],  # strong, redundant
        [0, 0, 1, 1, 0, 0],  # weaker, complementary
    ], dtype=np.int8)
    train = np.arange(y.shape[1])
    candidates = np.arange(len(y))
    assert select_team(y, train, candidates, 2, "top_marginal") == (0, 1)
    assert select_team(y, train, candidates, 2, "greedy_coverage") == (0, 2)


def test_nested_task_audits_keep_strata_and_observed_budget() -> None:
    tasks = tuple(f"task_{tier}{j}" for tier in ("e", "m", "h") for j in range(1, 6))
    subsets = nested_training_subsets(tasks, np.arange(len(tasks)), np.random.default_rng(42))
    assert set(subsets) == {3, 6, 12}
    assert set(subsets[3]).issubset(subsets[6])
    assert set(subsets[6]).issubset(subsets[12])
    for n, indices in subsets.items():
        assert len(indices) == n
        assert {tier: sum(tasks[j].startswith(f"task_{tier}") for j in indices)
                for tier in ("e", "m", "h")} == dict.fromkeys(("e", "m", "h"), n // 3)
