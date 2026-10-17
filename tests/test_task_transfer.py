"""Checks for leakage and data integrity in the retrospective task analysis."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (Path(__file__).resolve().parents[1] / "experiments" / "growing_bandits"
               / "retrospective" / "task_transfer.py")
SPEC = importlib.util.spec_from_file_location("task_transfer", MODULE_PATH)
assert SPEC and SPEC.loader
task_transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = task_transfer
SPEC.loader.exec_module(task_transfer)


def test_matrix_requires_one_observation_per_arm_task() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        task_transfer._matrix("x", [("a", "task_e1", 1), ("a", "task_e1", 0)],
                              Path("dummy"))
    with pytest.raises(ValueError, match="incomplete"):
        task_transfer._matrix("x", [("a", "task_e1", 1), ("b", "task_e2", 0)],
                              Path("dummy"))


def test_split_disjoint_and_stratified() -> None:
    tasks = tuple(f"task_{tier}{i}" for tier in "emh" for i in range(10))
    train, test = task_transfer.split_tasks(tasks, np.random.default_rng(5))
    assert set(train).isdisjoint(test)
    assert len(train) == len(test) == 15
    assert all(sum(tasks[j].startswith(f"task_{tier}") for j in train) == 5
               for tier in "emh")


def test_stable_best_arm_transfers() -> None:
    tasks = tuple(f"task_{tier}{i}" for tier in "emh" for i in range(10))
    y = np.zeros((8, 30), dtype=np.int8)
    y[0, :] = 1
    pool = task_transfer.Pool("synthetic", tuple(f"a{i}" for i in range(8)), tasks,
                              y, Path("dummy"))
    rows = task_transfer.evaluate(pool, n_splits=20, seed=7)
    assert all(row["selected_arm"] == "a0" for row in rows)
    assert all(row["heldout_gain"] == 0.875 for row in rows)
