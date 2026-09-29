"""Stage 0 -- free reanalyses (Pre-registration 10 section 6): per-cell heterogeneity on the existing
Gmail prompt pools (Pre-reg 9 snapshot) and the project's old GitLab paired run. No data collection.

Controller ruling (2026-09-28, task-3): `two_way_bootstrap` no longer exists. `analyze_cell` gets tau's
interval from `heterogeneity.tau_interval` (Graybill-Wang MLS): `tau`, `tau_lo`, `tau_hi` (two-sided 95%)
and `tau_upper_one_sided` (one-sided 95%). `n_boot` is dropped from `analyze_cell`'s signature (no
bootstrap needed for a single cell).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
import stage0  # noqa: E402


def test_analyze_cell_fields_and_interval():
    rng = np.random.default_rng(0)
    p = np.clip(0.6 + rng.normal(0, 0.08, 40)[:, None] + rng.normal(0, 0.3, 30)[None, :], 0.02, 0.98)
    Y = (rng.random(p.shape) < p).astype(float)
    tasks = [f"task_{'emh'[j % 3]}{j}" for j in range(30)]
    out = stage0.analyze_cell(Y, tasks, noise_var=0.1, seed=1)
    for k in ("tau", "tau_lo", "tau_hi", "tau_upper_one_sided", "r_sb", "split_half_p",
              "tau_discriminating", "upper_tail_mass", "interaction_var", "task_var"):
        assert k in out
    assert out["tau_lo"] <= out["tau"] <= out["tau_hi"]


def test_analyze_cell_tau_discriminating_is_nan_with_fewer_than_two_discriminating_tasks():
    """Controller ruling: tau on discriminating tasks is variance_components on the column-subset;
    if fewer than 2 discriminating tasks, report NaN (variance_components needs >= 2 columns)."""
    I, J = 10, 6
    Y = np.zeros((I, J))
    Y[:, 0] = 1.0  # exactly one task in [0.2, 0.8] band (mean 1.0 is NOT discriminating; use a mix)
    Y[: I // 2, 0] = 1.0
    Y[I // 2 :, 0] = 0.0  # column 0 mean 0.5 -> discriminating; every other column constant at 0 -> not
    tasks = [f"task_e{j}" for j in range(J)]
    out = stage0.analyze_cell(Y, tasks, seed=0)
    assert np.isnan(out["tau_discriminating"])


def test_gitlab_paired_matrix(tmp_path):
    rows = [{"arm_id": a, "task": f"task_e{j}", "timestep": j, "reward": 1.0, "success": int(j % 2 == 0),
             "runtime": 1, "cost": 0.01, "steps": 3, "log_path": ""} for a in ("x", "y", "oracle") for j in range(4)]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    Y, arms, tasks = stage0.gitlab_paired_matrix(path, exclude=("oracle",))
    assert Y.shape == (2, 4) and arms == ["x", "y"]


def test_gitlab_paired_matrix_no_exclusion_keeps_every_arm(tmp_path):
    rows = [{"arm_id": a, "task": f"task_e{j}", "timestep": j, "reward": 1.0, "success": int(j % 2 == 0),
             "runtime": 1, "cost": 0.01, "steps": 3, "log_path": ""} for a in ("x", "y", "oracle") for j in range(4)]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    Y, arms, tasks = stage0.gitlab_paired_matrix(path)
    assert Y.shape == (3, 4) and arms == ["oracle", "x", "y"]


def test_gitlab_paired_matrix_feeds_analyze_cell_with_no_noise_var(tmp_path):
    """The old GitLab data has one run per cell (no replicates): `noise_var=None`, interaction and
    noise reported jointly as ms_resid (controller ruling)."""
    rng = np.random.default_rng(2)
    rows = []
    for i, a in enumerate([f"arm{i}" for i in range(6)]):
        for j in range(8):
            rows.append({"arm_id": a, "task": f"task_e{j}", "timestep": j,
                         "reward": 1.0, "success": int(rng.random() < 0.5),
                         "runtime": 1, "cost": 0.01, "steps": 3, "log_path": ""})
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    Y, arms, tasks = stage0.gitlab_paired_matrix(path)
    out = stage0.analyze_cell(Y, tasks, noise_var=None, seed=0)
    assert out["interaction_var"] is None
