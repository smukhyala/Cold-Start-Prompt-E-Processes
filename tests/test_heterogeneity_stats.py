"""Variance components for the crossed prompt x task design, and model-free reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cold_start.growing import heterogeneity as het


def _simulate(I=50, J=60, tau=0.05, task_sd=0.3, inter_sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, tau, I)
    b = rng.normal(0, task_sd, J)
    ab = rng.normal(0, inter_sd, (I, J))
    p = np.clip(0.6 + a[:, None] + b[None, :] + ab, 0.01, 0.99)
    return (rng.random((I, J)) < p).astype(float), p


def test_tau_is_recovered_on_average():
    taus = [het.variance_components(_simulate(seed=s)[0])["tau"] for s in range(40)]
    assert np.mean(taus) == pytest.approx(0.05, abs=0.01)


def test_flat_pool_gives_tau_near_zero_and_truncation_is_reported():
    vc = [het.variance_components(_simulate(tau=0.0, seed=s)[0]) for s in range(40)]
    assert np.mean([v["tau"] for v in vc]) < 0.02
    assert any(v["tau2_truncated"] for v in vc)


def test_components_identities():
    Y, _ = _simulate(seed=3)
    v = het.variance_components(Y, noise_var=0.1)
    J, I = Y.shape[1], Y.shape[0]
    assert v["tau2"] == pytest.approx(max((v["ms_prompt"] - v["ms_resid"]) / J, 0.0))
    assert v["task_var"] == pytest.approx(max((v["ms_task"] - v["ms_resid"]) / I, 0.0))
    assert v["interaction_var"] == pytest.approx(max(v["ms_resid"] - 0.1, 0.0))


def test_single_replicate_design():
    Y, _ = _simulate(seed=4)
    v = het.variance_components(Y)
    assert v["noise_var"] is None and v["interaction_var"] is None
    assert v["tau"] >= 0.0


def test_missing_cells_are_imputed_and_counted():
    rows = [{"pool": "P", "arm_id": f"p{i}", "task_id": f"t{j}", "replicate": 0, "status": "ok",
             "success": int((i + j) % 3 == 0)} for i in range(5) for j in range(6) if (i, j) != (2, 3)]
    Y, arms, tasks, n_imp = het.success_matrix(pd.DataFrame(rows), "P")
    assert Y.shape == (5, 6) and n_imp == 1 and np.isfinite(Y).all()


def test_two_way_bootstrap_brackets_the_truth():
    Y, _ = _simulate(tau=0.06, seed=5)
    draws = het.two_way_bootstrap(Y, lambda m: het.variance_components(m)["tau"], n_boot=300, seed=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert lo < 0.06 < hi


def test_split_half_detects_real_differences_and_not_noise():
    Y, _ = _simulate(tau=0.10, seed=6)
    tasks = [f"task_{'emh'[j % 3]}{j}" for j in range(Y.shape[1])]
    real = het.split_half(Y, tasks, seed=0, n_perm=2000)
    assert real["r_sb"] > 0.5 and real["p_value"] < 0.01
    Yf, _ = _simulate(tau=0.0, seed=7)
    flat = het.split_half(Yf, tasks, seed=0, n_perm=2000)
    assert flat["p_value"] > 0.01


def test_discriminating_tasks_mask():
    Y = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1]], dtype=float)
    assert het.discriminating_tasks(Y).tolist() == [False, False, True, True]


def test_prompt_effects_shrink_toward_zero():
    Y, _ = _simulate(tau=0.0, seed=8)
    raw = Y.mean(axis=1) - Y.mean()
    assert np.abs(het.prompt_effects(Y)).sum() <= np.abs(raw).sum() + 1e-12
