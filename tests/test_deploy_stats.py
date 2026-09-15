"""Tests for the paired / stratified / clustered bootstrap helpers on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cold_start.growing.deploy.stats import (
    cluster_bootstrap_over_envs,
    paired_bootstrap,
    spearman_with_ci,
    stratified_pooled,
)

# ---- paired bootstrap ------------------------------------------------------------------


def test_paired_bootstrap_ci_covers_the_true_mean():
    rng = np.random.default_rng(0)
    true_mean = 0.02
    diff = rng.normal(true_mean, 0.05, size=400)
    out = paired_bootstrap(diff, n_boot=4000, seed=1)
    assert out["lo"] <= true_mean <= out["hi"]
    assert out["lo"] <= out["mean"] <= out["hi"]
    assert out["mean"] == pytest.approx(diff.mean())
    assert out["se_paired"] == pytest.approx(diff.std(ddof=1) / np.sqrt(diff.size))
    assert out["n"] == 400 and out["n_boot"] == 4000
    # A 95% percentile interval is about +/- 2 SE wide.
    assert (out["hi"] - out["lo"]) == pytest.approx(2 * 1.96 * out["se_paired"], rel=0.15)


def test_paired_bootstrap_coverage_is_near_nominal():
    """Across many small samples the 95% interval should cover the truth ~95% of the time."""
    rng = np.random.default_rng(7)
    covered = 0
    trials = 200
    for i in range(trials):
        diff = rng.normal(0.0, 1.0, size=40)
        out = paired_bootstrap(diff, n_boot=1000, seed=i)
        covered += int(out["lo"] <= 0.0 <= out["hi"])
    assert 0.89 <= covered / trials <= 0.99, covered / trials


def test_paired_bootstrap_win_rate_and_crn_gain():
    rng = np.random.default_rng(3)
    common = rng.normal(0.7, 0.15, size=2000)  # the episode's luck, shared under CRN
    a = common + rng.normal(0.02, 0.01, size=2000)
    b = common + rng.normal(0.00, 0.01, size=2000)
    out = paired_bootstrap(a - b, n_boot=2000, a=a, b=b)
    assert out["win_rate"] > 0.9
    assert out["frac_positive"] + out["frac_zero"] + out["frac_negative"] == pytest.approx(1.0)
    # Pairing removes the shared variance: the unpaired-equivalent SE is far larger.
    assert out["se_unpaired_equiv"] > 5 * out["se_paired"]
    assert np.isnan(paired_bootstrap(a - b, n_boot=10)["se_unpaired_equiv"])

    # Identical policies: every difference is a tie, scored as half a win.
    tied = paired_bootstrap(np.zeros(50), n_boot=100)
    assert tied["win_rate"] == 0.5 and tied["frac_zero"] == 1.0
    assert tied["mean"] == 0.0 and tied["lo"] == 0.0 and tied["hi"] == 0.0


def test_paired_bootstrap_rejects_misaligned_marginals():
    with pytest.raises(ValueError, match="episode-aligned"):
        paired_bootstrap(np.zeros(5), n_boot=10, a=np.zeros(4), b=np.zeros(5))
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(0))


# ---- stratified pooling -------------------------------------------------------------------


def test_stratified_pooling_of_equal_cells_equals_grand_mean():
    rng = np.random.default_rng(11)
    cells = {f"cell{i}": rng.normal(0.01 * i, 0.02, size=300) for i in range(6)}
    out = stratified_pooled(cells, n_boot=2000, seed=0)
    grand = np.concatenate(list(cells.values())).mean()
    assert out["mean"] == pytest.approx(grand, abs=1e-12)
    assert out["lo"] <= out["mean"] <= out["hi"]
    assert out["n_cells"] == 6
    assert out["cell_means"] == {k: pytest.approx(v.mean()) for k, v in cells.items()}
    # Stratified resampling is tighter than an unstratified one would be: its SE is
    # the root-mean-square of the cell SEs over sqrt(n_cells), not the pooled spread.
    expected_se = np.sqrt(sum(v.var(ddof=1) / v.size for v in cells.values())) / 6
    assert out["se"] == pytest.approx(expected_se, rel=0.15)


def test_stratified_pooling_weights_cells_equally_not_by_size():
    big = np.full(1000, 1.0)
    small = np.full(10, 0.0)
    out = stratified_pooled({"big": big, "small": small}, n_boot=50)
    assert out["mean"] == pytest.approx(0.5)  # not 1000/1010
    assert out["lo"] == pytest.approx(0.5) and out["hi"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        stratified_pooled({})


# ---- cluster bootstrap over environments -------------------------------------------------


def test_cluster_bootstrap_over_envs_resamples_environments():
    rng = np.random.default_rng(5)
    envs = [f"env{i}" for i in range(8)]
    horizons = [50, 100, 200, 500, 1000]
    env_effect = dict(zip(envs, rng.normal(0.0, 0.05, size=len(envs)), strict=True))
    rows = []
    for env in envs:
        for horizon in horizons:
            rows.append({"env": env, "T": horizon, "d": env_effect[env] + rng.normal(0, 0.002)})
    df = pd.DataFrame(rows)
    out = cluster_bootstrap_over_envs(df, "env", "d", n_boot=4000, seed=0)
    assert out["mean"] == pytest.approx(df["d"].mean())
    assert out["n_envs"] == 8 and out["n_cells"] == 40
    assert out["lo"] <= out["mean"] <= out["hi"]
    # Between-environment spread dominates, so the cluster SE is close to the SE of
    # the environment means -- a within-cell bootstrap would be ~25x too tight.
    env_means = df.groupby("env")["d"].mean()
    assert out["se"] == pytest.approx(env_means.std(ddof=1) / np.sqrt(8), rel=0.2)


def test_cluster_bootstrap_single_env_is_degenerate_and_missing_column_raises():
    df = pd.DataFrame({"env": ["only"] * 5, "d": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = cluster_bootstrap_over_envs(df, "env", "d", n_boot=100)
    assert out["mean"] == out["lo"] == out["hi"] == 3.0
    with pytest.raises(KeyError):
        cluster_bootstrap_over_envs(df, "environment", "d", n_boot=10)


# ---- Spearman -----------------------------------------------------------------------------


def test_spearman_of_monotone_relation_is_one():
    x = np.linspace(0.0, 1.0, 25)
    out = spearman_with_ci(x, np.exp(3.0 * x), n_boot=500, seed=0)
    assert out["rho"] == 1.0
    assert out["lo"] == 1.0 and out["hi"] == 1.0
    out_neg = spearman_with_ci(x, -np.sqrt(x + 0.1), n_boot=500, seed=0)
    assert out_neg["rho"] == -1.0
    assert out["n"] == 25


def test_spearman_ci_brackets_a_noisy_relation_and_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(2)
    x = rng.normal(size=30)
    y = 0.6 * x + rng.normal(size=30)
    out = spearman_with_ci(x, y, n_boot=2000, seed=0)
    assert out["rho"] == pytest.approx(spearmanr(x, y).statistic)
    assert out["lo"] < out["rho"] < out["hi"]
    assert -1.0 <= out["lo"] and out["hi"] <= 1.0
    with pytest.raises(ValueError):
        spearman_with_ci([1.0], [2.0])
    with pytest.raises(ValueError):
        spearman_with_ci([1.0, 2.0, 3.0], [1.0, 2.0])
