"""Variance components for the crossed prompt x task design, and model-free reliability.

Fix round 1 (2026-09-28): `two_way_bootstrap` (column-with-replacement resampling) is gone -- it
inflated tau2 by duplicating a task's residual into every row mean without reducing MS_resid's
degrees of freedom for the duplicate, and its nominal-95% coverage measured as low as 0.20-0.23 (see
task-2-report.md). It is replaced by the closed-form `tau_interval` (Graybill-Wang MLS) plus
`prompt_bootstrap` (rows only).
"""

from __future__ import annotations

import warnings

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


def _simulate_normal(I, J, tau, task_sd, resid_sd, seed):
    """Pure Normal two-way random-effects ANOVA model, no Bernoulli sampling and no clipping --
    used only to check `tau_interval`'s calibration in isolation from `_simulate`'s nonlinear link."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, tau, I)
    b = rng.normal(0, task_sd, J)
    e = rng.normal(0, resid_sd, (I, J))
    return 0.6 + a[:, None] + b[None, :] + e


def _realized_tau(p: np.ndarray) -> float:
    """The REALIZED true tau for one `_simulate` draw: SD (ddof=0) of the true per-prompt mean success
    probabilities `p.mean(axis=1)`. `_simulate` clips p to [0.01, 0.99], which attenuates the prompts'
    true spread below the nominal `tau` parameter passed in (verified: at tau=0.10, J=60, mean tau_hat
    was ~0.084, not ~0.10) -- coverage of the *nominal* input tau is therefore the wrong target for a
    correctly-calibrated interval; coverage of this realized quantity is the right one (controller
    ruling, fix round 1)."""
    return float(np.std(p.mean(axis=1), ddof=0))


# ---- point estimator (unchanged by fix round 1) --------------------------------------------------


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
    assert v["resid_df"] == (I - 1) * (J - 1)


def test_single_replicate_design():
    Y, _ = _simulate(seed=4)
    v = het.variance_components(Y)
    assert v["noise_var"] is None and v["interaction_var"] is None
    assert v["tau"] >= 0.0


# ---- success_matrix: imputation, reindexing, row counts, missing-data ceiling --------------------


def test_missing_cells_are_imputed_and_counted():
    rows = [{"pool": "P", "arm_id": f"p{i}", "task_id": f"t{j}", "replicate": 0, "status": "ok",
             "success": int((i + j) % 3 == 0)} for i in range(5) for j in range(6) if (i, j) != (2, 3)]
    Y, arms, tasks, n_imp, row_counts = het.success_matrix(pd.DataFrame(rows), "P")
    assert Y.shape == (5, 6) and n_imp == 1 and np.isfinite(Y).all()
    assert row_counts.tolist() == [6, 6, 5, 6, 6]


def test_success_matrix_reindexes_dropped_arms_as_missing_and_returns_row_counts():
    arms = [f"p{i}" for i in range(24)]
    tasks = [f"t{j}" for j in range(20)]
    rows = [{"pool": "P", "arm_id": a, "task_id": t, "replicate": 0, "status": "ok", "success": 1.0}
            for a in arms for t in tasks]
    outcomes = pd.DataFrame(rows)
    # "p24" has zero ok rows: a bare pivot would silently drop it; expected_arms forces it to appear
    # as an explicit (imputed) missing row instead.
    Y, out_arms, out_tasks, n_imp, row_counts = het.success_matrix(
        outcomes, "P", expected_arms=arms + ["p24"], expected_tasks=tasks)
    assert Y.shape == (25, 20)
    assert out_arms[-1] == "p24" and n_imp == 20
    assert row_counts.tolist() == [20] * 24 + [0]
    assert np.isfinite(Y).all()  # the fully-missing row still gets a finite (grand/column) fallback


def test_success_matrix_raises_when_missing_exceeds_five_percent():
    rows = [{"pool": "P", "arm_id": f"p{i}", "task_id": f"t{j}", "replicate": 0, "status": "ok",
             "success": 1.0} for i in range(5) for j in range(4)]
    outcomes = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="imputation ceiling"):
        het.success_matrix(outcomes, "P", expected_tasks=[f"t{j}" for j in range(6)])


def test_imputation_bias_is_small_after_the_row_count_correction():
    """5% missing, additive-imputed: mean tau2_raw with the n_imputed/row_counts correction should
    stay close to the full-data value (Important finding, fix round 1: uncorrected, MS_resid shrinks
    ~x0.948 and tau2 inflates ~+0.00052 at 5% missing)."""
    I, J = 50, 60
    arms = [f"p{i}" for i in range(I)]
    tasks = [f"t{j}" for j in range(J)]
    n_missing = int(0.05 * I * J) - 5  # comfortably under the 5% ceiling for every repeat
    diffs = []
    for s in range(60):
        Y, _ = _simulate(I=I, J=J, tau=0.05, seed=30_000 + s)
        full = het.variance_components(Y)["tau2"]
        rng = np.random.default_rng(40_000 + s)
        flat = rng.choice(I * J, size=n_missing, replace=False)
        mask = np.zeros(I * J, dtype=bool)
        mask[flat] = True
        mask = mask.reshape(I, J)
        rows = [{"pool": "P", "arm_id": arms[i], "task_id": tasks[j], "replicate": 0, "status": "ok",
                 "success": float(Y[i, j])}
                for i in range(I) for j in range(J) if not mask[i, j]]
        outcomes = pd.DataFrame(rows)
        Yimp, _, _, n_imp, row_counts = het.success_matrix(
            outcomes, "P", expected_arms=arms, expected_tasks=tasks)
        vc = het.variance_components(Yimp, n_imputed=n_imp, row_counts=row_counts)
        diffs.append(vc["tau2"] - full)
    mean_diff = float(np.mean(diffs))
    print(f"imputation-bias mean(tau2_raw_corrected - tau2_full) = {mean_diff:.6f}")
    assert abs(mean_diff) < 3e-4


# ---- tau_interval (Graybill-Wang MLS) -- replaces two_way_bootstrap ------------------------------


def test_tau_interval_coverage_is_correct_under_the_pure_normal_anova_model():
    """Isolates `tau_interval`'s own calibration from `_simulate`'s Bernoulli+clip nonlinearity: on
    data that actually is the additive-Normal model the MLS formula assumes, coverage should sit
    close to 95% uniformly, independent of tau or J."""
    for J in (30, 60):
        for tau in (0.0, 0.05, 0.10):
            n_rep = 200
            covered = 0
            for s in range(n_rep):
                Y = _simulate_normal(50, J, tau, task_sd=0.3, resid_sd=0.05, seed=7_000 + s)
                iv = het.tau_interval(Y)
                if iv["tau_lo"] <= tau <= iv["tau_hi"]:
                    covered += 1
            coverage = covered / n_rep
            print(f"pure-normal coverage J={J} tau={tau}: {coverage:.3f}")
            assert 0.90 <= coverage <= 1.0


@pytest.mark.parametrize("J,tau_nominal", [
    (30, 0.00), (30, 0.05), (30, 0.10),
    (60, 0.00), (60, 0.05), (60, 0.10),
])
def test_tau_interval_two_sided_coverage_of_the_realized_tau(J, tau_nominal):
    """Coverage against the REALIZED true tau (controller ruling, fix round 1): `_simulate` clips p to
    [0.01, 0.99], attenuating the prompts' true spread below the nominal `tau` parameter, so the nominal
    parameter is the wrong coverage target (it measured 0.78-0.86 there at tau=0.10, both below the
    ruled floor). `_realized_tau` -- SD (ddof=0) of `p.mean(axis=1)`, the true per-prompt mean success
    probabilities including the clip -- is the correct target.

    Band is [0.88, 0.998] at every setting (controller ruling, fix round 1 second pass): J=60,
    tau_nominal=0.10 measures ~0.996 here (n_rep=2000) and pooled across 5 independent, non-overlapping
    seed chunks (14,600 trials total) gives 0.9953 (95% CI [0.9941, 0.9964]) -- marginally conservative,
    consistent with known MLS behavior; the 0.998 upper edge exists only to catch gross conservatism, not
    this ~0.5pt excess.
    """
    n_rep = 2000
    covered = 0
    for s in range(n_rep):
        Y, p = _simulate(I=50, J=J, tau=tau_nominal, seed=10_000 + s)
        iv = het.tau_interval(Y)
        rt = _realized_tau(p)
        if iv["tau_lo"] <= rt <= iv["tau_hi"]:
            covered += 1
    coverage = covered / n_rep
    print(f"tau_interval coverage(realized tau) I=50 J={J} tau_nominal={tau_nominal}: {coverage:.4f}")
    assert 0.88 <= coverage <= 0.998


def test_tau_interval_one_sided_upper_bound_is_tight_under_the_null():
    n_rep = 300
    uppers = [het.tau_interval(_simulate(I=50, J=60, tau=0.0, seed=10_000 + s)[0])["tau_upper_one_sided"]
              for s in range(n_rep)]
    mean_upper = float(np.mean(uppers))
    print(f"mean one-sided 95% upper bound at tau=0, J=60: {mean_upper:.4f}")
    assert mean_upper < 0.035


# ---- prompt_bootstrap (rows only) -- replaces two_way_bootstrap -----------------------------------


def test_prompt_bootstrap_brackets_the_truth():
    Y, p = _simulate(tau=0.06, seed=5)
    draws = het.prompt_bootstrap(Y, lambda m: het.variance_components(m)["tau"], n_boot=300, seed=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert lo < _realized_tau(p) < hi


def test_prompt_bootstrap_percentile_coverage_of_the_realized_tau():
    """Coverage against the realized tau (controller ruling, fix round 1; see
    test_tau_interval_two_sided_coverage_of_the_realized_tau); required floor unchanged at >= 0.75."""
    n_rep = 150
    covered = 0
    for s in range(n_rep):
        Y, p = _simulate(I=50, J=60, tau=0.05, seed=20_000 + s)
        rt = _realized_tau(p)
        draws = het.prompt_bootstrap(Y, lambda m: het.variance_components(m)["tau"], n_boot=300, seed=s)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        if lo <= rt <= hi:
            covered += 1
    coverage = covered / n_rep
    print(f"prompt_bootstrap coverage(realized tau) at tau_nominal=0.05: {coverage:.4f}")
    assert coverage >= 0.75


# ---- split_half: stratum regex, alternating odd task, Spearman-Brown, zero variance ---------------


def test_halves_rejects_task_ids_outside_the_stratification_pattern():
    with pytest.raises(ValueError, match="stratification pattern"):
        het._halves(["task_e0", "bad_id"], seed=0)


def test_halves_alternates_which_half_gets_the_odd_task_per_stratum():
    ids = [f"task_e{i}" for i in range(3)] + [f"task_h{i}" for i in range(3)] + [f"task_m{i}" for i in range(3)]
    a, b = het._halves(ids, seed=0)
    # 3 odd-sized (3-task) strata processed in sorted order e, h, m: alternating the extra task
    # (a, b, a) gives sizes (5, 4) instead of the old always-favor-b (3, 6).
    assert (len(a), len(b)) == (5, 4)


def test_split_half_zero_variance_half_returns_p_one_without_warning():
    Y = np.full((5, 6), 0.5)
    tasks = [f"task_e{i}" for i in range(6)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = het.split_half(Y, tasks, seed=0, n_perm=200)
    assert result["r"] == 0.0
    assert result["p_value"] == 1.0
    assert result["n_half_a"] == 3 and result["n_half_b"] == 3
    assert np.isnan(result["r_sb"])


def test_split_half_reports_nan_r_sb_for_nonpositive_r():
    tasks = [f"task_e{i}" for i in range(6)]
    a_idx, b_idx = het._halves(tasks, seed=0)
    row_profile = np.array([0.0, 1.0, 2.0, 3.0])
    Y = np.zeros((4, 6))
    for col in a_idx:
        Y[:, col] = row_profile
    for col in b_idx:
        Y[:, col] = row_profile[::-1]
    result = het.split_half(Y, tasks, seed=0, n_perm=200)
    assert result["r"] < 0.0
    assert np.isnan(result["r_sb"])


def test_split_half_detects_real_differences_and_not_noise():
    Y, _ = _simulate(tau=0.10, seed=6)
    tasks = [f"task_{'emh'[j % 3]}{j}" for j in range(Y.shape[1])]
    real = het.split_half(Y, tasks, seed=0, n_perm=2000)
    assert real["r_sb"] > 0.5 and real["p_value"] < 0.01
    Yf, _ = _simulate(tau=0.0, seed=7)
    flat = het.split_half(Yf, tasks, seed=0, n_perm=2000)
    assert flat["p_value"] > 0.01


# ---- discriminating_tasks, prompt_effects (unchanged by fix round 1) ------------------------------


def test_discriminating_tasks_mask():
    Y = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1]], dtype=float)
    assert het.discriminating_tasks(Y).tolist() == [False, False, True, True]


def test_prompt_effects_shrink_toward_zero():
    Y, _ = _simulate(tau=0.0, seed=8)
    raw = Y.mean(axis=1) - Y.mean()
    assert np.abs(het.prompt_effects(Y)).sum() <= np.abs(raw).sum() + 1e-12


# ---- noise_from_pairs: now a thin wrapper over empirical.within_cell_variance ---------------------


def test_noise_from_pairs_matches_within_cell_variance():
    from cold_start.growing.empirical import within_cell_variance

    rows = [
        {"pool": "P", "arm_id": "a", "task_id": "t1", "replicate": 0, "status": "ok", "success": 1},
        {"pool": "P", "arm_id": "a", "task_id": "t1", "replicate": 1, "status": "ok", "success": 0},
        {"pool": "P", "arm_id": "a", "task_id": "t2", "replicate": 0, "status": "ok", "success": 1},
        {"pool": "P", "arm_id": "a", "task_id": "t2", "replicate": 1, "status": "ok", "success": 1},
    ]
    outcomes = pd.DataFrame(rows)
    v, n = het.noise_from_pairs(outcomes, "P")
    v_ref, n_ref = within_cell_variance(outcomes, ["P"])
    assert (v, n) == (v_ref, n_ref)


def test_noise_from_pairs_returns_nan_without_replicate_pairs():
    rows = [{"pool": "P", "arm_id": "a", "task_id": "t1", "replicate": 0, "status": "ok", "success": 1}]
    v, n = het.noise_from_pairs(pd.DataFrame(rows), "P")
    assert np.isnan(v) and n == 0
