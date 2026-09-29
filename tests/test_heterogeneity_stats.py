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


def test_success_matrix_raises_for_a_fully_missing_arm():
    """N1 (fix round 2): a reindexed arm with zero `ok` rows is under the 5% overall missing-cell
    ceiling here (1/25 arms, 20/500 = 4% of cells) but must still be rejected -- its `row_counts` entry
    would be 0, and downstream `mean(J / row_counts)` (`variance_components`, `tau_interval`) would
    divide by zero, silently producing `tau2=0`/`tau2_truncated=True`/`tau_hi=NaN` instead of failing
    loudly. The error must name the missing arm."""
    arms = [f"p{i}" for i in range(24)]
    tasks = [f"t{j}" for j in range(20)]
    rows = [{"pool": "P", "arm_id": a, "task_id": t, "replicate": 0, "status": "ok", "success": 1.0}
            for a in arms for t in tasks]
    outcomes = pd.DataFrame(rows)
    with pytest.raises(ValueError, match=r"arm\(s\) \['p24'\]"):
        het.success_matrix(outcomes, "P", expected_arms=arms + ["p24"], expected_tasks=tasks)


def test_success_matrix_raises_for_a_fully_missing_task():
    """N1 (fix round 2): same failure mode as the fully-missing-arm case, but for a task with zero
    `ok` rows (a fully-missing column)."""
    arms = [f"p{i}" for i in range(20)]
    tasks = [f"t{j}" for j in range(24)]
    rows = [{"pool": "P", "arm_id": a, "task_id": t, "replicate": 0, "status": "ok", "success": 1.0}
            for a in arms for t in tasks]
    outcomes = pd.DataFrame(rows)
    with pytest.raises(ValueError, match=r"task\(s\) \['t24'\]"):
        het.success_matrix(outcomes, "P", expected_arms=arms, expected_tasks=tasks + ["t24"])


def test_success_matrix_raises_when_missing_exceeds_five_percent():
    # Every row and every column keeps at least one observed cell (the missing diagonal cell (i, i)),
    # so this exercises the overall-percentage ceiling, not the fully-missing-row/column check above.
    rows = [{"pool": "P", "arm_id": f"p{i}", "task_id": f"t{j}", "replicate": 0, "status": "ok",
             "success": 1.0} for i in range(10) for j in range(10) if i != j]
    outcomes = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="imputation ceiling"):
        het.success_matrix(outcomes, "P")


def test_variance_components_and_tau_interval_are_finite_with_partial_missingness_after_reindex():
    """N1 (fix round 2), positive case: a reindexed matrix with scattered (not fully-row/column)
    missingness under the 5% ceiling should succeed and feed finite outputs through both
    `variance_components` and `tau_interval` -- confirming the fully-missing-row/column guard doesn't
    also reject ordinary partial missingness."""
    I, J = 30, 20
    arms = [f"p{i}" for i in range(I)]
    tasks = [f"t{j}" for j in range(J)]
    Y_full, _ = _simulate(I=I, J=J, tau=0.05, seed=0)
    # one missing cell per even-indexed row, scattered across columns: no row or column ends up fully
    # missing (15/600 = 2.5%, comfortably under the 5% ceiling).
    missing_cells = {(i, (i * 3 + 1) % J) for i in range(0, I, 2)}
    rows = [{"pool": "P", "arm_id": arms[i], "task_id": tasks[j], "replicate": 0, "status": "ok",
             "success": float(Y_full[i, j])}
            for i in range(I) for j in range(J) if (i, j) not in missing_cells]
    outcomes = pd.DataFrame(rows)
    Y, out_arms, out_tasks, n_imp, row_counts = het.success_matrix(
        outcomes, "P", expected_arms=arms, expected_tasks=tasks)
    assert Y.shape == (I, J)
    assert n_imp == len(missing_cells)
    assert np.isfinite(Y).all()
    assert (row_counts > 0).all()
    vc = het.variance_components(Y, n_imputed=n_imp, row_counts=row_counts)
    iv = het.tau_interval(Y, n_imputed=n_imp, row_counts=row_counts)
    assert np.isfinite(vc["tau2"]) and np.isfinite(vc["ms_resid"])
    assert np.isfinite(iv["tau_lo"]) and np.isfinite(iv["tau_hi"]) and np.isfinite(iv["tau_upper_one_sided"])


def test_imputation_bias_is_small_after_the_row_count_correction():
    """5% missing, additive-imputed: compares the mean bias of the CORRECTED tau2 estimate
    (`variance_components(..., n_imputed=..., row_counts=...)`) against the mean bias of the
    UNCORRECTED estimate (`variance_components(...)` on the same imputed matrices, no correction
    applied) relative to the full-data tau2 -- both computed on the identical draws, so a regression
    that silently drops the correction is caught rather than passing because the uncorrected bias
    happens to already be small at these settings (N2, fix round 2: the old version of this test
    compared only the corrected bias to a fixed absolute bound of 3e-4, which the uncorrected estimate
    also satisfies at +0.00023 here -- so the test could not tell whether the correction was doing
    anything). Reported SE at 300 reps is ~1.3e-5 (fix round 2 measurement), so 300 reps keeps both the
    absolute-bound and the relative-improvement assertions stable while keeping the file well under the
    ~60s budget."""
    I, J = 50, 60
    arms = [f"p{i}" for i in range(I)]
    tasks = [f"t{j}" for j in range(J)]
    n_missing = int(0.05 * I * J) - 5  # comfortably under the 5% ceiling for every repeat
    corrected_diffs = []
    uncorrected_diffs = []
    for s in range(300):
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
        corrected = het.variance_components(Yimp, n_imputed=n_imp, row_counts=row_counts)["tau2"]
        uncorrected = het.variance_components(Yimp)["tau2"]
        corrected_diffs.append(corrected - full)
        uncorrected_diffs.append(uncorrected - full)
    mean_corrected = float(np.mean(corrected_diffs))
    mean_uncorrected = float(np.mean(uncorrected_diffs))
    print(f"imputation-bias mean corrected={mean_corrected:.6f} uncorrected={mean_uncorrected:.6f}")
    assert abs(mean_corrected) < 1.5e-4
    assert abs(mean_corrected) < 0.5 * abs(mean_uncorrected)


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


# ---- tau_set_interval (spec amendment 1d0e7b4: tau_main vs tau_set) -------------------------------


def _simulate_with_replicate_pair(I, J, tau, task_sd, inter_sd, seed):
    """Two-way random-effects model with a nonzero prompt x task interaction, plus a second
    independent Bernoulli replicate per cell (same true `p`) to estimate execution noise the way
    `noise_from_pairs`/`within_cell_variance` do: `v = E[(x0 - x1)^2] / 2`, df = number of pairs.

    Returns `(Y0, p, noise_var_hat, noise_df)`. `Y0` is the cell's replicate-0 matrix (what
    `tau_set_interval`'s `Y` argument is); `p` is the true success-probability surface, whose row
    means (`p.mean(axis=1)`, i.e. `_realized_tau(p)`) already include the interaction averaged over
    this task set -- exactly tau_set's realized target, unlike tau_main's `Var(a_i)` alone.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, tau, I)
    b = rng.normal(0, task_sd, J)
    ab = rng.normal(0, inter_sd, (I, J))
    p = np.clip(0.6 + a[:, None] + b[None, :] + ab, 0.01, 0.99)
    Y0 = (rng.random((I, J)) < p).astype(float)
    Y1 = (rng.random((I, J)) < p).astype(float)
    noise_var_hat = float(np.mean((Y0 - Y1) ** 2) / 2.0)
    return Y0, p, noise_var_hat, I * J


@pytest.mark.parametrize("J,tau_nominal", [(30, 0.05), (60, 0.10)])
def test_tau_set_interval_coverage_of_the_realized_task_set_spread(J, tau_nominal):
    """tau_set targets SD_i(p.mean(axis=1)) -- the realized spread of prompts' true rates on THIS
    task set, main effect plus the interaction averaged over it -- exactly what `_realized_tau`
    already computes (its own docstring flags this as tau_set's target, not tau_main's, a minor
    deferred at Task 2: '_realized_tau includes interaction row-mean variance (finite-sample target)
    -- mildly inflates measured coverage' of `tau_interval`/tau_main; tau_set_interval is the
    estimator that quantity is actually the coverage target for)."""
    n_rep = 400
    covered = 0
    for s in range(n_rep):
        Y0, p, noise_var_hat, noise_df = _simulate_with_replicate_pair(
            I=50, J=J, tau=tau_nominal, task_sd=0.3, inter_sd=0.05, seed=50_000 + s)
        iv = het.tau_set_interval(Y0, noise_var_hat, noise_df)
        rt = _realized_tau(p)
        if iv["tau_set_lo"] <= rt <= iv["tau_set_hi"]:
            covered += 1
    coverage = covered / n_rep
    print(f"tau_set_interval coverage(realized task-set spread) J={J} tau_nominal={tau_nominal}: {coverage:.4f}")
    assert 0.88 <= coverage <= 0.998


def test_tau_set_is_at_least_tau_main_on_average_when_interaction_is_present():
    n_rep = 200
    tau_set_hats = []
    tau_main_hats = []
    for s in range(n_rep):
        Y0, _, noise_var_hat, noise_df = _simulate_with_replicate_pair(
            I=50, J=30, tau=0.05, task_sd=0.3, inter_sd=0.08, seed=60_000 + s)
        tau_set_hats.append(het.tau_set_interval(Y0, noise_var_hat, noise_df)["tau_set"])
        tau_main_hats.append(het.variance_components(Y0)["tau"])
    mean_set, mean_main = float(np.mean(tau_set_hats)), float(np.mean(tau_main_hats))
    print(f"mean tau_set={mean_set:.4f} mean tau_main={mean_main:.4f}")
    assert mean_set >= mean_main


def test_tau_set_interval_raises_without_a_finite_noise_var():
    Y, _ = _simulate(seed=0)
    with pytest.raises(ValueError, match="noise_var"):
        het.tau_set_interval(Y, None, 100)
    with pytest.raises(ValueError, match="noise_var"):
        het.tau_set_interval(Y, float("nan"), 100)


def test_tau_set_interval_raises_without_at_least_one_noise_df():
    Y, _ = _simulate(seed=0)
    with pytest.raises(ValueError, match="noise_df"):
        het.tau_set_interval(Y, 0.05, 0)


# ---- prompt_bootstrap (rows only) -- replaces two_way_bootstrap -----------------------------------


def test_prompt_bootstrap_brackets_the_truth():
    Y, p = _simulate(tau=0.06, seed=5)
    draws = het.prompt_bootstrap(Y, lambda m: het.variance_components(m)["tau"], n_boot=300, seed=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert lo < _realized_tau(p) < hi


def test_prompt_bootstrap_passes_resampled_row_counts_to_stat():
    """Gap (fix round 2): when `row_counts` is supplied, `stat` must receive the row counts resampled
    with the SAME row indices as `Y`, not the original (pre-resample) counts -- otherwise a `stat`
    closure using `variance_components`'s `n_imputed`/`row_counts` correction would silently evaluate
    against the wrong counts (or the correction would be unreachable at all) inside a bootstrap
    contrast."""
    Y, _ = _simulate(I=10, J=8, seed=2)
    row_counts = np.arange(1, 11, dtype=float)  # distinct per-row values so misalignment is detectable
    seen: list[np.ndarray] = []

    def stat(Y_sub, rc_sub):
        seen.append(np.array(rc_sub))
        return 0.0

    rng = np.random.default_rng(3)
    R = rng.integers(0, 10, (5, 10))
    het.prompt_bootstrap(Y, stat, rows=R, row_counts=row_counts)
    assert len(seen) == 5
    for b in range(5):
        assert np.array_equal(seen[b], row_counts[R[b]])


def test_prompt_bootstrap_without_row_counts_calls_stat_with_one_argument():
    Y, _ = _simulate(I=10, J=8, seed=2)
    calls: list[tuple] = []

    def stat(*args):
        calls.append(args)
        return 0.0

    het.prompt_bootstrap(Y, stat, n_boot=3, seed=1)
    assert all(len(c) == 1 for c in calls)


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
