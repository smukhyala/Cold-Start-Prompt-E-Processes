"""Prompt heterogeneity in a crossed prompt x task design (Pre-registration 10, section 6).

    y_ij = mu + a_i + b_j + (ab)_ij + e_ij

With one replicate-0 observation per (prompt, task) cell, the two-way ANOVA mean squares give
unbiased moment estimates: E[MS_prompt] = s2_resid + J tau^2, E[MS_task] = s2_resid + I s2_task,
E[MS_resid] = s2_resid = s2_ab + s2_e. tau therefore needs no noise model and no NPMLE; the replicate
pairs only split s2_resid into interaction and execution noise. Negative moment estimates are truncated at
zero and flagged.

Fix round 1 (2026-09-28 review): `two_way_bootstrap` resampled tasks (columns) with replacement, which
duplicates a task's residual into the row means used for MS_prompt without reducing MS_resid's degrees
of freedom for those duplicates -- the interval it produced was not a valid percentile CI for tau
(measured 95%-nominal coverage as low as 0.20-0.23 at tau=0; see task-2-report.md fix round 1). It has
been removed and replaced by `prompt_bootstrap` (rows only) plus the closed-form `tau_interval` (MLS).

Fix round 2 (2026-09-28 review): `success_matrix` now raises when any single arm or task has zero
observed cells (not just when the overall missing-cell percentage exceeds the 5% ceiling), since a
fully-missing row/column produced a degenerate `row_counts` entry of 0 that downstream `J / row_counts`
consumers would divide by zero on. `prompt_bootstrap` gained an optional `row_counts` parameter so the
imputation row-count correction is resampled consistently with `Y` inside a bootstrap contrast instead
of silently dropping out. See task-2-report.md fix round 2.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from cold_start.growing.empirical import within_cell_variance

STATUS_OK = "ok"

_STRATUM_PATTERN = re.compile(r"^task_([emh])\d+$")


def success_matrix(
    outcomes: pd.DataFrame,
    pool: str,
    expected_arms: list[str] | None = None,
    expected_tasks: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str], int, np.ndarray]:
    """(prompts x tasks) replicate-0 ``ok`` success matrix.

    `expected_arms` / `expected_tasks`, when given, reindex the pivot to that full universe of ids
    so an arm or task with zero ``ok`` rows becomes an explicit (imputed) missing cell instead of
    silently disappearing from the matrix. Raises if more than 5% of the resulting cells are missing
    (imputation is a smoothing device for a handful of dropouts, not a substitute for real data), and
    -- regardless of that overall percentage -- raises if any single arm or task has *zero* observed
    cells: a fully-missing row/column has no real data to impute from (its row/col mean falls back to
    the grand mean only) and silently produces a degenerate ``row_counts`` entry of 0, which callers
    computing ``J / row_counts`` (fix round 2, finding N1) would divide by zero on. A caller that wants
    a dropped arm/task included anyway must supply real data for it, not rely on imputation.

    Returns ``(Y, arm_ids, task_ids, n_imputed, row_counts)`` where ``row_counts[i]`` is the number of
    genuinely observed (non-imputed) cells in row ``i``.
    """
    sub = outcomes[(outcomes["pool"] == pool) & (outcomes["replicate"] == 0) & (outcomes["status"] == STATUS_OK)]
    wide = sub.assign(success=sub["success"].astype(float)).pivot_table(
        index="arm_id", columns="task_id", values="success", aggfunc="first")
    if expected_arms is not None:
        wide = wide.reindex(index=[str(a) for a in expected_arms])
    if expected_tasks is not None:
        wide = wide.reindex(columns=[str(t) for t in expected_tasks])
    Y = wide.to_numpy(dtype=float)
    I, J = Y.shape
    arm_ids = [str(a) for a in wide.index]
    task_ids = [str(t) for t in wide.columns]
    observed = np.isfinite(Y)
    missing = ~observed
    n_imp = int(missing.sum())
    total = I * J
    row_counts = observed.sum(axis=1).astype(int)
    col_counts = observed.sum(axis=0).astype(int)
    zero_arms = [arm_ids[i] for i in range(I) if row_counts[i] == 0]
    zero_tasks = [task_ids[j] for j in range(J) if col_counts[j] == 0]
    if zero_arms or zero_tasks:
        raise ValueError(
            f"success_matrix: pool {pool!r} has zero observed cells for "
            f"arm(s) {zero_arms} and task(s) {zero_tasks} -- imputation cannot fill a row/column with "
            f"no real data; supply observations for these ids or drop them from expected_arms/expected_tasks"
        )
    if total and n_imp / total > 0.05:
        raise ValueError(
            f"success_matrix: {n_imp}/{total} cells missing for pool {pool!r} "
            f"({n_imp / total:.1%} > the 5% imputation ceiling)"
        )
    if n_imp:
        grand = float(Y[observed].mean()) if observed.any() else 0.0
        row_sum = np.where(observed, Y, 0.0).sum(axis=1)
        col_sum = np.where(observed, Y, 0.0).sum(axis=0)
        row_mean = np.divide(row_sum, row_counts, out=np.full(I, grand), where=row_counts > 0)
        col_mean = np.divide(col_sum, col_counts, out=np.full(J, grand), where=col_counts > 0)
        fill = np.clip(row_mean[:, None] + col_mean[None, :] - grand, 0.0, 1.0)
        Y = np.where(missing, fill, Y)
    return Y, arm_ids, task_ids, n_imp, row_counts


def variance_components(
    Y: np.ndarray,
    noise_var: float | None = None,
    n_imputed: int = 0,
    row_counts: np.ndarray | None = None,
) -> dict:
    """Two-way ANOVA moment estimates of tau^2, task variance, and residual (interaction+noise) variance.

    `n_imputed` reduces the residual degrees of freedom (an additively-imputed cell contributes ~0 to
    the residual sum of squares by construction but should not count as a free residual df, or MS_resid
    is biased low). `row_counts` (per-row count of genuinely observed cells, from `success_matrix`) lets
    rows with imputed cells -- whose means are less noisy than a fully-observed row's -- be corrected: the
    residual term entering the tau^2 moment equation is scaled by ``mean(J / n_i)`` (1.0 when every row
    is fully observed), rather than assuming every row has the full J-cell residual contribution.
    """
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    if I < 2 or J < 2:
        raise ValueError("variance components need at least 2 prompts and 2 tasks")
    grand = Y.mean()
    rm, cm = Y.mean(axis=1), Y.mean(axis=0)
    ms_prompt = J * np.sum((rm - grand) ** 2) / (I - 1)
    ms_task = I * np.sum((cm - grand) ** 2) / (J - 1)
    resid = Y - rm[:, None] - cm[None, :] + grand
    resid_df = (I - 1) * (J - 1) - int(n_imputed)
    if resid_df <= 0:
        raise ValueError("variance_components: too many imputed cells, residual df <= 0")
    ms_resid = np.sum(resid**2) / resid_df
    if row_counts is not None:
        row_counts_arr = np.asarray(row_counts, dtype=float)
        mean_j_over_ni = float(np.mean(J / row_counts_arr))
    else:
        mean_j_over_ni = 1.0
    tau2_raw = (ms_prompt - ms_resid * mean_j_over_ni) / J
    tau2 = max(tau2_raw, 0.0)
    out = {
        "tau2": float(tau2), "tau": float(np.sqrt(tau2)), "tau2_truncated": bool(tau2_raw < 0.0),
        "task_var": float(max((ms_task - ms_resid) / I, 0.0)),
        "ms_prompt": float(ms_prompt), "ms_task": float(ms_task), "ms_resid": float(ms_resid),
        "n_prompts": I, "n_tasks": J, "noise_var": None, "interaction_var": None,
        "resid_df": int(resid_df),
    }
    if noise_var is not None:
        out["noise_var"] = float(noise_var)
        out["interaction_var"] = float(max(ms_resid - noise_var, 0.0))
    return out


def _mls_bound(ms1: float, df1: int, ms2: float, df2: int, c1: float, c2: float, tail: float, side: str) -> float:
    """MLS half-width for theta = c1*MS1 - c2*MS2 at one-tail probability `tail` (Graybill & Wang 1980)."""
    G1 = 1.0 - df1 / stats.chi2.ppf(1.0 - tail, df1)
    H1 = df1 / stats.chi2.ppf(tail, df1) - 1.0
    G2 = 1.0 - df2 / stats.chi2.ppf(1.0 - tail, df2)
    H2 = df2 / stats.chi2.ppf(tail, df2) - 1.0
    if side == "lower":
        Fu = stats.f.ppf(1.0 - tail, df1, df2)
        G12 = ((Fu - 1.0) ** 2 - G1**2 * Fu**2 - H2**2) / Fu
        var = (c1 * ms1) ** 2 * G1**2 + (c2 * ms2) ** 2 * H2**2 + G12 * c1 * c2 * ms1 * ms2
    elif side == "upper":
        Fl = stats.f.ppf(tail, df1, df2)
        H12 = ((1.0 - Fl) ** 2 - H1**2 * Fl**2 - G2**2) / Fl
        var = (c1 * ms1) ** 2 * H1**2 + (c2 * ms2) ** 2 * G2**2 + H12 * c1 * c2 * ms1 * ms2
    else:
        raise ValueError("side must be 'lower' or 'upper'")
    return float(np.sqrt(max(var, 0.0)))


def tau_interval(
    Y: np.ndarray,
    *,
    alpha: float = 0.05,
    n_imputed: int = 0,
    row_counts: np.ndarray | None = None,
) -> dict:
    """Graybill-Wang Modified Large Sample (MLS) interval for tau = sqrt(sigma2_A) in the balanced
    two-way crossed random-effects design with one replicate per cell, sigma2_A = (MS_prompt -
    MS_resid) / J, df1 = I-1, df2 = residual df.

    Formula: Graybill, F.A. and Wang, C.M. (1980), "Confidence intervals on nonnegative linear
    combinations of variances", JASA 75(372), 869-873; as tabulated for theta = c1*MS1 - c2*MS2 in
    Burdick, R.K. and Graybill, F.A. (1992), Confidence Intervals on Variance Components, Ch. 3
    (two-factor crossed classification, n=1 replicate). Bounds are truncated at 0 before the square
    root, matching `variance_components`' truncation of tau2.

    Replaces the percentile bootstrap for tau (see module docstring): `two_way_bootstrap`'s
    column-with-replacement resampling inflated tau2 by roughly MS_resid/J and gave badly undercovering
    intervals; this closed-form interval does not resample at all.
    """
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    v = variance_components(Y, n_imputed=n_imputed, row_counts=row_counts)
    df1, df2 = I - 1, v["resid_df"]
    ms1, ms2 = v["ms_prompt"], v["ms_resid"]
    if row_counts is not None:
        mean_j_over_ni = float(np.mean(J / np.asarray(row_counts, dtype=float)))
    else:
        mean_j_over_ni = 1.0
    c1, c2 = 1.0 / J, mean_j_over_ni / J
    theta_hat = c1 * ms1 - c2 * ms2
    lo = theta_hat - _mls_bound(ms1, df1, ms2, df2, c1, c2, alpha / 2.0, "lower")
    hi = theta_hat + _mls_bound(ms1, df1, ms2, df2, c1, c2, alpha / 2.0, "upper")
    upper_one_sided = theta_hat + _mls_bound(ms1, df1, ms2, df2, c1, c2, alpha, "upper")
    return {
        "tau": v["tau"],
        "tau_lo": float(np.sqrt(max(lo, 0.0))),
        "tau_hi": float(np.sqrt(max(hi, 0.0))),
        "tau_upper_one_sided": float(np.sqrt(max(upper_one_sided, 0.0))),
    }


def tau_set_interval(
    Y: np.ndarray,
    noise_var: float | None,
    noise_df: int | None,
    *,
    alpha: float = 0.05,
    n_imputed: int = 0,
    row_counts: np.ndarray | None = None,
) -> dict:
    """Graybill-Wang MLS interval for tau_set = sqrt(sigma2_set), the SD of prompts' true success
    rates on THIS study's own task set -- main effect plus the prompt x task interaction averaged
    over these tasks (spec amendment 1d0e7b4, design doc SS2/6.1). Contrast with `tau_interval`
    (tau_main = sqrt(Var(a_i)) alone, the quantity that generalizes to new tasks): tau_main subtracts
    MS_resid (= sigma2_ab + sigma2_e, interaction *and* noise) from MS_prompt, so it estimates the
    main effect only; tau_set subtracts only the execution-noise variance sigma2_e -- estimated
    off-cell from the replicate pairs (`noise_from_pairs`), not from this cell's own MS_resid -- so
    the interaction stays folded into what tau_set measures, matching what a replay reservoir /
    NPMLE built from this cell's own task set actually represents.

    theta = c1*MS_prompt - c2*noise_var, c1 = 1/J, c2 = c/J, c = mean(J / row_counts) (1.0 when no
    cell was imputed) -- the same row-count correction `tau_interval` applies to MS_resid's
    coefficient, carried over unchanged to noise_var's coefficient since both are the residual side
    of the same c1*MS_prompt - c2*MS2 contrast `_mls_bound` was derived for. This is a first-order
    correction only: the exact expectation of MS_prompt under imputation also has a
    (c - 1) * sigma2_ab / J term (the interaction variance's own contribution to an imputed row's
    residual), which is neglected here -- negligible at the <= 5% missing-cell ceiling
    `success_matrix` enforces, and it has no closed-form separate from sigma2_ab, which this formula
    does not estimate.

    df1 = I - 1 (MS_prompt); df2 = `noise_df`, the number of replicate pairs the noise estimate was
    built from (NOT this cell's own residual df: sigma2_e's precision comes from wherever the pairs
    were run, e.g. the whole pool, not this one prompt x task grid).
    """
    if noise_var is None or not np.isfinite(noise_var):
        raise ValueError("tau_set_interval needs a finite noise_var (execution noise from replicate pairs)")
    if noise_df is None or noise_df < 1:
        raise ValueError("tau_set_interval needs noise_df >= 1 (the number of replicate pairs the noise "
                          "estimate was built from)")
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    v = variance_components(Y, n_imputed=n_imputed, row_counts=row_counts)
    df1, df2 = I - 1, int(noise_df)
    ms1, ms2 = v["ms_prompt"], float(noise_var)
    if row_counts is not None:
        mean_j_over_ni = float(np.mean(J / np.asarray(row_counts, dtype=float)))
    else:
        mean_j_over_ni = 1.0
    c1, c2 = 1.0 / J, mean_j_over_ni / J
    theta_hat = c1 * ms1 - c2 * ms2
    lo = theta_hat - _mls_bound(ms1, df1, ms2, df2, c1, c2, alpha / 2.0, "lower")
    hi = theta_hat + _mls_bound(ms1, df1, ms2, df2, c1, c2, alpha / 2.0, "upper")
    upper_one_sided = theta_hat + _mls_bound(ms1, df1, ms2, df2, c1, c2, alpha, "upper")
    return {
        "tau_set": float(np.sqrt(max(theta_hat, 0.0))),
        "tau_set_lo": float(np.sqrt(max(lo, 0.0))),
        "tau_set_hi": float(np.sqrt(max(hi, 0.0))),
        "tau_set_upper_one_sided": float(np.sqrt(max(upper_one_sided, 0.0))),
    }


def prompt_bootstrap(
    Y: np.ndarray,
    stat: Callable[..., float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    rows: np.ndarray | None = None,
    row_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Resample prompts (rows) with replacement; tasks (columns) are held fixed.

    Column resampling was removed (fix round 1, Critical finding): resampling tasks with replacement
    duplicates a task's residual into every row mean that draws it, which inflates the between-prompt
    mean square by roughly a factor of 2 without reducing the residual mean square's degrees of freedom
    for the duplicated columns, so a percentile interval built on it badly undercovers tau (measured as
    low as 0.20-0.23 nominal-95% coverage). Prompts are exchangeable draws from the pool by design, so
    row-only resampling is the valid bootstrap here.

    `rows` lets a caller share resampling indices across cells (paired contrasts): pass an (n_boot, I)
    index array; otherwise indices are drawn here.

    `row_counts` (per-row observed-cell count from `success_matrix`), when given, is resampled with the
    *same* row indices as `Y` and passed to `stat` as a second positional argument --
    ``stat(Y[R[b]], row_counts[R[b]])`` -- instead of ``stat(Y[R[b]])``. Without this, a `stat` closure
    over `variance_components`'s ``n_imputed``/``row_counts`` correction would silently be evaluated
    against the *original* (pre-resample) row counts, or not at all, so the imputation row-count
    correction would drop out of any bootstrap contrast built on this function (fix round 2 Gap
    finding). Omit `row_counts` to keep the old single-argument call, e.g. for a `stat` that does not
    need it.
    """
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    rng = np.random.default_rng(seed)
    R = rows if rows is not None else rng.integers(0, I, (n_boot, I))
    if row_counts is not None:
        rc = np.asarray(row_counts, dtype=float)
        return np.array([stat(Y[R[b]], rc[R[b]]) for b in range(len(R))])
    return np.array([stat(Y[R[b]]) for b in range(len(R))])


def _halves(task_ids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: dict[str, list[int]] = {}
    for j, t in enumerate(task_ids):
        m = _STRATUM_PATTERN.match(t)
        if m is None:
            raise ValueError(
                f"task id {t!r} does not match the split-half stratification pattern "
                f"{_STRATUM_PATTERN.pattern!r}"
            )
        strata.setdefault(m.group(1), []).append(j)
    a: list[int] = []
    b: list[int] = []
    give_extra_to_a = True
    for key in sorted(strata):
        idx = np.array(strata[key])
        rng.shuffle(idx)
        half = len(idx) // 2
        if len(idx) % 2 == 0:
            a += list(idx[:half])
            b += list(idx[half:])
        elif give_extra_to_a:
            a += list(idx[: half + 1])
            b += list(idx[half + 1 :])
            give_extra_to_a = False
        else:
            a += list(idx[:half])
            b += list(idx[half:])
            give_extra_to_a = True
    return np.array(sorted(a)), np.array(sorted(b))


def split_half(Y: np.ndarray, task_ids: list[str], *, seed: int = 0, n_perm: int = 10_000) -> dict:
    Y = np.asarray(Y, dtype=float)
    a, b = _halves(task_ids, seed)
    xa, xb = Y[:, a].mean(axis=1), Y[:, b].mean(axis=1)
    if xa.std() == 0.0 or xb.std() == 0.0:
        r = 0.0
        p = 1.0
    else:
        r = float(np.corrcoef(xa, xb)[0, 1])
        rng = np.random.default_rng(seed + 1)
        null = np.array([np.corrcoef(xa, rng.permutation(xb))[0, 1] for _ in range(n_perm)])
        null = np.nan_to_num(null)
        p = float((1 + np.sum(null >= r)) / (1 + n_perm))
    r_sb = float(2 * r / (1 + r)) if r > 0.0 else float("nan")
    return {"r": r, "r_sb": r_sb, "p_value": p, "n_half_a": int(a.size), "n_half_b": int(b.size)}


def prompt_effects(Y: np.ndarray) -> np.ndarray:
    """Empirical-Bayes shrunken prompt main effects: (row mean - grand) * tau2 / (tau2 + MS_resid / J)."""
    v = variance_components(Y)
    raw = Y.mean(axis=1) - Y.mean()
    denom = v["tau2"] + v["ms_resid"] / v["n_tasks"]
    return raw * (v["tau2"] / denom if denom > 0 else 0.0)


def discriminating_tasks(Y: np.ndarray, lo: float = 0.2, hi: float = 0.8) -> np.ndarray:
    cm = np.asarray(Y, dtype=float).mean(axis=0)
    return (cm >= lo) & (cm <= hi)


def noise_from_pairs(outcomes: pd.DataFrame, pool: str) -> tuple[float, int]:
    """Execution noise ``v = E[(x0 - x1)^2] / 2`` from replicate-pair cells, via
    `cold_start.growing.empirical.within_cell_variance` (the same estimator Pre-registration 9 uses)."""
    try:
        return within_cell_variance(outcomes, [pool])
    except ValueError:
        return float("nan"), 0
