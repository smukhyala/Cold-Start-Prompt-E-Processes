"""Paired, stratified and clustered uncertainty for the deployment tables.

The unit of independence is the episode, and within a cell every policy saw the same
episodes (common random numbers). So the right estimate of "policy A beats B by d" is
the mean of the per-episode *paired* differences with a bootstrap over episodes --
never a comparison of two independently bootstrapped means, which throws the pairing
away and inflates the interval by the shared episode variance.

Pooling across cells is where Simpson's paradox lives (register #8): Family B is 80%
of the corpus and per-environment search rates range 0.35-0.81, so an episode-weighted
pool is dominated by whichever cells happen to be largest. `stratified_pooled` gives
every cell equal weight and resamples within cells; `cluster_bootstrap_over_envs`
resamples whole environments, which is the honest interval when the question is
"would this hold on a fresh environment".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

#: Resamples are drawn in blocks so a 10k x 2000 index matrix never materializes.
_BOOT_CHUNK = 512


def _ci_bounds(ci: float) -> tuple[float, float]:
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}")
    tail = 100.0 * (1.0 - ci) / 2.0
    return tail, 100.0 - tail


def _resample_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_boot,)`` means of with-replacement resamples of `x`."""
    n = int(x.shape[0])
    out = np.empty(int(n_boot), dtype=np.float64)
    for start in range(0, int(n_boot), _BOOT_CHUNK):
        stop = min(start + _BOOT_CHUNK, int(n_boot))
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = x[idx].mean(axis=1)
    return out


def paired_bootstrap(
    diff: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    *,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> dict:
    """Mean of paired differences with a percentile bootstrap CI over episodes.

    `diff` is ``a - b`` per episode (sign convention is the caller's). Returns
    ``mean, lo, hi, se_paired, se_unpaired_equiv, win_rate`` plus the raw fractions
    ``frac_positive / frac_zero / frac_negative``. ``win_rate`` counts a tie as half a
    win: under CRN a large share of episodes recommend the *same* arm under both
    policies and contribute exactly zero, and a strict ``P(diff > 0)`` would score two
    identical policies at 0 rather than 0.5.

    ``se_unpaired_equiv`` is what the SE would have been had the two policies been run
    on independent episodes, ``sqrt((var_a + var_b) / n)``; it needs the two marginal
    arrays, so it is NaN unless `a` and `b` are given. Its ratio to ``se_paired`` is
    the variance reduction the CRN design bought.
    """
    d = np.asarray(diff, dtype=np.float64).ravel()
    n = int(d.shape[0])
    if n < 1:
        raise ValueError("paired_bootstrap needs at least one difference")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    lo_pct, hi_pct = _ci_bounds(ci)

    rng = np.random.default_rng(seed)
    boot = _resample_means(d, n_boot, rng)
    lo, hi = np.percentile(boot, [lo_pct, hi_pct])

    se_paired = float(d.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")
    se_unpaired = float("nan")
    if a is not None and b is not None:
        aa = np.asarray(a, dtype=np.float64).ravel()
        bb = np.asarray(b, dtype=np.float64).ravel()
        if aa.shape != (n,) or bb.shape != (n,):
            raise ValueError(
                f"a and b must be episode-aligned with diff (n={n}); got {aa.shape}, {bb.shape}"
            )
        if n >= 2:
            se_unpaired = float(np.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / n))

    frac_pos = float(np.mean(d > 0.0))
    frac_zero = float(np.mean(d == 0.0))
    return {
        "mean": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "se_paired": se_paired,
        "se_unpaired_equiv": se_unpaired,
        "win_rate": frac_pos + 0.5 * frac_zero,
        "frac_positive": frac_pos,
        "frac_zero": frac_zero,
        "frac_negative": float(np.mean(d < 0.0)),
        "n": n,
        "n_boot": int(n_boot),
        "ci": float(ci),
    }


def stratified_pooled(
    diffs_by_cell: dict[str, np.ndarray],
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Equal-weight pool of per-cell mean differences, cell-stratified bootstrap.

    Each cell is resampled within itself (episodes are exchangeable only within a
    cell) and the resampled cell means are averaged with equal weight, so a cell with
    more episodes does not get more say. Returns ``mean, lo, hi, se, n_cells,
    cell_means``.
    """
    if not diffs_by_cell:
        raise ValueError("stratified_pooled needs at least one cell")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    lo_pct, hi_pct = _ci_bounds(ci)

    rng = np.random.default_rng(seed)
    cell_means: dict[str, float] = {}
    pooled_boot = np.zeros(int(n_boot), dtype=np.float64)
    for cell, diff in diffs_by_cell.items():
        d = np.asarray(diff, dtype=np.float64).ravel()
        if d.shape[0] < 1:
            raise ValueError(f"cell {cell!r} has no episodes")
        cell_means[cell] = float(d.mean())
        pooled_boot += _resample_means(d, n_boot, rng)
    n_cells = len(diffs_by_cell)
    pooled_boot /= n_cells
    lo, hi = np.percentile(pooled_boot, [lo_pct, hi_pct])
    return {
        "mean": float(np.mean(list(cell_means.values()))),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(pooled_boot.std(ddof=1)) if n_boot >= 2 else float("nan"),
        "n_cells": n_cells,
        "cell_means": cell_means,
        "n_boot": int(n_boot),
        "ci": float(ci),
    }


def cluster_bootstrap_over_envs(
    cell_means: pd.DataFrame,
    env_col: str,
    value_col: str,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Mean of `value_col` over cells, with environments resampled as clusters.

    Cells of one environment (its horizons, caps) share that environment's reservoir
    and are not independent draws of "an environment", so the interval for a claim
    about environments in general must resample environments, not cells. Each
    resample draws ``n_envs`` environments with replacement and averages every cell
    they contribute. Returns ``mean, lo, hi, se, n_envs, n_cells``.
    """
    if env_col not in cell_means.columns or value_col not in cell_means.columns:
        raise KeyError(f"cell_means needs columns {env_col!r} and {value_col!r}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    lo_pct, hi_pct = _ci_bounds(ci)

    values = cell_means[value_col].to_numpy(dtype=np.float64)
    if values.shape[0] < 1:
        raise ValueError("cell_means has no rows")
    env_codes, env_index = pd.factorize(cell_means[env_col], sort=True)
    n_envs = int(len(env_index))
    sums = np.bincount(env_codes, weights=values, minlength=n_envs)
    counts = np.bincount(env_codes, minlength=n_envs).astype(np.float64)

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for start in range(0, int(n_boot), _BOOT_CHUNK):
        stop = min(start + _BOOT_CHUNK, int(n_boot))
        idx = rng.integers(0, n_envs, size=(stop - start, n_envs))
        boot[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [lo_pct, hi_pct])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot.std(ddof=1)) if n_boot >= 2 else float("nan"),
        "n_envs": n_envs,
        "n_cells": int(values.shape[0]),
        "n_boot": int(n_boot),
        "ci": float(ci),
    }


def _rank_correlation(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """Pearson correlation of pre-ranked rows, NaN where a row has no variance.

    Written out rather than delegated to `scipy.stats.spearmanr` so a constant
    resample yields a silent NaN instead of a `ConstantInputWarning`, and so the
    bootstrap runs as one vectorized pass over ``(n_boot, n)`` rank matrices.
    """
    cx = rx - rx.mean(axis=-1, keepdims=True)
    cy = ry - ry.mean(axis=-1, keepdims=True)
    sxx = np.sum(cx * cx, axis=-1)
    syy = np.sum(cy * cy, axis=-1)
    sxy = np.sum(cx * cy, axis=-1)
    denom = np.sqrt(sxx * syy)
    ok = denom > 0.0
    out = np.full(np.shape(denom), np.nan, dtype=np.float64)
    np.divide(sxy, denom, out=out, where=ok)
    return out


def spearman_with_ci(x, y, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Spearman's rho with a percentile bootstrap CI over (x, y) pairs.

    Used for H3 (offline AUC vs deployed regret across learned variants), where the
    number of pairs is small and a normal approximation would be a guess. Returns
    ``rho, lo, hi, n``; the bounds are NaN if every resample was degenerate.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"x and y must be paired; got shapes {xs.shape} and {ys.shape}")
    n = int(xs.shape[0])
    if n < 2:
        raise ValueError("spearman_with_ci needs at least two pairs")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    lo_pct, hi_pct = _ci_bounds(ci)

    rho = float(_rank_correlation(rankdata(xs), rankdata(ys)))

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for start in range(0, int(n_boot), _BOOT_CHUNK):
        stop = min(start + _BOOT_CHUNK, int(n_boot))
        idx = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = _rank_correlation(
            rankdata(xs[idx], axis=1), rankdata(ys[idx], axis=1)
        )
    valid = boot[np.isfinite(boot)]
    if valid.size:
        lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    else:
        lo, hi = float("nan"), float("nan")
    return {
        "rho": rho,
        "lo": float(lo),
        "hi": float(hi),
        "n": n,
        "n_boot": int(n_boot),
        "n_degenerate": int(boot.shape[0] - valid.size),
        "ci": float(ci),
    }
