"""Prompt heterogeneity in a crossed prompt x task design (Pre-registration 10, section 6).

    y_ij = mu + a_i + b_j + (ab)_ij + e_ij

With one replicate-0 observation per (prompt, task) cell, the two-way ANOVA mean squares give
unbiased moment estimates: E[MS_prompt] = s2_resid + J tau^2, E[MS_task] = s2_resid + I s2_task,
E[MS_resid] = s2_resid = s2_ab + s2_e. tau therefore needs no noise model and no NPMLE; the replicate
pairs only split s2_resid into interaction and execution noise. Negative moment estimates are truncated at
zero and flagged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

STATUS_OK = "ok"


def success_matrix(outcomes: pd.DataFrame, pool: str) -> tuple[np.ndarray, list[str], list[str], int]:
    sub = outcomes[(outcomes["pool"] == pool) & (outcomes["replicate"] == 0) & (outcomes["status"] == STATUS_OK)]
    wide = sub.assign(success=sub["success"].astype(float)).pivot_table(
        index="arm_id", columns="task_id", values="success", aggfunc="first")
    Y = wide.to_numpy(dtype=float)
    missing = ~np.isfinite(Y)
    n_imp = int(missing.sum())
    if n_imp:
        grand = np.nanmean(Y)
        row = np.nanmean(Y, axis=1, keepdims=True)
        col = np.nanmean(Y, axis=0, keepdims=True)
        Y = np.where(missing, np.clip(row + col - grand, 0.0, 1.0), Y)
    return Y, [str(a) for a in wide.index], [str(t) for t in wide.columns], n_imp


def variance_components(Y: np.ndarray, noise_var: float | None = None) -> dict:
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    if I < 2 or J < 2:
        raise ValueError("variance components need at least 2 prompts and 2 tasks")
    grand = Y.mean()
    rm, cm = Y.mean(axis=1), Y.mean(axis=0)
    ms_prompt = J * np.sum((rm - grand) ** 2) / (I - 1)
    ms_task = I * np.sum((cm - grand) ** 2) / (J - 1)
    resid = Y - rm[:, None] - cm[None, :] + grand
    ms_resid = np.sum(resid**2) / ((I - 1) * (J - 1))
    tau2_raw = (ms_prompt - ms_resid) / J
    tau2 = max(tau2_raw, 0.0)
    out = {
        "tau2": float(tau2), "tau": float(np.sqrt(tau2)), "tau2_truncated": bool(tau2_raw < 0.0),
        "task_var": float(max((ms_task - ms_resid) / I, 0.0)),
        "ms_prompt": float(ms_prompt), "ms_task": float(ms_task), "ms_resid": float(ms_resid),
        "n_prompts": I, "n_tasks": J, "noise_var": None, "interaction_var": None,
    }
    if noise_var is not None:
        out["noise_var"] = float(noise_var)
        out["interaction_var"] = float(max(ms_resid - noise_var, 0.0))
    return out


def two_way_bootstrap(Y: np.ndarray, stat: Callable[[np.ndarray], float], *, n_boot: int = 2000, seed: int = 0,
                      rows: np.ndarray | None = None, cols: np.ndarray | None = None) -> np.ndarray:
    """Resample prompts (rows) and tasks (columns) independently with replacement.

    `rows` / `cols` let a caller share resampling indices across cells (paired contrasts): pass an
    (n_boot, I) / (n_boot, J) index array; otherwise indices are drawn here.
    """
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    rng = np.random.default_rng(seed)
    R = rows if rows is not None else rng.integers(0, I, (n_boot, I))
    C = cols if cols is not None else rng.integers(0, J, (n_boot, J))
    return np.array([stat(Y[np.ix_(R[b], C[b])]) for b in range(len(R))])


def _halves(task_ids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: dict[str, list[int]] = {}
    for j, t in enumerate(task_ids):
        key = t.split("_")[1][0] if "_" in t else "x"
        strata.setdefault(key, []).append(j)
    a, b = [], []
    for key in sorted(strata):
        idx = np.array(strata[key])
        rng.shuffle(idx)
        a += list(idx[: len(idx) // 2])
        b += list(idx[len(idx) // 2:])
    return np.array(sorted(a)), np.array(sorted(b))


def split_half(Y: np.ndarray, task_ids: list[str], *, seed: int = 0, n_perm: int = 10_000) -> dict:
    Y = np.asarray(Y, dtype=float)
    a, b = _halves(task_ids, seed)
    xa, xb = Y[:, a].mean(axis=1), Y[:, b].mean(axis=1)
    r = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 and xb.std() > 0 else 0.0
    rng = np.random.default_rng(seed + 1)
    null = np.array([np.corrcoef(xa, rng.permutation(xb))[0, 1] for _ in range(n_perm)])
    null = np.nan_to_num(null)
    p = float((1 + np.sum(null >= r)) / (1 + n_perm))
    r_sb = 2 * r / (1 + r) if r > -1 else float("nan")
    return {"r": r, "r_sb": float(r_sb), "p_value": p, "n_half_a": int(a.size), "n_half_b": int(b.size)}


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
    ok = outcomes[(outcomes["pool"] == pool) & (outcomes["status"] == STATUS_OK)]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success", aggfunc="first")
    if 0 not in wide.columns or 1 not in wide.columns:
        return float("nan"), 0
    pairs = wide[[0, 1]].dropna()
    if pairs.empty:
        return float("nan"), 0
    d = pairs[0].to_numpy() - pairs[1].to_numpy()
    return float(np.mean(d**2) / 2.0), int(len(pairs))
