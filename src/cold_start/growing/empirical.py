"""Real-prompt reservoirs from logged WebArena outcomes.

The collector (`experiments/growing_bandits/empirical/collect.py`) writes one JSON line per
attempt. This module turns those lines into reservoirs the simulator can run on:

1. **Scores.** A prompt's score is its success rate over the task bank, replicate 0 only,
   ``ok`` episodes only (``missing`` never counts as a failure).
2. **Noise.** The score is a sum of heterogeneous Bernoullis, one per task, so the binomial
   variance would over-correct: it assumes every task sits at the prompt's mean. The
   within-cell variance ``v`` is measured directly from replicate pairs, and prompt i's
   measurement variance is ``v / n_i``.
3. **Deconvolution.** The distribution of true prompt rates is the nonparametric maximum
   likelihood estimate (Kiefer-Wolfowitz) under ``xbar_i ~ N(mu_i, v / n_i)``, by EM on a
   fixed 401-point grid. No shape is assumed.

The raw reservoir (observed rates, equal weight) and a parametric fit are sensitivity
variants: raw over-states the spread by construction; parametric smooths the tail.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from cold_start.growing.empirical_reservoir import EmpiricalReservoir
from cold_start.growing.reservoirs import BetaReservoir, MixtureReservoir, Reservoir, TailReservoir

SCHEMA = "empirical_pool/1"
STATUS_OK = "ok"
STATUS_INFRA = "infra_error"
STATUS_MISSING = "missing"
TERMINAL_STATUSES: tuple[str, ...] = (STATUS_OK, STATUS_MISSING)
RECORD_COLUMNS: tuple[str, ...] = (
    "schema", "pool", "arm_id", "task_id", "replicate", "attempt", "status", "success", "cost_usd",
)
KEY: list[str] = ["arm_id", "task_id", "replicate"]

GRID = np.linspace(0.0, 1.0, 401)
NPMLE_TOL = 1e-8
NPMLE_MAX_ITER = 100_000
MIN_WEIGHT = 1e-6


# ---- logs -----------------------------------------------------------------------


def load_attempts(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Every attempt line from the collector's JSONL files, in file order (strict: any bad line raises)."""
    rows: list[dict] = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return attempts_frame(rows)


def attempts_frame(rows: list[dict]) -> pd.DataFrame:
    """Attempt records as a frame with `RECORD_COLUMNS` first (present even when empty)."""
    frame = pd.DataFrame(rows)
    for col in RECORD_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.Series(dtype=object)
    extra = [c for c in frame.columns if c not in RECORD_COLUMNS]
    return frame[list(RECORD_COLUMNS) + extra]


def terminal_outcomes(attempts: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, task, replicate) that finished: ``ok`` or ``missing``.

    Two terminal rows for one key mean a resume ran an item twice; averaging them would
    silently weight that cell double, so it is an error.
    """
    out = attempts[attempts["status"].isin(TERMINAL_STATUSES)].copy()
    dup = out.duplicated(KEY, keep=False)
    if dup.any():
        keys = out.loc[dup, KEY].drop_duplicates().to_dict("records")
        raise ValueError(f"duplicate terminal records for {len(keys)} keys, e.g. {keys[:3]}")
    return out.reset_index(drop=True)


# ---- scores and noise -------------------------------------------------------------


@dataclass(frozen=True)
class PromptScores:
    pool: str
    arm_ids: tuple[str, ...]
    successes: np.ndarray
    n: np.ndarray

    @property
    def means(self) -> np.ndarray:
        return self.successes / self.n


def prompt_scores(outcomes: pd.DataFrame, pool: str) -> PromptScores:
    """Per-prompt successes and ``ok`` counts on replicate 0; prompts with no ``ok`` episode drop."""
    sub = outcomes[
        (outcomes["pool"] == pool) & (outcomes["replicate"] == 0) & (outcomes["status"] == STATUS_OK)
    ]
    grouped = sub.assign(success=sub["success"].astype(float)).groupby("arm_id")["success"]
    agg = grouped.agg(["sum", "count"]).sort_index()
    agg = agg[agg["count"] > 0]
    return PromptScores(
        pool=pool,
        arm_ids=tuple(str(a) for a in agg.index),
        successes=agg["sum"].to_numpy(dtype=float),
        n=agg["count"].to_numpy(dtype=float),
    )


def within_cell_variance(outcomes: pd.DataFrame, pools: Iterable[str] | None = None) -> tuple[float, int]:
    """``v = E[(x0 - x1)^2] / 2`` over (arm, task) cells run twice, both ``ok``."""
    ok = outcomes[outcomes["status"] == STATUS_OK]
    if pools is not None:
        ok = ok[ok["pool"].isin(list(pools))]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success",
                          aggfunc="first")
    if 0 not in wide.columns or 1 not in wide.columns:
        raise ValueError("no replicate pairs: within-cell variance needs replicate 0 and 1 of the same cell")
    pairs = wide[[0, 1]].dropna()
    if pairs.empty:
        raise ValueError("no replicate pairs: within-cell variance needs replicate 0 and 1 of the same cell")
    d = pairs[0].to_numpy(dtype=float) - pairs[1].to_numpy(dtype=float)
    return float(np.mean(d**2) / 2.0), int(len(pairs))


# ---- NPMLE --------------------------------------------------------------------------


def _likelihood_matrix(xbar: np.ndarray, sigma2: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return stats.norm.pdf(xbar[:, None], loc=grid[None, :], scale=np.sqrt(sigma2)[:, None])


def npmle(
    xbar: np.ndarray,
    sigma2: np.ndarray,
    grid: np.ndarray = GRID,
    tol: float = NPMLE_TOL,
    max_iter: int = NPMLE_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Kiefer-Wolfowitz NPMLE of the mixing distribution by EM on a fixed grid.

    Returns ``(grid, weights, log_likelihood)``. EM never decreases the likelihood; it stops
    when an iteration gains less than `tol`.
    """
    x = np.asarray(xbar, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    if x.size == 0 or x.shape != s2.shape:
        raise ValueError("xbar and sigma2 must be non-empty and the same length")
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0.0):
        raise ValueError("every measurement variance must be finite and positive")
    lik = _likelihood_matrix(x, s2, grid)
    w = np.full(grid.size, 1.0 / grid.size)
    prev = -np.inf
    for _ in range(max_iter):
        mix = lik @ w
        ll = float(np.sum(np.log(mix)))
        if ll - prev < tol:
            return grid.copy(), w, ll
        prev = ll
        w = w * (lik.T @ (1.0 / mix)) / x.size
    raise RuntimeError(f"NPMLE did not converge in {max_iter} iterations")


def npmle_reservoir(scores: PromptScores, v: float, label: str) -> EmpiricalReservoir:
    atoms, w, _ = npmle(scores.means, v / scores.n)
    w = np.where(w < MIN_WEIGHT, 0.0, w)
    return EmpiricalReservoir(atoms, w / w.sum(), label=label)


def raw_reservoir(scores: PromptScores, label: str) -> EmpiricalReservoir:
    k = scores.means.size
    return EmpiricalReservoir(np.clip(scores.means, 0.0, 1.0), np.full(k, 1.0 / k), label=label)


# ---- parametric sensitivity -------------------------------------------------------


def grid_masses(res: Reservoir, grid: np.ndarray = GRID) -> np.ndarray:
    """The reservoir's probability of each grid cell (midpoint edges), from `_survival`."""
    mids = (grid[:-1] + grid[1:]) / 2.0
    edges = np.concatenate([[-1e-12], mids, [1.0 + 1e-12]])
    surv = np.asarray(res._survival(edges), dtype=float)
    mass = np.clip(surv[:-1] - surv[1:], 0.0, None)
    total = mass.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("reservoir puts no mass on [0, 1]")
    return mass / total


def _moments(scores: PromptScores, v: float) -> tuple[float, float]:
    m = float(np.clip(np.mean(scores.means), 0.02, 0.98))
    var = float(np.var(scores.means, ddof=1) - np.mean(v / scores.n))
    return m, max(var, 1e-4)


#: Nelder-Mead explores log-parameters freely; exp(710) overflows a float64. Clipping to
#: +-30 (e^30 ~ 1e13) keeps every shape parameter finite without constraining any fit
#: that could matter -- a Beta with a parameter near 1e13 is already a point mass.
THETA_CLIP = 30.0


def _pos(theta: np.ndarray, i: int) -> float:
    return float(np.exp(np.clip(theta[i], -THETA_CLIP, THETA_CLIP)))


def _beta(theta: np.ndarray) -> Reservoir:
    return BetaReservoir(_pos(theta, 0), _pos(theta, 1), validate=False)


def _tail(theta: np.ndarray) -> Reservoir:
    return TailReservoir(_pos(theta, 0), float(special.expit(theta[1])), _pos(theta, 2), validate=False)


def _beta_mixture(theta: np.ndarray) -> Reservoir:
    comps = [BetaReservoir(_pos(theta, 0), _pos(theta, 1), validate=False),
             BetaReservoir(_pos(theta, 2), _pos(theta, 3), validate=False)]
    w = float(special.expit(theta[4]))
    return MixtureReservoir(comps, [w, 1.0 - w], validate=False)


def _starts(scores: PromptScores, v: float) -> dict[str, np.ndarray]:
    m, var = _moments(scores, v)
    strength = max(m * (1.0 - m) / var - 1.0, 0.5)
    sd = np.sqrt(var)
    top = float(np.clip(np.max(scores.means) + 0.05, 0.05, 0.99))
    width = max(top - float(np.min(scores.means)), 0.05)
    lo_m, hi_m = float(np.clip(m - sd, 0.02, 0.98)), float(np.clip(m + sd, 0.02, 0.98))
    return {
        "beta": np.log([m * strength, (1.0 - m) * strength]),
        "tail": np.array([np.log(2.0), special.logit(top), np.log(1.0 / width**2)]),
        "beta_mixture": np.array([*np.log([lo_m * strength, (1.0 - lo_m) * strength]),
                                  *np.log([hi_m * strength, (1.0 - hi_m) * strength]), 0.0]),
    }


FAMILIES: dict[str, Callable[[np.ndarray], Reservoir]] = {
    "beta": _beta, "tail": _tail, "beta_mixture": _beta_mixture,
}


def fit_parametric(scores: PromptScores, v: float, label: str) -> tuple[EmpiricalReservoir, pd.DataFrame]:
    """Fit each family by maximum marginal likelihood; return the best-AIC fit on `GRID`."""
    lik = _likelihood_matrix(scores.means, v / scores.n, GRID)
    starts = _starts(scores, v)
    rows: list[dict] = []
    best: tuple[float, np.ndarray] | None = None

    for family, build in FAMILIES.items():
        def nll(theta: np.ndarray, build=build) -> float:
            try:
                mass = grid_masses(build(theta))
            except (ValueError, FloatingPointError, ZeroDivisionError):
                return 1e12
            mix = lik @ mass
            if not np.all(np.isfinite(mix)) or np.any(mix <= 0.0):
                return 1e12
            return float(-np.sum(np.log(mix)))

        fit = optimize.minimize(nll, starts[family], method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        k = starts[family].size
        aic = 2.0 * k + 2.0 * float(fit.fun)
        rows.append({"family": family, "n_params": k, "nll": float(fit.fun), "aic": aic,
                     "converged": bool(fit.success), "theta": json.dumps([float(t) for t in fit.x])})
        if best is None or aic < best[0]:
            best = (aic, grid_masses(build(fit.x)))

    table = pd.DataFrame(rows)
    table["selected"] = table["aic"] == table["aic"].min()
    assert best is not None
    return EmpiricalReservoir(GRID, best[1], label=label), table
