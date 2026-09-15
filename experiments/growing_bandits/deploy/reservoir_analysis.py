"""Reservoir-tail diagnostics for the deployed learned policies (plan section 13b).

The question a SEARCH decision answers is "is a fresh draw from the reservoir worth
more than refining what I hold?". The oracle version of that quantity is known here
because this is a simulation: at a logged on-policy state with incumbent true mean
``mu_inc`` (the best true mean among the arms held),

    p_t = P(mu > mu_inc) = reservoir.tail_prob(mu_inc)
    I_t = INT_{mu_inc}^1 P(mu > x) dx        (midpoint quadrature on a 400-point grid)

is the tail mass and the expected improvement of one fresh draw. The deployable
estimates a policy can actually see are `est_p_new_beats_incumbent` and
`est_I_hat = beta_excess_mean(est_beta_a, est_beta_b, f_best_mean)` from the
method-of-moments Beta fit to the discovered arms. This module scores, per (cell,
policy): how well the oracle `I_t` / `p_t` predict the policy's own SEARCH decisions
(AUC, point-biserial correlation), how well the model's probability does, and how
faithful the estimates are to the oracle (Spearman); then joins the corpus-side
ceilings from `reservoir_diagnostics_offline.csv` (AUC of the same scores for the
oracle *label*).

Everything here reads `oracle_*` columns and a `Reservoir`; it runs only on logged
states after the policy acted (register #4), never inside a policy.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cold_start.growing.reservoirs import Reservoir  # noqa: E402

log = logging.getLogger("deploy.reservoir")

QUADRATURE_GRID = 400
#: Offline scores joined from `reservoir_diagnostics_offline.csv` (`scope == "T=<T>"`).
OFFLINE_SCORES: tuple[str, ...] = (
    "oracle_p_new_beats_best_true",
    "oracle_I_t",
    "est_p_new_beats_incumbent",
    "est_I_hat",
)
N_I_BINS = 10


def oracle_tail_integral(
    reservoir: Reservoir, c: np.ndarray, n_grid: int = QUADRATURE_GRID
) -> np.ndarray:
    """``I(c) = INT_c^1 P(mu > x) dx`` for every `c`, by midpoint quadrature.

    The same scheme as `train_policies.oracle_improvement_integral` (which takes a
    corpus frame keyed by `meta_env`); restated on a reservoir object so it applies to
    any cell, held-out mixtures included. The cumulative tail is accumulated from the
    top so every `c` is one interpolation.
    """
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    h = 1.0 / n_grid
    mids = (np.arange(n_grid) + 0.5) * h
    edges = np.arange(n_grid + 1) * h
    tail = np.array([reservoir.tail_prob(float(x)) for x in mids], dtype=np.float64)
    cum = np.concatenate([np.cumsum(tail[::-1])[::-1] * h, [0.0]])
    return np.interp(c, edges, cum)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(s) & np.isfinite(y)
    y, s = y[ok], s[ok]
    if y.size < 2 or y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y.astype(np.int64), s))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return _pearson(rankdata(a[ok]), rankdata(b[ok]))


def with_oracle_columns(on: pd.DataFrame, reservoir: Reservoir) -> pd.DataFrame:
    """Add ``oracle_p_t`` and ``oracle_I_t`` (at the incumbent's TRUE mean) to a feature frame."""
    out = on.copy()
    c = out["oracle_best_true_mu"].to_numpy(dtype=np.float64)
    out["oracle_p_t"] = out["oracle_p_new_beats_best_true"].to_numpy(dtype=np.float64)
    out["oracle_I_t"] = oracle_tail_integral(reservoir, c)
    return out


def reservoir_row(on: pd.DataFrame, meta: dict, offline: pd.DataFrame | None = None) -> dict:
    """One (cell, policy) row of `reservoir_<test>.csv` from its oracle-augmented frame.

    `on` must carry ``oracle_I_t`` / ``oracle_p_t`` (`with_oracle_columns`),
    ``decision_search`` (NaN where the step at the snapshot's time was not logged),
    ``p_search_model`` (NaN for the hand rule) and ``tau``.
    """
    dec = on["decision_search"].to_numpy(dtype=np.float64)
    has = np.isfinite(dec)
    y = dec[has] > 0.5
    I_t = on["oracle_I_t"].to_numpy(dtype=np.float64)
    p = on["oracle_p_t"].to_numpy(dtype=np.float64)
    est_I = on["est_I_hat"].to_numpy(dtype=np.float64)
    est_p = on["est_p_new_beats_incumbent"].to_numpy(dtype=np.float64)
    pm = on["p_search_model"].to_numpy(dtype=np.float64)
    tau = on["tau"].to_numpy(dtype=np.float64)
    inclination = np.where(np.isfinite(pm) & np.isfinite(tau), pm > tau, np.nan)

    row = dict(meta)
    row.update(
        {
            "n_states": int(len(on)),
            "n_decisions": int(has.sum()),
            "search_rate": float(y.mean()) if y.size else float("nan"),
            "mean_oracle_I_t": float(np.nanmean(I_t)) if I_t.size else float("nan"),
            "mean_oracle_p_t": float(np.nanmean(p)) if p.size else float("nan"),
            "mean_est_I_hat": float(np.nanmean(est_I)) if est_I.size else float("nan"),
            "mean_est_p": float(np.nanmean(est_p)) if est_p.size else float("nan"),
            # (b) do the policy's own decisions track the oracle?
            "auc_oracle_I_for_decision": _auc(y, I_t[has]),
            "auc_oracle_p_for_decision": _auc(y, p[has]),
            "auc_est_I_hat_for_decision": _auc(y, est_I[has]),
            "auc_est_p_for_decision": _auc(y, est_p[has]),
            "auc_model_p_for_decision": _auc(y, pm[has]),
            "corr_decision_oracle_I": _pearson(dec[has], I_t[has]),
            "corr_decision_oracle_p": _pearson(dec[has], p[has]),
            "corr_model_p_oracle_I": _pearson(pm, I_t),
            "spearman_model_p_oracle_I": _spearman(pm, I_t),
            # the model's fresh inclination (p > tau), free of the commitment window
            "n_inclinations": int(np.isfinite(inclination).sum()),
            "inclination_rate": (
                float(np.nanmean(inclination)) if np.isfinite(inclination).any() else float("nan")
            ),
            "auc_oracle_I_for_inclination": _auc(
                inclination[np.isfinite(inclination)] > 0.5, I_t[np.isfinite(inclination)]
            ),
            # estimate fidelity
            "spearman_est_I_hat_oracle_I": _spearman(est_I, I_t),
            "spearman_est_p_oracle_p": _spearman(est_p, p),
        }
    )
    if offline is not None:
        row.update(offline_ceilings(offline, int(meta["horizon"])))
    return row


def offline_ceilings(offline: pd.DataFrame, horizon: int) -> dict:
    """`auc_oriented` of each `OFFLINE_SCORES` entry for the corpus label at `horizon`
    (``scope == "T=<T>"``, falling back to ``all`` if the horizon has no row)."""
    out: dict = {}
    for score in OFFLINE_SCORES:
        key = f"offline_label_auc_{score}"
        sub = offline[(offline["score"] == score) & (offline["scope"] == f"T={horizon}")]
        if sub.empty:
            sub = offline[(offline["score"] == score) & (offline["scope"] == "all")]
        out[key] = float(sub["auc_oriented"].iloc[0]) if len(sub) else float("nan")
    return out


def decision_by_oracle_bins(on: pd.DataFrame, meta: dict, n_bins: int = N_I_BINS) -> pd.DataFrame:
    """P(SEARCH) and the model's mean probability by decile of oracle `I_t` (within the cell).

    The figure's raw data: a policy that reads the reservoir tail searches more where
    `I_t` is large; a schedule would show a flat profile.
    """
    dec = on["decision_search"].to_numpy(dtype=np.float64)
    I_t = on["oracle_I_t"].to_numpy(dtype=np.float64)
    pm = on["p_search_model"].to_numpy(dtype=np.float64)
    has = np.isfinite(dec) & np.isfinite(I_t)
    if has.sum() < n_bins:
        return pd.DataFrame()
    edges = np.quantile(I_t[has], np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    which = np.clip(np.searchsorted(edges, I_t[has], side="right") - 1, 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        rows.append(
            {
                **meta,
                "bin": b,
                "I_lo": float(edges[b]),
                "I_hi": float(edges[b + 1]),
                "I_mean": float(I_t[has][m].mean()),
                "n": int(m.sum()),
                "search_rate": float(dec[has][m].mean()),
                "model_p_mean": float(np.nanmean(pm[has][m])) if np.isfinite(pm[has][m]).any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)
