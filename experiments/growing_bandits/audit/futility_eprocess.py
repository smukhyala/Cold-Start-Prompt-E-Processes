#!/usr/bin/env python3
"""EXPLORATORY (data already seen): anytime-valid pool-level FUTILITY e-process (E5).

Question: can we certify early that NO challenger beats the incumbent by at least delta, so that
search of the whole candidate pool can stop?  This is the reverse of E1b.

Audit mode: each step reveals one task's outcome for every challenger k and for the incumbent b.

Per challenger k:  D_kt = Y_kt - Y_bt in {-1, 0, 1}.   X_kt = (D_kt + 1) / 2 in [0, 1].
    H0_k : E[D_k] >= delta   <=>   E[X_k] >= m_delta := (1 + delta) / 2.
One-sided DOWNWARD betting e-process:
    K_kt = prod_{s<=t} (1 - lambda_s (X_ks - m_delta)),   lambda_s predictable, lambda_s >= 0.
Under H0_k, E[X - m_delta | past] >= 0, so E[factor | past] <= 1 and K_k is a nonnegative
supermartingale with K_k0 = 1 (an e-process; Ville gives P(sup_t K_kt >= 1/alpha) <= alpha).
Validity needs the tasks to be independent given the past (tasks in random order).

Bet range.  The factor stays positive iff lambda (1 - m_delta) < 1 (worst case X = 1), so the correct
cap is lambda < 1 / (1 - m_delta), NOT 1 / m_delta.  For m_delta >= 1/2 we have 1/m_delta <= 1/(1 - m_delta),
so the grid {0.1, 0.2, 0.4, 0.8} / m_delta is always inside the valid range; we still clip at
0.9 / (1 - m_delta).  Components of the mixture (equal weights, a mixture of e-processes is an e-process):
    0   : aGRAPA bets, a numpy copy of ``HedgedCapitalEProcess(m0=m_delta).log_e_lower``
          (its lower-capital lambda cap is 0.5 / (1 - m0), valid for any m0 in (0, 1), so the repo
          class WAS usable for m0 in 0.525..0.6 and the tests check equality with it)
    1-4 : fixed lambda = {0.1, 0.2, 0.4, 0.8} / m_delta.

Pool futility.  H0_pool = union_k H0_k ("some challenger is at least delta better").  Under H0_pool
some k* has H0_k* true, and min_k E_kt <= E_k*t, so E_pool,t = min_k E_kt is an e-process for the
union null.  Only the per-arm validity at every stopping time is needed (no multiplicity correction,
no independence between arms): for any (data-dependent) stopping time tau,
E[E_pool,tau] <= E[E_k*,tau] <= 1, and Ville bounds P(sup_t E_pool,t >= 1/alpha) <= alpha.
Declare futility at the first t with min_k E_kt >= 1/alpha.

Drop-the-loser view.  Arm k is dropped at the first t with E_kt >= 1/alpha.  Each drop is a valid
level-alpha statement about that arm alone (not a pool-level statement).  Also valid, and weaker
to demand: "every arm has been dropped at some time" implies the arm k* was dropped, so
P(all arms ever dropped) <= alpha as well.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

LAMBDA_NUMERATORS = (0.1, 0.2, 0.4, 0.8)
EPS = 1e-9


def boundary_m(delta: float) -> float:
    return (1.0 + delta) / 2.0


def fixed_lambdas(delta: float) -> list[float]:
    m = boundary_m(delta)
    cap = 0.9 / (1.0 - m)
    out = []
    for num in LAMBDA_NUMERATORS:
        out.append(min(num / m, cap))
    return out


def futility_log_paths(D: np.ndarray, delta: float, return_components: bool = False):
    """Log of the mixture futility e-process for each challenger.

    D has shape (B, K, n) with entries in {-1, 0, 1} (tasks already in replay order).
    Returns log E of shape (B, K, n); entry [:, :, t] is the value after t + 1 tasks (E_0 = 1 is implicit).
    With ``return_components`` also returns the (5, B, K, n) component log paths.
    """
    B, K, n = D.shape
    m = boundary_m(delta)
    X = (D.astype(float) + 1.0) / 2.0
    lams_fixed = fixed_lambdas(delta)
    n_comp = 1 + len(lams_fixed)
    cap_agrapa = 0.5 / max(1.0 - m, EPS)

    log_comp = np.zeros((n_comp, B, K))
    lam_agrapa = np.zeros((B, K))
    sum_x = np.zeros((B, K))
    sum_sq = np.zeros((B, K))
    out = np.zeros((B, K, n))
    comps = np.zeros((n_comp, B, K, n)) if return_components else None
    for t in range(n):
        x = X[:, :, t]
        log_comp[0] += np.log(np.maximum(1.0 - lam_agrapa * (x - m), 1e-300))
        for j in range(len(lams_fixed)):
            log_comp[j + 1] += np.log(np.maximum(1.0 - lams_fixed[j] * (x - m), 1e-300))
        # aGRAPA update (copy of HedgedCapitalEProcess)
        sum_x += x
        mu_hat = (0.5 + sum_x) / (1.0 + (t + 1))
        sum_sq += (x - mu_hat) ** 2
        sigma2 = (0.25 + sum_sq) / (1.0 + (t + 1))
        raw = (m - mu_hat) / np.maximum(sigma2, EPS)
        lam_agrapa = np.minimum(np.maximum(raw, 0.0), cap_agrapa)
        out[:, :, t] = logsumexp(log_comp, axis=0) - math.log(n_comp)
        if return_components:
            comps[:, :, :, t] = log_comp
    if return_components:
        return out, comps
    return out


def first_crossing(log_paths: np.ndarray, alpha: float) -> np.ndarray:
    """First index t (0-based, after t + 1 tasks) with log path >= log(1/alpha); -1 if never.

    log_paths has shape (..., n).
    """
    hit = log_paths >= math.log(1.0 / alpha)
    n = log_paths.shape[-1]
    any_hit = hit.any(axis=-1)
    idx = hit.argmax(axis=-1)
    return np.where(any_hit, idx, -1)


def pool_log_paths(arm_log_paths: np.ndarray) -> np.ndarray:
    """E_pool = min over challengers; (B, K, n) -> (B, n)."""
    return arm_log_paths.min(axis=1)


def episodes_summary(arm_cross: np.ndarray, pool_cross: np.ndarray, n: int) -> dict:
    """Episode counts per replay.

    arm_cross (B, K): first crossing index or -1.  pool_cross (B,): first pool crossing or -1.
    Arm k is evaluated on tasks 0..cross inclusive (or all n); the incumbent on every task.
    Returns arrays (B,) for: full grid, drop-the-loser, pool stop only, drop plus pool stop.
    """
    B, K = arm_cross.shape
    full = np.zeros(B)
    drop = np.zeros(B)
    stop = np.zeros(B)
    both = np.zeros(B)
    for b in range(B):
        evaluated = np.zeros(K)
        for k in range(K):
            if arm_cross[b, k] >= 0:
                evaluated[k] = arm_cross[b, k] + 1
            else:
                evaluated[k] = n
        full[b] = n * (K + 1)
        drop[b] = n + evaluated.sum()
        if pool_cross[b] >= 0:
            horizon = pool_cross[b] + 1
        else:
            horizon = n
        stop[b] = horizon * (K + 1)
        total = horizon
        for k in range(K):
            total = total + min(evaluated[k], horizon)
        both[b] = total
    return {"full": full, "drop": drop, "stop": stop, "both": both}


def boundary_shift(p_task: np.ndarray, gain: float) -> float:
    """Shift s with mean_t(min(p_t + s, 1) - p_t) = gain exactly (bisection)."""
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        val = float(np.mean(np.minimum(p_task + mid, 1.0) - p_task))
        if val < gain:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
