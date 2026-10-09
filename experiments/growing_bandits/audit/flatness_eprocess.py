#!/usr/bin/env python3
"""EXPLORATORY (data already seen): anytime-valid "is this pool worth searching?" tests.

Audit mode: each step reveals one task's outcomes for all K arms (cost K episodes).

E1a (homogeneity).  H0: within each task, prompt labels are exchangeable (no
prompt effect; task difficulty arbitrary).  Conditional on S_t = sum_k y_tk the
success set is uniform over the C(K, S_t) subsets.  The alternative is Fisher's
multivariate noncentral hypergeometric with odds omega_k = exp(eta * z_k) where
z_k is a predictable (past-only) logit score.  The e-factor is
    C(K,S) * prod_{k in success set} omega_k / e_S(omega)
and E_t is the uniform mixture over eta of the running products.

E1b ("does any prompt beat the incumbent?").  For challenger k vs incumbent b,
on discordant tasks X = 1{k succeeds, b fails}; H0_k: P(X=1 | disc, past) <= 1/2.
One-sided hedged-capital (aGRAPA) upper capital K^+ per challenger, uniform
mixture over challengers for the global null.  `e1b_log_paths` is a numpy
re-implementation of HedgedCapitalEProcess.log_e_upper (m0=0.5) batched over
simulations; tests check it against the repo class + GlobalNullEProcess.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.special import gammaln, logsumexp

ETAS = (0.25, 0.5, 1.0, 2.0, 4.0)


# ----------------------------------------------------------------------------
# E1a
# ----------------------------------------------------------------------------
def log_elem_sym(log_omega: np.ndarray) -> np.ndarray:
    """log e_j(omega) for j=0..K, DP in log-space.

    log_omega has shape (..., K); returns shape (..., K+1).
    """
    shape = log_omega.shape
    K = shape[-1]
    e = np.full(shape[:-1] + (K + 1,), -np.inf)
    e[..., 0] = 0.0
    for k in range(K):
        shifted = np.full(e.shape, -np.inf)
        shifted[..., 1:] = e[..., :-1] + log_omega[..., k : k + 1]
        e = np.logaddexp(e, shifted)
    return e


def log_elem_sym_brute(log_omega: np.ndarray) -> np.ndarray:
    """Brute-force log e_j for a 1-D vector (test reference)."""
    K = len(log_omega)
    out = np.full(K + 1, -np.inf)
    for j in range(K + 1):
        total = 0.0
        for subset in combinations(range(K), j):
            prod = 1.0
            for idx in subset:
                prod = prod * math.exp(log_omega[idx])
            total = total + prod
        if total > 0.0:
            out[j] = math.log(total)
    return out


def predictable_scores(succ_past: np.ndarray, m: int) -> np.ndarray:
    """z_k = logit((s_k+1)/(m+2)) centred over arms.  succ_past: (..., K)."""
    p = (succ_past + 1.0) / (m + 2.0)
    z = np.log(p) - np.log1p(-p)
    return z - z.mean(axis=-1, keepdims=True)


def e1a_log_factors(y: np.ndarray, z: np.ndarray, etas=ETAS) -> np.ndarray:
    """log e-factor for each eta.  y, z: (B, K); returns (B, len(etas))."""
    B, K = y.shape
    etas_arr = np.asarray(etas, dtype=float)
    S = y.sum(axis=1).astype(int)
    log_omega = etas_arr[None, :, None] * z[:, None, :]  # (B, E, K)
    e = log_elem_sym(log_omega)  # (B, E, K+1)
    idx = np.broadcast_to(S[:, None, None], (B, len(etas_arr), 1))
    log_eS = np.take_along_axis(e, idx, axis=2)[:, :, 0]
    log_choose = gammaln(K + 1) - gammaln(S + 1) - gammaln(K - S + 1)
    num = etas_arr[None, :] * (y * z).sum(axis=1)[:, None]
    out = log_choose[:, None] + num - log_eS
    trivial = (S == 0) | (S == K)
    out[trivial, :] = 0.0
    return out


def e1a_log_paths(Y: np.ndarray, etas=ETAS) -> np.ndarray:
    """Y: (B, K, n) tasks already in replay order.  Returns log E_t, shape (B, n)."""
    B, K, n = Y.shape
    cum = np.zeros((B, len(etas)))
    succ = np.zeros((B, K))
    out = np.zeros((B, n))
    for t in range(n):
        z = predictable_scores(succ, t)
        y = Y[:, :, t].astype(float)
        cum = cum + e1a_log_factors(y, z, etas)
        succ = succ + y
        out[:, t] = logsumexp(cum, axis=1) - math.log(len(etas))
        # note: out[:, t] = log E_{t+1} in 1-based task count
    return out


class FlatnessEProcess:
    """Sequential (one task at a time) interface to E1a; E_0 = 1."""

    def __init__(self, K: int, etas=ETAS):
        self.K = K
        self.etas = tuple(etas)
        self._cum = np.zeros(len(self.etas))
        self._succ = np.zeros(K)
        self.m = 0

    @property
    def log_e(self) -> float:
        return float(logsumexp(self._cum) - math.log(len(self.etas)))

    def update(self, y: np.ndarray) -> float:
        yv = np.asarray(y, dtype=float)[None, :]
        z = predictable_scores(self._succ[None, :], self.m)
        self._cum = self._cum + e1a_log_factors(yv, z, self.etas)[0]
        self._succ = self._succ + yv[0]
        self.m += 1
        return self.log_e


# ----------------------------------------------------------------------------
# E1b
# ----------------------------------------------------------------------------
def e1b_log_paths(Y: np.ndarray, inc: np.ndarray, chal: np.ndarray, m0: float = 0.5,
                  eps: float = 1e-9) -> np.ndarray:
    """Global mixture log E_t of one-sided hedged capital K^+ over challengers.

    Y:    (B, A, n) 0/1 outcomes, tasks in replay order (A rows incl. incumbent).
    inc:  (B,) row index of the incumbent in each simulation.
    chal: (B, A) bool, which rows are challengers.
    Returns (B, n).  Mirrors HedgedCapitalEProcess (update only on discordant tasks).
    """
    B, A, n = Y.shape
    cap = 0.5 / max(m0, eps)
    log_kp = np.zeros((B, A))
    sum_x = np.zeros((B, A))
    sum_sq = np.zeros((B, A))
    cnt = np.zeros((B, A))
    lam = np.zeros((B, A))
    n_chal = chal.sum(axis=1).astype(float)
    rows = np.arange(B)
    out = np.zeros((B, n))
    for t in range(n):
        x = Y[:, :, t].astype(float)
        yb = x[rows, inc][:, None]
        disc = (x != yb) & chal
        term = 1.0 + lam * (x - m0)
        log_kp = log_kp + np.where(disc, np.log(np.maximum(term, 1e-300)), 0.0)
        cnt_new = cnt + 1.0
        sum_x_new = sum_x + x
        mu = (0.5 + sum_x_new) / (1.0 + cnt_new)
        sum_sq_new = sum_sq + (x - mu) ** 2
        sig2 = (0.25 + sum_sq_new) / (1.0 + cnt_new)
        lam_new = np.clip((mu - m0) / np.maximum(sig2, eps), 0.0, cap)
        cnt = np.where(disc, cnt_new, cnt)
        sum_x = np.where(disc, sum_x_new, sum_x)
        sum_sq = np.where(disc, sum_sq_new, sum_sq)
        lam = np.where(disc, lam_new, lam)
        masked = np.where(chal, log_kp, -np.inf)
        out[:, t] = logsumexp(masked, axis=1) - np.log(n_chal)
    return out


def e1b_log_path_reference(Y: np.ndarray, inc: int, chal_rows) -> np.ndarray:
    """Reference using the repo's HedgedCapitalEProcess + GlobalNullEProcess.

    Y: (A, n).  Returns (n,) log E_t.
    """
    from cold_start.inference.global_null import GlobalNullEProcess
    from cold_start.inference.hedged_capital import HedgedCapitalEProcess

    class _Upper:
        def __init__(self, ep):
            self.ep = ep

        @property
        def log_e(self):
            return self.ep.log_e_upper

    A, n = Y.shape
    procs = []
    for k in chal_rows:
        procs.append(HedgedCapitalEProcess(m0=0.5))
    wrapped = []
    for p in procs:
        wrapped.append(_Upper(p))
    glob = GlobalNullEProcess(combine="mixture")
    out = np.zeros(n)
    for t in range(n):
        for i in range(len(chal_rows)):
            k = chal_rows[i]
            if Y[k, t] != Y[inc, t]:
                procs[i].update(float(Y[k, t]))
        out[t] = glob.log_e(wrapped)
    return out


# ----------------------------------------------------------------------------
# replay helpers
# ----------------------------------------------------------------------------
def summarize_paths(log_paths: np.ndarray, alpha: float):
    """Per-order (rejected, stop_index, final, max).  stop_index is 1-based, n if never."""
    B, n = log_paths.shape
    thresh = math.log(1.0 / alpha)
    hit = log_paths >= thresh
    rejected = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    stop = np.where(rejected, first, n)
    return rejected, stop, log_paths[:, -1], log_paths.max(axis=1)
