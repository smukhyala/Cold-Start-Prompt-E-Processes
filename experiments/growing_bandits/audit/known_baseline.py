#!/usr/bin/env python3
"""EXPLORATORY (data already seen): KNOWN-INCUMBENT e-processes (B2) and finite-bank helpers (B4).

Known-baseline statistic.  The incumbent's per-task success probability b_t is known (or
estimated once from many prior default runs and reused for every challenger).  For challenger k on
task t:
    X_kt = (Y_kt - b_t + 1) / 2   in [0, 1].
Efficacy null   H0_k: E[Y_k - b] <= 0        <=>  E[X_k] <= 1/2.
Futility null   H0_k: E[Y_k - b] >= delta    <=>  E[X_k] >= (1 + delta) / 2.
If b_t is the TRUE default success probability of task t, E[Y_kt - b_t | past] = gain_k exactly,
so each one-step factor has conditional expectation <= 1 on the null boundary (no incumbent run
noise, no discordance filtering), and the paired design's extra incumbent episode disappears:
K episodes per task instead of K + 1.

Efficacy pool statistic: uniform mixture (average) over challengers of the upward aGRAPA capital
(numpy copy of HedgedCapitalEProcess.log_e_upper with m0 = 1/2, updated on EVERY task).
Futility pool statistic: min over challengers of the E5 mixture (reused from futility_eprocess),
fed D = Y - b = 2X - 1.

Estimated baseline.  With b_hat_t from m prior default runs the plug-in X is exactly unbiased only if
b_hat_t is independent of the task draw EVERY time (fresh task, fresh estimate).  If the bank is
revisited and b_hat_t is estimated once per bank task, the bank-level offset mean_t(q_t - b_hat_t)
is a persistent error shared by every challenger and the plug-in test is not valid.  The
conservative fix shifts b by a per-task Clopper-Pearson bound at level alpha / (2 N) (upper for
efficacy, lower for futility) and runs the test at level alpha / 2.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist

from experiments.growing_bandits.audit import futility_eprocess as fu

EPS = 1e-9


def known_baseline_x(y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """X = (y - b + 1) / 2.  y: (..., n) or (..., K, n); b broadcastable against y."""
    return (y - b + 1.0) / 2.0


# ----------------------------------------------------------------------------
# efficacy
# ----------------------------------------------------------------------------
def efficacy_arm_log_paths(X: np.ndarray, m0: float = 0.5, cap_scale: float = 0.5) -> np.ndarray:
    """Upward aGRAPA log-capital per challenger.  X: (B, K, n) in [0, 1].  Returns (B, K, n).

    cap_scale = 0.5 is the repo convention (lambda <= 0.5 / m0); anything below 1 keeps the factor
    positive for every x in [0, 1].
    """
    B, K, n = X.shape
    cap = cap_scale / max(m0, EPS)
    log_k = np.zeros((B, K))
    lam = np.zeros((B, K))
    sum_x = np.zeros((B, K))
    sum_sq = np.zeros((B, K))
    out = np.zeros((B, K, n))
    for t in range(n):
        x = X[:, :, t]
        log_k = log_k + np.log(np.maximum(1.0 + lam * (x - m0), 1e-300))
        sum_x = sum_x + x
        mu = (0.5 + sum_x) / (1.0 + (t + 1))
        sum_sq = sum_sq + (x - mu) ** 2
        sig2 = (0.25 + sum_sq) / (1.0 + (t + 1))
        lam = np.clip((mu - m0) / np.maximum(sig2, EPS), 0.0, cap)
        out[:, :, t] = log_k
    return out


def efficacy_pool_log_paths(arm_log_paths: np.ndarray) -> np.ndarray:
    """Uniform mixture over challengers (global null).  (B, K, n) -> (B, n)."""
    K = arm_log_paths.shape[1]
    return logsumexp(arm_log_paths, axis=1) - math.log(K)


# ----------------------------------------------------------------------------
# futility
# ----------------------------------------------------------------------------
def futility_pool_log_paths(Yminus_b: np.ndarray, delta: float) -> np.ndarray:
    """Min over challengers of the E5 mixture.  Yminus_b = Y - b in [-1, 1], shape (B, K, n)."""
    arm = fu.futility_log_paths(Yminus_b, delta)
    return fu.pool_log_paths(arm)


# ----------------------------------------------------------------------------
# crossing summaries
# ----------------------------------------------------------------------------
def first_cross_1based(log_paths: np.ndarray, threshold: float) -> np.ndarray:
    """First t (1-based) with log path >= log(threshold); 0 if never.  log_paths (B, n)."""
    hit = log_paths >= math.log(threshold)
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1) + 1
    return np.where(any_hit, first, 0)


def tasks_quantile(cross: np.ndarray, q: float = 0.8) -> float:
    """Smallest n such that the fraction of sims with 0 < cross <= n is >= q; nan if never."""
    B = len(cross)
    times = np.where(cross > 0, cross, np.iinfo(np.int64).max).astype(np.int64)
    times = np.sort(times)
    k = int(math.ceil(q * B)) - 1
    if times[k] == np.iinfo(np.int64).max:
        return float("nan")
    return float(times[k])


# ----------------------------------------------------------------------------
# Clopper-Pearson bounds for the estimated baseline
# ----------------------------------------------------------------------------
def cp_upper(successes: np.ndarray, m: int, a: float) -> np.ndarray:
    """One-sided (1 - a) Clopper-Pearson upper bound; 1 when successes == m."""
    s = np.asarray(successes, dtype=float)
    out = np.ones_like(s)
    inside = s < m
    out[inside] = beta_dist.ppf(1.0 - a, s[inside] + 1.0, m - s[inside])
    return out


def cp_lower(successes: np.ndarray, m: int, a: float) -> np.ndarray:
    """One-sided (1 - a) Clopper-Pearson lower bound; 0 when successes == 0."""
    s = np.asarray(successes, dtype=float)
    out = np.zeros_like(s)
    inside = s > 0
    out[inside] = beta_dist.ppf(a, s[inside], m - s[inside] + 1.0)
    return out


# ----------------------------------------------------------------------------
# B4 helpers: exact finite-bank nulls
# ----------------------------------------------------------------------------
def balanced_null_pair(inc: np.ndarray, chal: np.ndarray, rng: np.random.Generator):
    """Exact-null bank for one (incumbent, challenger) pair.

    Discordant tasks (inc != chal) keep their unordered outcome pair {0, 1} but the challenger / incumbent
    labels are flipped so that exactly half favour the challenger.  If the number of discordant tasks
    is odd, one random discordant task is made concordant first (challenger := incumbent), a one-cell
    edit.  Returns (inc_new, chal_new, n_cells_edited); sum(chal_new - inc_new) == 0 exactly.
    """
    inc_new = np.array(inc, dtype=float)
    chal_new = np.array(chal, dtype=float)
    disc = []
    for t in range(len(inc_new)):
        if inc_new[t] != chal_new[t]:
            disc.append(t)
    edited = 0
    if len(disc) % 2 == 1:
        pick = int(rng.integers(0, len(disc)))
        t_drop = disc.pop(pick)
        chal_new[t_drop] = inc_new[t_drop]
        edited = 1
    order = rng.permutation(len(disc))
    half = len(disc) // 2
    for j in range(len(disc)):
        t = disc[int(order[j])]
        if j < half:
            inc_new[t] = 0.0
            chal_new[t] = 1.0
        else:
            inc_new[t] = 1.0
            chal_new[t] = 0.0
    return inc_new, chal_new, edited


def edit_to_target_sum(inc: np.ndarray, chal: np.ndarray, target: int, rng: np.random.Generator):
    """Minimal number of challenger-cell flips so sum(chal - inc) == target exactly.

    Each flip (chal 0 -> 1 raises the sum by one, 1 -> 0 lowers it by one) changes exactly one cell.
    Returns (chal_new, n_cells_edited).
    """
    chal_new = np.array(chal, dtype=float)
    current = int(round(float(np.sum(chal_new - inc))))
    edits = 0
    while current < target:
        zeros = []
        for t in range(len(chal_new)):
            if chal_new[t] == 0.0:
                zeros.append(t)
        pick = zeros[int(rng.integers(0, len(zeros)))]
        chal_new[pick] = 1.0
        current = current + 1
        edits = edits + 1
    while current > target:
        ones = []
        for t in range(len(chal_new)):
            if chal_new[t] == 1.0:
                ones.append(t)
        pick = ones[int(rng.integers(0, len(ones)))]
        chal_new[pick] = 0.0
        current = current - 1
        edits = edits + 1
    return chal_new, edits
