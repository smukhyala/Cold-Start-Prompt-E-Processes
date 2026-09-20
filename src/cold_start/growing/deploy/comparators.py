"""Attainable, CRN-paired comparators for simple regret.

The old benchmark measured regret against ``essential_sup() = 1.0``, which no arm ever
attains, so every policy carried a large constant offset that hid the differences
between them and made regret incomparable across reservoirs. The comparator used here
is the **full-search oracle**: the best arm among the first ``T`` draws of the
episode's own reservoir stream.

Why this is the right yardstick:

* **Attainable.** Every arm a policy holds at ``T`` is one of the first ``K_T <= T``
  reservoir draws, because each SEARCH costs one pull. So the best discovered arm can
  never beat the comparator and discovery regret ``R_disc = mu*_ep - max_{a in D_T}
  mu_a`` is ``>= 0`` by construction (`tests/test_deploy_harness.py` asserts it).
* **CRN-paired.** The ``k``-th searched arm of *any* policy in a cell is
  ``reservoir.sample_from_uniforms(reservoir_uniforms(base_seed, rep, k))`` -- exactly
  the stream `Simulator.step` and `seed_initial_arms` consume -- so the comparator is a
  per-episode constant shared by every policy, and paired differences cancel it.
* **Bit-consistent.** The simulator stores ``mu`` as float32; the same cast is applied
  here so ``best_discovered <= mu_star`` holds exactly rather than up to rounding.

These functions read the reservoir. They are harness-side only and must never be
reachable from a policy (failure-mode register #4, #11).
"""

from __future__ import annotations

import numpy as np

from cold_start.growing import rng as crn


def episode_reservoir_prefix(
    reservoir, base_seed: int, n_replicates: int, n_draws: int
) -> np.ndarray:
    """True means of the first `n_draws` reservoir draws of each episode, ``(M, n_draws)``.

    Draw ``k`` of replicate ``r`` is the arm any policy in this cell discovers on its
    ``k``-th SEARCH (the warm start consumes draws ``0..n_initial_arms-1``). Values
    pass through float32 exactly as `GrowingState.add_arms` stores them, so an arm
    read back from ``state.mu`` compares bit-equal to its entry here.
    """
    m, k = int(n_replicates), int(n_draws)
    if m < 1 or k < 1:
        raise ValueError(f"need n_replicates >= 1 and n_draws >= 1; got {m}, {k}")
    rep = np.repeat(np.arange(m, dtype=np.int64), k)
    draw = np.tile(np.arange(k, dtype=np.int64), m)
    u = crn.reservoir_uniforms(base_seed, rep, draw)
    mu = np.asarray(reservoir.sample_from_uniforms(u), dtype=np.float32)
    return mu.astype(np.float64).reshape(m, k)


def mu_star_from_prefix(prefix: np.ndarray, cap: int | None = None) -> np.ndarray:
    """``max`` over the first ``min(T, cap)`` columns of an ``(M, T)`` prefix.

    Split out so the harness computes the prefix once per cell and derives both the
    full-search and the cap-feasible comparator from it: the prefix is the expensive
    part (a 60-step bisection per draw for the mixture family).
    """
    p = np.asarray(prefix, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError(f"prefix must be (M, n_draws) with n_draws >= 1; got {p.shape}")
    n = p.shape[1] if cap is None else min(p.shape[1], int(cap))
    if n < 1:
        raise ValueError(f"cap must be >= 1; got {cap}")
    return p[:, :n].max(axis=1)


def mu_star_episode(
    reservoir, base_seed: int, n_replicates: int, horizon: int, cap: int | None = None
) -> np.ndarray:
    """Best true mean among the first ``min(T, cap)`` draws of each episode, ``(M,)``.

    With ``cap=None`` this is the full-search oracle ``mu*_ep = max_{k<T} mu_k``. With
    a live-arm cap a policy can hold at most ``cap`` arms, so ``max_{k<min(T,cap)}`` is
    the tighter, cap-feasible comparator reported alongside it.
    """
    n = int(horizon) if cap is None else min(int(horizon), int(cap))
    return mu_star_from_prefix(episode_reservoir_prefix(reservoir, base_seed, n_replicates, n))


def expected_best_of_n(reservoir, n: int, n_mc: int = 20_000, seed: int = 0) -> float:
    """Monte Carlo ``E[max of n iid draws]``: a cap-free, seed-free reference column.

    Uses the inverse-CDF structure every reservoir family guarantees: because
    `sample_from_uniforms` is monotone, ``max_i F^{-1}(u_i) = F^{-1}(max_i u_i)``, and
    the maximum of ``n`` uniforms is ``U^{1/n}``. So one uniform per Monte Carlo sample
    and one inverse-CDF evaluation replace an ``n_mc x n`` matrix -- the same
    estimator at ``1/n`` the cost, which matters for the mixture family whose inverse
    is a 60-step bisection.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1; got {n_mc}")
    gen = np.random.default_rng(seed)
    u_max = np.power(gen.random(int(n_mc)), 1.0 / float(n))
    draws = np.asarray(reservoir.sample_from_uniforms(u_max), dtype=np.float64)
    return float(draws.mean())
