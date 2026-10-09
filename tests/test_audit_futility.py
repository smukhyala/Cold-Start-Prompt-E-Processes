"""Tests for the EXPLORATORY audit pool-futility e-process (E5)."""

from __future__ import annotations

import math

import numpy as np

from cold_start.inference.hedged_capital import HedgedCapitalEProcess
from experiments.growing_bandits.audit import futility_eprocess as fu


def boundary_dist(delta: float, p_zero: float):
    """P(D=-1), P(D=0), P(D=1) with E[D] = delta exactly."""
    p_plus = (delta + (1.0 - p_zero)) / 2.0
    p_minus = (1.0 - p_zero) - p_plus
    return p_minus, p_zero, p_plus


def test_one_step_factor_expectation_at_most_one_at_boundary():
    for delta in (0.05, 0.10, 0.15, 0.20):
        m = fu.boundary_m(delta)
        for p_zero in (0.0, 0.3, 0.7):
            pm, p0, pp = boundary_dist(delta, p_zero)
            assert abs((pp - pm) - delta) < 1e-12
            lams = fu.fixed_lambdas(delta) + [0.0, 0.5 / (1.0 - m)]
            for lam in lams:
                total = 0.0
                for d, pr in ((-1, pm), (0, p0), (1, pp)):
                    x = (d + 1.0) / 2.0
                    total += pr * (1.0 - lam * (x - m))
                assert abs(total - 1.0) < 1e-12  # exactly 1 at the boundary
                assert lam * (1.0 - m) < 1.0  # factor positive
            # strictly inside the null (better than delta): expectation <= 1
            pm2, p02, pp2 = boundary_dist(delta + 0.1, p_zero)
            for lam in lams:
                total = 0.0
                for d, pr in ((-1, pm2), (0, p02), (1, pp2)):
                    x = (d + 1.0) / 2.0
                    total += pr * (1.0 - lam * (x - m))
                assert total <= 1.0 + 1e-12


def test_initial_e_is_one():
    D = np.zeros((2, 3, 1), dtype=np.int8)
    # a task with D = 0 and delta tiny: e after first task uses lambda_0 = 0 only for aGRAPA;
    # E_0 = 1 means all components start at log 0 and aGRAPA's first bet is 0.
    out, comps = fu.futility_log_paths(D, 0.10, return_components=True)
    assert abs(comps[0, 0, 0, 0]) < 1e-15  # aGRAPA factor at step 1 is exactly 1
    # explicit E_0: a zero-length path would be log 1 = 0; check fixed-lambda product convention
    m = fu.boundary_m(0.10)
    lam = fu.fixed_lambdas(0.10)[0]
    assert abs(comps[1, 0, 0, 0] - math.log(1.0 - lam * (0.5 - m))) < 1e-12


def test_agrapa_component_matches_repo_class():
    rng = np.random.default_rng(0)
    for delta in (0.05, 0.2):
        m = fu.boundary_m(delta)
        D = rng.integers(-1, 2, size=(1, 1, 40)).astype(np.int8)
        out, comps = fu.futility_log_paths(D, delta, return_components=True)
        ref = HedgedCapitalEProcess(m0=m)
        for t in range(40):
            ref.update((float(D[0, 0, t]) + 1.0) / 2.0)
            assert abs(ref.log_e_lower - comps[0, 0, 0, t]) < 1e-9


def test_min_combination_validity_tiny_simulation():
    rng = np.random.default_rng(5)
    delta = 0.10
    sims, K, n = 2000, 3, 120
    # arm 0 exactly at the boundary, arms 1-2 strictly better than the incumbent by 0.4
    # (so E_pool = min is driven by arm 0).  H0_pool is true.  Also arm 1 is slightly below delta.
    probs = []
    probs.append(boundary_dist(delta, 0.5))
    probs.append(boundary_dist(0.4, 0.3))
    probs.append(boundary_dist(0.0, 0.5))
    D = np.zeros((sims, K, n), dtype=np.int8)
    for k in range(K):
        pm, p0, pp = probs[k]
        u = rng.random((sims, n))
        D[:, k, :] = np.where(u < pm, -1, np.where(u < pm + p0, 0, 1))
    logp = fu.futility_log_paths(D, delta)
    pool_cross = fu.first_crossing(fu.pool_log_paths(logp), 0.05)
    rate = float((pool_cross >= 0).mean())
    assert rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / sims)
    # and the flat pool (all D=0, delta = 0.2) is certified within 120 tasks most of the time
    Dflat = np.zeros((50, 3, 120), dtype=np.int8)
    cross = fu.first_crossing(fu.pool_log_paths(fu.futility_log_paths(Dflat, 0.2)), 0.05)
    assert (cross >= 0).all()


def test_boundary_shift_exact():
    p = np.array([0.1, 0.5, 0.95, 0.99])
    s = fu.boundary_shift(p, 0.10)
    assert abs(float(np.mean(np.minimum(p + s, 1.0) - p)) - 0.10) < 1e-9
