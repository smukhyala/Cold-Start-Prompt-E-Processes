"""Tests for the EXPLORATORY known-incumbent e-processes (B2) and finite-bank helpers (B4)."""

from __future__ import annotations

import math

import numpy as np

from cold_start.inference.hedged_capital import HedgedCapitalEProcess
from experiments.growing_bandits.audit import futility_eprocess as fu
from experiments.growing_bandits.audit import known_baseline as kb


def one_step_expectation(q: float, gain: float, lam: float, sign: float, m: float) -> float:
    """E[1 + sign * lam * (X - m)] with X = (Y - q + 1) / 2, Y ~ Bern(q + gain)."""
    p = q + gain
    total = 0.0
    for y, pr in ((1.0, p), (0.0, 1.0 - p)):
        x = (y - q + 1.0) / 2.0
        total = total + pr * (1.0 + sign * lam * (x - m))
    return total


def test_one_step_expectation_at_most_one_at_null_boundary():
    for q in (0.05, 0.3, 0.62, 0.95):
        # efficacy boundary: gain 0, upward bet, m0 = 1/2, lambda up to the aGRAPA cap
        for lam in (0.0, 0.3, 1.0):
            assert one_step_expectation(q, 0.0, lam, +1.0, 0.5) <= 1.0 + 1e-12
        # futility boundary: gain delta, downward bet, m = (1 + delta) / 2
        for delta in (0.10, 0.15):
            if q + delta > 1.0:
                continue
            m = (1.0 + delta) / 2.0
            lams = fu.fixed_lambdas(delta) + [0.0, 0.5 / (1.0 - m)]
            for lam in lams:
                assert one_step_expectation(q, delta, lam, -1.0, m) <= 1.0 + 1e-12


def test_efficacy_numpy_matches_repo_hedged_capital():
    rng = np.random.default_rng(0)
    x = rng.random(40)
    ref = HedgedCapitalEProcess(m0=0.5)
    ref_path = []
    for t in range(len(x)):
        ref.update(float(x[t]))
        ref_path.append(ref.log_e_upper)
    mine = kb.efficacy_arm_log_paths(x[None, None, :])[0, 0]
    assert np.allclose(mine, np.array(ref_path))


def test_efficacy_null_rate_below_alpha_infinite_tasks():
    rng = np.random.default_rng(1)
    q = rng.uniform(0.1, 0.9, size=30)
    B = 300
    n = 150
    idx = rng.integers(0, 30, size=(B, 3, n))
    y = (rng.random((B, 3, n)) < q[idx]).astype(float)
    X = kb.known_baseline_x(y, q[idx])
    lp = kb.efficacy_pool_log_paths(kb.efficacy_arm_log_paths(X))
    rate = float(np.mean(kb.first_cross_1based(lp, 20.0) > 0))
    assert rate <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / B)


def test_conservative_cp_shift_is_valid_and_plugin_is_not_in_bank():
    rng = np.random.default_rng(2)
    N = 30
    q = rng.uniform(0.1, 0.9, size=N)
    m = 3
    B = 400
    n = 200
    alpha = 0.05
    s = rng.binomial(m, q[None, :], size=(B, N)).astype(float)
    plug = s / m
    cons = kb.cp_upper(s, m, alpha / (2.0 * N))
    assert np.all(cons >= plug - 1e-12)
    idx = rng.integers(0, N, size=(B, 2, n))
    y = (rng.random((B, 2, n)) < q[idx]).astype(float)
    rates = {}
    for name, bank, thr in (("plug", plug, 20.0), ("cons", cons, 40.0)):
        b_used = np.take_along_axis(bank, idx[:, 0, :], axis=1)
        X = kb.known_baseline_x(y, b_used[:, None, :])
        lp = kb.efficacy_pool_log_paths(kb.efficacy_arm_log_paths(X))
        rates[name] = float(np.mean(kb.first_cross_1based(lp, thr) > 0))
    assert rates["cons"] <= alpha
    assert rates["plug"] > rates["cons"]


def test_cp_bounds_bracket_truth_and_edges():
    s = np.array([0.0, 3.0, 10.0])
    up = kb.cp_upper(s, 10, 0.01)
    lo = kb.cp_lower(s, 10, 0.01)
    assert up[2] == 1.0 and lo[0] == 0.0
    assert np.all(up >= s / 10.0) and np.all(lo <= s / 10.0)


def test_label_flip_null_preserves_per_task_outcomes_and_zero_mean():
    rng = np.random.default_rng(3)
    inc = (rng.random(40) < 0.5).astype(float)
    chal = (rng.random(40) < 0.5).astype(float)
    for _ in range(20):
        i2, c2, edited = kb.balanced_null_pair(inc, chal, rng)
        assert float(np.sum(c2 - i2)) == 0.0
        diffs = 0
        for t in range(40):
            same_multiset = sorted([i2[t], c2[t]]) == sorted([inc[t], chal[t]])
            if not same_multiset:
                diffs = diffs + 1
        assert diffs == edited
        assert edited <= 1


def test_edit_to_target_sum_is_exact_and_minimal():
    rng = np.random.default_rng(4)
    inc = (rng.random(50) < 0.5).astype(float)
    chal = (rng.random(50) < 0.5).astype(float)
    start = int(round(float(np.sum(chal - inc))))
    for target in (0, 5, 8):
        new, edits = kb.edit_to_target_sum(inc, chal, target, rng)
        assert int(round(float(np.sum(new - inc)))) == target
        assert edits == abs(target - start)
        assert int(np.sum(new != chal)) == edits


def test_tasks_quantile_and_cross():
    lp = np.array([[0.0, 4.0, 4.0], [0.0, 0.0, 0.0]])
    cross = kb.first_cross_1based(lp, 20.0)
    assert list(cross) == [2, 0]
    assert math.isnan(kb.tasks_quantile(cross, 0.8))
    assert kb.tasks_quantile(np.array([3, 5, 2, 9, 4]), 0.8) == 5.0
