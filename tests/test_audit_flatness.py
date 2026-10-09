"""Tests for the EXPLORATORY audit flatness / challenger e-processes (E1)."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from experiments.growing_bandits.audit import flatness_eprocess as fe


def test_log_elem_sym_matches_brute_force():
    rng = np.random.default_rng(0)
    for K in (1, 3, 6):
        lw = rng.normal(0, 1.2, size=K)
        dp = fe.log_elem_sym(lw[None, :])[0]
        bf = fe.log_elem_sym_brute(lw)
        assert np.allclose(dp, bf)


def test_factor_expectation_one_under_conditional_null():
    K = 5
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1.0, size=K)
    z = z - z.mean()
    for S in range(K + 1):
        factors = []
        for subset in combinations(range(K), S):
            y = np.zeros(K)
            for idx in subset:
                y[idx] = 1.0
            lf = fe.e1a_log_factors(y[None, :], z[None, :])[0]
            factors.append(np.exp(lf))
        factors = np.array(factors)
        assert np.allclose(factors.mean(axis=0), 1.0)


def test_initial_e_is_one_and_stream_matches_batch():
    K = 6
    ep = fe.FlatnessEProcess(K)
    assert ep.log_e == 0.0
    rng = np.random.default_rng(2)
    Y = (rng.random((1, K, 12)) < 0.5).astype(float)
    for t in range(12):
        ep.update(Y[0, :, t])
    path = fe.e1a_log_paths(Y)
    assert abs(ep.log_e - path[0, -1]) < 1e-9


def test_e1a_null_type1_and_power():
    rng = np.random.default_rng(3)
    K, n, B = 6, 30, 600
    p_task = rng.uniform(0.3, 0.8, size=n)
    Y = (rng.random((B, K, n)) < p_task[None, None, :]).astype(int)
    rej, _, _, _ = fe.summarize_paths(fe.e1a_log_paths(Y), 0.05)
    assert rej.mean() <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / B)
    p_arm = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.2])
    Y2 = (rng.random((200, K, 60)) < p_arm[None, :, None]).astype(int)
    rej2, _, _, _ = fe.summarize_paths(fe.e1a_log_paths(Y2), 0.05)
    assert rej2.mean() > 0.8


def test_e1b_batch_matches_repo_classes():
    rng = np.random.default_rng(4)
    A, n = 7, 25
    Y = (rng.random((A, n)) < np.linspace(0.3, 0.8, A)[:, None]).astype(int)
    inc = 2
    chal_rows = [0, 1, 3, 4, 5, 6]
    ref = fe.e1b_log_path_reference(Y, inc, chal_rows)
    chal = np.zeros((1, A), dtype=bool)
    for k in chal_rows:
        chal[0, k] = True
    got = fe.e1b_log_paths(Y[None, :, :], np.array([inc]), chal)[0]
    assert np.allclose(ref, got, atol=1e-9)


def test_e1b_null_and_planted_arm():
    rng = np.random.default_rng(5)
    A, n, B = 8, 40, 600
    Y = (rng.random((B, A, n)) < 0.5).astype(int)
    inc = rng.integers(0, A, size=B)
    chal = np.ones((B, A), dtype=bool)
    chal[np.arange(B), inc] = False
    rej, _, _, _ = fe.summarize_paths(fe.e1b_log_paths(Y, inc, chal), 0.05)
    assert rej.mean() <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / B)
    Y2 = (rng.random((200, A, 60)) < 0.5).astype(int)
    Y2[:, 0, :] = (rng.random((200, 60)) < 0.9).astype(int)
    inc2 = np.full(200, 1)
    chal2 = np.ones((200, A), dtype=bool)
    chal2[:, 1] = False
    rej2, _, _, _ = fe.summarize_paths(fe.e1b_log_paths(Y2, inc2, chal2), 0.05)
    assert rej2.mean() > 0.9
