"""Tests for the EXPLORATORY B5 combined adaptive + known-baseline designs."""

from __future__ import annotations

import numpy as np

from experiments.growing_bandits.audit import adaptive_tests as ad
from experiments.growing_bandits.audit import combined_tests as ct
from experiments.growing_bandits.audit import futility_eprocess as fu


def test_combined_futility_boundary_null_validity():
    q = ad.load_task_difficulty()
    K = 6
    delta = 0.15
    B = 600
    rng = np.random.default_rng(5)
    shifts, _ = ad.build_shifts("boundary", K, B, delta, rng, q_grid=q)
    res = ct.simulate_known_futility("drop", K, shifts, q, delta, 6000, 0.05, rng)
    rate = float(np.mean(np.isfinite(res["ep_cert"])))
    # alpha = 0.05; generous Monte Carlo slack for a smoke test
    assert rate <= 0.05 + 4.0 * np.sqrt(0.05 * 0.95 / B)


def test_combined_futility_bank_boundary_null_validity():
    q = ad.load_task_difficulty()
    K = 6
    delta = 0.15
    B = 600
    rng = np.random.default_rng(6)
    bank = ct.make_bank(q, 200, rng)
    s = fu.boundary_shift(bank, delta)
    shifts = np.zeros((B, K))
    for b in range(B):
        shifts[b, rng.integers(K)] = s
    res = ct.simulate_known_futility("drop", K, shifts, bank, delta, 6000, 0.05, rng,
                                     baseline="bank", m=30)
    rate = float(np.mean(np.isfinite(res["ep_cert"])))
    assert rate <= 0.05 + 4.0 * np.sqrt(0.05 * 0.95 / B)


def test_dropped_arms_never_reactivate():
    q = ad.load_task_difficulty()
    K = 8
    B = 40
    rng = np.random.default_rng(7)
    shifts = np.zeros((B, K))
    for b in range(B):
        for k in range(K // 2):
            shifts[b, k] = -0.15
    res = ct.simulate_known_futility("drop", K, shifts, q, 0.15, 4000, 0.05, rng, trace=True)
    hist = res["active_history"]
    assert len(hist) > 5
    saw_drop = False
    for t in range(1, len(hist)):
        newly_active = hist[t] & ~hist[t - 1]
        assert not newly_active.any()
        if (hist[t - 1] & ~hist[t]).any():
            saw_drop = True
    assert saw_drop


def test_futility_episode_accounting_no_incumbent():
    q = ad.load_task_difficulty()
    K = 5
    B = 30
    rng = np.random.default_rng(8)
    shifts = np.zeros((B, K))
    res = ct.simulate_known_futility("drop", K, shifts, q, 0.15, 3000, 0.05, rng, trace=True)
    hist = res["active_history"]
    # replay the accounting: episodes = sum over steps of the active count at the START of the step
    active_before = np.ones((B, K), dtype=bool)
    total = np.zeros(B)
    for t in range(len(hist)):
        total = total + active_before.sum(axis=1) * (res["steps_used"] > t)
        active_before = hist[t]
    assert np.array_equal(total, res["ep_used"])
    assert np.all(res["incumbent_episodes"] == 0)
    # known-baseline-only: exactly K episodes per task
    res2 = ct.simulate_known_futility("all", K, shifts, q, 0.15, 3000, 0.05, rng)
    assert np.array_equal(res2["ep_used"], res2["steps_used"] * K)
    assert np.all(res2["incumbent_episodes"] == 0)


def test_efficacy_one_episode_per_step_and_null_validity():
    q = ad.load_task_difficulty()
    K = 6
    B = 400
    rng = np.random.default_rng(9)
    shifts = np.zeros((B, K))
    res = ct.simulate_known_efficacy("thompson", K, shifts, q, 3000, 0.05, rng)
    assert res["episodes_per_step"] == 1
    assert res["incumbent_episodes"] == 0
    fin = res["ep_rej"][np.isfinite(res["ep_rej"])]
    assert np.array_equal(fin, res["tasks_rej"][np.isfinite(res["tasks_rej"])])
    assert float(np.mean(np.isfinite(res["ep_rej"]))) <= 0.05 + 4.0 * np.sqrt(0.05 * 0.95 / B)
    res2 = ct.simulate_known_efficacy("all", K, shifts, q, 3000, 0.05, rng)
    assert res2["episodes_per_step"] == K


def test_unpulled_efficacy_arms_frozen():
    rng = np.random.default_rng(10)
    B, K = 10, 4
    st = ct.KnownEfficacyState(B, K)
    for _ in range(10):
        pulled = np.zeros((B, K), dtype=bool)
        pulled[:, 0] = True
        st.update(pulled, rng.random((B, K)))
    snap = st.log_kp[:, 1:].copy()
    pulled = np.zeros((B, K), dtype=bool)
    pulled[:, 0] = True
    st.update(pulled, rng.random((B, K)))
    assert np.array_equal(snap, st.log_kp[:, 1:])
