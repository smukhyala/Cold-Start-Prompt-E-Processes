"""Tests for the EXPLORATORY adaptive B1/B3 pool tests."""

from __future__ import annotations

import numpy as np

from experiments.growing_bandits.audit import adaptive_tests as ad
from experiments.growing_bandits.audit import flatness_eprocess as fl
from experiments.growing_bandits.audit import futility_eprocess as fu


def test_global_average_is_one_at_start():
    st = ad.EfficacyState(5, 12)
    assert np.allclose(st.log_global(), 0.0)


def test_unsampled_arm_capital_never_changes():
    rng = np.random.default_rng(0)
    B, K = 20, 6
    st = ad.EfficacyState(B, K)
    rows = np.arange(B)
    # warm up: pull arm 1 a few times so it has non-trivial state
    for _ in range(15):
        x = (rng.random((B, K)) < 0.6).astype(float)
        y = (rng.random(B) < 0.4).astype(float)
        pulled = np.zeros((B, K), dtype=bool)
        pulled[:, 1] = True
        st.update(pulled, x, y)
    snap_log = st.log_kp.copy()
    snap_lam = st.lam.copy()
    snap_cnt = st.cnt.copy()
    assert np.abs(snap_log[:, 1]).max() > 0.0
    for _ in range(40):
        x = (rng.random((B, K)) < 0.7).astype(float)
        y = (rng.random(B) < 0.3).astype(float)
        arm = rng.choice([0, 2, 3, 4, 5], size=B)
        pulled = np.zeros((B, K), dtype=bool)
        pulled[rows, arm] = True
        st.update(pulled, x, y)
    assert np.array_equal(st.log_kp[:, 1], snap_log[:, 1])
    assert np.array_equal(st.lam[:, 1], snap_lam[:, 1])
    assert np.array_equal(st.cnt[:, 1], snap_cnt[:, 1])


def test_all_pulled_matches_e1b_helper():
    rng = np.random.default_rng(1)
    B, K, n = 8, 5, 60
    Y = (rng.random((B, K + 1, n)) < 0.5).astype(int)
    inc = np.zeros(B, dtype=int)
    chal = np.ones((B, K + 1), dtype=bool)
    chal[:, 0] = False
    ref = fl.e1b_log_paths(Y, inc, chal)
    st = ad.EfficacyState(B, K)
    for t in range(n):
        st.update(np.ones((B, K), dtype=bool), Y[:, 1:, t].astype(float), Y[:, 0, t].astype(float))
        assert np.allclose(st.log_global(), ref[:, t], atol=1e-9)


def test_futility_state_matches_e5_helper():
    rng = np.random.default_rng(2)
    B, K, n = 6, 4, 50
    D = rng.integers(-1, 2, size=(B, K, n)).astype(np.int8)
    for delta in (0.10, 0.15):
        ref = fu.futility_log_paths(D, delta)
        st = ad.FutilityState(B, K, delta)
        for t in range(n):
            st.update(np.ones((B, K), dtype=bool), (D[:, :, t] + 1.0) / 2.0)
            assert np.allclose(st.log_e(), ref[:, :, t], atol=1e-9)


def test_validity_smoke_small_null_sims():
    q = ad.load_task_difficulty()
    K, B = 6, 400
    for scenario in ("null", "below"):
        for rule in ("greedy", "thompson", "audit"):
            rng = np.random.default_rng(7)
            shifts, _ = ad.build_shifts(scenario, K, B, 0.1, rng)
            out = ad.simulate_b1(rule, K, shifts, q, 600, 0.05, rng)
            rate = ad.rate_at(out["ep_rej"], 600)
            assert rate <= 0.05 + 0.03, (scenario, rule, rate)


def test_power_smoke_good_arm_rejects():
    q = ad.load_task_difficulty()
    rng = np.random.default_rng(3)
    shifts, good = ad.build_shifts("good1", 4, 100, 0.3, rng)
    out = ad.simulate_b1("thompson", 4, shifts, q, 6000, 0.05, rng)
    assert ad.rate_at(out["ep_rej"], 6000) > 0.5


def test_drop_the_loser_never_reactivates():
    q = ad.load_task_difficulty()
    rng = np.random.default_rng(5)
    K, B = 5, 50
    shifts, _ = ad.build_shifts("below", K, B, 0.1, rng)
    for rule in ("drop", "focus"):
        out = ad.simulate_b3(rule, K, shifts, q, 0.15, 3000, 0.05, rng, trace=True)
        hist = out["active_history"]
        assert len(hist) > 5
        dropped_any = False
        for i in range(1, len(hist)):
            assert not (hist[i] & ~hist[i - 1]).any()
            if (hist[i - 1] & ~hist[i]).any():
                dropped_any = True
        assert dropped_any


def test_b3_validity_smoke_boundary_arm():
    q = ad.load_task_difficulty()
    rng = np.random.default_rng(11)
    K, B = 4, 300
    shifts, _ = ad.build_shifts("boundary", K, B, 0.15, rng, q_grid=q)
    for rule in ("audit", "drop", "focus"):
        out = ad.simulate_b3(rule, K, shifts, q, 0.15, 3000, 0.05, rng)
        assert ad.rate_at(out["ep_cert"], 3000) <= 0.05 + 0.03
