"""Tests for the vectorized growing-bandit state and its common-random-number layer."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing import rng as crn
from cold_start.growing.state import EMPTY_UID, AgrapaParams, GrowingState
from cold_start.inference.upward_capital import UpwardCapitalEProcess


class _StubTable:
    """Minimal BoundTable: bounds that depend only on (n, S), enough to exercise state."""

    def __init__(self, horizon: int) -> None:
        self.stride = horizon + 1
        size = self.stride * self.stride
        n_ax = np.arange(self.stride)[:, None]
        s_ax = np.arange(self.stride)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n_ax > 0, s_ax / np.maximum(n_ax, 1), 0.5)
            half = np.where(n_ax > 0, 1.0 / np.sqrt(np.maximum(n_ax, 1)), 1.0)
        self.lower_flat = np.clip(mean - half, 0.0, 1.0).astype(np.float32).reshape(size)
        self.upper_flat = np.clip(mean + half, 0.0, 1.0).astype(np.float32).reshape(size)


# ---- splitmix64 / CRN ---------------------------------------------------------


def test_splitmix64_matches_reference_vectors():
    """Bit-exactness against the published splitmix64 output for seed 0."""
    out = crn.splitmix64(np.zeros(1, dtype=np.uint64))
    assert int(out[0]) == 0xE220A8397B1DCDAF, f"got {int(out[0]):#x}"


def test_splitmix64_rejects_non_uint64():
    """numpy 2.x silently promotes uint64 + int64 to float64; that must fail loudly."""
    with pytest.raises(TypeError):
        crn.splitmix64(np.arange(4, dtype=np.int64))


def test_uniforms_are_uniform():
    key = crn.arm_key(1234, np.zeros(200_000, dtype=np.int64), np.arange(200_000))
    u = crn.uniform_u53(key, np.zeros(200_000, dtype=np.uint64)).astype(np.float64) / float(
        crn.UNIFORM_SCALE
    )
    assert 0.0 <= u.min() and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.005, f"mean={u.mean():.5f}"


def test_reward_stream_is_keyed_by_uid_not_slot():
    """The core CRN invariant.

    The j-th new arm occupies a different slot in the SEARCH branch than in REFINE.
    If rewards were keyed by slot the two branches would silently decouple, so the
    same (replicate, uid, pull-index) must give the same draw regardless of slot.
    """
    reps = np.arange(64)
    uid = np.full(64, 7)
    thr = crn.mu_to_threshold(np.full(64, 0.4))
    a = crn.bernoulli(crn.arm_key(99, reps, uid), np.full(64, 3, dtype=np.uint64), thr)
    b = crn.bernoulli(crn.arm_key(99, reps, uid), np.full(64, 3, dtype=np.uint64), thr)
    assert np.array_equal(a, b)
    other = crn.bernoulli(crn.arm_key(99, reps, np.full(64, 8)), np.full(64, 3, dtype=np.uint64), thr)
    assert not np.array_equal(a, other), "different uids must give different streams"


def test_tiebreak_is_deterministic_and_tiny():
    eps = crn.tiebreak_epsilon(5, np.arange(100), np.arange(100))
    again = crn.tiebreak_epsilon(5, np.arange(100), np.arange(100))
    assert np.array_equal(eps, again)
    assert eps.min() >= 0.0 and eps.max() <= 1e-6


# ---- state --------------------------------------------------------------------


def test_add_arms_and_capacity_growth_preserve_data():
    s = GrowingState(n_replicates=8, capacity=1, horizon=20, base_seed=3)
    for j in range(5):
        s.add_arms(np.full(8, 0.1 * (j + 1), dtype=np.float32), np.full(8, j, dtype=np.int32))
    assert s.Kmax >= 5
    assert np.all(s.Kt == 5)
    mu2d = s.view(s.mu)
    for j in range(5):
        assert np.allclose(mu2d[:, j], 0.1 * (j + 1), atol=1e-6), f"arm {j} lost on regrow"


def test_unpulled_arm_has_trivial_bounds():
    s = GrowingState(n_replicates=4, capacity=4, horizon=20, base_seed=1)
    s.add_arms(np.full(4, 0.5, dtype=np.float32), np.zeros(4, dtype=np.int32))
    assert np.allclose(s.view(s.lcb)[:, 0], 0.0)
    assert np.allclose(s.view(s.ucb)[:, 0], 1.0)


def test_inactive_slots_keep_empty_uid():
    s = GrowingState(n_replicates=4, capacity=4, horizon=20, base_seed=1)
    s.add_arms(np.full(4, 0.5, dtype=np.float32), np.zeros(4, dtype=np.int32))
    assert np.all(s.view(s.uid)[:, 1:] == EMPTY_UID)


def test_clone_is_independent():
    s = GrowingState(n_replicates=4, capacity=4, horizon=20, base_seed=1)
    s.add_arms(np.full(4, 0.5, dtype=np.float32), np.zeros(4, dtype=np.int32))
    tab = _StubTable(20)
    c = s.clone()
    c.pull(np.zeros(4, dtype=np.int64), tab)
    assert np.all(s.n == 0), "mutating the clone changed the original"
    assert np.all(c.n[c.row_off] == 1)


def test_pull_counts_and_successes_are_consistent():
    s = GrowingState(n_replicates=256, capacity=2, horizon=200, base_seed=11)
    s.add_arms(np.full(256, 0.7, dtype=np.float32), np.zeros(256, dtype=np.int32))
    tab = _StubTable(200)
    cols = np.zeros(256, dtype=np.int64)
    for _ in range(100):
        s.pull(cols, tab)
    n = s.n[s.row_off]
    hits = s.S[s.row_off]
    assert np.all(n == 100)
    rate = hits.mean() / 100.0
    assert abs(rate - 0.7) < 0.02, f"empirical rate {rate:.4f} far from mu=0.7"


def test_forcing_identical_actions_gives_identical_trajectories():
    """The sharpest CRN check: same forced actions => bit-identical branches."""
    tab = _StubTable(60)
    states = []
    for _ in range(2):
        s = GrowingState(n_replicates=32, capacity=4, horizon=60, base_seed=2024)
        s.add_arms(np.full(32, 0.55, dtype=np.float32), np.zeros(32, dtype=np.int32))
        s.add_arms(np.full(32, 0.45, dtype=np.float32), np.ones(32, dtype=np.int32))
        for step in range(40):
            s.pull(np.full(32, step % 2, dtype=np.int64), tab)
        states.append(s)
    a, b = states
    assert np.array_equal(a.n, b.n)
    assert np.array_equal(a.S, b.S)
    assert np.allclose(a.log_e, b.log_e)


# ---- equivalence with the repo's scalar e-process -----------------------------


@pytest.mark.parametrize("m0", [0.3, 0.5, 0.7])
def test_vectorized_agrapa_matches_scalar_upward_capital(m0: float):
    """The vectorized betting recursion must reproduce the paper's scalar reference.

    This is what keeps the fast path honest against the code that produced the
    published results: any drift here means the recorded log-e feature is not the
    quantity the paper defines.
    """
    rng = np.random.default_rng(4242)
    rewards = (rng.random(80) < 0.62).astype(np.float64)

    scalar = UpwardCapitalEProcess(m0=m0)
    scalar_path = [scalar.update(float(x)) for x in rewards]

    s = GrowingState(
        n_replicates=1, capacity=1, horizon=80, base_seed=0, agrapa=AgrapaParams(m0=m0)
    )
    s.add_arms(np.array([0.62], dtype=np.float32), np.zeros(1, dtype=np.int32))
    tab = _StubTable(80)
    vec_path = []
    for x in rewards:
        # Inject the same reward sequence rather than drawing, so we compare the
        # recursion itself and not the RNG.
        s.thresh[0] = crn.UNIFORM_SCALE if x > 0.5 else np.uint64(0)
        s.pull(np.zeros(1, dtype=np.int64), tab)
        vec_path.append(float(s.log_e[0]))

    assert np.allclose(scalar_path, vec_path, atol=1e-12), (
        f"max abs diff = {np.max(np.abs(np.array(scalar_path) - np.array(vec_path))):.3e}"
    )
