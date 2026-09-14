"""Tests for the vectorized rollout loop, including the paired-branch CRN coupling."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.allocation import LUCB, UCBChallenger, WidthRacing
from cold_start.growing.search_policies import BernoulliSearch, PowerSchedule, UniformRandom
from cold_start.growing.simulator import Simulator, live_arm_counts, seed_initial_arms
from cold_start.growing.state import GrowingState


class StubTable:
    """(n, S)-indexed bounds: empirical mean +/- 1/sqrt(n). Shape-compatible stand-in."""

    def __init__(self, horizon: int) -> None:
        self.stride = horizon + 1
        n = np.arange(self.stride)[:, None]
        k = np.arange(self.stride)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n > 0, k / np.maximum(n, 1), 0.5)
            half = np.where(n > 0, 1.0 / np.sqrt(np.maximum(n, 1)), 1.0)
        self.lower_flat = np.clip(mean - half, 0.0, 1.0).astype(np.float32).ravel()
        self.upper_flat = np.clip(mean + half, 0.0, 1.0).astype(np.float32).ravel()


class StubReservoir:
    """Uniform on [0.1, 0.9] via inverse CDF, so draws are addressable by index."""

    def sample_from_uniforms(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u) * 0.8 + 0.1, 0.0, 1.0)

    def essential_sup(self) -> float:
        return 0.9


def _sim(horizon: int, policy=None, alloc=None, cap: int = 32) -> Simulator:
    return Simulator(
        table=StubTable(horizon),
        reservoir=StubReservoir(),
        allocation=alloc or LUCB(),
        search_policy=policy or PowerSchedule(alpha=0.5, c=1.0),
        horizon=horizon,
        max_live_arms=cap,
    )


def _fresh(m: int, horizon: int, seed: int = 17, n_arms: int = 2) -> tuple:
    sim = _sim(horizon)
    st = GrowingState(m, 4, horizon, base_seed=seed)
    seed_initial_arms(st, sim.reservoir, n_arms, sim.table)
    return st, sim


def test_budget_is_exactly_conserved():
    """Every step spends exactly one unit of evaluation budget, SEARCH or REFINE."""
    st, sim = _fresh(32, 120, n_arms=2)
    sim.run_to_horizon(st, start_t=2)
    assert np.all(st.pulls_used == 120), f"pull counts: {np.unique(st.pulls_used)}"


def test_sqrt_schedule_reaches_about_sqrt_horizon_arms():
    st, sim = _fresh(32, 200, n_arms=2)
    sim.run_to_horizon(st, start_t=2)
    k = st.Kt.mean()
    assert abs(k - np.sqrt(200)) < 2.0, f"K_t={k:.1f} vs sqrt(T)={np.sqrt(200):.1f}"


def test_live_arm_cap_is_respected():
    st, sim = _fresh(16, 150, n_arms=1)
    sim.search_policy = BernoulliSearch(p=1.0, rng=np.random.default_rng(0))
    sim.max_live_arms = 7
    sim.run_to_horizon(st, start_t=1)
    assert st.Kt.max() <= 7, f"cap breached: {st.Kt.max()}"


def test_empty_state_is_forced_to_search_even_when_refine_is_forced():
    """A replicate with no arms has nothing to refine; the constraint must win."""
    sim = _sim(50)
    st = GrowingState(8, 4, 50, base_seed=3)
    res = sim.step(st, t=0, force=False)
    assert res.searched.all(), "forced REFINE on an empty state must fall back to SEARCH"
    assert np.all(st.Kt == 1)


def test_forced_actions_are_honoured():
    st, sim = _fresh(16, 60, n_arms=3)
    before = st.Kt.copy()
    searched = sim.step(st.clone(), t=3, force=True)
    assert searched.searched.all()

    refined_state = st.clone()
    res = sim.step(refined_state, t=3, force=False)
    assert not res.searched.any()
    assert np.array_equal(refined_state.Kt, before), "REFINE must not add an arm"


def test_search_branch_adds_exactly_one_arm():
    st, sim = _fresh(16, 60, n_arms=3)
    before = st.Kt.copy()
    sim.step(st, t=3, force=True)
    assert np.all(st.Kt - before == 1)


def test_identical_seeds_give_identical_trajectories():
    a, sim_a = _fresh(24, 100, seed=99)
    b, sim_b = _fresh(24, 100, seed=99)
    sim_a.run_to_horizon(a, start_t=2)
    sim_b.run_to_horizon(b, start_t=2)
    assert np.array_equal(a.n, b.n)
    assert np.array_equal(a.S, b.S)
    assert np.array_equal(a.mu, b.mu)


def test_paired_branches_share_the_same_next_reservoir_draw():
    """The CRN invariant that makes the paired design worth running.

    The j-th arm a branch discovers must be the SAME underlying draw in both, so the
    two branches differ by the action taken and not by which arms luck handed them.
    """
    st, sim = _fresh(64, 80, n_arms=2)
    search_now = st.clone()
    refine_now = st.clone()

    sim.step(search_now, t=2, force=True)
    # The refine branch searches one step later; it should receive the same arm.
    sim.step(refine_now, t=2, force=False)
    sim.step(refine_now, t=3, force=True)

    mu_a = search_now.view(search_now.mu)[:, 2]
    mu_b = refine_now.view(refine_now.mu)[:, 2]
    assert np.allclose(mu_a, mu_b), "branches drew different arms from the reservoir"


def test_new_arm_is_pulled_exactly_once_on_discovery():
    st, sim = _fresh(16, 60, n_arms=2)
    sim.step(st, t=2, force=True)
    new_n = st.view(st.n)[:, 2]
    assert np.all(new_n == 1), f"new arm pull counts: {np.unique(new_n)}"


def test_eliminating_allocation_concentrates_pulls():
    """Weak arms must stop drawing budget; that is the point of an eliminating rule."""
    st, sim = _fresh(32, 300, n_arms=2)
    sim.run_to_horizon(st, start_t=2)
    n2 = st.view(st.n)
    share = (n2.max(axis=1) / n2.sum(axis=1)).mean()
    uniform_share = 1.0 / st.Kt.mean()
    assert share > 2 * uniform_share, f"top-arm share {share:.3f} vs uniform {uniform_share:.3f}"


@pytest.mark.parametrize("alloc", [UCBChallenger(), LUCB(), WidthRacing()])
def test_every_allocation_rule_runs_to_horizon(alloc):
    st, sim = _fresh(16, 100, n_arms=2)
    sim.allocation = alloc
    sim.run_to_horizon(st, start_t=2)
    assert np.all(st.pulls_used == 100)
    assert np.all(st.Kt >= 2)


@pytest.mark.parametrize(
    "policy", [PowerSchedule(alpha=0.5), UniformRandom(rng=np.random.default_rng(1))]
)
def test_every_policy_runs_to_horizon(policy):
    st, sim = _fresh(16, 100, n_arms=2)
    sim.search_policy = policy
    sim.run_to_horizon(st, start_t=2)
    assert np.all(st.pulls_used == 100)


def test_live_arm_counts_match_kt():
    st, sim = _fresh(16, 60, n_arms=3)
    sim.run_to_horizon(st, start_t=3)
    assert np.array_equal(live_arm_counts(st), st.Kt)


def test_run_to_horizon_from_the_horizon_is_a_noop():
    st, sim = _fresh(8, 40, n_arms=2)
    before = st.n.copy()
    sim.run_to_horizon(st, start_t=40)
    assert np.array_equal(st.n, before)


def test_step_cost_is_nearly_independent_of_arm_count():
    """Regression guard for the incremental bound update.

    Only one arm per replicate changes per step, so we refresh only the M touched
    entries. If someone reverts to re-gathering all M*K bounds, cost becomes linear
    in K and the overnight compute budget stops holding -- this catches that in CI
    rather than after a wasted night. The threshold is deliberately loose so it
    fails on a structural regression (roughly 10x), not on machine noise.
    """
    import time

    def measure(cap: int) -> float:
        horizon = 400
        sim = _sim(horizon, cap=cap)
        st = GrowingState(512, cap, horizon, base_seed=1)
        seed_initial_arms(st, sim.reservoir, min(cap, 8), sim.table)
        for t in range(8, 40):
            sim.step(st, t)
        start = time.perf_counter()
        for t in range(40, 140):
            sim.step(st, t)
        return (time.perf_counter() - start) / 100

    small = measure(16)
    large = measure(64)
    ratio = large / small
    assert ratio < 4.0, (
        f"step cost grew {ratio:.1f}x going from K=16 to K=64; "
        "the per-step bound update looks linear in K again"
    )


def test_forced_search_at_the_arm_cap_raises_rather_than_silently_refining():
    """The simulator-level half of the arm-cap guard.

    The labeller has its own upfront check, so this one protects every other caller.
    Without it a forced SEARCH at the cap is ANDed away into a REFINE, which makes the
    two oracle branches identical and fabricates a maximally-confident zero label.
    """
    from cold_start.growing.state import ForcedActionUnavailable

    sim = _sim(60, cap=3)
    st = GrowingState(16, 4, 60, base_seed=4)
    seed_initial_arms(st, sim.reservoir, 3, sim.table)
    assert np.all(st.Kt == 3), "fixture did not reach the cap"

    with pytest.raises(ForcedActionUnavailable):
        sim.step(st, t=3, force=True, strict=True)


def test_non_strict_steps_may_legitimately_hit_the_cap():
    """Later rounds of a multi-round commitment are allowed to be capped.

    "Commit to searching for k rounds" means "for as many as the cap allows"; only the
    first action defines the label, so only it is strict.
    """
    sim = _sim(60, cap=3)
    st = GrowingState(16, 4, 60, base_seed=4)
    seed_initial_arms(st, sim.reservoir, 3, sim.table)
    res = sim.step(st, t=3, force=True, strict=False)
    assert not res.searched.any(), "capped replicates should fall back to REFINE"
    assert np.all(st.Kt == 3)


def test_unforced_policy_decisions_are_never_strict():
    """A behavioural policy hitting the cap is normal and must not raise."""
    sim = _sim(60, cap=3, policy=BernoulliSearch(p=1.0, rng=np.random.default_rng(0)))
    st = GrowingState(16, 4, 60, base_seed=4)
    seed_initial_arms(st, sim.reservoir, 3, sim.table)
    sim.run_to_horizon(st, start_t=3)
    assert np.all(st.Kt == 3)
    assert np.all(st.pulls_used == 60)
