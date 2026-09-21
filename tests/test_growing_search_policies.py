"""Tests for the SEARCH-vs-REFINE behavioral and continuation policies."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.search_policies import (
    BernoulliSearch,
    BracketExpansion,
    DecisionContext,
    EpsilonSchedule,
    EvidenceGatedSchedule,
    EvidenceThreshold,
    OSEInspired,
    PowerSchedule,
    UniformRandom,
    behavioral_mixture,
)
from cold_start.growing.state import GrowingState
from cold_start.registry import get_registered

M = 512


def _state(n_arms: int, lcb: float = 0.3, ucb: float = 0.7, m: int = M) -> GrowingState:
    s = GrowingState(n_replicates=m, capacity=max(n_arms, 1), horizon=200, base_seed=5)
    for j in range(n_arms):
        s.add_arms(np.full(m, 0.5, dtype=np.float32), np.full(m, j, dtype=np.int32))
    if n_arms:
        s.view(s.lcb)[:] = lcb
        s.view(s.ucb)[:] = ucb
    return s


def _ctx(t: int = 10, horizon: int = 200, **kw) -> DecisionContext:
    return DecisionContext(t=t, horizon=horizon, **kw)


def test_empty_state_always_searches():
    """With no arms discovered there is nothing to refine."""
    s = _state(0)
    ctx = _ctx(log_e_pair=np.zeros(M), best_mean=np.zeros(M))
    for policy in behavioral_mixture(np.random.default_rng(0)):
        assert policy.should_search(s, ctx).all(), f"{policy.name} refused to open"


@pytest.mark.parametrize("p", [0.0, 0.2, 0.8, 1.0])
def test_bernoulli_search_hits_its_rate(p: float):
    s = _state(3)
    rate = BernoulliSearch(p=p, rng=np.random.default_rng(1)).should_search(s, _ctx()).mean()
    assert abs(rate - p) < 0.06, f"p={p} produced rate={rate:.3f}"


def test_power_schedule_matches_its_closed_form():
    """SEARCH iff K_t < c * t**alpha, checked either side of the boundary."""
    ctx = _ctx(t=9)
    # target = 1.0 * 9**0.5 = 3.0
    assert PowerSchedule(alpha=0.5, c=1.0).should_search(_state(2), ctx).all()
    assert not PowerSchedule(alpha=0.5, c=1.0).should_search(_state(3), ctx).any()
    # target = 1.5 * 9**(1/3) ~= 3.12
    assert PowerSchedule(alpha=1 / 3, c=1.5).should_search(_state(3), ctx).all()


def test_power_schedule_grows_the_arm_budget_over_time():
    policy = PowerSchedule(alpha=0.5, c=1.0)
    s = _state(5)
    assert not policy.should_search(s, _ctx(t=16)).any(), "K=5 should exceed sqrt(16)=4"
    assert policy.should_search(s, _ctx(t=100)).all(), "K=5 should fall short of sqrt(100)=10"


def test_epsilon_schedule_flips_roughly_epsilon_of_decisions():
    s, ctx = _state(2), _ctx(t=9)
    base = PowerSchedule(alpha=0.5, c=1.0).should_search(s, ctx)
    noisy = EpsilonSchedule(
        alpha=0.5, c=1.0, epsilon=0.25, rng=np.random.default_rng(2)
    ).should_search(s, ctx)
    flipped = (base != noisy).mean()
    assert abs(flipped - 0.25) < 0.06, f"flip rate {flipped:.3f}"


def test_uniform_random_is_a_fair_coin():
    rate = UniformRandom(rng=np.random.default_rng(3)).should_search(_state(3), _ctx()).mean()
    assert abs(rate - 0.5) < 0.06, f"rate={rate:.3f}"


def test_bracket_expands_when_the_race_narrows():
    """Wide plausible set -> race it; narrow set -> recruit more."""
    narrow = _state(2)  # both arms plausible, below the bracket width of 4
    assert BracketExpansion(base_width=4).should_search(narrow, _ctx()).all()
    wide = _state(6)
    assert not BracketExpansion(base_width=4).should_search(wide, _ctx()).any()


def test_evidence_threshold_fires_only_once_the_leader_separates():
    s = _state(3)
    policy = EvidenceThreshold(threshold=2.9957)
    assert not policy.should_search(s, _ctx(log_e_pair=np.full(M, 1.0))).any()
    assert policy.should_search(s, _ctx(log_e_pair=np.full(M, 5.0))).all()


def test_evidence_threshold_requires_its_input():
    with pytest.raises(ValueError):
        EvidenceThreshold().should_search(_state(3), _ctx())


def test_ose_stops_searching_once_a_good_arm_is_held():
    s = _state(3)
    policy = OSEInspired(target0=0.9, decay=0.5)
    assert policy.should_search(s, _ctx(t=0, best_mean=np.full(M, 0.2))).all()
    assert not policy.should_search(s, _ctx(t=0, best_mean=np.full(M, 0.95))).any()


def test_ose_target_decays_so_search_stops_late_in_the_budget():
    s = _state(3)
    policy = OSEInspired(target0=0.9, decay=0.5)
    held = np.full(M, 0.5)
    assert policy.should_search(s, _ctx(t=0, horizon=100, best_mean=held)).all()
    # By t=T the bar has fallen to 0.9*(1-0.5)=0.45, below what we already hold.
    assert not policy.should_search(s, _ctx(t=100, horizon=100, best_mean=held)).any()


def test_cp0_stops_recruiting_once_the_race_is_decided():
    """The round-0 continuation policy gates its schedule on being unresolved."""
    policy = EvidenceGatedSchedule(alpha=0.5, c=1.0)
    unresolved = _state(2, lcb=0.3, ucb=0.7)  # both plausible
    assert policy.should_search(unresolved, _ctx(t=100)).all()

    decided = _state(2)
    lv, uv = decided.view(decided.lcb), decided.view(decided.ucb)
    lv[:, 0], uv[:, 0] = 0.80, 0.90
    lv[:, 1], uv[:, 1] = 0.10, 0.20  # eliminated: ucb < max lcb
    assert not policy.should_search(decided, _ctx(t=100)).any()


def test_bernoulli_rejects_bad_probability():
    with pytest.raises(ValueError):
        BernoulliSearch(p=1.5)


def test_mixture_has_ten_distinct_policies():
    mix = behavioral_mixture(np.random.default_rng(0))
    assert len(mix) == 10, f"expected 10 behavioral policies, got {len(mix)}"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("bernoulli_search", BernoulliSearch),
        ("power_schedule", PowerSchedule),
        ("epsilon_schedule", EpsilonSchedule),
        ("bracket", BracketExpansion),
        ("random_search", UniformRandom),
        ("evidence_threshold", EvidenceThreshold),
        ("ose", OSEInspired),
        ("cp0", EvidenceGatedSchedule),
    ],
)
def test_registry_round_trip(name: str, cls: type):
    assert get_registered("search_policy", name) is cls


# ---- the environment-adaptive schedule (DEPLOYMENT_PLAN.md, Pre-registration 4) --------------


def _state_with_means(post_rows: list[list[float]], n_pulls: int = 98) -> GrowingState:
    """A state whose held arms have the given posterior means (`(S+1)/(n+2)`), per replicate."""
    m = len(post_rows)
    k = len(post_rows[0])
    s = GrowingState(n_replicates=m, capacity=k, horizon=200, base_seed=5)
    for j in range(k):
        s.add_arms(np.full(m, 0.5, dtype=np.float32), np.full(m, j, dtype=np.int32))
    n = s.view(s.n)
    S = s.view(s.S)
    for i, row in enumerate(post_rows):
        for j, post in enumerate(row):
            n[i, j] = n_pulls
            S[i, j] = round(post * (n_pulls + 2) - 1)  # (S+1)/(n+2) == post, exact at n_pulls = 98
    return s


def test_adaptive_schedule_reads_the_fraction_of_arms_near_the_best():
    from cold_start.growing.search_policies import TailAdaptiveSchedule, frac_within_of_best

    thin = [0.70, 0.69, 0.68, 0.67]   # every arm within 0.05 of the best -> q = 1
    heavy = [0.70, 0.40, 0.30, 0.20]  # only the best itself -> q = 0.25
    s = _state_with_means([thin, heavy])
    q = frac_within_of_best(s, 0.05)
    assert q.tolist() == pytest.approx([1.0, 0.25])
    # b = 0: a plain c * T^alpha schedule, identical for both replicates.
    plain = TailAdaptiveSchedule(alpha=0.5, c=1.0, b=0.0)
    ctx = _ctx(t=10, horizon=100)
    assert plain.target(s, ctx).tolist() == pytest.approx([10.0, 10.0])
    # b > 0: the heavy-tailed replicate's target grows by (1 + b * (1 - q)); the thin one's does not.
    adaptive = TailAdaptiveSchedule(alpha=0.5, c=1.0, b=2.0)
    assert adaptive.target(s, ctx).tolist() == pytest.approx([10.0, 10.0 * (1 + 2 * 0.75)])
    # And the decision is K_t < target: both hold 4 arms, so both search here...
    assert adaptive.should_search(s, ctx).tolist() == [True, True]
    # ...but at c = 0.4 (target 4 vs 10) only the heavy-tailed replicate keeps recruiting.
    tight = TailAdaptiveSchedule(alpha=0.5, c=0.4, b=2.0)
    assert tight.target(s, ctx).tolist() == pytest.approx([4.0, 10.0])
    assert tight.should_search(s, ctx).tolist() == [False, True]


def test_adaptive_schedule_target_depends_on_the_horizon_not_the_clock():
    """K_target = c * T^alpha * (...): a *fixed-K-per-horizon* rule, reached as fast as possible."""
    from cold_start.growing.search_policies import TailAdaptiveSchedule

    s = _state_with_means([[0.7, 0.69, 0.68]])
    rule = TailAdaptiveSchedule(alpha=0.5, c=2.0, b=0.0)
    assert rule.target(s, _ctx(t=3, horizon=100)).tolist() == pytest.approx([20.0])
    assert rule.target(s, _ctx(t=90, horizon=100)).tolist() == pytest.approx([20.0])
    assert rule.target(s, _ctx(t=3, horizon=400)).tolist() == pytest.approx([40.0])


def test_adaptive_schedule_opens_with_no_arms_and_is_registered():
    from cold_start.growing.search_policies import TailAdaptiveSchedule

    assert TailAdaptiveSchedule(alpha=0.5, c=1.0, b=1.0).should_search(_state(0), _ctx()).all()
    assert get_registered("search_policy", "adaptive_K") is TailAdaptiveSchedule


# ---- the best-mean gate (DEPLOYMENT_PLAN.md, Pre-registration 5) --------------------------


def test_best_mean_gate_searches_while_the_best_is_below_theta_and_k_below_the_ceiling():
    from cold_start.growing.search_policies import BestMeanGate, best_held_mean

    low = [0.45, 0.40, 0.30]    # best 0.45 -> keep recruiting
    high = [0.70, 0.40, 0.30]   # best 0.70 -> stop
    s = _state_with_means([low, high])
    assert best_held_mean(s).tolist() == pytest.approx([0.45, 0.70])
    rule = BestMeanGate(theta=0.6, alpha=0.5, c=2.0)
    ctx = _ctx(t=10, horizon=100)
    assert rule.ceiling(ctx) == pytest.approx(20.0)
    assert rule.should_search(s, ctx).tolist() == [True, False]
    # The ceiling binds even when the best is poor: c * T^alpha = 2 arms here, both hold 3.
    tight = BestMeanGate(theta=0.6, alpha=0.0, c=2.0)
    assert tight.should_search(s, ctx).tolist() == [False, False]
    # theta >= 1 is the fixed-K schedule: the gate never closes on its own.
    fixed = BestMeanGate(theta=1.0, alpha=0.5, c=2.0)
    assert fixed.should_search(s, ctx).tolist() == [True, True]


def test_best_mean_gate_opens_with_no_arms_and_is_registered():
    from cold_start.growing.search_policies import BestMeanGate

    assert BestMeanGate(theta=0.6, alpha=0.5, c=2.0).should_search(_state(0), _ctx()).all()
    assert get_registered("search_policy", "bestmean_K") is BestMeanGate


# ---- the level-scaled schedule (DEPLOYMENT_PLAN.md, Pre-registration 6) --------------------


def test_level_scaled_schedule_scales_its_target_by_the_observed_level():
    from cold_start.growing.search_policies import LevelScaledSchedule, held_level

    thin = [0.70, 0.60, 0.50]    # level 0.60
    heavy = [0.50, 0.30, 0.31]   # level 0.37
    s = _state_with_means([thin, heavy])
    assert held_level(s).tolist() == pytest.approx([0.6, 0.37])
    ctx = _ctx(t=10, horizon=100)
    plain = LevelScaledSchedule(alpha=0.5, c=2.0, b=0.0)
    assert plain.target(s, ctx).tolist() == pytest.approx([20.0, 20.0])
    scaled = LevelScaledSchedule(alpha=0.5, c=2.0, b=4.0)
    import math
    assert scaled.target(s, ctx).tolist() == pytest.approx([20 * math.exp(4 * (0.5 - 0.6)), 20 * math.exp(4 * (0.5 - 0.37))])
    # Both hold 3 arms: the thin replicate's target (13.4) and the heavy one's (33.6) both exceed 3.
    assert scaled.should_search(s, ctx).tolist() == [True, True]
    tight = LevelScaledSchedule(alpha=0.0, c=3.5, b=4.0)  # targets 2.3 and 5.9
    assert tight.should_search(s, ctx).tolist() == [False, True]


def test_level_scaled_schedule_opens_with_no_arms_and_is_registered():
    from cold_start.growing.search_policies import LevelScaledSchedule

    assert LevelScaledSchedule(alpha=0.5, c=1.0, b=2.0).should_search(_state(0), _ctx()).all()
    assert get_registered("search_policy", "level_K") is LevelScaledSchedule
