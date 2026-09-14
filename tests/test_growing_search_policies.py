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
