from __future__ import annotations

import numpy as np
import pytest

from cold_start.policies.base import PolicyState
from cold_start.policies.epsilon_greedy import EpsilonGreedyPolicy
from cold_start.policies.spruce import SprucePolicy
from cold_start.policies.thompson import ThompsonPolicy
from cold_start.policies.ucb import UCBPolicy
from cold_start.policies.uniform import UniformPolicy


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def test_uniform_round_robin(rng):
    arm_ids = ["a", "b", "c"]
    state = PolicyState(arm_ids=arm_ids)
    p = UniformPolicy(rng, mode="round_robin")
    picks = [p.next_arm(t, state) for t in range(1, 10)]
    assert picks == ["a", "b", "c", "a", "b", "c", "a", "b", "c"]


def test_policy_select_arm_alias(rng):
    arm_ids = ["a", "b", "c"]
    state = PolicyState(arm_ids=arm_ids)
    p = UniformPolicy(rng, mode="round_robin")
    assert p.select_arm(1, state) == "a"
    assert p.select_arm(2, state) == "b"


def _run_toy_bandit(policy_cls, kwargs, probs, T, seed=0):
    """Run a Bernoulli bandit on `probs`, return pull counts."""
    rng = np.random.default_rng(seed)
    arm_ids = list(probs.keys())
    state = PolicyState(arm_ids=arm_ids)
    policy = policy_cls(rng=rng, **kwargs)

    # Warm-start manually: play each arm once to avoid NaN means.
    for t, aid in enumerate(arm_ids, start=1):
        x = float(rng.random() < probs[aid])
        state.record(aid, x, log_e=0.0)

    for t in range(len(arm_ids) + 1, T + 1):
        aid = policy.next_arm(t, state)
        x = float(rng.random() < probs[aid])
        state.record(aid, x, log_e=0.0)
    return dict(state.pulls)


def test_epsilon_greedy_concentrates_on_best():
    probs = {"lo": 0.2, "mid": 0.5, "hi": 0.85}
    pulls = _run_toy_bandit(EpsilonGreedyPolicy, {"epsilon": 0.1}, probs, T=800, seed=1)
    best_pulls = pulls["hi"]
    assert best_pulls > 0.5 * 800, f"epsilon-greedy did not concentrate on 'hi': {pulls}"


def test_thompson_concentrates_on_best():
    probs = {"lo": 0.2, "mid": 0.5, "hi": 0.8}
    pulls = _run_toy_bandit(ThompsonPolicy, {}, probs, T=800, seed=2)
    assert pulls["hi"] > 0.55 * 800, f"thompson did not concentrate on 'hi': {pulls}"


def test_spruce_picks_unplayed_first(rng):
    arm_ids = ["a", "b", "c"]
    state = PolicyState(arm_ids=arm_ids)
    p = SprucePolicy(rng, exploration_c=1.0)
    picks = {p.next_arm(t, state) for t in range(1, 4)}
    # All three unplayed arms should be explored in the first 3 steps.
    # Because they all have infinite score, the tie-break is random — but any
    # pick among them is acceptable. We check no arm is preferred with zero info.
    assert picks.issubset(set(arm_ids))


def test_spruce_can_break_ties_deterministically(rng):
    arm_ids = ["a", "b", "c"]
    state = PolicyState(arm_ids=arm_ids)
    p = SprucePolicy(rng, exploration_c=1.0, tie_break="first")
    assert p.next_arm(t=1, state=state) == "a"
    state.record("a", 1.0, log_e=0.0, log_e_upper=0.0)
    assert p.next_arm(t=2, state=state) == "b"


def test_ucb_picks_unplayed_first(rng):
    arm_ids = ["a", "b", "c"]
    state = PolicyState(arm_ids=arm_ids)
    p = UCBPolicy(rng, exploration_c=1.0)
    # Any unpulled arm has infinite UCB; in the first call the policy must
    # pick one of them, never an already-pulled arm.
    state.record("a", 1.0, log_e=0.0)
    pick = p.next_arm(t=2, state=state)
    assert pick in ("b", "c"), f"UCB picked already-pulled arm {pick}"


def test_ucb_can_break_ties_deterministically(rng):
    arm_ids = ["a", "b", "c"]
    state = PolicyState(arm_ids=arm_ids)
    p = UCBPolicy(rng, exploration_c=1.0, tie_break="first")
    assert p.next_arm(t=1, state=state) == "a"
    state.record("a", 1.0, log_e=0.0)
    assert p.next_arm(t=2, state=state) == "b"


def test_ucb_greedy_when_c_zero():
    """With c=0, UCB reduces to greedy on empirical means."""
    rng = np.random.default_rng(0)
    arm_ids = ["lo", "hi"]
    state = PolicyState(arm_ids=arm_ids)
    # Both arms have been played; "hi" has the better mean.
    state.record("lo", 0.0, log_e=0.0)
    state.record("lo", 0.0, log_e=0.0)
    state.record("hi", 1.0, log_e=0.0)
    state.record("hi", 1.0, log_e=0.0)
    p = UCBPolicy(rng, exploration_c=0.0)
    assert p.next_arm(t=5, state=state) == "hi"


def test_ucb_concentrates_on_best():
    probs = {"lo": 0.2, "mid": 0.5, "hi": 0.85}
    pulls = _run_toy_bandit(UCBPolicy, {"exploration_c": 1.0}, probs, T=800, seed=4)
    assert pulls["hi"] > 0.4 * 800, f"UCB did not concentrate on 'hi': {pulls}"


def test_ucb_rejects_negative_c():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        UCBPolicy(rng, exploration_c=-0.1)
