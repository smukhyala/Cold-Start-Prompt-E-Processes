"""Tests for the eliminating REFINE allocation rules."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.allocation import (
    LUCB,
    UCBChallenger,
    WidthRacing,
    leader_and_challenger,
    n_eliminated,
    n_plausible,
    plausible_mask,
)
from cold_start.growing.state import GrowingState
from cold_start.registry import get_registered


def _state(lcb: list[float], ucb: list[float], m: int = 3) -> GrowingState:
    """Build a state with hand-set confidence bounds, replicated m ways."""
    k = len(lcb)
    s = GrowingState(n_replicates=m, capacity=max(k, 1), horizon=50, base_seed=1)
    for j in range(k):
        s.add_arms(np.full(m, 0.5, dtype=np.float32), np.full(m, j, dtype=np.int32))
    lv, uv = s.view(s.lcb), s.view(s.ucb)
    for j in range(k):
        lv[:, j] = lcb[j]
        uv[:, j] = ucb[j]
    return s


def test_leader_is_argmax_lcb_and_challenger_excludes_it():
    s = _state([0.70, 0.40, 0.05], [0.90, 0.75, 0.30])
    leader, challenger = leader_and_challenger(s)
    assert np.all(leader == 0), f"leader={leader}"
    assert np.all(challenger == 1), f"challenger={challenger}"


def test_challenger_selection_does_not_corrupt_ucb():
    """The challenger search masks the leader in place; it must restore it."""
    s = _state([0.70, 0.40], [0.90, 0.75])
    before = s.ucb.copy()
    leader_and_challenger(s)
    assert np.array_equal(s.ucb, before), "ucb array was left mutated"


def test_eliminated_arm_is_never_selected():
    """An arm with ucb < max lcb cannot be optimal, so it must draw no more pulls."""
    s = _state([0.70, 0.40, 0.05], [0.90, 0.75, 0.30])
    assert np.all(n_plausible(s) == 2)
    assert np.all(n_eliminated(s) == 1)
    for rule in (UCBChallenger(), LUCB(), WidthRacing()):
        chosen = rule.select(s)
        assert not np.any(chosen == 2), f"{rule.name} selected the eliminated arm"


def test_ucb_rule_picks_highest_upper_bound():
    s = _state([0.10, 0.20, 0.15], [0.95, 0.60, 0.70])
    assert np.all(UCBChallenger().select(s) == 0)


def test_racing_picks_widest_plausible_interval():
    s = _state([0.30, 0.10, 0.28], [0.40, 0.90, 0.35])
    assert np.all(WidthRacing().select(s) == 1)


def test_lucb_balances_pull_counts_between_leader_and_challenger():
    """LUCB alternates by pull count, so the pair stays equally well measured."""
    s = _state([0.70, 0.40], [0.75, 0.95])
    n2 = s.view(s.n)
    n2[:, 0] = 4  # leader by lcb
    n2[:, 1] = 9  # challenger, already better measured
    assert np.all(LUCB().select(s) == 0), "should top up the under-pulled leader"

    n2[:, 0] = 12
    n2[:, 1] = 3
    assert np.all(LUCB().select(s) == 1), "should top up the under-pulled challenger"


def test_lucb_does_not_lock_onto_mid_range_arms():
    """Regression: confidence-sequence width is p-dependent, so it cannot drive LUCB.

    A Bernoulli arm near 0.5 has a wider interval at *every* sample size than an arm
    near 0.9. An earlier width-based tie-break therefore poured budget into whichever
    arm sat mid-range and never left -- measured 18.4 pulls on an inferior arm at 0.55
    against 10.2 on a leader at 0.85, which cost 8 percentage points of final
    recommendation accuracy. Pull-count balancing has no such dependence.
    """
    # Both arms must SURVIVE for alternation to be the correct behaviour: the leader
    # sits tight near 0.85, the challenger wide near 0.5, and the challenger's upper
    # bound still reaches the leader's lower bound.
    s = _state([0.60, 0.30], [0.92, 0.70], m=1)
    n2 = s.view(s.n)
    n2[:, 0] = 5
    n2[:, 1] = 5
    picks = []
    for _ in range(20):
        c = int(LUCB().select(s)[0])
        picks.append(c)
        n2[0, c] += 1
    counts = np.bincount(picks, minlength=2)
    assert abs(int(counts[0]) - int(counts[1])) <= 1, (
        f"allocation drifted to one arm: {counts.tolist()}"
    )


def test_lucb_handles_a_single_arm():
    s = _state([0.30], [0.80])
    assert np.all(LUCB().select(s) == 0)


def test_all_rules_return_active_arms_only():
    s = GrowingState(n_replicates=5, capacity=8, horizon=50, base_seed=2)
    for j in range(3):
        s.add_arms(np.full(5, 0.5, dtype=np.float32), np.full(5, j, dtype=np.int32))
    for rule in (UCBChallenger(), LUCB(), WidthRacing()):
        chosen = rule.select(s)
        assert np.all(chosen < s.Kt), f"{rule.name} chose an unallocated slot: {chosen}"


def test_plausible_mask_excludes_inactive_slots():
    s = GrowingState(n_replicates=2, capacity=6, horizon=50, base_seed=3)
    s.add_arms(np.full(2, 0.5, dtype=np.float32), np.zeros(2, dtype=np.int32))
    mask = plausible_mask(s)
    assert mask[:, 0].all()
    assert not mask[:, 1:].any(), "empty slots must never be plausible winners"


def test_ties_are_not_always_broken_toward_the_lowest_slot():
    """A freshly searched arm always occupies the highest slot.

    Without the per-arm jitter it would lose every tie to slot 0, biasing the oracle
    advantage downward exactly when many arms share the same (n, S).
    """
    s = _state([0.5, 0.5, 0.5], [0.8, 0.8, 0.8], m=400)
    chosen = UCBChallenger().select(s)
    assert len(np.unique(chosen)) > 1, "all replicates broke the tie identically"


@pytest.mark.parametrize("name,cls", [("ucb", UCBChallenger), ("lucb", LUCB), ("racing", WidthRacing)])
def test_registry_round_trip(name: str, cls: type):
    assert get_registered("allocation", name) is cls



def test_lucb_stops_pulling_once_only_one_arm_survives():
    """LUCB must not alternate with a provably eliminated arm.

    `leader_and_challenger` takes argmax over every non-leader slot. While two or more
    arms are plausible that is harmless, but once the leader is the only survivor the
    challenger is necessarily the best ELIMINATED arm -- and a guard counting
    *discovered* arms never fires. LUCB then spent half its remaining budget on an arm
    it had already proven could not win.
    """
    s = _state([0.70, 0.20, 0.05], [0.90, 0.55, 0.30])
    assert np.all(n_plausible(s) == 1), "fixture should leave exactly one survivor"
    n2 = s.view(s.n)
    n2[:, 0] = 30
    n2[:, 1] = 4
    n2[:, 2] = 2
    chosen = LUCB().select(s)
    assert np.all(chosen == 0), (
        f"LUCB chose {np.unique(chosen)} when only arm 0 survives"
    )


def test_lucb_still_alternates_while_two_arms_survive():
    """The fix must not break the normal case it exists for."""
    s = _state([0.55, 0.50, 0.05], [0.85, 0.80, 0.20])
    assert np.all(n_plausible(s) == 2)
    n2 = s.view(s.n)
    n2[:, 0] = 20
    n2[:, 1] = 4
    assert np.all(LUCB().select(s) == 1), "should top up the under-pulled survivor"
