"""Tests for the exact backward-induction validator on the toy two-point reservoir.

Every other label in this study is an approximation checked against this module, so
these tests check the DP against arithmetic done by hand rather than against code.
Two structural facts do most of the work:

* the two-point posterior mean ``E[mu | n, S]`` is also the predictive
  ``P(next pull = 1 | n, S)``, and it is a martingale in the data -- so refining a
  *single* arm with one step left is worth exactly its current posterior mean; and
* the terminal payoff is a max over arms, so extra information can never hurt, which
  is what makes the optimal value monotone in the horizon.
"""

from __future__ import annotations

import math

import pytest

from cold_start.growing.dp import (
    REFINE,
    SEARCH,
    canonical,
    posterior_mean,
    solve,
)

# Fixed toy environment used wherever the exact numbers are hand-computed.
LO, HI, P = 0.2, 0.8, 0.5


def _same(a, b) -> bool:
    """StateValue equality that treats NaN as equal.

    An illegal action stores NaN deliberately, and ``nan != nan``, so plain ``==``
    would report every terminal state as "different" from itself.
    """
    for field in ("value", "action", "q_search", "q_refine", "advantage", "refine_arm"):
        x, y = getattr(a, field), getattr(b, field)
        if isinstance(x, float) and math.isnan(x) and isinstance(y, float) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


# ---- posterior and terminal condition -----------------------------------------


def test_posterior_mean_matches_hand_computation():
    """One success from one pull: weights 0.5*0.8 vs 0.5*0.2, so P(hi) = 0.8."""
    assert posterior_mean(1, 1, LO, HI, P) == pytest.approx(0.8 * 0.8 + 0.2 * 0.2), "pm(1,1)"
    assert posterior_mean(1, 0, LO, HI, P) == pytest.approx(0.2 * 0.8 + 0.8 * 0.2), "pm(1,0)"
    assert posterior_mean(0, 0, LO, HI, P) == pytest.approx(P * HI + (1 - P) * LO), "prior"


def test_terminal_value_is_the_best_available_posterior_mean():
    """At t = T there is no action left; the value is what the best arm is believed worth."""
    sol = solve(3, LO, HI, P, max_arms=3)
    for (remaining, arms), sv in sol.states.items():
        if remaining != 0:
            continue
        best = max(posterior_mean(n, s, LO, HI, P) for n, s in arms)
        assert sv.value == pytest.approx(best), f"terminal {arms}: {sv.value} != {best}"
        assert math.isnan(sv.advantage), f"terminal {arms} reported an advantage"


def test_terminal_value_with_no_arms_is_the_reservoir_prior_mean():
    """Degenerate T = 0: nothing was ever drawn, so an unseen draw is all there is."""
    sol = solve(0, LO, HI, P, max_arms=3)
    assert sol.value == pytest.approx(P * HI + (1 - P) * LO), f"got {sol.value}"


# ---- one step remaining: hand-computed comparisons -----------------------------


def test_refining_a_lone_arm_with_one_step_left_is_worth_its_posterior_mean():
    """The martingale check.

    With a single arm and one pull left the value is
    ``P(x=1) pm(n+1, S+1) + P(x=0) pm(n+1, S)``, and since ``P(x=1) = pm(n, S)`` the
    tower property collapses that to ``pm(n, S)`` exactly. If this drifts, the
    posterior or the predictive is wrong.
    """
    sol = solve(1, LO, HI, P, max_arms=3, start_arms=[(1, 0)])
    sv = sol.lookup(1, [(1, 0)])
    assert sv.q_refine == pytest.approx(posterior_mean(1, 0, LO, HI, P)), f"got {sv.q_refine}"


def test_one_step_search_beats_refine_against_a_losing_incumbent():
    """Hand-computed: pm(1,0)=0.32, pm(1,1)=0.68, a fresh pull is a 1 w.p. 0.5.

    SEARCH is worth 0.5*max(0.68, 0.32) + 0.5*max(0.32, 0.32) = 0.5, REFINE is worth
    0.32 by the martingale argument above, so A = +0.18 and SEARCH is optimal.
    """
    sol = solve(1, LO, HI, P, max_arms=3, start_arms=[(1, 0)])
    sv = sol.lookup(1, [(1, 0)])

    assert sv.q_search == pytest.approx(0.5), f"q_search={sv.q_search}"
    assert sv.q_refine == pytest.approx(0.32), f"q_refine={sv.q_refine}"
    assert sv.advantage == pytest.approx(0.18), f"A={sv.advantage}"
    assert sv.action == SEARCH, f"action={sv.action}"


def test_one_step_search_ties_refine_against_the_best_possible_incumbent():
    """With one step left a fresh arm can at best match a (1,1) incumbent, so A = 0.

    The tie must resolve to REFINE: SEARCH is taken only when strictly better, which
    is the same convention the deployable rule `SEARCH iff Phi > 0` uses.
    """
    sol = solve(1, LO, HI, P, max_arms=3, start_arms=[(1, 1)])
    sv = sol.lookup(1, [(1, 1)])
    assert sv.advantage == pytest.approx(0.0, abs=1e-12), f"A={sv.advantage}"
    assert sv.action == REFINE, f"tie resolved to {sv.action}"


# ---- structural properties ------------------------------------------------------


def test_optimal_value_is_non_decreasing_in_the_horizon():
    """Extra budget cannot hurt: the payoff is a max of martingales, so E[max] only grows."""
    values = []
    for horizon in range(0, 9):
        values.append(solve(horizon, LO, HI, 0.4, max_arms=4).value)
    for earlier, later in zip(values[:-1], values[1:], strict=True):
        assert later >= earlier - 1e-12, f"value decreased with horizon: {values}"
    assert values[-1] > values[0], f"more budget bought nothing at all: {values}"


def test_hi_rich_reservoir_makes_search_far_more_attractive_than_a_hi_poor_one():
    """Same state, different reservoirs.

    At a losing incumbent with one pull left, A = mean(reservoir) * (pm(1,1) - pm(1,0)):
    a fresh draw is worth taking only to the extent it can outscore the incumbent.
    Both factors collapse as p -> 0, so a hi-poor reservoir makes SEARCH nearly
    worthless while a hi-rich one makes it decisive.
    """
    advantages = {}
    for p in (0.9, 0.05, 0.001):
        sol = solve(1, 0.1, 0.9, p, max_arms=3, start_arms=[(1, 0)])
        advantages[p] = sol.exact_advantage(1, [(1, 0)])

    assert advantages[0.9] > advantages[0.05] > advantages[0.001] > 0.0, f"{advantages}"
    assert advantages[0.9] > 100 * advantages[0.001], f"{advantages}"


def test_almost_surely_hi_reservoir_beats_an_almost_surely_lo_one():
    """The literal `p ~ 1` vs `p ~ 0` comparison the validator is asked to recover."""
    rich = solve(1, 0.1, 0.9, 0.999, max_arms=3, start_arms=[(1, 0)]).exact_advantage(1, [(1, 0)])
    poor = solve(1, 0.1, 0.9, 0.001, max_arms=3, start_arms=[(1, 0)]).exact_advantage(1, [(1, 0)])
    assert rich > poor > 0.0, f"rich={rich}, poor={poor}"


def test_degenerate_reservoir_makes_search_worthless_everywhere():
    """With p = 0 every arm is mu_lo, so searching buys exactly nothing.

    The DP must return A = 0 at every state where both actions are legal, and pick
    SEARCH only at the root, where having no arm at all makes REFINE illegal.
    """
    sol = solve(6, LO, HI, 0.0, max_arms=4)
    assert sol.value == pytest.approx(LO), f"value={sol.value}"

    searched = 0
    for key, sv in sol.states.items():
        if math.isnan(sv.q_search) or math.isnan(sv.q_refine):
            continue
        assert sv.advantage == pytest.approx(0.0, abs=1e-12), f"A={sv.advantage} at {key}"
        if sv.action == SEARCH:
            searched += 1
    assert searched == 0, f"{searched} state(s) preferred a worthless SEARCH"
    assert sol.action_counts()[SEARCH] == 1, f"expected only the forced root: {sol.action_counts()}"


def test_hi_rich_reservoir_chooses_search_at_many_states():
    """The counterpart: with a genuinely rewarding reservoir, SEARCH is often optimal."""
    sol = solve(6, LO, HI, 0.9, max_arms=4)
    assert sol.action_counts()[SEARCH] > 10, f"counts={sol.action_counts()}"


def test_max_arms_makes_search_illegal_once_the_cap_is_reached():
    sol = solve(4, LO, HI, P, max_arms=1)
    assert sol.n_decision_states() == 0, "a one-arm cap should leave no genuine choice"
    for (remaining, arms), sv in sol.states.items():
        if remaining > 0 and len(arms) == 1:
            assert math.isnan(sv.q_search), f"SEARCH stayed legal at {arms} with max_arms=1"


# ---- reachable-state frontier ---------------------------------------------------


def test_reachable_state_counts_are_reported_and_grow_with_the_horizon():
    """The tractable frontier, measured rather than assumed.

    Memoized recursion from the root visits only reachable states; the enumeration
    form blows up on multisets that no trajectory can produce. These counts are the
    regression anchor for that claim (mu_lo=0.2, mu_hi=0.8, p=0.5, max_arms=5).
    T = 12 stays out of the suite at ~10.5k states, which is the practical frontier.
    """
    counts = {}
    for horizon in (6, 8, 10):
        counts[horizon] = solve(horizon, LO, HI, P, max_arms=5).n_states

    assert counts[6] == 268, f"T=6 reachable states: {counts[6]}"
    assert counts[8] == 1047, f"T=8 reachable states: {counts[8]}"
    assert counts[10] == 3519, f"T=10 reachable states: {counts[10]}"
    assert counts[6] < counts[8] < counts[10], f"counts not increasing: {counts}"


def test_solve_is_deterministic():
    first = solve(7, LO, HI, 0.3, max_arms=4)
    second = solve(7, LO, HI, 0.3, max_arms=4)
    assert first.value == second.value, f"{first.value} != {second.value}"
    assert first.states.keys() == second.states.keys(), "different states were reached"
    for key, sv in first.states.items():
        other = second.states[key]
        assert _same(sv, other), f"state {key} differs: {sv} vs {other}"


# ---- query surface ---------------------------------------------------------------


def test_lookup_canonicalizes_arm_order():
    """Arms are exchangeable, so the caller's ordering must not matter.

    Reachability ties the two halves of a key together: total pulls always equal
    ``T - remaining``, so this state is (1+2 pulls) at 2 steps left of a 5-step run.
    """
    sol = solve(5, LO, HI, P, max_arms=4)
    a = sol.lookup(2, [(1, 0), (2, 2)])
    b = sol.lookup(2, [(2, 2), (1, 0)])
    assert a is b, "canonicalization returned two different memo entries"
    assert canonical([(2, 2), (1, 0)]) == ((1, 0), (2, 2)), "canonical order is not sorted"


def test_optimal_action_and_advantage_agree_with_the_stored_state():
    sol = solve(4, LO, HI, P, max_arms=3)
    sv = sol.lookup(3, [(1, 1)])
    assert sol.optimal_action(3, [(1, 1)]) == sv.action, "action query disagrees"
    assert sol.exact_advantage(3, [(1, 1)]) == sv.advantage, "advantage query disagrees"


def test_unreachable_state_lookup_raises():
    sol = solve(2, LO, HI, P, max_arms=3)
    with pytest.raises(KeyError, match="not reachable"):
        sol.lookup(2, [(9, 9)])


def test_invalid_arm_counts_are_rejected():
    with pytest.raises(ValueError, match="invalid arm counts"):
        canonical([(1, 2)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"T": -1},
        {"max_arms": 0},
        {"mu_lo": 0.9, "mu_hi": 0.1},
        {"p": 1.5},
        {"mu_hi": 1.5},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    args = {"T": 3, "mu_lo": LO, "mu_hi": HI, "p": P, "max_arms": 3}
    args.update(kwargs)
    with pytest.raises(ValueError):
        solve(**args)
