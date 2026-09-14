"""Exact backward induction for the toy two-point-reservoir environment.

This is the only place in the study where a SEARCH-vs-REFINE label can be checked
against *ground truth* instead of against another approximation. Everything else --
the Monte Carlo labeller, the fitted policy -- is validated by agreeing with what
this module computes, so correctness here outranks speed everywhere.

The environment. Bernoulli arms whose means are drawn from a two-point reservoir:
``mu = mu_hi`` with probability ``p``, else ``mu_lo``. At each of ``T`` steps the
agent either REFINEs (pull an arm it already has) or SEARCHes (draw a fresh arm from
the reservoir and pull it once), subject to ``max_arms``. At ``T`` it names one arm
and is paid that arm's true mean, so the objective is ``E[mu(recommended)]``.

The sufficient statistic. The agent never observes ``mu``. Its belief about arm ``i``
is a two-point posterior determined by ``(n_i, S_i)`` and ``p`` alone, and arms are
exchangeable, so the state is ``(steps remaining, multiset of (n_i, S_i))``. The
canonical sorted tuple of those pairs is the memo key.

**Recurse lazily from the root; do not enumerate the state space.** The enumeration
form -- build every multiset of ``(n, S)`` pairs consistent with the budget, then
sweep backwards -- was tried first and never produced output at all: the multiset
count explodes long before the *reachable* set does, because the overwhelming
majority of those multisets are unreachable (they spend more pulls than the budget
allows, or split them in ways no trajectory can produce). Memoized recursion from the
root touches only what a trajectory can actually reach, and `DPSolution.n_states`
reports that count so the tractable frontier is a measured number.

Two further prunings, both exactness-preserving:

* Zero-probability transitions are skipped. At ``mu_hi = 1`` a fresh arm can never
  return a 0, and following that branch would both invent unreachable states and ask
  for a posterior conditioned on impossible data.
* Arms sharing an ``(n, S)`` are refined once, not once each. They are exchangeable,
  so the successor states are identical.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: An arm summarized by (pulls, successes).
ArmCounts = tuple[int, int]
#: Canonical (sorted) multiset of arms.
ArmSet = tuple[ArmCounts, ...]
#: (steps remaining, canonical arm multiset).
StateKey = tuple[int, ArmSet]

SEARCH = "SEARCH"
REFINE = "REFINE"
RECOMMEND = "RECOMMEND"

_NAN = float("nan")

#: Ties are declared within this absolute gap in expected reward.
#: Exact ties are *common* here rather than exotic: the posterior mean is a
#: martingale, so e.g. refining a lone arm on the last step is worth exactly its
#: current posterior mean, which a fresh arm can match but not beat. Summing the
#: two Q-values in different orders then leaves a few ulps of binary-float noise,
#: and without a tolerance that noise -- not the decision problem -- would pick the
#: reported action. 1e-12 is ~1e4 ulps at these magnitudes and still far below any
#: advantage the study cares about (target label precision is 5e-3).
TIE_TOL = 1e-12


@dataclass(frozen=True)
class StateValue:
    """The exact solution at one reachable state.

    ``q_search`` / ``q_refine`` are NaN when that action is illegal (no arms yet, or
    ``max_arms`` reached), and ``advantage`` inherits the NaN rather than pretending a
    comparison happened. ``refine_arm`` indexes the *canonical sorted* arm tuple.
    """

    value: float
    action: str
    q_search: float
    q_refine: float
    advantage: float
    refine_arm: int


@dataclass(frozen=True)
class DPSolution:
    """Optimal value and action at every state reachable from the root."""

    horizon: int
    mu_lo: float
    mu_hi: float
    p: float
    max_arms: int
    root: StateKey
    states: dict[StateKey, StateValue]

    @property
    def value(self) -> float:
        """``E[mu(recommended)]`` under optimal play from the root."""
        return self.states[self.root].value

    @property
    def n_states(self) -> int:
        """Number of reachable states -- the measured size of the tractable frontier."""
        return len(self.states)

    def lookup(self, remaining: int, arms: Iterable[ArmCounts]) -> StateValue:
        """Exact solution at a state, canonicalizing `arms` for the caller.

        This is the comparison point for the Monte Carlo labeller: hand it the same
        ``(n_i, S_i)`` multiset the rollouts started from and compare signs and
        magnitudes against `StateValue.advantage`.
        """
        key = canonical_key(remaining, arms)
        try:
            return self.states[key]
        except KeyError:
            raise KeyError(
                f"state {key} is not reachable from root {self.root} under "
                f"T={self.horizon}, max_arms={self.max_arms}"
            ) from None

    def optimal_action(self, remaining: int, arms: Iterable[ArmCounts]) -> str:
        return self.lookup(remaining, arms).action

    def exact_advantage(self, remaining: int, arms: Iterable[ArmCounts]) -> float:
        """``V(SEARCH) - V(REFINE)`` at a state; positive means SEARCH is better."""
        return self.lookup(remaining, arms).advantage

    def action_counts(self) -> dict[str, int]:
        counts = {SEARCH: 0, REFINE: 0, RECOMMEND: 0}
        for sv in self.states.values():
            counts[sv.action] += 1
        return counts

    def n_decision_states(self) -> int:
        """States where both actions are legal -- the only ones with a meaningful A."""
        total = 0
        for sv in self.states.values():
            if not (math.isnan(sv.q_search) or math.isnan(sv.q_refine)):
                total += 1
        return total


def canonical(arms: Iterable[ArmCounts]) -> ArmSet:
    """Sorted tuple of ``(n, S)`` pairs: the exchangeability-collapsed arm multiset."""
    out: list[ArmCounts] = []
    for n, s in arms:
        n_i = int(n)
        s_i = int(s)
        if n_i < 0 or s_i < 0 or s_i > n_i:
            raise ValueError(f"invalid arm counts (n={n_i}, S={s_i}); need 0 <= S <= n")
        out.append((n_i, s_i))
    out.sort()
    return tuple(out)


def canonical_key(remaining: int, arms: Iterable[ArmCounts]) -> StateKey:
    return (int(remaining), canonical(arms))


def posterior_hi(n: int, s: int, mu_lo: float, mu_hi: float, p: float) -> float:
    """``P(mu = mu_hi | n pulls, S successes)`` under the two-point prior.

    Exact likelihood ratio rather than any Beta approximation -- the reservoir here
    genuinely has two atoms, and the point of this module is to be the ground truth.
    """
    w_hi = p * (mu_hi**s) * ((1.0 - mu_hi) ** (n - s))
    w_lo = (1.0 - p) * (mu_lo**s) * ((1.0 - mu_lo) ** (n - s))
    total = w_hi + w_lo
    if total <= 0.0:
        # Data impossible under both atoms (e.g. mu_lo=0, mu_hi=1, mixed outcomes).
        # Unreachable in the recursion because zero-probability branches are pruned;
        # falling back to the prior keeps a direct call total instead of dividing by 0.
        return p
    return w_hi / total


def posterior_mean(n: int, s: int, mu_lo: float, mu_hi: float, p: float) -> float:
    """``E[mu | n, S]``. Also the predictive ``P(next pull = 1 | n, S)``."""
    w = posterior_hi(n, s, mu_lo, mu_hi, p)
    return w * mu_hi + (1.0 - w) * mu_lo


def _validate(T: int, mu_lo: float, mu_hi: float, p: float, max_arms: int) -> None:
    if T < 0:
        raise ValueError(f"horizon must be >= 0, got {T}")
    if max_arms < 1:
        raise ValueError(f"max_arms must be >= 1, got {max_arms}")
    if not 0.0 <= mu_lo <= 1.0 or not 0.0 <= mu_hi <= 1.0:
        raise ValueError(f"means must lie in [0, 1], got mu_lo={mu_lo}, mu_hi={mu_hi}")
    if mu_hi < mu_lo:
        raise ValueError(f"expected mu_lo <= mu_hi, got mu_lo={mu_lo}, mu_hi={mu_hi}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be a probability, got {p}")


def solve(
    T: int,
    mu_lo: float,
    mu_hi: float,
    p: float,
    max_arms: int = 5,
    start_arms: Sequence[ArmCounts] = (),
) -> DPSolution:
    """Solve the toy environment exactly by memoized backward induction.

    `start_arms` lets the solver be rooted at an arbitrary already-observed arm
    multiset, which is how a Monte Carlo-labelled state is checked: solve from that
    state's own counts and compare against `DPSolution.exact_advantage`.

    Ties (within `TIE_TOL`) are broken toward REFINE -- SEARCH is chosen only when
    strictly better.
    That matches the deployable rule the study is aiming at (``SEARCH iff Phi > 0``)
    and makes a degenerate reservoir (``p = 0``, where search buys exactly nothing)
    report REFINE everywhere instead of an arbitrary coin flip.
    """
    _validate(T, mu_lo, mu_hi, p, max_arms)

    prior_mean = p * mu_hi + (1.0 - p) * mu_lo
    states: dict[StateKey, StateValue] = {}
    pm_cache: dict[ArmCounts, float] = {}

    def pm(arm: ArmCounts) -> float:
        cached = pm_cache.get(arm)
        if cached is None:
            cached = posterior_mean(arm[0], arm[1], mu_lo, mu_hi, p)
            pm_cache[arm] = cached
        return cached

    def terminal_value(arms: ArmSet) -> float:
        # Nothing observed means nothing to name; the best the agent can do is take an
        # unseen draw, worth the reservoir's prior mean. Only reachable at T = 0.
        if not arms:
            return prior_mean
        best = pm(arms[0])
        for arm in arms[1:]:
            candidate = pm(arm)
            if candidate > best:
                best = candidate
        return best

    def visit(remaining: int, arms: ArmSet) -> StateValue:
        key = (remaining, arms)
        hit = states.get(key)
        if hit is not None:
            return hit

        if remaining == 0:
            sv = StateValue(
                value=terminal_value(arms),
                action=RECOMMEND,
                q_search=_NAN,
                q_refine=_NAN,
                advantage=_NAN,
                refine_arm=-1,
            )
            states[key] = sv
            return sv

        # --- SEARCH: draw a fresh arm, pull it once. Its posterior is (1, x). ---
        q_search = _NAN
        if len(arms) < max_arms:
            total = 0.0
            if prior_mean > 0.0:
                total += prior_mean * visit(remaining - 1, canonical(arms + ((1, 1),))).value
            if prior_mean < 1.0:
                total += (1.0 - prior_mean) * visit(
                    remaining - 1, canonical(arms + ((1, 0),))
                ).value
            q_search = total

        # --- REFINE: pull one existing arm; maximize over which. ---
        q_refine = _NAN
        best_arm = -1
        seen: set[ArmCounts] = set()
        for i, arm in enumerate(arms):
            if arm in seen:
                # Exchangeable duplicate: identical successors, already evaluated.
                continue
            seen.add(arm)
            n_i, s_i = arm
            p_one = pm(arm)
            total = 0.0
            if p_one > 0.0:
                nxt = arms[:i] + ((n_i + 1, s_i + 1),) + arms[i + 1 :]
                total += p_one * visit(remaining - 1, canonical(nxt)).value
            if p_one < 1.0:
                nxt = arms[:i] + ((n_i + 1, s_i),) + arms[i + 1 :]
                total += (1.0 - p_one) * visit(remaining - 1, canonical(nxt)).value
            if best_arm < 0 or total > q_refine:
                q_refine = total
                best_arm = i

        if best_arm < 0:
            action = SEARCH
            value = q_search
        elif math.isnan(q_search):
            action = REFINE
            value = q_refine
        else:
            # `value` takes the exact max so V stays optimal to the last ulp; `action`
            # uses the tolerant comparison so a tie is reported as REFINE rather than
            # decided by rounding. The two differ by at most TIE_TOL, by construction.
            value = q_search if q_search > q_refine else q_refine
            action = SEARCH if q_search > q_refine + TIE_TOL else REFINE

        sv = StateValue(
            value=value,
            action=action,
            q_search=q_search,
            q_refine=q_refine,
            advantage=q_search - q_refine,
            refine_arm=best_arm,
        )
        states[key] = sv
        return sv

    root_arms = canonical(start_arms)
    if len(root_arms) > max_arms:
        raise ValueError(f"start_arms has {len(root_arms)} arms, exceeding max_arms={max_arms}")
    visit(T, root_arms)

    return DPSolution(
        horizon=int(T),
        mu_lo=float(mu_lo),
        mu_hi=float(mu_hi),
        p=float(p),
        max_arms=int(max_arms),
        root=(int(T), root_arms),
        states=states,
    )
