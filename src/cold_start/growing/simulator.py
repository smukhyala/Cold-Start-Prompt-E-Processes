"""The vectorized rollout loop: one step advances all M replicates at once.

A "step" is one unit of evaluation budget. Each replicate independently either
SEARCHes (draw a fresh arm from the reservoir and pull it once) or REFINEs (re-pull
an existing arm chosen by the allocation rule). Both happen in the same pass -- the
two groups are masked, not looped -- which is what keeps the cost per replicate-pull
in the hundreds of nanoseconds rather than the tens of microseconds a per-trajectory
loop would cost.

The `force` argument is the hinge the whole study turns on: it pins the next action
for every replicate, which is how the oracle labeller evaluates "what if we SEARCH
here" against "what if we REFINE here" from an identical starting state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cold_start.growing import rng as crn
from cold_start.growing.allocation import AllocationRule
from cold_start.growing.search_policies import DecisionContext, SearchPolicy
from cold_start.growing.state import (
    EMPTY_UID,
    BoundTable,
    ForcedActionUnavailable,
    GrowingState,
)

# Domain tag for the reservoir stream (must differ from the reward/tiebreak domains).
_RESERVOIR_DOMAIN = 1


class Reservoir(Protocol):
    """Infinite pool of candidate arms.

    `sample_from_uniforms` is required rather than optional: the paired rollouts feed
    a shared uniform stream so the j-th arm a branch discovers is the same underlying
    draw in both branches. A family without an inverse-CDF path would break that.
    """

    def sample_from_uniforms(self, u: np.ndarray) -> np.ndarray: ...
    def essential_sup(self) -> float: ...


@dataclass
class StepResult:
    """What happened in one step, per replicate."""

    searched: np.ndarray
    cols: np.ndarray
    reward: np.ndarray


class Simulator:
    """Advances a `GrowingState` under a SEARCH policy and a REFINE allocation rule."""

    def __init__(
        self,
        table: BoundTable,
        reservoir: Reservoir,
        allocation: AllocationRule,
        search_policy: SearchPolicy,
        horizon: int,
        max_live_arms: int = 64,
        pairwise_fn: Callable[[GrowingState], np.ndarray] | None = None,
    ) -> None:
        self.table = table
        self.reservoir = reservoir
        self.allocation = allocation
        self.search_policy = search_policy
        self.horizon = int(horizon)
        self.max_live_arms = int(max_live_arms)
        self.pairwise_fn = pairwise_fn

    # ---- context ---------------------------------------------------------------

    def best_posterior_mean(self, state: GrowingState) -> np.ndarray:
        """Highest Beta(1,1) posterior mean among active arms, per replicate."""
        pm = state.view(state.empirical_mean().astype(np.float32))
        masked = np.where(state.active_mask(), pm, -np.inf)
        out = masked.max(axis=1)
        return np.where(np.isfinite(out), out, 0.0)

    def context(self, state: GrowingState, t: int) -> DecisionContext:
        """Build only the statistics this policy actually reads.

        Evidence statistics cost a full pass over the arm array, so a schedule-style
        policy must not pay for them.
        """
        ctx = DecisionContext(t=t, horizon=self.horizon)
        if self.search_policy.needs_evidence:
            ctx.best_mean = self.best_posterior_mean(state)
            if self.pairwise_fn is not None:
                ctx.log_e_pair = self.pairwise_fn(state)
            else:
                ctx.log_e_pair = np.zeros(state.M, dtype=np.float64)
        return ctx

    # ---- the step --------------------------------------------------------------

    def decide(
        self,
        state: GrowingState,
        t: int,
        force: np.ndarray | bool | None = None,
        strict: bool = False,
    ) -> np.ndarray:
        """Return the per-replicate SEARCH decision, after applying hard constraints."""
        if force is None:
            decision = self.search_policy.should_search(state, self.context(state, t))
        elif isinstance(force, np.ndarray):
            decision = np.asarray(force, dtype=bool).copy()
        else:
            decision = np.full(state.M, bool(force))

        # A replicate with no arms has nothing to refine, so it must search.
        decision |= state.Kt == 0

        # The live-arm cap must not silently override the action that DEFINES a label.
        # Demoting a forced SEARCH to a REFINE makes both oracle branches identical, so
        # the labeller records A_t = 0 with SE = 0 -- a fabricated label that looks
        # perfectly precise and stops the adaptive rule on its first batch.
        #
        # `strict` marks the step whose action the label is defined by, which is the
        # first one. Later steps of a multi-round commitment may legitimately hit the
        # cap: "commit to searching for k rounds" means "for as many of them as the cap
        # allows", and the label is still well defined because the first action was
        # honoured.
        at_cap = state.Kt >= self.max_live_arms
        if strict and force is not None and bool((decision & at_cap).any()):
            raise ForcedActionUnavailable(
                f"{int((decision & at_cap).sum())} of {state.M} replicates are at the "
                f"live-arm cap ({self.max_live_arms}), so the forced SEARCH cannot be "
                f"applied; label this state as undefined rather than as A_t = 0"
            )
        decision &= ~at_cap
        return decision

    def step(
        self,
        state: GrowingState,
        t: int,
        force: np.ndarray | bool | None = None,
        strict: bool = False,
    ) -> StepResult:
        do_search = self.decide(state, t, force, strict=strict)

        # Choose the REFINE target before any new arm exists, so a freshly searched
        # arm is never also a refinement candidate in the same step.
        has_arms = state.Kt > 0
        refine_cols = np.zeros(state.M, dtype=np.int64)
        if has_arms.any():
            refine_cols = self.allocation.select(state)

        new_cols = state.Kt.astype(np.int64)
        if do_search.any():
            draw_index = state.n_draws.astype(np.int64)
            u = crn.reservoir_uniforms(state.base_seed, np.arange(state.M), draw_index)
            mu_new = np.asarray(self.reservoir.sample_from_uniforms(u), dtype=np.float32)
            state.add_arms(mu_new, draw_index.astype(np.int32), where=do_search)
            state.n_draws += do_search

        cols = np.where(do_search, new_cols, refine_cols)
        reward = state.pull(cols, self.table)
        state.t = t + 1
        return StepResult(searched=do_search, cols=cols, reward=reward)

    def run_to_horizon(
        self, state: GrowingState, start_t: int, force_first: np.ndarray | bool | None = None
    ) -> GrowingState:
        """Advance from `start_t` to the horizon, optionally pinning the first action.

        Both oracle branches call this with the SAME policy and differ only in
        `force_first`. That is the whole design: anything else and the label would
        measure the continuation policy rather than the action under test.
        """
        t = start_t
        if t < self.horizon:
            self.step(state, t, force=force_first)
            t += 1
        while t < self.horizon:
            self.step(state, t, force=None)
            t += 1
        return state


def seed_initial_arms(
    state: GrowingState, reservoir: Reservoir, n_arms: int, table: BoundTable
) -> None:
    """Open a trajectory with `n_arms` arms, each pulled once.

    Draws come from the same indexed reservoir stream the simulator uses, so a state
    built here and a state reached by searching are indistinguishable in how their
    randomness is addressed.
    """
    for _ in range(n_arms):
        draw_index = state.n_draws.astype(np.int64)
        u = crn.reservoir_uniforms(state.base_seed, np.arange(state.M), draw_index)
        mu_new = np.asarray(reservoir.sample_from_uniforms(u), dtype=np.float32)
        cols = state.add_arms(mu_new, draw_index.astype(np.int32))
        state.n_draws += 1
        state.pull(cols, table)


def live_arm_counts(state: GrowingState) -> np.ndarray:
    return (state.view(state.uid) != EMPTY_UID).sum(axis=1).astype(np.int32)
