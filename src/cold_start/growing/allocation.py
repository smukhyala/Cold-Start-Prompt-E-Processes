"""REFINE rules: which existing arm to re-evaluate.

All three rules here are *eliminating*. That is the point, and it is the direct
answer to the estimation model this study was set up to avoid: a term like
sqrt(K/t) assumes every discovered arm needs comparable estimation precision, which
is far too pessimistic because a good pure-exploration rule stops sampling weak arms
almost immediately. Here an arm with `ucb < max_j lcb_j` is provably not the best
and receives no further pulls, so K bad arms cost O(K) pulls in total rather than
O(K) pulls each.

Every rule is vectorized over the M Monte Carlo replicates and returns one column
index per replicate. Scores are f32 (argmax on f32 is ~2x faster than f64) and the
per-arm tiebreak jitter is added before every argmax, because `np.argmax` resolves
ties at the lowest index and a freshly searched arm always sits in the highest slot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cold_start.growing.state import NEG_SENTINEL, GrowingState
from cold_start.registry import register

_POS_SENTINEL = np.float32(1e30)


def leader_and_challenger(state: GrowingState) -> tuple[np.ndarray, np.ndarray]:
    """Return (leader, challenger) column indices per replicate.

    Leader is argmax of the lower confidence bound -- the arm we would currently
    commit to. Challenger is the argmax of the upper bound among the rest -- the arm
    with the strongest remaining case for overtaking it. Their separation is the
    quantity the pairwise e-process is built on.
    """
    lcb = state.view(state.lcb)
    ucb = state.view(state.ucb)
    tie = state.view(state.tie)

    leader = np.argmax(lcb + tie, axis=1)
    lin_leader = state.row_off + leader

    saved = state.ucb[lin_leader].copy()
    state.ucb[lin_leader] = NEG_SENTINEL
    challenger = np.argmax(ucb + tie, axis=1)
    state.ucb[lin_leader] = saved

    return leader.astype(np.int64), challenger.astype(np.int64)


def plausible_mask(state: GrowingState) -> np.ndarray:
    """(M, Kmax) mask of arms not yet eliminated: ucb_i >= max_j lcb_j."""
    lcb = state.view(state.lcb)
    ucb = state.view(state.ucb)
    best_lcb = lcb.max(axis=1, keepdims=True)
    return (ucb >= best_lcb) & state.active_mask()


class AllocationRule(ABC):
    """Choose one existing arm per replicate to re-evaluate."""

    name: str = "abstract"

    @abstractmethod
    def select(self, state: GrowingState) -> np.ndarray:
        """Return a column index per replicate (int64, length M)."""

    def _masked_scores(self, state: GrowingState, score: np.ndarray) -> np.ndarray:
        """Blank out eliminated and inactive arms, then add the tiebreak jitter."""
        out = np.where(plausible_mask(state), score, NEG_SENTINEL).astype(np.float32)
        return out + state.view(state.tie)


@register("ucb", kind="allocation")
class UCBChallenger(AllocationRule):
    """Pull the surviving arm with the highest upper confidence bound."""

    name = "ucb"

    def select(self, state: GrowingState) -> np.ndarray:
        scores = self._masked_scores(state, state.view(state.ucb))
        return np.argmax(scores, axis=1).astype(np.int64)


@register("lucb", kind="allocation")
class LUCB(AllocationRule):
    """Alternate between the leader and its strongest challenger.

    Sampling whichever of the two is currently less certain is what drives their
    confidence sequences apart fastest, which is the quantity that decides whether
    the best arm can be certified before the budget runs out.
    """

    name = "lucb"

    def select(self, state: GrowingState) -> np.ndarray:
        """Alternate between leader and challenger by pull count, not by CS width.

        Tie-breaking on width looks natural -- sample whichever is less certain -- but
        it is wrong for Bernoulli rewards, because confidence-sequence width depends on
        the success rate as well as the sample size: an arm near 0.5 is intrinsically
        wider at *every* n than an arm near 0.9. A width rule therefore locks onto
        mid-range arms and never leaves. Measured on a leader at 0.85 against a
        challenger at 0.55, it spent 18.4 pulls on the inferior arm and only 10.2 on
        the leader, and the resulting recommendation was correct 91.5% of the time
        against 99.1% for a branch that left the pair alone.

        Balancing pull counts is the standard LUCB behaviour (sample both the leader
        and the challenger each round) expressed in a one-pull-per-round setting, and
        it has no dependence on where the arms sit in [0, 1].
        """
        leader, challenger = leader_and_challenger(state)
        n_l = state.n[state.row_off + leader]
        n_c = state.n[state.row_off + challenger]

        # The challenger must come from the SURVIVING set, and the degenerate guard
        # must count survivors rather than discovered arms. `leader_and_challenger`
        # takes argmax over every non-leader slot, which is harmless while two or more
        # arms are plausible (any plausible arm's ucb beats any eliminated one's) but
        # wrong the moment the leader is the only survivor: the challenger is then
        # necessarily the best ELIMINATED arm, and a `Kt <= 1` guard never fires, so
        # LUCB alternates leader/eliminated for the rest of the run. That contradicts
        # this module's own contract, and it is the same shape as the width-tie-break
        # defect: a selector not restricted to the survivor set.
        #
        # Measured impact on the labels is small -- once one arm survives the
        # recommendation is already settled, so the misspent budget has little
        # marginal value -- but a continuation policy that knowingly spends half its
        # remaining pulls on a provably eliminated arm is not the strong continuation
        # policy the oracle label is supposed to be defined against.
        resolved = n_plausible(state) <= 1
        pick_challenger = (n_c <= n_l) & ~resolved
        return np.where(pick_challenger, challenger, leader)


@register("racing", kind="allocation")
class WidthRacing(AllocationRule):
    """Pull the widest-interval member of the plausible-winner set.

    Uniform-over-survivors in spirit, but weighted by uncertainty: it spends the
    budget where the frontier is least resolved rather than where the mean is
    highest, which keeps it from locking onto an early lucky arm.
    """

    name = "racing"

    def select(self, state: GrowingState) -> np.ndarray:
        width = state.view(state.ucb) - state.view(state.lcb)
        scores = self._masked_scores(state, width)
        return np.argmax(scores, axis=1).astype(np.int64)


def n_plausible(state: GrowingState) -> np.ndarray:
    """Number of surviving candidate winners per replicate."""
    return plausible_mask(state).sum(axis=1).astype(np.int32)


def n_eliminated(state: GrowingState) -> np.ndarray:
    """Number of discovered arms proven not to be the best, per replicate."""
    return (state.Kt - n_plausible(state)).astype(np.int32)
