"""SEARCH-vs-REFINE decision rules.

Two distinct uses, deliberately sharing one interface:

* **Behavioral policies** generate the state pool. We need states from a *mixture* of
  many rules, because a corpus harvested from a single policy only covers the states
  that policy visits -- and a decision function fitted on it would be learning that
  policy's habits rather than the underlying trade-off. This is the main threat to
  the whole study, so the mixture is wide on purpose.
* **Continuation policies** run inside the oracle rollouts after the forced first
  action. Both branches must use the *same* one, or the label measures the
  continuation policy rather than the action.

Every rule is vectorized: `should_search` returns one boolean per replicate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from cold_start.growing.allocation import n_plausible
from cold_start.growing.state import GrowingState
from cold_start.registry import register


@dataclass
class DecisionContext:
    """Optional evidence available to a policy at decision time.

    Kept optional so schedule-style policies never pay for statistics they ignore.
    """

    t: int
    horizon: int
    log_e_pair: np.ndarray | None = None
    best_mean: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return self.horizon - self.t


class SearchPolicy(ABC):
    """Decide, per replicate, whether the next action is SEARCH."""

    name: str = "abstract"
    needs_evidence: bool = False

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)

    @abstractmethod
    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        """Return a boolean array of length M."""

    def _no_arms_yet(self, state: GrowingState) -> np.ndarray:
        """A replicate with no arms must SEARCH; there is nothing else to do."""
        return state.Kt == 0


def _force_first_arm(state: GrowingState, decision: np.ndarray) -> np.ndarray:
    return decision | (state.Kt == 0)


# ---- fixed-probability rules ---------------------------------------------------


@register("bernoulli_search", kind="search_policy")
class BernoulliSearch(SearchPolicy):
    """SEARCH with a fixed probability. `p=0.5` is aggressive, `p=0.05` conservative."""

    name = "bernoulli_search"

    def __init__(self, p: float = 0.2, rng: np.random.Generator | None = None) -> None:
        super().__init__(rng)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0,1]; got {p}")
        self.p = float(p)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        draw = self.rng.random(state.M) < self.p
        return _force_first_arm(state, draw)


# ---- growth schedules ----------------------------------------------------------


@register("power_schedule", kind="search_policy")
class PowerSchedule(SearchPolicy):
    """SEARCH while `K_t < c * t**alpha`.

    Covers the classical family the literature reaches for -- alpha=1/2 is the
    sqrt(t) rule, 1/3 and 2/3 the other common choices. These are the baselines the
    learned rule has to beat, so the state pool must contain the states they visit.
    """

    name = "power_schedule"

    def __init__(
        self, alpha: float = 0.5, c: float = 1.0, rng: np.random.Generator | None = None
    ) -> None:
        super().__init__(rng)
        self.alpha = float(alpha)
        self.c = float(c)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        target = self.c * max(ctx.t, 1) ** self.alpha
        return _force_first_arm(state, state.Kt < target)


def frac_within_of_best(state: GrowingState, band: float) -> np.ndarray:
    """Per replicate, the fraction of held arms whose posterior mean is within `band` of
    the best held arm's -- `est_frac_arms_within_5pct_of_best` at ``band=0.05``, computed
    from ``(S+1)/(n+2)`` alone. 1.0 when a replicate holds no arms."""
    post = state.view(state.empirical_mean())
    held = np.arange(post.shape[1])[None, :] < state.Kt[:, None]
    k = np.maximum(state.Kt, 1).astype(np.float64)
    best = np.where(held, post, -np.inf).max(axis=1)
    near = (held & (post >= (best - band)[:, None])).sum(axis=1)
    return np.where(state.Kt > 0, near / k, 1.0)


@register("adaptive_K", kind="search_policy")
class TailAdaptiveSchedule(SearchPolicy):
    """SEARCH while ``K_t < c * T**alpha * (1 + b * (1 - q_t))`` (Pre-registration 4).

    A fixed-K-per-horizon schedule -- recruit to the target as fast as possible, then
    refine -- whose target grows when the held arms say the reservoir is heavy-tailed:
    ``q_t`` is the fraction of held arms within `band` of the best held arm's posterior
    mean. In a thin-tailed reservoir ``q_t`` is near 1 and the target is the plain
    ``c * T**alpha``; in a heavy-tailed one it is near 0 and the target is up to
    ``1 + b`` times larger. ``b = 0`` is the fixed-K schedule exactly. Three scalars,
    one QUALITY statistic, no confidence sequence and no e-process.
    """

    name = "adaptive_K"

    def __init__(
        self,
        alpha: float = 0.5,
        c: float = 2.0,
        b: float = 0.0,
        band: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.alpha = float(alpha)
        self.c = float(c)
        self.b = float(b)
        self.band = float(band)

    def target(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        base = self.c * float(max(ctx.horizon, 1)) ** self.alpha
        if self.b == 0.0:
            return np.full(state.M, base, dtype=np.float64)
        q = frac_within_of_best(state, self.band)
        return base * (1.0 + self.b * (1.0 - q))

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        return _force_first_arm(state, state.Kt < self.target(state, ctx))


def best_held_mean(state: GrowingState) -> np.ndarray:
    """Per replicate, the largest posterior mean ``(S+1)/(n+2)`` over held arms; 0 with none."""
    post = state.view(state.empirical_mean())
    held = np.arange(post.shape[1])[None, :] < state.Kt[:, None]
    best = np.where(held, post, -np.inf).max(axis=1)
    return np.where(state.Kt > 0, best, 0.0)


@register("bestmean_K", kind="search_policy")
class BestMeanGate(SearchPolicy):
    """SEARCH while ``best_mean_t < theta`` and ``K_t < c * T**alpha`` (Pre-registration 5).

    Recruit until the best arm held is good enough, never beyond a per-horizon ceiling,
    then refine. This is the rule `model_reads.py` found the k = 4 learned policy to be
    reading: its top feature at every horizon is the best held arm's posterior mean.
    ``theta >= 1`` never closes the gate and is the fixed-K schedule exactly.
    """

    name = "bestmean_K"

    def __init__(
        self,
        theta: float = 0.65,
        alpha: float = 0.5,
        c: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.c = float(c)

    def ceiling(self, ctx: DecisionContext) -> float:
        return self.c * float(max(ctx.horizon, 1)) ** self.alpha

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        below_ceiling = state.Kt < self.ceiling(ctx)
        return _force_first_arm(state, below_ceiling & (best_held_mean(state) < self.theta))


def held_level(state: GrowingState) -> np.ndarray:
    """Per replicate, the mean posterior mean ``(S+1)/(n+2)`` over held arms; 0.5 with none."""
    post = state.view(state.empirical_mean())
    held = np.arange(post.shape[1])[None, :] < state.Kt[:, None]
    total = np.where(held, post, 0.0).sum(axis=1)
    return np.where(state.Kt > 0, total / np.maximum(state.Kt, 1), 0.5)


@register("level_K", kind="search_policy")
class LevelScaledSchedule(SearchPolicy):
    """SEARCH while ``K_t < c * T**alpha * exp(b * (0.5 - level_t))`` (Pre-registration 6).

    A fixed-K-per-horizon schedule whose target is scaled by the observed reservoir
    level -- the mean posterior mean over held arms. A high level (thin-tailed reservoir,
    the best arm near the typical one) shrinks the target; a low level (heavy-tailed,
    rare excellent arms far above typical) grows it. `model_reads.py` found this to be
    what the k = 4 learned policy effectively reads. ``b = 0`` is the fixed-K schedule.
    """

    name = "level_K"

    def __init__(
        self,
        alpha: float = 0.5,
        c: float = 2.0,
        b: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.alpha = float(alpha)
        self.c = float(c)
        self.b = float(b)

    def target(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        base = self.c * float(max(ctx.horizon, 1)) ** self.alpha
        if self.b == 0.0:
            return np.full(state.M, base, dtype=np.float64)
        return base * np.exp(self.b * (0.5 - held_level(state)))

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        return _force_first_arm(state, state.Kt < self.target(state, ctx))


@register("epsilon_schedule", kind="search_policy")
class EpsilonSchedule(SearchPolicy):
    """A power schedule with epsilon-randomization.

    Pure schedules are deterministic given (t, K_t), so on their own they trace a
    one-dimensional curve through state space. The noise is what gives the corpus
    coverage *off* that curve, which is where the decision boundary has to be
    estimated.
    """

    name = "epsilon_schedule"

    def __init__(
        self,
        alpha: float = 0.5,
        c: float = 1.0,
        epsilon: float = 0.15,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.inner = PowerSchedule(alpha=alpha, c=c, rng=self.rng)
        self.epsilon = float(epsilon)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        base = self.inner.should_search(state, ctx)
        flip = self.rng.random(state.M) < self.epsilon
        return _force_first_arm(state, np.where(flip, ~base, base))


@register("bracket", kind="search_policy")
class BracketExpansion(SearchPolicy):
    """Successive-halving flavour: recruit a batch of arms, then race them down.

    Alternates between an expansion phase (SEARCH until the live set doubles) and a
    racing phase (REFINE until the plausible set halves). Produces states with a
    characteristic saw-tooth in K_t that no smooth schedule visits.
    """

    name = "bracket"

    def __init__(
        self, base_width: int = 4, rng: np.random.Generator | None = None
    ) -> None:
        super().__init__(rng)
        self.base_width = int(base_width)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        surviving = n_plausible(state)
        # Expand whenever the race has narrowed to fewer than the bracket width.
        return _force_first_arm(state, surviving < self.base_width)


@register("random_search", kind="search_policy")
class UniformRandom(SearchPolicy):
    """Fair coin every round. Deliberately bad as an algorithm, excellent for coverage."""

    name = "random_search"

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        return _force_first_arm(state, self.rng.random(state.M) < 0.5)


# ---- evidence-driven rules -----------------------------------------------------


@register("evidence_threshold", kind="search_policy")
class EvidenceThreshold(SearchPolicy):
    """SEARCH once the incumbent is convincingly ahead of its strongest challenger.

    The intuition the whole study is testing: when the leader/challenger comparison
    is settled, more pulls on the existing frontier buy little, and the budget is
    better spent looking for a genuinely better arm.
    """

    name = "evidence_threshold"
    needs_evidence = True

    def __init__(
        self, threshold: float = 2.9957, rng: np.random.Generator | None = None
    ) -> None:
        super().__init__(rng)
        self.threshold = float(threshold)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        if ctx.log_e_pair is None:
            raise ValueError("evidence_threshold requires ctx.log_e_pair")
        return _force_first_arm(state, ctx.log_e_pair >= self.threshold)


@register("ose", kind="search_policy")
class OSEInspired(SearchPolicy):
    """Search while the best arm found so far falls short of a decaying target.

    An infinite-armed-bandit heuristic: early on, almost any arm is beatable, so keep
    drawing; as the budget burns the bar we demand of a fresh arm rises and searching
    stops paying.
    """

    name = "ose"
    needs_evidence = True

    def __init__(
        self, target0: float = 0.9, decay: float = 0.5, rng: np.random.Generator | None = None
    ) -> None:
        super().__init__(rng)
        self.target0 = float(target0)
        self.decay = float(decay)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        if ctx.best_mean is None:
            raise ValueError("ose requires ctx.best_mean")
        frac = min(max(ctx.t / max(ctx.horizon, 1), 0.0), 1.0)
        target = self.target0 * (1.0 - self.decay * frac)
        return _force_first_arm(state, ctx.best_mean < target)


@register("cp0", kind="search_policy")
class EvidenceGatedSchedule(SearchPolicy):
    """The round-0 continuation policy: a sqrt(t) schedule, gated on being unresolved.

    Deliberately a hybrid of the two families it sits between, so the round-0 oracle
    labels are not generated by a rule that is obviously wrong in one direction. It
    searches while the arm count is below the schedule AND the plausible-winner set
    still has more than one member; once the race is decided it stops recruiting.
    """

    name = "cp0"

    def __init__(
        self,
        alpha: float = 0.5,
        c: float = 1.0,
        min_pulls_per_arm: int = 2,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.schedule = PowerSchedule(alpha=alpha, c=c, rng=self.rng)
        self.min_pulls_per_arm = int(min_pulls_per_arm)

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        below_schedule = self.schedule.should_search(state, ctx)
        unresolved = n_plausible(state) > 1
        return _force_first_arm(state, below_schedule & unresolved & self._affordable(state, ctx))

    def _affordable(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        """Never recruit an arm the remaining budget cannot evaluate.

        A pure `K_t < c*sqrt(t)` schedule is blind to how much budget is left, so a
        replicate that falls behind late tries to *catch up* -- at t=110 of 120 with
        two arms it would recruit ten more and pull each exactly once, leaving the
        final recommendation to be decided by single coin flips on arms nobody
        measured. That is not a strong continuation policy, and since the oracle label
        is defined relative to the continuation policy it would corrupt every label
        taken near the horizon. Requiring room for `min_pulls_per_arm` pulls on each
        arm we would then hold keeps recruitment tied to what can actually be
        evaluated.
        """
        room = self.min_pulls_per_arm * (state.Kt + 1)
        return ctx.remaining >= room


def behavioral_mixture(rng: np.random.Generator) -> list[SearchPolicy]:
    """The ten behavioral policies used to generate the state pool (spec section 7)."""
    return [
        BernoulliSearch(p=0.5, rng=rng),
        BernoulliSearch(p=0.05, rng=rng),
        PowerSchedule(alpha=0.5, c=1.0, rng=rng),
        PowerSchedule(alpha=1.0 / 3.0, c=1.5, rng=rng),
        PowerSchedule(alpha=2.0 / 3.0, c=0.8, rng=rng),
        UniformRandom(rng=rng),
        EpsilonSchedule(alpha=0.5, c=1.0, epsilon=0.15, rng=rng),
        BracketExpansion(base_width=4, rng=rng),
        EvidenceThreshold(threshold=2.9957, rng=rng),
        OSEInspired(target0=0.9, decay=0.5, rng=rng),
    ]
