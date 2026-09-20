"""Terminal recommendation rules scored by the deployment harness.

The recommendation never affects the trajectory: it is a read-only function of the
final state. So every rule is scored from the *same* final state of every run, and the
study's sensitivity to the rule becomes a reported number rather than a modelling
choice (failure-mode register #4). Four rules are deployable -- they read only
``(n_i, S_i)`` and the CS bounds -- and one, ``oracle_prior``, is not: it shrinks
toward a Beta prior moment-matched to the *true* reservoir. It is kept as the ceiling
the deployable rules are measured against, and it is the harness (never a policy)
that supplies the prior.
"""

from __future__ import annotations

from cold_start.growing import recommend
from cold_start.growing.recommend import Recommendation
from cold_start.growing.state import GrowingState

#: Every rule the harness scores, primary first. The first four are
#: `recommend.RULES`; ``oracle_prior`` is `recommend_with_oracle_prior`.
RECOMMENDER_NAMES: tuple[str, ...] = (
    "posterior_mean_shrunk",
    "lcb",
    "posterior_mean",
    "empirical",
    "oracle_prior",
)

#: The deployable primary: empirical-Bayes shrinkage toward the arms this episode
#: actually discovered, with prior strength capped at K (`recommend.py`). It is the
#: repo's designed deployable counterpart of the label harness's oracle prior.
PRIMARY_RECOMMENDER = "posterior_mean_shrunk"

#: Rules that are computable by a deployed algorithm (no reservoir knowledge).
DEPLOYABLE_RECOMMENDERS: tuple[str, ...] = tuple(
    name for name in RECOMMENDER_NAMES if name != "oracle_prior"
)


def recommend_all_rules(
    state: GrowingState, oracle_prior: tuple[float, float] | None
) -> dict[str, Recommendation]:
    """Score every recommender from one final state, keyed by `RECOMMENDER_NAMES`.

    `oracle_prior` is ``(a, b)`` from `recommend.oracle_prior_from_reservoir`; pass
    ``None`` to omit the oracle rule (e.g. when the reservoir is not available, or to
    make the hygiene boundary explicit in a caller that must not hold one). Every
    `Recommendation.mu` is the TRUE mean of the named arm -- harness-side use only.
    """
    observable = recommend.recommend_all(state)
    out: dict[str, Recommendation] = {}
    for name in RECOMMENDER_NAMES:
        if name == "oracle_prior":
            if oracle_prior is not None:
                out[name] = recommend.recommend_with_oracle_prior(state, oracle_prior)
            continue
        out[name] = observable[name]
    return out
