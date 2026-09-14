"""Terminal recommendation rules: which arm the algorithm actually names at T.

Everything this study measures collapses to one number per rollout -- the *true* mean
of the arm recommended at the horizon -- so the recommendation rule is part of the
estimand, not an implementation detail. Four rules are therefore computed for every
rollout, and the label's sensitivity to the choice becomes a reported number instead
of an assumption:

* ``posterior_mean`` (primary, spec-faithful) -- ``argmax (S_i + a) / (n_i + a + b)``
  with ``Beta(a, b) = Beta(1, 1)`` by default. Under the naive rule one lucky pull
  scores 1.0 and outranks any amount of accumulated evidence, so a forced-SEARCH
  branch could "win" simply by manufacturing a fresh 1-of-1 arm rather than by
  finding a better one -- a pure upward bias in ``A_t``.
* ``posterior_mean_shrunk`` -- the same estimator with an *empirical-Bayes* prior
  fitted, per replicate, to the arms discovered so far. See the confound below.
* ``lcb`` -- argmax of the anytime-valid lower confidence bound; conservative, and the
  standard recommendation in best-arm identification.
* ``empirical`` -- ``argmax S_i / n_i``, recorded *because* it is the naive rule, so
  that "how much does the rule matter" is measured rather than argued.

**Why Beta(1,1) is not enough, and why the fourth rule exists.** A fresh arm pulled
once that succeeds scores ``(1+1)/(1+2) = 0.6667``. That beats an incumbent at 40/60
(0.6613), at 60/100 (0.5980), and even at 120/200 (0.5990) -- an arm measured two
hundred times. Every SEARCH branch of every oracle label ends by adding a fresh arm
and pulling it exactly once, so whenever the continuation policy has little budget
left to invest in that arm, the recommendation can be decided by a single coin flip
and the recorded "recommended true mean" degenerates into a raw reservoir draw. That
injects rule-induced variance and bias into ``A_t`` precisely in the small-remaining-
budget region -- which is where the SEARCH/REFINE boundary is most interesting and
where the sanity cases live. It would mean measuring the recommendation rule rather
than the action.

The empirical-Bayes rule fixes the mechanism rather than the symptom: shrink toward
the population of arms *this reservoir actually produces* instead of toward 1/2. It
is also what a real algorithm would do, and it reads only ``(n_i, S_i)`` of the
discovered arms, so it stays deployable rather than oracle. Both are kept because the
spec names Beta(1,1) as primary; the pilot gate decides on measured evidence.

Vectorization. Each rule is one masked argmax over the (M, Kmax) score view followed
by one flat gather of the true means, so all M replicates are recommended at once.

Two invariants that are silent when broken:

1. **Only active slots compete.** Empty slots carry ``uid == EMPTY_UID`` and stale
   payloads; they are pushed to -inf rather than merely left at their sentinel, so no
   score choice can ever resurrect one.
2. **Ties get the state's per-arm jitter.** ``np.argmax`` resolves ties at the LOWEST
   index, and a freshly discovered arm always occupies the HIGHEST slot. Without the
   jitter a new arm loses every tie -- a systematic downward bias in ``A_t``, worst
   early in a rollout when many arms share the same ``(n, S)``. The jitter is a
   deterministic function of ``(base_seed, replicate, uid)``, so it is identical in
   the SEARCH and REFINE branches and the CRN coupling survives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cold_start.growing.state import EMPTY_UID, GrowingState

#: Recommendation rules, primary first. Every rollout records all four, so the
#: label's sensitivity to the rule is measurable at the pilot gate.
# Rules callable as `recommend(state, rule=...)`. `posterior_mean_oracle` is
# deliberately NOT here: it needs a reservoir prior, so it is reached through
# `recommend_with_oracle_prior` instead and cannot be selected by name alone.
RULES: tuple[str, ...] = ("posterior_mean", "posterior_mean_shrunk", "lcb", "empirical")
PRIMARY_RULE = "posterior_mean"

#: Default prior for `posterior_mean`. Beta(1,1) is what the spec names; it is also
#: the weakest sensible prior, which is exactly why the shrunk rule exists.
DEFAULT_PRIOR_A = 1.0
DEFAULT_PRIOR_B = 1.0

#: Floor on the fitted prior's total mass ``a + b``. Two arms can produce an absurdly
#: small variance by coincidence, and an unfloored fit would then let the population
#: mean overwhelm real evidence. A floor of 2 says "never weaker than Beta(1,1)".
MIN_PRIOR_STRENGTH = 2.0

#: Cap on the fitted prior's total mass. A near-degenerate population (all discovered
#: arms alike) sends the method-of-moments strength to infinity, which would compress
#: every score onto the population mean and hand the ranking to the 1e-6 tie jitter.
#: At 1e4 the residual per-arm signal is ~1e-4 -- two orders above the jitter, so
#: evidence still decides -- while the shrinkage is as strong as it can meaningfully be.
MAX_PRIOR_STRENGTH = 1e4

#: Default resolution of the stored outcome histogram (see `recommendation_histogram`).
DEFAULT_HISTOGRAM_BINS = 16


@dataclass(frozen=True)
class Recommendation:
    """One arm named per replicate, with the true mean we are allowed to score it by.

    ``mu`` is oracle knowledge: legitimate here because these rollouts exist only to
    produce a label. It must never reach a deployable feature column -- see
    `cold_start.growing.schema.design_matrix_columns`.
    """

    rule: str
    cols: np.ndarray  # (M,) int64: chosen column index within each replicate's row
    mu: np.ndarray  # (M,) float64: TRUE mean of the recommended arm

    def __len__(self) -> int:
        return int(self.cols.shape[0])


def _posterior_mean_scores(state: GrowingState, prior_a: float, prior_b: float) -> np.ndarray:
    """Flat ``(S_i + a) / (n_i + a + b)`` under a fixed Beta(a, b) prior."""
    return (state.S + prior_a) / (state.n + prior_a + prior_b)


def fit_empirical_bayes_prior(
    state: GrowingState,
    prior_a: float = DEFAULT_PRIOR_A,
    prior_b: float = DEFAULT_PRIOR_B,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-replicate Beta prior fitted by method of moments to the arms discovered so far.

    Returns ``(a, b)``, each of shape (M,). For a Beta the moments invert in closed
    form: with population mean ``m`` and variance ``v`` over the discovered arms'
    shrunken means, the total mass is ``nu = m(1-m)/v - 1`` and ``a = m*nu``,
    ``b = (1-m)*nu``.

    **Deployability.** The fit reads only ``(n_i, S_i)`` of arms this replicate has
    actually pulled, and the active mask. It never touches ``state.mu``, so the rule
    remains something a real algorithm could run -- it is not an oracle in disguise.

    Fallbacks, all of which must be total rather than NaN-producing (a NaN score would
    lose every argmax silently, which is the worst possible failure here):

    * fewer than two active arms, or a degenerate population (zero variance, or a mean
      pinned at 0 or 1) -- there is no population to fit, so use the base prior;
    * an over-dispersed population, where ``v >= m(1-m)`` makes ``nu`` non-positive and
      no Beta matches the moments -- keep the fitted mean, floor the strength.
    """
    base_strength = prior_a + prior_b
    if base_strength <= 0.0:
        raise ValueError(f"prior mass must be positive, got a={prior_a}, b={prior_b}")
    base_mean = prior_a / base_strength

    active = state.active_mask()  # (M, Kmax) -- reads uid only, never mu
    weight = active.astype(np.float64)
    n_arms = weight.sum(axis=1)

    per_arm = state.view(_posterior_mean_scores(state, prior_a, prior_b))
    masked = np.where(active, per_arm, 0.0)
    totals = masked.sum(axis=1)

    mean = np.full(state.M, base_mean, dtype=np.float64)
    np.divide(totals, n_arms, out=mean, where=n_arms > 0.0)

    dev = np.where(active, (per_arm - mean[:, None]) ** 2, 0.0)
    var = np.zeros(state.M, dtype=np.float64)
    np.divide(dev.sum(axis=1), n_arms - 1.0, out=var, where=n_arms >= 2.0)

    fittable = (n_arms >= 2.0) & (var > 0.0) & (mean > 0.0) & (mean < 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        strength_raw = mean * (1.0 - mean) / var - 1.0
    strength_raw = np.where(np.isfinite(strength_raw), strength_raw, MIN_PRIOR_STRENGTH)
    # A prior fitted from K arms cannot be worth more than about K observations.
    # Without this the estimator is badly over-confident exactly where it is most
    # dangerous: two arms that happen to agree closely give a near-zero moment
    # variance and hence an enormous strength -- measured Beta(424, 92) from just two
    # arms at 0.833 and 0.810. Because the discovered arms are survivorship-biased
    # (we kept pulling the good ones), that over-confident prior then declares a
    # never-pulled arm nearly as good as a 40-pull incumbent. Tying the strength to
    # the number of arms that informed it keeps the shrinkage honest when K is small,
    # and relaxes as real evidence about the reservoir accumulates.
    n_arms = np.maximum(active.reshape(state.M, state.Kmax).sum(axis=1), 1).astype(np.float64)
    strength = np.clip(strength_raw, MIN_PRIOR_STRENGTH, np.minimum(MAX_PRIOR_STRENGTH, n_arms))

    fitted_mean = np.where(fittable, mean, base_mean)
    fitted_strength = np.where(fittable, strength, base_strength)
    return fitted_mean * fitted_strength, (1.0 - fitted_mean) * fitted_strength


def _empirical_bayes_scores(state: GrowingState, prior_a: float, prior_b: float) -> np.ndarray:
    """Flat ``(S_i + a_m) / (n_i + a_m + b_m)`` with a per-replicate fitted prior."""
    a, b = fit_empirical_bayes_prior(state, prior_a, prior_b)
    n2 = state.view(state.n.astype(np.float64))
    s2 = state.view(state.S.astype(np.float64))
    scores = (s2 + a[:, None]) / (n2 + (a + b)[:, None])
    # Freshly allocated and C-contiguous, so this reshape is a view, not the silent
    # copy `.ravel()` would make on a strided array (see state.py's layout rules).
    return scores.reshape(-1)


def _rule_scores(
    state: GrowingState, rule: str, prior_a: float, prior_b: float
) -> np.ndarray:
    """Flat (M*Kmax,) float64 score for `rule`, before masking and tie-breaking.

    float64 rather than the stored float32: the jitter is 1e-6 while scores sit near
    1.0, which is only ~16 float32 ulps. Comparing in f64 keeps the tie-break exact
    instead of leaving it to rounding.

    `prior_a` / `prior_b` are ignored by `lcb` and `empirical`, which are prior-free
    by construction; they are accepted uniformly so callers need no special-casing.
    """
    if rule == "posterior_mean":
        return _posterior_mean_scores(state, prior_a, prior_b)
    if rule == "posterior_mean_shrunk":
        return _empirical_bayes_scores(state, prior_a, prior_b)
    if rule == "lcb":
        return state.lcb.astype(np.float64)
    if rule == "empirical":
        # S/n, with the 0/0 of a never-pulled arm defined as 0.0: an arm with no
        # evidence should not be recommendable by a rule that reads only evidence.
        n = state.n.astype(np.float64)
        s = state.S.astype(np.float64)
        out = np.zeros_like(n)
        np.divide(s, n, out=out, where=n > 0.0)
        return out
    if rule == "posterior_mean_oracle":
        raise ValueError(
            "posterior_mean_oracle needs the reservoir prior; call recommend(..., "
            "oracle_prior=(a, b)) rather than routing it through _rule_scores"
        )
    raise ValueError(f"unknown recommendation rule {rule!r}; expected one of {RULES}")


def recommend(
    state: GrowingState,
    rule: str = PRIMARY_RULE,
    *,
    prior_a: float = DEFAULT_PRIOR_A,
    prior_b: float = DEFAULT_PRIOR_B,
) -> Recommendation:
    """Name one arm per replicate under `rule`, returning its column and true mean.

    `prior_a` / `prior_b` set the Beta prior for `posterior_mean` and the *base* prior
    the empirical-Bayes fit falls back to. The defaults reproduce the spec's Beta(1,1)
    exactly, so changing them is always a deliberate ablation.
    """
    active = state.uid != EMPTY_UID
    have_any = active.reshape(state.M, state.Kmax).any(axis=1)
    if not bool(have_any.all()):
        empty = int((~have_any).sum())
        raise ValueError(
            f"{empty} of {state.M} replicate(s) have no active arm; there is nothing "
            "to recommend. Every replicate must SEARCH at least once before T."
        )

    score = _rule_scores(state, rule, prior_a, prior_b) + state.tie.astype(np.float64)
    score = np.where(active, score, -np.inf)

    cols = np.argmax(state.view(score), axis=1).astype(np.int64)
    lin = state.row_off + cols
    return Recommendation(rule=rule, cols=cols, mu=state.mu[lin].astype(np.float64))


def recommend_all(
    state: GrowingState,
    *,
    prior_a: float = DEFAULT_PRIOR_A,
    prior_b: float = DEFAULT_PRIOR_B,
) -> dict[str, Recommendation]:
    """Every rule at once. Cheap -- the shared cost is the state, not the argmax.

    All four go into every label. Recording only the primary rule would leave the
    single-lucky-pull confound invisible instead of quantified.
    """
    out: dict[str, Recommendation] = {}
    for rule in RULES:
        out[rule] = recommend(state, rule, prior_a=prior_a, prior_b=prior_b)
    return out


def simple_regret(mu_star, mu_recommended) -> np.ndarray:
    """``mu_star - mu(recommended)``: the quantity the study ultimately minimizes.

    Not clipped at zero. A negative value would mean the recommended arm beat the
    stated optimum, which is a bug in `mu_star` (or a reservoir whose supremum is not
    attained being summarized by an attainable value) and should surface, not vanish.
    """
    star = np.asarray(mu_star, dtype=np.float64)
    rec = np.asarray(mu_recommended, dtype=np.float64)
    return star - rec


def advantage(mu_rec_search, mu_rec_refine) -> float:
    """Oracle advantage ``A_t = E[mu_rec | SEARCH] - E[mu_rec | REFINE]``; ``A_t > 0``
    means SEARCH is the better next action.

    Deliberately a difference of recommended TRUE MEANS, not a difference of simple
    regrets, even though the two are algebraically identical:
    ``(mu_star - mu_ref) - (mu_star - mu_sea) = mu_sea - mu_ref``. Two reasons.

    1. **Numerics.** Forming each regret first adds and then cancels a large common
       constant ``mu_star`` (often ~0.9 against an advantage of ~1e-3), spending
       significant digits on a term guaranteed to disappear.
    2. **Definition independence.** For reservoirs whose supremum is not attained --
       Family B at ``mu_star = 1`` is one -- ``mu_star`` is a limit, not any arm's
       mean, so *any* choice of it is a convention. Differencing true means makes the
       label completely independent of that convention, which matters because the
       label is compared across reservoir families that would not share one.
    """
    sea = np.asarray(mu_rec_search, dtype=np.float64)
    ref = np.asarray(mu_rec_refine, dtype=np.float64)
    return float(sea.mean() - ref.mean())


def advantage_se(mu_rec_search, mu_rec_refine) -> float:
    """Standard error of `advantage`, computed PAIRED across replicates.

    The two branches share common random numbers by construction, so their outcomes
    are positively correlated and the unpaired SE
    ``sqrt(var_s/M + var_r/M)`` badly overstates the error -- which would make the
    adaptive-precision labeller draw batches it does not need. Requires equal-length,
    replicate-aligned inputs; anything else is a wiring bug, not a valid call.
    """
    sea = np.asarray(mu_rec_search, dtype=np.float64)
    ref = np.asarray(mu_rec_refine, dtype=np.float64)
    if sea.shape != ref.shape:
        raise ValueError(
            f"paired SE needs replicate-aligned branches, got shapes {sea.shape} and "
            f"{ref.shape}; the CRN pairing is broken if these ever differ"
        )
    m = sea.size
    if m < 2:
        return float("nan")
    diff = sea - ref
    return float(diff.std(ddof=1) / np.sqrt(m))


def histogram_edges(
    bins: int = DEFAULT_HISTOGRAM_BINS, lo: float = 0.0, hi: float = 1.0
) -> np.ndarray:
    """Bin edges matching `recommendation_histogram`, so quantiles can be recomputed."""
    return np.linspace(lo, hi, bins + 1, dtype=np.float64)


def recommendation_histogram(
    mu_recommended,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Counts of ``mu_recommended`` over `bins` equal-width bins on ``[lo, hi]``.

    Per-replicate outcomes are far too large to persist (150k states x ~thousands of
    replicates), but a 16-bin histogram is 32 bytes and lets any later analysis
    recompute medians, quantiles, or tail masses without re-simulating anything.

    Values are clipped into range rather than dropped, so the counts always sum to the
    replicate count -- a histogram that silently loses mass would understate the tail
    it exists to describe.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if not hi > lo:
        raise ValueError(f"histogram range must satisfy hi > lo, got lo={lo}, hi={hi}")
    values = np.clip(np.asarray(mu_recommended, dtype=np.float64).ravel(), lo, hi)
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    return counts.astype(np.int64)


def oracle_prior_from_reservoir(reservoir, n_samples: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Moment-match a Beta prior to the TRUE reservoir.

    This is oracle knowledge and belongs only in the label, never in a feature. The
    justification is that the study's objective is *expected* simple regret, and the
    recommendation that minimises it is the posterior mean under the correct prior --
    not the lower confidence bound, which is optimal for a risk-averse objective, and
    not an empirical-Bayes fit, which is estimated from a survivorship-biased sample
    of arms we chose to keep pulling. Defining the oracle label with the oracle's own
    prior is what makes it an upper bound on what any deployable rule could achieve.
    """
    import numpy as _np

    draws = _np.asarray(reservoir.sample(_np.random.default_rng(seed), n_samples), dtype=float)
    m = float(_np.clip(draws.mean(), 1e-6, 1 - 1e-6))
    v = float(draws.var(ddof=1))
    if v <= 1e-12 or v >= m * (1 - m):
        return 1.0, 1.0
    # No MIN_PRIOR_STRENGTH floor here. That floor exists for the empirical-Bayes path,
    # where the prior is estimated from a handful of survivorship-biased arms and must
    # not outweigh the evidence behind it. This function knows the reservoir exactly, so
    # flooring would silently substitute a different prior than the one promised --
    # measured on the beta=0.5 tail reservoir, whose true strength is 1.50, below the
    # floor of 2.0. Only a numerical guard remains.
    nu = float(_np.clip(m * (1 - m) / v - 1.0, 1e-3, MAX_PRIOR_STRENGTH))
    return m * nu, (1.0 - m) * nu


def recommend_with_oracle_prior(
    state: GrowingState, prior: tuple[float, float]
) -> Recommendation:
    """`posterior_mean` under a prior matched to the true reservoir."""
    a, b = float(prior[0]), float(prior[1])
    score = (state.S + a) / (state.n + a + b) + state.tie.astype(np.float64)
    score = np.where(state.uid != EMPTY_UID, score, -np.inf)
    cols = np.argmax(score.reshape(state.M, state.Kmax), axis=1).astype(np.int64)
    lin = state.row_off + cols
    return Recommendation(rule="posterior_mean_oracle", cols=cols, mu=state.mu[lin].astype(np.float64))
