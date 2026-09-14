"""Tests for the terminal recommendation rules and the advantage accounting.

The recommendation rule is part of the estimand, not a detail: it decides which arm's
true mean becomes the rollout's outcome, so a biased rule biases every label. These
tests pin the three rules' behavior on hand-built states, and pin the two places the
rule can silently corrupt `A_t` -- unpulled-arm scores and argmax tie-breaking.
"""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.recommend import (
    MIN_PRIOR_STRENGTH,
    RULES,
    Recommendation,
    advantage,
    advantage_se,
    fit_empirical_bayes_prior,
    histogram_edges,
    recommend,
    recommend_all,
    recommendation_histogram,
    simple_regret,
)
from cold_start.growing.state import GrowingState


def _make_state(arms: list[dict], n_replicates: int = 1, base_seed: int = 7) -> GrowingState:
    """Build a state whose every replicate holds the same arms.

    Each entry is ``{"mu": ..., "n": ..., "S": ...}`` with an optional ``"lcb"``.
    Arms are added through `add_arms` (never by poking `uid` directly) so the CRN
    keys and the tie jitter are populated exactly as they are in a real rollout.
    """
    capacity = max(len(arms), 1)
    state = GrowingState(
        n_replicates=n_replicates, capacity=capacity, horizon=64, base_seed=base_seed
    )
    for uid, spec in enumerate(arms):
        mu = np.full(n_replicates, spec["mu"], dtype=np.float32)
        uids = np.full(n_replicates, uid, dtype=np.int32)
        cols = state.add_arms(mu, uids)
        lin = state.row_off + cols
        state.n[lin] = int(spec["n"])
        state.S[lin] = int(spec["S"])
        if "lcb" in spec:
            state.lcb[lin] = np.float32(spec["lcb"])
    return state


# ---- the three rules ----------------------------------------------------------


def test_posterior_mean_picks_the_best_evidenced_arm():
    state = _make_state(
        [
            {"mu": 0.30, "n": 20, "S": 6},
            {"mu": 0.70, "n": 20, "S": 14},
            {"mu": 0.40, "n": 20, "S": 8},
        ]
    )
    rec = recommend(state, "posterior_mean")
    assert rec.cols.tolist() == [1], f"expected column 1, got {rec.cols.tolist()}"
    assert rec.mu[0] == pytest.approx(0.70), f"recommended true mu={rec.mu[0]}"


def test_lcb_rule_picks_the_highest_lower_bound():
    """LCB deliberately disagrees with the point estimate when evidence is thin."""
    state = _make_state(
        [
            {"mu": 0.50, "n": 40, "S": 24, "lcb": 0.45},
            {"mu": 0.90, "n": 2, "S": 2, "lcb": 0.10},
        ]
    )
    rec = recommend(state, "lcb")
    assert rec.cols.tolist() == [0], f"expected the well-measured arm, got {rec.cols.tolist()}"
    assert rec.mu[0] == pytest.approx(0.50), f"recommended true mu={rec.mu[0]}"


def test_empirical_rule_picks_the_highest_raw_rate():
    state = _make_state(
        [
            {"mu": 0.60, "n": 10, "S": 6},
            {"mu": 0.20, "n": 10, "S": 2},
            {"mu": 0.80, "n": 10, "S": 8},
        ]
    )
    rec = recommend(state, "empirical")
    assert rec.cols.tolist() == [2], f"expected column 2, got {rec.cols.tolist()}"


def test_empirical_rule_scores_an_unpulled_arm_as_zero():
    """0/0 must not be a free win: an arm with no evidence carries no evidence."""
    state = _make_state([{"mu": 0.10, "n": 4, "S": 1}, {"mu": 0.99, "n": 0, "S": 0}])
    rec = recommend(state, "empirical")
    assert rec.cols.tolist() == [0], f"unpulled arm was recommended: {rec.cols.tolist()}"


# ---- why the primary rule shrinks ---------------------------------------------


def test_shrinkage_prefers_the_measured_arm_while_empirical_prefers_one_of_one():
    """The reason `posterior_mean` is primary.

    A 1-of-1 arm scores a perfect 1.0 under S/n, so the naive rule hands the win to
    whichever branch most recently manufactured a fresh arm -- which is precisely the
    SEARCH branch, and precisely a bias in `A_t`. Beta(1,1) shrinkage pulls that arm
    back to 2/3 and the well-measured arm (46/62 = 0.742) wins.
    """
    state = _make_state([{"mu": 0.75, "n": 60, "S": 45}, {"mu": 0.30, "n": 1, "S": 1}])

    shrunk = recommend(state, "posterior_mean")
    naive = recommend(state, "empirical")

    assert shrunk.cols.tolist() == [0], f"shrinkage failed to prefer 45/60: {shrunk.cols.tolist()}"
    assert naive.cols.tolist() == [1], f"naive rule failed to prefer 1/1: {naive.cols.tolist()}"
    assert shrunk.mu[0] == pytest.approx(0.75), f"got {shrunk.mu[0]}"
    assert naive.mu[0] == pytest.approx(0.30), f"got {naive.mu[0]}"


def test_forty_of_sixty_versus_one_of_one_is_the_shrinkage_boundary():
    """Where shrinkage stops rescuing the measured arm -- worth pinning, not assuming.

    (40+1)/(60+2) = 0.66129 sits just BELOW (1+1)/(1+2) = 0.66667, so at exactly 40/60
    the Beta(1,1) prior is not quite enough and the 1-of-1 arm still wins. The rule
    reverses the naive ranking only once the measured rate clears the shrunk value of
    a single success, i.e. above ~2/3. This is a property of the estimator, not a bug,
    and it is the reason the study records all three rules rather than trusting one.
    """
    state = _make_state([{"mu": 0.667, "n": 60, "S": 40}, {"mu": 0.30, "n": 1, "S": 1}])
    scores = (state.S + 1.0) / (state.n + 2.0)

    assert scores[0] == pytest.approx(41.0 / 62.0), f"got {scores[0]}"
    assert scores[1] == pytest.approx(2.0 / 3.0), f"got {scores[1]}"
    assert scores[0] < scores[1], f"40/60 unexpectedly outscored 1/1: {scores[:2]}"

    rec = recommend(state, "posterior_mean")
    assert rec.cols.tolist() == [1], f"expected the 1-of-1 arm at the boundary: {rec.cols}"


# ---- eligibility and tie-breaking ---------------------------------------------


def test_inactive_slots_are_never_recommended():
    """Empty slots read (n, S) = (0, 0), which scores 0.5 -- above every active arm here."""
    state = GrowingState(n_replicates=3, capacity=6, horizon=64, base_seed=11)
    for uid, mu in enumerate((0.10, 0.15)):
        cols = state.add_arms(
            np.full(3, mu, dtype=np.float32), np.full(3, uid, dtype=np.int32)
        )
        lin = state.row_off + cols
        state.n[lin] = 8
        state.S[lin] = uid  # 0/8 and 1/8: both score well under 0.5 after shrinkage

    for rule in RULES:
        rec = recommend(state, rule)
        assert set(rec.cols.tolist()) <= {0, 1}, f"{rule} chose an empty slot: {rec.cols.tolist()}"
        # float32 storage: 0.15 round-trips to 0.15000001, hence the f32-sized slack.
        assert np.all(rec.mu <= 0.15 + 1e-6), f"{rule} recommended a stale mu: {rec.mu}"


def test_tie_jitter_prevents_systematic_low_slot_bias():
    """Without the jitter every tie goes to column 0, and a new arm never wins one.

    `np.argmax` resolves ties at the lowest index while a freshly searched arm always
    lands in the highest slot, so ties would be resolved *against* SEARCH in every
    replicate -- a coherent, non-averaging bias in `A_t`.
    """
    m = 256
    identical = [{"mu": 0.5, "n": 3, "S": 2} for _ in range(5)]
    state = _make_state(identical, n_replicates=m, base_seed=20260910)

    rec = recommend(state, "posterior_mean")
    chosen, counts = np.unique(rec.cols, return_counts=True)

    assert chosen.size == 5, f"only columns {chosen.tolist()} ever won a tie"
    assert counts.min() > 0.5 * m / 5, f"tie winners are badly skewed: {counts.tolist()}"
    assert counts[0] < 0.6 * m, f"column 0 won {counts[0]}/{m} ties -- jitter is not applied"


def test_recommendation_is_deterministic():
    """The jitter is a function of (seed, replicate, uid), so repeats must agree."""
    arms = [{"mu": 0.5, "n": 2, "S": 1} for _ in range(4)]
    first = recommend(_make_state(arms, n_replicates=32, base_seed=99), "posterior_mean")
    second = recommend(_make_state(arms, n_replicates=32, base_seed=99), "posterior_mean")
    assert first.cols.tolist() == second.cols.tolist(), "tie-breaking is not reproducible"


def test_recommend_all_returns_every_rule():
    state = _make_state([{"mu": 0.4, "n": 5, "S": 2, "lcb": 0.1}], n_replicates=2)
    out = recommend_all(state)
    assert sorted(out) == sorted(RULES), f"got rules {sorted(out)}"
    for rule, rec in out.items():
        assert isinstance(rec, Recommendation), f"{rule} returned {type(rec)}"
        assert len(rec) == 2, f"{rule} returned {len(rec)} recommendations for 2 replicates"


def test_unknown_rule_raises():
    state = _make_state([{"mu": 0.4, "n": 5, "S": 2}])
    with pytest.raises(ValueError, match="unknown recommendation rule"):
        recommend(state, "argmax_ucb")


def test_replicate_without_an_active_arm_raises():
    state = GrowingState(n_replicates=2, capacity=3, horizon=8, base_seed=3)
    with pytest.raises(ValueError, match="no active arm"):
        recommend(state)


# ---- regret and advantage accounting ------------------------------------------


def test_simple_regret_is_the_gap_to_mu_star():
    reg = simple_regret(0.9, np.array([0.9, 0.5, 0.2]))
    assert reg.tolist() == pytest.approx([0.0, 0.4, 0.7]), f"got {reg.tolist()}"


def test_advantage_is_positive_when_search_recommends_better_arms():
    """Sign convention: A_t > 0 means SEARCH is the better next action."""
    search = np.array([0.8, 0.7, 0.9])
    refine = np.array([0.6, 0.6, 0.6])
    a = advantage(search, refine)
    assert a > 0.0, f"expected SEARCH to win, got A={a}"
    assert a == pytest.approx(0.2), f"got A={a}"
    assert advantage(refine, search) == pytest.approx(-0.2), "sign convention is not antisymmetric"


def test_advantage_equals_the_difference_of_mean_regrets():
    """Algebraically identical, computed the other way to avoid cancelling mu_star."""
    rng = np.random.default_rng(0)
    search = rng.uniform(0.2, 0.9, size=512)
    refine = rng.uniform(0.2, 0.9, size=512)
    mu_star = 0.97

    direct = advantage(search, refine)
    via_regret = float(
        simple_regret(mu_star, refine).mean() - simple_regret(mu_star, search).mean()
    )
    assert direct == pytest.approx(via_regret, abs=1e-12), f"{direct} vs {via_regret}"


def test_paired_se_beats_unpaired_when_branches_are_coupled():
    """CRN makes the branches correlated; the paired SE is the one the labeller needs."""
    rng = np.random.default_rng(1)
    common = rng.uniform(0.2, 0.8, size=1024)
    search = common + rng.normal(0.0, 0.01, size=1024)
    refine = common + rng.normal(0.0, 0.01, size=1024)

    paired = advantage_se(search, refine)
    unpaired = float(np.sqrt(search.var(ddof=1) / 1024 + refine.var(ddof=1) / 1024))
    assert paired < unpaired / 5.0, f"paired={paired:.6f} unpaired={unpaired:.6f}"


def test_paired_se_rejects_misaligned_branches():
    with pytest.raises(ValueError, match="replicate-aligned"):
        advantage_se(np.zeros(8), np.zeros(9))


# ---- outcome histogram ---------------------------------------------------------


def test_histogram_bins_sum_to_the_replicate_count():
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.0, 1.0, size=997)
    counts = recommendation_histogram(mu)
    assert counts.shape == (16,), f"expected 16 bins, got {counts.shape}"
    assert int(counts.sum()) == 997, f"histogram lost mass: sum={int(counts.sum())}"


def test_histogram_clips_rather_than_dropping_out_of_range_values():
    counts = recommendation_histogram(np.array([-0.5, 0.5, 1.5]), bins=4)
    assert int(counts.sum()) == 3, f"clipping failed: {counts.tolist()}"
    assert counts[0] == 1 and counts[-1] == 1, f"clipped mass landed wrong: {counts.tolist()}"


def test_histogram_edges_match_the_bin_count():
    edges = histogram_edges(16)
    assert edges.shape == (17,), f"got {edges.shape}"
    assert edges[0] == 0.0 and edges[-1] == 1.0, f"got range [{edges[0]}, {edges[-1]}]"


def test_histogram_rejects_a_degenerate_range():
    with pytest.raises(ValueError, match="hi > lo"):
        recommendation_histogram(np.zeros(4), bins=8, lo=1.0, hi=1.0)


# ---- the single-lucky-pull confound and the empirical-Bayes rule ----------------


def test_beta_one_one_lets_one_lucky_pull_outrank_two_hundred_pulls():
    """The confound, pinned as arithmetic.

    A fresh arm pulled once that succeeds scores (1+1)/(1+2) = 0.6667 under Beta(1,1),
    which beats 40/60, 60/100 and even 120/200. Every SEARCH branch ends by adding a
    fresh arm and pulling it once, so with little budget left to invest in that arm the
    recommendation -- and therefore the recorded outcome of the whole rollout -- can be
    decided by one coin flip on a raw reservoir draw.
    """
    assert 2.0 / 3.0 > 41.0 / 62.0, "1-of-1 should outscore 40/60 under Beta(1,1)"
    assert 2.0 / 3.0 > 61.0 / 102.0, "1-of-1 should outscore 60/100 under Beta(1,1)"
    assert 2.0 / 3.0 > 121.0 / 202.0, "1-of-1 should outscore 120/200 under Beta(1,1)"

    state = _make_state([{"mu": 0.60, "n": 200, "S": 120}, {"mu": 0.95, "n": 1, "S": 1}])
    rec = recommend(state, "posterior_mean")
    assert rec.cols.tolist() == [1], f"expected the fresh arm to win: {rec.cols.tolist()}"


def test_empirical_bayes_prior_defends_the_two_hundred_pull_arm():
    """The fix: shrink toward the arms this reservoir actually produces, not toward 1/2.

    With a discovered population centred near 0.38, the fitted prior pulls the 1-of-1
    arm from 0.6667 down to ~0.52 while the 200-pull arm barely moves (0.5990 ->
    0.5963), so evidence wins. Beta(1,1) on the same state recommends the fresh arm.
    """
    arms = [
        {"mu": 0.60, "n": 200, "S": 120},
        {"mu": 0.20, "n": 50, "S": 10},
        {"mu": 0.20, "n": 60, "S": 12},
        {"mu": 0.20, "n": 40, "S": 8},
        {"mu": 0.95, "n": 1, "S": 1},
    ]
    state = _make_state(arms)

    a, b = fit_empirical_bayes_prior(state)
    strength = float(a[0] + b[0])
    mean = float(a[0]) / strength
    assert 0.30 < mean < 0.45, f"fitted population mean {mean:.4f} is not below 2/3"
    assert strength >= MIN_PRIOR_STRENGTH, f"fitted strength {strength:.4f} below the floor"

    shrunk = recommend(state, "posterior_mean_shrunk")
    flat = recommend(state, "posterior_mean")
    assert shrunk.cols.tolist() == [0], f"EB rule failed to defend 120/200: {shrunk.cols}"
    assert flat.cols.tolist() == [4], f"Beta(1,1) should still prefer 1/1: {flat.cols}"
    assert shrunk.mu[0] == pytest.approx(0.60, abs=1e-6), f"got {shrunk.mu[0]}"


def test_empirical_bayes_fit_is_total_when_there_is_no_population_to_fit():
    """One arm, or a zero-variance population, must fall back -- never NaN.

    A NaN score loses every argmax silently, which would corrupt labels with no error
    and no signal. Both degenerate cases fall back to the base prior instead.
    """
    for label, arms in (
        ("single arm", [{"mu": 0.5, "n": 3, "S": 2}]),
        ("identical arms", [{"mu": 0.5, "n": 3, "S": 2}, {"mu": 0.6, "n": 3, "S": 2}]),
        ("never pulled", [{"mu": 0.5, "n": 0, "S": 0}, {"mu": 0.6, "n": 0, "S": 0}]),
    ):
        state = _make_state(arms, n_replicates=4)
        a, b = fit_empirical_bayes_prior(state)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"{label}: NaN prior {a}, {b}"
        assert np.allclose(a, 1.0) and np.allclose(b, 1.0), f"{label}: expected the base prior"

        rec = recommend(state, "posterior_mean_shrunk")
        assert np.all(np.isfinite(rec.mu)), f"{label}: non-finite recommendation {rec.mu}"
        assert set(rec.cols.tolist()) <= set(range(len(arms))), f"{label}: {rec.cols.tolist()}"


def test_empirical_bayes_strength_is_floored_on_an_over_dispersed_population():
    """No Beta matches the moments when v >= m(1-m); keep the mean, floor the strength."""
    state = _make_state([{"mu": 0.01, "n": 200, "S": 0}, {"mu": 0.99, "n": 200, "S": 200}])
    a, b = fit_empirical_bayes_prior(state)
    strength = float(a[0] + b[0])
    assert strength == pytest.approx(MIN_PRIOR_STRENGTH), f"strength={strength}"


def test_empirical_bayes_strength_is_capped_by_the_number_of_arms():
    """Two arms that closely agree send the method-of-moments strength to infinity.

    Uncapped, every score would collapse onto the population mean and the 1e-6 tie
    jitter -- not evidence -- would pick the recommendation. The cap is the arm count
    rather than a large constant because a prior fitted from K arms cannot be worth
    more than about K observations: measured Beta(424, 92) from two arms at 0.833 and
    0.810, which then rates a never-pulled arm as highly as a 40-pull incumbent.
    """
    state = _make_state([{"mu": 0.6, "n": 100, "S": 60}, {"mu": 0.6, "n": 200, "S": 120}])
    a, b = fit_empirical_bayes_prior(state)
    assert float(a[0] + b[0]) == pytest.approx(2.0), f"a={a[0]}, b={b[0]}"


def test_empirical_bayes_strength_grows_as_arms_accumulate():
    """The cap must relax as real evidence about the reservoir arrives."""
    few = _make_state([{"mu": 0.6, "n": 100, "S": 60}, {"mu": 0.6, "n": 200, "S": 120}])
    many = _make_state(
        [{"mu": 0.6, "n": 100, "S": 60 + (i % 3)} for i in range(12)]
    )
    a_few, b_few = fit_empirical_bayes_prior(few)
    a_many, b_many = fit_empirical_bayes_prior(many)
    assert float(a_many[0] + b_many[0]) > float(a_few[0] + b_few[0])


def test_survivorship_bias_cannot_make_an_unpulled_arm_beat_a_measured_one():
    """The failure this cap exists to prevent.

    Two excellent discovered arms make the fitted prior believe fresh arms are
    excellent too -- but the discovered set is survivorship-biased, since those are
    the arms we chose to keep pulling. Uncapped, a 1-of-1 arm scored 0.822 against a
    34-of-40 incumbent's 0.823, so a single coin flip could unseat forty pulls.
    """
    state = _make_state(
        [
            {"mu": 0.85, "n": 40, "S": 34},
            {"mu": 0.83, "n": 40, "S": 33},
            {"mu": 0.10, "n": 1, "S": 1},
        ]
    )
    rec = recommend(state, rule="posterior_mean_shrunk")
    assert int(rec.cols[0]) != 2, "an unpulled arm unseated a 40-pull incumbent"


def test_empirical_bayes_fit_never_reads_true_mu():
    """The rule must stay DEPLOYABLE: the fit sees (n, S) and the active mask, nothing else.

    If true means leaked into the prior, the recommendation would be an oracle and
    every label built on it would be unusable for fitting a real policy.
    """
    counts = [(200, 120), (50, 10), (60, 12), (1, 1)]
    honest_mu = (0.6, 0.2, 0.2, 0.9)
    inverted_mu = (0.1, 0.9, 0.9, 0.1)

    honest = []
    inverted = []
    for i, (n, s) in enumerate(counts):
        honest.append({"mu": honest_mu[i], "n": n, "S": s})
        inverted.append({"mu": inverted_mu[i], "n": n, "S": s})

    a1, b1 = fit_empirical_bayes_prior(_make_state(honest, n_replicates=8))
    a2, b2 = fit_empirical_bayes_prior(_make_state(inverted, n_replicates=8))
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2), "the fit moved with true mu"

    cols1 = recommend(_make_state(honest, n_replicates=8), "posterior_mean_shrunk").cols
    cols2 = recommend(_make_state(inverted, n_replicates=8), "posterior_mean_shrunk").cols
    assert cols1.tolist() == cols2.tolist(), "the recommendation moved with true mu"


def test_prior_parameterization_flips_the_forty_of_sixty_verdict():
    """Beta(2,2) shrinks the 1-of-1 arm to 0.60 and 40/60 to 0.656, reversing Beta(1,1).

    The prior is a knob, defaulting to the spec's Beta(1,1), so any change is a
    deliberate ablation rather than a silent redefinition of the primary label.
    """
    state = _make_state([{"mu": 0.667, "n": 60, "S": 40}, {"mu": 0.30, "n": 1, "S": 1}])

    default = recommend(state, "posterior_mean")
    stronger = recommend(state, "posterior_mean", prior_a=2.0, prior_b=2.0)

    assert default.cols.tolist() == [1], f"Beta(1,1) should prefer 1/1: {default.cols.tolist()}"
    assert stronger.cols.tolist() == [0], f"Beta(2,2) should prefer 40/60: {stronger.cols.tolist()}"
    assert 42.0 / 64.0 > 3.0 / 5.0, "Beta(2,2) arithmetic: 40/60 must outscore 1/1"


def test_rejects_a_prior_with_no_mass():
    with pytest.raises(ValueError, match="prior mass must be positive"):
        recommend(_make_state([{"mu": 0.5, "n": 2, "S": 1}]), "posterior_mean_shrunk",
                  prior_a=0.0, prior_b=0.0)


def test_all_four_rules_are_recorded_per_label():
    """The sensitivity of A_t to the recommendation rule has to be measurable, not assumed."""
    assert "posterior_mean_shrunk" in RULES, f"RULES={RULES}"
    assert len(RULES) == 4, f"RULES={RULES}"

    state = _make_state(
        [{"mu": 0.6, "n": 30, "S": 18, "lcb": 0.4}, {"mu": 0.9, "n": 1, "S": 1, "lcb": 0.0}],
        n_replicates=4,
    )
    out = recommend_all(state)
    assert sorted(out) == sorted(RULES), f"got {sorted(out)}"
    for rule, rec in out.items():
        assert np.all(np.isfinite(rec.mu)), f"{rule} produced non-finite mu: {rec.mu}"
