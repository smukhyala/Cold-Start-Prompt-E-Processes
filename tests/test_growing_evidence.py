from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hyp2f1
from scipy.stats import binom

from cold_start.growing import evidence as ev
from cold_start.growing.evidence import (
    GrowingMixtureEProcess,
    PairwiseEvidence,
    cs_bounds,
    cs_grid,
    cs_lower_upper_for_n,
    d_log_e_lower_dm,
    log_e_lower,
    log_e_lower_flat_table,
    log_e_lower_grid,
    log_e_upper,
    pairwise_log_e,
)
from cold_start.growing.schema import EvidenceSpec
from cold_start.growing.state import AgrapaParams, GrowingState
from cold_start.growing.tables import CSTable
from cold_start.registry import get_registered

ALPHA = 0.05


# ---- helpers -------------------------------------------------------------------


@lru_cache(maxsize=4)
def _cs_table(horizon: int, alpha: float = ALPHA) -> CSTable:
    """Cached on disk under data/cs_tables/, so a rerun of the suite is instant."""
    return CSTable.load_or_build(horizon, alpha)


@lru_cache(maxsize=4)
def _pair_table(max_n: int, n_intervals: int = 512):
    return PairwiseEvidence(n_intervals=n_intervals).tabulate(max_n)


def _sup_log_e_rate(n_runs: int, T: int, mu: float, m: float, alpha: float, seed: int) -> float:
    """Empirical P(sup_n E_n >= 1/alpha) for a single Bernoulli(mu) arm against H0: mu <= m.

    Vectorized over runs by gathering a precomputed (n, S) table -- the whole point of
    an order-invariant e-process is that the running value is a lookup.
    """
    rng = np.random.default_rng(seed)
    table = log_e_lower_flat_table(T, m)
    rewards = rng.random((n_runs, T)) < mu
    successes = np.cumsum(rewards, axis=1).astype(np.int64)
    pulls = np.arange(1, T + 1, dtype=np.int64)[None, :]
    log_e = table[pulls * (T + 1) + successes]
    return float(np.mean(log_e.max(axis=1) >= math.log(1.0 / alpha)))


def _two_arm_rate(n_runs: int, T: int, mu: float, m: float, rule: str, seed: int) -> float:
    """P(sup_n E_n >= 1/alpha) for ARM 0 ONLY, when a rule decides which arm to pull.

    Both arms are Bernoulli(mu) with mu = m, the boundary of H0. Arm 0's e-process
    sees an adaptively-chosen *subsequence* of a Bernoulli stream; optional skipping
    says it stays a supermartingale as long as the choice is predictable. `peek` is
    the deliberate violation.
    """
    rng = np.random.default_rng(seed)
    table = log_e_lower_flat_table(T, m)
    stride = T + 1
    rewards = rng.random((2, n_runs, T)) < mu
    coin = rng.random((n_runs, T))
    pulls = np.zeros((2, n_runs), dtype=np.int64)
    successes = np.zeros((2, n_runs), dtype=np.int64)
    log_e = np.zeros((2, n_runs))
    sup = np.zeros(n_runs)
    runs = np.arange(n_runs)
    for t in range(T):
        if rule == "eps_greedy":
            greedy = (successes[0] + 1) / (pulls[0] + 2) >= (successes[1] + 1) / (pulls[1] + 2)
            take_0 = np.where(coin[:, t] < 0.1, rng.random(n_runs) < 0.5, greedy)
        elif rule == "chase_evidence":
            take_0 = log_e[0] >= log_e[1]
        elif rule == "peek":
            take_0 = rewards[0, :, t] >= rewards[1, :, t]
        else:  # pragma: no cover - guard
            raise ValueError(rule)
        arm = (~take_0).astype(np.int64)
        pulls[arm, runs] += 1
        successes[arm, runs] += rewards[arm, runs, t]
        log_e[arm, runs] = table[pulls[arm, runs] * stride + successes[arm, runs]]
        np.maximum(sup, log_e[0], out=sup)
    return float(np.mean(sup >= math.log(1.0 / ALPHA)))


def _pairwise_rate(
    n_runs: int, T: int, mu_lead: float, mu_chal: float, alpha: float, rule: str, table, seed: int
) -> float:
    """P(sup_t e_pair >= 1/alpha) for H0: mu_lead <= mu_chal, under a sampling rule.

    `T` is the *total* pull budget shared by the two arms, so the sampling rule really
    decides where information goes -- which is the sharp version of the worry that
    data-dependent pull times break the product-of-e-processes argument.
    """
    rng = np.random.default_rng(seed)
    r_lead = rng.random((n_runs, T)) < mu_lead
    r_chal = rng.random((n_runs, T)) < mu_chal
    tiebreak = rng.random((n_runs, T)) < 0.5
    n_l = np.zeros(n_runs, dtype=np.int64)
    s_l = np.zeros(n_runs, dtype=np.int64)
    n_c = np.zeros(n_runs, dtype=np.int64)
    s_c = np.zeros(n_runs, dtype=np.int64)
    row_l = table.lead_rows(n_l, s_l).copy()
    row_c = table.chal_rows(n_c, s_c).copy()
    sup = np.full(n_runs, -np.inf)
    for t in range(T):
        if rule == "alternating":
            take_lead = np.full(n_runs, t % 2 == 0)
        elif rule == "count":
            take_lead = np.where(n_l == n_c, tiebreak[:, t], n_l < n_c)
        elif rule == "evidence":
            take_lead = (s_l + 1) / (n_l + 2) >= (s_c + 1) / (n_c + 2)
        else:  # pragma: no cover - guard
            raise ValueError(rule)
        i = np.flatnonzero(take_lead)
        j = np.flatnonzero(~take_lead)
        if i.size:
            n_l[i] += 1
            s_l[i] += r_lead[i, t]
            row_l[i] = table.lead_rows(n_l[i], s_l[i])
        if j.size:
            n_c[j] += 1
            s_c[j] += r_chal[j, t]
            row_c[j] = table.chal_rows(n_c[j], s_c[j])
        np.maximum(sup, (row_l + row_c).min(axis=1), out=sup)
    return float(np.mean(sup >= math.log(1.0 / alpha)))


def _log_e_lower_via_J(n: float, S: float, m: float, a: float = 1.0, b: float = 1.0) -> float:
    """Independent reference: the exact-cancellation hypergeometric form.

        log E^{<=m} = J(a+S, b+n-S, m) - J(a, b, m) - S log m,
        J(A,B,m) = -log B + log 2F1(1-A, B; B+1; 1-m)

    Every (1-m) power cancels analytically, so this never underflows -- it is the
    ground truth for the m -> 1 band where the binomial-tail route dies.
    """

    def big_j(A: float, B: float) -> float:
        return -math.log(B) + float(np.log(hyp2f1(1.0 - A, B, B + 1.0, 1.0 - m)))

    shift = -S * math.log(m) if S > 0 else 0.0
    return big_j(a + S, b + n - S) - big_j(a, b) + shift


def _log_e_lower_via_quad(n: float, S: float, m: float, a: float, b: float) -> float:
    """Independent reference: adaptive quadrature of the defining ratio.

    The integrand is rescaled by its own log-peak before integration, so quadrature
    never sees an under/overflowing value and stays accurate for large exponents.
    """

    def log_tail(A: float, B: float) -> float:
        peak = min(max((A - 1) / (A + B - 2) if A > 1 and B > 1 else m, m), 1.0 - 1e-15)
        log_peak = (A - 1) * math.log(peak) + (B - 1) * math.log1p(-peak)
        value, _ = quad(
            lambda t: math.exp((A - 1) * math.log(t) + (B - 1) * math.log1p(-t) - log_peak),
            m,
            1.0,
            limit=500,
            epsabs=1e-14,
            epsrel=1e-14,
        )
        return math.log(value) + log_peak

    shift = (-S * math.log(m) if S > 0 else 0.0) + (
        -(n - S) * math.log1p(-m) if n - S > 0 else 0.0
    )
    return log_tail(a + S, b + n - S) - log_tail(a, b) + shift


# ---- construction and edge cases -----------------------------------------------


def test_edge_cases_are_exact_not_numerical():
    """The construction's degenerate points have closed forms; assert them, do not fit them."""
    assert float(log_e_lower(0, 0, 0.37)) == pytest.approx(0.0, abs=1e-12), "n=0 must give E=1"
    assert float(log_e_upper(0, 0, 0.37)) == pytest.approx(0.0, abs=1e-12), "n=0 must give E=1"

    # S = 0 under a flat prior gives exactly 1/(n+1), constant in m -- which is why
    # the lower CS endpoint is exactly 0 for every all-failure arm.
    for m in (0.05, 0.5, 0.95):
        got = float(log_e_lower(10, 0, m))
        assert got == pytest.approx(-math.log(11.0), abs=1e-12), f"S=0 at m={m} gave {got}"

    # m -> 1: the limit is b/(b+n-S) <= 1, so the rejection region is bounded away
    # from 1 and the null "mu <= 1" is untestable, as it must be.
    for n, S in ((10, 3), (100, 40), (1000, 999)):
        got = float(log_e_lower(n, S, 1.0))
        assert got == pytest.approx(-math.log(n + 1 - S), rel=1e-10), f"m->1 at ({n},{S})"
    # Mirrored: m -> 0 gives a/(a+S).
    for n, S in ((10, 3), (100, 40)):
        got = float(log_e_upper(n, S, 0.0))
        assert got == pytest.approx(-math.log(S + 1), rel=1e-10), f"m->0 upper at ({n},{S})"

    # m -> 0 with S >= 1 is unbounded evidence (the denominator m^S vanishes).
    assert float(log_e_lower(10, 3, 0.0)) > 100.0


def test_upper_is_the_exact_reflection_of_lower():
    rng = np.random.default_rng(3)
    n = rng.integers(0, 200, 200)
    S = np.array([rng.integers(0, k + 1) for k in n])
    m = rng.random(200) * 0.98 + 0.01
    direct = log_e_upper(n, S, m)
    reflected = log_e_lower(n, n - S, 1.0 - m)
    assert np.max(np.abs(direct - reflected)) < 1e-12


def test_matches_hypergeometric_J_form_including_the_underflow_band():
    """T7: agreement with the exact-cancellation form, especially where m -> 1.

    This is the assertion that would have caught the silent-nan bug immediately: the
    J-form has no (1-m)^k factor to underflow, so any disagreement means the primary
    path has saturated.
    """
    worst = 0.0
    where = None
    for n in (50, 100, 200, 400, 800):
        for frac in (0.02, 0.1, 0.5, 0.9, 0.98):
            S = int(round(frac * n))
            for m in (1e-5, 1e-3, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0 - 1e-5):
                mine = float(log_e_lower(n, S, m))
                ref = _log_e_lower_via_J(n, S, m)
                assert math.isfinite(mine), f"non-finite log_e at n={n}, S={S}, m={m}"
                if abs(mine - ref) > worst:
                    worst = abs(mine - ref)
                    where = (n, S, m, mine, ref)
    assert worst < 1e-8, f"worst |ours - J-form| = {worst:.3e} at {where}"


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("prior", [(1.0, 1.0), (0.5, 0.5), (2.0, 3.0)])
def test_matches_adaptive_quadrature(prior: tuple[float, float]):
    """Closed form vs numerical integration of the definition, for three priors."""
    a, b = prior
    worst = 0.0
    where = None
    for n in (0, 1, 5, 17, 30):
        for S in range(n + 1):
            for m in (1e-4, 0.01, 0.1, 0.5, 0.9, 0.99):
                mine = float(log_e_lower(n, S, m, prior_a=a, prior_b=b))
                ref = _log_e_lower_via_quad(n, S, m, a, b)
                rel = abs(mine - ref) / max(1.0, abs(ref))
                if rel > worst:
                    worst = rel
                    where = (n, S, m, mine, ref)
    assert worst < 1e-9, f"prior={prior}: worst relative gap {worst:.3e} at {where}"


def test_binomial_tail_route_underflows_where_ours_does_not():
    """Regression pin for the failure mode this module exists to avoid.

    `betainc` saturates by n ~ 100 and the binomial-tail form underflows to -inf well
    inside the working range. If someone ever "simplifies" back to either, this test
    documents exactly what breaks and where.
    """
    naive = float(binom.logcdf(40, 401, 0.922))
    assert naive == -np.inf, f"expected the binomial tail to underflow; got {naive}"
    ours = float(log_e_lower(400, 40, 0.922))
    assert math.isfinite(ours), "our path must survive where the binomial tail dies"
    assert ours == pytest.approx(_log_e_lower_via_J(400, 40, 0.922), abs=1e-9)
    # The true value there is a mild number, not -inf: mistaking one for the other is
    # what poisons a minimum-over-grid statistic with nan.
    assert -10.0 < ours < 0.0, f"log_e at the underflow point should be O(1); got {ours}"


def test_every_cover_term_the_pairwise_statistic_consumes_is_finite():
    """T7: the one-line invariant, asserted on the whole cube rather than a sample."""
    pw = PairwiseEvidence(n_intervals=512)
    right = pw.grid[1:]
    left = pw.grid[:-1]
    for n in (50, 150, 400, 800):
        S = np.arange(0, n + 1, max(1, n // 40))
        lead = log_e_lower(n, S[:, None], right)
        chal = log_e_upper(n, S[:, None], left)
        assert np.isfinite(lead).all(), f"leader cover terms non-finite at n={n}"
        assert np.isfinite(chal).all(), f"challenger cover terms non-finite at n={n}"
        assert np.isfinite(lead + chal).all(), f"cover sum non-finite at n={n}"


# ---- monotonicity is not a theorem ---------------------------------------------


def test_log_e_is_not_monotone_in_m_in_general():
    """The counterexample that forbids naive bisection for the CS endpoints.

    a=b=10, n=50, S=1, m=0.3 is interior and numerically benign, and the derivative of
    log E against H0: mu <= m is +13.95 there. A bisection that assumed monotonicity
    could latch onto a later crossing, return L too large, and silently lose coverage.
    """
    deriv = float(d_log_e_lower_dm(50, 1, 0.3, prior_a=10.0, prior_b=10.0))
    assert deriv == pytest.approx(13.954, abs=1e-2), f"derivative was {deriv}"
    assert deriv > 0.0, "monotonicity in m must NOT be assumed"

    # Confirm the analytic derivative against finite differences, so the counterexample
    # cannot be dismissed as a bug in the derivative formula.
    h = 1e-6
    fd = (
        float(log_e_lower(50, 1, 0.3 + h, prior_a=10.0, prior_b=10.0))
        - float(log_e_lower(50, 1, 0.3 - h, prior_a=10.0, prior_b=10.0))
    ) / (2 * h)
    assert fd == pytest.approx(deriv, rel=1e-5), f"analytic {deriv} vs finite-diff {fd}"


def test_first_crossing_search_agrees_with_the_grid_interval_hull():
    """For the flat prior the retained set is an interval, so the two agree.

    Contiguity is checked rather than assumed: it is what makes "first crossing" and
    "min over survivors" the same answer, and it is an empirical property of a=b=1,
    not a theorem (see the previous test).
    """
    grid = cs_grid(512)
    threshold = math.log(1.0 / ALPHA)
    spacing = float(np.max(np.diff(grid)))
    worst_gap = 0.0
    for n in range(0, 61):
        values = log_e_lower_grid(n, grid)
        retained = values < threshold
        idx = np.argmax(retained, axis=0)
        for S in range(n + 1):
            kept = np.flatnonzero(retained[:, S])
            assert kept.size > 0, f"nothing retained at (n={n}, S={S})"
            assert kept[-1] - kept[0] + 1 == kept.size, (
                f"retained set is not an interval at (n={n}, S={S})"
            )
        hull = grid[idx]
        refined, _ = cs_lower_upper_for_n(n, threshold, grid)
        assert np.all(refined <= hull + 1e-12), "refinement must never narrow the interval"
        worst_gap = max(worst_gap, float(np.max(hull - refined)))
    assert worst_gap <= spacing + 1e-12, f"refinement left the bracket: {worst_gap} > {spacing}"


# ---- Ville bounds for the per-arm e-process -------------------------------------


@pytest.mark.parametrize("m", [0.3, 0.5, 0.8])
def test_ville_bound_at_the_null_boundary(m: float):
    """P(sup_n E_n >= 1/alpha | mu = m) <= alpha, the worst case inside H0: mu <= m."""
    rate = _sup_log_e_rate(2000, 300, mu=m, m=m, alpha=ALPHA, seed=7 + int(m * 10))
    assert rate <= ALPHA + 0.01, f"m={m}: boundary rejection rate {rate:.4f} exceeds alpha"


@pytest.mark.parametrize("m", [0.3, 0.5, 0.9])
def test_ville_bound_holds_at_the_full_study_horizon(m: float):
    """The same at T=1000, where a saturating implementation would already be broken."""
    rate = _sup_log_e_rate(2000, 1000, mu=m, m=m, alpha=ALPHA, seed=41 + int(m * 10))
    assert rate <= ALPHA + 0.01, f"m={m}, T=1000: rejection rate {rate:.4f} exceeds alpha"


@pytest.mark.parametrize(("mu", "m"), [(0.3, 0.5), (0.5, 0.8)])
def test_strict_interior_of_the_null_is_far_more_conservative(mu: float, m: float):
    """Away from the boundary the mixture is not merely valid but nearly silent."""
    rate = _sup_log_e_rate(2000, 300, mu=mu, m=m, alpha=ALPHA, seed=21 + int(m * 10))
    assert rate <= 0.005, f"mu={mu}, m={m}: interior rate {rate:.4f} is not conservative"


@pytest.mark.parametrize(("mu", "expected"), [(0.75, 0.95), (0.6, 0.60)])
def test_power_under_the_alternative(mu: float, expected: float):
    """Validity is worthless without power; pin both."""
    rate = _sup_log_e_rate(1000, 300, mu=mu, m=0.5, alpha=ALPHA, seed=13 + int(mu * 10))
    assert rate >= expected, f"mu={mu}: power {rate:.4f} below {expected}"


@pytest.mark.parametrize("mu", [0.2, 0.5, 0.9])
def test_confidence_sequence_covers_uniformly_over_time(mu: float):
    """P(exists n <= T : mu not in CS_n) <= 2*alpha (two one-sided tests)."""
    table = _cs_table(200)
    rng = np.random.default_rng(int(mu * 100) + 3)
    rewards = rng.random((2000, 200)) < mu
    successes = np.cumsum(rewards, axis=1).astype(np.int64)
    pulls = np.arange(1, 201, dtype=np.int64)[None, :]
    lower, upper = table.bounds(pulls, successes)
    miss = float(np.mean(np.any((mu < lower) | (mu > upper), axis=1)))
    assert miss <= 2 * ALPHA, f"mu={mu}: uniform miscoverage {miss:.4f} exceeds {2 * ALPHA}"


@pytest.mark.parametrize("rule", ["eps_greedy", "chase_evidence"])
def test_valid_under_adaptive_allocation_and_optional_skipping(rule: str):
    """Two arms at the boundary; a predictable allocator decides which one advances.

    Arm 0's e-process therefore sees an adaptively-chosen subsequence. Optional
    skipping keeps it a supermartingale as long as the choice does not look at the
    reward it is about to receive -- `chase_evidence` is the sharpest predictable
    case, since it deliberately follows whichever arm currently looks strongest.
    """
    rate = _two_arm_rate(2000, 300, mu=0.5, m=0.5, rule=rule, seed=31)
    assert rate <= ALPHA + 0.03, f"{rule}: rate {rate:.4f} exceeds alpha + 0.03"


def test_negative_control_a_peeking_allocator_must_break_the_null():
    """INTENTIONAL FAILURE OF THE NULL -- this test asserts a HIGH rejection rate.

    If the allocator inspects the reward before deciding which arm to pull, the choice
    is no longer predictable, optional skipping does not apply, and arm 0's statistic
    is not an e-process. Without this, the passing validity tests above would prove
    nothing about whether they have teeth: a construction that never rejects anything
    passes all of them.
    """
    rate = _two_arm_rate(2000, 300, mu=0.5, m=0.5, rule="peek", seed=31)
    assert rate > 0.5, (
        f"peeking allocator rejected at only {rate:.4f}; the validity tests above are "
        "not discriminating, because a statistic that never fires would also pass them"
    )


# ---- confidence-sequence tables -------------------------------------------------


def test_table_is_finite_and_non_degenerate_at_the_full_horizon():
    """Regression test for the saturation bug, at the horizon the study actually uses.

    Finiteness alone is not enough: a saturated special function yields -inf, the null
    is never rejected, and the endpoints collapse to 0 and 1 -- still finite, still a
    table, and completely useless. So the widths are pinned too.
    """
    table = _cs_table(1000)
    mask = table.valid_mask()
    lower = np.asarray(table.lower_flat)[mask]
    upper = np.asarray(table.upper_flat)[mask]
    assert np.isfinite(lower).all(), "non-finite lower bounds at valid (n, S)"
    assert np.isfinite(upper).all(), "non-finite upper bounds at valid (n, S)"
    assert lower.min() >= 0.0 and upper.max() <= 1.0
    assert np.all(upper >= lower)

    S = np.arange(1001)
    lo, hi = table.bounds(np.full(1001, 1000), S)
    widest = float(np.max(hi - lo))
    assert widest < 0.15, f"CS at n=1000 is degenerate: widest interval {widest:.3f}"


def test_table_agrees_with_the_reference_bound_computation():
    """The fast cumulative build vs the slow grid-scan-plus-bisection reference."""
    table = _cs_table(1000)
    rng = np.random.default_rng(0)
    n = rng.integers(0, 1001, 40)
    S = np.array([rng.integers(0, k + 1) for k in n])
    fast_lo, fast_hi = table.bounds(n, S)
    ref_lo, ref_hi = cs_bounds(n, S, alpha=ALPHA)
    gap = max(float(np.max(np.abs(fast_lo - ref_lo))), float(np.max(np.abs(fast_hi - ref_hi))))
    assert gap < 1e-3, f"table and reference bounds differ by {gap:.3e}"


def test_confidence_bounds_are_sane():
    table = _cs_table(1000)

    lo, hi = table.bounds(0, 0)
    assert (float(lo), float(hi)) == (0.0, 1.0), "an unpulled arm must report [0, 1]"

    # The posterior mean is always inside the interval -- if it were not, the CS would
    # be excluding the very value the data most supports.
    n_idx, s_idx = np.divmod(np.arange(table.lower_flat.size), table.stride)
    valid = s_idx <= n_idx
    post_mean = (s_idx[valid] + 1.0) / (n_idx[valid] + 2.0)
    assert np.all(np.asarray(table.lower_flat)[valid] <= post_mean + 1e-6)
    assert np.all(post_mean <= np.asarray(table.upper_flat)[valid] + 1e-6)

    widths = [float(table.width(n, n // 2)) for n in (10, 50, 200, 1000)]
    assert all(a > b for a, b in zip(widths[:-1], widths[1:], strict=True)), f"widths do not shrink: {widths}"
    assert widths[0] < 1.0 and widths[-1] < 0.15


def test_table_build_is_deterministic_and_the_cache_round_trips(tmp_path: Path):
    first = CSTable.build(60, ALPHA)
    second = CSTable.build(60, ALPHA)
    assert np.array_equal(first.lower_flat, second.lower_flat, equal_nan=True)
    assert np.array_equal(first.upper_flat, second.upper_flat, equal_nan=True)

    cached = CSTable.load_or_build(60, ALPHA, cache_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 2 and all(f.endswith(".npy") for f in files), files
    reloaded = CSTable.load_or_build(60, ALPHA, cache_dir=tmp_path)
    assert np.array_equal(np.asarray(cached.lower_flat), first.lower_flat, equal_nan=True)
    assert np.array_equal(np.asarray(reloaded.upper_flat), first.upper_flat, equal_nan=True)
    # Memory-mapped so worker processes share one copy of the pages.
    assert isinstance(reloaded.lower_flat, np.memmap)


def test_table_satisfies_the_bound_table_protocol_used_by_the_simulator():
    """The table must drop straight into GrowingState.pull with no adapter."""
    table = _cs_table(200)
    state = GrowingState(
        n_replicates=8, capacity=4, horizon=200, base_seed=99, agrapa=AgrapaParams(m0=0.5)
    )
    state.add_arms(np.full(8, 0.7, dtype=np.float32), np.arange(8, dtype=np.int32))
    cols = np.zeros(8, dtype=np.int64)
    for _ in range(30):
        state.pull(cols, table)
    lin = state.row_off + cols
    assert np.all(state.n[lin] == 30)
    assert np.all(np.isfinite(state.lcb[lin])) and np.all(np.isfinite(state.ucb[lin]))
    assert np.all(state.lcb[lin] <= state.ucb[lin])


# ---- pairwise leader-vs-challenger evidence -------------------------------------


def test_interval_cover_never_exceeds_the_naive_point_grid_minimum():
    """The invariant that makes the cover's validity inherited rather than hoped for.

    A minimum over grid *points* overshoots the true infimum and is anti-conservative
    (measured elsewhere at 0.0525 against alpha = 0.10). The interval cover is an
    exact finite union, so it must sit at or below the point form everywhere.
    """
    pw = PairwiseEvidence(n_intervals=128)
    interior = pw.grid[1:-1]
    worst = -np.inf
    for n_l, s_l, n_c, s_c in [
        (40, 30, 40, 20),
        (100, 90, 100, 50),
        (7, 3, 11, 9),
        (150, 15, 150, 140),
        (5, 5, 5, 0),
        (0, 0, 0, 0),
    ]:
        cover = float(pw.log_e(n_l, s_l, n_c, s_c))
        naive = float(
            (log_e_lower(n_l, s_l, interior) + log_e_upper(n_c, s_c, interior)).min()
        )
        worst = max(worst, cover - naive)
    assert worst <= 1e-9, f"cover exceeded the point-grid minimum by {worst:.3e}"


@pytest.mark.parametrize("means", [(0.5, 0.5), (0.9, 0.9), (0.1, 0.1)])
def test_cover_null_rate_at_the_boundary(means: tuple[float, float]):
    """mu_lead == mu_chal is the boundary of H0: mu_lead <= mu_chal. alpha = 0.10."""
    mu_l, mu_c = means
    table = _pair_table(150)
    rate = _pairwise_rate(2000, 150, mu_l, mu_c, 0.10, "alternating", table, seed=99)
    assert rate <= 0.10, f"({mu_l}, {mu_c}): cover null rate {rate:.4f} exceeds alpha=0.10"


@pytest.mark.parametrize("rule", ["alternating", "count", "evidence"])
@pytest.mark.parametrize("means", [(0.5, 0.5), (0.7, 0.7), (0.3, 0.3)])
def test_cover_ville_bound_under_three_sampling_rules(rule: str, means: tuple[float, float]):
    """Adaptive, data-dependent pull times are the sharpest test of the product argument.

    `count` balances pulls, `evidence` chases whichever arm currently looks better --
    the case that most plausibly breaks a product of two separately-valid e-processes.
    """
    mu_l, mu_c = means
    table = _pair_table(120)
    rate = _pairwise_rate(1500, 120, mu_l, mu_c, ALPHA, rule, table, seed=5)
    assert rate <= ALPHA, f"{rule} at ({mu_l}, {mu_c}): rate {rate:.4f} exceeds alpha"


@pytest.mark.parametrize("means", [(0.4, 0.6), (0.5, 0.8)])
def test_cover_is_very_conservative_strictly_inside_the_null(means: tuple[float, float]):
    mu_l, mu_c = means
    table = _pair_table(120)
    rate = _pairwise_rate(1500, 120, mu_l, mu_c, ALPHA, "alternating", table, seed=5)
    assert rate <= 0.01, f"({mu_l}, {mu_c}): interior rate {rate:.4f} is not conservative"


def test_cover_has_power_when_the_leader_really_is_better():
    table150 = _pair_table(150)
    strong = _pairwise_rate(1000, 150, 0.75, 0.45, ALPHA, "alternating", table150, seed=17)
    assert strong >= 0.60, f"power at gap 0.30 with 150 pulls was only {strong:.3f}"

    table120 = _pair_table(120)
    gap20 = _pairwise_rate(1000, 120, 0.60, 0.40, ALPHA, "alternating", table120, seed=18)
    gap30 = _pairwise_rate(1000, 120, 0.65, 0.35, ALPHA, "alternating", table120, seed=19)
    assert gap20 >= 0.20, f"power at gap 0.20 was {gap20:.3f}"
    assert gap30 >= 0.50, f"power at gap 0.30 was {gap30:.3f}"
    assert gap30 > gap20, "power must increase with the gap"


def test_cover_power_increases_with_grid_resolution():
    """The observable signature that the interval cover is doing what the theory says.

    Each interval null shrinks toward a point null as the grid refines, so the cover
    minimum rises and power increases monotonically. A construction that got the
    left/right endpoint pairing backwards would not show this.
    """
    powers = []
    for n_intervals in (16, 64, 256, 512):
        table = PairwiseEvidence(n_intervals=n_intervals).tabulate(150)
        powers.append(_pairwise_rate(600, 150, 0.75, 0.45, ALPHA, "alternating", table, seed=17))
    assert all(a < b for a, b in zip(powers[:-1], powers[1:], strict=True)), f"power not monotone in G: {powers}"


def test_cover_power_pin_at_large_n():
    """The regime where the naive implementation silently returned nan instead of e^71."""
    value = float(pairwise_log_e(400, 360, 400, 200))
    assert math.isfinite(value), "cover must be finite in the large-n endgame"
    assert value > 50.0, f"cover evidence at 360/400 vs 200/400 was only {value:.2f}"
    for n_l, s_l, n_c, s_c, floor in ((200, 180, 200, 100, 25.0), (100, 90, 100, 50, 12.0)):
        got = float(pairwise_log_e(n_l, s_l, n_c, s_c))
        assert math.isfinite(got) and got > floor, f"{s_l}/{n_l} vs {s_c}/{n_c} gave {got}"


def test_tabulated_and_on_the_fly_cover_agree():
    table = _pair_table(120)
    pw = PairwiseEvidence(n_intervals=512)
    rng = np.random.default_rng(11)
    n_l = rng.integers(0, 121, 50)
    s_l = np.array([rng.integers(0, k + 1) for k in n_l])
    n_c = rng.integers(0, 121, 50)
    s_c = np.array([rng.integers(0, k + 1) for k in n_c])
    fast = table.log_e(n_l, s_l, n_c, s_c)
    slow = pw.log_e(n_l, s_l, n_c, s_c)
    assert np.max(np.abs(fast - slow)) < 1e-2, "float32 table drifted from the f64 path"


def test_tabulate_refuses_to_allocate_a_gigabyte():
    with pytest.raises(ValueError, match="max_bytes"):
        PairwiseEvidence(n_intervals=512).tabulate(1000)


# ---- component wiring ----------------------------------------------------------


def test_growing_mixture_is_registered_and_matches_the_functional_api():
    cls = get_registered("eprocess", "growing_mixture")
    assert cls is GrowingMixtureEProcess
    proc = cls.from_spec(EvidenceSpec(alpha=ALPHA, agrapa_m0=0.4))
    assert proc.m0 == 0.4 and proc.alpha == ALPHA
    assert float(proc.log_e(20, 15)) == pytest.approx(float(log_e_lower(20, 15, 0.4)))
    assert float(proc.log_e_downward(20, 15)) == pytest.approx(float(log_e_upper(20, 15, 0.4)))
    assert bool(proc.rejects(20, 15)) == (float(log_e_lower(20, 15, 0.4)) >= proc.log_threshold)
    lo, hi = proc.bounds(20, 15)
    assert 0.0 <= lo <= 0.75 <= hi <= 1.0

    # Deliberately NOT an EProcess: that ABC is built around order-dependent update(x).
    from cold_start.inference.base import EProcess

    assert not issubclass(cls, EProcess)
    assert issubclass(cls, ev.SufficientStatEProcess)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        GrowingMixtureEProcess(m0=0.0)
    with pytest.raises(ValueError):
        GrowingMixtureEProcess(alpha=1.0)
    with pytest.raises(ValueError):
        log_e_lower(5, 2, 0.5, prior_a=-1.0)
    with pytest.raises(ValueError):
        cs_bounds(5, 9)
    with pytest.raises(ValueError):
        cs_grid(1)
    with pytest.raises(NotImplementedError):
        CSTable.build(10, ALPHA, prior_a=2.0)
