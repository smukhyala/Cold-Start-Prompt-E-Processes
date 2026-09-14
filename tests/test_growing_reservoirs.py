"""Tests for the growing-bandit reservoirs (infinite arm pools).

Three things here are worth more than ordinary coverage:

* **Family B against Family A.** At ``c=1, mu_star=1`` the controlled-tail family is
  provably ``Beta(1, beta)``. Two independently written samplers agreeing to machine
  precision is a far stronger check than either one agreeing with itself.
* **The inverse-CDF contract.** Paired SEARCH/REFINE rollouts share a uniform stream,
  so `sample_from_uniforms` must be a genuine monotone inverse CDF for every family.
  A family that quietly sampled some other way would decouple the branches with no
  visible symptom, which is exactly the kind of bug a test has to catch.
* **Degenerate-parameter rejection.** Roughly half of Family B's nominal grid
  collapses under the clip at 0; the guard must reject those and keep the rest.
"""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.reservoirs import (
    BETA_PRESETS,
    MAX_BOUNDARY_MASS,
    MIXTURE_PRESETS,
    TAIL_BETA_GRID,
    TAIL_C_GRID,
    TAIL_MU_STAR_GRID,
    BetaReservoir,
    DegenerateReservoirError,
    MixtureReservoir,
    Reservoir,
    TailReservoir,
    all_named_reservoirs,
    build_reservoir,
    mean_by_quadrature,
    tail_grid,
    valid_tail_grid,
)
from cold_start.registry import get_registered, list_registered

SEED = 20260910

# Built once: constructing every named environment runs the full validator, so this
# doubles as a smoke test that nothing in the shipped grid is degenerate.
NAMED: dict[str, Reservoir] = all_named_reservoirs()
NAMED_IDS = sorted(NAMED)

# Mixtures invert numerically (60 bisection steps x 2 components), so keep the
# per-test sample size at a level where 30+ parametrizations stay fast.
N_MC = 100_000


def _empirical_cdf(samples: np.ndarray, xs: np.ndarray) -> np.ndarray:
    ordered = np.sort(samples)
    return np.searchsorted(ordered, xs, side="right") / ordered.size


def _exact_cdf(res: Reservoir, xs: np.ndarray) -> np.ndarray:
    """CDF through the public oracle API only, so tests never touch internals."""
    return np.array([1.0 - res.tail_prob(float(x)) for x in xs])


# ---- family B: the defining tail law -------------------------------------------


@pytest.mark.parametrize("entry", valid_tail_grid(), ids=lambda e: f"b{e['beta']:g}_m{e['mu_star']:g}_c{e['c']:g}")
def test_tail_law_holds_across_the_family_b_grid(entry: dict[str, float]) -> None:
    """``P(mu >= mu_star - eps) = c * eps^beta`` is the family's *definition*.

    Solving that for the eps that targets a fixed probability keeps the Monte Carlo
    check well-powered at every beta: a fixed eps would make the beta=8 target 1e-8
    and the test vacuous.
    """
    res = TailReservoir(**entry)
    target = 0.05
    eps = (target / res.c) ** (1.0 / res.beta)
    threshold = res.mu_star - eps

    rng = np.random.default_rng(SEED)
    draws = res.sample(rng, 200_000)
    empirical = float(np.mean(draws >= threshold))

    se = float(np.sqrt(target * (1.0 - target) / draws.size))
    assert abs(empirical - target) <= 5.0 * se, (
        f"{entry}: P(mu >= {threshold:.6f}) empirical={empirical:.6f} "
        f"vs theory={target:.6f} (5 SE = {5 * se:.6f})"
    )
    assert res.tail_prob(threshold) == pytest.approx(target, abs=1e-12), (
        f"{entry}: exact tail_prob({threshold:.6f}) = {res.tail_prob(threshold):.12f}, "
        f"expected {target}"
    )


@pytest.mark.parametrize(("beta", "expected"), [(1.0, 0.1), (2.0, 0.01), (4.0, 0.0001)])
def test_tail_probability_at_one_tenth_below_mu_star(beta: float, expected: float) -> None:
    """The specific cross-check recorded in the design: c=1, mu_star=1, eps=0.1."""
    res = TailReservoir(beta=beta, mu_star=1.0, c=1.0)
    assert res.tail_prob(0.9) == pytest.approx(expected, rel=1e-12), (
        f"beta={beta}: exact tail_prob(0.9) = {res.tail_prob(0.9)}, expected {expected}"
    )

    rng = np.random.default_rng(SEED)
    draws = res.sample(rng, 500_000)
    empirical = float(np.mean(draws >= 0.9))
    se = float(np.sqrt(max(expected, 1e-6) * (1.0 - expected) / draws.size))
    assert abs(empirical - expected) <= 5.0 * se, (
        f"beta={beta}: empirical P(mu >= 0.9) = {empirical:.6f} vs theory {expected} "
        f"(5 SE = {5 * se:.6f})"
    )


@pytest.mark.parametrize("beta", TAIL_BETA_GRID)
def test_family_b_reduces_to_beta_one_beta(beta: float) -> None:
    """At c=1, mu_star=1 the tail family is exactly ``Beta(1, beta)``.

    ``1 - V^(1/beta)`` has CDF ``1 - (1-x)^beta``, which is Beta(1, beta)'s. Two
    independent implementations agreeing at machine precision is the strongest
    cross-family check available, and it validates both at once.
    """
    tail = TailReservoir(beta=beta, mu_star=1.0, c=1.0)
    ref = BetaReservoir(1.0, float(beta))

    u = np.linspace(0.0, 1.0, 2001)
    max_diff = float(np.max(np.abs(tail.sample_from_uniforms(u) - ref.sample_from_uniforms(u))))
    assert max_diff < 1e-9, f"beta={beta}: max quantile difference {max_diff:.3e} vs Beta(1,{beta})"

    xs = np.linspace(0.0, 1.0, 101)
    max_tail_diff = float(np.max(np.abs(_exact_cdf(tail, xs) - _exact_cdf(ref, xs))))
    assert max_tail_diff < 1e-9, f"beta={beta}: max CDF difference {max_tail_diff:.3e}"

    assert tail.mean() == pytest.approx(ref.mean(), abs=1e-12), (
        f"beta={beta}: tail mean {tail.mean()} vs Beta(1,{beta}) mean {ref.mean()}"
    )


# ---- the inverse-CDF contract ---------------------------------------------------


@pytest.mark.parametrize("name", NAMED_IDS)
def test_sample_from_uniforms_is_monotone(name: str) -> None:
    """It is an inverse CDF, so it must be non-decreasing -- the coupling relies on it."""
    res = NAMED[name]
    u = np.sort(np.random.default_rng(SEED).random(5_000))
    out = res.sample_from_uniforms(u)
    steps = np.diff(out)
    assert np.all(steps >= -1e-12), (
        f"{name}: sample_from_uniforms is not monotone; "
        f"min step {float(steps.min()):.3e} at index {int(np.argmin(steps))}"
    )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_sample_agrees_in_distribution_with_sample_from_uniforms(name: str) -> None:
    """A shared uniform stream and a fresh one must give the same distribution."""
    res = NAMED[name]
    direct = res.sample(np.random.default_rng(SEED), N_MC)
    via_uniforms = res.sample_from_uniforms(np.random.default_rng(SEED + 1).random(N_MC))

    xs = np.linspace(0.0, 1.0, 201)
    gap = float(np.max(np.abs(_empirical_cdf(direct, xs) - _empirical_cdf(via_uniforms, xs))))
    # Two-sample KS at ~1e-6 significance for n = m = N_MC is about 1.95/sqrt(n/2).
    bound = 1.95 / np.sqrt(N_MC / 2.0)
    assert gap < bound, f"{name}: two-sample KS gap {gap:.5f} exceeds {bound:.5f}"


@pytest.mark.parametrize("name", NAMED_IDS)
def test_empirical_cdf_matches_the_exact_cdf(name: str) -> None:
    """`sample` must reproduce the analytic `tail_prob`, not merely reproduce itself."""
    res = NAMED[name]
    draws = res.sample(np.random.default_rng(SEED), N_MC)
    xs = np.linspace(0.0, 1.0, 201)
    gap = float(np.max(np.abs(_empirical_cdf(draws, xs) - _exact_cdf(res, xs))))
    bound = 1.95 / np.sqrt(N_MC)
    assert gap < bound, f"{name}: KS distance from the exact CDF is {gap:.5f} > {bound:.5f}"


@pytest.mark.parametrize("name", NAMED_IDS)
def test_samples_lie_in_the_unit_interval(name: str) -> None:
    res = NAMED[name]
    draws = res.sample(np.random.default_rng(SEED), 20_000)
    assert draws.shape == (20_000,), f"{name}: got shape {draws.shape}"
    assert np.all(np.isfinite(draws)), f"{name}: non-finite draws present"
    assert float(draws.min()) >= 0.0, f"{name}: minimum draw {float(draws.min())} < 0"
    assert float(draws.max()) <= 1.0, f"{name}: maximum draw {float(draws.max())} > 1"


@pytest.mark.parametrize("name", NAMED_IDS)
def test_sample_from_uniforms_rejects_out_of_range_input(name: str) -> None:
    res = NAMED[name]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        res.sample_from_uniforms(np.array([0.5, 1.5]))


# ---- quantile / tail_prob / essential_sup consistency ---------------------------


@pytest.mark.parametrize("name", NAMED_IDS)
def test_quantile_and_tail_prob_are_mutual_inverses(name: str) -> None:
    """``tail_prob(quantile(q)) == 1 - q`` wherever the CDF is continuous.

    Quantiles below 0.3 are skipped because Family B's clip at 0 creates an atom
    there (up to MAX_BOUNDARY_MASS of the mass), and no generalized inverse is a
    two-sided inverse across an atom.
    """
    res = NAMED[name]
    for q in (0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        x = res.quantile(q)
        assert res.tail_prob(x) == pytest.approx(1.0 - q, abs=1e-6), (
            f"{name}: tail_prob(quantile({q})) = {res.tail_prob(x):.9f}, expected {1 - q}"
        )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_quantile_is_non_decreasing(name: str) -> None:
    res = NAMED[name]
    qs = np.linspace(0.0, 1.0, 51)
    vals = [res.quantile(float(q)) for q in qs]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:], strict=False)), (
        f"{name}: quantile function is not monotone: {vals}"
    )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_tail_prob_matches_monte_carlo(name: str) -> None:
    res = NAMED[name]
    draws = res.sample(np.random.default_rng(SEED + 7), N_MC)
    for q in (0.25, 0.5, 0.75, 0.9):
        x = res.quantile(q)
        exact = res.tail_prob(x)
        empirical = float(np.mean(draws > x))
        se = float(np.sqrt(max(exact, 1e-6) * (1.0 - exact) / draws.size))
        assert abs(empirical - exact) <= 5.0 * se + 1e-3, (
            f"{name}: P(mu > quantile({q}) = {x:.6f}) empirical={empirical:.6f} "
            f"exact={exact:.6f} (5 SE = {5 * se:.6f})"
        )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_essential_sup_bounds_every_draw(name: str) -> None:
    res = NAMED[name]
    mu_star = res.essential_sup()
    draws = res.sample(np.random.default_rng(SEED + 3), 50_000)
    assert float(draws.max()) <= mu_star + 1e-12, (
        f"{name}: draw {float(draws.max())} exceeds essential_sup {mu_star}"
    )
    assert res.tail_prob(mu_star) == pytest.approx(0.0, abs=1e-12), (
        f"{name}: tail_prob(essential_sup) = {res.tail_prob(mu_star)}, expected 0"
    )
    assert res.quantile(1.0) == pytest.approx(mu_star, abs=1e-9), (
        f"{name}: quantile(1) = {res.quantile(1.0)}, expected essential_sup {mu_star}"
    )


def test_essential_sup_is_correct_per_family() -> None:
    assert BetaReservoir(2.0, 5.0).essential_sup() == 1.0, "Beta has full support on (0,1)"
    for mu_star in TAIL_MU_STAR_GRID:
        res = TailReservoir(beta=1.0, mu_star=mu_star, c=2.0)
        assert res.essential_sup() == pytest.approx(mu_star), (
            f"tail essential_sup {res.essential_sup()} should be mu_star {mu_star}"
        )
    mixed = MixtureReservoir(
        [BetaReservoir(2.0, 5.0), TailReservoir(beta=1.0, mu_star=0.8, c=2.0)],
        [0.5, 0.5],
    )
    assert mixed.essential_sup() == 1.0, (
        f"mixture essential_sup {mixed.essential_sup()} should be the component maximum"
    )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_mean_by_quadrature_matches_the_analytic_mean(name: str) -> None:
    """Every family ships a closed-form mean; the generic quadrature validates it."""
    res = NAMED[name]
    assert res.mean() == pytest.approx(mean_by_quadrature(res), abs=2e-4), (
        f"{name}: analytic mean {res.mean():.6f} vs quadrature {mean_by_quadrature(res):.6f}"
    )


# ---- degenerate-parameter guard --------------------------------------------------


def test_known_degenerate_combination_is_rejected() -> None:
    """c=0.5, mu_star=0.8, beta=4 puts ~80% of arms at exactly 0 (mean 0.033)."""
    with pytest.raises(DegenerateReservoirError) as excinfo:
        TailReservoir(beta=4.0, mu_star=0.8, c=0.5)
    message = str(excinfo.value)
    assert "P(mu == 0)" in message, f"error should name the failing check, got: {message}"
    assert "mean" in message, f"error should report the measured mean, got: {message}"


@pytest.mark.parametrize("beta", TAIL_BETA_GRID)
def test_known_good_combinations_are_accepted(beta: float) -> None:
    """c=1, mu_star=1 is exactly Beta(1, beta) and must survive the guard at every beta."""
    res = TailReservoir(beta=beta, mu_star=1.0, c=1.0)
    assert res.boundary_mass() == pytest.approx(0.0, abs=1e-12), (
        f"beta={beta}: Beta(1,beta) has no atom at 0, got {res.boundary_mass()}"
    )


def test_valid_tail_grid_excludes_degenerate_combinations() -> None:
    combos = valid_tail_grid()
    full = len(TAIL_BETA_GRID) * len(TAIL_MU_STAR_GRID) * len(TAIL_C_GRID)

    assert {"beta": 4.0, "mu_star": 0.8, "c": 0.5} not in combos, (
        "the measured-degenerate combination leaked into the valid grid"
    )
    assert 0 < len(combos) < full, f"expected a strict subset of {full} combos, got {len(combos)}"

    for entry in combos:
        res = TailReservoir(**entry)  # must not raise
        assert res.boundary_mass() <= MAX_BOUNDARY_MASS, (
            f"{entry}: boundary mass {res.boundary_mass():.4f} exceeds {MAX_BOUNDARY_MASS}"
        )

    covered = {entry["beta"] for entry in combos}
    assert covered == set(TAIL_BETA_GRID), (
        f"every tail exponent must remain reachable; missing {set(TAIL_BETA_GRID) - covered}"
    )
    assert {entry["mu_star"] for entry in combos} == set(TAIL_MU_STAR_GRID), (
        "every mu_star must remain reachable"
    )
    assert valid_tail_grid() is not combos, "valid_tail_grid must hand back a fresh list"


def test_valid_tail_grid_agrees_with_the_constructor() -> None:
    """The published grid is derived by construction, so it cannot drift from the guard."""
    published = {tuple(sorted(entry.items())) for entry in valid_tail_grid()}
    rebuilt: set[tuple] = set()
    for beta in TAIL_BETA_GRID:
        for mu_star in TAIL_MU_STAR_GRID:
            for c in TAIL_C_GRID:
                try:
                    TailReservoir(beta=beta, mu_star=mu_star, c=c)
                except DegenerateReservoirError:
                    continue
                rebuilt.add(tuple(sorted({"beta": beta, "mu_star": mu_star, "c": c}.items())))
    assert published == rebuilt, f"published grid {published} disagrees with the constructor"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"beta": 0.0, "mu_star": 1.0, "c": 1.0}, "beta must be positive"),
        ({"beta": 1.0, "mu_star": 0.0, "c": 1.0}, r"mu_star must lie in \(0, 1\]"),
        ({"beta": 1.0, "mu_star": 1.2, "c": 1.0}, r"mu_star must lie in \(0, 1\]"),
        ({"beta": 1.0, "mu_star": 1.0, "c": -1.0}, "c must be positive"),
    ],
)
def test_tail_reservoir_rejects_invalid_parameters(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TailReservoir(**kwargs)


@pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 2.0)])
def test_beta_reservoir_rejects_non_positive_shapes(a: float, b: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        BetaReservoir(a, b)


def test_mixture_rejects_malformed_weights() -> None:
    comps = [BetaReservoir(2.0, 5.0), BetaReservoir(5.0, 2.0)]
    with pytest.raises(ValueError, match="same length"):
        MixtureReservoir(comps, [1.0])
    with pytest.raises(ValueError, match="must sum to 1"):
        MixtureReservoir(comps, [0.5, 0.9])
    with pytest.raises(ValueError, match="non-negative"):
        MixtureReservoir(comps, [1.5, -0.5])
    with pytest.raises(ValueError, match="at least one component"):
        MixtureReservoir([], [])


def test_a_collapsed_beta_is_rejected_by_the_shared_guard() -> None:
    """The usability guard is not Family-B-specific: any near-point-mass is rejected."""
    with pytest.raises(DegenerateReservoirError, match="mean"):
        BetaReservoir(0.1, 200.0)


# ---- family C: the held-out generalization set ----------------------------------


@pytest.mark.parametrize("preset", sorted(MIXTURE_PRESETS))
def test_mixture_presets_are_bimodal_and_usable(preset: str) -> None:
    res = MixtureReservoir.from_preset(preset)
    assert res.essential_sup() == 1.0, f"{preset}: essential_sup {res.essential_sup()}"
    assert 0.05 < res.mean() < 0.95, f"{preset}: mean {res.mean():.4f} outside the usable band"
    assert res.quantile(0.99) > res.quantile(0.5) + 0.05, (
        f"{preset}: upper tail is not separated from the bulk "
        f"(q99={res.quantile(0.99):.4f}, q50={res.quantile(0.5):.4f})"
    )


def test_mixture_presets_do_not_reuse_the_family_a_grid() -> None:
    """Family C is the held-out set, so it must not be a relabeling of Family A."""
    beta_shapes = set(BETA_PRESETS.values())
    for preset, (shapes, _weights) in MIXTURE_PRESETS.items():
        overlap = beta_shapes.intersection(shapes)
        assert not overlap, f"{preset} reuses Family A shape(s) {overlap} as a component"


def test_mixture_matches_its_components_by_construction() -> None:
    """A degenerate mixture with one component must reproduce that component exactly."""
    comp = BetaReservoir(6.0, 6.0)
    solo = MixtureReservoir([comp], [1.0])
    u = np.linspace(0.0, 1.0, 501)
    max_diff = float(np.max(np.abs(solo.sample_from_uniforms(u) - comp.sample_from_uniforms(u))))
    assert max_diff < 1e-9, f"single-component mixture differs from its component by {max_diff:.3e}"


def test_mixture_accepts_component_specs() -> None:
    """Configs record components as {"type", "params"} dicts, not live objects."""
    from_specs = MixtureReservoir(
        [{"type": "beta", "params": {"a": 6.0, "b": 6.0}}, {"type": "beta", "params": {"a": 30.0, "b": 3.0}}],
        [0.97, 0.03],
    )
    from_objects = MixtureReservoir.from_preset("many_mediocre_rare_excellent")
    u = np.linspace(0.0, 1.0, 501)
    assert np.allclose(
        from_specs.sample_from_uniforms(u), from_objects.sample_from_uniforms(u), atol=1e-12
    ), "spec-constructed mixture differs from the preset"


# ---- registry, provenance, determinism -------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls"),
    [("beta", BetaReservoir), ("tail", TailReservoir), ("mixture", MixtureReservoir)],
)
def test_registry_round_trip(name: str, cls: type) -> None:
    assert get_registered("reservoir", name) is cls, (
        f"registry returned {get_registered('reservoir', name)!r} for {name!r}, expected {cls!r}"
    )
    assert name in list_registered("reservoir"), (
        f"{name!r} missing from list_registered('reservoir') = {list_registered('reservoir')}"
    )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_params_round_trip_through_build_reservoir(name: str) -> None:
    """Every trajectory records `params()`; a run must be replayable from them."""
    res = NAMED[name]
    rebuilt = build_reservoir(res.to_spec())
    u = np.linspace(0.0, 1.0, 501)
    assert np.allclose(res.sample_from_uniforms(u), rebuilt.sample_from_uniforms(u), atol=1e-12), (
        f"{name}: rebuilt from {res.to_spec()} does not reproduce the original"
    )


@pytest.mark.parametrize("preset", sorted(BETA_PRESETS))
def test_build_reservoir_accepts_a_preset_only_spec(preset: str) -> None:
    res = build_reservoir({"type": "beta", "params": {"preset": preset}})
    assert (res.a, res.b) == BETA_PRESETS[preset], (
        f"{preset}: built Beta({res.a}, {res.b}), expected {BETA_PRESETS[preset]}"
    )


@pytest.mark.parametrize("name", NAMED_IDS)
def test_same_seed_gives_identical_draws(name: str) -> None:
    res = NAMED[name]
    first = res.sample(np.random.default_rng(SEED), 5_000)
    second = res.sample(np.random.default_rng(SEED), 5_000)
    assert np.array_equal(first, second), f"{name}: identical seeds produced different draws"

    other = res.sample(np.random.default_rng(SEED + 1), 5_000)
    assert not np.array_equal(first, other), f"{name}: different seeds produced identical draws"


def test_params_are_json_serializable() -> None:
    import json

    for name, res in NAMED.items():
        payload = json.dumps(res.to_spec(), sort_keys=True)
        assert json.loads(payload)["type"] == res.family, f"{name}: spec type mismatch"


def test_tail_grid_keys_are_readable_and_unique() -> None:
    grid = tail_grid()
    assert len(grid) == len(valid_tail_grid()), (
        f"tail_grid has {len(grid)} entries but valid_tail_grid has {len(valid_tail_grid())}"
    )
    assert "beta1_mustar1_c1" in grid, f"expected the Beta(1,1)-equivalent key in {sorted(grid)}"


def test_sample_rejects_negative_size() -> None:
    res = BetaReservoir(2.0, 5.0)
    with pytest.raises(ValueError, match="non-negative"):
        res.sample(np.random.default_rng(SEED), -1)
