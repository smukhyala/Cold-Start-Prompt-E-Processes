"""Reservoirs: the infinite arm pools a SEARCH action draws from.

In the growing-bandit formulation the arm set is not fixed. A SEARCH action draws a
brand-new arm whose true mean ``mu`` comes from a *reservoir* distribution ``Q`` on
``[0, 1]``. Everything the study is trying to learn -- when it pays to draw a fresh
arm instead of refining an incumbent -- is a function of the shape of ``Q``'s upper
tail, so ``Q`` is a first-class, parameterized, recorded object rather than a fixed
choice buried in the simulator.

Three design decisions are load-bearing.

**Inverse-CDF sampling is mandatory, not an optimization.** The oracle label pairs
two Monte Carlo branches (forced SEARCH vs forced REFINE) driven by a shared uniform
stream, so that the j-th new arm is the *same* underlying draw in both branches. That
coupling only exists if every family can map a fixed uniform to a draw, which is why
`sample_from_uniforms` is the primitive and `sample` is defined in terms of it: a
family that sampled some other way would silently decouple the branches with no
visible symptom. Families without a closed-form inverse (mixtures) invert numerically.

**Oracle methods are separated by name, not by convention.** `tail_prob` and
`quantile` are exact properties of ``Q`` that no deployable policy could ever compute.
They exist for analysis and for the ``oracle_``-prefixed feature columns, and the
column contract in `schema.py` is what keeps them out of a deployable design matrix.

**Degenerate parameterizations are rejected at construction.** Family B's clipping at
0 means a large slice of its nominal parameter grid piles almost all of its mass on a
single point -- e.g. ``c=0.5, mu_star=0.8, beta=4`` puts 80% of arms at exactly 0 and
has mean 0.033. Those are not hard environments, they are environments where nothing
is learnable, and a labeled dataset generated from them is noise. Rather than trusting
config authors to avoid them, the constructor validates and `valid_tail_grid()`
enumerates only what survives, so config generation cannot request a bad combination.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import special

from cold_start.registry import get_registered, register

# ---- usability thresholds -----------------------------------------------------
#
# A reservoir is "usable" if a bandit run against it can plausibly learn something.
# The thresholds are deliberately loose: they exist to catch collapse, not to police
# difficulty. Thin upper tails (large beta) are the research variable and must pass,
# which is why the tail check is a quantile floor rather than a fixed-epsilon tail
# mass -- ``P(mu >= mu_star - 0.1)`` is ~1e-8 for beta=8 by design, not by accident.

MIN_MEAN = 0.05
MAX_MEAN = 0.95
MAX_BOUNDARY_MASS = 0.25  # P(mu == 0) or P(mu == 1); above this it is a spike, not a tail
MIN_SPREAD = 0.05  # quantile(0.99) - quantile(0.01); arms must be distinguishable
MIN_TOP_QUANTILE = 0.10  # quantile(0.99); the best realistic draw must beat nothing

_QUADRATURE_POINTS = 20_001
_BISECTION_ITERS = 60


class DegenerateReservoirError(ValueError):
    """Raised when a parameter combination yields an environment nothing can learn."""


# ---- base ---------------------------------------------------------------------


class Reservoir(ABC):
    """An infinite pool of arms, each summarized by its true mean ``mu`` in [0, 1].

    Subclasses implement two vectorized hooks -- `_icdf` (the inverse CDF) and
    `_survival` (``P(mu > x)``) -- and everything else in the public interface is
    derived from them. Deriving `sample` from `sample_from_uniforms` is what
    guarantees the paired-rollout coupling holds for *every* family rather than for
    whichever families happened to be written carefully.
    """

    family: str = "reservoir"

    # -- hooks --------------------------------------------------------------

    @abstractmethod
    def _icdf(self, u: np.ndarray) -> np.ndarray:
        """Vectorized inverse CDF on [0, 1]; must be non-decreasing in ``u``."""

    @abstractmethod
    def _survival(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``P(mu > x)``, valid for ``x`` outside [0, 1] too."""

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        """Vectorized ``P(mu <= x)``, used only to invert numerically.

        Split from `_survival` purely for speed: on this build
        ``scipy.special.betainc`` runs a measured 175x faster than its complement
        ``betaincc`` (8.5 ms vs 1.49 s on 2e5 points), and numeric inversion calls
        this once per bisection step per component. Cancellation in ``1 - sf`` is
        harmless here because the result is only ever compared against a uniform.
        """
        return 1.0 - self._survival(x)

    @abstractmethod
    def essential_sup(self) -> float:
        """``mu_star``: the smallest ``x`` with ``P(mu > x) == 0``."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Exact parameters, JSON-serializable, recorded with every trajectory."""

    # -- public interface ----------------------------------------------------

    def sample_from_uniforms(self, u: np.ndarray) -> np.ndarray:
        """Map a uniform stream to arm means; the coupling primitive.

        Paired SEARCH/REFINE rollouts feed the *same* uniforms here, so the j-th new
        arm is the same underlying draw in both branches. Being an inverse CDF, the
        map is monotone, which additionally makes the coupling order-preserving.
        """
        arr = np.asarray(u, dtype=float)
        if arr.size and (np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0):
            raise ValueError("sample_from_uniforms expects u in [0, 1]")
        return np.clip(self._icdf(arr), 0.0, 1.0)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` fresh arm means. Routed through the inverse CDF on purpose."""
        if size < 0:
            raise ValueError(f"size must be non-negative; got {size}")
        return self.sample_from_uniforms(rng.random(size))

    def tail_prob(self, x: float) -> float:
        """``P(mu > x)``. ORACLE: exact reservoir knowledge, analysis only."""
        return float(self._survival(np.asarray(float(x))))

    def quantile(self, q: float) -> float:
        """``Q^{-1}(q)``. ORACLE: exact reservoir knowledge, analysis only."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile expects q in [0, 1]; got {q}")
        return float(self.sample_from_uniforms(np.asarray(float(q))))

    def mean(self) -> float:
        """``E[mu]``. Subclasses override with a closed form where one exists."""
        return mean_by_quadrature(self)

    def to_spec(self) -> dict[str, Any]:
        """Round-trippable ``{"type", "params"}`` matching `schema.ReservoirSpec`."""
        return {"type": self.family, "params": self.params()}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        inner = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({inner})"


def mean_by_quadrature(res: Reservoir, n: int = _QUADRATURE_POINTS) -> float:
    """``E[mu] = int_0^1 Q^{-1}(u) du`` by midpoint rule.

    Works for any family, including ones with atoms, because the inverse CDF is
    defined everywhere on [0, 1]. Used by the validator so the usability checks do
    not depend on a family having supplied a closed-form mean.
    """
    u = (np.arange(n, dtype=float) + 0.5) / n
    return float(np.mean(res.sample_from_uniforms(u)))


def validate_reservoir(res: Reservoir) -> None:
    """Reject reservoirs no bandit run could learn anything from.

    Checks, in order: mass piled on a boundary, a mean outside a usable band, a
    collapsed quantile spread, and a top quantile pinned at zero. Any failure raises
    with the measured value, because "Beta(0.1, 20) is invalid" is useless next to
    "mean 0.005 < 0.05".
    """
    failures: list[str] = []

    at_zero = 1.0 - res.tail_prob(0.0)
    at_one = res.tail_prob(1.0 - 1e-12)
    if at_zero > MAX_BOUNDARY_MASS:
        failures.append(f"P(mu == 0) = {at_zero:.4f} > {MAX_BOUNDARY_MASS}")
    if at_one > MAX_BOUNDARY_MASS:
        failures.append(f"P(mu == 1) = {at_one:.4f} > {MAX_BOUNDARY_MASS}")

    mu_bar = res.mean()
    if not MIN_MEAN <= mu_bar <= MAX_MEAN:
        failures.append(f"mean = {mu_bar:.4f} outside [{MIN_MEAN}, {MAX_MEAN}]")

    q01, q99 = res.quantile(0.01), res.quantile(0.99)
    if q99 - q01 < MIN_SPREAD:
        failures.append(f"quantile spread = {q99 - q01:.4f} < {MIN_SPREAD}")
    if q99 < MIN_TOP_QUANTILE:
        failures.append(f"quantile(0.99) = {q99:.4f} < {MIN_TOP_QUANTILE}")

    if failures:
        raise DegenerateReservoirError(
            f"{type(res).__name__}({res.params()}) is degenerate: " + "; ".join(failures)
        )


# ---- family A: Beta -----------------------------------------------------------

BETA_PRESETS: dict[str, tuple[float, float]] = {
    "good_common": (5.0, 2.0),  # good arms are the norm; SEARCH should be cheap
    "moderately_rare": (2.0, 5.0),  # good arms exist but you must look
    "high_quality_rare": (1.0, 9.0),  # thin, exponential-ish upper tail
    "mostly_mediocre": (8.0, 8.0),  # everything clusters at 0.5; REFINE regime
    "strongly_skewed": (0.5, 3.0),  # U-ish mass low, occasional excellent draw
    "uniform_control": (1.0, 1.0),  # control: no structure in the tail at all
}


@register("beta", kind="reservoir")
class BetaReservoir(Reservoir):
    """``mu ~ Beta(a, b)``. The natural default: full support on (0, 1), two knobs.

    The named grid spans the regimes the study needs to separate, from "good arms are
    common" to "good arms are vanishingly rare", plus a uniform control.
    """

    family = "beta"

    def __init__(self, a: float, b: float, *, preset: str | None = None, validate: bool = True):
        a, b = float(a), float(b)
        if not (a > 0.0 and b > 0.0):
            raise ValueError(f"Beta shape parameters must be positive; got a={a}, b={b}")
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"Beta shape parameters must be finite; got a={a}, b={b}")
        self.a = a
        self.b = b
        self.preset = preset
        if validate:
            validate_reservoir(self)

    @classmethod
    def from_preset(cls, name: str) -> BetaReservoir:
        if name not in BETA_PRESETS:
            raise KeyError(f"unknown beta preset {name!r}; available={sorted(BETA_PRESETS)}")
        a, b = BETA_PRESETS[name]
        return cls(a, b, preset=name)

    def _icdf(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(special.betaincinv(self.a, self.b, u), dtype=float)

    def _survival(self, x: np.ndarray) -> np.ndarray:
        # I_x(a, b) = 1 - I_{1-x}(b, a), so the upper tail is a *direct* betainc
        # evaluation: exact to ~1e-16 relative (verified against `stats.beta.sf`)
        # with none of `betaincc`'s cost and none of `1 - cdf`'s cancellation.
        return np.asarray(special.betainc(self.b, self.a, np.clip(1.0 - x, 0.0, 1.0)), dtype=float)

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(special.betainc(self.a, self.b, np.clip(x, 0.0, 1.0)), dtype=float)

    def essential_sup(self) -> float:
        return 1.0

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {"a": self.a, "b": self.b}
        if self.preset is not None:
            out["preset"] = self.preset
        return out


def beta_grid() -> dict[str, BetaReservoir]:
    """The named Family A grid, keyed by regime name."""
    return {name: BetaReservoir.from_preset(name) for name in BETA_PRESETS}


# ---- family B: controlled upper tail -------------------------------------------

TAIL_BETA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
TAIL_MU_STAR_GRID: tuple[float, ...] = (0.8, 0.9, 1.0)
TAIL_C_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)


@register("tail", kind="reservoir")
class TailReservoir(Reservoir):
    """``P(mu >= mu_star - eps) = c * eps^beta`` exactly, by construction.

    Family A ties the upper tail to the whole shape of the density. This family
    inverts that: the tail exponent ``beta`` is the *parameter*, so the study can vary
    "how rare are near-optimal arms" independently of everything else. The
    construction is inverse sampling,

        mu = clip(mu_star - (V / c)^(1/beta), 0, 1),   V ~ U(0, 1),

    for which ``P(mu >= mu_star - eps) = P(V <= c * eps^beta) = min(1, c * eps^beta)``.

    At ``c = 1, mu_star = 1`` this is *exactly* ``Beta(1, beta)`` -- since
    ``1 - V^(1/beta)`` has CDF ``1 - (1-x)^beta`` -- which is a free cross-check
    between Families A and B and is asserted in the test suite.

    The clip at 0 is where the family can collapse: whenever ``c * mu_star^beta`` is
    small, a large fraction of draws land below zero and pile into an atom at 0. The
    constructor rejects those; `valid_tail_grid()` enumerates what survives.
    """

    family = "tail"

    def __init__(self, beta: float, mu_star: float, c: float, *, validate: bool = True):
        beta, mu_star, c = float(beta), float(mu_star), float(c)
        if not beta > 0.0:
            raise ValueError(f"beta must be positive; got {beta}")
        if not 0.0 < mu_star <= 1.0:
            raise ValueError(f"mu_star must lie in (0, 1]; got {mu_star}")
        if not c > 0.0:
            raise ValueError(f"c must be positive; got {c}")
        self.beta = beta
        self.mu_star = mu_star
        self.c = c
        if validate:
            validate_reservoir(self)

    def _icdf(self, u: np.ndarray) -> np.ndarray:
        v = np.clip(1.0 - np.asarray(u, dtype=float), 0.0, 1.0)
        return self.mu_star - np.power(v / self.c, 1.0 / self.beta)

    def _survival(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        eps = self.mu_star - x
        tail = np.minimum(1.0, self.c * np.power(np.maximum(eps, 0.0), self.beta))
        # Below 0 every draw qualifies, including the atom the clip creates at 0.
        return np.where(x < 0.0, 1.0, np.where(eps <= 0.0, 0.0, tail))

    def essential_sup(self) -> float:
        return self.mu_star

    def boundary_mass(self) -> float:
        """``P(mu == 0)``: the mass the clip pushes onto the lower boundary."""
        return float(max(0.0, 1.0 - min(1.0, self.c * self.mu_star**self.beta)))

    def mean(self) -> float:
        # E[mu] = int_0^{V0} (mu_star - (v/c)^(1/beta)) dv with V0 = min(1, c*mu_star^beta).
        v0 = min(1.0, self.c * self.mu_star**self.beta)
        p = 1.0 / self.beta
        return float(self.mu_star * v0 - self.c ** (-p) * v0 ** (1.0 + p) / (1.0 + p))

    def params(self) -> dict[str, Any]:
        return {"beta": self.beta, "mu_star": self.mu_star, "c": self.c}


@lru_cache(maxsize=1)
def _valid_tail_grid() -> tuple[dict[str, float], ...]:
    keep: list[dict[str, float]] = []
    for beta in TAIL_BETA_GRID:
        for mu_star in TAIL_MU_STAR_GRID:
            for c in TAIL_C_GRID:
                try:
                    TailReservoir(beta=beta, mu_star=mu_star, c=c)
                except DegenerateReservoirError:
                    continue
                keep.append({"beta": beta, "mu_star": mu_star, "c": c})
    return tuple(keep)


def valid_tail_grid() -> list[dict[str, float]]:
    """The Family B parameter combinations that are actually usable.

    Config generation must draw from this rather than from the raw cross product:
    roughly half of ``beta x mu_star x c`` collapses under the clip at 0. The list is
    derived by *constructing* each combination, so it can never drift out of sync with
    the validator.
    """
    return [dict(entry) for entry in _valid_tail_grid()]


def tail_grid() -> dict[str, TailReservoir]:
    """The usable Family B grid, keyed by a readable parameter string."""
    out: dict[str, TailReservoir] = {}
    for entry in _valid_tail_grid():
        key = f"beta{entry['beta']:g}_mustar{entry['mu_star']:g}_c{entry['c']:g}"
        out[key] = TailReservoir(**entry)
    return out


# ---- family C: mixtures --------------------------------------------------------

# Held-out generalization set (plan section D9): these must be constructible without
# reference to the A or B grids, so that holding out "family C" holds out a genuinely
# separate set of environments rather than a relabeling of the training ones.
MIXTURE_PRESETS: dict[str, tuple[tuple[tuple[float, float], ...], tuple[float, ...]]] = {
    # Many mediocre arms with a rare excellent one: the canonical SEARCH-pays case.
    "many_mediocre_rare_excellent": (((6.0, 6.0), (30.0, 3.0)), (0.97, 0.03)),
    # A tight bulk at 0.5 plus a tiny cluster at 0.9: bimodal, sharply separated.
    "bulk_half_tiny_cluster_high": (((50.0, 50.0), (90.0, 10.0)), (0.95, 0.05)),
    # Broad low-quality mass plus a narrow high-quality mode: unequal spreads.
    # Shapes are deliberately disjoint from BETA_PRESETS: Family C is the held-out
    # generalization set, so it must not be a relabeling of a training environment.
    "broad_low_narrow_high": (((2.5, 6.0), (40.0, 8.0)), (0.85, 0.15)),
}


@register("mixture", kind="reservoir")
class MixtureReservoir(Reservoir):
    """A weighted mixture of reservoirs.

    Mixtures are the held-out generalization set because they break the single
    assumption both other families share: that ``Q`` has one mode and a smoothly
    varying tail. If the fitted SEARCH/REFINE boundary collapses here, the study needs
    to know, so these environments must be reachable without touching A or B's grids.

    There is no closed-form inverse CDF for a mixture, so `_icdf` bisects on the
    survival function. That is the whole point of making inverse sampling the
    interface rather than an implementation detail: numeric inversion is monotone and
    deterministic, so the paired-rollout coupling survives intact.
    """

    family = "mixture"

    def __init__(
        self,
        components: Sequence[Reservoir | Mapping[str, Any]],
        weights: Sequence[float],
        *,
        preset: str | None = None,
        validate: bool = True,
    ):
        if len(components) == 0:
            raise ValueError("a mixture needs at least one component")
        if len(components) != len(weights):
            raise ValueError(
                f"components and weights must be the same length; "
                f"got {len(components)} and {len(weights)}"
            )
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0.0) or not np.all(np.isfinite(w)):
            raise ValueError(f"weights must be finite and non-negative; got {list(weights)}")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError("weights must sum to something positive")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1; got {total!r}")

        # Duck-typed rather than isinstance-checked: a component may arrive either as a
        # constructed Reservoir or as the {"type", "params"} spec a config records.
        self.components: list[Reservoir] = [
            comp if hasattr(comp, "sample_from_uniforms") else build_reservoir(comp)
            for comp in components
        ]
        self.weights = (w / total).astype(float)
        self.preset = preset
        if validate:
            validate_reservoir(self)

    @classmethod
    def from_preset(cls, name: str) -> MixtureReservoir:
        if name not in MIXTURE_PRESETS:
            raise KeyError(f"unknown mixture preset {name!r}; available={sorted(MIXTURE_PRESETS)}")
        shapes, weights = MIXTURE_PRESETS[name]
        comps = [BetaReservoir(a, b, validate=False) for a, b in shapes]
        return cls(comps, weights, preset=name)

    def _survival(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros(np.shape(x), dtype=float)
        for w, comp in zip(self.weights, self.components, strict=True):
            out = out + w * comp._survival(x)
        return out

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros(np.shape(x), dtype=float)
        for w, comp in zip(self.weights, self.components, strict=True):
            out = out + w * comp._cdf(x)
        return out

    def _icdf(self, u: np.ndarray) -> np.ndarray:
        """Bisect the mixture CDF to machine precision.

        A mixture has no closed-form inverse, but bisection is monotone in ``u`` and
        fully deterministic, so the paired-rollout coupling survives numeric
        inversion intact. ``_BISECTION_ITERS`` halvings of [0, mu_star] take the
        bracket below float64 resolution, so this is the exact inverse CDF, not an
        approximation to it -- which matters because these environments are the
        held-out generalization set and must not carry their own error term.
        """
        u = np.asarray(u, dtype=float)
        mu_star = self.essential_sup()
        lo = np.zeros(np.shape(u), dtype=float)
        hi = np.full(np.shape(u), mu_star, dtype=float)
        for _ in range(_BISECTION_ITERS):
            mid = 0.5 * (lo + hi)
            below = self._cdf(mid) < u
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out = 0.5 * (lo + hi)
        # The CDF saturates to exactly 1.0 in float64 well below mu_star (at ~0.9993
        # for Beta(6,6)), so bisection at u=1 stops at the first saturated x rather
        # than at the endpoint. Pin it, or `quantile(1)` silently misses mu_star.
        return np.where(u >= 1.0, mu_star, out)

    def essential_sup(self) -> float:
        sups = [
            comp.essential_sup()
            for w, comp in zip(self.weights, self.components, strict=True)
            if w > 0.0
        ]
        return float(max(sups))

    def mean(self) -> float:
        return float(
            sum(w * comp.mean() for w, comp in zip(self.weights, self.components, strict=True))
        )

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "components": [comp.to_spec() for comp in self.components],
            "weights": [float(w) for w in self.weights],
        }
        if self.preset is not None:
            out["preset"] = self.preset
        return out


def mixture_grid() -> dict[str, MixtureReservoir]:
    """The named Family C presets, keyed by preset name."""
    return {name: MixtureReservoir.from_preset(name) for name in MIXTURE_PRESETS}


# ---- construction from config ---------------------------------------------------


def build_reservoir(spec: Mapping[str, Any] | Any) -> Reservoir:
    """Instantiate a reservoir from a `schema.ReservoirSpec` or an equivalent mapping.

    Accepts both ``{"type": "beta", "params": {"preset": "good_common"}}`` and the
    fully-explicit form `params()` round-trips to, so a recorded trajectory can be
    replayed from its own provenance columns.
    """
    type_ = getattr(spec, "type", None)
    if type_ is None:
        type_ = spec["type"]
        params = dict(spec.get("params") or {})
    else:
        params = dict(getattr(spec, "params", None) or {})

    cls = get_registered("reservoir", type_)
    preset = params.get("preset")
    if preset is not None and len(params) == 1:
        return cls.from_preset(preset)
    return cls(**params)


def all_named_reservoirs() -> dict[str, Reservoir]:
    """Every named environment across the three families, for tests and sweeps."""
    out: dict[str, Reservoir] = {}
    for name, res in beta_grid().items():
        out[f"beta:{name}"] = res
    for name, res in tail_grid().items():
        out[f"tail:{name}"] = res
    for name, res in mixture_grid().items():
        out[f"mixture:{name}"] = res
    return out
