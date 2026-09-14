"""Anytime-valid evidence that is a function of the sufficient statistic (n, S).

Why this module exists. The repo already has two e-processes and neither can go in
the growing-bandits hot loop:

* `UpwardCapitalEProcess` bets with aGRAPA, whose value depends on the *order* of the
  0s and 1s. `log E` is therefore not a function of `(n, S)` and cannot be tabulated.
* `ConfidenceSequence` keeps 64 stateful hedged-capital objects per arm, which across
  `(M replicates, K arms)` is an `(M, K, 64)` tensor of Python objects.

Together those are roughly four orders of magnitude too slow for the ~10^10 simulated
pulls this study needs. So we add a second evidence layer that *is* order-invariant,
and keep the existing ones as recorded features and as correctness oracles in tests.

The construction. For the one-sided null ``H0: mu <= m`` we mix the Bernoulli
likelihood ratio over ``theta ~ Beta(a, b)`` **truncated to the alternative region
(m, 1)**::

    E_n(m) = [ INT_m^1 theta^S (1-theta)^(n-S) dQ(theta) / Q((m, 1]) ]
             / [ m^S (1-m)^(n-S) ]

Truncating to the alternative is what buys validity over the *whole* composite null
rather than only at its boundary: for each fixed ``theta > m``,

    E_mu[ dP_theta / dP_m ] = [ (theta/m) mu + ((1-theta)/(1-m)) (1-mu) ]^n,

whose bracket equals 1 at ``mu = m`` and has derivative ``(theta-m)/(m(1-m)) > 0``, so
it is ``<= 1`` for every ``mu <= m``. Mixing preserves that, giving a nonnegative
supermartingale over the entire null and hence Ville's inequality. The mirrored
construction (truncate to ``(0, m)``) tests ``H0: mu >= m``.


Numerics, which decide whether any of this works
------------------------------------------------

Three separate traps, all of which produce *plausible wrong answers* rather than
errors, and all of which are handled here rather than hoped away.

1. **`betainc` saturates.** ``log1p(-betainc(...))`` hits ``log1p(-1.0) = -inf`` from
   n ~ 100 (measured here: 11 bad cells at n=100, 159 at n=500), with ~0.4 nat errors
   among the cells that are still finite. Never used.

2. **Binomial tails underflow near the ends of the m-range.** The textbook form
   ``base + binom.logcdf(S, n+1, m)`` needs a probability of order ``(1-m)^(n+1-S)``,
   which underflows to `-inf` well inside the working range -- at ``n=400, S=40`` the
   first bad grid point is ``m = 0.922``. The true log e-value there is a mild number
   near ``log(b/(b+n-S))``. Because the pairwise statistic below takes a *minimum over
   a whole grid*, one poisoned edge entry turns the answer into `nan`, `nan >= t` is
   `False`, and the test silently never fires -- precisely in the large-n,
   well-separated states where the true evidence is `e^71`. The observable symptom is
   "this construction has no power", which would get a correct construction thrown
   away. So the binomial-tail route is not used either.

3. **Terminating hypergeometric series cancel.** The exact-cancellation form
   ``log E = J(a+S, b+n-S, m) - J(a, b, m) - S log m`` with
   ``J(A,B,m) = -log B + log 2F1(1-A, B; B+1; 1-m)`` removes every ``(1-m)^k`` factor
   analytically and is bounded in ``[(A-1) log m - log B, -log B]``, so it never
   underflows. But for integer priors ``1-A`` is a non-positive integer, the series
   terminates, and its terms alternate: at ``A = B = 501, m = 0.7`` it needs ~130
   digits it does not have.

What is actually used:

* **Uniform prior ``a = b = 1`` (the default, and the production path).** Under it the
  truncated normalizers are exactly binomial tails, and the whole ratio collapses to
  an **all-positive-term log-sum-exp over binomial coefficients**::

      log E^{<=m}(n,S) = betaln(S+1, n-S+1)
                         + LSE_{k=0..S} [ log C(n+1, k) - (S-k) * logit(m) ]

  Every ``(1-m)`` power cancels analytically, every term is positive, and there is no
  cancellation anywhere. Verified against adaptive quadrature to ``1.1e-13`` and
  against the hyp2f1 J-form to ``1.9e-12`` for ``n`` up to 800 across
  ``m in [1e-5, 1-1e-5]`` including the underflow band. It also reproduces every edge
  case analytically: ``n=0 -> 0``; ``S=0 -> -log(n+1)`` constant in `m`;
  ``m -> 1 -> log(b/(b+n-S))``; ``m -> 0 -> +inf`` when ``S >= 1``.
  Its cumulative structure means a whole grid of `m` costs one
  ``np.logaddexp.accumulate``, which is what makes the CS table affordable.

* **General prior.** A hybrid: ``betaln(A,B) + beta.logsf(m, A, B)`` where the
  survival function has not underflowed, and the J/hyp2f1 form where it has. The two
  fail in *disjoint* regions (`logsf` near `m -> 1`, hyp2f1 near `m -> 0`), so the
  hybrid is accurate wherever either is -- measured against quadrature at ``7.8e-13``
  relative for `n <= 30` over `(a,b)` in `{(1,1), (.5,.5), (2,3)}`. For large `n` with
  a non-uniform prior both can fail together; that is documented, not silently
  tolerated, and is why `a = b = 1` is the supported production path.

**Monotonicity in `m` is NOT a theorem.** The exact derivative is

    d/dm log E^{<=m} = h_{a,b}(m) - h_{a+S,b+n-S}(m) - S/m + (n-S)/(1-m),
    h_{A,B}(m) = pdf_{A,B}(m) / sf_{A,B}(m),

and at ``a=b=10, n=50, S=1, m=0.3`` it is **+13.954** -- an interior, numerically
benign point. No violation is known for ``a=b=1``, but that is an empirical accident
of the flat prior, not a result. So confidence-sequence endpoints are found by
**first-crossing search on a grid plus refinement inside the bracket**, never by naive
bisection: latching onto a later crossing would return `L` too large and *lose*
coverage, whereas a first-crossing search can only widen the interval.

This module depends only on numpy/scipy and `cold_start.registry`, so the subpackage
stays liftable.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.special import betaln, gammaln, hyp2f1, logsumexp
from scipy.stats import beta as _beta_dist

from cold_start.growing.schema import EvidenceSpec
from cold_start.registry import register

__all__ = [
    "LOG_FLOOR",
    "GrowingMixtureEProcess",
    "PairwiseEvidence",
    "PairwiseGridTable",
    "SufficientStatEProcess",
    "cs_bounds",
    "cs_grid",
    "cs_lower_upper_for_n",
    "d_log_e_lower_dm",
    "log_e_lower",
    "log_e_lower_flat_table",
    "log_e_lower_grid",
    "log_e_upper",
    "pairwise_log_e",
]

# exp() underflows to exactly 0 below this, so it is the natural "as good as -inf"
# sentinel: nothing at or under it can ever reach a log(1/alpha) threshold.
LOG_FLOOR = -745.0

# logit(m) is clipped here so that m = 0 and m = 1 evaluate to their analytic limits
# instead of producing `0 * inf = nan` inside the mixture sum. exp(-40) = 4e-18 is
# below float64 resolution, so the clip is invisible at every interior grid point.
LOGIT_CLIP = 40.0

# Below this the scipy survival-function route is in (or near) subnormal territory and
# its log is no longer trustworthy -- measured errors of ~0.2 nats at -744 -- so the
# general-prior path switches to the hyp2f1 form.
_LOGSF_TRUST = -500.0


# ---- small numerical helpers ---------------------------------------------------


def _logit(m: np.ndarray) -> np.ndarray:
    """log(m / (1-m)), clipped so the closed interval [0, 1] stays representable."""
    m = np.asarray(m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(m) - np.log1p(-m)
    return np.clip(np.nan_to_num(out, nan=0.0), -LOGIT_CLIP, LOGIT_CLIP)


def _log_binom_row(n: int) -> np.ndarray:
    """``log C(n+1, k)`` for k = 0 .. n+1, the mixture weights for a horizon-n arm."""
    k = np.arange(n + 2, dtype=np.float64)
    return gammaln(n + 2.0) - gammaln(k + 1.0) - gammaln(n + 2.0 - k)


def _require_finite(x: np.ndarray, what: str) -> np.ndarray:
    """Fail loudly on non-finite evidence rather than letting `nan >= t` say False.

    This is the one-line invariant that catches every numerical trap in this module's
    history: a `nan` log e-value is indistinguishable from "no evidence" at every
    downstream comparison, so it must never be allowed to leave.
    """
    if not np.all(np.isfinite(x)):
        bad = int(np.count_nonzero(~np.isfinite(np.asarray(x))))
        raise FloatingPointError(f"{what}: {bad} non-finite entries")
    return x


# ---- uniform prior: exact, all-positive-term form ------------------------------


def _uniform_lower_batch(n: np.ndarray, S: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``log E^{<=m}(n, S)`` for a broadcast batch, uniform prior.

    Memory is ``batch x (max(S)+1)``; callers with both a large batch and a large `S`
    should use `log_e_lower_flat_table` or `log_e_lower_grid`, which share the
    cumulative structure instead of materialising it per element.
    """
    n, S, m = np.broadcast_arrays(
        np.asarray(n, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(m, dtype=np.float64),
    )
    lg = _logit(m)
    s_max = int(S.max()) if S.size else 0
    k = np.arange(s_max + 1, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        log_c = gammaln(n[..., None] + 2.0) - gammaln(k + 1.0) - gammaln(
            np.maximum(n[..., None] + 2.0 - k, 1.0)
        )
    gap = S[..., None] - k
    terms = np.where(gap >= 0, log_c - gap * lg[..., None], -np.inf)
    out = betaln(S + 1.0, n - S + 1.0) + logsumexp(terms, axis=-1)
    return np.asarray(out)[()]


def log_e_lower_grid(n: int, m_grid: np.ndarray) -> np.ndarray:
    """``log E^{<=m}(n, S)`` for every ``S = 0..n`` and every grid point, uniform prior.

    Returns shape ``(len(m_grid), n+1)``. The whole grid costs one
    ``np.logaddexp.accumulate`` because the mixture sum over `k` is a prefix sum::

        LSE_{k<=S}[ logC(n+1,k) - (S-k) L ] = cumLSE_S[ logC(n+1,k) + k L ] - S L

    which is the single reason a full T=1000 confidence-sequence table takes seconds
    rather than an afternoon.
    """
    lg = _logit(m_grid)
    log_c = _log_binom_row(n)[: n + 1]
    k = np.arange(n + 1, dtype=np.float64)
    terms = log_c[None, :] + k[None, :] * lg[:, None]
    cum = np.logaddexp.accumulate(terms, axis=1)
    s = np.arange(n + 1, dtype=np.float64)
    out = betaln(s + 1.0, n - s + 1.0)[None, :] - s[None, :] * lg[:, None] + cum
    return _require_finite(out, f"log_e_lower_grid(n={n})")


def log_e_lower_flat_table(horizon: int, m: float) -> np.ndarray:
    """``log E^{<=m}(n, S)`` at one fixed `m`, for every ``0 <= S <= n <= horizon``.

    Flat, addressed as ``table[n * (horizon+1) + S]``, matching the `BoundTable`
    layout in `cold_start.growing.state`. Cells with ``S > n`` hold `nan`: they are
    unreachable, and a `nan` makes a mis-index obvious instead of plausible.
    """
    stride = horizon + 1
    lg = float(_logit(np.float64(m)))
    out = np.full(stride * stride, np.nan, dtype=np.float64)
    for n in range(horizon + 1):
        log_c = _log_binom_row(n)[: n + 1]
        k = np.arange(n + 1, dtype=np.float64)
        cum = np.logaddexp.accumulate(log_c + k * lg)
        s = np.arange(n + 1, dtype=np.float64)
        out[n * stride : n * stride + n + 1] = betaln(s + 1.0, n - s + 1.0) - s * lg + cum
    return out


# ---- general prior: hybrid tail integral ---------------------------------------


def _log_tail_integral(A: np.ndarray, B: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``log INT_m^1 t^(A-1) (1-t)^(B-1) dt``, stable across the whole m-range.

    Two routes that fail in disjoint regions, so their combination is reliable
    wherever either is:

    * ``betaln(A,B) + beta.logsf(m, A, B)`` -- exact until the survival function
      underflows, which happens as ``m -> 1``.
    * ``B log(1-m) + J(A,B,m)`` with ``J = -log B + log 2F1(1-A, B; B+1; 1-m)`` --
      the exact-cancellation form, immune to underflow, but the terminating series
      cancels catastrophically as ``m -> 0``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        via_sf = betaln(A, B) + _beta_dist.logsf(m, A, B)
        via_j = B * np.log1p(-m) - np.log(B) + np.log(hyp2f1(1.0 - A, B, B + 1.0, 1.0 - m))
    return np.where(np.isfinite(via_sf) & (via_sf > _LOGSF_TRUST), via_sf, via_j)


def _general_lower_batch(
    n: np.ndarray, S: np.ndarray, m: np.ndarray, prior_a: float, prior_b: float
) -> np.ndarray:
    n, S, m = np.broadcast_arrays(
        np.asarray(n, dtype=np.float64),
        np.asarray(S, dtype=np.float64),
        np.asarray(m, dtype=np.float64),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        # `0 * log(0)` is 0 here, not nan: an exponent of zero contributes nothing.
        pow_m = np.where(S > 0, -S * np.log(m), 0.0)
        pow_1m = np.where(n - S > 0, -(n - S) * np.log1p(-m), 0.0)
    num = _log_tail_integral(prior_a + S, prior_b + n - S, m)
    den = _log_tail_integral(np.float64(prior_a), np.float64(prior_b), m)
    return np.asarray(num - den + pow_m + pow_1m)[()]


# ---- public per-arm evidence ---------------------------------------------------


def _check_prior(prior_a: float, prior_b: float) -> bool:
    if prior_a <= 0.0 or prior_b <= 0.0:
        raise ValueError(f"prior_a, prior_b must be positive; got ({prior_a}, {prior_b})")
    return prior_a == 1.0 and prior_b == 1.0


def log_e_lower(
    n: np.ndarray,
    S: np.ndarray,
    m: np.ndarray,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> np.ndarray:
    """log e-value against ``H0: mu <= m``. Large values are evidence that ``mu > m``.

    Named for the confidence sequence: `L` is the smallest `m` this test fails to
    reject. NOT monotone in `m` in general -- see the module docstring.
    """
    if _check_prior(prior_a, prior_b):
        return _uniform_lower_batch(n, S, m)
    return _general_lower_batch(n, S, m, prior_a, prior_b)


def log_e_upper(
    n: np.ndarray,
    S: np.ndarray,
    m: np.ndarray,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> np.ndarray:
    """log e-value against ``H0: mu >= m``. Large values are evidence that ``mu < m``.

    Obtained by the exact reflection ``E^{>=m}(n, S) = E^{<=1-m}(n, n-S)`` with the
    prior parameters swapped -- substitute ``theta -> 1-theta`` throughout. Sharing
    one implementation this way means the two sides cannot drift apart.
    """
    n_arr = np.asarray(n)
    S_arr = np.asarray(S)
    m_arr = np.asarray(m, dtype=np.float64)
    return log_e_lower(n_arr, n_arr - S_arr, 1.0 - m_arr, prior_a=prior_b, prior_b=prior_a)


def d_log_e_lower_dm(
    n: np.ndarray,
    S: np.ndarray,
    m: np.ndarray,
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> np.ndarray:
    """Exact ``d/dm log E^{<=m}(n, S)``.

    Exists to make the failure of monotonicity checkable directly rather than by
    Monte Carlo: differentiating the two truncated normalizers gives left-truncation
    hazards ``h_{A,B}(m) = pdf_{A,B}(m) / sf_{A,B}(m)``, and

        d/dm log E = h_{a,b}(m) - h_{a+S,b+n-S}(m) - S/m + (n-S)/(1-m),

    which is strictly positive at ``a=b=10, n=50, S=1, m=0.3``. Any future
    "optimisation" back to monotone bisection has to get past that fact.
    """
    n = np.asarray(n, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    def hazard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.exp(_beta_dist.logpdf(m, a, b) - _beta_dist.logsf(m, a, b))

    return (
        hazard(np.float64(prior_a), np.float64(prior_b))
        - hazard(prior_a + S, prior_b + n - S)
        - S / m
        + (n - S) / (1.0 - m)
    )


# ---- confidence sequence -------------------------------------------------------


def cs_grid(n_intervals: int = 512) -> np.ndarray:
    """``n_intervals + 1`` splitting points on [0, 1], uniform in ``arcsin(sqrt(m))``.

    Uniform-in-`m` spacing wastes resolution in the middle, where the evidence is
    flat, and starves the tails, where `S/n` is extreme and the evidence moves fast.
    The variance-stabilising angle spreads resolution the way the statistic actually
    varies: the discretisation loss of the interval cover below is
    ``~2 n_max (delta_phi)^2`` nats, so ``delta_phi = sqrt(eps / (2T))`` buys a loss of
    `eps`.

    Both endpoints are included on purpose. ``m = 0`` and ``m = 1`` are exactly where
    the constructions degenerate to ``E = a/(a+S)`` and ``E = b/(b+n-S)``, which is
    what makes the interval cover below finite and *exact*.
    """
    if n_intervals < 2:
        raise ValueError(f"n_intervals must be >= 2; got {n_intervals}")
    phi = np.linspace(0.0, 0.5 * math.pi, n_intervals + 1)
    grid = np.sin(phi) ** 2
    # Enforce the exact reflection symmetry grid[G-j] == 1 - grid[j] that the
    # upper/lower mirror relies on; sin/cos rounding otherwise breaks it by ~1e-17.
    grid = 0.5 * (grid + (1.0 - grid[::-1]))
    grid[0], grid[-1] = 0.0, 1.0
    return grid


def _first_crossing(values: np.ndarray, threshold: float, grid: np.ndarray) -> np.ndarray:
    """Smallest grid point whose value is below `threshold`, refined inside its bracket.

    `values` is ``(n_grid, batch)``. Returns the refined endpoint per batch column.

    The first crossing -- not a bisection -- is what keeps this correct without a
    monotonicity theorem. Refinement is a linear interpolation *inside* the bracket
    ``[grid[j-1], grid[j]]``, so the result can never exceed the grid hull `grid[j]`;
    erring toward the wider interval is the safe direction, because a too-large `L`
    silently drops coverage while a too-small one merely costs power.
    """
    retained = values < threshold
    j = np.argmax(retained, axis=0)
    lo = grid[np.maximum(j - 1, 0)]
    hi = grid[j]
    v_lo = np.take_along_axis(values, np.maximum(j - 1, 0)[None, :], axis=0)[0]
    v_hi = np.take_along_axis(values, j[None, :], axis=0)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (v_lo - threshold) / (v_lo - v_hi)
    frac = np.where(np.isfinite(frac), np.clip(frac, 0.0, 1.0), 1.0)
    refined = lo + frac * (hi - lo)
    return np.where(j == 0, 0.0, np.minimum(refined, hi))


def cs_lower_upper_for_n(
    n: int, threshold: float, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """CS endpoints for every ``S = 0..n`` at one `n`, uniform prior.

    The upper endpoint comes free from the reflection ``E^{>=m}(n,S) = E^{<=1-m}(n,n-S)``
    together with the grid's ``grid[G-j] = 1 - grid[j]`` symmetry, so
    ``U(n, S) = 1 - L(n, n-S)`` exactly. Deriving it rather than recomputing it halves
    the table build and makes the two sides consistent by construction.
    """
    values = log_e_lower_grid(n, grid)
    lower = _first_crossing(values, threshold, grid)
    upper = 1.0 - lower[::-1]
    return lower, upper


def cs_bounds(
    n: np.ndarray,
    S: np.ndarray,
    *,
    alpha: float = 0.05,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    n_intervals: int = 512,
    bisect_iters: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """(1-alpha) confidence sequence ``[L, U]`` for each `(n, S)`.

    The reference/general implementation: grid scan for the first crossing, then a
    *bisection inside that bracket only*. This is the slow, obviously-correct path
    used to validate `cold_start.growing.tables.CSTable`; the table's own build uses
    the cumulative grid form and interpolates within the bracket instead.

    Cost is ``batch x n_intervals x max(S)`` memory, so this is for tens of `(n, S)`
    pairs, not for a table.
    """
    if not (prior_a == 1.0 and prior_b == 1.0):
        raise NotImplementedError(
            f"cs_bounds supports only the uniform Beta(1,1) prior; got "
            f"Beta({prior_a}, {prior_b}). The stable log-sum-exp-over-binomial-"
            f"coefficients form the endpoint scan relies on is specific to that "
            f"prior, and the general hybrid path produces NaN near m=1. Production "
            f"uses the uniform prior throughout, so this is a documented limit "
            f"rather than a silent failure."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1); got {alpha}")
    threshold = math.log(1.0 / alpha)
    grid = cs_grid(n_intervals)
    n_arr = np.atleast_1d(np.asarray(n, dtype=np.int64))
    S_arr = np.atleast_1d(np.asarray(S, dtype=np.int64))
    n_arr, S_arr = np.broadcast_arrays(n_arr, S_arr)
    if np.any(S_arr < 0) or np.any(S_arr > n_arr):
        raise ValueError("require 0 <= S <= n")

    def endpoint(nn: np.ndarray, ss: np.ndarray) -> np.ndarray:
        vals = log_e_lower(
            nn[None, :], ss[None, :], grid[:, None], prior_a=prior_a, prior_b=prior_b
        )
        if np.isnan(vals).any():
            raise FloatingPointError("cs_bounds grid scan produced nan log e-values")
        j = np.argmax(vals < threshold, axis=0)
        lo = grid[np.maximum(j - 1, 0)]
        hi = grid[j]
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            v = log_e_lower(nn, ss, mid, prior_a=prior_a, prior_b=prior_b)
            rejected = v >= threshold
            lo = np.where(rejected, mid, lo)
            hi = np.where(rejected, hi, mid)
        return np.where(j == 0, 0.0, hi)

    lower = endpoint(n_arr, S_arr)
    # Mirror: the largest m surviving the upper test is 1 minus the smallest m
    # surviving the lower test at (n, n-S) with the prior parameters swapped.
    upper = 1.0 - _cs_lower_swapped(n_arr, S_arr, threshold, grid, prior_b, prior_a, bisect_iters)
    if np.ndim(n) == 0 and np.ndim(S) == 0:
        return float(lower[0]), float(upper[0])
    return lower, upper


def _cs_lower_swapped(
    n_arr: np.ndarray,
    S_arr: np.ndarray,
    threshold: float,
    grid: np.ndarray,
    prior_a: float,
    prior_b: float,
    bisect_iters: int,
) -> np.ndarray:
    vals = log_e_lower(
        n_arr[None, :], (n_arr - S_arr)[None, :], grid[:, None], prior_a=prior_a, prior_b=prior_b
    )
    j = np.argmax(vals < threshold, axis=0)
    lo = grid[np.maximum(j - 1, 0)]
    hi = grid[j]
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        v = log_e_lower(n_arr, n_arr - S_arr, mid, prior_a=prior_a, prior_b=prior_b)
        rejected = v >= threshold
        lo = np.where(rejected, mid, lo)
        hi = np.where(rejected, hi, mid)
    return np.where(j == 0, 0.0, hi)


# ---- pairwise leader-vs-challenger evidence ------------------------------------


class PairwiseGridTable:
    """Precomputed cover terms for every ``(n, S)`` up to `max_n`.

    Column `j` already holds the *paired* endpoints the cover needs -- the leader
    factor at ``grid[j+1]`` and the challenger factor at ``grid[j]`` -- so a query is
    two gathers, an add and a min. Invalid cells (``S > n``) are ``+inf`` so a
    mis-index can never *lower* a minimum and manufacture evidence.

    Memory is ``2 (max_n+1)^2 G`` float32, which is why this is not the default path:
    at T=1000 with G=512 it would be a gigabyte. Use `PairwiseEvidence.log_e` for
    large horizons.
    """

    def __init__(self, lead_right: np.ndarray, chal_left: np.ndarray, max_n: int):
        self.lead_right = lead_right
        self.chal_left = chal_left
        self.max_n = int(max_n)
        self.stride = int(max_n) + 1

    def _idx(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        return np.asarray(n, dtype=np.int64) * self.stride + np.asarray(S, dtype=np.int64)

    def lead_rows(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Leader cover terms (evaluated at the right endpoint of each interval)."""
        return self.lead_right[self._idx(n, S)]

    def chal_rows(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Challenger cover terms (evaluated at the left endpoint of each interval)."""
        return self.chal_left[self._idx(n, S)]

    def log_e(
        self, n_lead: np.ndarray, S_lead: np.ndarray, n_chal: np.ndarray, S_chal: np.ndarray
    ) -> np.ndarray:
        total = self.lead_rows(n_lead, S_lead) + self.chal_rows(n_chal, S_chal)
        return total.min(axis=-1)


@dataclass
class PairwiseEvidence:
    """e-process for ``H0: mu_lead <= mu_chal``, by an exact finite interval cover.

    A point grid is **not** enough. Writing the null as a union over a splitting
    point and taking ``min_j [ E_lead^{<=m_j} * E_chal^{>=m_j} ]`` over grid *points*
    is anti-conservative -- the grid minimum overshoots the true infimum, and measured
    at ``alpha = 0.10, T = 150`` the point form rejects at 0.0525, above `alpha`.

    Cover with **intervals** instead. For ``0 = m_0 < m_1 < ... < m_G = 1``::

        H0 = UNION_{j=0..G-1} { mu_L <= m_{j+1} } AND { mu_C >= m_j }     (exact, finite)

        log_e_pair = min_j [ log E^{<=m_{j+1}}(n_L, S_L) + log E^{>=m_j}(n_C, S_C) ]

    Note the asymmetry: the **right** endpoint for the leader factor and the **left**
    endpoint for the challenger factor. Reversing them breaks the argument. Exactness:
    for any ``mu_L <= mu_C`` take ``j* = max{j : m_j <= mu_C}``; then ``mu_C >= m_{j*}``
    and ``mu_L <= mu_C < m_{j*+1}``, so the ``j*`` product is a genuine e-process for a
    sub-null containing the truth and the minimum is dominated by it. No slack term, no
    Lipschitz constant, no convexity assumption. The endpoints degenerate gracefully:
    ``E^{>=m_0} = E^{>=0}`` and ``E^{<=m_G} = E^{<=1}`` are both bounded by 1.

    Refining the grid only *raises* power (each interval shrinks toward a point null),
    which is the observable signature that the construction is doing what the theory
    says -- and is asserted in the test suite.

    Remaining caveat, stated rather than assumed away: leader and challenger are
    *selected from the data*, so this is not a valid test for "the selected pair"
    without a selective-inference correction. In this study `log_e_pair` is a
    **feature**, never a formal test.
    """

    n_intervals: int = 512
    prior_a: float = 1.0
    prior_b: float = 1.0
    grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid = cs_grid(self.n_intervals)

    @classmethod
    def from_spec(cls, spec: EvidenceSpec) -> PairwiseEvidence:
        return cls(
            n_intervals=spec.pairwise_grid, prior_a=spec.prior_a, prior_b=spec.prior_b
        )

    def log_e(
        self,
        n_lead: np.ndarray,
        S_lead: np.ndarray,
        n_chal: np.ndarray,
        S_chal: np.ndarray,
    ) -> np.ndarray:
        """Cover minimum for each (leader, challenger) pair in the batch."""
        n_lead, S_lead, n_chal, S_chal = np.broadcast_arrays(
            np.asarray(n_lead), np.asarray(S_lead), np.asarray(n_chal), np.asarray(S_chal)
        )
        right = self.grid[1:]
        left = self.grid[:-1]
        lead = log_e_lower(
            n_lead[..., None], S_lead[..., None], right, prior_a=self.prior_a, prior_b=self.prior_b
        )
        chal = log_e_upper(
            n_chal[..., None], S_chal[..., None], left, prior_a=self.prior_a, prior_b=self.prior_b
        )
        _require_finite(lead, "pairwise leader cover terms")
        _require_finite(chal, "pairwise challenger cover terms")
        return (lead + chal).min(axis=-1)

    def tabulate(self, max_n: int, max_bytes: int = 512 * 1024 * 1024) -> PairwiseGridTable:
        """Precompute the cover terms for every ``(n, S)`` with ``n <= max_n``."""
        if not _check_prior(self.prior_a, self.prior_b):
            raise NotImplementedError("tabulate() supports the uniform prior only")
        stride = max_n + 1
        need = 2 * stride * stride * self.n_intervals * 4
        if need > max_bytes:
            raise ValueError(
                f"tabulate(max_n={max_n}, G={self.n_intervals}) needs {need / 2**20:.0f} MiB "
                f"> max_bytes={max_bytes / 2**20:.0f} MiB; use PairwiseEvidence.log_e instead"
            )
        g = self.grid.size - 1
        lead_right = np.full((stride * stride, g), np.inf, dtype=np.float32)
        chal_left = np.full((stride * stride, g), np.inf, dtype=np.float32)
        for n in range(stride):
            vals = log_e_lower_grid(n, self.grid)  # (G+1, n+1)
            base = n * stride
            lead_right[base : base + n + 1] = vals[1:, :].T.astype(np.float32)
            # E^{>=grid[j]}(n, S) = E^{<=1-grid[j]}(n, n-S) = vals[G-j, n-S].
            chal_left[base : base + n + 1] = vals[:0:-1, ::-1].T.astype(np.float32)
        _require_finite(lead_right[np.isfinite(lead_right)], "pairwise table (leader)")
        _require_finite(chal_left[np.isfinite(chal_left)], "pairwise table (challenger)")
        return PairwiseGridTable(lead_right, chal_left, max_n)


def pairwise_log_e(
    n_lead: np.ndarray,
    S_lead: np.ndarray,
    n_chal: np.ndarray,
    S_chal: np.ndarray,
    *,
    n_intervals: int = 512,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper: log e-value against ``H0: mu_lead <= mu_chal``."""
    ev = PairwiseEvidence(n_intervals=n_intervals, prior_a=prior_a, prior_b=prior_b)
    return ev.log_e(n_lead, S_lead, n_chal, S_chal)


# ---- component interface -------------------------------------------------------


class SufficientStatEProcess(ABC):
    """Sibling of `cold_start.inference.base.EProcess` for order-invariant evidence.

    Deliberately *not* a subclass. `EProcess` is defined around a per-observation
    `update(x)` with order-dependent internal state; an `(n, S)`-indexed construction
    has no such state, and pretending otherwise would invite callers to feed it
    non-binary rewards for which `(n, S)` is not sufficient. The contract here is a
    pure function of the sufficient statistic, which is exactly what makes it
    tabulatable.
    """

    alpha: float

    @property
    def log_threshold(self) -> float:
        return math.log(1.0 / self.alpha)

    @abstractmethod
    def log_e(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        """log e-value from the sufficient statistic."""

    def rejects(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Ville rejection at level `alpha`, valid at any stopping time."""
        return self.log_e(n, S) >= self.log_threshold


@register("growing_mixture", kind="eprocess")
class GrowingMixtureEProcess(SufficientStatEProcess):
    """Truncated-Beta mixture e-process against ``H0: mu <= m0``, indexed by `(n, S)`."""

    def __init__(
        self,
        m0: float = 0.5,
        alpha: float = 0.05,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
    ):
        if not 0.0 < m0 < 1.0:
            raise ValueError(f"m0 must be in (0,1); got {m0}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1); got {alpha}")
        _check_prior(prior_a, prior_b)
        self.m0 = float(m0)
        self.alpha = float(alpha)
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)

    @classmethod
    def from_spec(cls, spec: EvidenceSpec) -> GrowingMixtureEProcess:
        return cls(
            m0=spec.agrapa_m0, alpha=spec.alpha, prior_a=spec.prior_a, prior_b=spec.prior_b
        )

    def log_e(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        return log_e_lower(n, S, self.m0, prior_a=self.prior_a, prior_b=self.prior_b)

    def log_e_downward(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Evidence against the mirrored null ``H0: mu >= m0``."""
        return log_e_upper(n, S, self.m0, prior_a=self.prior_a, prior_b=self.prior_b)

    def bounds(self, n: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(1-alpha) confidence sequence for `mu`, by the reference (slow) path."""
        return cs_bounds(
            n, S, alpha=self.alpha, prior_a=self.prior_a, prior_b=self.prior_b
        )
