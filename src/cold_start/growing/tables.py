"""Precomputed confidence-sequence lookup tables, memory-mapped and cached.

This is the single decision that makes the growing-bandits study affordable. Because
the mixture confidence sequence in `cold_start.growing.evidence` depends on an arm's
history only through the sufficient statistic `(n, S)`, every bound an entire rollout
will ever need can be computed once, up front, for all ``0 <= S <= n <= T``. Inside
the simulator a confidence bound then costs one flat fancy-index gather -- no SciPy
call, no per-arm object, no state -- about 2.9 us for the M touched entries regardless
of how many arms are live.

Three implementation choices, each load-bearing:

* **Flat float32 arrays with an explicit stride.** Indexing a 2-D array with two index
  arrays is 3-4x slower than one flat gather, so bounds are addressed as
  ``flat[n * stride + S]``. That is exactly the `BoundTable` protocol in
  `cold_start.growing.state`, so a table drops straight into `GrowingState.pull`.
  At T=1000 each side is 4 MB.
* **`.npy` on disk, loaded with `mmap_mode="r"`.** Worker processes share one set of
  pages through the OS page cache at zero copy. Passing a table as a task argument
  instead would cost ~3.8 ms of pickling per task (380 s over 100k tasks) *and* give
  every worker a private copy.
* **Atomic writes.** Build to `<name>.npy.tmp` and `os.replace()`. A crash mid-write
  otherwise leaves a truncated `.npy` that fails to load for every later run, and the
  failure surfaces far from its cause.

Invalid cells (``S > n``) are `nan`, not a plausible number: they are unreachable, and
a `nan` bound makes a mis-index obvious instead of silently sane.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cold_start.growing.evidence import cs_grid, cs_lower_upper_for_n
from cold_start.growing.schema import EvidenceSpec

__all__ = ["DEFAULT_CACHE_DIR", "CSTable"]

# Gitignored alongside the rest of the generated datasets (see .gitignore: `data/`).
DEFAULT_CACHE_DIR = Path("data/cs_tables")

INVALID = np.float32(np.nan)


def _cache_stem(
    horizon: int, alpha: float, prior_a: float, prior_b: float, n_intervals: int
) -> str:
    """Readable prefix plus a digest, so two near-identical configs cannot collide."""
    payload = f"{horizon}|{alpha!r}|{prior_a!r}|{prior_b!r}|{n_intervals}"
    digest = hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()
    return f"cs_T{horizon}_alpha{alpha:g}_prior{prior_a:g}-{prior_b:g}_{digest}"


def _save_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)


@dataclass
class CSTable:
    """Anytime-valid confidence bounds for every `(n, S)`, satisfying `BoundTable`."""

    lower_flat: np.ndarray
    upper_flat: np.ndarray
    stride: int
    horizon: int
    alpha: float
    prior_a: float = 1.0
    prior_b: float = 1.0

    # ---- construction ----------------------------------------------------------

    @classmethod
    def build(
        cls,
        horizon: int,
        alpha: float = 0.05,
        *,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
        n_intervals: int = 512,
    ) -> CSTable:
        """Compute both bounds for all ``0 <= S <= n <= horizon``.

        One `n` at a time, because the cumulative form in
        `evidence.log_e_lower_grid` gives every `S` at that `n` for the price of one
        pass -- so the Python loop runs `horizon + 1` times, never once per `(n, S)`.
        The upper bound is *derived* from the lower one by the exact reflection
        ``U(n, S) = 1 - L(n, n-S)`` rather than recomputed, which halves the build and
        makes the two sides consistent by construction.

        Deterministic: no RNG anywhere, so a rebuild is bit-identical.
        """
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0; got {horizon}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1); got {alpha}")
        if prior_a != 1.0 or prior_b != 1.0:
            raise NotImplementedError(
                "CSTable currently builds the uniform-prior table only; it is the "
                "numerically exact path (see evidence.py). Use evidence.cs_bounds "
                "for other priors."
            )
        stride = horizon + 1
        threshold = math.log(1.0 / alpha)
        grid = cs_grid(n_intervals)
        lower = np.full(stride * stride, INVALID, dtype=np.float32)
        upper = np.full(stride * stride, INVALID, dtype=np.float32)
        for n in range(stride):
            lo, hi = cs_lower_upper_for_n(n, threshold, grid)
            base = n * stride
            lower[base : base + n + 1] = lo.astype(np.float32)
            upper[base : base + n + 1] = hi.astype(np.float32)
        table = cls(lower, upper, stride, horizon, alpha, prior_a, prior_b)
        table.validate()
        return table

    @classmethod
    def load_or_build(
        cls,
        horizon: int,
        alpha: float = 0.05,
        *,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
        n_intervals: int = 512,
        cache_dir: Path | str | None = None,
    ) -> CSTable:
        """Load the cached table for these parameters, building and caching if absent.

        The returned arrays are memory-mapped and therefore **read-only**; that is
        deliberate, since several worker processes share the same pages.
        """
        cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        stem = _cache_stem(horizon, alpha, prior_a, prior_b, n_intervals)
        lo_path = cache / f"{stem}_lower.npy"
        hi_path = cache / f"{stem}_upper.npy"
        stride = horizon + 1

        if lo_path.exists() and hi_path.exists():
            lower = np.load(lo_path, mmap_mode="r")
            upper = np.load(hi_path, mmap_mode="r")
            if lower.shape == (stride * stride,) and upper.shape == lower.shape:
                return cls(lower, upper, stride, horizon, alpha, prior_a, prior_b)
            # A stale or truncated cache entry is rebuilt rather than trusted.

        table = cls.build(
            horizon, alpha, prior_a=prior_a, prior_b=prior_b, n_intervals=n_intervals
        )
        cache.mkdir(parents=True, exist_ok=True)
        _save_atomic(lo_path, table.lower_flat)
        _save_atomic(hi_path, table.upper_flat)
        return cls(
            np.load(lo_path, mmap_mode="r"),
            np.load(hi_path, mmap_mode="r"),
            stride,
            horizon,
            alpha,
            prior_a,
            prior_b,
        )

    @classmethod
    def from_spec(
        cls, spec: EvidenceSpec, horizon: int, *, cache_dir: Path | str | None = None
    ) -> CSTable:
        return cls.load_or_build(
            horizon,
            spec.alpha,
            prior_a=spec.prior_a,
            prior_b=spec.prior_b,
            cache_dir=cache_dir,
        )

    # ---- use -------------------------------------------------------------------

    def bounds(self, n: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Confidence bounds for each `(n, S)`. One flat gather per side."""
        idx = np.asarray(n, dtype=np.int64) * self.stride + np.asarray(S, dtype=np.int64)
        return self.lower_flat[idx], self.upper_flat[idx]

    def width(self, n: np.ndarray, S: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds(n, S)
        return hi - lo

    def valid_mask(self) -> np.ndarray:
        """Boolean over the flat layout: True exactly where ``S <= n <= horizon``."""
        n_idx, s_idx = np.divmod(np.arange(self.lower_flat.size), self.stride)
        return s_idx <= n_idx

    def validate(self) -> None:
        """Every reachable cell must be a finite, ordered bound inside [0, 1].

        Kept as a method rather than only a test because the historical failure mode
        here is *silent*: a saturated special function returns `-inf`, the null is
        never rejected, the endpoint collapses to 0 or 1, and the table still looks
        like a table.
        """
        mask = self.valid_mask()
        lo = np.asarray(self.lower_flat)[mask]
        hi = np.asarray(self.upper_flat)[mask]
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise FloatingPointError("CSTable contains non-finite bounds at valid (n, S)")
        if lo.min() < 0.0 or hi.max() > 1.0:
            raise ValueError("CSTable bounds escape [0, 1]")
        if np.any(hi < lo):
            raise ValueError("CSTable has upper < lower at some (n, S)")

    def __repr__(self) -> str:
        return (
            f"CSTable(horizon={self.horizon}, alpha={self.alpha}, "
            f"prior=({self.prior_a}, {self.prior_b}), stride={self.stride})"
        )
