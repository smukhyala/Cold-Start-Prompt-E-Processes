"""Empirical reservoirs: a finite, weighted set of arm means estimated from real prompts.

The parametric families in `reservoirs.py` describe where arm means *might* come from.
This one describes where they *did* come from: a discrete distribution (atoms, weights)
estimated from logged WebArena outcomes by `cold_start.growing.empirical`. Everything the
simulator needs -- inverse-CDF sampling for the CRN coupling, the survival function for
the oracle columns -- is exact for a discrete distribution, so no numeric inversion is
involved and the paired-rollout coupling holds as it does for every other family.

A real pool may be flatter than any environment the corpus admitted (`MIN_SPREAD`), and
that is a finding rather than an error, so validation is *recorded* by default instead
of raised: `validation_error` holds the validator's message, or ``None``.

This module lives outside `reservoirs.py` on purpose: `harness.run_cell` reaches it only
through the registry, so it stays off the fingerprinted simulation surface
(`run_deployment.SIM_SURFACE_MODULES`) and no shipped episode's fingerprint moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cold_start.growing.reservoirs import DegenerateReservoirError, Reservoir, validate_reservoir
from cold_start.registry import register

_WEIGHT_TOL = 1e-6


@register("empirical", kind="reservoir")
class EmpiricalReservoir(Reservoir):
    """``mu`` takes the value ``atoms[k]`` with probability ``weights[k]``."""

    family = "empirical"

    def __init__(
        self,
        atoms: Sequence[float],
        weights: Sequence[float],
        *,
        label: str = "",
        validate: bool = False,
    ):
        a = np.asarray(atoms, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        if a.size == 0:
            raise ValueError("an empirical reservoir needs at least one atom")
        if a.size != w.size:
            raise ValueError(f"atoms and weights must be the same length; got {a.size} and {w.size}")
        if not (np.all(np.isfinite(a)) and np.all((a >= 0.0) & (a <= 1.0))):
            raise ValueError("atoms must be finite and lie in [0, 1]")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        total = float(w.sum())
        if abs(total - 1.0) > _WEIGHT_TOL:
            raise ValueError(f"weights must sum to 1; got {total!r}")

        keep = w > 0.0
        a, w = a[keep], w[keep] / w[keep].sum()
        uniq, inverse = np.unique(a, return_inverse=True)
        merged = np.zeros(uniq.size, dtype=float)
        np.add.at(merged, inverse, w)
        self.atoms = uniq
        self.weights = merged
        self._cum = np.cumsum(merged)
        self._cum[-1] = 1.0
        self._cum0 = np.concatenate([[0.0], self._cum])
        self.label = str(label)

        self.validation_error: str | None = None
        try:
            validate_reservoir(self)
        except DegenerateReservoirError as exc:
            if validate:
                raise
            self.validation_error = str(exc)

    def _icdf(self, u: np.ndarray) -> np.ndarray:
        # inf{x : F(x) >= u}: the first atom whose cumulative weight reaches u.
        idx = np.searchsorted(self._cum, np.asarray(u, dtype=float), side="left")
        return self.atoms[np.minimum(idx, self.atoms.size - 1)]

    def _survival(self, x: np.ndarray) -> np.ndarray:
        n_at_or_below = np.searchsorted(self.atoms, np.asarray(x, dtype=float), side="right")
        return np.clip(1.0 - self._cum0[n_at_or_below], 0.0, 1.0)

    def essential_sup(self) -> float:
        return float(self.atoms[-1])

    def mean(self) -> float:
        return float(np.dot(self.atoms, self.weights))

    def sd(self) -> float:
        m = self.mean()
        return float(np.sqrt(np.dot(self.weights, (self.atoms - m) ** 2)))

    def params(self) -> dict[str, Any]:
        return {
            "atoms": [float(x) for x in self.atoms],
            "weights": [float(x) for x in self.weights],
            "label": self.label,
        }
