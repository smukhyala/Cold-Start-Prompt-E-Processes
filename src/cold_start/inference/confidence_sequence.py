"""Confidence sequence by grid-inverting the hedged-capital e-process.

For each candidate null mean m in a grid over [0,1], we maintain a parallel
HedgedCapitalEProcess(m0=m). The (1-alpha) CS at time t is the set of m whose
e-value is below 1/alpha — i.e., candidates not yet rejected.
"""

from __future__ import annotations

import math

from cold_start.inference.hedged_capital import HedgedCapitalEProcess


class ConfidenceSequence:
    def __init__(self, alpha: float = 0.05, grid_size: int = 64):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1); got {alpha}")
        self.alpha = alpha
        self.grid_size = grid_size
        self.log_threshold = math.log(1.0 / alpha)

        # avoid endpoints 0 and 1 where hedged-capital degenerates
        step = 1.0 / (grid_size + 1)
        self.grid = [step * (i + 1) for i in range(grid_size)]
        self._processes = [HedgedCapitalEProcess(m0=m) for m in self.grid]

    def update(self, x: float) -> tuple[float, float]:
        for p in self._processes:
            p.update(x)
        return self.bounds()

    def bounds(self) -> tuple[float, float]:
        survivors = [m for m, p in zip(self.grid, self._processes) if p.log_e < self.log_threshold]
        if not survivors:
            # no candidate survives — degenerate; return the finest-grid midpoint bracket
            return (min(self.grid), max(self.grid))
        return (min(survivors), max(survivors))
