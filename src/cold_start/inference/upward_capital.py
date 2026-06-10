"""One-sided upward-betting capital e-process for [0,1]-bounded rewards.

This is the construction stated in Mukhyala & Waudby-Smith (2026), § 3.4:

    E_{a,t} = ∏_{i ≤ t : A_i = a} (1 + λ_{a,i} (Y_i − μ_0))

with predictable betting fractions λ_{a,i} ∈ [0, 1/(1 − μ_0)]. The null is
composite and one-sided:

    H_0 : μ_a ≤ μ_0           (no detectable improvement)
    H_1 : μ_a > μ_0           (this arm beats the baseline)

Under H_0, (E_{a,t})_{t≥0} is a nonnegative supermartingale (Proposition 1 in
the paper). Ville's inequality gives anytime-valid type-I error control
P(sup_t E_t ≥ 1/α | H_0) ≤ α.

This e-process is the natural counterpart of the upward branch K^+ inside
`HedgedCapitalEProcess`. It is provided as a standalone class so configs can
select the paper's exact construction without the two-sided hedging, and so
the global null can mix one-sided per-arm evidence directly.
"""

from __future__ import annotations

import math

from cold_start.inference.base import EProcess
from cold_start.registry import register


@register("upward_capital", kind="eprocess")
class UpwardCapitalEProcess(EProcess):
    def __init__(self, m0: float = 0.5, eps: float = 1e-9):
        if not 0.0 < m0 < 1.0:
            raise ValueError(f"m0 must be in (0,1); got {m0}")
        self.m0 = float(m0)
        self.eps = eps
        # The upward bet λ must satisfy 1 + λ·(Y − μ_0) ≥ 0 for Y ∈ [0,1];
        # the binding case is Y = 0, giving λ ≤ 1/μ_0. aGRAPA further bounds
        # λ by 0.5/μ_0 to keep the per-step capital bounded above by 1.5.
        self._cap = 0.5 / max(m0, eps)

        # log-space capital
        self._log_e = 0.0

        # running statistics for aGRAPA (predictable) bets
        self._sum_x = 0.0
        self._sum_sq_dev = 0.0  # cumulative (X_s − μ̂_s)^2 using online mean
        self._n = 0

        # next-step bet λ_{t+1}, chosen from info through time t (predictable)
        self._lambda = 0.0

    @property
    def n(self) -> int:
        return self._n

    @property
    def log_e(self) -> float:
        return self._log_e

    @property
    def log_e_upper(self) -> float:
        """Alias: this e-process *is* the upward-betting capital."""
        return self._log_e

    def update(self, x: float) -> float:
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"x must be in [0,1]; got {x}")

        lam = self._lambda
        term = 1.0 + lam * (x - self.m0)
        # term ≥ 0 by the cap choice; floor for safety against fp underflow
        self._log_e += math.log(max(term, 1e-300))

        # update running stats (Laplace-smoothed mean keeps σ̂² well-defined
        # before any observations and prevents division by zero in the bet rule)
        self._n += 1
        self._sum_x += x
        mu_hat = (0.5 + self._sum_x) / (1.0 + self._n)
        self._sum_sq_dev += (x - mu_hat) ** 2
        sigma2_hat = (0.25 + self._sum_sq_dev) / (1.0 + self._n)

        # aGRAPA bet: λ_{t+1} = clip((μ̂ − μ_0) / σ̂², 0, cap). Negative bets
        # would correspond to the downward alternative, which this one-sided
        # process does not pursue; clip to 0.
        raw = (mu_hat - self.m0) / max(sigma2_hat, self.eps)
        self._lambda = min(max(raw, 0.0), self._cap)

        return self._log_e
