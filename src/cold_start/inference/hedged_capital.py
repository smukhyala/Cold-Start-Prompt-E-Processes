"""Hedged-capital e-process for [0,1]-bounded rewards.

Implements the hedged betting process of Waudby-Smith & Ramdas (JRSS-B 2023)
with aGRAPA (approximate GROW asymptotic power adjustment) bets.

Testing H_0: E[X] = m_0 against the two-sided alternative. Maintains two
capital processes:

    K_t^+ = prod_{s<=t} (1 + lambda_s^+ * (X_s - m_0))
    K_t^- = prod_{s<=t} (1 - lambda_s^- * (X_s - m_0))

with lambda_s^{+/-} in [0, 0.5 / max(m_0, 1-m_0)] predictable. E_t is the
average of the two. All accumulation is done in log-space.
"""

from __future__ import annotations

import math

from cold_start.inference.base import EProcess
from cold_start.registry import register

_LOG_HALF = math.log(0.5)


@register("hedged_capital", kind="eprocess")
class HedgedCapitalEProcess(EProcess):
    def __init__(self, m0: float = 0.5, eps: float = 1e-9):
        if not 0.0 < m0 < 1.0:
            raise ValueError(f"m0 must be in (0,1); got {m0}")
        self.m0 = float(m0)
        self.eps = eps
        self._cap_plus = 0.5 / max(m0, eps)
        self._cap_minus = 0.5 / max(1.0 - m0, eps)

        # log-space capital processes
        self._log_kp = 0.0
        self._log_km = 0.0

        # running statistics for aGRAPA bets
        self._sum_x = 0.0
        self._sum_sq_dev = 0.0  # cumulative (X_s - mu_hat_s)^2 using online mean
        self._n = 0

        # next-step bets, chosen at time t based on info through t-1
        self._lambda_plus = 0.0
        self._lambda_minus = 0.0

    @property
    def n(self) -> int:
        return self._n

    @property
    def log_e(self) -> float:
        # log E_t = log((K^+ + K^-) / 2) = logaddexp(log K^+, log K^-) - log 2
        return _logaddexp(self._log_kp, self._log_km) + _LOG_HALF

    @property
    def log_e_upper(self) -> float:
        """Log of the upward-betting capital process K^+.

        Tests the one-sided alternative p > m_0. This is the quantity SPRUCE-style
        best-arm-identification policies should use — two-sided evidence grows
        for arms that deviate in either direction, which confuses UCB on mean.
        """
        return self._log_kp

    @property
    def log_e_lower(self) -> float:
        """Log of the downward-betting capital process K^-. Tests p < m_0."""
        return self._log_km

    def update(self, x: float) -> float:
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"x must be in [0,1]; got {x}")

        lp, lm = self._lambda_plus, self._lambda_minus
        # multiplicative capital updates in log-space
        term_p = 1.0 + lp * (x - self.m0)
        term_m = 1.0 - lm * (x - self.m0)
        # both terms >= 0 by the cap choice
        self._log_kp += math.log(max(term_p, 1e-300))
        self._log_km += math.log(max(term_m, 1e-300))

        # update running stats using Laplace-smoothed mean (numerically friendly)
        self._n += 1
        self._sum_x += x
        mu_hat = (0.5 + self._sum_x) / (1.0 + self._n)
        # approximate running sum of squared deviations
        self._sum_sq_dev += (x - mu_hat) ** 2
        sigma2_hat = (0.25 + self._sum_sq_dev) / (1.0 + self._n)

        # aGRAPA bet update, clipped to [0, cap]
        raw_plus = (mu_hat - self.m0) / max(sigma2_hat, self.eps)
        raw_minus = (self.m0 - mu_hat) / max(sigma2_hat, self.eps)
        self._lambda_plus = min(max(raw_plus, 0.0), self._cap_plus)
        self._lambda_minus = min(max(raw_minus, 0.0), self._cap_minus)

        return self.log_e


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
