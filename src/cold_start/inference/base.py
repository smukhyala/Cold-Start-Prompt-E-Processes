"""E-process interface.

An e-process E_t against a null P is a nonnegative process with E_0 = 1 and
E[E_tau] <= 1 for any stopping time tau under P. By Ville's inequality,
P(sup_t E_t >= 1/alpha) <= alpha — giving anytime-valid type-I error control.

All concrete e-processes in this framework work in log-space for numerical
stability and expose `log_e` as the primary running statistic.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class EProcess(ABC):
    """Base class for all e-processes."""

    @abstractmethod
    def update(self, x: float) -> float:
        """Absorb observation x in [0,1] and return the updated log-e-value."""

    @property
    @abstractmethod
    def log_e(self) -> float:
        """Current log-e-value."""

    @property
    def e_value(self) -> float:
        return math.exp(self.log_e)

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of observations absorbed so far."""
