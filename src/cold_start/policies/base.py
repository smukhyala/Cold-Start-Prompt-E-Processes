"""Sampling-policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PolicyState:
    """Per-arm counters and latest e-process state, passed to policies each step.

    `log_e` is two-sided evidence against m_0; `log_e_upper` is one-sided
    evidence for p > m_0 (the upward-betting capital K^+). Best-arm UCBs should
    read `log_e_upper`, since two-sided e-values grow for arms that deviate in
    either direction.
    """

    arm_ids: list[str]
    pulls: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    log_e: dict[str, float] = field(default_factory=dict)
    log_e_upper: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for aid in self.arm_ids:
            self.pulls.setdefault(aid, 0)
            self.successes.setdefault(aid, 0)
            self.log_e.setdefault(aid, 0.0)
            self.log_e_upper.setdefault(aid, 0.0)

    def record(
        self,
        arm_id: str,
        reward: float,
        log_e: float,
        log_e_upper: float | None = None,
    ) -> None:
        self.pulls[arm_id] = self.pulls.get(arm_id, 0) + 1
        self.successes[arm_id] = self.successes.get(arm_id, 0) + (1 if reward >= 0.5 else 0)
        self.log_e[arm_id] = log_e
        if log_e_upper is not None:
            self.log_e_upper[arm_id] = log_e_upper


class SamplingPolicy(ABC):
    """Select the next arm at each timestep."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    @abstractmethod
    def next_arm(self, t: int, state: PolicyState) -> str:
        """Return the arm_id to pull at time t (1-indexed)."""

    def select_arm(self, t: int, state: PolicyState) -> str:
        """Compatibility alias for experiment code that uses allocation terminology."""
        return self.next_arm(t, state)

    def update(self, arm_id: str, reward: float, info: dict[str, Any]) -> None:
        """Optional hook; most policies read state via PolicyState directly."""
        return None

    @property
    def scores(self) -> dict[str, float]:
        """Latest arm-score snapshot for logging. Empty by default."""
        return {}
