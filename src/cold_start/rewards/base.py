"""Reward function interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from cold_start.types import RunResult, Task


class RewardFn(ABC):
    @abstractmethod
    def __call__(self, task: Task, result: RunResult) -> float:
        """Return a reward in [0,1]."""
