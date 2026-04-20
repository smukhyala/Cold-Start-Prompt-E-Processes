"""Binary reward: 1.0 on success, 0.0 otherwise."""

from __future__ import annotations

from cold_start.registry import register
from cold_start.rewards.base import RewardFn
from cold_start.types import RunResult, Task


@register("binary", kind="reward")
class BinaryReward(RewardFn):
    def __call__(self, task: Task, result: RunResult) -> float:
        return 1.0 if result.success else 0.0
