"""Continuous reward in [0, 1].

The paper (§ 3.1) admits both binary success Y_t ∈ {0, 1} and continuous
quality Y_t ∈ [0, 1]. When the environment populates `RunResult.partial_score`
(e.g. fraction of subgoals completed, partial-credit verifier output), this
reward returns it directly, clamped to [0, 1]. Otherwise it falls back to
binary success so configs can swap reward functions without breaking envs
that only emit pass/fail.
"""

from __future__ import annotations

from cold_start.registry import register
from cold_start.rewards.base import RewardFn
from cold_start.types import RunResult, Task


@register("continuous", kind="reward")
class ContinuousReward(RewardFn):
    def __call__(self, task: Task, result: RunResult) -> float:
        if result.partial_score is not None:
            return max(0.0, min(1.0, float(result.partial_score)))
        return 1.0 if result.success else 0.0
