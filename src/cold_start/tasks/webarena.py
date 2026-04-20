"""WebArena-Infinity adapter — stub.

Integration path (see docs/webarena_integration.md when written):
  - Install WebArena-Infinity per upstream instructions (Docker stack).
  - Implement sample_task(t) against its streaming task generator.
  - Implement run_arm by handing the agent loop browser tools and translating
    tool calls to the env's step API.
  - Use WebArena-Infinity's built-in evaluator for the binary success signal.
"""

from __future__ import annotations

from cold_start.registry import register
from cold_start.tasks.base import AgentRunner, EnvironmentAdapter
from cold_start.types import Arm, RunResult, Task


@register("webarena", kind="task_source")
class WebArenaInfinityAdapter(EnvironmentAdapter):
    def __init__(self, **kwargs: object):
        self._kwargs = kwargs

    def reset(self, seed: int) -> None:
        return None

    def sample_task(self, t: int) -> Task:
        raise NotImplementedError(
            "wire WebArena-Infinity per docs/webarena_integration.md"
        )

    def run_arm(
        self,
        arm: Arm,
        task: Task,
        runner: AgentRunner,
        max_steps: int,
    ) -> RunResult:
        raise NotImplementedError(
            "wire WebArena-Infinity per docs/webarena_integration.md"
        )
