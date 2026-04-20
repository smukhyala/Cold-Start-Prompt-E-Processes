"""Environment adapter interface.

An EnvironmentAdapter is responsible for:
  - sampling incoming tasks (a continuous stream, per the WebArena-Infinity model)
  - executing an arm's cold-start prompt on a task via a model client
  - returning a RunResult with success, steps, and tokens used
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from cold_start.types import Arm, RunResult, Task


class AgentRunner(Protocol):
    """Protocol satisfied by a model-client-wrapping callable."""

    def __call__(self, system_prompt: str, task: Task, max_steps: int) -> RunResult: ...


class EnvironmentAdapter(ABC):
    @abstractmethod
    def reset(self, seed: int) -> None: ...

    @abstractmethod
    def sample_task(self, t: int) -> Task: ...

    @abstractmethod
    def run_arm(
        self,
        arm: Arm,
        task: Task,
        runner: AgentRunner,
        max_steps: int,
    ) -> RunResult: ...

    def close(self) -> None:  # optional
        return None
