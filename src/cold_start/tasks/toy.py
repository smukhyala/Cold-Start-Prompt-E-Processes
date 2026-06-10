"""Synthetic Bernoulli environment. The workhorse for dev/tests/pilot.

Each arm has a configured success probability p_k (Bernoulli draw). When
`arm_means` is also provided, each task additionally emits a graded
`partial_score` from a Beta distribution centered at the configured mean,
so configs can exercise the `ContinuousReward` path on a controlled stream.
No model calls. Reproducible under seed. Steps drawn from a simple
truncated-geometric to give realistic step counts for downstream plotting.
"""

from __future__ import annotations

import numpy as np

from cold_start.registry import register
from cold_start.tasks.base import AgentRunner, EnvironmentAdapter
from cold_start.types import Arm, RunResult, Task


@register("toy", kind="task_source")
class ToyEnv(EnvironmentAdapter):
    def __init__(
        self,
        arm_probs: dict[str, float],
        step_mean: float = 8.0,
        arm_means: dict[str, float] | None = None,
        partial_score_concentration: float = 6.0,
    ):
        if not arm_probs:
            raise ValueError("arm_probs must be non-empty")
        for k, v in arm_probs.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"arm_probs[{k!r}]={v} out of [0,1]")
        self.arm_probs = dict(arm_probs)
        self.step_mean = float(step_mean)

        if arm_means is not None:
            for k, v in arm_means.items():
                if not 0.0 < v < 1.0:
                    raise ValueError(
                        f"arm_means[{k!r}]={v} must be in (0,1) (open) for a valid Beta draw"
                    )
            self.arm_means: dict[str, float] | None = dict(arm_means)
        else:
            self.arm_means = None
        if partial_score_concentration <= 0:
            raise ValueError(
                f"partial_score_concentration must be > 0; got {partial_score_concentration}"
            )
        self._kappa = float(partial_score_concentration)

        self.rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_task(self, t: int) -> Task:
        return Task(task_id=f"toy-{t:06d}", payload={}, metadata={"difficulty": 0.5})

    def run_arm(
        self,
        arm: Arm,
        task: Task,
        runner: AgentRunner,
        max_steps: int,
    ) -> RunResult:
        p = self.arm_probs.get(arm.arm_id)
        if p is None:
            raise KeyError(f"ToyEnv: no success prob configured for arm {arm.arm_id!r}")
        success = bool(self.rng.random() < p)

        # Optional continuous draw: Beta(κ·μ, κ·(1−μ)) has mean μ and grows
        # concentrated as κ increases. Default κ=6 gives modest variance.
        partial_score: float | None = None
        if self.arm_means is not None:
            mu = self.arm_means.get(arm.arm_id)
            if mu is None:
                raise KeyError(
                    f"ToyEnv: arm_means configured but missing arm {arm.arm_id!r}"
                )
            partial_score = float(
                self.rng.beta(self._kappa * mu, self._kappa * (1.0 - mu))
            )

        # step count: truncated geometric, capped at max_steps
        raw_steps = int(self.rng.geometric(1.0 / max(self.step_mean, 1.0)))
        steps = min(raw_steps, max_steps)
        return RunResult(
            success=success,
            reward=float(success),
            steps=steps,
            wallclock_s=0.0,
            trace={"env": "toy"},
            tokens={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            partial_score=partial_score,
        )
