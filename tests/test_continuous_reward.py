from __future__ import annotations

import numpy as np
import pytest

from cold_start.rewards.continuous import ContinuousReward
from cold_start.tasks.toy import ToyEnv
from cold_start.types import Arm, PromptVector, RunResult, Task


@pytest.fixture
def arm() -> Arm:
    return Arm(arm_id="x", name="X", vector=PromptVector(0, 0, 0, 0, 0, 0))


@pytest.fixture
def task() -> Task:
    return Task(task_id="t1", payload={}, metadata={})


def test_continuous_reward_uses_partial_score_when_present(arm, task):
    result = RunResult(
        success=False, reward=0.0, steps=1, wallclock_s=0.0, partial_score=0.73
    )
    assert ContinuousReward()(task, result) == 0.73


def test_continuous_reward_falls_back_to_binary(arm, task):
    result_success = RunResult(success=True, reward=1.0, steps=1, wallclock_s=0.0)
    result_fail = RunResult(success=False, reward=0.0, steps=1, wallclock_s=0.0)
    rew = ContinuousReward()
    assert rew(task, result_success) == 1.0
    assert rew(task, result_fail) == 0.0


def test_continuous_reward_clamps_out_of_range(arm, task):
    rew = ContinuousReward()
    over = RunResult(success=False, reward=0.0, steps=0, wallclock_s=0.0, partial_score=1.4)
    under = RunResult(success=False, reward=0.0, steps=0, wallclock_s=0.0, partial_score=-0.1)
    assert rew(task, over) == 1.0
    assert rew(task, under) == 0.0


def test_toy_env_emits_partial_score_when_arm_means_configured(arm):
    env = ToyEnv(
        arm_probs={"x": 0.5},
        arm_means={"x": 0.7},
        partial_score_concentration=20.0,
    )
    env.reset(seed=42)
    results = [env.run_arm(arm, env.sample_task(t), runner=None, max_steps=10) for t in range(1, 200)]
    scores = [r.partial_score for r in results]
    assert all(s is not None for s in scores)
    assert all(0.0 <= float(s) <= 1.0 for s in scores)
    # Sample mean should approach the configured mean as concentration grows.
    assert abs(float(np.mean([float(s) for s in scores])) - 0.7) < 0.05


def test_toy_env_partial_score_none_when_arm_means_omitted(arm):
    env = ToyEnv(arm_probs={"x": 0.5})
    env.reset(seed=1)
    result = env.run_arm(arm, env.sample_task(1), runner=None, max_steps=10)
    assert result.partial_score is None


def test_toy_env_rejects_arm_means_at_endpoints():
    with pytest.raises(ValueError):
        ToyEnv(arm_probs={"x": 0.5}, arm_means={"x": 0.0})
    with pytest.raises(ValueError):
        ToyEnv(arm_probs={"x": 0.5}, arm_means={"x": 1.0})
