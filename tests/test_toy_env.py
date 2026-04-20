from __future__ import annotations

from cold_start.tasks.toy import ToyEnv
from cold_start.types import Arm, PromptVector


def _arm(arm_id: str) -> Arm:
    v = PromptVector(0, 0, 0, 0, 0, 0)
    return Arm(arm_id=arm_id, name=arm_id, vector=v)


def test_toy_env_reproducible():
    env_a = ToyEnv(arm_probs={"x": 0.7, "y": 0.3})
    env_a.reset(42)
    env_b = ToyEnv(arm_probs={"x": 0.7, "y": 0.3})
    env_b.reset(42)
    a = [env_a.run_arm(_arm("x"), env_a.sample_task(t), None, max_steps=10).success for t in range(1, 21)]
    b = [env_b.run_arm(_arm("x"), env_b.sample_task(t), None, max_steps=10).success for t in range(1, 21)]
    assert a == b


def test_toy_env_respects_probs_roughly():
    env = ToyEnv(arm_probs={"hi": 0.9, "lo": 0.1})
    env.reset(0)
    hi_wins = sum(env.run_arm(_arm("hi"), env.sample_task(t), None, max_steps=5).success for t in range(1, 501))
    lo_wins = sum(env.run_arm(_arm("lo"), env.sample_task(t), None, max_steps=5).success for t in range(1, 501))
    assert hi_wins > 400 and lo_wins < 100
