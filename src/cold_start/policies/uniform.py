"""Uniform sampling — round-robin (default) or i.i.d."""

from __future__ import annotations

import numpy as np

from cold_start.policies.base import PolicyState, SamplingPolicy
from cold_start.registry import register


@register("uniform", kind="policy")
class UniformPolicy(SamplingPolicy):
    def __init__(self, rng: np.random.Generator, mode: str = "round_robin"):
        super().__init__(rng)
        if mode not in ("round_robin", "iid"):
            raise ValueError(f"mode must be 'round_robin' or 'iid'; got {mode!r}")
        self.mode = mode

    def next_arm(self, t: int, state: PolicyState) -> str:
        if self.mode == "round_robin":
            idx = (t - 1) % len(state.arm_ids)
            return state.arm_ids[idx]
        return str(self.rng.choice(state.arm_ids))
