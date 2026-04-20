"""Epsilon-greedy on empirical means."""

from __future__ import annotations

import numpy as np

from cold_start.policies.base import PolicyState, SamplingPolicy
from cold_start.registry import register


@register("epsilon_greedy", kind="policy")
class EpsilonGreedyPolicy(SamplingPolicy):
    def __init__(self, rng: np.random.Generator, epsilon: float = 0.2):
        super().__init__(rng)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0,1]; got {epsilon}")
        self.epsilon = epsilon
        self._last_scores: dict[str, float] = {}

    def next_arm(self, t: int, state: PolicyState) -> str:
        means = {aid: (state.successes[aid] / state.pulls[aid]) if state.pulls[aid] else 0.0
                 for aid in state.arm_ids}
        self._last_scores = means
        if self.rng.random() < self.epsilon:
            return str(self.rng.choice(state.arm_ids))
        best = max(means.values())
        ties = [aid for aid, v in means.items() if v == best]
        return str(self.rng.choice(ties))

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._last_scores)
