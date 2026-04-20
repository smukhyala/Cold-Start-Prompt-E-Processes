"""Thompson sampling with Beta(1,1) prior per arm for Bernoulli rewards."""

from __future__ import annotations

import numpy as np

from cold_start.policies.base import PolicyState, SamplingPolicy
from cold_start.registry import register


@register("thompson", kind="policy")
class ThompsonPolicy(SamplingPolicy):
    def __init__(self, rng: np.random.Generator, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(rng)
        self.alpha0 = alpha
        self.beta0 = beta
        self._last_scores: dict[str, float] = {}

    def next_arm(self, t: int, state: PolicyState) -> str:
        samples: dict[str, float] = {}
        for aid in state.arm_ids:
            s = state.successes[aid]
            f = state.pulls[aid] - s
            samples[aid] = float(self.rng.beta(self.alpha0 + s, self.beta0 + f))
        self._last_scores = samples
        best = max(samples.values())
        ties = [aid for aid, v in samples.items() if v == best]
        return str(self.rng.choice(ties))

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._last_scores)
