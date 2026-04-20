"""WarmStart wrapper: round-robin until every arm has min_pulls, then delegate."""

from __future__ import annotations

import numpy as np

from cold_start.policies.base import PolicyState, SamplingPolicy


class WarmStart(SamplingPolicy):
    def __init__(self, inner: SamplingPolicy, min_pulls_per_arm: int, rng: np.random.Generator):
        super().__init__(rng)
        self.inner = inner
        self.min_pulls = int(min_pulls_per_arm)

    def next_arm(self, t: int, state: PolicyState) -> str:
        under = [aid for aid in state.arm_ids if state.pulls[aid] < self.min_pulls]
        if under:
            idx = (t - 1) % len(under)
            return under[idx]
        return self.inner.next_arm(t, state)

    @property
    def scores(self) -> dict[str, float]:
        return self.inner.scores
