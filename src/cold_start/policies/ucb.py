"""UCB1 allocation (Auer, Cesa-Bianchi, Fischer 2002).

At each timestep:

    score(a) = μ̂_a + c · sqrt(2 · log(t) / n_a)

— pick argmax. Any arm with `n_a = 0` is pulled first (infinite bonus).
The constant `c` defaults to 1.0 (the textbook UCB1 setting); larger `c`
trades expected reward for exploration.

This is the "vanilla" UCB referenced as ref [2] in Mukhyala & Waudby-Smith
(2026), § 3.5, alongside ε-greedy [14], Thompson [1], and SPRUCE [10].
Unlike SPRUCE, this policy reads only empirical means — it does not consult
the per-arm log-wealth, so it serves as a clean baseline for comparing
mean-based UCB against wealth-based UCB on the same task stream.
"""

from __future__ import annotations

import math

import numpy as np

from cold_start.policies.base import PolicyState, SamplingPolicy
from cold_start.registry import register


@register("ucb", kind="policy")
class UCBPolicy(SamplingPolicy):
    def __init__(self, rng: np.random.Generator, exploration_c: float = 1.0):
        super().__init__(rng)
        if exploration_c < 0.0:
            raise ValueError(f"exploration_c must be ≥ 0; got {exploration_c}")
        self.c = float(exploration_c)
        self._last_scores: dict[str, float] = {}

    def next_arm(self, t: int, state: PolicyState) -> str:
        scores: dict[str, float] = {}
        for aid in state.arm_ids:
            n = state.pulls[aid]
            if n == 0:
                scores[aid] = math.inf
                continue
            mu_hat = state.successes[aid] / n
            bonus = self.c * math.sqrt(2.0 * math.log(max(t, 2)) / n)
            scores[aid] = mu_hat + bonus
        self._last_scores = {
            k: (v if math.isfinite(v) else float("inf")) for k, v in scores.items()
        }
        best = max(scores.values())
        ties = [aid for aid, v in scores.items() if v == best]
        return str(self.rng.choice(ties))

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._last_scores)
