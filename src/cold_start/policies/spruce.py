"""SPRUCE: UCB on empirical log-wealth growth rate.

At each t, score each arm with its empirical log-wealth growth rate plus a
sqrt(log t / n_k) exploration bonus, and pull the argmax. This follows the
SPRUCE-style UCB construction: with sublinear portfolio regret (from aGRAPA
bets inside the per-arm hedged e-process) and sublinear allocation regret
(from the UCB exploration bonus), the policy is asymptotically log-optimal —
matching the growth rate an oracle that knew the best arm would achieve.

Reference: the user's notes on "Multi-Armed Sequential Hypothesis Testing by
Betting" (SPRUCE).
"""

from __future__ import annotations

import math

import numpy as np

from cold_start.policies.base import PolicyState, SamplingPolicy
from cold_start.registry import register


@register("spruce", kind="policy")
class SprucePolicy(SamplingPolicy):
    def __init__(
        self,
        rng: np.random.Generator,
        exploration_c: float = 1.0,
        tie_break: str = "random",
    ):
        super().__init__(rng)
        if exploration_c < 0.0:
            raise ValueError(f"exploration_c must be >= 0; got {exploration_c}")
        if tie_break not in ("random", "first"):
            raise ValueError(f"tie_break must be 'random' or 'first'; got {tie_break!r}")
        self.c = float(exploration_c)
        self.tie_break = tie_break
        self._last_scores: dict[str, float] = {}

    def next_arm(self, t: int, state: PolicyState) -> str:
        # Any arm with zero pulls gets a huge bonus and is picked first.
        # Uses the one-sided upward-betting wealth K^+ (log_e_upper), which
        # grows for p_k > m_0 specifically — not for two-sided deviation.
        scores: dict[str, float] = {}
        for aid in state.arm_ids:
            n = state.pulls[aid]
            if n == 0:
                scores[aid] = math.inf
                continue
            growth_rate = state.log_e_upper[aid] / n
            bonus = self.c * math.sqrt(math.log(max(t, 2)) / n)
            scores[aid] = growth_rate + bonus
        self._last_scores = {
            k: (v if math.isfinite(v) else float("inf")) for k, v in scores.items()
        }
        best = max(scores.values())
        ties = [aid for aid, v in scores.items() if v == best]
        if self.tie_break == "first":
            return ties[0]
        return str(self.rng.choice(ties))

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._last_scores)
