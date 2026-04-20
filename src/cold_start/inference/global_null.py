"""Combine per-arm e-processes into a global e-process for the intersection null.

H_0^global: every arm satisfies its own null.
H_1^global: at least one arm violates its null.

Two combination rules:

- "product": E^global_t = prod_k E^k_{n_k(t)}. Valid e-process under the
  intersection null when each arm is a separate filtration and arms are
  chosen in a predictable (adapted) way.

- "mixture": E^global_t = (1/K) * sum_k E^k_{n_k(t)}. Convex combination of
  e-values stays an e-value.

Both are valid — the product form grows faster when many arms deviate;
the mixture form is more robust when only a few arms deviate.
"""

from __future__ import annotations

import math
from typing import Iterable

from cold_start.inference.base import EProcess
from cold_start.registry import register

_LOG_HALF = math.log(0.5)


@register("global_null", kind="eprocess")
class GlobalNullEProcess:
    def __init__(self, combine: str = "product"):
        if combine not in ("product", "mixture"):
            raise ValueError(f"combine must be 'product' or 'mixture'; got {combine!r}")
        self.combine = combine

    def log_e(self, per_arm: Iterable[EProcess]) -> float:
        """Compute the global log-e-value from current per-arm e-processes."""
        logs = [p.log_e for p in per_arm]
        if not logs:
            return 0.0
        if self.combine == "product":
            return sum(logs)
        # mixture: log((1/K) sum exp(log_e_k))
        m = max(logs)
        return m + math.log(sum(math.exp(v - m) for v in logs)) - math.log(len(logs))
