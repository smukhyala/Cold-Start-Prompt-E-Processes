"""Combine per-arm e-processes into a global e-process for the intersection null.

H_0^global: every arm satisfies its own null.
H_1^global: at least one arm violates its null.

Three combination rules:

- "product": E^global_t = prod_k E^k_{n_k(t)}. Valid e-process under the
  intersection null when each arm is a separate filtration and arms are
  chosen in a predictable (adapted) way.

- "mixture": E^global_t = (1/K) * sum_k E^k_{n_k(t)}. Uniform convex
  combination of e-values stays an e-value.

- "linear_mixture": E^global_t = sum_k w_k * E^k_{n_k(t)} for user-supplied
  weights w_k ≥ 0 summing to 1. This is the construction stated in
  Mukhyala & Waudby-Smith (2026), § 3.4, Proposition 2: any convex
  combination of e-processes is itself an e-process, since
  E[E_τ] = Σ_k w_k · E[E^k_τ] ≤ Σ_k w_k = 1 under the intersection null.
  Uniform weights recover "mixture"; unequal weights let the analyst bake
  in a prior over which arms are likely to be informative.

All three are valid: product grows faster when many arms deviate; mixture
forms are robust when only a few arms deviate.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from cold_start.inference.base import EProcess
from cold_start.registry import register


@register("global_null", kind="eprocess")
class GlobalNullEProcess:
    def __init__(
        self,
        combine: str = "product",
        weights: Mapping[str, float] | None = None,
    ):
        if combine not in ("product", "mixture", "linear_mixture"):
            raise ValueError(
                "combine must be 'product', 'mixture', or 'linear_mixture'; "
                f"got {combine!r}"
            )
        self.combine = combine
        if weights is not None:
            if combine != "linear_mixture":
                raise ValueError(
                    f"weights are only meaningful with combine='linear_mixture'; "
                    f"got combine={combine!r}"
                )
            self.weights = self._validate_weights(weights)
        else:
            self.weights = None

    @staticmethod
    def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
        if not weights:
            raise ValueError("weights must be a non-empty mapping arm_id → w_a")
        out: dict[str, float] = {}
        total = 0.0
        for k, v in weights.items():
            w = float(v)
            if w < 0.0:
                raise ValueError(f"weights[{k!r}]={w} is negative; w_a ≥ 0 required")
            out[k] = w
            total += w
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"weights must sum to 1; got {total}")
        return out

    def log_e(
        self,
        per_arm: Iterable[EProcess] | Mapping[str, EProcess],
    ) -> float:
        """Compute the global log-e-value from current per-arm e-processes.

        For `combine='linear_mixture'`, `per_arm` must be a mapping arm_id → e-process
        so weights can be matched. For other modes, an iterable is sufficient.
        """
        if self.combine == "linear_mixture":
            if not isinstance(per_arm, Mapping):
                raise TypeError(
                    "linear_mixture requires a mapping arm_id → EProcess (not an iterable)"
                )
            items = list(per_arm.items())
            if not items:
                return 0.0
            weights = self.weights or {aid: 1.0 / len(items) for aid, _ in items}
            # E_t = Σ_k w_k E^k_t computed in log-space:
            #   log E_t = log Σ_k exp(log w_k + log E^k_t)
            terms = []
            for aid, p in items:
                w = weights.get(aid)
                if w is None:
                    raise KeyError(
                        f"no weight configured for arm {aid!r}; "
                        f"linear_mixture weights cover {sorted(weights)}"
                    )
                if w == 0.0:
                    continue
                terms.append(math.log(w) + p.log_e)
            if not terms:
                return -math.inf  # all-zero weights → degenerate but well-defined
            m = max(terms)
            return m + math.log(sum(math.exp(v - m) for v in terms))

        # product / uniform mixture: take logs from an iterable
        if isinstance(per_arm, Mapping):
            logs = [p.log_e for p in per_arm.values()]
        else:
            logs = [p.log_e for p in per_arm]
        if not logs:
            return 0.0
        if self.combine == "product":
            return sum(logs)
        # uniform mixture: log((1/K) sum exp(log_e_k))
        m = max(logs)
        return m + math.log(sum(math.exp(v - m) for v in logs)) - math.log(len(logs))
