"""Decision helpers on top of e-processes."""

from __future__ import annotations

import math
from typing import Mapping

from cold_start.types import PerArmState


def reject_null(log_e: float, alpha: float) -> bool:
    return log_e >= math.log(1.0 / alpha)


def best_arm_certified(
    states: Mapping[str, PerArmState], alpha: float
) -> str | None:
    """Return the arm whose CS lower bound exceeds all other arms' upper bounds."""
    if len(states) < 2:
        return None
    best_id = None
    for k, sk in states.items():
        # must strictly dominate every other arm
        ok = all(sk.cs_lo > sj.cs_hi for j, sj in states.items() if j != k)
        if ok:
            if best_id is not None:
                return None  # should not happen by construction, but guard
            best_id = k
    return best_id
