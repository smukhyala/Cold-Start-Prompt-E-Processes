"""Distance and utilities over PromptVector."""

from __future__ import annotations

from cold_start.prompts.axes import AXIS_NAMES, AxesSpec
from cold_start.types import PromptVector


def distance(p: PromptVector, q: PromptVector, axes: AxesSpec) -> float:
    """Weighted, range-normalized Manhattan: sum_i w_i * |p_i - q_i| / R_i.

    R_i is the axis max value. Weights come from the axes spec.
    """
    p_d = p.as_dict()
    q_d = q.as_dict()
    total = 0.0
    for name in AXIS_NAMES:
        axis = axes[name]
        r = max(axis.max, 1)  # avoid div-by-zero on degenerate axes
        total += axis.weight * abs(p_d[name] - q_d[name]) / r
    return total


def validate_vector(v: PromptVector, axes: AxesSpec) -> None:
    d = v.as_dict()
    for name, value in d.items():
        axis = axes[name]
        if not 0 <= value <= axis.max:
            raise ValueError(
                f"vector axis {name!r}={value} out of [0, {axis.max}]"
            )
