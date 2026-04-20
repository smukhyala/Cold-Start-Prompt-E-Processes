"""Axis specifications loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

AXIS_NAMES = ("planning", "verification", "agency", "expertise", "format", "goal")


@dataclass(frozen=True)
class Axis:
    name: str
    max: int
    weight: float
    labels: tuple[str, ...]

    def label(self, value: int) -> str:
        if not 0 <= value <= self.max:
            raise ValueError(f"axis {self.name!r}: value {value} out of [0,{self.max}]")
        return self.labels[value]


@dataclass(frozen=True)
class AxesSpec:
    axes: dict[str, Axis]

    def __getitem__(self, name: str) -> Axis:
        return self.axes[name]

    def names(self) -> tuple[str, ...]:
        return AXIS_NAMES

    def max_value(self, name: str) -> int:
        return self.axes[name].max

    def weight(self, name: str) -> float:
        return self.axes[name].weight


def load_axes(path: str | Path) -> AxesSpec:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    axes: dict[str, Axis] = {}
    for name in AXIS_NAMES:
        spec = raw["axes"][name]
        axes[name] = Axis(
            name=name,
            max=int(spec["max"]),
            weight=float(spec.get("weight", 1.0)),
            labels=tuple(spec["labels"]),
        )
        if len(axes[name].labels) != axes[name].max + 1:
            raise ValueError(
                f"axis {name!r}: {len(axes[name].labels)} labels for max={axes[name].max}; expected {axes[name].max + 1}"
            )
    return AxesSpec(axes=axes)
