"""Load the initial arm catalog from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from cold_start.prompts.axes import AxesSpec
from cold_start.prompts.vector import validate_vector
from cold_start.types import Arm, PromptVector


def load_arms(path: str | Path, axes: AxesSpec) -> list[Arm]:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    arms: list[Arm] = []
    for entry in raw["arms"]:
        vec_raw = entry["vector"]
        vec = PromptVector(
            planning=int(vec_raw["planning"]),
            verification=int(vec_raw["verification"]),
            agency=int(vec_raw["agency"]),
            expertise=int(vec_raw["expertise"]),
            format=int(vec_raw["format"]),
            goal=int(vec_raw["goal"]),
        )
        validate_vector(vec, axes)
        arms.append(
            Arm(
                arm_id=entry["arm_id"],
                name=entry.get("name", entry["arm_id"].title()),
                vector=vec,
            )
        )
    if len({a.arm_id for a in arms}) != len(arms):
        raise ValueError("duplicate arm_id in catalog")
    return arms


def generate_arm(arm_id: str, vector: PromptVector, axes: AxesSpec, name: str | None = None) -> Arm:
    validate_vector(vector, axes)
    return Arm(arm_id=arm_id, name=name or arm_id.title(), vector=vector)
