"""Pydantic models for YAML experiment configs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ComponentSpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class InferenceSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    per_arm: ComponentSpec
    global_: ComponentSpec = Field(alias="global")
    confidence_sequence: dict[str, Any] = Field(
        default_factory=lambda: {"alpha": 0.05, "grid_size": 64}
    )


class PolicySpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    warmstart: dict[str, Any] | None = None


class PromptsSpec(BaseModel):
    axes: str
    template: str
    arms: str


class TrialSpec(BaseModel):
    name: str
    seed: int = 42
    num_tasks: int = 100
    num_trials: int = 1
    paired_streams: bool = True
    max_agent_steps: int = 30
    output_dir: str = "logs/"


class ModelSpec(BaseModel):
    type: str = "anthropic"
    id: str = "claude-opus-4-7"
    max_tokens: int = 1024
    temperature: float | None = None
    prompt_cache: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class LoggingSpec(BaseModel):
    level: str = "INFO"
    jsonl: bool = True


class RunConfig(BaseModel):
    trial: TrialSpec
    prompts: PromptsSpec
    model: ModelSpec
    task_source: ComponentSpec
    reward: ComponentSpec = ComponentSpec(type="binary")
    inference: InferenceSpec
    policy: PolicySpec
    logging: LoggingSpec = LoggingSpec()

    def hash(self) -> str:
        payload = json.dumps(self.model_dump(by_alias=True), sort_keys=True, default=str)
        return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()


def load_config(path: str | Path) -> RunConfig:
    p = Path(path)
    with p.open() as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
