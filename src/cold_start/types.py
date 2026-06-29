"""Core frozen dataclasses shared across the framework."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PromptVector:
    planning: int
    verification: int
    agency: int
    expertise: int
    format: int
    goal: int

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.planning,
            self.verification,
            self.agency,
            self.expertise,
            self.format,
            self.goal,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "planning": self.planning,
            "verification": self.verification,
            "agency": self.agency,
            "expertise": self.expertise,
            "format": self.format,
            "goal": self.goal,
        }

    def id(self) -> str:
        """Deterministic 8-char id over the vector tuple. Stable across runs."""
        h = hashlib.blake2b(repr(self.as_tuple()).encode(), digest_size=4)
        return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Arm:
    arm_id: str
    name: str
    vector: PromptVector
    prompt_guidance: str = ""


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunResult:
    success: bool
    reward: float
    steps: int
    wallclock_s: float
    trace: dict[str, Any] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    # Optional graded outcome in [0,1] — used by ContinuousReward when present.
    # None means the environment reports only binary success.
    partial_score: float | None = None


@dataclass
class PerArmState:
    pulls: int = 0
    successes: int = 0
    log_e: float = 0.0
    cs_lo: float = 0.0
    cs_hi: float = 1.0

    def to_record(self) -> dict[str, Any]:
        return {
            "pulls": self.pulls,
            "successes": self.successes,
            "log_e": self.log_e,
            "cs_lo": self.cs_lo,
            "cs_hi": self.cs_hi,
        }
