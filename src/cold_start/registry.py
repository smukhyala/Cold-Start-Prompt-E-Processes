"""Simple string-keyed registry for pluggable components.

Every framework extension point (models, task sources, rewards, e-processes,
policies) uses a registry keyed by short string. Configs reference components
by name; new implementations drop in via the @register decorator without
touching the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, type]] = {}

KINDS = ("model", "task_source", "reward", "eprocess", "policy")


def register(name: str, *, kind: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register `cls` in registry `kind` under key `name`."""
    if kind not in KINDS:
        raise ValueError(f"unknown registry kind {kind!r}; expected one of {KINDS}")

    def _decorator(cls: type[T]) -> type[T]:
        bucket = _REGISTRIES.setdefault(kind, {})
        if name in bucket:
            raise ValueError(f"{kind} {name!r} already registered as {bucket[name]!r}")
        bucket[name] = cls
        return cls

    return _decorator


def get_registered(kind: str, name: str) -> type:
    if kind not in _REGISTRIES or name not in _REGISTRIES[kind]:
        available = sorted(_REGISTRIES.get(kind, {}).keys())
        raise KeyError(f"{kind} {name!r} not registered. available={available}")
    return _REGISTRIES[kind][name]


def list_registered(kind: str) -> list[str]:
    return sorted(_REGISTRIES.get(kind, {}).keys())


def _clear_for_tests() -> None:
    _REGISTRIES.clear()
