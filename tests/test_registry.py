from __future__ import annotations

import pytest

from cold_start.registry import get_registered, list_registered, register


def test_core_components_registered():
    assert "hedged_capital" in list_registered("eprocess")
    assert "global_null" in list_registered("eprocess")
    assert "toy" in list_registered("task_source")
    assert "webarena" in list_registered("task_source")
    assert "binary" in list_registered("reward")
    assert {"uniform", "epsilon_greedy", "thompson", "spruce"} <= set(
        list_registered("policy")
    )
    assert "mock" in list_registered("model")


def test_get_registered_unknown_raises():
    with pytest.raises(KeyError):
        get_registered("eprocess", "does_not_exist")


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        @register("foo", kind="bogus")  # type: ignore[arg-type]
        class _Foo:
            pass
