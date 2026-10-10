"""The registered step cap must reach browser-use's ``Agent.run(max_steps=...)``.

browser-use's ``Agent(...)`` swallows an unknown ``max_steps`` kwarg and ``run()`` defaults to 500, so a cap
passed only to the constructor is silently ignored (Pre-registration 10 Amendment 1). This test swaps in a fake
``browser_use.Agent`` and checks what the armed agent hands to ``run()``.
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from cold_start.tasks.webarena import _webarena_root


def _integration_available() -> bool:
    try:
        _webarena_root()
    except RuntimeError:
        return False
    return importlib.util.find_spec("browser_use") is not None


class _FakeHistory:
    def __init__(self) -> None:
        self.history = []

    def save_to_file(self, path) -> None:
        return None

    def screenshot_paths(self) -> list:
        return []

    def is_done(self) -> bool:
        return True

    def final_result(self) -> str:
        return "done"

    def errors(self) -> list:
        return []


class _FakeAgent:
    seen_init: dict = {}
    seen_run: dict = {}

    def __init__(self, **kwargs) -> None:
        _FakeAgent.seen_init = kwargs
        self.history = _FakeHistory()

    async def run(self, **kwargs) -> _FakeHistory:
        _FakeAgent.seen_run = kwargs
        return self.history


@pytest.mark.skipif(not _integration_available(), reason="webarena-infinity + browser-use required")
def test_step_cap_is_passed_to_run(monkeypatch, tmp_path):
    import browser_use

    import cold_start.tasks.webarena as wa

    async def no_tokens(agent):
        return {}

    monkeypatch.setattr(browser_use, "Agent", _FakeAgent)
    monkeypatch.setattr(wa, "_collect_token_usage", no_tokens)
    armed_cls = wa._get_armed_agent_cls()
    agent = armed_cls(llm=None, max_steps=30, timeout=60)
    asyncio.run(agent.run("star the email", "http://localhost:0", tmp_path))
    assert _FakeAgent.seen_run.get("max_steps") == 30
    assert "max_steps" not in _FakeAgent.seen_init
