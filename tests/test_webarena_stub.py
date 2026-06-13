"""WebArena adapter sanity tests.

`reset()` and `run_arm()` spawn a real server subprocess and launch a Chromium
browser, which is too heavy for pytest. These tests cover everything *up to*
those side effects: construction, task discovery via the sibling repo, and the
import path that wires `webarena-infinity/evaluation/` onto sys.path.

A full integration smoke (server + browser + LLM) is performed manually via
`cold-start-run --config configs/webarena_gmail_warmup.yaml`; see README.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cold_start.tasks.webarena import (
    WebArenaInfinityAdapter,
    _import_webarena,
    _webarena_root,
)


def _sibling_repo_present() -> bool:
    """True iff the webarena-infinity sibling repo (or WEBARENA_INFINITY_ROOT)
    is actually checked out next to this repo."""
    try:
        _webarena_root()
        return True
    except RuntimeError:
        return False


def _browser_use_importable() -> bool:
    try:
        import browser_use  # noqa: F401
        return True
    except Exception:
        return False


def test_adapter_constructs_without_side_effects():
    """Constructing the adapter must not start a server or import browser_use.

    The adapter is meant to be safe to instantiate in CI; only reset() and
    run_arm() are allowed to touch the network or filesystem.
    """
    env = WebArenaInfinityAdapter()
    assert env._server_proc is None
    assert env._agent is None
    assert env._tasks is None
    # Default model is Sonnet 4.6 for the bootstrap (cheapest model whose JSON
    # output passes browser-use's AgentOutput schema; Haiku 4.5 was tried but
    # is structurally incompatible). Explicit override wins.
    assert env._llm_model == "claude-sonnet-4-6"


def test_adapter_rejects_non_cycle_sample_mode():
    with pytest.raises(ValueError, match="sample_mode"):
        WebArenaInfinityAdapter(sample_mode="random")


def test_adapter_sample_task_before_reset_raises():
    env = WebArenaInfinityAdapter()
    with pytest.raises(RuntimeError):
        env.sample_task(1)


@pytest.mark.skipif(not _sibling_repo_present(), reason="webarena-infinity sibling repo not present")
def test_import_webarena_locates_sibling_modules():
    """Verify the path-insertion import strategy actually finds the three
    modules the adapter relies on."""
    agents_mod, server_mod, tasks_mod = _import_webarena()
    for mod, expected_attr in [
        (agents_mod, "BrowserUseAgent"),
        (server_mod, "start_server"),
        (tasks_mod, "load_tasks"),
    ]:
        assert hasattr(mod, expected_attr), (
            f"module {mod.__name__!r} missing expected attribute {expected_attr!r} — "
            "the sibling repo's evaluation/ API may have shifted"
        )


@pytest.mark.skipif(not _sibling_repo_present(), reason="webarena-infinity sibling repo not present")
def test_gmail_task_suite_loads():
    """Verify the Gmail app's real-tasks suite — the default used by
    configs/webarena_gmail.yaml — can be read end-to-end through the adapter's
    import path, without starting a server."""
    _, _, tasks_mod = _import_webarena()
    web_app_dir = str(_webarena_root() / "apps" / "gmail")
    tasks = tasks_mod.load_tasks(web_app_dir, "real-tasks")
    assert len(tasks) >= 1, "Gmail real-tasks suite is empty — sibling repo may be partial"
    # Schema expected by the adapter: id, instruction, verify, optional difficulty.
    first = tasks[0]
    for key in ("id", "instruction", "verify"):
        assert key in first, f"task entry missing {key!r}: {first}"


@pytest.mark.skipif(
    not (_sibling_repo_present() and _browser_use_importable()),
    reason="webarena-infinity + browser-use both required for full integration",
)
def test_armed_agent_class_subclasses_browseruseagent():
    """Without launching anything, verify the lazily-built armed-agent class
    is in fact a subclass of webarena-infinity's BrowserUseAgent. Catches
    drift if either repo renames or restructures the agent base class.
    """
    from cold_start.tasks.webarena import _get_armed_agent_cls

    armed_cls = _get_armed_agent_cls()
    agents_mod, _, _ = _import_webarena()
    assert issubclass(armed_cls, agents_mod.BrowserUseAgent)
    # The injection hook the adapter relies on must exist on the subclass.
    assert hasattr(armed_cls, "set_prompt_extension")


@pytest.mark.skipif(
    "WEBARENA_INFINITY_ROOT" in os.environ,
    reason="WEBARENA_INFINITY_ROOT is set; skipping the missing-root error path",
)
def test_missing_sibling_raises_actionable_error(monkeypatch, tmp_path):
    """When the sibling repo is genuinely missing, _webarena_root() must raise
    a clear RuntimeError naming the expected path — not blow up deep in an
    import or a server timeout."""
    fake = tmp_path / "not-a-webarena"
    monkeypatch.setenv("WEBARENA_INFINITY_ROOT", str(fake))
    with pytest.raises(RuntimeError, match="webarena-infinity not found"):
        _webarena_root()
