"""``collect.agent_settings``: a data dir's manifest may swap the agent's model, and nothing else."""

import json

import pytest

from experiments.growing_bandits.empirical import collect


def _write_manifest(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_no_manifest_keeps_profile_agent(tmp_path):
    assert collect.agent_settings(tmp_path, "gmail") == collect.profile_spec("gmail")["agent"]


def test_manifest_without_overrides_keeps_profile_agent(tmp_path):
    _write_manifest(tmp_path, {"n_tasks": 3})
    assert collect.agent_settings(tmp_path, "gitlab") == collect.profile_spec("gitlab")["agent"]


def test_overrides_swap_model_only(tmp_path):
    _write_manifest(tmp_path, {"agent_overrides": {"llm_provider": "anthropic", "llm_model": "claude-sonnet-5-5",
                                                   "llm_reasoning_effort": None}})
    agent = collect.agent_settings(tmp_path, "gitlab")
    base = collect.profile_spec("gitlab")["agent"]
    assert agent["llm_provider"] == "anthropic"
    assert agent["llm_model"] == "claude-sonnet-5-5"
    assert agent["llm_reasoning_effort"] is None
    for key in ("web_app", "timeout_s", "task_suite", "use_vision", "headless"):
        assert agent[key] == base[key]


def test_disallowed_override_refused(tmp_path):
    _write_manifest(tmp_path, {"agent_overrides": {"timeout_s": 9999}})
    with pytest.raises(RuntimeError):
        collect.agent_settings(tmp_path, "gmail")


def test_record_carries_model():
    cfg = collect.WorkerConfig(worker=0, n_workers=1, log_dir=None, budget_usd=1.0, llm_model="claude-sonnet-5-5")
    assert cfg.llm_model == "claude-sonnet-5-5"
    assert collect.WorkerConfig(worker=0, n_workers=1, log_dir=None, budget_usd=1.0).llm_model == "gpt-5.4-mini"
