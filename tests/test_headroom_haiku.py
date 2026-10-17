"""Pre-registration 16 (Gmail headroom under Haiku 4.5): frozen data dir, model override, analyzer pipeline."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import analyze_headroom_gmail as ahg  # noqa: E402
import analyze_headroom_haiku as ahh  # noqa: E402
import collect  # noqa: E402
import prepare_headroom_haiku as phh  # noqa: E402

DATA = ROOT / "data" / "headroom_haiku_gmail"


def test_queue_balanced_and_differs_from_prereg15():
    import prepare_headroom_gmail as phg
    tasks = []
    for i in range(60):
        tasks.append(f"task_{i}")
    q16 = phh.build_queue(tasks)
    assert len(q16) == 240
    assert phg.queue_text(q16) != phg.queue_text(phg.build_queue(tasks))


@pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="data dir not prepared")
def test_manifest_swaps_only_the_model():
    manifest = json.loads((DATA / "manifest.json").read_text())
    agent = collect.agent_settings(DATA, "gmail")
    assert agent["llm_model"] == "claude-haiku-4-5-20251001"
    assert agent["llm_provider"] == "anthropic"
    assert agent["use_thinking"] is False
    base = collect.profile_spec("gmail")["agent"]
    for key in ("web_app", "timeout_s", "task_suite", "use_vision", "headless"):
        assert agent[key] == base[key]
    p15 = json.loads((ROOT / "data" / "headroom_gmail" / "manifest.json").read_text())
    assert manifest["files"]["pools/gmoraclefull.yaml"] == p15["files"]["pools/gmoraclefull.yaml"]
    assert manifest["files"]["pools/anchors_gmail.yaml"] == p15["files"]["pools/anchors_gmail.yaml"]


@pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="data dir not prepared")
def test_synthetic_pipeline(tmp_path):
    ahg.make_synthetic_log(DATA, tmp_path / "logs")
    result = ahh.analyze(DATA, tmp_path / "logs", tmp_path / "results", allow_synthetic=True)
    assert result["primary"]["synthetic"] is True
    assert result["secondary_haiku"]["key_use"]["haiku"] == {"skipped": "synthetic"}


def test_key_use_patterns():
    assert ahh.REFERENCE.search('{"memory": "Per the Answer key, email 31"}')
    assert ahh.DIRECT_URL.search('{"url": "http://localhost:8204/#/email/31"}')
    assert not ahh.DIRECT_URL.search('{"url": "http://localhost:8204/#/inbox"}')
