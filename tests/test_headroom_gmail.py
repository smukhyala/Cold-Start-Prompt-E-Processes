"""Pre-registration 15 (Gmail answer-key headroom): the frozen queue and the frozen analyzer's rule."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))

import analyze_headroom_gmail as ahg  # noqa: E402
import prepare_headroom_gmail as phg  # noqa: E402


def test_queue_shape_and_balance():
    tasks = []
    for i in range(60):
        tasks.append(f"task_{i}")
    queue = phg.build_queue(tasks)
    assert len(queue) == 240
    counts = {}
    for item in queue:
        key = (item.arm_id, item.task_id)
        counts[key] = counts.get(key, 0) + 1
    for task in tasks:
        assert counts[(phg.O_ARM, task)] == 2
        assert counts[(phg.D_ARM, task)] == 2
    # a task's 4 episodes are contiguous
    first = {}
    last = {}
    for item in queue:
        first.setdefault(item.task_id, item.index)
        last[item.task_id] = item.index
    for task in tasks:
        assert last[task] - first[task] == 3


def test_queue_is_seeded():
    tasks = ["a", "b", "c"]
    assert phg.queue_text(phg.build_queue(tasks)) == phg.queue_text(phg.build_queue(tasks))


def test_reading_rule():
    assert ahg.reading(0.10, 0.30) == ahg.READING_LARGE
    assert ahg.reading(0.099, 0.30) == ahg.READING_UNRESOLVED
    assert ahg.reading(-0.05, 0.0999) == ahg.READING_SMALL
    assert ahg.reading(-0.05, 0.10) == ahg.READING_UNRESOLVED


@pytest.mark.skipif(not (ROOT / "data" / "headroom_gmail" / "manifest.json").exists(), reason="data dir not prepared")
def test_frozen_data_dir_and_synthetic_pipeline(tmp_path):
    manifest = json.loads((ROOT / "data" / "headroom_gmail" / "manifest.json").read_text())
    assert manifest["n_episodes"] == 240
    assert manifest["n_tasks"] == 60
    assert manifest["agent"]["llm_model"] == "gpt-5.4-mini"
    phg.validate(ROOT / "data" / "headroom_gmail")
    result = ahg.synthetic_dry_run(out_dir=tmp_path)
    assert result["primary"]["synthetic"] is True
    assert result["primary"]["reading"] in (ahg.READING_LARGE, ahg.READING_SMALL, ahg.READING_UNRESOLVED)
    assert abs(result["secondary"]["default_drift_vs_leg_m"]["D_leg_m_mean"] - 0.7166666) < 1e-4


def test_real_mode_refuses_missing_log(tmp_path):
    with pytest.raises(ahg.AnalysisRefused):
        ahg.analyze(log_dir=tmp_path / "nope", out_dir=tmp_path / "out")
