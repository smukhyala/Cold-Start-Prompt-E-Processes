"""Audit/export tests use temporary synthetic data only; no publication artifacts."""
import hashlib
import json

import pytest

from experiments.growing_bandits.empirical.analyze_live_stopping import analyze
from experiments.growing_bandits.empirical.live_stopping import run


def example(tmp_path, invalid=False):
    prompts = {r: {"text": r, "sha256": hashlib.sha256(r.encode()).hexdigest()}
               for r in ("baseline", "challenger")}
    config = {"max_pairs": 12, "model": "synthetic", "max_usd": 10.,
              "episode_cost_bound_usd": .1, "prompts": prompts}
    def evaluate(task, role, episode_id):
        return {"status": "provider_error" if invalid else "ok",
                "success": None if invalid else int(role == "challenger"),
                "cost_usd": .01, "input_tokens": 10, "output_tokens": 2,
                "model": "synthetic", "prompt_sha256": prompts[role]["sha256"]}
    directory = tmp_path / "logs"
    run(config, directory, [{"task_id": f"synthetic_{i}"} for i in range(12)], evaluate)
    return directory


def test_export_reproduces_receipts_and_first_boundary(tmp_path):
    directory = example(tmp_path)
    summary = analyze(directory, tmp_path / "export")
    assert summary["decision"] == "efficacy"
    assert summary["first_crossing_pair"] == 11
    assert summary["episode_receipts"] == 22
    assert summary["episodes_avoided_by_statistical_stop"] == 2
    assert summary["measured_api_cost_usd"] == pytest.approx(.22)
    assert not summary["checks"]["http_meter_reconciles"]
    assert (tmp_path / "export/live_stopping_evidence.pdf").exists()


def test_export_rejects_derived_outcomes_that_disagree_with_receipt(tmp_path):
    directory = example(tmp_path)
    path = directory / "pairs.json"
    pairs = json.loads(path.read_text())
    pairs[0]["baseline_success"] = 1
    path.write_text(json.dumps(pairs))
    with pytest.raises(AssertionError):
        analyze(directory, tmp_path / "export")


def test_no_figure_or_statistical_savings_for_no_completed_pairs(tmp_path):
    directory = example(tmp_path, invalid=True)
    summary = analyze(directory, tmp_path / "export")
    assert summary["completed_pairs"] == 0
    assert summary["first_crossing_pair"] is None
    assert summary["episodes_avoided_by_statistical_stop"] == 0
    assert not (tmp_path / "export/live_stopping_evidence.pdf").exists()
