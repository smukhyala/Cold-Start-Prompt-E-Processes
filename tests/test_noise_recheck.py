"""Frozen Gmail noise-recheck: no model calls, no servers."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/empirical"))
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/retrospective"))

import analyze_noise_recheck as an  # noqa: E402
import collect  # noqa: E402
import make_pools  # noqa: E402
import prepare_noise_recheck as prep  # noqa: E402

DATA = ROOT / "data/noise_recheck_gmail"


@pytest.fixture(scope="module")
def frozen():
    return an.load_frozen(DATA)


def test_queue_has_each_cell_twice(frozen):
    manifest, queue = frozen
    assert len(queue) == 60
    assert [q["index"] for q in queue] == list(range(60))
    counts = Counter((q["arm_id"], q["task_id"]) for q in queue)
    assert len(counts) == 30 and set(counts.values()) == {2}
    for cell in {(q["arm_id"], q["task_id"]) for q in queue}:
        reps = sorted(q["replicate"] for q in queue if (q["arm_id"], q["task_id"]) == cell)
        assert reps == [0, 1]
    assert all(q["pool"] == "G" and q["pilot"] is False for q in queue)


def test_cells_are_historical_repeat_cells(frozen):
    manifest, queue = frozen
    hist = {(c["arm_id"], c["task_id"]) for c in prep.historical_cells()}
    assert len(hist) == 145
    assert {(q["arm_id"], q["task_id"]) for q in queue} <= hist


def test_manifest_hashes_and_collector_checks(frozen):
    manifest, _ = frozen
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected
    for name in ("pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml"):
        src = ROOT / "data/empirical_pool" / name
        assert (DATA / name).read_bytes() == src.read_bytes()
    collect.verify_queue(DATA)
    prompts = collect.load_prompts(DATA)
    collect.check_queue_arms(make_pools.read_queue(DATA / "queue.jsonl"), prompts)
    assert "30-step cap" in manifest["disclosure"]


def test_analyzer_refuses_incomplete_log(frozen, tmp_path):
    manifest, queue = frozen
    with pytest.raises(ValueError, match="incomplete"):
        an.read_terminals(tmp_path, queue)
    row = {"arm_id": queue[0]["arm_id"], "task_id": queue[0]["task_id"], "replicate": queue[0]["replicate"],
           "status": "ok", "success": 1, "cost_usd": 0.1}
    (tmp_path / "worker_0.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        an.read_terminals(tmp_path, queue)


def test_synthetic_dry_run(frozen, tmp_path):
    manifest, queue = frozen
    summary = an.analyze(manifest, an.synthetic_terminals(manifest, queue, 0.3))
    assert summary["n_episodes"] == 60 and summary["n_cells"] == 30
    lo, hi = summary["now_disagree_ci95_clopper_pearson"]
    assert 0 <= lo <= summary["now_disagree_rate"] <= hi <= 1
    assert summary["september_disagree_all_k"] == 15 and summary["september_disagree_all_n"] == 145
    assert 0 <= summary["binomial_test_vs_0.103"]["p_two_sided"] <= 1
    assert sys.modules["analyze_noise_recheck"].clopper_pearson(0, 30)[0] == 0.0
