"""Pre-registration 13, Gmail arm: frozen data dir, analyzer gate and maths. No model call, no server."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/empirical"))
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/retrospective"))

import analyze_winner_check_gmail as ana  # noqa: E402
import collect  # noqa: E402
import make_pools  # noqa: E402
import prepare_winner_check_gmail as prep  # noqa: E402

DATA = ROOT / "data/winner_check_gmail"
PILOT = ROOT / "data/first_run_pilot_gmail"


@pytest.fixture(scope="module")
def frozen():
    return ana.load_frozen(DATA)


def test_queue_60_items_each_task_replicate_once(frozen):
    manifest, queue = frozen
    assert len(queue) == 60 == manifest["n_episodes"]
    assert [q.index for q in queue] == list(range(60))
    keys = Counter((q.task_id, q.replicate) for q in queue)
    assert len(keys) == 60 and set(keys.values()) == {1}
    for task in manifest["task_ids"]:
        assert sorted(q.replicate for q in queue if q.task_id == task) == [0, 1]


def test_tasks_equal_pilot_and_arm_is_default(frozen):
    manifest, queue = frozen
    pilot = json.loads((PILOT / "manifest.json").read_text())
    assert {q.task_id for q in queue} == set(pilot["task_ids"]) and len(pilot["task_ids"]) == 30
    assert {q.arm_id for q in queue} == {"anchor_baseline"} == {manifest["default_arm"]}
    assert {q.pool for q in queue} == {"anchor"}
    assert all(q.pilot is False for q in queue)
    september = json.loads((ROOT / "data/empirical_pool/manifest.json").read_text())["task_subset"]
    assert not set(manifest["task_ids"]) & set(september)


def test_manifest_hashes_and_collector_checks(frozen):
    manifest, _ = frozen
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected
    for name in ("pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml"):
        assert (DATA / name).read_bytes() == (PILOT / name).read_bytes()
    collect.verify_queue(DATA)
    prompts = collect.load_prompts(DATA)
    collect.check_queue_arms(make_pools.read_queue(DATA / "queue.jsonl"), prompts)
    for key in ("step_cap", "time_gap", "june_baseline"):
        assert manifest["disclosure"][key]
    assert manifest["seed"] == prep.SEED == 20261012


def test_analyzer_refuses_incomplete_logs(tmp_path):
    with pytest.raises(ana.AnalysisRefused):
        ana.load_scored(DATA, tmp_path / "nope", ana.PILOT_LOGS)
    empty = tmp_path / "logs"
    empty.mkdir()
    with pytest.raises(ana.AnalysisRefused, match="no records"):
        ana.load_scored(DATA, empty, ana.PILOT_LOGS)
    # an unmarked log missing one episode is refused as incomplete
    ana.make_synthetic_logs(DATA, tmp_path / "s_new", tmp_path / "s_pilot")
    lines = (tmp_path / "s_new/worker_0.jsonl").read_text().splitlines()
    partial = []
    for line in lines[:59]:
        rec = json.loads(line)
        rec.pop(ana.SYNTHETIC_KEY)
        partial.append(json.dumps(rec))
    (empty / "worker_0.jsonl").write_text("\n".join(partial) + "\n")
    with pytest.raises(ana.AnalysisRefused, match="incomplete"):
        ana.load_scored(DATA, empty, ana.PILOT_LOGS)


def test_analyzer_refuses_synthetic_in_real_mode(tmp_path):
    ana.make_synthetic_logs(DATA, tmp_path / "n", tmp_path / "p")
    with pytest.raises(ana.AnalysisRefused, match="fabricated"):
        ana.analyze(DATA, tmp_path / "n", tmp_path / "p", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_synthetic_dry_run_writes_only_synthetic_files(tmp_path):
    primary = ana.synthetic_dry_run(DATA, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["primary_synthetic.json", "task_table_synthetic.csv"]
    assert primary["synthetic"] is True and "WARNING" in primary
    assert primary["n_tasks"] == 30
    assert primary["primary"]["registered_reading"].split(":")[0] in ("reproduced", "shrank", "inconclusive")
    assert primary["primary"]["task_bootstrap"]["n_boot"] == 10_000


def test_rescore_applied_identically_to_both_arms():
    rows = []
    for arm in ("G_14", "anchor_baseline"):
        rows.append({"arm_id": arm, "status": "ok", "success": 1, "timed_out": False, "steps": 31})
        rows.append({"arm_id": arm, "status": "ok", "success": 1, "timed_out": False, "steps": 30})
        rows.append({"arm_id": arm, "status": "ok", "success": 1, "timed_out": True, "steps": 5})
        rows.append({"arm_id": arm, "status": "ok", "success": 0, "timed_out": False, "steps": 5})
    scored = ana.st.rescore(pd.DataFrame(rows), 30)
    for arm in ("G_14", "anchor_baseline"):
        got = scored[scored["arm_id"] == arm]["success"].astype(int).tolist()
        assert got == [0, 1, 0, 0]


def test_ub_matches_scipy_on_toy_example():
    d = np.array([0.5, 0.0, 0.5, -0.5, 0.0, 1.0, 0.5, 0.0])
    s = ana.t_summary(d)
    ci = stats.t.interval(0.90, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    assert s["upper_bound_one_sided"] == pytest.approx(ci[1])
    assert s["lower_bound_one_sided"] == pytest.approx(ci[0])
    assert s["interval_two_sided_90"] == pytest.approx(list(ci))
    assert ana.reading({"lower_bound_one_sided": 0.12, "upper_bound_one_sided": 0.3}).startswith("reproduced")
    assert ana.reading({"lower_bound_one_sided": -0.1, "upper_bound_one_sided": 0.09}).startswith("shrank")
    assert ana.reading({"lower_bound_one_sided": -0.1, "upper_bound_one_sided": 0.2}) == "inconclusive"


def test_real_pilot_log_has_all_60_g14_episodes():
    prompts = collect.load_prompts(DATA)
    terminal = ana._terminal(ana.PILOT_LOGS, False, prompts)
    sub = terminal[terminal["arm_id"] == "G_14"]
    manifest, _ = ana.load_frozen(DATA)
    have = {(str(r.task_id), int(r.replicate)) for r in sub.itertuples() if r.status == "ok"}
    assert have == {(t, r) for t in manifest["task_ids"] for r in (0, 1)}
