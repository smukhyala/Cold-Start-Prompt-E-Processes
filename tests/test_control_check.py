"""analyze_control_check: the refusal gate and the control verdict, on fabricated (clearly marked) logs in tmp dirs.

No model call, no server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import analyze_control_check as acc  # noqa: E402
import analyze_winner_check as awc  # noqa: E402
import build_full_oracle as bfo  # noqa: E402
import prepare_oracle_full as prep  # noqa: E402

LEG_G = ROOT / "data" / "winner_check_gitlab"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("control_data") / "control_check_gitlab"
    prep.prepare("study", out, bfo.build_all())
    return out


def _logs(data, tmp_path, **kw):
    acc.make_synthetic_log(data, tmp_path / "logs", **kw)
    awc.make_synthetic_log(LEG_G, tmp_path / "leg_g_logs")
    return tmp_path / "logs", tmp_path / "leg_g_logs"


def _run(data, tmp_path, logs, leg_logs, **kw):
    return acc.analyze(data, logs, tmp_path / "out", allow_synthetic=True, leg_g_data=LEG_G, leg_g_logs=leg_logs, **kw)


def test_control_holds_when_the_oracle_is_clearly_better(data, tmp_path):
    logs, leg_logs = _logs(data, tmp_path, seed=1, p_o2=0.9, p_d=0.5)
    result = _run(data, tmp_path, logs, leg_logs)
    primary = result["primary"]
    assert primary["n_tasks"] == 110 and primary["n_episodes"] == 440
    assert primary["control_holds"] is True
    assert primary["primary"]["lower_bound_one_sided_95"] > 0.0
    assert primary["primary"]["mean"] > 0.2
    sec = result["secondary"]
    assert sec["task_cluster_bootstrap_O2_minus_D"]["control_holds"] is True
    assert sec["task_cluster_bootstrap_O2_minus_D"]["seed"] == 20261012
    assert sec["default_drift"]["quantity"].startswith("mean over tasks of Dnew_t - Dold_t")
    assert sec["combined_reading"] in (acc.READING_INFORMATIVE, acc.READING_CONTROL_HOLDS_LEG_G_OPEN)
    assert sec["old_cookbook_split"]["in_old_cookbook"]["n_tasks"] + \
        sec["old_cookbook_split"]["not_in_old_cookbook"]["n_tasks"] == 110
    assert set(sec["operations_by_arm"]) == {"O2", "D"}
    for name in ("primary.json", "secondary.json", "task_table.csv"):
        assert (tmp_path / "out" / name).exists()


def test_control_fails_when_there_is_no_effect(data, tmp_path):
    logs, leg_logs = _logs(data, tmp_path, seed=2, p_o2=0.6, p_d=0.6)
    result = _run(data, tmp_path, logs, leg_logs)
    assert result["primary"]["control_holds"] is False
    assert result["secondary"]["combined_reading"] == acc.READING_UNINFORMATIVE


def test_combined_reading_rules():
    assert acc.combined_reading(True, 0.05) == acc.READING_INFORMATIVE
    assert acc.combined_reading(False, 0.05) == acc.READING_UNINFORMATIVE
    assert acc.combined_reading(True, 0.12) == acc.READING_CONTROL_HOLDS_LEG_G_OPEN
    assert acc.combined_reading(True, None) == acc.READING_NO_LEG_G


def test_refuses_an_incomplete_log(data, tmp_path):
    logs, leg_logs = _logs(data, tmp_path)
    lines = (logs / "worker_0.jsonl").read_text().splitlines()
    (logs / "worker_0.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(acc.AnalysisRefused, match="1 of 440 queued episodes have no terminal record"):
        _run(data, tmp_path, logs, leg_logs)
    assert not (tmp_path / "out" / "primary.json").exists()


def test_refuses_a_non_ok_episode_and_a_wrong_prompt_sha(data, tmp_path):
    logs, leg_logs = _logs(data, tmp_path)
    lines = (logs / "worker_0.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    last["status"] = "missing"
    last["success"] = None
    lines[-1] = json.dumps(last)
    (logs / "worker_0.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(acc.AnalysisRefused):
        _run(data, tmp_path, logs, leg_logs)

    logs2, leg_logs2 = _logs(data, tmp_path / "b")
    lines = (logs2 / "worker_0.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["prompt_sha256"] = "0" * 64
    lines[0] = json.dumps(first)
    (logs2 / "worker_0.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(acc.AnalysisRefused, match="prompt_sha256"):
        _run(data, tmp_path / "b", logs2, leg_logs2)


def test_never_analyses_synthetic_records_as_real(data, tmp_path):
    logs, _ = _logs(data, tmp_path)
    with pytest.raises(acc.AnalysisRefused, match=acc.SYNTHETIC_KEY):
        acc.analyze(data, logs, tmp_path / "out", allow_synthetic=False)


def test_synthetic_mode_requires_the_marker(data, tmp_path):
    logs, leg_logs = _logs(data, tmp_path)
    cleaned = []
    for line in (logs / "worker_0.jsonl").read_text().splitlines():
        row = json.loads(line)
        row.pop(acc.SYNTHETIC_KEY)
        cleaned.append(json.dumps(row))
    (logs / "worker_0.jsonl").write_text("\n".join(cleaned) + "\n")
    with pytest.raises(acc.AnalysisRefused, match="synthetic mode requires"):
        _run(data, tmp_path, logs, leg_logs)


def test_missing_leg_g_logs_do_not_block_the_primary(data, tmp_path):
    logs, _ = _logs(data, tmp_path)
    result = acc.analyze(data, logs, tmp_path / "out", allow_synthetic=True, leg_g_data=LEG_G,
                         leg_g_logs=tmp_path / "no_such_dir")
    assert result["secondary"]["leg_g_recomputed"]["available"] is False
    assert result["secondary"]["combined_reading"] == acc.READING_NO_LEG_G
    assert "primary" in result["primary"]


def test_old_cookbook_tasks_are_read_from_the_exported_matrix():
    tasks = acc.old_cookbook_tasks()
    assert len(tasks) == 40
