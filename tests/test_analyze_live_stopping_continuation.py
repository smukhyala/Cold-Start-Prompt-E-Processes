"""Synthetic continuation accounting against the immutable exported original run.

Only temporary files are written. External dependency/source hashes are tested
by the registration gates; these tests isolate physical-attempt accounting.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.growing_bandits.empirical import analyze_live_stopping_continuation as analysis

ROOT = Path(__file__).resolve().parents[1]
EXPORTED = ROOT / "results/paper/live_stopping/history/2026-10-16_initial_halt"
RETRY = "pair_000014_challenger_retry1"


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    source, continued = tmp_path / "original", tmp_path / "continued"
    source.mkdir()
    continued.mkdir()
    for name in ("registration.json", "pairs.json", "evidence.json", "terminal.json", "api_calls.jsonl"):
        (source / name).write_bytes((EXPORTED / name).read_bytes())
    registration = _read(source / "registration.json")
    # The source-files manifest below still checks every temporary input. The
    # unrelated absolute paths to the execution machine's dependencies are not
    # prerequisites for this isolated accounting test.
    monkeypatch.setattr(analysis, "verify_frozen_files", lambda _config: None)
    original = _read(EXPORTED / "episode_receipts.json")
    assert len(original) == 28
    for core in original:
        episode_id = core["episode_id"]
        result = core["result"]
        _write(source / "receipts" / f"{episode_id}.json", core)
        _write(source / "episodes" / episode_id / "receipt.json", result)
        intent = {key: result[key] for key in
                  ("episode_id", "task_id", "source_task_id", "role", "model", "prompt_sha256")}
        intent["timestamp_utc"] = result["started_utc"]
        _write(source / "episodes" / episode_id / "intent.json", intent)
        if result["status"] == "ok":
            _write(continued / "receipts" / f"{episode_id}.json", core)
    for name in ("registration.json", "pairs.json", "evidence.json"):
        (continued / name).write_bytes((source / name).read_bytes())
    config = registration["frozen"]["config"]
    task = registration["frozen"]["tasks"][13]
    identity = {"episode_id": RETRY, "task_id": task["task_id"],
                "source_task_id": task["source_task_id"], "role": "challenger",
                "model": config["model"], "prompt_sha256": config["prompts"]["challenger"]["sha256"]}
    _write(continued / "episodes" / RETRY / "intent.json",
           {**identity, "timestamp_utc": "2026-10-16T20:00:00+00:00"})
    terminal = {"decision": "unresolved", "reason": "infrastructure_error",
                "first_crossing": None, "completed_pairs": 13,
                "recorded_at": "2026-10-16T20:00:10+00:00"}
    _write(continued / "terminal.json", terminal)
    manifest = {"source_directory": str(source), "continuation_directory": str(continued),
                "source_files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source.rglob("*") if p.is_file()},
                "amendment_files": {str(Path(analysis.__file__).resolve()):
                                    hashlib.sha256(Path(analysis.__file__).read_bytes()).hexdigest()}}
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    return {"source": source, "continued": continued, "identity": identity,
            "manifest": path, "out": tmp_path / "export"}


def _ledger(fixture, *, completed):
    rows = []
    if completed:
        rows.extend([
            {"event": "request_started", "request_id": "request_000001", "episode_id": RETRY,
             "timestamp_utc": "2026-10-16T20:00:01+00:00", "reserved_usd": .318432},
            {"event": "request_completed", "request_id": "request_000001", "episode_id": RETRY,
             "timestamp_utc": "2026-10-16T20:00:02+00:00", "cost_usd": .000165,
             "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0,
             "raw_usage": {"prompt_tokens": 100, "completion_tokens": 20,
                           "prompt_tokens_details": {"cached_tokens": 0}},
             "resolved_model": fixture["identity"]["model"]},
        ])
    unknown_id = "request_000002" if completed else "request_000001"
    rows.extend([
        {"event": "request_started", "request_id": unknown_id, "episode_id": RETRY,
         "timestamp_utc": "2026-10-16T20:00:03+00:00", "reserved_usd": .318432},
        {"event": "request_unknown", "request_id": unknown_id, "episode_id": RETRY,
         "timestamp_utc": "2026-10-16T20:00:04+00:00", "reserved_usd": .318432},
    ])
    (fixture["continued"] / "api_calls.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))


def test_combined_physical_attempts_keep_failed_cost_and_both_unknown_reserves(continuation):
    _ledger(continuation, completed=True)
    receipt = {**continuation["identity"], "status": "infrastructure_error", "success": None,
               "cost_usd": .000165, "input_tokens": 100, "output_tokens": 20,
               "cache_read_tokens": 0, "finished_utc": "2026-10-16T20:00:05+00:00"}
    _write(continuation["continued"] / "episodes" / RETRY / "receipt.json", receipt)
    summary = analysis.analyze(continuation["manifest"], continuation["out"])
    assert summary["actual_episode_attempts"] == 29  # 28 original + one physical retry
    assert summary["valid_outcome_episodes"] == 27  # Copied core receipts are not counted again.
    assert summary["known_api_cost_usd"] == pytest.approx(1.76230815 + .000165, abs=1e-12)
    assert summary["unknown_charge_reserve_usd"] == .636864
    assert summary["total_cost_interval_usd"][1] == pytest.approx(2.39933715, abs=1e-12)
    assert summary["input_tokens"] == 4625255 + 100
    assert summary["output_tokens"] == 173209 + 20
    assert not summary["cost_accounting_complete"]
    assert summary["completed_pairs"] == 13 and summary["first_crossing"] is None
    assert summary["episodes_avoided_by_statistical_stop"] == 0
    combined = [json.loads(line) for line in (continuation["out"] / "api_calls.jsonl").read_text().splitlines()]
    assert {r["session"] for r in combined if r["request_id"] == "request_000001"} == {
        "original", "continuation"}


def test_unreceipted_retry_is_reported_without_imputation_or_lost_reservation(continuation):
    _ledger(continuation, completed=False)
    terminal_path = continuation["continued"] / "terminal.json"
    terminal = _read(terminal_path)
    terminal["reason"] = "ambiguous_inflight_episode"
    _write(terminal_path, terminal)
    summary = analysis.analyze(continuation["manifest"], continuation["out"])
    assert summary["actual_episode_attempts"] == 29
    assert summary["retry_attempts_started"] == 1
    assert summary["missing_episode_receipts"] == [("continuation", RETRY)]
    assert summary["known_api_cost_usd"] == 1.76230815
    assert summary["unknown_charge_reserve_usd"] == .636864
    assert not summary["cost_accounting_complete"]
    assert summary["completed_pairs"] == 13 and summary["valid_outcome_episodes"] == 27
    assert summary["episodes_avoided_by_statistical_stop"] == 0


def test_wrong_physical_task_identity_is_rejected(continuation):
    _ledger(continuation, completed=False)
    intent_path = continuation["continued"] / "episodes" / RETRY / "intent.json"
    intent = copy.deepcopy(_read(intent_path))
    intent["source_task_id"] = "wrong_source_task"
    _write(intent_path, intent)
    with pytest.raises(AssertionError):
        analysis.analyze(continuation["manifest"], continuation["out"])
