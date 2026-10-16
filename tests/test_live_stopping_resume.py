"""Synthetic tests of the explicitly authorized, one-time continuation amendment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.growing_bandits.empirical import live_stopping as live
from experiments.growing_bandits.empirical import live_stopping_resume as resume


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    live._atomic_json(Path(path), value)


@pytest.fixture
def study(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "continuation"
    prompts = {r: {"text": r, "sha256": hashlib.sha256(r.encode()).hexdigest()} for r in live.ROLES}
    cfg = {"max_pairs": 30, "model": "frozen-test-model", "prompts": prompts,
           "budget_mode": "transport", "minimum_request_reservation_usd": .318432, "max_usd": 75.0}
    tasks = [{"task_id": f"sample_{i + 1}", "index": i + 1, "seed": i + 500} for i in range(30)]

    def evaluate(task, role, episode_id):
        failed = task["index"] == 14 and role == "challenger"
        result = {"episode_id": episode_id, "status": "infrastructure_error" if failed else "ok",
                  "success": None if failed else (0 if task["index"] == 14 else 1),
                  "cost_usd": .005 if failed else .01, "input_tokens": 10, "output_tokens": 2,
                  "model": cfg["model"], "prompt_sha256": cfg["prompts"][role]["sha256"],
                  "unknown_billing_requests": ["request_unknown_original"] if failed else [],
                  "unknown_billing_reserved_usd": .318432 if failed else 0,
                  "cost_accounting_complete": not failed}
        write(source / "episodes" / episode_id / "intent.json", {"episode_id": episode_id})
        write(source / "episodes" / episode_id / "receipt.json", result)
        return result

    original = live.run(cfg, source, tasks, evaluate)
    assert original["completed_pairs"] == 13
    (source / "api_calls.jsonl").write_text(json.dumps({"event": "request_unknown",
        "request_id": "request_unknown_original", "episode_id": resume.LOGICAL_EPISODE_ID}) + "\n")
    registration = read(source / "registration.json")
    manifest = {"source_directory": str(source), "continuation_directory": str(destination),
                "source_frozen_sha256": registration["frozen_sha256"],
                "logical_episode_id": resume.LOGICAL_EPISODE_ID,
                "physical_episode_id": resume.PHYSICAL_EPISODE_ID, "max_retry_attempts": 1,
                "acknowledged_unknown_request_ids": ["request_unknown_original"],
                "source_files": {str(p.resolve()): live.file_sha256(p) for p in source.rglob("*") if p.is_file()},
                "amendment_files": {str(Path(resume.__file__).resolve()): live.file_sha256(resume.__file__)}}
    manifest_path = tmp_path / "amendment.json"
    write(manifest_path, manifest)
    return {"source": source, "destination": destination, "config": cfg, "tasks": tasks,
            "manifest_path": manifest_path, "manifest": manifest,
            "manifest_sha256": live.file_sha256(manifest_path)}


def receipt(study, task, role, episode_id, **changes):
    cfg = study["config"]
    result = {"episode_id": episode_id, "status": "ok", "success": int(role == "challenger"),
              "cost_usd": .01, "input_tokens": 10, "output_tokens": 2, "model": cfg["model"],
              "prompt_sha256": cfg["prompts"][role]["sha256"]}
    result.update(changes)
    return result


def test_initialization_preserves_original_and_copies_only_27_valid_episodes(study):
    original_files = copy.deepcopy(study["manifest"]["source_files"])
    cfg, tasks, directory = resume.initialize_continuation(study["manifest_path"], study["manifest_sha256"])
    registration = read(study["source"] / "registration.json")
    assert cfg == registration["frozen"]["config"]
    assert tasks == registration["frozen"]["tasks"]
    assert cfg["max_usd"] == 75.0
    assert len(list((directory / "receipts").glob("*.json"))) == 27
    assert not (directory / "receipts" / f"{resume.LOGICAL_EPISODE_ID}.json").exists()
    assert read(directory / "receipts/pair_000014_baseline.json")["result"]["success"] == 0
    assert not (directory / "terminal.json").exists()
    marker = read(directory / "continuation.json")
    assert marker["excluded_original_attempt"]["result"]["success"] is None
    assert marker["excluded_original_attempt"]["result"]["cost_usd"] == .005
    assert marker["original_terminal"]["decision"] == "unresolved"
    resume.initialize_continuation(study["manifest_path"], study["manifest_sha256"])
    assert all(live.file_sha256(path) == sha for path, sha in original_files.items())


def test_one_retry_then_original_task_order_and_first_boundary(study):
    calls = []

    def evaluate(task, role, episode_id):
        calls.append((task["index"], role, episode_id))
        return receipt(study, task, role, episode_id)

    state = resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert calls[0] == (14, "challenger", resume.PHYSICAL_EPISODE_ID)
    assert calls[1:3] == [(15, "baseline", "pair_000015_baseline"),
                           (15, "challenger", "pair_000015_challenger")]
    assert state["completed_pairs"] == 24
    assert state["decision"] == "efficacy"
    assert len(calls) == 21  # one retry + ten entirely new pairs, no old baseline rerun
    pairs = read(study["destination"] / "pairs.json")
    assert pairs[:13] == read(study["source"] / "pairs.json")
    assert pairs[13]["baseline_success"] == 0 and pairs[13]["challenger_success"] == 1
    retry = read(study["destination"] / "receipts" / f"{resume.LOGICAL_EPISODE_ID}.json")
    assert retry["episode_id"] == resume.LOGICAL_EPISODE_ID
    assert retry["result"]["episode_id"] == resume.PHYSICAL_EPISODE_ID
    assert retry["result"]["physical_episode_id"] == resume.PHYSICAL_EPISODE_ID
    assert retry["result"]["logical_episode_id"] == resume.LOGICAL_EPISODE_ID
    before = list(calls)
    resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert calls == before
    assert all(live.file_sha256(p) == sha for p, sha in study["manifest"]["source_files"].items())


def test_failed_retry_ends_amendment_without_another_attempt(study):
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(episode_id)
        return receipt(study, task, role, episode_id, status="infrastructure_error", success=None)

    state = resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert state["completed_pairs"] == 13 and state["decision"] == "unresolved"
    assert state["terminal"]["reason"] == "infrastructure_error"
    resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert calls == [resume.PHYSICAL_EPISODE_ID]


def test_unknown_retry_interruption_never_submits_retry_twice(study):
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(episode_id)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    state = resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert calls == [resume.PHYSICAL_EPISODE_ID]
    assert state["terminal"]["reason"] == "ambiguous_inflight_episode"


def test_durable_valid_retry_resumes_at_next_task_without_new_retry(study, monkeypatch):
    calls = []
    real_write = live._atomic_json

    def evaluate(task, role, episode_id):
        calls.append(episode_id)
        return receipt(study, task, role, episode_id)

    def interrupt_after_retry(path, value):
        real_write(path, value)
        if path == study["destination"] / "receipts" / f"{resume.LOGICAL_EPISODE_ID}.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(live, "_atomic_json", interrupt_after_retry)
    with pytest.raises(KeyboardInterrupt):
        resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    monkeypatch.setattr(live, "_atomic_json", real_write)
    state = resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert calls[:2] == [resume.PHYSICAL_EPISODE_ID, "pair_000015_baseline"]
    assert calls.count(resume.PHYSICAL_EPISODE_ID) == 1
    assert state["decision"] == "efficacy"


def test_source_guard_rechecked_before_each_callback(study):
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(episode_id)
        (study["source"] / "api_calls.jsonl").write_text("unexpected change")
        return receipt(study, task, role, episode_id)

    state = resume.run_continuation(study["manifest_path"], study["manifest_sha256"], evaluate)
    assert calls == [resume.PHYSICAL_EPISODE_ID]
    assert state["completed_pairs"] == 14
    assert state["terminal"]["reason"] == "infrastructure_error"


def test_wrong_manifest_hash_never_initializes_or_calls(study):
    with pytest.raises(ValueError, match="manifest hash"):
        resume.run_continuation(study["manifest_path"], "0" * 64, lambda *a: pytest.fail("no call"))
    assert not study["destination"].exists()


@pytest.mark.parametrize("changes", [
    {"logical_episode_id": "pair_000015_challenger"},
    {"physical_episode_id": "pair_000014_challenger_retry2"},
    {"max_retry_attempts": 2},
    {"acknowledged_unknown_request_ids": ["unrelated_request"]},
])
def test_retry_scope_cannot_expand(study, changes):
    manifest = {**study["manifest"], **changes}
    write(study["manifest_path"], manifest)
    with pytest.raises(ValueError):
        resume.run_continuation(study["manifest_path"], live.file_sha256(study["manifest_path"]),
                                lambda *a: pytest.fail("no call"))


def test_copied_baseline_cannot_be_replaced_on_resume(study):
    resume.initialize_continuation(study["manifest_path"], study["manifest_sha256"])
    path = study["destination"] / "receipts/pair_000014_baseline.json"
    changed = read(path)
    changed["result"]["success"] = 1
    write(path, changed)
    with pytest.raises(ValueError, match="Copied original record"):
        resume.run_continuation(study["manifest_path"], study["manifest_sha256"],
                                lambda *a: pytest.fail("no call"))


def test_returned_physical_attempt_must_match(study):
    def wrong_id(task, role, episode_id):
        result = receipt(study, task, role, episode_id)
        result["episode_id"] = resume.LOGICAL_EPISODE_ID
        return result

    state = resume.run_continuation(study["manifest_path"], study["manifest_sha256"], wrong_id)
    assert state["completed_pairs"] == 13
    assert state["terminal"]["reason"] == "infrastructure_error"
