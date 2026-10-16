"""The authorized retry retains earlier charges and never enables later retries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/empirical"))

import resume_live_stopping_runner as resume  # noqa: E402


def fixture(tmp_path, global_max=1.0):
    source = tmp_path / "original"
    source.mkdir()
    usage = {"prompt_tokens": 400000, "completion_tokens": 0,
             "prompt_tokens_details": {"cached_tokens": 0}}
    rows = [
        {"event": "request_started", "request_id": "request_000457",
         "episode_id": "pair_000014_challenger", "reserved_usd": .318432},
        {"event": "request_completed", "request_id": "request_000457",
         "episode_id": "pair_000014_challenger", "cost_usd": .3, "raw_usage": usage},
        {"event": "request_started", "request_id": "request_000458",
         "episode_id": "pair_000014_challenger", "reserved_usd": .318432},
        {"event": "request_unknown", "request_id": "request_000458"},
        {"event": "invocation_failed", "request_id": "request_000458"},
    ]
    ledger = source / "api_calls.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"source_directory": str(source), "continuation_directory": str(tmp_path / "continuation"),
                "source_files": {"api_calls.jsonl": hashlib.sha256(ledger.read_bytes()).hexdigest()},
                "acknowledged_unknown_request_ids": ["request_000458"],
                "logical_episode_id": "pair_000014_challenger",
                "physical_episode_id": "pair_000014_challenger_retry1", "max_retry_attempts": 1}
    prompts = {role: {"text": role, "sha256": hashlib.sha256(role.encode()).hexdigest(),
                      "arm_id": role, "vector": {"planning": 0, "verification": 0, "agency": 0,
                                                  "expertise": 1, "format": 0, "goal": 0}}
               for role in ("baseline", "challenger")}
    config = {"model": resume.frozen.MODEL, "max_usd": global_max,
              "prompts": prompts, "task_suite": str(tmp_path / "tasks"),
              "max_steps": 30, "timeout_s": 180, "reasoning_effort": "low"}
    return manifest, config, ledger


def invoke(meter):
    async def exercise():
        async with httpx.AsyncClient(transport=meter) as client:
            return await client.post("https://api.openai.com/v1/chat/completions", json={
                "model": resume.frozen.MODEL, "max_completion_tokens": 4096,
                "reasoning_effort": "low", "service_tier": "default", "messages": []})
    return asyncio.run(exercise())


def test_opening_balance_includes_paid_part_of_invalid_episode_and_uncertainty(tmp_path):
    manifest, config, ledger = fixture(tmp_path)
    before = ledger.read_bytes()
    balance = resume.opening_balance(manifest, config)
    assert balance["original_known_cost_usd"] == .3
    assert balance["original_unknown_reserved_usd"] == .318432
    assert balance["new_segment_maximum_usd"] == .381568
    assert balance["original_started_requests"] == 2
    assert ledger.read_bytes() == before


def test_current_experiment_balance_exact_and_readonly():
    source = ROOT / "logs/prospective_live_stopping"
    if not source.exists():
        pytest.skip("local raw execution logs unavailable")
    ledger = source / "api_calls.jsonl"
    manifest = {"source_directory": str(source),
                "source_files": {"api_calls.jsonl": hashlib.sha256(ledger.read_bytes()).hexdigest()},
                "acknowledged_unknown_request_ids": ["request_000458"],
                "logical_episode_id": "pair_000014_challenger",
                "physical_episode_id": "pair_000014_challenger_retry1"}
    balance = resume.opening_balance(manifest, {"max_usd": 75})
    assert balance["original_known_cost_usd"] == 1.76230815
    assert balance["original_unknown_reserved_usd"] == .318432
    assert balance["new_segment_maximum_usd"] == 72.91925985


def test_combined_original_and_continuation_charge_cannot_exceed_global_cap(tmp_path):
    manifest, config, ledger = fixture(tmp_path)
    evaluator = resume.AuthorizedContinuationEvaluator(config, Path(manifest["continuation_directory"]), manifest)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"model": resume.frozen.MODEL,
            "usage": {"prompt_tokens": 400000, "completion_tokens": 0,
                      "prompt_tokens_details": {"cached_tokens": 0}}})
    evaluator.meter.transport = httpx.MockTransport(handler)
    evaluator.meter.episode_id = manifest["physical_episode_id"]
    invoke(evaluator.meter)
    with pytest.raises(resume.frozen.SpendingLimitReached):
        invoke(evaluator.meter)
    assert len(calls) == 1
    combined = Decimal(".3") + Decimal(".318432") + evaluator.meter.spent
    assert combined == Decimal(".918432") < Decimal(str(config["max_usd"]))
    assert config["max_usd"] == 1.0  # caller's original registration is unchanged
    assert ledger.read_bytes().count(b"request_started") == 2


def test_reopening_continuation_deducts_original_balance_once(tmp_path):
    manifest, config, _ledger = fixture(tmp_path)
    directory = Path(manifest["continuation_directory"])
    first = resume.AuthorizedContinuationEvaluator(config, directory, manifest)
    saved = (directory / "opening_balance.json").read_bytes()
    second = resume.AuthorizedContinuationEvaluator(config, directory, manifest)
    assert first.meter.max_usd == second.meter.max_usd == Decimal(".381568")
    assert first.meter.spent == second.meter.spent == 0
    assert (directory / "opening_balance.json").read_bytes() == saved


def test_new_unknown_request_halts_despite_acknowledgement_of_old_request(tmp_path):
    manifest, config, _ledger = fixture(tmp_path)
    evaluator = resume.AuthorizedContinuationEvaluator(config, Path(manifest["continuation_directory"]), manifest)
    calls = []
    def interrupted(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic network failure")
    evaluator.meter.transport = httpx.MockTransport(interrupted)
    evaluator.meter.episode_id = manifest["physical_episode_id"]
    with pytest.raises(httpx.ReadTimeout):
        invoke(evaluator.meter)
    with pytest.raises(resume.frozen.UnknownBilling):
        invoke(evaluator.meter)
    assert len(calls) == 1
    assert evaluator.meter.reserved == Decimal(".318432")
    assert evaluator.balance["original_unknown_reserved_usd"] == .318432


def test_only_exact_authorized_physical_retry_is_allowed(tmp_path):
    manifest, config, _ledger = fixture(tmp_path)
    evaluator = resume.AuthorizedContinuationEvaluator(config, Path(manifest["continuation_directory"]), manifest)
    task = {"task_id": "live_014"}
    for identifier in ("pair_000014_challenger", "pair_000014_challenger_retry2", "pair_000015_challenger_retry1"):
        with pytest.raises(ValueError, match="not authorized"):
            evaluator(task, "challenger", identifier)
    with pytest.raises(ValueError, match="fourteenth draw"):
        evaluator(task, "baseline", manifest["physical_episode_id"])


def test_unacknowledged_original_ambiguity_rejected(tmp_path):
    manifest, config, _ledger = fixture(tmp_path)
    manifest["acknowledged_unknown_request_ids"] = []
    with pytest.raises(ValueError, match="every unresolved"):
        resume.opening_balance(manifest, config)


def test_changed_source_ledger_or_opening_balance_rejected(tmp_path):
    manifest, config, ledger = fixture(tmp_path)
    directory = Path(manifest["continuation_directory"])
    resume.AuthorizedContinuationEvaluator(config, directory, manifest)
    balance_path = directory / "opening_balance.json"
    balance = json.loads(balance_path.read_text())
    balance["original_unknown_reserved_usd"] = 0
    balance_path.write_text(json.dumps(balance))
    with pytest.raises(ValueError, match="opening balance changed"):
        resume.AuthorizedContinuationEvaluator(config, directory, manifest)
    ledger.write_text(ledger.read_text() + "\n")
    with pytest.raises(ValueError, match="not uniquely pinned"):
        resume.opening_balance(manifest, config)


def test_preflight_cli_never_calls_paid_controller(tmp_path, monkeypatch):
    from experiments.growing_bandits.empirical import live_stopping_resume

    manifest, config, _ledger = fixture(tmp_path)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(manifest))
    events = []
    monkeypatch.setattr(live_stopping_resume, "initialize_continuation",
                        lambda *args: (config, [], Path(manifest["continuation_directory"])))
    def forbidden(*args):
        pytest.fail("preflight must not start paid execution")
    monkeypatch.setattr(live_stopping_resume, "run_continuation", forbidden)
    class DummyEvaluator:
        def __init__(self, *args):
            pass
        def prepare(self):
            events.append("prepare")
            return {"ready": True, "llm_calls": 0}
        def close(self):
            events.append("close")
    monkeypatch.setattr(resume, "AuthorizedContinuationEvaluator", DummyEvaluator)
    assert resume.main(["--manifest", str(path), "--manifest-sha256", "test", "--preflight"]) == 0
    assert events == ["prepare", "close"]
