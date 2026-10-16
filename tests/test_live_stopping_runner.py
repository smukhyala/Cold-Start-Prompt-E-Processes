"""No-network checks for the live experiment's actual HTTP spending boundary."""

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

import live_stopping_runner as runner  # noqa: E402


def payload(**overrides):
    return {"model": runner.MODEL, "max_completion_tokens": 4096,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "synthetic test"}],
            "service_tier": "default", **overrides}


def response_body(prompt=1000, completion=100, cached=200):
    return {"id": "test-completion", "model": runner.MODEL,
            "system_fingerprint": "test-fingerprint", "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def invoke(meter, body=None):
    async def run():
        async with httpx.AsyncClient(transport=meter) as client:
            return await client.post("https://api.openai.com/v1/chat/completions", json=body or payload())
    return asyncio.run(run())


def test_cost_uses_raw_provider_usage_and_cache_discount(tmp_path):
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response_body())))
    meter.episode_id = "episode_1"
    # Deliberately invalid choices still carry billable usage. Account before
    # browser-use attempts output parsing, which would reject this response.
    assert invoke(meter).status_code == 200
    assert meter.spent == Decimal("0.001065")
    summary = meter.episode_summary("episode_1")
    assert summary["input_tokens"] == 1000
    assert summary["output_tokens"] == 100
    assert summary["cache_read_tokens"] == 200
    assert summary["resolved_models"] == [runner.MODEL]
    assert summary["system_fingerprints"] == ["test-fingerprint"]
    restored = runner.MeteredTransport(meter.path, 75, transport=httpx.MockTransport(lambda r: None))
    assert restored.spent == meter.spent
    assert restored.reserved == 0


def test_budget_reserves_before_network_and_never_overshoots(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=response_body(prompt=400000, completion=0, cached=0))
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", .4, transport=httpx.MockTransport(handler))
    meter.episode_id = "episode_1"
    invoke(meter)
    with pytest.raises(runner.SpendingLimitReached):
        invoke(meter)
    assert len(calls) == 1
    assert meter.spent == Decimal(".3")
    assert meter.budget_exhausted
    assert meter.counter == 1


def test_insufficient_first_request_budget_makes_zero_network_calls(tmp_path):
    def forbidden(request):
        pytest.fail("no paid request should be sent")
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", .3, transport=httpx.MockTransport(forbidden))
    meter.episode_id = "episode_1"
    with pytest.raises(runner.SpendingLimitReached):
        invoke(meter)
    assert meter.counter == 0
    assert not meter.path.exists()


def test_network_error_keeps_reservation_and_blocks_resume(tmp_path):
    calls = []
    def broken(request):
        calls.append(request)
        raise httpx.ReadError("synthetic interruption")
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75, transport=httpx.MockTransport(broken))
    meter.episode_id = "episode_1"
    with pytest.raises(httpx.ReadError):
        invoke(meter)
    assert meter.reserved == runner.REQUEST_RESERVE_USD
    assert meter.episode_summary("episode_1")["unknown_billing_requests"] == ["request_000001"]
    restored = runner.MeteredTransport(meter.path, 75, transport=httpx.MockTransport(broken))
    restored.episode_id = "episode_2"
    with pytest.raises(runner.UnknownBilling):
        invoke(restored)
    assert len(calls) == 1


def test_durable_intent_precedes_network(tmp_path):
    path = tmp_path / "calls.jsonl"
    def handler(request):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[-1]["event"] == "request_started"
        assert rows[-1]["reserved_usd"] == float(runner.REQUEST_RESERVE_USD)
        return httpx.Response(200, json=response_body())
    meter = runner.MeteredTransport(path, 75, transport=httpx.MockTransport(handler))
    meter.episode_id = "episode_1"
    invoke(meter)


@pytest.mark.parametrize("status", [400, 401, 429])
def test_explicit_provider_rejections_are_separate_from_task_failure(tmp_path, status):
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75,
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json={"error": "synthetic"})))
    meter.episode_id = "episode_1"
    assert invoke(meter).status_code == status
    summary = meter.episode_summary("episode_1")
    assert summary["provider_http_errors"] == [status]
    assert summary["api_responses"] == 0
    assert summary["cost_usd"] == 0
    assert meter.reserved == 0
    with pytest.raises(runner.ProviderEpisodeFailed):
        invoke(meter)
    assert meter.counter == 1


def test_server_error_retains_uncertain_charge_and_halts(tmp_path):
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75,
        transport=httpx.MockTransport(lambda request: httpx.Response(500, json={"error": "synthetic"})))
    meter.episode_id = "episode_1"
    assert invoke(meter).status_code == 500
    assert meter.reserved == runner.REQUEST_RESERVE_USD
    summary = meter.episode_summary("episode_1")
    assert summary["provider_http_errors"] == [500]
    assert summary["unknown_billing_requests"]
    with pytest.raises(runner.UnknownBilling):
        invoke(meter)
    assert meter.counter == 1


@pytest.mark.parametrize("body", [payload(model="gpt-5.4-mini"), payload(max_completion_tokens=8192),
                                  payload(service_tier="priority"), payload(n=2), payload(stream=True)])
def test_unregistered_request_configuration_blocked_before_send(tmp_path, body):
    def forbidden(request):
        pytest.fail("no unregistered request should be sent")
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75, transport=httpx.MockTransport(forbidden))
    meter.episode_id = "episode_1"
    with pytest.raises(ValueError):
        invoke(meter, body)
    assert meter.counter == 0


@pytest.mark.parametrize("body", [{"model": runner.MODEL}, response_body(completion=4097),
                                  {**response_body(), "model": "different-model"}])
def test_incomplete_usage_or_wrong_returned_model_blocks_further_spend(tmp_path, body):
    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    meter.episode_id = "episode_1"
    with pytest.raises(ValueError):
        invoke(meter)
    assert meter.reserved == runner.REQUEST_RESERVE_USD
    with pytest.raises(runner.UnknownBilling):
        invoke(meter)


def test_completed_episode_receipt_is_reused_without_adapter_or_llm(tmp_path):
    evaluator = runner.LiveEvaluator.__new__(runner.LiveEvaluator)
    evaluator.directory = tmp_path
    evaluator.prompts = {"baseline": {"sha256": "test-hash"}}
    task = {"task_id": "sample_000", "source_task_id": "task_e1"}
    receipt = {"episode_id": "episode_1", **task, "role": "baseline", "model": runner.MODEL,
               "prompt_sha256": "test-hash", "status": "ok", "success": 1, "cost_usd": .01}
    runner._atomic_json(tmp_path / "episodes/episode_1/receipt.json", receipt)
    assert evaluator(task, "baseline", "episode_1") == receipt


def test_interrupted_episode_never_reexecuted(tmp_path):
    evaluator = runner.LiveEvaluator.__new__(runner.LiveEvaluator)
    evaluator.directory = tmp_path
    evaluator.prompts = {"baseline": {"sha256": "test-hash"}}
    runner._atomic_json(tmp_path / "episodes/episode_1/intent.json", {"episode_id": "episode_1"})
    with pytest.raises(runner.UnknownBilling):
        evaluator({"task_id": "sample_000", "source_task_id": "task_e1"}, "baseline", "episode_1")


def test_prompt_tampering_rejected_before_browser_start(tmp_path):
    config = {"model": runner.MODEL, "prompts": {"baseline": {"text": "changed", "sha256": hashlib.sha256(b"frozen").hexdigest()}}}
    with pytest.raises(ValueError, match="frozen hash"):
        runner.LiveEvaluator(config, tmp_path)


def test_torn_api_journal_never_discards_unknown_paid_work(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"event": "request_started",')
    with pytest.raises(json.JSONDecodeError):
        runner.MeteredTransport(path, 75, transport=httpx.MockTransport(lambda r: None))


def test_maximum_request_reserve_is_conservative():
    assert runner.REQUEST_RESERVE_USD == Decimal("0.318432")
    cost, *_ = runner.usage_cost(response_body(prompt=400000, completion=4096, cached=0)["usage"])
    assert cost == runner.REQUEST_RESERVE_USD


@pytest.mark.parametrize("bad_output", [None, "this is not valid JSON"])
def test_http200_invocation_failure_latches_no_paid_retry_and_keeps_usage(tmp_path, bad_output):
    from browser_use.llm.exceptions import ModelProviderError
    from browser_use.llm.messages import UserMessage
    from browser_use.llm.openai.chat import ChatOpenAI
    from pydantic import BaseModel

    class Output(BaseModel):
        answer: str

    calls = []
    def handler(request):
        calls.append(request)
        body = response_body()
        body.update({"created": 0, "object": "chat.completion"})
        if bad_output is not None:
            body["choices"] = [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": bad_output}}]
        return httpx.Response(200, json=body)

    meter = runner.MeteredTransport(tmp_path / "calls.jsonl", 75, transport=httpx.MockTransport(handler))
    meter.episode_id = "episode_1"

    async def exercise():
        async with httpx.AsyncClient(transport=meter) as client:
            llm = runner.latch_invocation_errors(ChatOpenAI(
                model=runner.MODEL, api_key="synthetic-not-a-credential", reasoning_effort="low",
                max_completion_tokens=4096, service_tier="default", max_retries=0,
                http_client=client, base_url="https://api.openai.com/v1"), meter)
            with pytest.raises(ModelProviderError):
                await llm.ainvoke([UserMessage(content="synthetic")], output_format=Output)
            with pytest.raises(runner.ProviderEpisodeFailed):
                await llm.ainvoke([UserMessage(content="synthetic")], output_format=Output)
    asyncio.run(exercise())
    assert len(calls) == 1
    assert meter.spent == Decimal("0.001065")
    assert meter.episode_summary("episode_1")["provider_invocation_errors"] == ["ModelProviderError"]
    restored = runner.MeteredTransport(meter.path, 75, transport=httpx.MockTransport(handler))
    assert restored.provider_failed
    assert restored.spent == meter.spent


def test_reload_prevents_stale_meter_spending_after_lock_takeover(tmp_path):
    path = tmp_path / "calls.jsonl"
    first = runner.MeteredTransport(path, .4, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_body(prompt=400000, completion=0, cached=0))))
    stale = runner.MeteredTransport(path, .4, transport=httpx.MockTransport(
        lambda request: pytest.fail("stale controller must not dispatch")))
    first.episode_id = "episode_1"
    invoke(first)
    assert stale.spent == 0
    stale.reload_accounting()
    stale.episode_id = "episode_2"
    assert stale.spent == Decimal(".3")
    with pytest.raises(runner.SpendingLimitReached):
        invoke(stale)
    assert stale.counter == 1
