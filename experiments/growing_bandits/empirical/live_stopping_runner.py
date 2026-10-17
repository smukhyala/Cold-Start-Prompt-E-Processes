"""Real WebArena callback with a durable, per-request USD spending guard.

The existing adapter supplies the browser agent and verifier.  This module only
adds request accounting, a fail-closed spending boundary, and episode receipts.
No API request is made by importing or constructing the evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT, ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import collect  # noqa: E402

from cold_start.tasks.webarena import WebArenaInfinityAdapter, _loop  # noqa: E402
from cold_start.types import Arm, PromptVector, Task  # noqa: E402

MODEL = "gpt-5.4-mini-2026-03-17"
INPUT_RATE = Decimal("0.00000075")
CACHED_RATE = Decimal("0.000000075")
OUTPUT_RATE = Decimal("0.0000045")
MAX_CONTEXT_TOKENS = 400_000
MAX_COMPLETION_TOKENS = 4096
# Deliberately counts all 400k context tokens as input AND adds the output cap.
# Standard global endpoint, standard tier, no hosted tools or regional uplift.
REQUEST_RESERVE_USD = INPUT_RATE * MAX_CONTEXT_TOKENS + OUTPUT_RATE * MAX_COMPLETION_TOKENS


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _append_durable(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SpendingLimitReached(RuntimeError):
    """The next request cannot be reserved inside the registered USD limit."""


class UnknownBilling(RuntimeError):
    """A started request lacks an authoritative usage or rejection receipt."""


class ProviderEpisodeFailed(RuntimeError):
    """A provider error already ended this registered no-retry experiment."""


def usage_cost(usage: dict) -> tuple[Decimal, int, int, int]:
    """Price provider-reported usage, rejecting incomplete accounting."""
    def integer(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {name} in provider usage")
        return value

    prompt = integer("prompt_tokens", usage.get("prompt_tokens"))
    completion = integer("completion_tokens", usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    cached = integer("cached_tokens", details.get("cached_tokens", 0))
    if cached > prompt or prompt > MAX_CONTEXT_TOKENS or completion > MAX_COMPLETION_TOKENS:
        raise ValueError("provider usage violates the frozen token bounds")
    return INPUT_RATE * (prompt - cached) + CACHED_RATE * cached + OUTPUT_RATE * completion, prompt, completion, cached


class MeteredTransport(httpx.AsyncBaseTransport):
    """Serialize and journal each actual HTTP request before the SDK sees it.

    The SDK is configured with zero retries. The guard still applies to every
    physical request, including calls from browser-use's extraction/judge tools.
    A crash after request intent is durable is ambiguous and blocks new calls;
    the reservation is never silently treated as zero cost.
    """

    def __init__(self, path: Path, max_usd: float, *, transport=None, model: str = MODEL):
        self.path = Path(path)
        self.max_usd = Decimal(str(max_usd))
        if not Decimal("0") < self.max_usd <= Decimal("100"):
            raise ValueError("registered USD limit must be in (0, 100]")
        if model != MODEL:
            raise ValueError("transport pricing and bounds require the registered snapshot")
        self.model = model
        self.transport = transport if transport is not None else httpx.AsyncHTTPTransport(retries=0)
        self.lock = asyncio.Lock()
        self.episode_id: str | None = None
        self.budget_exhausted = False
        self.reload_accounting()

    def reload_accounting(self) -> None:
        """Load durable receipts while the controller's exclusive lock is held.

        Construction can precede lock acquisition. The first callback reloads
        here so taking over after another process cannot use stale spending.
        """
        self.rows: list[dict] = []
        if self.path.exists():
            # A torn or malformed record is an accounting ambiguity, never a
            # reason to truncate paid history and continue.
            for line in self.path.read_text().splitlines():
                self.rows.append(json.loads(line))
        self.pending: dict[str, dict] = {}
        self.spent = Decimal("0")
        self.counter = 0
        self.provider_failed = False
        for row in self.rows:
            self.counter = max(self.counter, int(row["request_sequence"]))
            request_id = row["request_id"]
            if row["event"] == "request_started":
                if request_id in self.pending:
                    raise ValueError("duplicate request intent")
                self.pending[request_id] = row
            elif row["event"] in ("request_completed", "request_rejected"):
                if request_id not in self.pending:
                    raise ValueError("request result without durable intent")
                del self.pending[request_id]
                self.spent += Decimal(str(row["cost_usd"]))
            elif row["event"] not in ("request_unknown", "invocation_failed"):
                raise ValueError("unknown meter journal event")
            if row.get("http_status", 0) >= 400 or row["event"] == "invocation_failed":
                self.provider_failed = True
        if self.spent + self.reserved > self.max_usd:
            raise ValueError("saved spend and reservations exceed the registered limit")

    @property
    def reserved(self) -> Decimal:
        return sum((Decimal(str(r["reserved_usd"])) for r in self.pending.values()), Decimal("0"))

    def _write(self, row: dict) -> None:
        _append_durable(self.path, row)
        self.rows.append(row)

    def record_invocation_error(self, exc: BaseException) -> None:
        """Latch errors occurring after a billed HTTP 200 but before parsing."""
        self.provider_failed = True
        self._write({"event": "invocation_failed", "timestamp_utc": utc_now(),
                     "request_id": f"request_{self.counter:06d}", "request_sequence": self.counter,
                     "episode_id": self.episode_id, "exception_type": type(exc).__name__})

    def episode_summary(self, episode_id: str) -> dict:
        rows = [row for row in self.rows if row.get("episode_id") == episode_id]
        completed = [row for row in rows if row["event"] == "request_completed"]
        return {
            "cost_usd": float(sum((Decimal(str(row["cost_usd"])) for row in completed), Decimal("0"))),
            "input_tokens": sum(row["input_tokens"] for row in completed),
            "output_tokens": sum(row["output_tokens"] for row in completed),
            "cache_read_tokens": sum(row["cache_read_tokens"] for row in completed),
            "api_requests": sum(row["event"] == "request_started" for row in rows),
            "api_responses": len(completed),
            "provider_http_errors": [row["http_status"] for row in rows if row.get("http_status", 0) >= 400],
            "provider_invocation_errors": [row["exception_type"] for row in rows
                                            if row["event"] == "invocation_failed"],
            "unknown_billing_requests": [row["request_id"] for row in self.pending.values()
                                          if row.get("episode_id") == episode_id],
            "unknown_billing_reserved_usd": float(sum((Decimal(str(row["reserved_usd"]))
                for row in self.pending.values() if row.get("episode_id") == episode_id), Decimal("0"))),
            "resolved_models": sorted({row["resolved_model"] for row in completed}),
            "system_fingerprints": sorted({row["system_fingerprint"] for row in completed
                                            if row.get("system_fingerprint") is not None}),
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self.lock:
            if self.pending:
                raise UnknownBilling("a request lacks a final usage receipt; paid work is halted")
            if self.provider_failed:
                raise ProviderEpisodeFailed("a provider error has halted this zero-retry experiment")
            if self.episode_id is None:
                raise RuntimeError("API request outside a registered episode")
            if (request.method != "POST" or request.url.scheme != "https"
                    or request.url.host != "api.openai.com" or request.url.path != "/v1/chat/completions"):
                raise ValueError("only the registered standard OpenAI chat endpoint is permitted")
            payload = json.loads(request.content)
            if (payload.get("model") != self.model
                    or payload.get("max_completion_tokens") != MAX_COMPLETION_TOKENS
                    or payload.get("reasoning_effort") != "low"
                    or payload.get("stream", False)
                    or payload.get("service_tier", "default") != "default"
                    or payload.get("n", 1) != 1
                    or payload.get("tools") or payload.get("functions")):
                raise ValueError("request differs from the frozen model, output, tier, or tool settings")
            if self.spent + REQUEST_RESERVE_USD > self.max_usd:
                self.budget_exhausted = True
                raise SpendingLimitReached("insufficient budget for the next request's conservative reserve")
            self.counter += 1
            request_id = f"request_{self.counter:06d}"
            common = {"request_id": request_id, "request_sequence": self.counter,
                      "episode_id": self.episode_id}
            intent = {**common, "event": "request_started", "timestamp_utc": utc_now(),
                      "reserved_usd": float(REQUEST_RESERVE_USD), "requested_model": self.model,
                      "max_completion_tokens": MAX_COMPLETION_TOKENS, "service_tier": "default",
                      "body_sha256": hashlib.sha256(request.content).hexdigest()}
            self._write(intent)
            self.pending[request_id] = intent
            try:
                response = await self.transport.handle_async_request(request)
                await response.aread()
                if response.status_code >= 400:
                    self.provider_failed = True
                    if response.status_code >= 500:
                        # Server failures can occur after inference started.
                        # Keep their full reserve and do not assert zero cost.
                        self._write({**common, "event": "request_unknown", "timestamp_utc": utc_now(),
                                     "http_status": response.status_code,
                                     "provider_request_id": response.headers.get("x-request-id"),
                                     "exception_type": "provider_server_error",
                                     "reserved_usd": float(REQUEST_RESERVE_USD)})
                        return response
                    row = {**common, "event": "request_rejected", "timestamp_utc": utc_now(),
                           "http_status": response.status_code, "cost_usd": 0.0,
                           "provider_request_id": response.headers.get("x-request-id"),
                           "billing_basis": "explicit HTTP rejection; no completion returned"}
                else:
                    body = response.json()
                    if body.get("model") != self.model:
                        raise ValueError("provider returned a model other than the registered snapshot")
                    cost, prompt, completion, cached = usage_cost(body.get("usage") or {})
                    if cost > REQUEST_RESERVE_USD:
                        raise ValueError("provider usage exceeds the reserved maximum")
                    row = {**common, "event": "request_completed", "timestamp_utc": utc_now(),
                           "http_status": response.status_code, "cost_usd": float(cost),
                           "input_tokens": prompt, "output_tokens": completion, "cache_read_tokens": cached,
                           "raw_usage": body["usage"], "resolved_model": body["model"],
                           "system_fingerprint": body.get("system_fingerprint"),
                           "provider_request_id": response.headers.get("x-request-id"),
                           "completion_id": body.get("id"),
                           "response_sha256": hashlib.sha256(response.content).hexdigest()}
                self._write(row)
                self.spent += Decimal(str(row["cost_usd"]))
                del self.pending[request_id]
                return response
            except BaseException as exc:
                self._write({**common, "event": "request_unknown", "timestamp_utc": utc_now(),
                             "exception_type": type(exc).__name__, "reserved_usd": float(REQUEST_RESERVE_USD)})
                raise

    async def aclose(self) -> None:
        await self.transport.aclose()


def latch_invocation_errors(llm, meter: MeteredTransport):
    """Guard browser-use's parse/envelope errors as well as HTTP errors.

    TokenCost wraps this method afterward. A successful HTTP response may be
    billed even when its structured output cannot be parsed; its usage remains
    in the transport ledger and another invocation is refused locally.
    """
    original = llm.ainvoke

    async def guarded(messages, output_format=None, **kwargs):
        if meter.provider_failed:
            raise ProviderEpisodeFailed("a provider invocation already failed; retries are disabled")
        try:
            return await original(messages, output_format=output_format, **kwargs)
        except BaseException as exc:
            meter.record_invocation_error(exc)
            raise

    object.__setattr__(llm, "ainvoke", guarded)
    return llm


class LiveEvaluator:
    """Callable real-agent evaluator accepted by the live stopping controller.

    Required config: model, max_usd, task_suite (absolute JSON path without its
    extension), and prompts keyed baseline/challenger with arm_id/text/sha256/
    vector. Remaining execution settings are validated against this protocol.
    """

    def __init__(self, config: dict, directory: Path):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        os.environ["ANONYMIZED_TELEMETRY"] = "false"
        os.environ["BROWSER_USE_CLOUD_SYNC"] = "false"
        self.config = config
        self.directory = Path(directory)
        if config["model"] != MODEL:
            raise ValueError("wrong registered model snapshot")
        for key, expected in (("max_steps", 30), ("timeout_s", 180), ("reasoning_effort", "low")):
            if config.get(key, expected) != expected:
                raise ValueError(f"wrong registered {key}")
        for package, version in config.get("packages", {}).items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"installed {package} version differs from registration")
        self.prompts = config["prompts"]
        for role in ("baseline", "challenger"):
            prompt = self.prompts[role]
            if hashlib.sha256(prompt["text"].encode()).hexdigest() != prompt["sha256"]:
                raise ValueError(f"{role} prompt does not match frozen hash")
        self.meter = MeteredTransport(self.directory / "api_calls.jsonl", config["max_usd"])
        self.client = httpx.AsyncClient(transport=self.meter, timeout=httpx.Timeout(65.0))
        self.adapter = WebArenaInfinityAdapter(
            web_app=config.get("web_app", "apps/linear-account-settings"), task_suite=config["task_suite"],
            port=int(config.get("port", 8601)), use_vision=False, headless=True,
            timeout_s=180, llm_provider="openai", llm_model=MODEL, llm_reasoning_effort="low",
            artifacts_dir=str(self.directory / "artifacts"),
            axes_path=str(ROOT / "configs/axes.yaml"), template_path=str(ROOT / "configs/template.jinja"))
        self.ready = False
        self.completed_since_reset = 0
        self.started_evaluations = False

    def prepare(self) -> dict:
        """Start/reset the local app and browser. This method makes no LLM call."""
        if self.meter.pending:
            raise UnknownBilling("cannot resume with an unreceipted API request")
        if self.meter.provider_failed:
            raise ProviderEpisodeFailed("cannot resume after a provider rejection")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY unavailable")
        if not self.ready:
            # The upstream start_server kills anything listening on its port.
            # Refuse before entering that helper if the registered port is busy.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
                if port_probe.connect_ex(("127.0.0.1", self.adapter._port)) == 0:
                    raise RuntimeError("registered app port is occupied; refusing to stop another process")
            collect._recover(self.adapter)
            self.ready = True
            self.completed_since_reset = 0
        for prompt in self.prompts.values():
            self.adapter.register_prompt(prompt["arm_id"], prompt["text"])
        return {"ready": True, "task_count": len(self.adapter.task_ids()), "llm_calls": self.meter.counter}

    def __call__(self, task: dict, role: str, episode_id: str) -> dict:
        from browser_use.llm.openai.chat import ChatOpenAI

        if role not in ("baseline", "challenger") or not episode_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid episode role or id")
        prompt = self.prompts[role]
        receipt_path = self.directory / "episodes" / episode_id / "receipt.json"
        identity = {"episode_id": episode_id, "task_id": task["task_id"], "source_task_id": task["source_task_id"],
                    "role": role, "model": MODEL, "prompt_sha256": prompt["sha256"]}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if any(receipt.get(key) != value for key, value in identity.items()):
                raise ValueError("saved episode receipt identity does not match request")
            return receipt
        intent_path = receipt_path.with_name("intent.json")
        if intent_path.exists():
            raise UnknownBilling("episode intent exists without a final receipt; refusing duplicate paid execution")
        if not self.started_evaluations:
            self.meter.reload_accounting()
            self.started_evaluations = True
        self.prepare()
        if self.completed_since_reset >= collect.RECYCLE_EVERY:
            self.adapter.close()
            self.ready = False
            self.prepare()
        # Use a fresh LLM object: browser-use decorates ainvoke for accounting,
        # so sharing an LLM across Agent objects nests previous token services.
        self.adapter._agent.llm = latch_invocation_errors(ChatOpenAI(
            model=MODEL, reasoning_effort="low", max_completion_tokens=MAX_COMPLETION_TOKENS,
            max_retries=0, service_tier="default", http_client=self.client,
            base_url="https://api.openai.com/v1", timeout=65.0), self.meter)
        self.adapter._agent._last_token_summary = {}
        self.meter.episode_id = episode_id
        started = utc_now()
        start_clock = time.monotonic()
        _atomic_json(intent_path, {**identity, "event": "episode_started", "timestamp_utc": started,
                                  "instruction_sha256": hashlib.sha256(task["instruction"].encode()).hexdigest()})
        arm = Arm(prompt["arm_id"], prompt["arm_id"], PromptVector(**prompt["vector"]))
        raw_task = {**task, "id": task["task_id"]}
        wrapped_task = Task(task_id=task["task_id"], payload={"instruction": task["instruction"]},
                            metadata={"raw": raw_task, "web_app_dir": self.adapter._web_app_abs})
        result, exception = None, None
        try:
            result = self.adapter.run_arm(arm, wrapped_task, None, max_steps=30,
                                          artifact_subdir=episode_id)
        except Exception as exc:
            exception = exc
        finally:
            self.meter.episode_id = None
        usage = self.meter.episode_summary(episode_id)
        status, reason = collect.classify_detail(result, exception)
        status = {"ok": "ok", "infra_error": "infrastructure_error"}[status]
        if reason == collect.INFRA_PROVIDER:
            status = "provider_error"
        trace = {} if result is None else result.trace or {}
        if str(trace.get("verifier_message", "")).startswith("Verifier exception:"):
            status = "infrastructure_error"
            reason = "verifier_exception"
        if (usage["provider_http_errors"] or usage["provider_invocation_errors"]
                or (exception is not None and collect._is_provider_error(str(exception)))):
            status, reason = "provider_error", "provider_exception"
        if self.meter.budget_exhausted:
            status, reason = "budget_exhausted", "usd_request_reserve"
        if usage["unknown_billing_requests"]:
            status = "provider_error" if usage["provider_http_errors"] else "infrastructure_error"
            reason = "unknown_billing"
        if status == "ok" and usage["api_responses"] == 0:
            status, reason = "provider_error", "no_successful_provider_response"
        receipt = {**identity, **usage, "status": status,
                   "success": int(bool(result.success)) if status == "ok" and result is not None else None,
                   "infra_reason": reason, "started_utc": started, "finished_utc": utc_now(),
                   "elapsed_wallclock_s": time.monotonic() - start_clock,
                   "steps": None if result is None else result.steps,
                   "agent_wallclock_s": None if result is None else result.wallclock_s,
                   "timed_out": bool(trace.get("timed_out", False)), "trace": trace,
                   "exception_type": None if exception is None else type(exception).__name__,
                   "cost_accounting_complete": not bool(usage["unknown_billing_requests"]),
                   "cost_basis": "provider response usage at frozen standard API list prices",
                   "sdk_cost_usd": None if result is None else (result.tokens or {}).get("cost_usd")}
        _atomic_json(receipt_path, receipt)
        self.completed_since_reset += 1
        return receipt

    def close(self) -> None:
        try:
            self.adapter.close()
        finally:
            self.ready = False
            _loop().run_until_complete(self.client.aclose())


def main(argv: list[str] | None = None) -> int:
    from experiments.growing_bandits.empirical import live_stopping

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true", help="execute the frozen paid experiment")
    action.add_argument("--preflight", action="store_true", help="check app/browser startup without inference")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.config.with_name("manifest.json").read_text())
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("configuration differs from the frozen manifest")
    task_path = Path(config["task_suite"] + ".json")
    if hashlib.sha256(task_path.read_bytes()).hexdigest() != manifest["tasks_sha256"]:
        raise ValueError("task suite differs from the frozen manifest")
    live_stopping.verify_frozen_files(config)
    tasks = json.loads(task_path.read_text())
    evaluator = LiveEvaluator(config, args.log_dir)
    try:
        if args.preflight:
            print(json.dumps(evaluator.prepare(), indent=2))
        else:
            state = live_stopping.run(config, args.log_dir, tasks, evaluator)
            print(json.dumps(state, indent=2))
    finally:
        evaluator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
