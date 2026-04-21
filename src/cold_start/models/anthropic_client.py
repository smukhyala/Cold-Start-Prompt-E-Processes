"""Anthropic client wrapper for Claude Opus 4.7 with prompt caching on system block.

The cold-start system prompt is byte-identical across every call that uses the
same arm (see `prompts/template.py`). Attaching `cache_control={"type":
"ephemeral"}` to the system block means every call after the first per-arm hits
the cache on the system tokens, which is the dominant cost saving.
"""

from __future__ import annotations

import os
import time
from typing import Any

from cold_start.models.base import ModelClient, ModelResponse
from cold_start.registry import register

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover
    anthropic = None


@register("anthropic", kind="model")
class AnthropicClient(ModelClient):
    def __init__(
        self,
        id: str = "claude-opus-4-7",
        max_tokens: int = 1024,
        temperature: float | None = None,
        prompt_cache: bool = True,
        max_retries: int = 5,
        initial_backoff_s: float = 1.0,
        params: dict[str, Any] | None = None,
    ) -> None:
        if anthropic is None:
            raise RuntimeError("anthropic SDK not installed; `pip install anthropic`")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt_cache = prompt_cache
        self.max_retries = max_retries
        self.initial_backoff_s = initial_backoff_s
        self.extra_params = dict(params or {})

    def call(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        if self.prompt_cache:
            system_blocks = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_blocks = [{"type": "text", "text": system_prompt}]

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
        kwargs.update(self.extra_params)

        response = self._call_with_retries(kwargs)
        return self._to_model_response(response)

    def _call_with_retries(self, kwargs: dict[str, Any]):
        backoff = self.initial_backoff_s
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(**kwargs)
            except Exception as e:  # pragma: no cover — network path
                last_err = e
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"unreachable: {last_err}")

    @staticmethod
    def _to_model_response(resp: Any) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}),
                    }
                )

        usage_obj = getattr(resp, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            for k in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                v = getattr(usage_obj, k, None)
                if v is not None:
                    usage[k] = int(v)

        return ModelResponse(
            content="".join(text_parts),
            stop_reason=getattr(resp, "stop_reason", "") or "",
            usage=usage,
            tool_calls=tool_calls,
            raw=resp,
        )
