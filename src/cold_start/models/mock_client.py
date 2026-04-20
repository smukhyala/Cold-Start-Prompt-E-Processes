"""Deterministic mock client for tests. Never calls the network."""

from __future__ import annotations

from typing import Any

from cold_start.models.base import ModelClient, ModelResponse
from cold_start.registry import register


@register("mock", kind="model")
class MockClient(ModelClient):
    def __init__(self, reply: str = "OK", stop_reason: str = "end_turn") -> None:
        self.reply = reply
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.calls.append({"system": system_prompt, "messages": messages, "tools": tools})
        return ModelResponse(
            content=self.reply,
            stop_reason=self.stop_reason,
            usage={"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
        )
