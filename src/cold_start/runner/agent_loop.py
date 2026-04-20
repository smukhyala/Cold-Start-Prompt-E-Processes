"""Minimal tool-using agent loop.

Drives a ModelClient through repeated (tool_use, tool_result) rounds until the
model issues `end_turn` or `max_steps` is reached. Env adapters that need
specific tool schemas and handlers pass them in; ToyEnv doesn't use this path.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from cold_start.models.base import ModelClient
from cold_start.types import RunResult, Task


def run_agent_once(
    client: ModelClient,
    system_prompt: str,
    task: Task,
    tools: list[dict[str, Any]] | None,
    tool_handler: Callable[[dict[str, Any]], Any] | None,
    max_steps: int,
    initial_user_text: str | None = None,
    success_predicate: Callable[[str], bool] | None = None,
) -> RunResult:
    """Run one (arm, task) rollout. Returns RunResult with usage totals aggregated."""
    messages: list[dict[str, Any]] = []
    if initial_user_text is None:
        initial_user_text = _default_user_text(task)
    messages.append({"role": "user", "content": initial_user_text})

    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    last_text = ""
    steps = 0
    start = time.monotonic()

    for step in range(max_steps):
        steps = step + 1
        resp = client.call(system_prompt=system_prompt, messages=messages, tools=tools)

        tokens["input"] += int(resp.usage.get("input_tokens", 0))
        tokens["output"] += int(resp.usage.get("output_tokens", 0))
        tokens["cache_read"] += int(resp.usage.get("cache_read_input_tokens", 0))
        tokens["cache_write"] += int(resp.usage.get("cache_creation_input_tokens", 0))

        last_text = resp.content or last_text

        if resp.tool_calls and tool_handler is not None:
            assistant_blocks: list[dict[str, Any]] = []
            if resp.content:
                assistant_blocks.append({"type": "text", "text": resp.content})
            for tc in resp.tool_calls:
                assistant_blocks.append(
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                )
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_result_blocks: list[dict[str, Any]] = []
            for tc in resp.tool_calls:
                result = tool_handler(tc)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": str(result),
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        break

    elapsed = time.monotonic() - start
    success = bool(success_predicate(last_text)) if success_predicate else False
    return RunResult(
        success=success,
        reward=float(success),
        steps=steps,
        wallclock_s=elapsed,
        trace={"final_text": last_text[:500]},
        tokens=tokens,
    )


def _default_user_text(task: Task) -> str:
    parts: list[str] = []
    if "instruction" in task.payload:
        parts.append(str(task.payload["instruction"]))
    else:
        parts.append(f"Task id: {task.task_id}")
        if task.payload:
            parts.append(f"Payload: {task.payload!r}")
    return "\n".join(parts)
