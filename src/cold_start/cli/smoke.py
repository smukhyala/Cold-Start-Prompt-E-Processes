"""One-call verification of Claude Opus 4.7 with prompt caching on the system block."""

from __future__ import annotations

import sys
import time

from cold_start.cli import _bootstrap  # noqa: F401
from cold_start.models.anthropic_client import AnthropicClient


def main() -> int:
    client = AnthropicClient(id="claude-opus-4-7", max_tokens=64)
    # Prompt caching on Opus has a 1024-token minimum, so pad to a realistic
    # cold-start system prompt length. Content is irrelevant to the check.
    filler = (
        "You are an expert research assistant specializing in rigorous, "
        "step-by-step analysis. You value clarity, precision, and calibrated "
        "uncertainty. Before answering you consider alternative hypotheses, "
        "check your reasoning for common failure modes, and state assumptions "
        "explicitly. You avoid speculation beyond what the evidence supports. "
    ) * 40
    system = filler + "\n\nWhen asked to say 'ok', respond with exactly one word: ok."
    msg = [{"role": "user", "content": "Say ok."}]

    first = client.call(system, msg)
    time.sleep(2)
    second = client.call(system, msg)

    print(f"first  stop_reason={first.stop_reason} content={first.content!r}")
    print(f"first  usage={first.usage}")
    print(f"second stop_reason={second.stop_reason} content={second.content!r}")
    print(f"second usage={second.usage}")

    cache_read = second.usage.get("cache_read_input_tokens", 0)
    if cache_read <= 0:
        print("WARN: no cache hit on second call (cache_read_input_tokens=0)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
