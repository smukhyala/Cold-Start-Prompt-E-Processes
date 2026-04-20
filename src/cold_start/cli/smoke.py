"""One-call verification of Claude Opus 4.7 with prompt caching on the system block."""

from __future__ import annotations

import sys

from cold_start.cli import _bootstrap  # noqa: F401
from cold_start.models.anthropic_client import AnthropicClient


def main() -> int:
    client = AnthropicClient(id="claude-opus-4-7", max_tokens=64, temperature=0.0)
    system = "You are a concise assistant. Respond with exactly one word: ok."
    msg = [{"role": "user", "content": "Say ok."}]

    first = client.call(system, msg)
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
