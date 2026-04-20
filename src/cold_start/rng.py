"""Seeded RNG helpers. One numpy Generator threaded through the orchestrator."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a PCG64 Generator with an explicit seed for reproducibility."""
    return np.random.default_rng(seed)


def spawn(rng: np.random.Generator, tag: str) -> np.random.Generator:
    """Derive an independent substream from `rng` using a string tag.

    Useful when a subsystem needs its own seed sequence without coupling to the
    parent generator's draw order.
    """
    tag_seed = int.from_bytes(tag.encode()[:8].ljust(8, b"\0"), "little") & 0xFFFFFFFF
    base = int(rng.integers(0, 2**32 - 1))
    return np.random.default_rng(base ^ tag_seed)
