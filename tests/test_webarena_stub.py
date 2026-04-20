from __future__ import annotations

import pytest

from cold_start.tasks.webarena import WebArenaInfinityAdapter


def test_webarena_stub_instantiable_and_raises():
    env = WebArenaInfinityAdapter()
    env.reset(0)  # no-op is fine
    with pytest.raises(NotImplementedError):
        env.sample_task(1)
