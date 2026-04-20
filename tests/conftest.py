from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Ensure registries are populated before any test that looks components up by name.
from cold_start.cli import _bootstrap  # noqa: F401


CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def axes_path() -> Path:
    return CONFIGS / "axes.yaml"


@pytest.fixture
def arms_path() -> Path:
    return CONFIGS / "arms_initial.yaml"


@pytest.fixture
def template_path() -> Path:
    return CONFIGS / "template.jinja"


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d
