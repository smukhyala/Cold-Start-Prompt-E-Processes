"""The watchdog's decision rule, isolated from processes and clocks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import watchdog_empirical_pool as wd  # noqa: E402


@pytest.mark.parametrize("status, alive, minutes, expected", [
    ("done", False, 0.0, "finished"),
    ("budget", False, 0.0, "finished"),
    ("running", True, 5.0, "wait"),
    ("running", True, wd.STALL_MINUTES + 1, "kill_and_relaunch"),
    ("running", False, 1.0, "relaunch"),
    ("failed", False, 1.0, "relaunch"),
    (None, False, 0.0, "relaunch"),
])
def test_decide(status, alive, minutes, expected):
    assert wd.decide(status, alive, minutes) == expected


def test_line_count_counts_every_worker(tmp_path):
    (tmp_path / "worker_0.jsonl").write_text("{}\n{}\n")
    (tmp_path / "worker_1.jsonl").write_text("{}\n")
    assert wd.line_count(tmp_path) == 3
