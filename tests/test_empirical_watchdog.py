"""The watchdog's decision rule, isolated from processes and clocks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

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


def test_kill_and_wait_process_escalates_to_sigkill(monkeypatch):
    """Stalled process: SIGTERM times out, escalates to SIGKILL."""
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 12345

    killpg_calls = []
    def mock_killpg(pid, sig):
        killpg_calls.append((pid, sig))

    monkeypatch.setattr("os.killpg", mock_killpg)

    wait_calls = []
    def mock_wait(timeout=None):
        wait_calls.append(timeout)
        if len(wait_calls) == 1:
            raise subprocess.TimeoutExpired("cmd", timeout)
        return None

    mock_proc.wait = mock_wait

    wd._kill_and_wait_process(mock_proc)

    assert killpg_calls == [(12345, 15), (12345, 9)]  # SIGTERM=15, SIGKILL=9
    assert len(wait_calls) == 2
    assert wait_calls[0] == 60
    assert wait_calls[1] == 30


def test_kill_and_wait_process_tolerates_process_lookup_error(monkeypatch):
    """SIGTERM on already-gone group raises ProcessLookupError, is tolerated."""
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 12345

    def mock_killpg(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr("os.killpg", mock_killpg)

    wd._kill_and_wait_process(mock_proc)

    mock_proc.wait.assert_not_called()


def test_main_loop_detects_done_status(tmp_path, monkeypatch):
    """Main loop: STATUS=done returns 0 without relaunching."""
    log_dir = tmp_path / "logs" / "empirical_pool"
    log_dir.mkdir(parents=True)

    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None  # Alive

    monkeypatch.setattr(wd, "LOG_DIR", log_dir)
    monkeypatch.setattr("time.sleep", lambda x: None)

    times = iter([0.0, 0.0, 1.0])  # Start, then first poll, then done check
    monkeypatch.setattr("time.time", lambda: next(times))

    monkeypatch.setattr(wd, "line_count", lambda log_dir: 0)
    monkeypatch.setattr(wd, "_launch", lambda extra, log_dir: mock_proc)
    monkeypatch.setattr(wd, "_status", lambda log_dir: "done")

    result = wd.main(["--workers", "8"])

    assert result == 0
    mock_proc.poll.assert_called()


def test_main_loop_max_relaunches_writes_gave_up(tmp_path, monkeypatch):
    """Main loop: after MAX_RELAUNCHES, writes WATCHDOG_GAVE_UP and returns 1."""
    log_dir = tmp_path / "logs" / "empirical_pool"
    log_dir.mkdir(parents=True)

    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None  # Always alive
    mock_proc.pid = 999

    monkeypatch.setattr(wd, "LOG_DIR", log_dir)
    monkeypatch.setattr("time.sleep", lambda x: None)

    time_counter = [0.0]
    def fake_time():
        time_counter[0] += wd.STALL_MINUTES + 100  # Always stalled
        return time_counter[0]
    monkeypatch.setattr("time.time", fake_time)

    monkeypatch.setattr(wd, "line_count", lambda log_dir: 0)
    monkeypatch.setattr(wd, "_launch", lambda extra, log_dir: mock_proc)
    monkeypatch.setattr(wd, "_status", lambda log_dir: "running")
    monkeypatch.setattr(wd, "_kill_and_wait_process", lambda proc: None)

    result = wd.main(["--workers", "8"])

    assert result == 1
    assert (log_dir / "WATCHDOG_GAVE_UP").exists()
    gave_up_text = (log_dir / "WATCHDOG_GAVE_UP").read_text()
    assert "20 relaunches" in gave_up_text


def test_main_loop_stalled_process_triggers_kill_and_relaunch(tmp_path, monkeypatch):
    """Main loop: stalled live process triggers kill_and_wait and relaunches."""
    log_dir = tmp_path / "logs" / "empirical_pool"
    log_dir.mkdir(parents=True)

    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None  # Alive
    mock_proc.pid = 999

    monkeypatch.setattr(wd, "LOG_DIR", log_dir)
    monkeypatch.setattr("time.sleep", lambda x: None)

    # Simulate time passing: no progress for > 30 min on second iteration
    time_values = [0.0, 0.0, wd.STALL_MINUTES * 60 + 100, wd.STALL_MINUTES * 60 + 100, 0.0]
    time_iter = iter(time_values)
    monkeypatch.setattr("time.time", lambda: next(time_iter))

    monkeypatch.setattr(wd, "line_count", lambda log_dir: 0)  # No progress

    kill_calls = []
    def mock_kill_and_wait(proc):
        kill_calls.append(proc)
    monkeypatch.setattr(wd, "_kill_and_wait_process", mock_kill_and_wait)

    # After first relaunch due to stall, return a different mock that shows done
    monkeypatch.setattr(wd, "_launch", lambda extra, log_dir: mock_proc)

    # After kill_and_relaunch, status becomes "done"
    status_calls = [0]
    def mock_status(log_dir):
        status_calls[0] += 1
        return "done" if status_calls[0] > 2 else "running"
    monkeypatch.setattr(wd, "_status", mock_status)

    result = wd.main(["--workers", "8"])

    assert result == 0
    assert len(kill_calls) == 1  # Should have called kill_and_wait once
