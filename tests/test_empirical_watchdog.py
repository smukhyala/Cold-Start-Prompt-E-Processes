"""The watchdog's decision rule, isolated from processes and clocks."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import watchdog_empirical_pool as wd  # noqa: E402

#: Every mock `Popen.pid` any test in this file uses. `_guard_real_signals` (below) treats a
#: call on any other pid as a bug, not a silent no-op: a real, unaccounted-for pid must never
#: quietly reach this guard.
_KNOWN_MOCK_PIDS = {555, 999, 4242, 12345}


@pytest.fixture(autouse=True)
def _guard_real_signals(monkeypatch):
    """Fix round 1, item 1: several tests exercise the real `_ensure_previous_dead` (added for
    amendment 1's relaunch hygiene) without mocking it away, and that function's whole job is
    to call `os.killpg`. Without this fixture those tests sent real `os.killpg(<mock pid>,
    SIGKILL)` to the host. Every test in this file now runs with `os.killpg`/`os.kill` replaced
    by a recorder that never reaches the real syscall; a test that wants to observe kill calls
    reads `wd.os.killpg`/`wd.os.kill` back out (or installs its own, more specific, mock, which
    simply overrides this one for that test). A pid outside `_KNOWN_MOCK_PIDS` raises instead of
    silently doing nothing, so a stray real pid can never pass as a no-op.
    """
    def make_guard(name):
        def guard(pid, sig):
            if pid not in _KNOWN_MOCK_PIDS:
                raise AssertionError(f"real {name}({pid!r}, {sig!r}) called on an unrecognized pid -- "
                                     "this file must never reach a real syscall; add pid to "
                                     "_KNOWN_MOCK_PIDS only if it really is one of this file's mock pids")
        return guard

    monkeypatch.setattr(wd.os, "killpg", make_guard("os.killpg"))
    monkeypatch.setattr(wd.os, "kill", make_guard("os.kill"))


@pytest.mark.parametrize("status, alive, minutes, expected", [
    ("done", False, 0.0, "finished"),
    ("budget", False, 0.0, "finished"),
    ("running", True, 5.0, "wait"),
    ("running", True, wd.STALL_MINUTES + 1, "kill_and_relaunch"),
    ("running", False, 1.0, "relaunch"),
    ("failed", False, 1.0, "relaunch"),
    (None, False, 0.0, "relaunch"),
    # Amendment 1: `collect.py --archive-out-of-queue` leaves STATUS `paused`; a paused, dead
    # run is treated like any other dead non-finished run (relaunch), not mistaken for finished.
    ("paused", False, 1.0, "relaunch"),
    ("paused", True, 5.0, "wait"),
    ("some_unknown_status", False, 1.0, "relaunch"),
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


# ---- final-review fix wave: per-worker staleness, launching, provider_down, --budget ---------


@pytest.mark.parametrize("status, alive, stale, expected", [
    ("provider_down", False, (), "finished"),
    ("provider_down", True, (), "finished"),
    ("launching", True, (), "wait"),
    ("launching", False, (), "relaunch"),
    ("running", True, (3,), "kill_and_relaunch"),
    ("launching", True, (3,), "wait"),  # workers are not judged until the collector says it is running
])
def test_decide_new_states(status, alive, stale, expected):
    assert wd.decide(status, alive, 1.0, stale) == expected


def test_worker_clocks():
    limit = wd.WORKER_STALL_MINUTES * 60.0
    clocks = wd.WorkerClocks(3, now=0.0)
    clocks.observe({0: None, 1: None, 2: None}, now=10.0)
    assert clocks.stale(limit - 1, exempt=set()) == ()  # no file yet: fresh until up for the limit
    assert clocks.stale(limit + 1, exempt=set()) == (0, 1, 2)
    clocks.observe({0: 100, 1: None, 2: 50}, now=60.0)
    clocks.observe({0: 200, 1: None, 2: 50}, now=600.0)  # worker 2 stops growing at t=60
    assert clocks.stale(60.0 + limit + 1, exempt=set()) == (1, 2)
    assert clocks.stale(60.0 + limit + 1, exempt={2}) == (1,)
    clocks.restart(now=5000.0)
    assert clocks.stale(5000.0 + limit - 1, exempt=set()) == ()


def test_finished_workers_exempts_only_clean_exits(tmp_path):
    (tmp_path / "worker_0.exit").write_text("done\n")
    (tmp_path / "worker_1.exit").write_text("failed\n")
    (tmp_path / "worker_2.exit").write_text("provider_down\n")
    (tmp_path / "worker_3.exit").write_text("budget\n")
    assert wd.finished_workers(tmp_path) == {0, 2, 3}


def test_worker_sizes(tmp_path):
    (tmp_path / "worker_1.jsonl").write_text("{}\n")
    assert wd.worker_sizes(tmp_path, 3) == {0: None, 1: 3, 2: None}


class _Clock:
    """time.time/time.sleep for the loop: each sleep advances the clock."""

    def __init__(self):
        self.t = 1_000.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _loop_env(tmp_path, monkeypatch, procs):
    log_dir = tmp_path / "logs" / "empirical_pool"
    log_dir.mkdir(parents=True)
    clock = _Clock()
    monkeypatch.setattr(wd, "LOG_DIR", log_dir)
    monkeypatch.setattr("time.time", clock.time)
    monkeypatch.setattr("time.sleep", clock.sleep)
    launches = []

    def launch(extra, log_dir_):
        launches.append({"extra": list(extra), "status_at_launch": (log_dir_ / "STATUS").read_text().strip()})
        return procs[min(len(launches), len(procs)) - 1]

    monkeypatch.setattr(wd, "_launch", launch)
    kills = []
    monkeypatch.setattr(wd, "_kill_and_wait_process", lambda proc: kills.append(proc))
    return log_dir, clock, launches, kills


def _proc(alive: bool):
    p = mock.Mock(spec=subprocess.Popen)
    p.poll.return_value = None if alive else 1
    p.pid = 4242
    return p


def test_one_hung_worker_triggers_a_relaunch_and_a_machine_readable_line(tmp_path, monkeypatch):
    alive, finisher = _proc(True), _proc(True)
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [alive, finisher])
    grow = {"n": 0}

    def status(log_dir_):
        if len(launches) > 1:
            return "done"
        return "running"

    monkeypatch.setattr(wd, "_status", status)
    real_sizes = wd.worker_sizes

    def sizes(log_dir_, n):
        grow["n"] += 1
        out = real_sizes(log_dir_, n)
        out[0] = grow["n"]  # worker 0 keeps writing, so the global line count never stalls
        out[1] = 7          # worker 1 wrote once and hung
        return out

    monkeypatch.setattr(wd, "worker_sizes", sizes)
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: grow["n"])
    assert wd.main(["--workers", "2", "--pilot"]) == 0
    assert kills == [alive]
    lines = (log_dir / wd.RELAUNCH_LOG).read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("RELAUNCH 1 stale_workers:1 mode=pilot at=")
    assert [x["status_at_launch"] for x in launches] == ["launching", "launching"]


def test_a_finished_worker_is_not_mistaken_for_a_hung_one(tmp_path, monkeypatch):
    proc = _proc(True)
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [proc])
    polls = {"n": 0}

    def status(log_dir_):
        polls["n"] += 1
        return "done" if polls["n"] > 20 else "running"  # 20 polls x 2 min = 40 min

    def sizes(log_dir_, n):
        return {0: polls["n"], 1: 5}

    monkeypatch.setattr(wd, "_status", status)
    monkeypatch.setattr(wd, "worker_sizes", sizes)
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: polls["n"])
    real_prepare = wd._prepare_launch

    def prepare(log_dir_):
        real_prepare(log_dir_)
        (log_dir_ / "worker_1.exit").write_text("done\n")  # the collector marks worker 1 finished

    monkeypatch.setattr(wd, "_prepare_launch", prepare)
    assert wd.main(["--workers", "2"]) == 0
    assert kills == [] and len(launches) == 1


def test_a_stale_done_status_from_the_pilot_cannot_end_the_next_launch(tmp_path, monkeypatch):
    """I5: STATUS still says done from the pilot; the new collector dies before writing running."""
    dead, finisher = _proc(False), _proc(True)
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [dead, finisher])
    (log_dir / "STATUS").write_text("done\n")
    calls = {"n": 0}
    real_status = wd._status

    def status(log_dir_):
        calls["n"] += 1
        if len(launches) > 1 and calls["n"] > 2:
            return "done"
        return real_status(log_dir_)  # the file the watchdog itself wrote: "launching"

    monkeypatch.setattr(wd, "_status", status)
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)
    assert wd.main(["--workers", "1"]) == 0
    assert len(launches) == 2  # relaunched, not "finished" at the first poll
    assert (log_dir / wd.RELAUNCH_LOG).read_text().startswith("RELAUNCH 1 collector_exited:launching mode=full")


def test_provider_down_is_terminal_without_a_relaunch(tmp_path, monkeypatch, capsys):
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [_proc(False)])
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "provider_down")
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)
    assert wd.main(["--workers", "8"]) == 2
    assert len(launches) == 1 and kills == []
    assert "provider_down" in capsys.readouterr().out
    assert (log_dir / wd.RELAUNCH_LOG).read_text() == ""


def test_budget_is_forwarded_to_the_collector(tmp_path, monkeypatch):
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [_proc(False)])
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "done")
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)
    assert wd.main(["--pilot", "--budget", "40", "--workers", "8"]) == 0
    assert launches[0]["extra"] == ["--workers", "8", "--budget", "40.0", "--pilot"]
    assert wd.collector_args(8, False, None) == ["--workers", "8"]


# ---- amendment 1: relaunch hygiene (a previous browser-use worker can survive its leader) -----


def test_ensure_previous_dead_sigkills_the_process_group_and_waits(monkeypatch):
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 555
    killpg_calls = []
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    wait_calls = []
    mock_proc.wait = lambda timeout=None: wait_calls.append(timeout)

    wd._ensure_previous_dead(mock_proc)

    assert killpg_calls == [(555, signal.SIGKILL)]
    assert len(wait_calls) == 1 and wait_calls[0] is not None


def test_ensure_previous_dead_is_a_noop_with_no_previous_process(monkeypatch):
    killpg_calls = []
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    wd._ensure_previous_dead(None)
    assert killpg_calls == []


@pytest.mark.parametrize("exc", [ProcessLookupError(), PermissionError()])
def test_ensure_previous_dead_tolerates_killpg_errors(monkeypatch, exc):
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 555

    def boom(pid, sig):
        raise exc

    monkeypatch.setattr(wd.os, "killpg", boom)
    wd._ensure_previous_dead(mock_proc)  # must not raise
    mock_proc.wait.assert_not_called()


def test_ensure_previous_dead_tolerates_a_wait_timeout(monkeypatch):
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 555
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: None)

    def wait(timeout=None):
        raise subprocess.TimeoutExpired("cmd", timeout)

    mock_proc.wait = wait
    wd._ensure_previous_dead(mock_proc)  # must not raise


# ---- fix round 1, item 4: kill orphans of a PREVIOUS watchdog session before the first launch --


def test_kill_orphaned_lock_holder_is_a_noop_with_no_lock_file(tmp_path, monkeypatch):
    killpg_calls = []
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    wd._kill_orphaned_lock_holder(tmp_path)
    assert killpg_calls == [] and sleeps == []


def test_kill_orphaned_lock_holder_is_a_noop_with_unparseable_lock_contents(tmp_path, monkeypatch):
    (tmp_path / "collect.lock").write_text("not-a-pid")
    killpg_calls = []
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    wd._kill_orphaned_lock_holder(tmp_path)
    assert killpg_calls == []


def test_kill_orphaned_lock_holder_sigkills_a_live_group_and_waits(tmp_path, monkeypatch):
    (tmp_path / "collect.lock").write_text("7777")
    killpg_calls = []
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    wd._kill_orphaned_lock_holder(tmp_path)

    assert killpg_calls == [(7777, signal.SIGKILL)]
    assert sleeps == [wd.ORPHAN_KILL_WAIT_S]  # bounded wait after a real kill


@pytest.mark.parametrize("exc", [ProcessLookupError(), PermissionError()])
def test_kill_orphaned_lock_holder_tolerates_killpg_errors_and_does_not_wait(tmp_path, monkeypatch, exc):
    (tmp_path / "collect.lock").write_text("7777")

    def boom(pid, sig):
        raise exc

    monkeypatch.setattr(wd.os, "killpg", boom)
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    wd._kill_orphaned_lock_holder(tmp_path)  # must not raise

    assert sleeps == []  # nothing left alive (or nothing we could touch): no need to wait


def test_first_launch_kills_an_orphaned_lock_holder_from_a_previous_watchdog_session(tmp_path, monkeypatch):
    """A `collect.lock` naming a still-alive process group, left over from an earlier watchdog
    *process* (this one just started, so `_ensure_previous_dead`'s in-memory `proc` is
    unavailable), must be SIGKILLed before the very first `_launch`."""
    log_dir = tmp_path / "logs" / "empirical_pool"
    log_dir.mkdir(parents=True)
    (log_dir / "collect.lock").write_text("7777")

    clock = _Clock()
    monkeypatch.setattr(wd, "LOG_DIR", log_dir)
    monkeypatch.setattr("time.time", clock.time)
    monkeypatch.setattr("time.sleep", clock.sleep)

    killpg_calls = []
    monkeypatch.setattr(wd.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

    events: list[str] = []
    proc = _proc(True)

    def launch_spy(extra, log_dir_):
        events.append("launch")
        return proc

    monkeypatch.setattr(wd, "_launch", launch_spy)
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "done")  # finish right after the first poll
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)

    assert wd.main(["--workers", "1"]) == 0

    assert killpg_calls == [(7777, signal.SIGKILL)]
    assert events == ["launch"]  # the orphan kill ran before this, not instead of it


def test_relaunch_hygiene_on_the_dead_relaunch_path(tmp_path, monkeypatch):
    """A collector that already exited (the 'dead' -> 'relaunch' path): before the very first
    launch there is nothing to kill, and before the relaunch that follows a dead collector, any
    leftover process group from the previous launch is SIGKILLed first."""
    dead, finisher = _proc(False), _proc(True)
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [dead, finisher])
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "done" if len(launches) > 1 else "running")
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)

    ensure_calls = []
    monkeypatch.setattr(wd, "_ensure_previous_dead", lambda proc: ensure_calls.append(proc))

    assert wd.main(["--workers", "1"]) == 0

    assert ensure_calls == [None, dead]
    assert len(launches) == 2


def test_relaunch_hygiene_on_the_kill_and_relaunch_path(tmp_path, monkeypatch):
    """A live but globally stalled collector (the 'kill_and_relaunch' path): the ordinary
    SIGTERM/SIGKILL escalation (`_kill_and_wait_process`) still runs, and the new hygiene check
    SIGKILLs the process group again before the next launch (it may have left children behind)."""
    stuck, finisher = _proc(True), _proc(True)
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [stuck, finisher])
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "done" if len(launches) > 1 else "running")
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)  # never grows: global stall

    def fast_sleep(s):
        clock.t += wd.STALL_MINUTES * 60 + 100

    monkeypatch.setattr("time.sleep", fast_sleep)

    ensure_calls = []
    monkeypatch.setattr(wd, "_ensure_previous_dead", lambda proc: ensure_calls.append(proc))

    assert wd.main(["--workers", "1"]) == 0

    assert kills == [stuck]  # the existing SIGTERM/SIGKILL escalation still ran
    assert ensure_calls == [None, stuck]  # ... and the hygiene check ran again before the relaunch
    assert len(launches) == 2


def test_ensure_previous_dead_runs_before_every_launch_in_order(tmp_path, monkeypatch):
    """Fix round 1, item 1: it is not enough that `_ensure_previous_dead` gets called
    somewhere -- it must run strictly before the `_launch` it guards, every time (first launch
    and every relaunch). Runs the real `_ensure_previous_dead` (not mocked away): safe only
    because `_guard_real_signals` (autouse, above) has already replaced `os.killpg`."""
    dead, finisher = _proc(False), _proc(True)
    log_dir, clock, launches, kills = _loop_env(tmp_path, monkeypatch, [dead, finisher])
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "done" if len(launches) > 1 else "running")
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)

    events: list[str] = []
    inner_launch, inner_ensure = wd._launch, wd._ensure_previous_dead

    def launch_spy(extra, log_dir_):
        events.append("launch")
        return inner_launch(extra, log_dir_)

    def ensure_spy(proc):
        events.append("ensure")
        return inner_ensure(proc)

    monkeypatch.setattr(wd, "_launch", launch_spy)
    monkeypatch.setattr(wd, "_ensure_previous_dead", ensure_spy)

    assert wd.main(["--workers", "1"]) == 0

    assert events == ["ensure", "launch", "ensure", "launch"]  # ensure always precedes its launch


def test_forwarded_args_parse_in_the_collector(tmp_path, monkeypatch):
    """What the watchdog forwards (including --budget, which used to crash it) parses in collect.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "growing_bandits" / "empirical"))
    import collect

    class _Parsed(Exception):
        pass

    def stop(lock_path):
        raise _Parsed  # argparse accepted every argument; stop before anything else happens

    monkeypatch.setattr(collect, "_acquire_lock", stop)
    with pytest.raises(_Parsed):
        collect.main(wd.collector_args(8, True, 40.0) + ["--log-dir", str(tmp_path)])
