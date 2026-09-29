"""Keep the empirical-pool collector alive unattended.

    .venv/bin/python scripts/watchdog_empirical_pool.py [--pilot] [--workers 8] [--budget 40]
    .venv/bin/python scripts/watchdog_empirical_pool.py --profile gitlab --budget USD [--pilot]

``--profile`` (default ``prereg9``) is forwarded to the collector, and the watchdog's own
LOG_DIR -- STATUS, ``collect.lock``, the worker files, RELAUNCH_LOG -- follows it:
``logs/empirical_pool`` for ``prereg9``, ``logs/heterogeneity/<profile>`` otherwise (passed
to the collector as ``--log-dir`` too, so the two can never watch different directories).
A heterogeneity profile needs an explicit ``--budget``. ``--print-log-dir`` prints that
directory and exits without touching anything (the launcher uses it).

Every POLL_SECONDS:

* STATUS ``done``/``budget`` -> exit 0. STATUS ``provider_down`` -> exit 2 without a
  relaunch: the provider is failing (outage or exhausted quota) and a relaunch would only turn
  more queue items into ``missing``; a person decides when to resume.
* The collector process is gone -> relaunch it (it resumes by itself).
* It is alive but no log line has been written for STALL_MINUTES, or (STATUS ``running``)
  any one worker's ``worker_<w>.jsonl`` has not grown for WORKER_STALL_MINUTES -> kill and
  relaunch. A worker whose ``worker_<w>.exit`` says it finished (done/budget/provider_down)
  is exempt; a worker file that does not exist yet counts as fresh until the collector has
  been up for WORKER_STALL_MINUTES. One hung worker is therefore caught on its own, not only
  once every worker hangs. The limit follows the profile's wall clock (`worker_stall_minutes`):
  WORKER_STALL_MINUTES (15) for the 180 s profiles, WORKER_STALL_MINUTES_LONG (25) for the
  600 s ones, whose healthy worker can go ~17 minutes without an append (a 300 s provider
  backoff, a recovery, then a 600 s episode).

Before every launch it writes STATUS ``launching`` (treated like ``running``), so a STATUS
left ``done`` by an earlier run -- e.g. the pilot -- can never make a fresh launch that dies
before writing ``running`` look finished. Every relaunch appends one machine-readable line,
``RELAUNCH <n> <reason> mode=<pilot|full> at=<utc>``, to RELAUNCH_LOG (read by gate G2) and
prints it. Gives up after MAX_RELAUNCHES and writes WATCHDOG_GAVE_UP beside the logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "empirical_pool"
HET_LOG_ROOT = ROOT / "logs" / "heterogeneity"
DEFAULT_PROFILE = "prereg9"
#: collect.py's `PROFILES` (kept stdlib-only here; tests assert the two agree).
PROFILE_NAMES = ("prereg9", "gitlab", "gmail", "bridge")
COLLECT = ROOT / "experiments" / "growing_bandits" / "empirical" / "collect.py"
POLL_SECONDS = 120
STALL_MINUTES = 30.0
WORKER_STALL_MINUTES = 15.0
#: Profiles whose agent wall clock is 600 s (collect.py's `PROFILES`): provider backoff (300 s)
#: + recovery (~20-100 s) + one episode (600 s) + a poll (120 s) can exceed 15 minutes.
WORKER_STALL_MINUTES_LONG = 25.0
LONG_CLOCK_PROFILES = ("gitlab", "bridge")
MAX_RELAUNCHES = 20
RELAUNCH_LOG = "relaunches.log"
FINISHED = ("done", "budget", "provider_down")
#: A worker exit marker with one of these outcomes means the worker is finished, not hung.
WORKER_FINISHED = ("done", "budget", "provider_down")


def decide(status: str | None, alive: bool, minutes_since_progress: float,
           stale_workers: tuple[int, ...] | list[int] = ()) -> str:
    if status in FINISHED:
        return "finished"
    if alive and minutes_since_progress > STALL_MINUTES:
        return "kill_and_relaunch"
    if alive and status == "running" and stale_workers:
        return "kill_and_relaunch"
    if alive:
        return "wait"
    return "relaunch"


def line_count(log_dir: Path) -> int:
    total = 0
    for path in sorted(Path(log_dir).glob("worker_*.jsonl")):
        with open(path) as fh:
            total += sum(1 for _ in fh)
    return total


def worker_sizes(log_dir: Path, n_workers: int) -> dict[int, int | None]:
    """Byte size of each worker's log, ``None`` while it does not exist."""
    sizes: dict[int, int | None] = {}
    for w in range(n_workers):
        try:
            sizes[w] = (Path(log_dir) / f"worker_{w}.jsonl").stat().st_size
        except FileNotFoundError:
            sizes[w] = None
    return sizes


def finished_workers(log_dir: Path) -> set[int]:
    """Workers whose exit marker (written by the collector) says they finished rather than crashed."""
    out: set[int] = set()
    for path in Path(log_dir).glob("worker_*.exit"):
        try:
            w = int(path.stem.split("_", 1)[1])
            outcome = path.read_text().strip()
        except (ValueError, OSError):
            continue
        if outcome in WORKER_FINISHED:
            out.add(w)
    return out


def worker_stall_minutes(profile: str) -> float:
    """How long one worker's log may stay unchanged before it counts as hung, for `profile`."""
    return WORKER_STALL_MINUTES_LONG if profile in LONG_CLOCK_PROFILES else WORKER_STALL_MINUTES


class WorkerClocks:
    """When each worker's log last grew; every clock restarts at each (re)launch.

    `stall_minutes` ``None`` means WORKER_STALL_MINUTES (read at each `stale` call)."""

    def __init__(self, n_workers: int, now: float, stall_minutes: float | None = None) -> None:
        self.size: dict[int, int | None] = {w: None for w in range(n_workers)}
        self.grew_at: dict[int, float] = {w: now for w in range(n_workers)}
        self.stall_minutes = stall_minutes

    def restart(self, now: float) -> None:
        for w in self.grew_at:
            self.grew_at[w] = now

    def observe(self, sizes: dict[int, int | None], now: float) -> None:
        for w, size in sizes.items():
            if size != self.size.get(w):
                self.size[w] = size
                if size is not None:
                    self.grew_at[w] = now

    def stale(self, now: float, exempt: set[int]) -> tuple[int, ...]:
        limit = (WORKER_STALL_MINUTES if self.stall_minutes is None else self.stall_minutes) * 60.0
        return tuple(w for w, t in sorted(self.grew_at.items()) if w not in exempt and now - t > limit)


def _status(log_dir: Path) -> str | None:
    path = log_dir / "STATUS"
    return path.read_text().strip() if path.exists() else None


def _write_status(log_dir: Path, status: str) -> None:
    (log_dir / "STATUS").write_text(status + "\n")


def _clear_exit_markers(log_dir: Path) -> None:
    for path in Path(log_dir).glob("worker_*.exit"):
        path.unlink(missing_ok=True)


def _prepare_launch(log_dir: Path) -> None:
    """STATUS ``launching`` and no stale exit markers, immediately before every `_launch`."""
    _write_status(log_dir, "launching")
    _clear_exit_markers(log_dir)


#: Bounded wait for `_ensure_previous_dead`'s SIGKILL to take: this is a belt-and-braces check
#: before every launch, not the main kill path (`_kill_and_wait_process` already SIGTERM'd/
#: SIGKILL'd a stalled collector), so it stays short.
ENSURE_DEAD_WAIT_S = 5.0


def _ensure_previous_dead(proc: subprocess.Popen | None) -> None:
    """SIGKILL a previous collector's whole process group before every `_launch`.

    browser-use workers can survive their leader's exit and keep appending to the same
    worker file as their replacement, so both the first launch (`proc` is ``None``, a no-op)
    and every relaunch -- whether the collector already died on its own or was just
    SIGTERM/SIGKILL'd by `_kill_and_wait_process` -- must make sure nothing from the previous
    process group is still alive before a fresh set of workers starts.
    """
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=ENSURE_DEAD_WAIT_S)
    except subprocess.TimeoutExpired:
        pass


#: Fix round 1, item 4: bounded wait after SIGKILLing an orphaned lock holder from a PREVIOUS
#: watchdog *session* (no `Popen` handle to `.wait()` on, unlike `_ensure_previous_dead`).
ORPHAN_KILL_WAIT_S = 5.0


def _kill_orphaned_lock_holder(log_dir: Path) -> None:
    """Before the very first launch of a watchdog session, SIGKILL whatever `collect.lock`
    under `log_dir` still names a live process group.

    `_ensure_previous_dead` only guards a relaunch *within this session* (it needs the
    in-memory `proc` this same watchdog process launched); a `collect.lock` left behind by an
    earlier watchdog session or a manually-started collector has no such handle, so orphans of
    that earlier run (its leader, or a worker that outlived it) would otherwise survive into a
    fresh one and keep appending to the files it's about to resume from. Uses the same
    process-group probe as `collect.py`'s own lock (`os.killpg`, not a single-pid `os.kill`).
    """
    lock_path = Path(log_dir) / "collect.lock"
    if not lock_path.exists():
        return
    try:
        owner = int(lock_path.read_text().strip())
    except (ValueError, OSError):
        return
    try:
        os.killpg(owner, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    time.sleep(ORPHAN_KILL_WAIT_S)


def _launch(extra: list[str], log_dir: Path) -> subprocess.Popen:
    out = open(log_dir / "collect.out", "a")  # noqa: SIM115 -- lives as long as the child
    return subprocess.Popen([sys.executable, str(COLLECT), *extra], stdout=out, stderr=subprocess.STDOUT,
                            cwd=ROOT, start_new_session=True)


def _record_relaunch(log_dir: Path, n: int, reason: str, mode: str) -> str:
    line = f"RELAUNCH {n} {reason} mode={mode} at={dt.datetime.now(dt.UTC).isoformat()}"
    with open(log_dir / RELAUNCH_LOG, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
    return line


def _kill_and_wait_process(proc: subprocess.Popen) -> None:
    """Kill process group, escalating to SIGKILL if needed. Never raises."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def profile_log_dir(profile: str) -> Path:
    """Where `profile`'s collector writes (collect.py's ``PROFILES[profile]["log_dir"]``)."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILE_NAMES}")
    return LOG_DIR if profile == DEFAULT_PROFILE else HET_LOG_ROOT / profile


def collector_args(workers: int, pilot: bool, budget: float | None, profile: str = DEFAULT_PROFILE,
                   log_dir: Path | None = None) -> list[str]:
    """collect.py's arguments. ``prereg9`` forwards exactly what it always did; any other
    profile adds ``--profile`` and ``--log-dir`` (the directory this watchdog watches)."""
    extra = ["--workers", str(workers)]
    if budget is not None:
        extra += ["--budget", str(budget)]
    if pilot:
        extra.append("--pilot")
    if profile != DEFAULT_PROFILE:
        extra += ["--profile", profile, "--log-dir", str(log_dir if log_dir is not None else profile_log_dir(profile))]
    return extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=None, help="forwarded to collect.py (its default otherwise)")
    ap.add_argument("--profile", choices=PROFILE_NAMES, default=DEFAULT_PROFILE, help="forwarded to collect.py")
    ap.add_argument("--print-log-dir", action="store_true", help="print this profile's log dir and exit")
    args = ap.parse_args(argv)
    if args.profile != DEFAULT_PROFILE and args.budget is None:
        ap.error(f"--profile {args.profile} needs an explicit --budget USD (the collector has no default for it)")
    log_dir = profile_log_dir(args.profile)
    if args.print_log_dir:
        print(log_dir)
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / RELAUNCH_LOG).touch()  # its existence is how gate G2 knows a watchdog supervised the run
    mode = "pilot" if args.pilot else "full"
    extra = collector_args(args.workers, args.pilot, args.budget, args.profile, log_dir)
    proc: subprocess.Popen | None = None
    _kill_orphaned_lock_holder(log_dir)  # orphans of a PREVIOUS watchdog session/process
    _ensure_previous_dead(proc)  # a no-op here (no `proc` yet): kept for the "every launch" invariant
    _prepare_launch(log_dir)
    proc = _launch(extra, log_dir)
    now = time.time()
    relaunches, last_lines, last_progress = 0, line_count(log_dir), now
    clocks = WorkerClocks(args.workers, now, stall_minutes=worker_stall_minutes(args.profile))
    while True:
        time.sleep(POLL_SECONDS)
        now = time.time()
        lines = line_count(log_dir)
        if lines != last_lines:
            last_lines, last_progress = lines, now
        clocks.observe(worker_sizes(log_dir, args.workers), now)
        status = _status(log_dir)
        alive = proc.poll() is None
        stale = clocks.stale(now, finished_workers(log_dir)) if alive and status == "running" else ()
        action = decide(status, alive, (now - last_progress) / 60.0, stale)
        if action == "finished":
            if status == "provider_down":
                print(f"collector stopped: provider_down ({lines} lines). The LLM provider failed on "
                      "consecutive items (outage or exhausted quota); NOT relaunching. Check the provider, "
                      "then relaunch by hand -- items already missing stay missing.", flush=True)
                return 2
            print(f"collector finished: {status} ({lines} lines)", flush=True)
            return 0
        if action == "wait":
            continue
        if relaunches >= MAX_RELAUNCHES:
            (log_dir / "WATCHDOG_GAVE_UP").write_text(f"{relaunches} relaunches\n")
            return 1
        if action == "kill_and_relaunch" and proc.poll() is None:
            _kill_and_wait_process(proc)
        relaunches += 1
        if action == "relaunch":
            reason = f"collector_exited:{status}"
        elif stale and (now - last_progress) / 60.0 <= STALL_MINUTES:
            reason = "stale_workers:" + ",".join(str(w) for w in stale)
        else:
            reason = "global_stall"
        _record_relaunch(log_dir, relaunches, reason, mode)
        _ensure_previous_dead(proc)
        _prepare_launch(log_dir)
        proc = _launch(extra, log_dir)
        last_progress = time.time()
        clocks.restart(last_progress)


if __name__ == "__main__":
    raise SystemExit(main())
