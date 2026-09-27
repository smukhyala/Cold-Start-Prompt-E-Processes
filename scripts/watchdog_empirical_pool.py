"""Keep the empirical-pool collector alive unattended.

    .venv/bin/python scripts/watchdog_empirical_pool.py [--pilot]

Every POLL_SECONDS: if STATUS says done/budget, exit. If the collector process is gone,
relaunch it (it resumes by itself). If it is alive but no log line has been written for
STALL_MINUTES, kill and relaunch it. Gives up after MAX_RELAUNCHES and writes
WATCHDOG_GAVE_UP beside the logs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "empirical_pool"
COLLECT = ROOT / "experiments" / "growing_bandits" / "empirical" / "collect.py"
POLL_SECONDS = 120
STALL_MINUTES = 30.0
MAX_RELAUNCHES = 20


def decide(status: str | None, alive: bool, minutes_since_progress: float) -> str:
    if status in ("done", "budget"):
        return "finished"
    if alive and minutes_since_progress > STALL_MINUTES:
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


def _status(log_dir: Path) -> str | None:
    path = log_dir / "STATUS"
    return path.read_text().strip() if path.exists() else None


def _launch(extra: list[str], log_dir: Path) -> subprocess.Popen:
    out = open(log_dir / "collect.out", "a")  # noqa: SIM115 -- lives as long as the child
    return subprocess.Popen([sys.executable, str(COLLECT), *extra], stdout=out, stderr=subprocess.STDOUT,
                            cwd=ROOT, start_new_session=True)


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    extra = ["--workers", str(args.workers)] + (["--pilot"] if args.pilot else [])
    proc = _launch(extra, LOG_DIR)
    relaunches, last_lines, last_progress = 0, line_count(LOG_DIR), time.time()
    while True:
        time.sleep(POLL_SECONDS)
        lines = line_count(LOG_DIR)
        if lines != last_lines:
            last_lines, last_progress = lines, time.time()
        action = decide(_status(LOG_DIR), proc.poll() is None, (time.time() - last_progress) / 60.0)
        if action == "finished":
            print(f"collector finished: {_status(LOG_DIR)} ({lines} lines)", flush=True)
            return 0
        if action == "wait":
            continue
        if relaunches >= MAX_RELAUNCHES:
            (LOG_DIR / "WATCHDOG_GAVE_UP").write_text(f"{relaunches} relaunches\n")
            return 1
        if action == "kill_and_relaunch" and proc.poll() is None:
            _kill_and_wait_process(proc)
        relaunches += 1
        print(f"relaunch {relaunches}: {action} at {lines} lines", flush=True)
        proc, last_progress = _launch(extra, LOG_DIR), time.time()


if __name__ == "__main__":
    raise SystemExit(main())
