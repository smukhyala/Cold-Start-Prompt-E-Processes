#!/usr/bin/env python3
"""Watchdog for the GitLab paired 40-task sweep.

The WebArena/browser-use stack can occasionally stop producing JSONL heartbeats
while the Python process remains alive. This watchdog keeps the paired sweep
moving by:

- merging complete 40-row arm segments into *_FULL.jsonl files;
- resuming partial arms from the first missing timestep;
- restarting the top-level paired runner when no runner is active;
- killing stale idle runners and their GitLab app server on port 8001.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/gitlab_strong_arm/paired"
LOG_DIR = OUT / "logs"
STDOUT_LOG = OUT / "paired_run.out"
WATCHDOG_LOG = OUT / "watchdog.log"
TARGET_T = 40
PORT = 8001
WATCHDOG_SCREEN = "gitlab_paired_watchdog"


@dataclass
class SegmentGroup:
    base_run_id: str
    rows_by_t: dict[int, dict[str, Any]]

    @property
    def count(self) -> int:
        return len(self.rows_by_t)

    @property
    def last_t(self) -> int:
        return max(self.rows_by_t) if self.rows_by_t else 0

    @property
    def contiguous(self) -> bool:
        return sorted(self.rows_by_t) == list(range(1, self.last_t + 1))


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"watchdog starting interval={args.interval}s stale={args.stale_seconds}s")

    while True:
        try:
            tick(args)
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR: watchdog tick failed: {exc!r}")
        if args.once:
            break
        time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def tick(args: argparse.Namespace) -> None:
    arms = arm_ids()
    for arm in arms:
        ensure_full_if_complete(arm)

    completed = [arm for arm in arms if full_log(arm) is not None]
    partial = first_partial_arm(arms)
    runner_pids = active_runner_pids()

    if len(completed) == len(arms):
        log("all GitLab paired arms have complete FULL logs")
        kill_gitlab_screens()
        return

    if runner_pids:
        age = freshness_age_s()
        if age > args.stale_seconds:
            log(
                f"stale runner detected age={age:.0f}s pids={runner_pids}; "
                "terminating and resuming"
            )
            kill_runner_tree()
            if partial is not None:
                launch_resume(partial)
            else:
                launch_paired_runner()
        else:
            active = partial or "top-level"
            log(
                f"runner active pids={runner_pids}; freshness={age:.0f}s; "
                f"completed={len(completed)}/{len(arms)}; current={active}"
            )
        return

    if partial is not None:
        log(f"no runner active; resuming partial arm {partial}")
        launch_resume(partial)
    else:
        log(f"no runner active; starting paired runner; completed={len(completed)}/{len(arms)}")
        launch_paired_runner()


def arm_ids() -> list[str]:
    path = ROOT / "configs/arms_gitlab_strong.yaml"
    raw = yaml.safe_load(path.read_text())
    return [str(entry["arm_id"]) for entry in raw["arms"]]


def trial_name(arm: str) -> str:
    return f"gitlab_strong_paired_{arm}"


def full_log(arm: str) -> Path | None:
    candidates = sorted(LOG_DIR.glob(f"{trial_name(arm)}_trial0_*_FULL.jsonl"))
    for path in reversed(candidates):
        if row_count(path) == TARGET_T:
            return path
    return None


def first_partial_arm(arms: list[str]) -> str | None:
    for arm in arms:
        if full_log(arm) is not None:
            continue
        group = best_group(arm)
        if group is not None and group.count > 0:
            return arm
    return None


def segment_paths(arm: str) -> list[Path]:
    paths = []
    for path in LOG_DIR.glob(f"{trial_name(arm)}_trial0_*.jsonl"):
        name = path.name
        if name.startswith("INVALID_"):
            continue
        if name.endswith("_FULL.jsonl") or name.endswith("_MERGED_SO_FAR.jsonl"):
            continue
        if path.stat().st_size == 0:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def groups_for(arm: str) -> list[SegmentGroup]:
    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for path in segment_paths(arm):
        rows = read_rows(path)
        if not rows:
            continue
        base = str(rows[0]["run_id"]).split("_resume_from_")[0]
        rows_by_t = groups.setdefault(base, {})
        for row in rows:
            t = int(row["t"])
            rows_by_t[t] = row
    return [SegmentGroup(base, rows_by_t) for base, rows_by_t in groups.items()]


def best_group(arm: str) -> SegmentGroup | None:
    groups = groups_for(arm)
    if not groups:
        return None
    return max(groups, key=lambda g: (g.count, g.last_t, g.base_run_id))


def ensure_full_if_complete(arm: str) -> Path | None:
    existing = full_log(arm)
    if existing is not None:
        return existing
    group = best_group(arm)
    if group is None or group.count < TARGET_T or not group.contiguous:
        return None
    rows = [group.rows_by_t[t] for t in range(1, TARGET_T + 1)]
    out = LOG_DIR / f"{group.base_run_id}_FULL.jsonl"
    out.write_text("\n".join(json.dumps(r, separators=(",", ":"), default=str) for r in rows) + "\n")
    successes = sum(float(r.get("reward", 0.0)) >= 0.5 for r in rows)
    log(f"merged {arm} FULL log: {out} rows={len(rows)} successes={successes}")
    return out


def merged_so_far(arm: str) -> tuple[Path, SegmentGroup]:
    group = best_group(arm)
    if group is None or group.count == 0:
        raise RuntimeError(f"no partial rows found for {arm}")
    if not group.contiguous:
        raise RuntimeError(f"partial rows for {arm} are not contiguous: {sorted(group.rows_by_t)}")
    rows = [group.rows_by_t[t] for t in range(1, group.last_t + 1)]
    out = LOG_DIR / f"{group.base_run_id}_MERGED_SO_FAR.jsonl"
    out.write_text("\n".join(json.dumps(r, separators=(",", ":"), default=str) for r in rows) + "\n")
    return out, group


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def row_count(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def freshness_age_s() -> float:
    mtimes = []
    if STDOUT_LOG.exists():
        mtimes.append(STDOUT_LOG.stat().st_mtime)
    mtimes.extend(path.stat().st_mtime for path in LOG_DIR.glob("gitlab_strong_paired_*.jsonl"))
    if not mtimes:
        return float("inf")
    return time.time() - max(mtimes)


def active_runner_pids() -> list[int]:
    patterns = [
        "experiments/run_gitlab_strong_arm.py --mode paired",
        "-m cold_start.cli.run --config results/gitlab_strong_arm/paired",
    ]
    pids: set[int] = set()
    for pattern in patterns:
        proc = subprocess.run(["pgrep", "-f", pattern], text=True, capture_output=True, check=False)
        for line in proc.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid != os.getpid():
                pids.add(pid)
    return sorted(pids)


def kill_runner_tree() -> None:
    pids = active_runner_pids()
    for pid in pids:
        safe_kill(pid, signal.SIGTERM)
    time.sleep(5)
    for pid in pids:
        if pid_alive(pid):
            safe_kill(pid, signal.SIGKILL)
    kill_port_listener(PORT)
    kill_gitlab_screens()


def kill_port_listener(port: int) -> None:
    proc = subprocess.run(
        ["lsof", "-tiTCP:%d" % port, "-sTCP:LISTEN"],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != os.getpid():
            log(f"killing leftover listener on port {port}: pid={pid}")
            safe_kill(pid, signal.SIGTERM)


def kill_gitlab_screens() -> None:
    proc = subprocess.run(["screen", "-ls"], text=True, capture_output=True, check=False)
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if ".gitlab_" not in stripped:
            continue
        name = stripped.split()[0].split(".", 1)[1]
        if name == WATCHDOG_SCREEN:
            continue
        log(f"quitting stale screen session {name}")
        subprocess.run(["screen", "-S", name, "-X", "quit"], check=False)


def safe_kill(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        log(f"could not kill pid={pid}: {exc}")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def launch_resume(arm: str) -> None:
    ensure_full_if_complete(arm)
    if full_log(arm) is not None:
        launch_paired_runner()
        return
    merged, group = merged_so_far(arm)
    next_t = group.last_t + 1
    config = LOG_DIR / "configs" / f"{trial_name(arm)}.yaml"
    if not config.exists():
        log(f"missing config for {arm}: {config}; starting paired runner instead")
        launch_paired_runner()
        return
    session = f"gitlab_{arm}_resume_{next_t}"
    cmd = (
        f"cd {shell(ROOT)} && "
        "env PYTHONUNBUFFERED=1 PYTHONPATH=src "
        f".venv/bin/python -m cold_start.cli.run --config {shell(config)} "
        f"--resume-from {shell(merged)} --start-at {next_t} --skip-preflight "
        f">> {shell(STDOUT_LOG)} 2>&1"
    )
    launch_screen(session, cmd)
    log(f"launched resume session={session} arm={arm} next_t={next_t} merged={merged}")


def launch_paired_runner() -> None:
    session = "gitlab_paired_40_continue"
    cmd = (
        f"cd {shell(ROOT)} && "
        "env PYTHONUNBUFFERED=1 PYTHONPATH=src "
        ".venv/bin/python experiments/run_gitlab_strong_arm.py "
        "--mode paired --task-family gitlab --num-tasks 40 "
        "--output-dir results/gitlab_strong_arm/paired --skip-existing "
        f">> {shell(STDOUT_LOG)} 2>&1"
    )
    launch_screen(session, cmd)
    log(f"launched paired runner session={session}")


def launch_screen(session: str, cmd: str) -> None:
    kill_gitlab_screens()
    subprocess.run(["screen", "-dmS", session, "/bin/zsh", "-lc", cmd], check=True)


def shell(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    WATCHDOG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCHDOG_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
