"""Pre-registration 13 Amendment 1: move episodes that spanned a system sleep out of the worker logs.

An ``ok`` record whose ``wallclock_s`` exceeds its ``timeout_s`` by more than `SLACK_S` cannot be a normal
episode: the asyncio clock fired late because the machine was asleep, so its outcome (usually a timeout) is
an infrastructure artifact. Such records are moved verbatim to ``<log>/archive/worker_<w>.sleep_spanning.jsonl``;
the collector then sees those queue items as missing and re-runs them on its next launch.

Mirrors `collect.archive_out_of_queue`: holds the collector's ``collect.lock`` (refuses while that log dir's
collector is alive), rewrites each worker file atomically, and is idempotent.

    ./.venv/bin/python experiments/growing_bandits/retrospective/archive_sleep_spanning.py logs/noise_recheck_gmail
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.growing_bandits.empirical.collect import _acquire_lock, _parses, _release_lock

SLACK_S = 60.0


def is_sleep_spanning(record: dict) -> bool:
    if record.get("status") != "ok":
        return False
    wall = float(record.get("wallclock_s") or 0.0)
    limit = float(record.get("timeout_s") or 0.0)
    return limit > 0.0 and wall > limit + SLACK_S


def archive_sleep_spanning(log_dir: Path) -> dict[str, int]:
    log_dir = Path(log_dir)
    lock_path = log_dir / "collect.lock"
    _acquire_lock(lock_path)
    try:
        archive_dir = log_dir / "archive"
        counts = {"kept": 0, "archived": 0, "files": 0}
        for path in sorted(log_dir.glob("worker_*.jsonl")):
            data = path.read_bytes()
            *complete, tail = data.split(b"\n")
            if tail.strip() and _parses(tail):
                complete.append(tail)
            kept_lines: list[bytes] = []
            archived_lines: list[bytes] = []
            for line in complete:
                if not line.strip():
                    continue
                if is_sleep_spanning(json.loads(line)):
                    archived_lines.append(line)
                else:
                    kept_lines.append(line)
            if archived_lines:
                archive_dir.mkdir(parents=True, exist_ok=True)
                with open(archive_dir / f"{path.stem}.sleep_spanning.jsonl", "ab") as fh:
                    for line in archived_lines:
                        fh.write(line + b"\n")
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with open(tmp_path, "wb") as fh:
                    for line in kept_lines:
                        fh.write(line + b"\n")
                os.replace(tmp_path, path)
            counts["kept"] += len(kept_lines)
            counts["archived"] += len(archived_lines)
            counts["files"] += 1
        return counts
    finally:
        _release_lock(lock_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for log_dir in args.log_dirs:
        print(log_dir, archive_sleep_spanning(log_dir))


if __name__ == "__main__":
    main()
