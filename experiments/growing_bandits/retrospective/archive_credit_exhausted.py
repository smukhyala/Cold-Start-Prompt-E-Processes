"""Pre-registration 16 Amendment 2: move episodes the provider refused for lack of prepaid credit out of the worker logs.

A record whose text contains Anthropic's refusal ``credit balance is too low`` was never a real episode: from the
moment the account ran dry, every model call returned HTTP 400, the agent gave up, and (before this amendment's
collector fix) the record was logged as an ``ok`` failure. Such records are moved verbatim to
``<log>/archive/worker_<w>.credit_exhausted.jsonl``; the collector then sees those queue items as missing and re-runs
them on its next launch. Same mechanics as ``archive_sleep_spanning`` (holds ``collect.lock``, atomic rewrite,
idempotent).

    PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/archive_credit_exhausted.py logs/headroom_haiku_gmail
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from experiments.growing_bandits.empirical.collect import _acquire_lock, _parses, _release_lock

MARKER = b"credit balance is too low"


def is_credit_exhausted(line: bytes) -> bool:
    return MARKER in line


def archive_credit_exhausted(log_dir: Path) -> dict[str, int]:
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
                if is_credit_exhausted(line):
                    archived_lines.append(line)
                else:
                    kept_lines.append(line)
            if archived_lines:
                archive_dir.mkdir(parents=True, exist_ok=True)
                with open(archive_dir / f"{path.stem}.credit_exhausted.jsonl", "ab") as fh:
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
        print(log_dir, archive_credit_exhausted(log_dir))


if __name__ == "__main__":
    main()
