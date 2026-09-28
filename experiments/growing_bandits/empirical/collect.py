"""Collect the real prompt pools' outcomes on WebArena Gmail (spec section 3.3).

    .venv/bin/python experiments/growing_bandits/empirical/collect.py --workers 8 [--pilot]

At start the collector checks ``queue.jsonl`` against ``manifest.json``'s sha256, computes the
items still without a terminal record (``ok`` or ``missing``) and deals THOSE round-robin to
the workers, so every remaining item is owned by exactly one worker per launch and a lost
worker's share does not serialize at the end. Worker w runs its share in queue order on its
own WebArena server (port 8001 + w) and appends one JSON line per *attempt* to
``logs/empirical_pool/worker_<w>.jsonl``; an item's earlier ``infra_error`` attempts count
toward its three. When a worker returns, the collector writes ``worker_<w>.exit`` (its
outcome), which the watchdog uses to tell a finished worker from a hung one.

An agent timeout is a task failure, scored 0, as every historical Gmail run scored it. A
harness exception, or an LLM API error in the agent's trace while the agent was not done,
is ``infra_error``: the worker restarts its server and browser and retries, and the third
failure writes ``missing``, which estimation excludes and never scores 0. A retry after a
*provider* error (the give-up streak, e.g. rate limit or exhausted quota) waits
``PROVIDER_BACKOFF_S`` first; a worker whose last ``PROVIDER_DOWN_STREAK`` items all ended
``missing`` on provider errors stops, and the collector's final STATUS is ``provider_down``
(the watchdog does not relaunch it: an outage would otherwise turn the queue into
``missing`` item by item, irreversibly).

Every worker re-reads the summed ``cost_usd`` of all workers before each attempt and stops
at the budget, so the overshoot is bounded by the episodes already in flight (8 x ~$0.05).

A worker recycles its server/browser (close + reset) every ``RECYCLE_EVERY`` episodes even
without an error: historical Gmail runs on this adapter stalled every ~15-25 episodes (the
server's pipes, now drained by the adapter, and verifier requests with no timeout).
The first start and every recovery (after an infra error or a recycle) retry ``reset()``
with backoff; a worker that fails to recover several times in a row gives up with a clear
error rather than spinning forever. A pid lockfile refuses a second collector against the
same log dir while one is already running.

Log files are shared read-only across workers: a reader never modifies another worker's
file (a torn final line from a write in flight is skipped in memory); only the owner
repairs its own file's tail, before it appends.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import multiprocessing as mp_
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_pools  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.tasks.webarena import ERROR_TAIL_LEN  # noqa: E402
from cold_start.types import RunResult  # noqa: E402

log = logging.getLogger("empirical.collect")

LOG_DIR = ROOT / "logs" / "empirical_pool"
BASE_PORT = 8001
BUDGET_USD = 260.0
MAX_ATTEMPTS = 3
MAX_STEPS = 30
N_BANK_TASKS = 60
AGENT = {
    "web_app": "apps/gmail",
    "task_suite": "real-tasks",
    "use_vision": False,
    "headless": True,
    "timeout_s": 180,
    "llm_provider": "openai",
    "llm_model": "gpt-5.4-mini",
    "llm_reasoning_effort": "low",
}
API_ERROR_MARKERS: tuple[str, ...] = (
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailable",
    "Error code: 429", "overloaded",
    # browser-use wraps openai's APIConnectionError/APITimeoutError as ModelProviderError(message=str(e)),
    # which collapses to just these two sentences (openai/_exceptions.py APIConnectionError/APITimeoutError).
    "Connection error", "Request timed out", "Rate limit",
)
#: Any provider 5xx (500, 502-504, Cloudflare's 520-529, Anthropic's 529 "overloaded").
PROVIDER_5XX = re.compile(r"Error code: 5\d\d")
# Historical Gmail runs on this adapter stalled every ~15-25 episodes per worker; recycling the
# server/browser this often bounds how long any one process lives (~1 min overhead per recycle).
RECYCLE_EVERY = 15
# `reset()` retry policy for recovering a broken server/browser (after an infra error or a recycle).
RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_S: tuple[float, ...] = (10.0, 30.0, 60.0)
# Consecutive *failed* recoveries (not consecutive infra_error attempts) before a worker gives up.
MAX_RECOVERY_FAILURES = 3
# Wait before retrying an item whose last attempt ended in a provider-error streak: a
# rate-limit blip clears in a minute, an outage does not (then PROVIDER_DOWN_STREAK fires).
PROVIDER_BACKOFF_S: tuple[float, ...] = (60.0, 300.0)
# Consecutive items of one worker ending `missing` on provider errors before it stops.
PROVIDER_DOWN_STREAK = 3
# Record fields truncated to this many characters.
TEXT_LEN = 300

INFRA_EXCEPTION = "exception"
INFRA_PROVIDER = "provider"
#: A worker's return value / the collector's final STATUS, and the collector's exit code.
OUTCOME_DONE, OUTCOME_BUDGET, OUTCOME_FAILED, OUTCOME_PROVIDER_DOWN = "done", "budget", "failed", "provider_down"
EXIT_CODES: dict[str, int] = {OUTCOME_DONE: 0, OUTCOME_FAILED: 1, OUTCOME_BUDGET: 3, OUTCOME_PROVIDER_DOWN: 4}


@dataclass(frozen=True)
class WorkerConfig:
    worker: int
    n_workers: int
    log_dir: Path
    budget_usd: float
    max_attempts: int = MAX_ATTEMPTS
    pilot_only: bool = False
    max_steps: int = MAX_STEPS
    #: Queue indices this worker owns this launch (`partition_remaining`); ``None`` falls back
    #: to ``index % n_workers == worker``.
    assigned: tuple[int, ...] | None = None


def _default_failure_streak() -> int:
    """Consecutive step failures browser-use tolerates before giving up on an episode.

    Mirrors ``browser_use.agent.views.AgentSettings.max_failures`` (5 by default): after
    that many consecutive provider-error steps, browser-use forces a final "done" action
    (if ``final_response_after_failure``) and stops. Read live from the installed package
    rather than hard-coding it, so an upgrade that changes the default doesn't silently
    desync; falls back to a conservative 3 if browser-use isn't importable or the field
    moved.
    """
    try:
        from browser_use.agent.views import AgentSettings

        return int(AgentSettings.model_fields["max_failures"].default)
    except Exception:  # noqa: BLE001 -- any import/attribute drift falls back, never crashes classify
        return 3


def _is_provider_error(msg: str | None) -> bool:
    if msg is None:
        return False
    text = str(msg)
    return any(marker in text for marker in API_ERROR_MARKERS) or PROVIDER_5XX.search(text) is not None


def _ends_in_provider_streak(tail: list, required: int) -> bool:
    """True iff `tail` (oldest-first, one entry per step, ``None`` = no error that step)
    ends with `required` consecutive provider-error steps.

    Tolerates exactly one trailing step with no error: browser-use's forced final "done"
    action after ``max_failures`` has no error of its own, but is still part of the same
    give-up, not a recovery.
    """
    steps = list(tail)
    if steps and steps[-1] is None:
        steps = steps[:-1]
    streak = 0
    for msg in reversed(steps):
        if _is_provider_error(msg):
            streak += 1
        else:
            break
    return streak >= required


def classify_detail(result: RunResult | None, exc: BaseException | None) -> tuple[str, str | None]:
    """``(status, infra_reason)``: ``infra_error`` iff the harness raised (reason ``exception``),
    or the episode failed and ended in browser-use's give-up streak of provider errors (reason
    ``provider``); a single transient error the agent recovered from, or an agent timeout, is a
    scored task failure, not infra.
    """
    if exc is not None or result is None:
        return emp.STATUS_INFRA, INFRA_EXCEPTION
    trace = result.trace or {}
    if trace.get("timed_out"):
        return emp.STATUS_OK, None
    if result.success:
        return emp.STATUS_OK, None
    required = _default_failure_streak()
    if ERROR_TAIL_LEN <= required:
        # The tail must hold the whole streak plus browser-use's forced final "done" step.
        raise RuntimeError(f"webarena.ERROR_TAIL_LEN={ERROR_TAIL_LEN} cannot hold a {required}-step give-up "
                           "streak plus its forced-done step; raise ERROR_TAIL_LEN")
    tail = trace.get("error_tail") or []
    if _ends_in_provider_streak(tail, required):
        return emp.STATUS_INFRA, INFRA_PROVIDER
    return emp.STATUS_OK, None


def classify(result: RunResult | None, exc: BaseException | None) -> str:
    return classify_detail(result, exc)[0]


def _parses(chunk: bytes) -> bool:
    try:
        json.loads(chunk)
    except ValueError:  # JSONDecodeError and UnicodeDecodeError are both ValueErrors
        return False
    return True


def _read_records(path: Path) -> list[dict]:
    """Every record in one worker's log, READ-ONLY and tolerant of a write in flight.

    Only the final line can be torn (each file has one appending owner). An unterminated
    final line that does not parse is a write in progress (or a hard kill's fragment) and is
    skipped *in memory*; one that parses is a complete record whose newline is not yet on
    disk (a strict prefix of a JSON object never parses), so it counts -- its spend and its
    terminal status are real. The file is never modified: it may belong to a live worker. A
    complete (newline-terminated) malformed line is a real data bug and raises.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    *complete, tail = data.split(b"\n")
    rows = [json.loads(line) for line in complete if line.strip()]
    if tail.strip() and _parses(tail):
        rows.append(json.loads(tail))
    return rows


def _repair_own_tail(path: Path) -> None:
    """The OWNER's repair of its own log before it appends (never called on another worker's file).

    A worker that died mid-``_append`` leaves either a byte fragment (dropped: truncated
    back to the last newline) or a complete record missing only its newline (kept: the
    newline is appended, so the record is not lost and the next append is not glued to it).
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    if not data or data.endswith(b"\n"):
        return
    last_nl = data.rfind(b"\n")
    tail = data[last_nl + 1 :]
    if _parses(tail):
        log.warning("terminating a complete but unterminated final record in %s", path)
        with open(path, "ab") as fh:
            fh.write(b"\n")
        return
    log.warning("dropping a truncated trailing line in %s (%d bytes)", path, len(tail))
    with open(path, "r+b") as fh:
        fh.truncate(last_nl + 1)


def load_all(log_dir: Path) -> pd.DataFrame:
    """Every worker's records, read tolerantly (`_read_records`), as one attempts frame."""
    rows: list[dict] = []
    for path in sorted(Path(log_dir).glob("worker_*.jsonl")):
        rows.extend(_read_records(path))
    return emp.attempts_frame(rows)


def _load_archive(log_dir: Path) -> pd.DataFrame:
    """Every record `archive_out_of_queue` moved out of the worker files, read the same
    tolerant way (`_read_records`) as a live worker file -- `log_dir/archive` need not exist."""
    rows: list[dict] = []
    for path in sorted((Path(log_dir) / "archive").glob("*.jsonl")):
        rows.extend(_read_records(path))
    return emp.attempts_frame(rows)


def spent_usd(log_dir: Path) -> float:
    """Total real dollars spent under `log_dir`: every current ``worker_*.jsonl`` plus
    whatever `archive_out_of_queue` moved out of them into ``archive/*.jsonl``. Archiving a
    record for a narrower queue (amendment 1) must never make its already-spent cost invisible
    to the budget cap -- that cap is bounded by real money, not by what the current queue
    happens to include.
    """
    total = 0.0
    for attempts in (load_all(log_dir), _load_archive(log_dir)):
        if not attempts.empty:
            total += float(attempts["cost_usd"].fillna(0.0).astype(float).sum())
    return total


def _progress(log_dir: Path) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str, int], int]]:
    attempts = load_all(log_dir)
    done = {(r.arm_id, r.task_id, int(r.replicate))
            for r in attempts.itertuples() if r.status in emp.TERMINAL_STATUSES}
    tries: dict[tuple[str, str, int], int] = {}
    for r in attempts.itertuples():
        key = (r.arm_id, r.task_id, int(r.replicate))
        tries[key] = max(tries.get(key, 0), int(r.attempt))
    return done, tries


def check_bank(adapter, queue: list[make_pools.QueueItem]) -> None:
    bank = set(adapter.task_ids())
    needed = {item.task_id for item in queue}
    if not needed <= bank:
        raise RuntimeError(f"task bank mismatch: {sorted(needed - bank)[:5]} are not in the adapter's bank")


def _clip(value, n: int = TEXT_LEN) -> str | None:
    return None if value is None else str(value)[:n]


def _record(item, attempt: int, status: str, result: RunResult | None, exc: BaseException | None,
            cfg: WorkerConfig, prompt: make_pools.PoolArm, infra_reason: str | None = None) -> dict:
    trace = (result.trace if result is not None else {}) or {}
    tokens = (result.tokens if result is not None else {}) or {}
    is_done = trace.get("is_done")
    return {
        "schema": emp.SCHEMA,
        "pool": item.pool,
        "arm_id": item.arm_id,
        "task_id": item.task_id,
        "replicate": item.replicate,
        "attempt": attempt,
        "status": status,
        "success": None if status != emp.STATUS_OK or result is None else int(bool(result.success)),
        "cost_usd": float(tokens.get("cost_usd", 0.0) or 0.0),
        "steps": None if result is None else int(result.steps),
        "wallclock_s": None if result is None else float(result.wallclock_s),
        "timed_out": bool(trace.get("timed_out", False)),
        "errors": [str(e)[:300] for e in (trace.get("errors") or [])[:3]],
        "error_tail": [None if e is None else str(e) for e in (trace.get("error_tail") or [])],
        "is_done": None if is_done is None else bool(is_done),
        "verifier_message": _clip(trace.get("verifier_message")),
        "final_result": _clip(trace.get("final_result")),
        "infra_reason": infra_reason,
        "task_dir": trace.get("task_dir"),
        "exception": None if exc is None else repr(exc)[:500],
        "tokens": tokens,
        "queue_index": item.index,
        "pilot": item.pilot,
        "worker": cfg.worker,
        "port": BASE_PORT + cfg.worker,
        "prompt_sha256": prompt.sha256,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def _append(path: Path, record: dict) -> None:
    _repair_own_tail(path)  # never glue a new record onto a torn prior write
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _recover(adapter) -> None:
    """Restart `adapter`'s server/browser after an infra error or a proactive recycle.

    `close()`'s own exceptions are swallowed and logged: a broken server/browser is exactly
    what triggers recovery, so a failing close() must never block the reset() that follows.
    `reset()` is retried with backoff (a WebArena server can take longer than one shot to
    come back); if every attempt fails, the last exception is raised so the caller can decide
    whether to give up the worker.
    """
    try:
        adapter.close()
    except Exception as err:  # noqa: BLE001 -- close() failing must never block reset()
        log.warning("close() raised during recovery (continuing to reset): %r", err)

    last_err: BaseException | None = None
    for attempt in range(RECOVERY_ATTEMPTS):
        try:
            adapter.reset(seed=0)
            return
        except Exception as err:  # noqa: BLE001 -- retried, then surfaced to the caller
            last_err = err
            log.warning("reset() attempt %d/%d failed: %r", attempt + 1, RECOVERY_ATTEMPTS, err)
            if attempt < RECOVERY_ATTEMPTS - 1:
                time.sleep(RECOVERY_BACKOFF_S[attempt])
    assert last_err is not None
    raise last_err


def _share(cfg: WorkerConfig, queue: list[make_pools.QueueItem]) -> list[make_pools.QueueItem]:
    """This worker's items, in queue order."""
    if cfg.assigned is not None:
        owned = set(cfg.assigned)
        return [q for q in queue if q.index in owned and (q.pilot or not cfg.pilot_only)]
    return [q for q in queue if q.index % cfg.n_workers == cfg.worker and (q.pilot or not cfg.pilot_only)]


def run_worker(cfg: WorkerConfig, queue: list[make_pools.QueueItem], prompts: dict[str, make_pools.PoolArm],
               adapter, *, sleep: Callable[[float], None] = time.sleep) -> str:
    """Run this worker's share of `queue`; returns ``"done"``, ``"budget"`` or ``"provider_down"``."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.log_dir / f"worker_{cfg.worker}.jsonl"
    _repair_own_tail(path)  # a complete record left unterminated by a kill must count before _progress reads
    for arm_id, prompt in prompts.items():
        adapter.register_prompt(arm_id, prompt.text)
    done, tries = _progress(cfg.log_dir)
    mine = _share(cfg, queue)

    recovery_failures = 0

    def recover() -> None:
        """Recover `adapter`, giving up the whole worker after too many failures in a row."""
        nonlocal recovery_failures
        try:
            _recover(adapter)
            recovery_failures = 0
        except Exception as err:  # noqa: BLE001 -- counted, then possibly escalated below
            recovery_failures += 1
            log.error("worker %d: recovery failed (%d/%d consecutive): %r", cfg.worker,
                      recovery_failures, MAX_RECOVERY_FAILURES, err)
            if recovery_failures >= MAX_RECOVERY_FAILURES:
                raise RuntimeError(
                    f"worker {cfg.worker} giving up after {recovery_failures} consecutive failed "
                    f"recoveries: {err!r}"
                ) from err

    episodes_since_recycle = 0
    provider_missing_streak = 0
    for item in mine:
        key = (item.arm_id, item.task_id, item.replicate)
        if key in done:
            continue
        attempt = tries.get(key, 0)
        while True:
            if spent_usd(cfg.log_dir) >= cfg.budget_usd:
                return OUTCOME_BUDGET
            attempt += 1
            prompt = prompts[item.arm_id]
            result, exc = None, None
            try:
                task = adapter.task_by_id(item.task_id)
                result = adapter.run_arm(prompt.arm, task, None, cfg.max_steps,
                                         artifact_subdir=f"r{item.replicate}_a{attempt}")
            except Exception as err:  # noqa: BLE001 -- any harness failure is infra, by definition
                exc = err
            episodes_since_recycle += 1
            status, reason = classify_detail(result, exc)
            if status == emp.STATUS_INFRA and attempt >= cfg.max_attempts:
                status = emp.STATUS_MISSING
            _append(path, _record(item, attempt, status, result, exc, cfg, prompt, reason))
            if status == emp.STATUS_INFRA:
                if reason == INFRA_PROVIDER:
                    wait = PROVIDER_BACKOFF_S[min(attempt - 1, len(PROVIDER_BACKOFF_S) - 1)]
                    log.warning("worker %d: %s/%s attempt %d ended in a provider-error streak; waiting %.0f s",
                                cfg.worker, item.arm_id, item.task_id, attempt, wait)
                    sleep(wait)
                log.warning("worker %d: %s/%s attempt %d infra error (%s); recovering", cfg.worker, item.arm_id,
                            item.task_id, attempt, reason)
                recover()
                episodes_since_recycle = 0
                continue
            if status == emp.STATUS_MISSING and reason == INFRA_PROVIDER:
                provider_missing_streak += 1
                if provider_missing_streak >= PROVIDER_DOWN_STREAK:
                    log.error("worker %d: %d consecutive items went missing on provider errors; stopping "
                              "(provider_down) so an outage cannot turn the queue into missing", cfg.worker,
                              provider_missing_streak)
                    return OUTCOME_PROVIDER_DOWN
            else:
                provider_missing_streak = 0
            if episodes_since_recycle >= RECYCLE_EVERY:
                log.info("worker %d: recycling server/browser after %d episodes", cfg.worker,
                         episodes_since_recycle)
                recover()
                episodes_since_recycle = 0
            break
    return OUTCOME_DONE


def _pgid_is_alive(pgid: int) -> bool:
    """Whether process GROUP `pgid` still has anything alive in it (`os.killpg(pgid, 0)`).

    A collector runs with ``start_new_session=True`` (the watchdog's `_launch`), so its own
    pid IS its pgid; checking the group rather than just that one pid also catches a worker
    that outlived its leader (fix round 1, item 3) -- something a single-pid `os.kill(pid, 0)`
    check cannot see, and exactly the gap that could let a live worker keep appending to a
    file `archive_out_of_queue` is mid-rewrite on.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # the group exists, just owned by someone else
    return True


def _acquire_lock(lock_path: Path) -> None:
    """Refuse to start a second collector against `lock_path`'s log dir.

    A lock naming a pid whose process group is still alive (`_pgid_is_alive`) means another
    collector run -- its leader, or a worker it spawned that survived it -- is still there. A
    lock naming a fully dead group is stale (its owner crashed or was killed without cleaning
    up) and is silently replaced.
    """
    if lock_path.exists():
        try:
            owner = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            owner = None
        if owner is not None and _pgid_is_alive(owner):
            raise RuntimeError(f"another collector (pid {owner}) holds {lock_path}; refusing to start a second one")
        log.warning("removing stale lock %s (pid %s is not running)", lock_path, owner)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def load_prompts(data_dir: Path) -> dict[str, make_pools.PoolArm]:
    out: dict[str, make_pools.PoolArm] = {}
    for name in ("G", "F", "anchor"):
        for arm in make_pools.load_pool(data_dir / f"pool_{name}.yaml"):
            if arm.arm.arm_id in out:
                raise RuntimeError(f"duplicate arm id {arm.arm.arm_id}")
            out[arm.arm.arm_id] = arm
    return out


def verify_queue(data_dir: Path) -> None:
    """Refuse to start unless ``queue.jsonl`` is byte-identical to the frozen one in ``manifest.json``."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    expected = manifest["files"]["queue.jsonl"]
    got = make_pools.file_sha256(data_dir / "queue.jsonl")
    if got != expected:
        raise RuntimeError(f"queue.jsonl sha256 {got} != manifest's frozen {expected}; refusing to start")


def partition_remaining(queue: list[make_pools.QueueItem], log_dir: Path, n_workers: int,
                        pilot: bool) -> list[tuple[int, ...]]:
    """Deal the items still without a terminal record round-robin (in queue order) to the workers.

    Each remaining item goes to exactly one worker for this launch, and every worker gets an
    equal share of what is actually left -- a static ``index % n`` split would leave a worker
    that died early with a long tail of its own items for the resumed run.
    """
    done, _ = _progress(log_dir)
    remaining = [q.index for q in queue
                 if (q.pilot or not pilot) and (q.arm_id, q.task_id, q.replicate) not in done]
    return [tuple(remaining[w::n_workers]) for w in range(n_workers)]


def final_status(outcomes: list[str]) -> str:
    """The collector's STATUS from its workers' outcomes.

    ``provider_down`` wins over everything, including ``failed``: during a provider outage a
    relaunch (what the watchdog does on ``failed``) would only turn more items into ``missing``;
    a person decides when to resume. ``failed`` then wins over ``budget``/``done``: a crash
    means an unknown slice of the queue was never safely attempted.
    """
    for status in (OUTCOME_PROVIDER_DOWN, OUTCOME_FAILED, OUTCOME_BUDGET):
        if status in outcomes:
            return status
    return OUTCOME_DONE


def exit_marker(log_dir: Path, worker: int) -> Path:
    return Path(log_dir) / f"worker_{worker}.exit"


def clear_exit_markers(log_dir: Path) -> None:
    for path in Path(log_dir).glob("worker_*.exit"):
        path.unlink(missing_ok=True)


def archive_out_of_queue(log_dir: Path, queue: list[make_pools.QueueItem]) -> dict[str, int]:
    """Archive already-collected records whose ``(arm_id, task_id, replicate)`` fell outside
    `queue` (amendment 1's 30-task subset) out of every ``worker_<w>.jsonl``.

    Each worker file is split into records still in `queue` (kept, rewritten in place) and
    records that are not (archived, appended verbatim -- original bytes, not re-serialized --
    to ``log_dir/archive/worker_<w>.pre_amendment_1.jsonl``). An unparseable trailing fragment
    (a torn write in flight) is dropped from both and logged. The rewrite is atomic: a temp
    file, then ``os.replace``. Idempotent: a second run finds nothing left to archive.

    Holds ``collect.lock`` for the whole operation (acquired/released via the collector's own
    `_acquire_lock`/`_release_lock`): refuses if a collector run's process GROUP -- its leader,
    or a worker it spawned that outlived it -- is still alive (`_pgid_is_alive`, not a single
    pid check), and stops a fresh collector from starting mid-rewrite.
    """
    log_dir = Path(log_dir)
    lock_path = log_dir / "collect.lock"
    _acquire_lock(lock_path)
    try:
        in_queue = {(q.arm_id, q.task_id, q.replicate) for q in queue}
        archive_dir = log_dir / "archive"
        counts = {"kept": 0, "archived": 0, "files": 0}
        for path in sorted(log_dir.glob("worker_*.jsonl")):
            data = path.read_bytes()
            *complete, tail = data.split(b"\n")
            if tail.strip():
                if _parses(tail):
                    complete.append(tail)
                else:
                    log.warning("dropping an unparseable trailing fragment in %s (%d bytes)", path, len(tail))
            kept_lines: list[bytes] = []
            archived_lines: list[bytes] = []
            for line in complete:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["arm_id"], record["task_id"], int(record["replicate"]))
                (kept_lines if key in in_queue else archived_lines).append(line)

            if archived_lines:
                archive_dir.mkdir(parents=True, exist_ok=True)
                archive_path = archive_dir / f"{path.stem}.pre_amendment_1.jsonl"
                with open(archive_path, "ab") as fh:
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


def _worker_main(worker: int, n_workers: int, data_dir: str, log_dir: str, budget: float, pilot: bool,
                 assigned: tuple[int, ...] | None = None) -> str:
    from dotenv import load_dotenv

    import cold_start.cli._bootstrap  # noqa: F401
    from cold_start.tasks.webarena import WebArenaInfinityAdapter

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{worker} %(levelname)s %(message)s")
    data, logs = Path(data_dir), Path(log_dir)
    queue = make_pools.read_queue(data / "queue.jsonl")
    prompts = load_prompts(data)
    adapter = WebArenaInfinityAdapter(port=BASE_PORT + worker, artifacts_dir=str(logs / "artifacts" / f"w{worker}"),
                                      axes_path=str(ROOT / "configs" / "axes.yaml"),
                                      template_path=str(ROOT / "configs" / "template.jinja"), **AGENT)
    try:
        # The first start takes the same reset-with-backoff path as every recovery: a server
        # slow to come up on a loaded machine must not kill the worker on one 15 s shot.
        _recover(adapter)
        if len(adapter.task_ids()) != N_BANK_TASKS:
            raise RuntimeError(f"task bank has {len(adapter.task_ids())} tasks; expected {N_BANK_TASKS}")
        check_bank(adapter, queue)
        cfg = WorkerConfig(worker=worker, n_workers=n_workers, log_dir=logs, budget_usd=budget, pilot_only=pilot,
                           assigned=assigned)
        return run_worker(cfg, queue, prompts, adapter)
    finally:
        try:
            adapter.close()
        except Exception as err:  # noqa: BLE001 -- shutdown cleanup must never mask a real worker error
            log.warning("worker %d: close() during shutdown raised: %r", worker, err)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", type=Path, default=make_pools.DATA_DIR)
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--budget", type=float, default=BUDGET_USD)
    ap.add_argument("--pilot", action="store_true", help="run only the pilot items")
    ap.add_argument("--archive-out-of-queue", action="store_true",
                    help="amendment 1: archive already-collected records outside the current "
                         "queue.jsonl, write STATUS paused, and exit -- never launches a worker")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.archive_out_of_queue:
        manifest = json.loads((args.data / "manifest.json").read_text())
        if "n_tasks" not in manifest:
            raise RuntimeError(f"{args.data / 'manifest.json'} has no \"n_tasks\": run "
                               "make_pools.py --requeue before --archive-out-of-queue (never "
                               "archive against the pre-amendment queue)")
        verify_queue(args.data)
        queue = make_pools.read_queue(args.data / "queue.jsonl")
        counts = archive_out_of_queue(args.log_dir, queue)
        print(json.dumps(counts))
        (args.log_dir / "STATUS").write_text("paused\n")
        return 0
    status_path = args.log_dir / "STATUS"
    lock_path = args.log_dir / "collect.lock"
    _acquire_lock(lock_path)
    status_path.write_text("running\n")
    final = OUTCOME_FAILED
    try:
        verify_queue(args.data)
        load_prompts(args.data)  # fail fast on a moved hash, before any server starts
        queue = make_pools.read_queue(args.data / "queue.jsonl")
        shares = partition_remaining(queue, args.log_dir, args.workers, args.pilot)
        log.info("%d items left; shares %s", sum(len(x) for x in shares), [len(x) for x in shares])
        clear_exit_markers(args.log_dir)
        ctx = mp_.get_context("spawn")
        outcomes: list[str] = []
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_worker_main, w, args.workers, str(args.data), str(args.log_dir), args.budget,
                            args.pilot, shares[w]): w
                for w in range(args.workers)
            }
            for fut in cf.as_completed(futures):
                w = futures[fut]
                try:
                    outcome = fut.result()
                except Exception as err:  # noqa: BLE001 -- covers a raised error and a BrokenProcessPool alike
                    log.error("worker %d failed: %r", w, err)
                    outcome = OUTCOME_FAILED
                outcomes.append(outcome)
                exit_marker(args.log_dir, w).write_text(outcome + "\n")
        final = final_status(outcomes)
        return EXIT_CODES[final]
    finally:
        status_path.write_text(final + "\n")
        try:
            log.info("collector finished: %s (spent $%.2f)", final, spent_usd(args.log_dir))
        except Exception as err:  # noqa: BLE001 -- a summary line must never mask the real error
            log.warning("collector finished: %s (could not sum spend: %r)", final, err)
        _release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
