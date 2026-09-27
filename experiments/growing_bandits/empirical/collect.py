"""Collect the real prompt pools' outcomes on WebArena Gmail (spec section 3.3).

    .venv/bin/python experiments/growing_bandits/empirical/collect.py --workers 8 [--pilot]

Worker w owns queue items with ``index % n_workers == w``, runs them in queue order on its
own WebArena server (port 8001 + w), and appends one JSON line per *attempt* to
``logs/empirical_pool/worker_<w>.jsonl``. Resuming is automatic: items with a terminal
record (``ok`` or ``missing``) are skipped, and an item's earlier ``infra_error`` attempts
count toward its three.

An agent timeout is a task failure, scored 0, as every historical Gmail run scored it. A
harness exception, or an LLM API error in the agent's trace while the agent was not done,
is ``infra_error``: the worker restarts its server and browser and retries, and the third
failure writes ``missing``, which estimation excludes and never scores 0.

Every worker re-reads the summed ``cost_usd`` of all workers before each attempt and stops
at the budget, so the overshoot is bounded by the episodes already in flight (8 x ~$0.05).

A worker recycles its server/browser (close + reset) every ``RECYCLE_EVERY`` episodes even
without an error, since the sibling server's stdout/stderr pipes are never drained and can
back up over a long-lived process. Recovery (after an infra error or a recycle) retries
``reset()`` with backoff; a worker that fails to recover several times in a row gives up
with a clear error rather than spinning forever. A pid lockfile refuses a second collector
against the same log dir while one is already running.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import multiprocessing as mp_
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_pools  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
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
    "Error code: 429", "Error code: 500", "Error code: 502", "Error code: 503", "Error code: 529", "overloaded",
    # browser-use wraps openai's APIConnectionError/APITimeoutError as ModelProviderError(message=str(e)),
    # which collapses to just these two sentences (openai/_exceptions.py APIConnectionError/APITimeoutError).
    "Connection error", "Request timed out", "Rate limit",
)
# A WebArena server's stdout/stderr pipes are never drained (webarena-infinity/evaluation/server.py);
# recycling periodically bounds how long any one server/browser process lives.
RECYCLE_EVERY = 100
# `reset()` retry policy for recovering a broken server/browser (after an infra error or a recycle).
RECOVERY_ATTEMPTS = 3
RECOVERY_BACKOFF_S: tuple[float, ...] = (10.0, 30.0, 60.0)
# Consecutive *failed* recoveries (not consecutive infra_error attempts) before a worker gives up.
MAX_RECOVERY_FAILURES = 3


@dataclass(frozen=True)
class WorkerConfig:
    worker: int
    n_workers: int
    log_dir: Path
    budget_usd: float
    max_attempts: int = MAX_ATTEMPTS
    pilot_only: bool = False
    max_steps: int = MAX_STEPS


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
    return msg is not None and any(marker in str(msg) for marker in API_ERROR_MARKERS)


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


def classify(result: RunResult | None, exc: BaseException | None) -> str:
    """``infra_error`` iff the harness raised, or the episode failed and ended in browser-use's
    give-up streak of provider errors; a single transient error the agent recovered from, or an
    agent timeout, is a scored task failure, not infra.
    """
    if exc is not None or result is None:
        return emp.STATUS_INFRA
    trace = result.trace or {}
    if trace.get("timed_out"):
        return emp.STATUS_OK
    if result.success:
        return emp.STATUS_OK
    tail = trace.get("error_tail") or []
    if _ends_in_provider_streak(tail, _default_failure_streak()):
        return emp.STATUS_INFRA
    return emp.STATUS_OK


def _repair_truncated_tail(path: Path) -> None:
    """Drop a final line left mid-write by a hard kill (no trailing newline, doesn't parse).

    A worker that dies mid-``_append`` leaves a byte fragment with no terminating newline;
    left alone, the next read chokes on it (``load_attempts`` is deliberately strict) and the
    next append glues a new record onto the fragment, corrupting both. A complete line that
    happens to be malformed (ends in a newline) is left untouched -- that is a real data bug,
    not a torn write, and must still raise.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    if not data or data.endswith(b"\n"):
        return
    last_nl = data.rfind(b"\n")
    tail = data[last_nl + 1 :]
    try:
        json.loads(tail.decode("utf-8"))
        return  # parses fine even without a trailing newline; nothing to repair
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.warning("dropping a truncated trailing line in %s (%d bytes)", path, len(tail))
        with open(path, "r+b") as fh:
            fh.truncate(last_nl + 1)


def _log_files(log_dir: Path) -> list[Path]:
    paths = sorted(Path(log_dir).glob("worker_*.jsonl"))
    for p in paths:
        _repair_truncated_tail(p)
    return paths


def spent_usd(log_dir: Path) -> float:
    attempts = emp.load_attempts(_log_files(log_dir))
    if attempts.empty:
        return 0.0
    return float(attempts["cost_usd"].fillna(0.0).astype(float).sum())


def _progress(log_dir: Path) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str, int], int]]:
    attempts = emp.load_attempts(_log_files(log_dir))
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


def _record(item, attempt: int, status: str, result: RunResult | None, exc: BaseException | None,
            cfg: WorkerConfig, prompt: make_pools.PoolArm) -> dict:
    trace = (result.trace if result is not None else {}) or {}
    tokens = (result.tokens if result is not None else {}) or {}
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
    _repair_truncated_tail(path)  # never glue a new record onto a torn prior write
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


def run_worker(cfg: WorkerConfig, queue: list[make_pools.QueueItem], prompts: dict[str, make_pools.PoolArm],
               adapter) -> str:
    """Run this worker's share of `queue`; returns ``"done"`` or ``"budget"``."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.log_dir / f"worker_{cfg.worker}.jsonl"
    for arm_id, prompt in prompts.items():
        adapter.register_prompt(arm_id, prompt.text)
    done, tries = _progress(cfg.log_dir)
    mine = [q for q in queue if q.index % cfg.n_workers == cfg.worker and (q.pilot or not cfg.pilot_only)]

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
    for item in mine:
        key = (item.arm_id, item.task_id, item.replicate)
        if key in done:
            continue
        attempt = tries.get(key, 0)
        while True:
            if spent_usd(cfg.log_dir) >= cfg.budget_usd:
                return "budget"
            attempt += 1
            prompt = prompts[item.arm_id]
            result, exc = None, None
            try:
                task = adapter.task_by_id(item.task_id)
                result = adapter.run_arm(prompt.arm, task, None, cfg.max_steps)
            except Exception as err:  # noqa: BLE001 -- any harness failure is infra, by definition
                exc = err
            episodes_since_recycle += 1
            status = classify(result, exc)
            if status == emp.STATUS_INFRA and attempt >= cfg.max_attempts:
                status = emp.STATUS_MISSING
            _append(path, _record(item, attempt, status, result, exc, cfg, prompt))
            if status == emp.STATUS_INFRA:
                log.warning("worker %d: %s/%s attempt %d infra error; recovering", cfg.worker, item.arm_id,
                            item.task_id, attempt)
                recover()
                episodes_since_recycle = 0
                continue
            if episodes_since_recycle >= RECYCLE_EVERY:
                log.info("worker %d: recycling server/browser after %d episodes", cfg.worker,
                         episodes_since_recycle)
                recover()
                episodes_since_recycle = 0
            break
    return "done"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just owned by someone else
    return True


def _acquire_lock(lock_path: Path) -> None:
    """Refuse to start a second collector against `lock_path`'s log dir.

    A lock naming a live pid means another collector is already running there. A lock
    naming a dead pid is stale (its owner crashed or was killed without cleaning up) and is
    silently replaced.
    """
    if lock_path.exists():
        try:
            owner = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            owner = None
        if owner is not None and _pid_is_alive(owner):
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


def _worker_main(worker: int, n_workers: int, data_dir: str, log_dir: str, budget: float, pilot: bool) -> str:
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
    adapter.reset(seed=0)
    try:
        if len(adapter.task_ids()) != N_BANK_TASKS:
            raise RuntimeError(f"task bank has {len(adapter.task_ids())} tasks; expected {N_BANK_TASKS}")
        check_bank(adapter, queue)
        cfg = WorkerConfig(worker=worker, n_workers=n_workers, log_dir=logs, budget_usd=budget, pilot_only=pilot)
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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / "STATUS"
    lock_path = args.log_dir / "collect.lock"
    _acquire_lock(lock_path)
    status_path.write_text("running\n")
    # A crashed worker (OOM, segfault, SIGKILL) must never be hidden behind a "budget" stop from
    # some other worker: "budget" is an expected, benign stop a person would resume from without a
    # second look, while a crash means an unknown slice of the queue was never safely attempted.
    final = "failed"
    try:
        load_prompts(args.data)  # fail fast on a moved hash, before any server starts
        ctx = mp_.get_context("spawn")
        outcomes: list[str] = []
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(_worker_main, w, args.workers, str(args.data), str(args.log_dir), args.budget,
                            args.pilot): w
                for w in range(args.workers)
            }
            for fut in cf.as_completed(futures):
                w = futures[fut]
                try:
                    outcomes.append(fut.result())
                except Exception as err:  # noqa: BLE001 -- covers a raised error and a BrokenProcessPool alike
                    log.error("worker %d failed: %r", w, err)
                    outcomes.append("failed")
        final = "failed" if "failed" in outcomes else ("budget" if "budget" in outcomes else "done")
        return {"done": 0, "budget": 3, "failed": 1}[final]
    finally:
        status_path.write_text(final + "\n")
        log.info("collector finished: %s (spent $%.2f)", final, spent_usd(args.log_dir))
        _release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
