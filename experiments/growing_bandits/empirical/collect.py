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
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import multiprocessing as mp_
import sys
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
)


@dataclass(frozen=True)
class WorkerConfig:
    worker: int
    n_workers: int
    log_dir: Path
    budget_usd: float
    max_attempts: int = MAX_ATTEMPTS
    pilot_only: bool = False
    max_steps: int = MAX_STEPS


def classify(result: RunResult | None, exc: BaseException | None) -> str:
    if exc is not None or result is None:
        return emp.STATUS_INFRA
    trace = result.trace or {}
    if trace.get("timed_out"):
        return emp.STATUS_OK
    errors = " ".join(str(e) for e in trace.get("errors") or [])
    if not result.success and not trace.get("is_done", False) and any(m in errors for m in API_ERROR_MARKERS):
        return emp.STATUS_INFRA
    return emp.STATUS_OK


def _log_files(log_dir: Path) -> list[Path]:
    return sorted(Path(log_dir).glob("worker_*.jsonl"))


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
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _recover(adapter) -> None:
    try:
        adapter.close()
    finally:
        adapter.reset(seed=0)


def run_worker(cfg: WorkerConfig, queue: list[make_pools.QueueItem], prompts: dict[str, make_pools.PoolArm],
               adapter) -> str:
    """Run this worker's share of `queue`; returns ``"done"`` or ``"budget"``."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.log_dir / f"worker_{cfg.worker}.jsonl"
    for arm_id, prompt in prompts.items():
        adapter.register_prompt(arm_id, prompt.text)
    done, tries = _progress(cfg.log_dir)
    mine = [q for q in queue if q.index % cfg.n_workers == cfg.worker and (q.pilot or not cfg.pilot_only)]
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
            status = classify(result, exc)
            if status == emp.STATUS_INFRA and attempt >= cfg.max_attempts:
                status = emp.STATUS_MISSING
            _append(path, _record(item, attempt, status, result, exc, cfg, prompt))
            if status != emp.STATUS_INFRA:
                break
            log.warning("worker %d: %s/%s attempt %d infra error; recovering", cfg.worker, item.arm_id,
                        item.task_id, attempt)
            _recover(adapter)
    return "done"


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
        adapter.close()


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
    status_path.write_text("running\n")
    load_prompts(args.data)  # fail fast on a moved hash, before any server starts
    ctx = mp_.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = [pool.apply_async(_worker_main, (w, args.workers, str(args.data), str(args.log_dir),
                                                   args.budget, args.pilot)) for w in range(args.workers)]
        outcomes = []
        for w, r in enumerate(results):
            try:
                outcomes.append(r.get())
            except Exception as err:  # noqa: BLE001
                log.error("worker %d failed: %r", w, err)
                outcomes.append("failed")
    final = "budget" if "budget" in outcomes else ("failed" if "failed" in outcomes else "done")
    status_path.write_text(final + "\n")
    log.info("collector finished: %s (spent $%.2f)", final, spent_usd(args.log_dir))
    return {"done": 0, "budget": 3, "failed": 1}[final]


if __name__ == "__main__":
    raise SystemExit(main())
