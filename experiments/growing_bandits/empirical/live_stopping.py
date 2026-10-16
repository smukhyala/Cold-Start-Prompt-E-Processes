"""Durable, sequential, K=1 live evaluation controller.

``run(config, directory, tasks, evaluate)`` calls ``evaluate(task, role,
episode_id)`` synchronously, baseline first.  The callback must return a JSON
object with status="ok", binary success, cost_usd, input_tokens, output_tokens,
model, and prompt_sha256. Extra fields (provider request IDs, timestamps, traces,
environment metadata) are preserved. Provider/infrastructure errors must use
their own status and any known accounting; they never become task failures.

Config requires model, prompts={role: {text, sha256}}, max_usd, and a conservative
episode_cost_bound_usd (budget_mode="episode_bound", default). Alternatively,
budget_mode="transport" delegates the cap to a frozen HTTP transport that reserves
the maximum possible charge before each request; minimum_request_reservation_usd
is then only the minimum reserve needed to start an episode, NOT its cost bound.
frozen_files maps file paths to SHA256 digests. Additional
execution settings are included in the immutable registration. Tasks are frozen
in order, including all IDs/seeds supplied by the generator.

The two existing offline functions are called after EVERY completed pair, before
another execution can start. This deliberately avoids a second implementation
of their mathematics. Receipts are authoritative; checkpoints are reconstructible.
An interrupted callback without a durable receipt is ambiguous and permanently
halts automatic scheduling. External exactly-once execution is impossible without
provider idempotency; this controller never retries an ambiguous paid request.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths

ROLES = ("baseline", "challenger")
DEFAULTS = {
    "max_pairs": 120,
    "delta": 0.15,
    "alpha_e": 0.025,
    "alpha_f": 0.025,
    "boundary_precedence": "efficacy",
    "automatic_retries": 0,
    "budget_mode": "episode_bound",
    "frozen_files": {},
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, value) -> None:
    """Atomic replacement with file and containing-directory durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path: Path):
    return json.loads(path.read_text())


@contextmanager
def _lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another live controller holds this experiment lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize(config: dict, tasks: list[dict]) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(copy.deepcopy(config))
    if type(cfg["max_pairs"]) is not int or cfg["max_pairs"] < 1:
        raise ValueError("max_pairs must be a positive integer")
    for name in ("delta", "alpha_e", "alpha_f"):
        if not _number(cfg[name]) or not 0 < cfg[name] < 1:
            raise ValueError(f"{name} must be between zero and one")
    if cfg["budget_mode"] not in ("episode_bound", "transport"):
        raise ValueError("budget_mode must be episode_bound or transport")
    reserve_key = ("episode_cost_bound_usd" if cfg["budget_mode"] == "episode_bound"
                   else "minimum_request_reservation_usd")
    for name in ("max_usd", reserve_key):
        if not _number(cfg.get(name)) or cfg[name] < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if cfg[reserve_key] <= 0:
        raise ValueError(f"{reserve_key} must be positive")
    if cfg["boundary_precedence"] != "efficacy" or cfg["automatic_retries"] != 0:
        raise ValueError("Only efficacy precedence and zero automatic retries are implemented")
    if not isinstance(cfg.get("model"), str) or not cfg["model"]:
        raise ValueError("A frozen model identifier is required")
    if len(tasks) != cfg["max_pairs"] or not all(isinstance(t, dict) for t in tasks):
        raise ValueError("Supply exactly max_pairs complete task metadata objects")
    for role in ROLES:
        prompt = cfg.get("prompts", {}).get(role, {})
        if not isinstance(prompt.get("text"), str):
            raise ValueError(f"Missing {role} prompt text")
        if hashlib.sha256(prompt["text"].encode()).hexdigest() != prompt.get("sha256"):
            raise ValueError(f"Incorrect {role} prompt hash")
    canonical_sha256({"config": cfg, "tasks": tasks})  # Reject non-JSON or nonfinite data.
    return cfg


def verify_frozen_files(config: dict) -> None:
    for name, expected in config["frozen_files"].items():
        if file_sha256(name) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def evidence_paths(pairs: list[dict], config: dict) -> list[dict]:
    """Exact offline implementation, recomputed on the observed prefix only."""
    if not pairs:
        return []
    outcomes = np.array([[p["baseline_success"] for p in pairs],
                         [p["challenger_success"] for p in pairs]], dtype=np.int8)
    efficacy = e1b_log_paths(outcomes[None, ...], np.array([0]),
                            np.array([[False, True]]))[0]
    difference = (outcomes[1] - outcomes[0])[None, None, :]
    futility = futility_log_paths(difference, config["delta"])[0, 0]
    return [{"pair_index": i + 1, "log_efficacy": float(e), "log_futility": float(f),
             "efficacy_crossed": bool(e >= math.log(1 / config["alpha_e"])),
             "futility_crossed": bool(f >= math.log(1 / config["alpha_f"]))}
            for i, (e, f) in enumerate(zip(efficacy, futility, strict=True))]


def _accounted(result: dict) -> bool:
    return (_number(result.get("cost_usd")) and result["cost_usd"] >= 0
            and result.get("cost_accounting_complete", True) is True
            and not result.get("unknown_billing_requests"))


def _result_error(result: dict, config: dict, role: str) -> str | None:
    status = result.get("status")
    if status in ("provider_error", "infrastructure_error", "budget_exhausted"):
        return status
    if status != "ok":
        return "invalid_episode_status"
    if not _number(result.get("cost_usd")) or result["cost_usd"] < 0:
        return "missing_or_invalid_cost_accounting"
    if not _accounted(result):
        return "incomplete_cost_accounting"
    for field in ("input_tokens", "output_tokens"):
        if type(result.get(field)) is not int or result[field] < 0:
            return "missing_or_invalid_token_accounting"
    if type(result.get("success")) not in (int, bool) or result["success"] not in (0, 1):
        return "missing_or_invalid_binary_outcome"
    if result.get("model") != config["model"]:
        return "model_identifier_mismatch"
    if result.get("prompt_sha256") != config["prompts"][role]["sha256"]:
        return "prompt_hash_mismatch"
    if (config["budget_mode"] == "episode_bound"
            and result["cost_usd"] > config["episode_cost_bound_usd"] + 1e-12):
        return "episode_cost_bound_exceeded"
    return None


def _episode_id(index: int, role: str) -> str:
    return f"pair_{index:06d}_{role}"


def _reservation(config: dict) -> float:
    key = ("episode_cost_bound_usd" if config["budget_mode"] == "episode_bound"
           else "minimum_request_reservation_usd")
    return config[key]


def _records(directory: Path, frozen_sha: str, tasks: list[dict]):
    intents, receipts = {}, {}
    for kind, destination in (("intents", intents), ("receipts", receipts)):
        for path in sorted((directory / kind).glob("*.json")):
            value = _read(path)
            index, role = value["pair_index"], value["role"]
            if (role not in ROLES or not 1 <= index <= len(tasks)
                    or path.stem != _episode_id(index, role)
                    or value["episode_id"] != path.stem
                    or value["frozen_sha256"] != frozen_sha
                    or value["task_sha256"] != canonical_sha256(tasks[index - 1])):
                raise ValueError(f"Inconsistent durable {kind} record: {path}")
            destination[path.stem] = value
    if receipts.keys() - intents.keys():
        raise ValueError("A receipt has no corresponding execution intent")
    return intents, receipts


def _checkpoint(directory: Path, cfg: dict, tasks: list[dict], frozen_sha: str,
                recovered: bool = False) -> dict:
    intents, receipts = _records(directory, frozen_sha, tasks)
    pairs = []
    error = None
    for index in range(1, len(tasks) + 1):
        halves = [receipts.get(_episode_id(index, role)) for role in ROLES]
        for role, half in zip(ROLES, halves, strict=True):
            if half is not None:
                error = error or _result_error(half["result"], cfg, role)
        if error or not all(halves):
            break
        pairs.append({"pair_index": index, "task": tasks[index - 1],
                      "baseline_success": int(halves[0]["result"]["success"]),
                      "challenger_success": int(halves[1]["result"]["success"]),
                      "baseline_episode_id": halves[0]["episode_id"],
                      "challenger_episode_id": halves[1]["episode_id"],
                      "cost_usd": math.fsum(h["result"]["cost_usd"] for h in halves),
                      "completed_at": halves[1]["completed_at"]})
    path = evidence_paths(pairs, cfg)
    crossing = next((point for point in path
                     if point["efficacy_crossed"] or point["futility_crossed"]), None)
    ambiguous = sorted(intents.keys() - receipts.keys())
    known_costs = [r["result"]["cost_usd"] for r in receipts.values()
                   if _number(r["result"].get("cost_usd")) and r["result"]["cost_usd"] >= 0]
    spent = math.fsum(known_costs)
    terminal = _read(directory / "terminal.json") if (directory / "terminal.json").exists() else None
    if terminal is None:
        reason, decision = None, "unresolved"
        if crossing:
            decision = "efficacy" if crossing["efficacy_crossed"] else "futility"
            reason = "statistical_boundary"
        elif ambiguous:
            reason = "ambiguous_inflight_episode"
        elif error:
            reason = error
        elif spent > cfg["max_usd"] + 1e-12:
            reason = "usd_cap_exceeded"
        elif len(pairs) == cfg["max_pairs"]:
            reason = "max_pairs"
        elif spent + 2 * _reservation(cfg) > cfg["max_usd"] + 1e-12:
            # A completed first half already owns its pair's reservation.
            baseline_done = _episode_id(len(pairs) + 1, "baseline") in receipts
            if not baseline_done or spent + _reservation(cfg) > cfg["max_usd"] + 1e-12:
                reason = "usd_budget"
        if reason:
            terminal = {"decision": decision, "reason": reason,
                        "completed_pairs": len(pairs), "first_crossing": crossing,
                        "recorded_at": _now(), "recovered_before_scheduling": recovered,
                        "frozen_sha256": frozen_sha}
            _atomic_json(directory / "terminal.json", terminal)
    state = {
        "schema": "live_stopping/1", "frozen_sha256": frozen_sha,
        "status": "terminal" if terminal else "running",
        "decision": terminal["decision"] if terminal else None,
        "terminal": terminal, "completed_pairs": len(pairs),
        "scheduled_episodes": len(intents), "recorded_episodes": len(receipts),
        "valid_outcome_episodes": sum(_result_error(r["result"], cfg, r["role"]) is None
                                      for r in receipts.values()),
        "ambiguous_episode_ids": ambiguous, "known_cost_usd": spent,
        "cost_accounting_complete": (len(receipts) == len(intents)
                                     and all(_accounted(r["result"]) for r in receipts.values())),
        "token_accounting_complete": (len(receipts) == len(intents)
                                      and all(_accounted(r["result"])
                                              and all(type(r["result"].get(k)) is int
                                                      and r["result"][k] >= 0
                                                      for k in ("input_tokens", "output_tokens"))
                                              for r in receipts.values())),
        "input_tokens": sum(r["result"].get("input_tokens", 0) for r in receipts.values()
                            if type(r["result"].get("input_tokens")) is int),
        "output_tokens": sum(r["result"].get("output_tokens", 0) for r in receipts.values()
                             if type(r["result"].get("output_tokens")) is int),
        "max_planned_episodes": 2 * cfg["max_pairs"],
        "unscheduled_planned_episodes": 2 * cfg["max_pairs"] - len(intents),
        "log_threshold_efficacy": math.log(1 / cfg["alpha_e"]),
        "log_threshold_futility": math.log(1 / cfg["alpha_f"]),
        "last_evidence": path[-1] if path else {"log_efficacy": 0.0, "log_futility": 0.0},
        "updated_at": _now(),
    }
    _atomic_json(directory / "pairs.json", pairs)
    _atomic_json(directory / "evidence.json", path)
    _atomic_json(directory / "state.json", state)
    return state


def run(config: dict, directory: Path | str, tasks: list[dict],
        evaluate: Callable[[dict, str, str], dict]) -> dict:
    """Execute until a registered boundary, budget, or infrastructure stop.

    No provider calls occur for an existing terminal experiment. A durable receipt
    permits exact continuation after an interruption between episodes; an intent
    without a receipt does not. Unknown charges remain explicitly unknown.
    """
    directory = Path(directory)
    tasks = copy.deepcopy(tasks)
    cfg = _normalize(config, tasks)
    frozen = {"config": cfg, "tasks": tasks}
    frozen_sha = canonical_sha256(frozen)
    with _lock(directory):
        registration_path = directory / "registration.json"
        if registration_path.exists():
            registration = _read(registration_path)
            if (registration["frozen_sha256"] != frozen_sha
                    or canonical_sha256(registration["frozen"]) != frozen_sha):
                raise ValueError("Frozen configuration or task population differs from registration")
        else:
            verify_frozen_files(cfg)
            _atomic_json(registration_path, {"registered_at": _now(),
                                           "frozen_sha256": frozen_sha, "frozen": frozen})
        state = _checkpoint(directory, cfg, tasks, frozen_sha, recovered=True)
        if state["status"] == "terminal":
            return state
        verify_frozen_files(cfg)
        for index in range(state["completed_pairs"] + 1, cfg["max_pairs"] + 1):
            for role in ROLES:
                episode_id = _episode_id(index, role)
                receipt_path = directory / "receipts" / f"{episode_id}.json"
                if receipt_path.exists():
                    continue
                # Check source hashes before every episode, including between halves.
                verify_frozen_files(cfg)
                intent = {"episode_id": episode_id, "pair_index": index, "role": role,
                          "task_sha256": canonical_sha256(tasks[index - 1]),
                          "frozen_sha256": frozen_sha, "model": cfg["model"],
                          "prompt_sha256": cfg["prompts"][role]["sha256"],
                          "budget_mode": cfg["budget_mode"],
                          "episode_start_reservation_usd": _reservation(cfg),
                          "started_at": _now()}
                _atomic_json(directory / "intents" / f"{episode_id}.json", intent)
                try:
                    result = evaluate(copy.deepcopy(tasks[index - 1]), role, episode_id)
                    if not isinstance(result, dict):
                        raise TypeError("Evaluation callback must return a JSON object")
                    canonical_sha256(result)
                except Exception as exc:
                    result = {"status": "infrastructure_error", "success": None,
                              "cost_usd": None, "error_type": type(exc).__name__,
                              "error": str(exc)}
                _atomic_json(receipt_path, {**intent, "completed_at": _now(), "result": result})
                # Checkpoints after each half ensure errors/cost overruns stop immediately;
                # evidence changes only when both valid halves have been durably saved.
                state = _checkpoint(directory, cfg, tasks, frozen_sha)
                if state["status"] == "terminal":
                    return state
        return state
