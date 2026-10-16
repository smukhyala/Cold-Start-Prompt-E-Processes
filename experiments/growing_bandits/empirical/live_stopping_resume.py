"""Explicit, one-time continuation amendment for the failed pair-14 challenger.

The original terminal experiment remains immutable. A separate effective view
copies its 13 complete pairs and valid pair-14 baseline; only the failed
challenger is evaluated again, under the physical ID ending in ``_retry1``.
All statistical code, parameters, prompts, tasks, and subsequent task order come
from the ORIGINAL registration. This is a post-outcome amendment, not execution
under the original stopping/error protocol.

``initialize_continuation(manifest_path, expected_sha256)`` returns
``(original_config, original_tasks, continuation_directory)``. The paid runner
must enforce cumulative spending over BOTH attempts, retaining the original
unknown request's reserve. This module does not implement provider accounting.
``run_continuation(..., evaluate)`` validates frozen source and amendment files
before every callback and delegates scheduling to the unchanged controller.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.growing_bandits.empirical import live_stopping as live

LOGICAL_EPISODE_ID = "pair_000014_challenger"
PHYSICAL_EPISODE_ID = LOGICAL_EPISODE_ID + "_retry1"


def _read(path: Path):
    return json.loads(path.read_text())


def validate_manifest(manifest_path: Path | str, expected_sha256: str) -> dict:
    """Read-only guard; the externally supplied manifest hash pins this amendment."""
    path = Path(manifest_path).resolve()
    if live.file_sha256(path) != expected_sha256:
        raise ValueError("Continuation amendment manifest hash mismatch")
    manifest = _read(path)
    if (manifest["logical_episode_id"] != LOGICAL_EPISODE_ID
            or manifest["physical_episode_id"] != PHYSICAL_EPISODE_ID
            or type(manifest["max_retry_attempts"]) is not int
            or manifest["max_retry_attempts"] != 1):
        raise ValueError("Only one fresh attempt of the pair-14 challenger is authorized")
    source = Path(manifest["source_directory"]).resolve()
    destination = Path(manifest["continuation_directory"]).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Original and continuation directories must be separate, non-nested paths")
    for category in ("source_files", "amendment_files"):
        if not isinstance(manifest[category], dict) or not manifest[category]:
            raise ValueError(f"Nonempty {category} hash mapping required")
        for name, expected in manifest[category].items():
            if not Path(name).is_absolute() or live.file_sha256(name) != expected:
                raise ValueError(f"Frozen continuation/source file changed: {name}")
    if str(Path(__file__).resolve()) not in manifest["amendment_files"]:
        raise ValueError("The continuation controller itself must be frozen")
    return manifest


def _source_view(manifest: dict):
    source = Path(manifest["source_directory"]).resolve()
    required = {source / name for name in ("registration.json", "terminal.json", "state.json",
                                           "pairs.json", "evidence.json", "api_calls.jsonl")}
    required.update((source / "intents").glob("*.json"))
    required.update((source / "receipts").glob("*.json"))
    required.update((source / "episodes").glob("*/intent.json"))
    required.update((source / "episodes").glob("*/receipt.json"))
    registration = _read(source / "registration.json")
    cfg, tasks = registration["frozen"]["config"], registration["frozen"]["tasks"]
    if cfg.get("task_suite"):
        task_path = Path(cfg["task_suite"] + ".json").resolve()
        required.update((task_path, task_path.with_name("config.json"), task_path.with_name("manifest.json")))
    if any(str(p.resolve()) not in manifest["source_files"] for p in required):
        raise ValueError("Source manifest must include all original state, episode, and task/config files")
    if (registration["frozen_sha256"] != manifest["source_frozen_sha256"]
            or live.canonical_sha256(registration["frozen"]) != registration["frozen_sha256"]):
        raise ValueError("Original registration differs from the amendment's source")
    live.verify_frozen_files(cfg)
    if cfg["budget_mode"] != "transport" or cfg["max_usd"] != 75.0:
        raise ValueError("The original cumulative transport budget must remain USD 75")
    intents, receipts = live._records(source, registration["frozen_sha256"], tasks)
    expected_ids = {live._episode_id(i, role) for i in range(1, 15) for role in live.ROLES}
    if set(receipts) != expected_ids or set(intents) != expected_ids:
        raise ValueError("Source must contain exactly the original 28 episode intents and receipts")
    failed = receipts[LOGICAL_EPISODE_ID]
    if live._result_error(failed["result"], cfg, "challenger") is None:
        raise ValueError("The selected original challenger was valid; no retry is authorized")
    if (not manifest["acknowledged_unknown_request_ids"]
            or sorted(manifest["acknowledged_unknown_request_ids"])
            != sorted(failed["result"].get("unknown_billing_requests", []))):
        raise ValueError("Amendment must identify the exact original ambiguous provider request")
    valid_ids = sorted(expected_ids - {LOGICAL_EPISODE_ID})
    for episode_id in valid_ids:
        record = receipts[episode_id]
        if live._result_error(record["result"], cfg, record["role"]) is not None:
            raise ValueError("Only the pair-14 challenger may be invalid")
    pairs = _read(source / "pairs.json")
    if len(pairs) != 13 or [p["pair_index"] for p in pairs] != list(range(1, 14)):
        raise ValueError("Exactly 13 original complete pairs must precede the retry")
    for pair in pairs:
        if pair["task"] != tasks[pair["pair_index"] - 1]:
            raise ValueError("Original paired task differs from the frozen task sequence")
        for role in live.ROLES:
            half = receipts[live._episode_id(pair["pair_index"], role)]
            if pair[f"{role}_success"] != half["result"]["success"]:
                raise ValueError("Original pair outcome differs from the durable receipt")
    evidence = live.evidence_paths(pairs, cfg)
    if evidence != _read(source / "evidence.json") or any(
            p["efficacy_crossed"] or p["futility_crossed"] for p in evidence):
        raise ValueError("Original evidence must reproduce exactly without a prior boundary")
    terminal, state = _read(source / "terminal.json"), _read(source / "state.json")
    if (terminal["decision"] != "unresolved" or terminal["completed_pairs"] != 13
            or terminal["first_crossing"] is not None
            or terminal["reason"] not in ("infrastructure_error", "provider_error")
            or state["terminal"] != terminal or state["completed_pairs"] != 13):
        raise ValueError("The source must retain its original unresolved operational terminal")
    return registration, cfg, tasks, intents, receipts, valid_ids


def initialize_continuation(manifest_path: Path | str, expected_sha256: str):
    """Create/resume the effective view without touching the original experiment."""
    manifest = validate_manifest(manifest_path, expected_sha256)
    registration, cfg, tasks, intents, receipts, valid_ids = _source_view(manifest)
    destination = Path(manifest["continuation_directory"]).resolve()
    marker = {"schema": "live_stopping_continuation/1", "manifest_path": str(Path(manifest_path).resolve()),
              "manifest_sha256": expected_sha256, "source_directory": manifest["source_directory"],
              "source_frozen_sha256": manifest["source_frozen_sha256"],
              "logical_episode_id": LOGICAL_EPISODE_ID, "physical_episode_id": PHYSICAL_EPISODE_ID,
              "copied_valid_episode_ids": valid_ids,
              "excluded_original_attempt": receipts[LOGICAL_EPISODE_ID],
              "original_terminal": _read(Path(manifest["source_directory"]) / "terminal.json")}
    with live._lock(destination):
        marker_path = destination / "continuation.json"
        if marker_path.exists():
            existing = _read(marker_path)
            if {k: v for k, v in existing.items() if k != "stage"} != marker:
                raise ValueError("Continuation directory belongs to another amendment")
            if existing.get("stage") not in ("initializing", "ready"):
                raise ValueError("Invalid continuation initialization state")
        else:
            if any((destination / name).exists() for name in
                   ("registration.json", "intents", "receipts", "terminal.json", "state.json")):
                raise ValueError("Refusing to overwrite an existing experiment without amendment provenance")
            live._atomic_json(marker_path, {**marker, "stage": "initializing"})
        copies = {destination / "registration.json": registration}
        for episode_id in valid_ids:
            copies[destination / "intents" / f"{episode_id}.json"] = intents[episode_id]
            copies[destination / "receipts" / f"{episode_id}.json"] = receipts[episode_id]
        for target, value in copies.items():
            if target.exists():
                if _read(target) != value:
                    raise ValueError(f"Copied original record was altered: {target}")
            else:
                live._atomic_json(target, value)
        if not marker_path.exists() or _read(marker_path)["stage"] != "ready":
            live._atomic_json(marker_path, {**marker, "stage": "ready"})
    return copy.deepcopy(cfg), copy.deepcopy(tasks), destination


def run_continuation(manifest_path: Path | str, expected_sha256: str, evaluate):
    """Run the sole authorized retry and then the untouched original task order.

    ``evaluate`` receives the physical episode ID; returned receipts retain that
    physical identity and add explicit logical identity for paired analysis.
    The transport, not the effective-pair cost subtotal, enforces the cumulative
    original USD cap including the excluded attempt and unknown reserve.
    """
    cfg, tasks, destination = initialize_continuation(manifest_path, expected_sha256)

    def amended_evaluate(task, role, logical_episode_id):
        validate_manifest(manifest_path, expected_sha256)
        physical_episode_id = (PHYSICAL_EPISODE_ID if logical_episode_id == LOGICAL_EPISODE_ID
                               else logical_episode_id)
        result = evaluate(task, role, physical_episode_id)
        if not isinstance(result, dict):
            raise TypeError("Amended evaluator must return a receipt object")
        if result.get("episode_id") != physical_episode_id:
            raise ValueError("Returned physical episode identity does not match the authorized attempt")
        return {**result, "logical_episode_id": logical_episode_id,
                "physical_episode_id": physical_episode_id,
                "continuation_manifest_sha256": expected_sha256}

    return live.run(cfg, destination, tasks, amended_evaluate)
