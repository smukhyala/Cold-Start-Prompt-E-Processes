"""Audit the terminal study without provider calls or changes to execution logs.

Controller replays run on temporary copies with a callback that always raises.
This is a post-outcome audit, not an execution controller or protocol amendment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.growing_bandits.empirical.analyze_live_stopping import analyze  # noqa: E402
from experiments.growing_bandits.empirical.live_stopping import run  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def digests(directory):
    paths = [directory / name for name in (
        "registration.json", "terminal.json", "state.json", "pairs.json",
        "evidence.json", "api_calls.jsonl")]
    paths.extend((directory / "intents").glob("*.json"))
    paths.extend((directory / "receipts").glob("*.json"))
    paths.extend((directory / "episodes").glob("*/receipt.json"))
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def audit(directory: Path, results: Path, out: Path):
    started = datetime.now(UTC).isoformat()
    before = digests(directory)
    registration = read(directory / "registration.json")
    config = registration["frozen"]["config"]
    tasks = registration["frozen"]["tasks"]
    summary = read(results / "summary.json")
    calls = []

    def forbid_evaluation(task, role, episode_id):
        calls.append(episode_id)
        raise AssertionError("No evaluation is permitted during this audit")

    with tempfile.TemporaryDirectory(prefix="live-stopping-resume-audit-") as tmp:
        temporary = Path(tmp)
        reproduced = analyze(directory, temporary / "reanalysis")
        assert reproduced == summary, "Reanalysis differs from published summary"
        clone = temporary / "controller-copy"
        clone.mkdir()
        for name in ("registration.json", "terminal.json", "state.json", "pairs.json", "evidence.json"):
            shutil.copy2(directory / name, clone / name)
        for name in ("intents", "receipts"):
            shutil.copytree(directory / name, clone / name)
        resumed = run(config, clone, tasks, forbid_evaluation)
        assert resumed["terminal"] == read(directory / "terminal.json")
        assert not calls
        # Demonstrate that the invalid receipt independently enforces the halt.
        # Only the disposable copy is modified; real evidence is never edited.
        (clone / "terminal.json").unlink()
        reconstructed = run(config, clone, tasks, forbid_evaluation)
        assert reconstructed["terminal"]["reason"] == "infrastructure_error"
        assert reconstructed["completed_pairs"] == 13
        assert not calls

    failed = read(directory / "receipts" / "pair_000014_challenger.json")["result"]
    assert failed["status"] == "infrastructure_error" and failed["success"] is None
    assert failed["unknown_billing_requests"] == ["request_000458"]
    assert failed == read(directory / "episodes" / "pair_000014_challenger" / "receipt.json")
    ledger = [json.loads(line) for line in (directory / "api_calls.jsonl").read_text().splitlines()]
    unknown = [r for r in ledger if r["request_id"] == "request_000458"]
    assert [r["event"] for r in unknown] == ["request_started", "request_unknown", "invocation_failed"]
    assert all(not r.get("completion_id") and not r.get("provider_request_id") for r in unknown)
    assert before == digests(directory), "Original execution records were changed"
    known = Decimal(str(summary["measured_api_cost_usd"]))
    reserved = Decimal(str(summary["unknown_billing_reserve_usd"]))
    account = read(results / "operational_accounting.json")
    # Includes one hypothetical repeat of the failed challenger and 212 unstarted
    # episodes. This is accounting context only: such continuation is not allowed.
    hypothetical_remaining = 213 * account["mean_fully_accounted_episode_cost_usd"]
    report = {
        "audit_started_utc": started,
        "audit_finished_utc": datetime.now(UTC).isoformat(),
        "status": "blocked_by_registered_terminal_infrastructure_stop",
        "paid_continuation_started": False,
        "new_agent_episodes": 0,
        "new_inference_requests": 0,
        "new_inference_cost_usd": 0,
        "controller_replay_location": "disposable copies only",
        "evaluation_callbacks": calls,
        "next_permitted_evaluation": None,
        "frozen_files_verified": len(config["frozen_files"]),
        "source_and_registered_parameters_unchanged": True,
        "saved_evidence_and_summary_exactly_reproduced": True,
        "original_execution_records_unchanged": True,
        "existing_terminal_replay_reason": resumed["terminal"]["reason"],
        "receipt_only_terminal_reconstruction_reason": reconstructed["terminal"]["reason"],
        "completed_pairs": summary["completed_pairs"],
        "efficacy_evidence": summary["final_efficacy"],
        "futility_evidence": summary["final_futility"],
        "first_crossing": summary["first_crossing"],
        "unresolved_request_events": unknown,
        "max_usd": config["max_usd"],
        "known_cost_usd": float(known),
        "unknown_request_reserve_usd": float(reserved),
        "conservative_committed_cost_usd": float(known + reserved),
        "remaining_authorized_usd_after_reserve": float(Decimal(str(config["max_usd"])) - known - reserved),
        "hypothetical_cost_of_213_needed_valid_episodes_usd": hypothetical_remaining,
        "cost_projection_note": "Includes an unpermitted repeat of the invalid challenger; not a planned continuation or savings claim.",
        "spending_authorization_is_blocker": False,
        "limitations": [
            "Billing reconciliation cannot supply an unexecuted browser trajectory or undo the provider error.",
            "Retrying, recoding, or skipping the invalid pair would depart from the frozen protocol.",
            "No resumed inference session occurred; review time is not experimental execution time.",
        ],
        "execution_record_sha256": before,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("execution_record_sha256", "unresolved_request_events")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/prospective_live_stopping")
    parser.add_argument(
        "--results", type=Path,
        default=ROOT / "results/paper/live_stopping/history/2026-10-16_initial_halt",
        help="Original-result archive; the primary results now contain the amended continuation.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.log_dir, args.results, args.out)
