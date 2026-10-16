"""Audit the authorized continuation and account for every original/retry attempt."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.growing_bandits.empirical.live_stopping import (  # noqa: E402
    canonical_sha256,
    evidence_paths,
    verify_frozen_files,
)
from experiments.growing_bandits.empirical.live_stopping_runner import usage_cost  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest_path: Path, out: Path):
    manifest = read(manifest_path)
    source = Path(manifest["source_directory"])
    continued = Path(manifest["continuation_directory"])
    for group in ("source_files", "amendment_files"):
        for path, digest in manifest[group].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    registration = read(source / "registration.json")
    config = registration["frozen"]["config"]
    assert read(continued / "registration.json")["frozen"] == registration["frozen"]
    verify_frozen_files(config)
    original_pairs = read(source / "pairs.json")
    pairs = read(continued / "pairs.json")
    assert pairs[:13] == original_pairs and len(original_pairs) == 13
    evidence = evidence_paths(pairs, config)
    assert evidence == read(continued / "evidence.json")
    terminal = read(continued / "terminal.json")
    core = {p.stem: read(p) for p in (continued / "receipts").glob("*.json")}
    assert read(source / "receipts/pair_000014_baseline.json") == core["pair_000014_baseline"]
    for index, pair in enumerate(pairs, 1):
        assert pair["pair_index"] == index
        assert pair["task"] == registration["frozen"]["tasks"][index - 1]
        for role in ("baseline", "challenger"):
            receipt = core[pair[f"{role}_episode_id"]]
            result = receipt["result"]
            assert result["status"] == "ok" and result["success"] == pair[f"{role}_success"]
            assert result["model"] == config["model"]
            assert result["prompt_sha256"] == config["prompts"][role]["sha256"]
            assert receipt["task_sha256"] == canonical_sha256(pair["task"])
    first = next((p for p in evidence if p["efficacy_crossed"] or p["futility_crossed"]), None)
    assert terminal["first_crossing"] == first
    if first:
        assert first["pair_index"] == len(pairs)
        assert terminal["reason"] == "statistical_boundary"
        expected = "efficacy" if first["efficacy_crossed"] else "futility"
        assert terminal["decision"] == expected
    elif terminal["reason"] == "max_pairs":
        assert len(pairs) == config["max_pairs"] and terminal["decision"] == "unresolved"

    attempts, intents, ledger, session_rows = [], [], [], []
    known, reserve = Decimal(0), Decimal(0)
    for session, directory in (("original", source), ("continuation", continued)):
        receipts = [read(p) for p in sorted((directory / "episodes").glob("*/receipt.json"))]
        episode_intents = [read(p) for p in sorted((directory / "episodes").glob("*/intent.json"))]
        # The continuation runner directory contains only new physical attempts.
        by_episode = {r["episode_id"]: r for r in receipts}
        assert len(by_episode) == len(receipts)
        assert set(by_episode) <= {r["episode_id"] for r in episode_intents}
        for record in receipts + episode_intents:
            parts = record["episode_id"].split("_")
            index = int(parts[1])
            role = parts[2]
            task = registration["frozen"]["tasks"][index - 1]
            assert record["task_id"] == task["task_id"]
            assert record["source_task_id"] == task["source_task_id"]
            assert record["role"] == role and role in ("baseline", "challenger")
            assert record["model"] == config["model"]
            assert record["prompt_sha256"] == config["prompts"][role]["sha256"]
            assert record["episode_id"] == f"pair_{index:06d}_{role}" or (
                session == "continuation" and record["episode_id"] == "pair_000014_challenger_retry1")
        ordered = sorted(episode_intents, key=lambda r: r["timestamp_utc"])
        if session == "continuation":
            expected_ids = ["pair_000014_challenger_retry1"] + [
                f"pair_{i:06d}_{role}" for i in range(15, 121) for role in ("baseline", "challenger")]
            assert [r["episode_id"] for r in ordered] == expected_ids[:len(ordered)]
            assert all(r["timestamp_utc"] <= terminal["recorded_at"] for r in ordered)
        for previous, following in zip(ordered, ordered[1:], strict=False):
            assert previous["episode_id"] in by_episode
            assert by_episode[previous["episode_id"]]["finished_utc"] <= following["timestamp_utc"]
        meter_path = directory / "api_calls.jsonl"
        rows = [json.loads(line) for line in meter_path.read_text().splitlines()] if meter_path.exists() else []
        pending, completed, calls = {}, [], []
        for row in rows:
            assert row["episode_id"] in {r["episode_id"] for r in episode_intents}
            if session == "continuation":
                assert row["timestamp_utc"] <= terminal["recorded_at"]
            if row["event"] == "request_started":
                assert row["request_id"] not in pending
                pending[row["request_id"]] = row
                calls.append(row)
            elif row["event"] in ("request_completed", "request_rejected"):
                assert row["request_id"] in pending
                del pending[row["request_id"]]
                known += Decimal(str(row["cost_usd"]))
                if row["event"] == "request_completed":
                    cost, input_tokens, output_tokens, cached = usage_cost(row["raw_usage"])
                    assert cost == Decimal(str(row["cost_usd"]))
                    assert (input_tokens, output_tokens, cached) == (
                        row["input_tokens"], row["output_tokens"], row["cache_read_tokens"])
                    assert row["resolved_model"] == config["model"]
                    completed.append(row)
            else:
                assert row["event"] in ("request_unknown", "invocation_failed")
            ledger.append({"session": session, **row})
        reserve += sum((Decimal(str(r["reserved_usd"])) for r in pending.values()), Decimal(0))
        for receipt in receipts:
            matching = [r for r in completed if r["episode_id"] == receipt["episode_id"]]
            assert sum((Decimal(str(r["cost_usd"])) for r in matching), Decimal(0)) == Decimal(str(receipt["cost_usd"]))
            for name in ("input_tokens", "output_tokens", "cache_read_tokens"):
                assert sum(r[name] for r in matching) == receipt[name]
            attempts.append({"session": session, **receipt})
        intents.extend({"session": session, **r} for r in episode_intents)
        starts = [r["timestamp_utc"] for r in episode_intents]
        ends = [r["finished_utc"] for r in receipts]
        end = read(directory / "terminal.json")["recorded_at"]
        session_rows.append({"session": session, "first_episode_utc": min(starts) if starts else None,
                             "last_receipt_utc": max(ends) if ends else None,
                             "terminal_utc": end,
                             "execution_elapsed_s": (datetime.fromisoformat(end) - datetime.fromisoformat(min(starts))).total_seconds() if starts else 0,
                             "physical_episode_attempts": len(episode_intents),
                             "requests_started": len(calls), "responses_with_usage": len(completed)})
    assert known + reserve <= Decimal(str(config["max_usd"]))
    retry_intents = [r for r in intents if r["episode_id"] == "pair_000014_challenger_retry1"]
    retry_receipts = [r for r in attempts if r["episode_id"] == "pair_000014_challenger_retry1"]
    assert len(retry_receipts) <= len(retry_intents) <= 1
    missing_receipts = sorted((r["session"], r["episode_id"]) for r in intents
                              if not any(a["session"] == r["session"] and a["episode_id"] == r["episode_id"] for a in attempts))
    if missing_receipts:
        assert not first and terminal["reason"] in ("ambiguous_inflight_episode", "infrastructure_error")
    assert not any(r["session"] == "continuation" and r["episode_id"] == "pair_000014_baseline" for r in intents)
    original_ids = {r["episode_id"] for r in attempts if r["session"] == "original"}
    continuation_ids = {r["episode_id"] for r in attempts if r["session"] == "continuation"}
    assert not (original_ids & continuation_ids)
    for pair in pairs[13:]:
        for role in ("baseline", "challenger"):
            logical = pair[f"{role}_episode_id"]
            physical = "pair_000014_challenger_retry1" if logical == "pair_000014_challenger" else logical
            matches = [r for r in attempts if r["episode_id"] == physical]
            assert len(matches) == 1
            assert matches[0]["success"] == pair[f"{role}_success"]
            assert matches[0]["status"] == "ok"
    if first:
        assert len(attempts) == len(intents) == 2 * len(pairs) + 1
    baseline = sum(p["baseline_success"] for p in pairs)
    challenger = sum(p["challenger_success"] for p in pairs)
    completed_rows = [r for r in ledger if r["event"] == "request_completed"]
    valid = [r for r in attempts if r["status"] == "ok"]
    avoided = 2 * (config["max_pairs"] - len(pairs)) if first else 0
    mean = sum(r["cost_usd"] for r in valid) / len(valid)
    summary = {
        "study_status": "authorized_post_outcome_amended_continuation",
        "original_decision": "unresolved", "original_terminal_reason": "infrastructure_error",
        "decision": terminal["decision"], "terminal_reason": terminal["reason"],
        "completed_pairs": len(pairs), "max_pairs": config["max_pairs"],
        "first_crossing_pair": first["pair_index"] if first else None, "first_crossing": first,
        "baseline_successes": baseline, "challenger_successes": challenger,
        "baseline_rate": baseline / len(pairs), "challenger_rate": challenger / len(pairs),
        "paired_difference": (challenger - baseline) / len(pairs),
        "final_efficacy": math.exp(evidence[-1]["log_efficacy"]),
        "final_futility": math.exp(evidence[-1]["log_futility"]),
        "threshold_efficacy": 1 / config["alpha_e"], "threshold_futility": 1 / config["alpha_f"],
        "actual_episode_attempts": len(intents), "valid_outcome_episodes": len(valid),
        "authorized_episode_retries": 1, "sdk_retries": 0,
        "retry_attempts_started": len(retry_intents), "missing_episode_receipts": missing_receipts,
        "api_requests_started": sum(r["event"] == "request_started" for r in ledger),
        "api_responses_with_usage": len(completed_rows),
        "known_api_cost_usd": float(known), "unknown_charge_reserve_usd": float(reserve),
        "total_cost_interval_usd": [float(known), float(known + reserve)],
        "cost_accounting_complete": reserve == 0 and not missing_receipts,
        "input_tokens": sum(r["input_tokens"] for r in completed_rows),
        "output_tokens": sum(r["output_tokens"] for r in completed_rows),
        "cache_read_tokens": sum(r["cache_read_tokens"] for r in completed_rows),
        "sum_session_wallclock_s": sum(r["execution_elapsed_s"] for r in session_rows),
        "calendar_elapsed_s": (datetime.fromisoformat(terminal["recorded_at"]) - datetime.fromisoformat(session_rows[0]["first_episode_utc"])).total_seconds(),
        "original_planned_episodes": 240, "amended_full_budget_attempts": 241,
        "episodes_avoided_by_statistical_stop": avoided,
        "mean_known_cost_per_valid_episode_usd": mean,
        "projected_cost_of_unexecuted_episodes_usd": avoided * mean if first else None,
        "projection_note": "Hypothetical additional workload cost, not realized dollar savings.",
        "nominal_validity": "Conditional on the registered null assumptions also holding under the post-outcome recovery mechanism; no unconditional confirmatory guarantee is claimed.",
        "checks": {"exact_online_offline_parity": True, "source_and_amendment_hashes_match": True,
                   "prior_13_pairs_retained": True, "saved_baseline_14_reused": True,
                   "all_attempt_costs_and_tokens_reconcile": True, "no_requests_after_terminal": True},
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("pairs.json", pairs), ("evidence.json", evidence),
                        ("episode_receipts.json", attempts), ("episode_intents.json", intents),
                        ("execution_timeline.json", session_rows), ("terminal.json", terminal),
                        ("amendment.json", manifest), ("registration.json", registration)):
        dump(out / name, value)
    with (out / "api_calls.jsonl").open("w") as stream:
        for row in ledger:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    table(out / "pairs.csv", pairs, ["pair_index", "baseline_success", "challenger_success", "baseline_episode_id", "challenger_episode_id", "completed_at", "cost_usd"])
    table(out / "evidence.csv", evidence, ["pair_index", "log_efficacy", "log_futility", "efficacy_crossed", "futility_crossed"])
    table(out / "episodes.csv", attempts, ["session", "episode_id", "task_id", "source_task_id", "role", "status", "success", "cost_usd", "input_tokens", "output_tokens", "started_utc", "finished_utc"])
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.manifest, args.out)
