"""Audit saved live outcomes and export a real-data evidence figure and tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.growing_bandits.empirical.live_stopping import (  # noqa: E402
    canonical_sha256,
    evidence_paths,
    verify_frozen_files,
)


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in rows)


def analyze(directory: Path, out: Path) -> dict:
    registration = read(directory / "registration.json")
    frozen = registration["frozen"]
    cfg = frozen["config"]
    assert canonical_sha256(frozen) == registration["frozen_sha256"]
    verify_frozen_files(cfg)
    pairs, saved = read(directory / "pairs.json"), read(directory / "evidence.json")
    state = read(directory / "state.json")
    receipts = [read(p) for p in sorted((directory / "receipts").glob("*.json"))]
    intents = [read(p) for p in sorted((directory / "intents").glob("*.json"))]
    by_episode = {r["episode_id"]: r for r in receipts}
    assert len(by_episode) == len(receipts)
    intent_ids = {r["episode_id"] for r in intents}
    for record in intents + receipts:
        assert record["episode_id"] in intent_ids
        assert record["frozen_sha256"] == registration["frozen_sha256"]
        assert record["task_sha256"] == canonical_sha256(frozen["tasks"][record["pair_index"] - 1])
        assert record["model"] == cfg["model"]
        assert record["prompt_sha256"] == cfg["prompts"][record["role"]]["sha256"]
    recomputed = evidence_paths(pairs, cfg)
    assert recomputed == saved, "online/offline evidence mismatch"
    assert [p["pair_index"] for p in pairs] == list(range(1, len(pairs) + 1))
    for pair in pairs:
        assert pair["task"] == frozen["tasks"][pair["pair_index"] - 1]
        halves = [by_episode[pair[f"{role}_episode_id"]] for role in ("baseline", "challenger")]
        for role, half in zip(("baseline", "challenger"), halves, strict=True):
            assert half["pair_index"] == pair["pair_index"] and half["role"] == role
            assert half["result"]["status"] == "ok"
            assert half["result"]["success"] == pair[f"{role}_success"]
        assert math.isclose(pair["cost_usd"], math.fsum(h["result"]["cost_usd"] for h in halves), abs_tol=1e-12)
    first = next((p for p in saved if p["efficacy_crossed"] or p["futility_crossed"]), None)
    terminal = state.get("terminal")
    if first:
        assert len(pairs) == first["pair_index"], "evaluations continued beyond first boundary"
        assert len(intents) == len(receipts) == 2 * len(pairs)
        assert terminal["first_crossing"] == first
    if terminal:
        assert all(x["started_at"] <= terminal["recorded_at"] for x in intents)
    known_cost = math.fsum(r["result"]["cost_usd"] for r in receipts
                          if isinstance(r["result"].get("cost_usd"), (int, float)))
    assert math.isclose(known_cost, state["known_cost_usd"], abs_tol=1e-12)
    ledger_path = directory / "api_calls.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    calls = [r for r in ledger if r["event"] == "request_started"]
    completed = [r for r in ledger if r["event"] == "request_completed"]
    pending = {r["request_id"]: r for r in calls}
    for row in ledger:
        assert row["episode_id"] in intent_ids
        if terminal:
            assert row["timestamp_utc"] <= terminal["recorded_at"]
        if row["event"] in ("request_completed", "request_rejected"):
            assert row["request_id"] in pending
            del pending[row["request_id"]]
    metered_cost = math.fsum(r["cost_usd"] for r in completed) if ledger_path.exists() else known_cost
    for receipt in receipts:
        if not ledger_path.exists():
            break
        subtotal = math.fsum(r["cost_usd"] for r in completed if r["episode_id"] == receipt["episode_id"])
        assert math.isclose(subtotal, receipt["result"].get("cost_usd") or 0, abs_tol=1e-12)
        for target, source in (("input_tokens", "input_tokens"), ("output_tokens", "output_tokens")):
            total = sum(r[source] for r in completed if r["episode_id"] == receipt["episode_id"])
            assert total == (receipt["result"].get(target) or 0)
        durable = directory / "episodes" / receipt["episode_id"] / "receipt.json"
        if durable.exists():
            assert read(durable) == receipt["result"]
    unmatched = sorted({r["episode_id"] for r in calls} - by_episode.keys())
    yi = np.array([p["baseline_success"] for p in pairs], dtype=int)
    yc = np.array([p["challenger_success"] for p in pairs], dtype=int)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("registration.json", "pairs.json", "evidence.json", "state.json", "terminal.json", "api_calls.jsonl"):
        source = directory / name
        if source.exists():
            (out / name).write_bytes(source.read_bytes())
    (out / "episode_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (out / "episode_intents.json").write_text(json.dumps(intents, indent=2) + "\n")
    rows = [{"pair_index": p["pair_index"], **p["task"],
             "baseline_success": p["baseline_success"],
             "challenger_success": p["challenger_success"],
             "cost_usd": p["cost_usd"], "completed_at": p["completed_at"]} for p in pairs]
    write_csv(out / "pairs.csv", rows, ["pair_index", "task_id", "source_task_id", "seed_hex",
              "difficulty", "instruction", "verify", "baseline_success", "challenger_success",
              "cost_usd", "completed_at"])
    write_csv(out / "evidence.csv", saved, ["pair_index", "log_efficacy", "log_futility",
                                          "efficacy_crossed", "futility_crossed"])
    episode_rows = [{"episode_id": r["episode_id"], "pair_index": r["pair_index"], "role": r["role"],
                     "started_at": r["started_at"], "completed_at": r["completed_at"], **r["result"]}
                    for r in receipts]
    write_csv(out / "episodes.csv", episode_rows, ["episode_id", "pair_index", "role", "status",
              "success", "cost_usd", "input_tokens", "output_tokens", "model", "prompt_sha256",
              "started_at", "completed_at"])
    start = min((x["started_at"] for x in intents), default=None)
    end = terminal["recorded_at"] if terminal else state["updated_at"]
    summary = {"decision": state["decision"], "terminal_reason": terminal["reason"] if terminal else None,
               "completed_pairs": len(pairs), "max_pairs": cfg["max_pairs"],
               "first_crossing_pair": first["pair_index"] if first else None,
               "first_crossing": first, "threshold_efficacy": 1 / cfg["alpha_e"],
               "threshold_futility": 1 / cfg["alpha_f"],
               "baseline_successes": int(yi.sum()), "challenger_successes": int(yc.sum()),
               "baseline_rate": float(yi.mean()) if len(pairs) else None,
               "challenger_rate": float(yc.mean()) if len(pairs) else None,
               "paired_difference": float((yc - yi).mean()) if len(pairs) else None,
               "challenger_only_wins": int(((yc == 1) & (yi == 0)).sum()),
               "baseline_only_wins": int(((yi == 1) & (yc == 0)).sum()),
               "ties": int((yi == yc).sum()),
               "final_efficacy": math.exp(saved[-1]["log_efficacy"]) if saved else None,
               "final_futility": math.exp(saved[-1]["log_futility"]) if saved else None,
               "episode_intents": len(intents), "episode_receipts": len(receipts),
               "valid_outcome_episodes": state["valid_outcome_episodes"],
               "measured_api_cost_usd": metered_cost,
               "cost_accounting_complete": state["cost_accounting_complete"] and not pending and not unmatched,
               "input_tokens": sum(r["input_tokens"] for r in completed) if ledger_path.exists() else state["input_tokens"],
               "output_tokens": sum(r["output_tokens"] for r in completed) if ledger_path.exists() else state["output_tokens"],
               "cache_read_tokens": sum(r["cache_read_tokens"] for r in completed),
               "api_requests_started": len(calls), "api_requests_completed": len(completed),
               "episodes_with_api_requests": len({r["episode_id"] for r in calls}),
               "unknown_billing_reserve_usd": math.fsum(r["reserved_usd"] for r in pending.values()),
               "unmatched_metered_episodes": unmatched,
               "started_at": start, "ended_at": end,
               "wallclock_s": (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start else 0,
               "full_budget_episodes": 2 * cfg["max_pairs"],
               "episodes_avoided_by_statistical_stop": 2 * (cfg["max_pairs"] - len(pairs)) if first else 0,
               "unscheduled_planned_episodes": 2 * cfg["max_pairs"] - len(intents),
               "projected_unexecuted_cost_usd": (known_cost / len(receipts) *
                   2 * (cfg["max_pairs"] - len(pairs))) if first and receipts and not pending and not unmatched else None,
               "projection_note": "Hypothetical unexecuted workload; not realized dollar savings.",
               "protocol_commit": cfg.get("protocol_commit"), "seed_hex": cfg.get("seed_hex"),
               "frozen_sha256": registration["frozen_sha256"],
               "checks": {"frozen_files_match": True, "evidence_exactly_reproduced": True,
                          "no_scheduling_after_terminal": True, "episode_cost_reconciles": True,
                          "http_meter_reconciles": ledger_path.exists() and not unmatched,
                          "complete_provider_usage": not pending},
               "validity": "Conditional on the registered sampling, reset, model and environment assumptions."}
    if saved:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
            x = [0] + [p["pair_index"] for p in saved]
            ax.plot(x, [0] + [p["log_efficacy"] for p in saved], color="#0072B2", label="Efficacy", lw=1.8)
            ax.plot(x, [0] + [p["log_futility"] for p in saved], color="#D55E00", ls="--", label="Practical futility", lw=1.8)
            ax.axhline(math.log(1 / cfg["alpha_e"]), color="0.35", ls=":", label="Threshold: log(40)")
            if first:
                ax.axvline(first["pair_index"], color="0.45", lw=1, ls="-.",
                           label=f"Actual stop: {first['pair_index']} pairs")
            ax.set(xlim=(0, cfg["max_pairs"]), xlabel="Completed task pairs",
                   ylabel="Natural log evidence")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="best", fontsize=8.5)
            for suffix in ("pdf", "svg", "png"):
                fig.savefig(out / f"live_stopping_evidence.{suffix}", dpi=300)
            plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
                 if p.is_file() and p.name != "sha256.json"}
    (out / "sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/prospective_live_stopping")
    parser.add_argument("--out", type=Path, default=ROOT / "results/paper/live_stopping")
    args = parser.parse_args()
    print(json.dumps(analyze(args.log_dir, args.out), indent=2))
