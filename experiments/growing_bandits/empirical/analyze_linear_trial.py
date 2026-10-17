"""Summarize the frozen Linear pair in task order; reads collector logs only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect  # noqa: E402
import make_pools  # noqa: E402
from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths  # noqa: E402
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths  # noqa: E402

DATA = ROOT / "data/prospective_linear"
LOGS = ROOT / "logs/prospective_linear"
INC, CHAL = "linear_baseline", "linear_explorer"
ALPHA = 0.025
DELTA = 0.15


def summarize(log_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((DATA / "manifest.json").read_text())
    collect.verify_queue(DATA, "linear")
    queue = make_pools.read_queue(DATA / "linear/queue.jsonl")
    attempts = collect.load_all(log_dir)
    terminal = {}
    all_rows = attempts.to_dict("records") if not attempts.empty else []
    for row in all_rows:
        if row.get("status") in ("ok", "missing"):
            key = (str(row["task_id"]), str(row["arm_id"]))
            if key in terminal:
                raise ValueError(f"duplicate terminal record: {key}")
            terminal[key] = row
    table = []
    for task in manifest["task_ids"]:
        i, c = terminal.get((task, INC)), terminal.get((task, CHAL))
        table.append({"task_id": task,
                      "incumbent_status": None if i is None else i["status"],
                      "challenger_status": None if c is None else c["status"],
                      "incumbent_success": None if i is None or i["status"] != "ok" else int(i["success"]),
                      "challenger_success": None if c is None or c["status"] != "ok" else int(c["success"]),
                      "incumbent_timestamp_utc": None if i is None else i.get("timestamp_utc"),
                      "challenger_timestamp_utc": None if c is None else c.get("timestamp_utc")})
    prefix = []
    for row in table:
        if row["incumbent_success"] is None or row["challenger_success"] is None:
            break
        prefix.append(row)
    result = {"protocol_commit": "30fcc9a858d83d28a19b1d0acae80d181a403877",
              "planned_tasks": len(queue) // 2, "attempts": len(all_rows),
              "terminal_episodes": len(terminal), "paired_prefix_tasks": len(prefix),
              "missing_episodes": sum(r.get("status") == "missing" for r in all_rows),
              "cost_usd": collect.spent_usd(log_dir),
              "status_file": (log_dir / "STATUS").read_text().strip() if (log_dir / "STATUS").exists() else None}
    if prefix:
        yi = np.array([r["incumbent_success"] for r in prefix], dtype=int)
        yc = np.array([r["challenger_success"] for r in prefix], dtype=int)
        Y = np.stack([yi, yc])[None, :, :]
        eff = e1b_log_paths(Y, np.array([0]), np.array([[False, True]]))[0]
        fut = futility_log_paths((yc - yi)[None, None, :], DELTA)[0, 0]
        boundary = math.log(1 / ALPHA)
        def crossing(path):
            hits = np.flatnonzero(path >= boundary)
            return None if len(hits) == 0 else int(hits[0]) + 1
        result.update({"incumbent_rate": float(yi.mean()), "challenger_rate": float(yc.mean()),
                       "paired_gain": float((yc - yi).mean()),
                       "challenger_only_wins": int(((yc == 1) & (yi == 0)).sum()),
                       "incumbent_only_wins": int(((yi == 1) & (yc == 0)).sum()),
                       "efficacy_first_crossing": crossing(eff), "futility_first_crossing": crossing(fut),
                       "efficacy_max_e": float(np.exp(eff.max())),
                       "futility_max_e": float(np.exp(fut.max())),
                       "efficacy_final_e": float(np.exp(eff[-1])),
                       "futility_final_e": float(np.exp(fut[-1]))})
    # Completeness is an observable data check; conditional task-sampling validity is a separate assumption.
    result["complete_pairs_without_missing"] = bool(len(prefix) == 120 and result["missing_episodes"] == 0)
    return result, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=LOGS)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--attempts", type=Path)
    args = parser.parse_args()
    result, table = summarize(args.log_dir)
    if args.table:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        with args.table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    if args.attempts:
        fields = ("queue_index", "task_id", "arm_id", "attempt", "status", "success",
                  "infra_reason", "cost_usd", "steps", "wallclock_s", "timed_out",
                  "prompt_sha256", "llm_model", "timestamp_utc")
        attempts = collect.load_all(args.log_dir)
        records = attempts.sort_values(["queue_index", "attempt"]).to_dict("records")
        args.attempts.parent.mkdir(parents=True, exist_ok=True)
        with args.attempts.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in records:
                clean = {k: row.get(k) for k in fields}
                for key, value in clean.items():
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        clean[key] = ""
                if clean["success"] != "":
                    clean["success"] = int(clean["success"])
                writer.writerow(clean)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
