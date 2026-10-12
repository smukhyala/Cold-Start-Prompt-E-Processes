"""Registered task-order analysis for the prospectively sampled Linear pair."""

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
from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths  # noqa: E402
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths  # noqa: E402

DATA = ROOT / "data/prospective_sampled_linear"
LOGS = ROOT / "logs/prospective_sampled_linear"
INC, CHAL = "linear_baseline", "linear_explorer"
ALPHA, DELTA = 0.025, 0.15


def first_crossing(path: np.ndarray) -> int | None:
    hits = np.flatnonzero(path >= math.log(1 / ALPHA))
    return None if not len(hits) else int(hits[0]) + 1


def summarize(log_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    collect.verify_queue(DATA, "sampled_linear")
    manifest = json.loads((DATA / "manifest.json").read_text())
    tasks = json.loads((DATA / "tasks.json").read_text())
    assert [t["id"] for t in tasks] == manifest["task_ids"]
    frame = collect.load_all(log_dir)
    attempts = frame.to_dict("records") if not frame.empty else []
    terminal = {}
    for row in attempts:
        if row["status"] in ("ok", "missing"):
            key = (str(row["task_id"]), str(row["arm_id"]))
            if key in terminal:
                raise ValueError(f"duplicate terminal record {key}")
            terminal[key] = row
    table = []
    for task in tasks:
        i, c = terminal.get((task["id"], INC)), terminal.get((task["id"], CHAL))
        table.append({"task_id": task["id"], "family": task["family"],
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
    result = {"protocol_commit": manifest["protocol_commit"], "seed_hex": manifest["seed_hex"],
              "planned_tasks": len(tasks), "attempts": len(attempts), "terminal_episodes": len(terminal),
              "paired_prefix_tasks": len(prefix),
              "missing_episodes": sum(r["status"] == "missing" for r in attempts),
              "cost_usd": collect.spent_usd(log_dir),
              "status_file": (log_dir / "STATUS").read_text().strip() if (log_dir / "STATUS").exists() else None}
    if prefix:
        yi = np.array([r["incumbent_success"] for r in prefix], dtype=int)
        yc = np.array([r["challenger_success"] for r in prefix], dtype=int)
        Y = np.stack([yi, yc])[None, :, :]
        eff = e1b_log_paths(Y, np.array([0]), np.array([[False, True]]))[0]
        fut = futility_log_paths((yc - yi)[None, None, :], DELTA)[0, 0]
        result.update({"incumbent_rate": float(yi.mean()), "challenger_rate": float(yc.mean()),
                       "paired_gain": float((yc - yi).mean()),
                       "challenger_only_wins": int(((yc == 1) & (yi == 0)).sum()),
                       "incumbent_only_wins": int(((yi == 1) & (yc == 0)).sum()),
                       "efficacy_first_crossing": first_crossing(eff),
                       "futility_first_crossing": first_crossing(fut),
                       "efficacy_max_e": float(np.exp(eff.max())),
                       "futility_max_e": float(np.exp(fut.max())),
                       "efficacy_final_e": float(np.exp(eff[-1])),
                       "futility_final_e": float(np.exp(fut[-1]))})
    result["complete_pairs_without_missing"] = bool(len(prefix) == len(tasks) and result["missing_episodes"] == 0)
    return result, table, attempts


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in records:
            clean = {}
            for key in fields:
                value = row.get(key)
                clean[key] = "" if value is None or (isinstance(value, float) and math.isnan(value)) else value
            writer.writerow(clean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=LOGS)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--attempts", type=Path)
    args = parser.parse_args()
    result, table, attempts = summarize(args.log_dir)
    if args.table:
        write_csv(args.table, table, list(table[0]))
    if args.attempts:
        fields = ["queue_index", "task_id", "arm_id", "attempt", "status", "success", "infra_reason",
                  "cost_usd", "steps", "wallclock_s", "timed_out", "prompt_sha256", "llm_model", "timestamp_utc"]
        write_csv(args.attempts, sorted(attempts, key=lambda r: (r["queue_index"], r["attempt"])), fields)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
