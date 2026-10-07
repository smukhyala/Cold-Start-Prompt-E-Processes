"""Analyze the frozen new-task Gmail pilot after all 90 episodes complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from first_run_trace_probe import features
from free_checks import observable_completion
from task_transfer import ROOT

DATA = ROOT / "data/first_run_pilot_gmail"
LOGS = ROOT / "logs/first_run_pilot_gmail"
OUT = ROOT / "results/growing_bandits/retrospective/first_run_pilot_gmail"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(log_dir: Path) -> tuple[dict, dict, dict]:
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(DATA / name) != expected:
            raise ValueError(f"prepared pilot input changed: {name}")
    rule = json.loads((DATA / "uplift_rule.json").read_text())
    queue = [json.loads(line) for line in (DATA / "queue.jsonl").open()]
    expected = {(r["arm_id"], r["task_id"], int(r["replicate"])) for r in queue}
    if len(expected) != 90:
        raise ValueError("pilot queue has duplicate cells")
    terminals = {}
    for path in sorted(log_dir.glob("worker_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            key = row["arm_id"], row["task_id"], int(row["replicate"])
            if key not in expected:
                raise ValueError(f"record outside frozen queue: {key}")
            if row["status"] in ("ok", "missing"):
                if key in terminals:
                    raise ValueError(f"duplicate terminal record: {key}")
                terminals[key] = row
    missing = expected - set(terminals)
    if missing:
        raise ValueError(f"pilot incomplete: {len(missing)} of 90 cells lack terminal records")
    if any(row["status"] != "ok" for row in terminals.values()):
        raise ValueError("pilot has terminal infrastructure-missing cells; report separately")
    return manifest, rule, terminals


def predicted_benefit(first: dict, rule: dict) -> float:
    row = {"pool": "G", "first_steps": first["steps"], "first_cost_usd": first["cost_usd"]}
    x = np.asarray(features(row, first), dtype=float)
    mean = np.asarray(rule["standardizer_mean"], dtype=float)
    scale = np.asarray(rule["standardizer_scale"], dtype=float)
    coef = np.asarray(rule["ridge_coefficients"], dtype=float)
    return float(((x - mean) / scale) @ coef + rule["ridge_intercept"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=LOGS)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    manifest, rule, terminals = load_inputs(args.log_dir)
    first_arm, other_arm = manifest["first_arm"], manifest["different_arm"]
    rows = []
    for task in manifest["task_ids"]:
        first = terminals[first_arm, task, 0]
        same = terminals[first_arm, task, 1]
        different = terminals[other_arm, task, 0]
        complete = observable_completion(first)
        score = predicted_benefit(first, rule)
        use_uplift = not complete or score >= rule["threshold"]
        y0, ys, yd = (int(row["success"]) for row in (first, same, different))
        c0, cs, cd = (float(row["cost_usd"]) for row in (first, same, different))
        rows.append(
            {
                "task_id": task,
                "first_complete": int(complete),
                "predicted_same_retry_benefit": score,
                "uplift_rule_retries": int(use_uplift),
                "first_success": y0,
                "same_success": ys,
                "different_success": yd,
                "first_cost_usd": c0,
                "same_cost_usd": cs,
                "different_cost_usd": cd,
                "stop_success": y0,
                "stop_cost_usd": c0,
                "timeout_same_success": y0 if complete else ys,
                "timeout_same_cost_usd": c0 + (0 if complete else cs),
                "timeout_different_success": y0 if complete else yd,
                "timeout_different_cost_usd": c0 + (0 if complete else cd),
                "uplift_same_success": ys if use_uplift else y0,
                "uplift_same_cost_usd": c0 + (cs if use_uplift else 0),
                "same_oracle_coverage": max(y0, ys),
                "different_oracle_coverage": max(y0, yd),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "task_outcomes.csv", rows)
    rng = np.random.default_rng(20261007)
    boots = defaultdict(list)
    methods = ("stop", "timeout_same", "timeout_different", "uplift_same")
    for _ in range(10000):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        for method in methods:
            boots[method, "gain"].append(
                np.mean([r[f"{method}_success"] - r["stop_success"] for r in sample])
            )
            boots[method, "gain_vs_timeout_same"].append(
                np.mean([r[f"{method}_success"] - r["timeout_same_success"] for r in sample])
            )
    summary = []
    for method in methods:
        summary.append(
            {
                "method": method,
                "n_tasks": len(rows),
                "success_rate": float(np.mean([r[f"{method}_success"] for r in rows])),
                "success_gain_vs_stop": float(
                    np.mean([r[f"{method}_success"] - r["stop_success"] for r in rows])
                ),
                "gain_task_boot_lo": float(np.quantile(boots[method, "gain"], 0.025)),
                "gain_task_boot_hi": float(np.quantile(boots[method, "gain"], 0.975)),
                "gain_vs_timeout_same": float(
                    np.mean([r[f"{method}_success"] - r["timeout_same_success"] for r in rows])
                ),
                "gain_vs_timeout_same_boot_lo": float(
                    np.quantile(boots[method, "gain_vs_timeout_same"], 0.025)
                ),
                "gain_vs_timeout_same_boot_hi": float(
                    np.quantile(boots[method, "gain_vs_timeout_same"], 0.975)
                ),
                "mean_cost_usd": float(np.mean([r[f"{method}_cost_usd"] for r in rows])),
                "cost_per_success_usd": (
                    float(
                        sum(r[f"{method}_cost_usd"] for r in rows)
                        / sum(r[f"{method}_success"] for r in rows)
                    )
                    if sum(r[f"{method}_success"] for r in rows)
                    else ""
                ),
                "same_oracle_coverage": float(np.mean([r["same_oracle_coverage"] for r in rows])),
                "different_oracle_coverage": float(
                    np.mean([r["different_oracle_coverage"] for r in rows])
                ),
            }
        )
    write_csv(args.output / "policy_summary.csv", summary)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "pilot_manifest_sha256": sha(DATA / "manifest.json"),
                "worker_log_sha256": {
                    str(p): sha(p) for p in sorted(args.log_dir.glob("worker_*.jsonl"))
                },
                "n_tasks": len(rows),
                "bootstrap_seed": 20261007,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
