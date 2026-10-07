"""Offline first-run retry decisions from existing randomized rerun anchors.

No inference or API calls. Distinct-agent outcomes are counterfactual averages over
the complete main-run grid, not a randomized prospective second-agent assignment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
from free_checks import observable_completion, saved_rows
from task_transfer import EMPIRICAL, ROOT, load_pools

OUT = ROOT / "results/growing_bandits/retrospective/first_run_triage"
SEED = 20261007


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def anchor_rows() -> list[dict]:
    logs = saved_rows()
    pools = {p.name: p for p in load_pools() if p.name in ("G", "F")}
    out = []
    for name, pool in pools.items():
        arm_pos = {a: i for i, a in enumerate(pool.arms)}
        task_pos = {t: j for j, t in enumerate(pool.tasks)}
        for (group, arm, task, rep), second in sorted(logs.items()):
            if group != name or rep != 1:
                continue
            first = logs[name, arm, task, 0]
            y1 = int(pool.y[arm_pos[arm], task_pos[task]])
            if y1 != int(first["success"]):
                raise ValueError("frozen first-run score and live log disagree")
            same_y = int(second["success"])
            others = [logs[name, b, task, 0] for b in pool.arms if b != arm]
            other_y = np.mean([int(r["success"]) for r in others])
            out.append(
                {
                    "pool": name,
                    "arm_id": arm,
                    "task_id": task,
                    "first_success": y1,
                    "first_cost_usd": float(first["cost_usd"]),
                    "first_complete": int(observable_completion(first)),
                    "first_steps": int(first["steps"]),
                    "same_success": same_y,
                    "same_cost_usd": float(second["cost_usd"]),
                    "different_success_expectation": float(other_y),
                    "different_cost_expectation_usd": float(
                        np.mean([float(r["cost_usd"]) for r in others])
                    ),
                }
            )
    return out


def policy_values(row: dict, policy: str) -> tuple[float, float, int]:
    """Return realized/expected success, paid cost, and whether a retry occurs."""
    if policy == "stop":
        return row["first_success"], row["first_cost_usd"], 0
    if policy.startswith("timeout_"):
        retry = not bool(row["first_complete"])
    elif policy.startswith("always_run_"):
        retry = True
    else:
        raise ValueError(policy)
    if not retry:
        return row["first_success"], row["first_cost_usd"], 0
    # An always-run policy still accepts the first trajectory if it appeared complete.
    # The second run then consumes budget without changing the chosen outcome.
    use_second = not bool(row["first_complete"])
    if policy.endswith("_same"):
        chosen = row["same_success"] if use_second else row["first_success"]
        return chosen, row["first_cost_usd"] + row["same_cost_usd"], 1
    if policy.endswith("_different"):
        chosen = row["different_success_expectation"] if use_second else row["first_success"]
        return (
            chosen,
            row["first_cost_usd"] + row["different_cost_expectation_usd"],
            1,
        )
    raise ValueError(policy)


def estimate(rows: list[dict], policy: str) -> dict[str, float]:
    values = np.asarray([policy_values(row, policy) for row in rows], dtype=float)
    return {
        "success_rate": float(values[:, 0].mean()),
        "cost_per_task_usd": float(values[:, 1].mean()),
        "retry_rate": float(values[:, 2].mean()),
        "cost_per_success_usd": float(values[:, 1].sum() / values[:, 0].sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = anchor_rows()
    write_csv("anchors.csv", rows)
    rng = np.random.default_rng(SEED)
    policies = (
        "stop",
        "timeout_same",
        "timeout_different",
        "always_run_same",
        "always_run_different",
    )
    results = []
    contrasts = []
    for pool in ("G", "F"):
        selected = [r for r in rows if r["pool"] == pool]
        tasks = sorted({r["task_id"] for r in selected})
        by_task = {task: [r for r in selected if r["task_id"] == task] for task in tasks}
        baseline = estimate(selected, "stop")
        boot = defaultdict(list)
        for _ in range(5000):
            sample = [
                r for task in rng.choice(tasks, len(tasks), replace=True) for r in by_task[task]
            ]
            stop = estimate(sample, "stop")
            for policy in policies:
                e = estimate(sample, policy)
                boot[policy, "success_gain"].append(e["success_rate"] - stop["success_rate"])
                boot[policy, "extra_cost"].append(
                    e["cost_per_task_usd"] - stop["cost_per_task_usd"]
                )
        for policy in policies:
            e = estimate(selected, policy)
            gain = e["success_rate"] - baseline["success_rate"]
            extra = e["cost_per_task_usd"] - baseline["cost_per_task_usd"]
            results.append(
                {
                    "pool": pool,
                    "policy": policy,
                    "n_anchors": len(selected),
                    "n_tasks": len(tasks),
                    **e,
                    "success_gain": gain,
                    "success_gain_boot_lo": float(np.quantile(boot[policy, "success_gain"], 0.025)),
                    "success_gain_boot_hi": float(np.quantile(boot[policy, "success_gain"], 0.975)),
                    "extra_cost_per_task_usd": extra,
                    "extra_cost_boot_lo": float(np.quantile(boot[policy, "extra_cost"], 0.025)),
                    "extra_cost_boot_hi": float(np.quantile(boot[policy, "extra_cost"], 0.975)),
                    "incremental_cost_per_extra_success_usd": extra / gain if gain > 0 else "",
                }
            )
        same = estimate(selected, "timeout_same")
        different = estimate(selected, "timeout_different")
        delta_success = []
        delta_cost = []
        for _ in range(5000):
            sample = [
                r for task in rng.choice(tasks, len(tasks), replace=True) for r in by_task[task]
            ]
            s = estimate(sample, "timeout_same")
            d = estimate(sample, "timeout_different")
            delta_success.append(d["success_rate"] - s["success_rate"])
            delta_cost.append(d["cost_per_task_usd"] - s["cost_per_task_usd"])
        contrasts.append(
            {
                "pool": pool,
                "contrast": "timeout_different_minus_timeout_same",
                "success_difference": different["success_rate"] - same["success_rate"],
                "success_boot_lo": float(np.quantile(delta_success, 0.025)),
                "success_boot_hi": float(np.quantile(delta_success, 0.975)),
                "cost_difference_per_task_usd": different["cost_per_task_usd"]
                - same["cost_per_task_usd"],
                "cost_boot_lo": float(np.quantile(delta_cost, 0.025)),
                "cost_boot_hi": float(np.quantile(delta_cost, 0.975)),
            }
        )
    write_csv("policy_summary.csv", results)
    write_csv("contrast_summary.csv", contrasts)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "empirical_snapshot": str(EMPIRICAL.relative_to(ROOT)),
                "empirical_snapshot_sha256": hashlib.sha256(EMPIRICAL.read_bytes()).hexdigest(),
                "worker_log_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((ROOT / "logs/empirical_pool").glob("worker_*.jsonl"))
                },
                "n_randomized_repeats": {p: sum(r["pool"] == p for r in rows) for p in ("G", "F")},
                "seed": SEED,
                "distinct_agent_estimate": "mean over all other prompt main outcomes on the same task",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
