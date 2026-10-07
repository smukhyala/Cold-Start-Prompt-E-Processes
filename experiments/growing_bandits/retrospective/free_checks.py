"""No-inference checks of saved trials and public WorkBench item outcomes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict

import numpy as np
from task_transfer import EMPIRICAL, ROOT, load_pools

OUT = ROOT / "results/growing_bandits/retrospective/free_checks"
WORKBENCH = OUT / "workbench_item_level_results.csv.gz"
SEED = 20261006
WORKBENCH_SHA256 = "6035fb112bbb4fb48cd56d643898cac43c42c2d11bc6d2bb0e595d8d31ea2216"


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def observable_completion(row: dict) -> bool:
    """A fixed rule using no benchmark verdict or verifier message."""
    return bool(row["is_done"] and not row["timed_out"] and row["final_result"].strip())


def saved_rows() -> dict[tuple[str, str, str, int], dict]:
    found = {}
    for path in sorted((ROOT / "logs/empirical_pool").glob("worker_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row["pool"] not in ("G", "F") or row["status"] != "ok":
                continue
            key = row["pool"], row["arm_id"], row["task_id"], int(row["replicate"])
            if key in found:
                raise ValueError(f"duplicate saved run {key}")
            found[key] = row
    return found


def paired_selector() -> None:
    logs = saved_rows()
    pools = {p.name: p for p in load_pools() if p.name in ("G", "F")}
    details = []
    rng = np.random.default_rng(SEED)
    summaries = []
    for name, pool in pools.items():
        task_pos = {t: j for j, t in enumerate(pool.tasks)}
        arm_pos = {a: i for i, a in enumerate(pool.arms)}
        if len([k for k in logs if k[0] == name and k[3] == 0]) != pool.y.size:
            raise ValueError(f"incomplete {name} main logs")
        for key, second in sorted(logs.items()):
            group, arm, task, rep = key
            if group != name or rep != 1:
                continue
            first = logs[name, arm, task, 0]
            y1 = int(pool.y[arm_pos[arm], task_pos[task]])
            y2 = int(second["success"])
            choose_second = not observable_completion(first) and observable_completion(second)
            others = [(b, logs[name, b, task, 0]) for b in pool.arms if b != arm]
            distinct_sel = np.mean(
                [
                    int(pool.y[arm_pos[b], task_pos[task]])
                    if not observable_completion(first) and observable_completion(other)
                    else y1
                    for b, other in others
                ]
            )
            distinct_oracle = np.mean(
                [max(y1, int(pool.y[arm_pos[b], task_pos[task]])) for b, _ in others]
            )
            details.append(
                {
                    "pool": name,
                    "arm_id": arm,
                    "task_id": task,
                    "first": y1,
                    "same_second": y2,
                    "first_complete": int(observable_completion(first)),
                    "same_second_complete": int(observable_completion(second)),
                    "same_selected": y2 if choose_second else y1,
                    "same_oracle": max(y1, y2),
                    "distinct_selected_expectation": distinct_sel,
                    "distinct_oracle_expectation": distinct_oracle,
                }
            )
        rows = [d for d in details if d["pool"] == name]
        tasks = sorted({d["task_id"] for d in rows})
        by_task = {t: [d for d in rows if d["task_id"] == t] for t in tasks}
        metrics = [
            "first",
            "same_selected",
            "same_oracle",
            "distinct_selected_expectation",
            "distinct_oracle_expectation",
        ]
        estimates = {m: np.mean([d[m] for d in rows]) for m in metrics}
        boot = defaultdict(list)
        for _ in range(5000):
            sampled = [d for t in rng.choice(tasks, len(tasks), replace=True) for d in by_task[t]]
            for m in metrics:
                boot[m].append(float(np.mean([d[m] for d in sampled])))
            boot["same_gain"].append(
                float(np.mean([d["same_selected"] - d["first"] for d in sampled]))
            )
            boot["distinct_gain"].append(
                float(np.mean([d["distinct_selected_expectation"] - d["first"] for d in sampled]))
            )
        estimates["same_gain"] = estimates["same_selected"] - estimates["first"]
        estimates["distinct_gain"] = estimates["distinct_selected_expectation"] - estimates["first"]
        for m, estimate in estimates.items():
            summaries.append(
                {
                    "pool": name,
                    "metric": m,
                    "estimate": estimate,
                    "task_boot_lo": np.quantile(boot[m], 0.025),
                    "task_boot_hi": np.quantile(boot[m], 0.975),
                    "n_repeat_cells": len(rows),
                    "n_tasks": len(tasks),
                }
            )
    write_csv("saved_selector_cells.csv", details)
    write_csv("saved_selector_summary.csv", summaries)


def workbench_matrix() -> tuple[np.ndarray, list[str], list[str], list[str]]:
    if hashlib.sha256(WORKBENCH.read_bytes()).hexdigest() != WORKBENCH_SHA256:
        raise ValueError("WorkBench source differs from the audited snapshot")
    models = set()
    tasks = set()
    values = {}
    template_by_task = {}
    with gzip.open(WORKBENCH, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["run_group"] != "revisited_2026":
                continue
            if row["ground_truth_version"] != "v2" or row["tool_selection"] != "all":
                raise ValueError("WorkBench cohort changed")
            key = row["model"], row["task_id"]
            if key in values or row["correct"] not in ("True", "False"):
                raise ValueError(f"invalid WorkBench outcome {key}")
            values[key] = int(row["correct"] == "True")
            models.add(key[0])
            tasks.add(key[1])
            if (
                row["task_id"] in template_by_task
                and template_by_task[row["task_id"]] != row["base_template"]
            ):
                raise ValueError("task template mismatch")
            template_by_task[row["task_id"]] = row["base_template"]
    models, tasks = sorted(models), sorted(tasks)
    if len(models) != 24 or len(tasks) != 690 or len(values) != len(models) * len(tasks):
        raise ValueError("unexpected WorkBench dimensions")
    y = np.asarray([[values[m, t] for t in tasks] for m in models], dtype=np.int8)
    return y, models, tasks, [template_by_task[t] for t in tasks]


def workbench_analysis() -> None:
    y, models, tasks, templates = workbench_matrix()
    rng = np.random.default_rng(SEED)
    groups = sorted(set(templates))
    template_array = np.asarray(templates)
    records = []
    for split in range(1000):
        shuffled = rng.permutation(groups)
        train = np.flatnonzero(np.isin(template_array, shuffled[: len(groups) // 2]))
        test = np.flatnonzero(np.isin(template_array, shuffled[len(groups) // 2 :]))
        rates = y[:, train].mean(axis=1)
        ranked = sorted(range(len(models)), key=lambda i: (-rates[i], models[i]))
        top1, top2 = ranked[:2]
        greedy2 = min(
            (i for i in range(len(models)) if i != top1),
            key=lambda i: (-np.maximum(y[top1, train], y[i, train]).mean(), -rates[i], models[i]),
        )
        selected = {"top_single": (top1,), "top_two": (top1, top2), "greedy_pair": (top1, greedy2)}
        n = len(models)
        s = y[:, test].sum(axis=0)
        random_pair = np.mean(1 - (n - s) * (n - s - 1) / (n * (n - 1)))
        base = {
            "split": split,
            "n_train_tasks": len(train),
            "n_test_tasks": len(test),
            "random_single": float(y[:, test].mean()),
            "random_pair": float(random_pair),
            "all_model_oracle": float(y[:, test].max(axis=0).mean()),
        }
        for method, pair in selected.items():
            records.append(
                {
                    **base,
                    "method": method,
                    "test_coverage": float(y[np.ix_(pair, test)].max(axis=0).mean()),
                    "models": "|".join(models[i] for i in pair),
                }
            )
    write_csv("workbench_splits.csv", records)
    summary = []
    for method in ("top_single", "top_two", "greedy_pair"):
        rows = [r for r in records if r["method"] == method]
        baseline = "random_single" if method == "top_single" else "random_pair"
        diff = np.asarray([r["test_coverage"] - r[baseline] for r in rows])
        single_by_split = {
            r["split"]: r["test_coverage"] for r in records if r["method"] == "top_single"
        }
        summary.append(
            {
                "method": method,
                "mean_coverage": np.mean([r["test_coverage"] for r in rows]),
                "mean_minus_random": diff.mean(),
                "split_p05_minus_random": np.quantile(diff, 0.05),
                "split_p95_minus_random": np.quantile(diff, 0.95),
                "mean_minus_top_single": np.mean(
                    [r["test_coverage"] - single_by_split[r["split"]] for r in rows]
                ),
                "mean_all_model_oracle": np.mean([r["all_model_oracle"] for r in rows]),
                "fraction_above_random": np.mean(diff > 0),
                "n_template_splits": len(rows),
            }
        )
    write_csv("workbench_summary.csv", summary)
    write_csv(
        "workbench_model_rates.csv",
        [{"model": model, "success_rate": float(y[i].mean())} for i, model in enumerate(models)],
    )
    manifest = {
        "source": "https://github.com/LucasEtteldorf/workbench/blob/main/data/results/item_level_results.csv.gz",
        "sha256": WORKBENCH_SHA256,
        "cohort": "revisited_2026, v2, all tools",
        "n_models": len(models),
        "n_tasks": len(tasks),
        "n_base_templates": len(groups),
        "empirical_snapshot_sha256": hashlib.sha256(EMPIRICAL.read_bytes()).hexdigest(),
        "seed": SEED,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    paired_selector()
    workbench_analysis()
