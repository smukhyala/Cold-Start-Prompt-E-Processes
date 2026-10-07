"""Freeze a 30-task Gmail retry pilot without launching any paid episodes."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from first_run_trace_probe import features
from first_run_triage import anchor_rows
from free_checks import saved_rows
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EMPIRICAL = ROOT / "data/empirical_pool"
OUT = ROOT / "data/first_run_pilot_gmail"
sys.path.insert(0, str(ROOT / "experiments/growing_bandits/empirical"))

from make_pools import QueueItem, load_pool  # noqa: E402

FIRST_ARM = "G_14"
SECOND_ARM = "G_16"
SEED = 20261007


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads((EMPIRICAL / "manifest.json").read_text())
    tasks = sorted(set(source["task_ids"]) - set(source["task_subset"]))
    if len(tasks) != 30 or {t.split("_")[1][0] for t in tasks} != {"e", "m", "h"}:
        raise ValueError("unexpected held-out task bank")
    if any(sum(t.startswith(f"task_{tier}") for t in tasks) != 10 for tier in "emh"):
        raise ValueError("task strata are not 10/10/10")
    pool = {arm.arm.arm_id: arm for arm in load_pool(EMPIRICAL / "pool_G.yaml")}
    if FIRST_ARM not in pool or SECOND_ARM not in pool or FIRST_ARM == SECOND_ARM:
        raise ValueError("frozen G pair unavailable")
    # Prior pilot records on these task IDs do not contain either selected arm.
    for path in (ROOT / "logs/empirical_pool/archive").glob("*.jsonl"):
        for line in path.open():
            row = json.loads(line)
            if row.get("task_id") in tasks and row.get("arm_id") in (FIRST_ARM, SECOND_ARM):
                raise ValueError("selected pair has a prior run on a pilot task")
    rng = np.random.default_rng(SEED)
    items = []
    for task in rng.permutation(tasks):
        # With one worker, the first episode really precedes both alternatives.
        options = [(FIRST_ARM, 1), (SECOND_ARM, 0)]
        rng.shuffle(options)
        items.extend([(FIRST_ARM, str(task), 0), *[(a, str(task), r) for a, r in options]])
    queue = [
        QueueItem(index=i, pool="G", arm_id=a, task_id=t, replicate=r, pilot=False)
        for i, (a, t, r) in enumerate(items)
    ]
    if len(queue) != 90:
        raise ValueError("pilot queue is not 30 triples")
    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml"):
        source_path = EMPIRICAL / name
        if sha(source_path) != source["files"][name]:
            raise ValueError(f"original frozen pool changed: {name}")
        dest = OUT / name
        if dest.exists() and sha(dest) != sha(source_path):
            raise ValueError(f"prepared pool differs from source: {name}")
        shutil.copy2(source_path, dest)
        files[name] = sha(dest)
    queue_path = OUT / "queue.jsonl"
    queue_text = "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in queue)
    if queue_path.exists() and queue_path.read_text() != queue_text:
        raise ValueError("existing trial queue differs; refusing to overwrite")
    queue_path.write_text(queue_text)
    files["queue.jsonl"] = sha(queue_path)
    old_logs = saved_rows()
    training = [r for r in anchor_rows() if r["first_complete"]]
    x = np.asarray(
        [features(r, old_logs[r["pool"], r["arm_id"], r["task_id"], 0]) for r in training]
    )
    y = np.asarray([r["same_success"] - r["first_success"] for r in training])
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x, y)
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
    predictions = model.predict(x)
    g_predictions = predictions[[r["pool"] == "G" for r in training]]
    rule = {
        "decision": "retry same if first incomplete, or if complete and predicted same-retry gain >= threshold",
        "features": [
            "log1p_steps",
            "log1p_100x_cost",
            "log1p_wallclock_s",
            "log1p_final_result_length",
            "log1p_nontrivial_errors",
            "is_F",
        ],
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
        "ridge_coefficients": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "completed_G_training_prediction_p80": float(np.quantile(g_predictions, 0.8)),
        "threshold": max(0.0, float(np.quantile(g_predictions, 0.8))),
        "n_completed_training_anchors": len(training),
        "ridge_alpha": 10.0,
    }
    rule_path = OUT / "uplift_rule.json"
    rule_text = json.dumps(rule, indent=2) + "\n"
    if rule_path.exists() and rule_path.read_text() != rule_text:
        raise ValueError("existing uplift rule differs; refusing to overwrite")
    rule_path.write_text(rule_text)
    files["uplift_rule.json"] = sha(rule_path)
    manifest = {
        "design": "G_14 first, G_14 rerun, G_16 alternative; isolated resets; one worker",
        "source_manifest_sha256": sha(EMPIRICAL / "manifest.json"),
        "seed": SEED,
        "task_ids": tasks,
        "n_tasks": len(tasks),
        "n_episodes": len(queue),
        "first_arm": FIRST_ARM,
        "different_arm": SECOND_ARM,
        "files": files,
    }
    manifest_path = OUT / "manifest.json"
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != manifest_text:
        old = json.loads(manifest_path.read_text())
        # Permit this one preparation-only addition of the frozen uplift rule.
        prior_files = {k: v for k, v in files.items() if k != "uplift_rule.json"}
        if old != {**manifest, "files": prior_files}:
            raise ValueError("existing trial manifest differs; refusing to overwrite")
    manifest_path.write_text(manifest_text)
    print(f"Prepared {len(queue)} episodes on {len(tasks)} tasks under {OUT}")


if __name__ == "__main__":
    main()
