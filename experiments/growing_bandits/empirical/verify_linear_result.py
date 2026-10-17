"""Verify the saved Linear outcome tables and registered paths without worker logs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths  # noqa: E402
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths  # noqa: E402

DATA = ROOT / "data/prospective_linear/manifest.json"
RESULTS = ROOT / "results/paper/prospective_linear"


def main() -> None:
    manifest = json.loads(DATA.read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    with (RESULTS / "tasks.csv").open(newline="") as f:
        tasks = list(csv.DictReader(f))
    with (RESULTS / "attempts.csv").open(newline="") as f:
        attempts = list(csv.DictReader(f))
    assert [r["task_id"] for r in tasks] == manifest["task_ids"]
    assert len(tasks) == 120 and len(attempts) == 241
    assert sum(r["status"] == "ok" for r in attempts) == 240
    assert sum(r["status"] == "infra_error" for r in attempts) == 1
    assert {(r["task_id"], r["arm_id"]) for r in attempts if r["status"] == "ok"} == {
        (task, arm) for task in manifest["task_ids"] for arm in manifest["arms"]}
    assert {(r["arm_id"], r["prompt_sha256"]) for r in attempts} == set(manifest["prompt_sha256"].items())
    inc = np.array([int(r["incumbent_success"]) for r in tasks])
    chal = np.array([int(r["challenger_success"]) for r in tasks])
    Y = np.stack([inc, chal])[None, :, :]
    eff = e1b_log_paths(Y, np.array([0]), np.array([[False, True]]))[0]
    fut = futility_log_paths((chal - inc)[None, None, :], 0.15)[0, 0]
    boundary = math.log(40)
    def first(path):
        hits = np.flatnonzero(path >= boundary)
        return None if len(hits) == 0 else int(hits[0]) + 1
    assert (int(inc.sum()), int(chal.sum())) == (64, 66)
    assert first(eff) == summary["efficacy_first_crossing"]
    assert first(fut) == summary["futility_first_crossing"] == 93
    assert math.isclose(float(np.exp(eff[-1])), summary["efficacy_final_e"], rel_tol=1e-12)
    assert math.isclose(float(np.exp(fut[-1])), summary["futility_final_e"], rel_tol=1e-12)
    assert math.isclose(sum(float(r["cost_usd"]) for r in attempts), summary["cost_usd"], abs_tol=1e-8)
    print("verified 120 fixed-order pairs, 241 attempts, registered crossings, and spend")


if __name__ == "__main__":
    main()
