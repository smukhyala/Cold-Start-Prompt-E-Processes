"""Verify the saved sampled Linear tables and registered paths without worker logs."""

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

DATA = ROOT / "data/prospective_sampled_linear"
RESULTS = ROOT / "results/paper/prospective_sampled_linear"
INC, CHAL = "linear_baseline", "linear_explorer"
PROMPTS = {
    INC: "ae8c7c7fb6984ed82d52d8dd454e6672e8d7a1bf544cd727affdb35cb539ef82",
    CHAL: "5e4ee37361c08c0a0282fc548c5d8f6467ea4fed7c4211252b4440241a05c024",
}


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    bank = json.loads((DATA / "tasks.json").read_text())
    queue = [json.loads(line) for line in (DATA / "sampled_linear/queue.jsonl").read_text().splitlines()]
    summary = json.loads((RESULTS / "summary.json").read_text())
    with (RESULTS / "tasks.csv").open(newline="") as f:
        tasks = list(csv.DictReader(f))
    with (RESULTS / "attempts.csv").open(newline="") as f:
        attempts = list(csv.DictReader(f))

    ids = manifest["task_ids"]
    assert len(ids) == len(set(ids)) == 120
    assert [task["id"] for task in bank] == ids
    assert [row["task_id"] for row in tasks] == ids
    assert [row["family"] for row in tasks] == [task["family"] for task in bank]
    assert len(queue) == len(attempts) == 240
    assert [(int(row["index"]), row["task_id"], row["arm_id"]) for row in queue] == [
        (int(row["queue_index"]), row["task_id"], row["arm_id"]) for row in attempts
    ]
    assert all(row["status"] == "ok" and row["attempt"] == "1" for row in attempts)
    assert all(row["prompt_sha256"] == PROMPTS[row["arm_id"]] for row in attempts)
    assert all(row["llm_model"] == "gpt-5.4-mini" for row in attempts)
    assert all(row["timed_out"] == "False" for row in attempts)
    assert all(row["incumbent_status"] == row["challenger_status"] == "ok" for row in tasks)
    assert {(row["task_id"], row["arm_id"]) for row in attempts} == {
        (task_id, arm) for task_id in ids for arm in (INC, CHAL)
    }
    by_pair = {(row["task_id"], row["arm_id"]): row for row in attempts}
    for row in tasks:
        for arm, side in ((INC, "incumbent"), (CHAL, "challenger")):
            episode = by_pair[(row["task_id"], arm)]
            assert int(float(episode["success"])) == int(row[f"{side}_success"])
            assert episode["timestamp_utc"] == row[f"{side}_timestamp_utc"]

    inc = np.array([int(row["incumbent_success"]) for row in tasks])
    chal = np.array([int(row["challenger_success"]) for row in tasks])
    eff = e1b_log_paths(np.stack([inc, chal])[None, :, :], np.array([0]), np.array([[False, True]]))[0]
    fut = futility_log_paths((chal - inc)[None, None, :], 0.15)[0, 0]

    def first(path: np.ndarray) -> int | None:
        hits = np.flatnonzero(path >= math.log(40))
        return None if len(hits) == 0 else int(hits[0]) + 1

    assert (int(inc.sum()), int(chal.sum())) == (3, 1)
    assert int(((inc == 0) & (chal == 1)).sum()) == summary["challenger_only_wins"] == 1
    assert int(((inc == 1) & (chal == 0)).sum()) == summary["incumbent_only_wins"] == 3
    assert first(eff) == summary["efficacy_first_crossing"] is None
    assert first(fut) == summary["futility_first_crossing"] == 44
    for key, value in (("efficacy_max_e", np.exp(eff.max())),
                       ("efficacy_final_e", np.exp(eff[-1])),
                       ("futility_max_e", np.exp(fut.max())),
                       ("futility_final_e", np.exp(fut[-1]))):
        assert math.isclose(float(value), summary[key], rel_tol=1e-12)
    assert math.isclose(float(inc.mean()), summary["incumbent_rate"], abs_tol=1e-12)
    assert math.isclose(float(chal.mean()), summary["challenger_rate"], abs_tol=1e-12)
    assert math.isclose(float((chal - inc).mean()), summary["paired_gain"], abs_tol=1e-12)
    assert math.isclose(sum(float(row["cost_usd"]) for row in attempts), summary["cost_usd"], abs_tol=1e-8)
    assert summary["protocol_commit"] == manifest["protocol_commit"]
    assert summary["seed_hex"] == manifest["seed_hex"]
    assert summary["complete_pairs_without_missing"] and summary["missing_episodes"] == 0
    assert summary["attempts"] == summary["terminal_episodes"] == 240
    print("verified 120 sampled-order pairs, 240 attempts, registered crossings, and spend")


if __name__ == "__main__":
    main()
