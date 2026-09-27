"""Spec section 9, end to end: fake collector -> frozen snapshot -> reservoirs -> replay -> contrast table.

Two prompts per pool x three tasks, plus replicates, collected by the real `run_worker` on a
scripted fake adapter (no browser, no LLM, no spend); then the real `replay.run_estimate`,
real `run_deployment.main` replays (point and prompt-bootstrap, tiny M, T = 20, policies that
need no tuned constants), the real K-grid and flatness table, and the real
`prompt_bootstrap_contrast` with the reservoir-manifest check on. The only fixture step is
marking both primary cells informative: at M = 8 the flatness guard is noise, and the point
of this test is the bootstrap path, not the guard (the guard has its own tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import collect  # noqa: E402
import describe  # noqa: E402
import gates  # noqa: E402
import make_pools as mp  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.types import RunResult  # noqa: E402

TASKS = ["t0", "t1", "t2"]
ARMS = {"G": ["G_00", "G_01"], "F": ["F_00", "F_01"]}
T = 20
POLICY, REFERENCE = "fixed_K16", "always_search"


class _ScriptedAdapter:
    """First call of (arm, task) succeeds per `FIRST`; the replicate call flips on t0, so v > 0 in both pools."""

    FIRST = {("G_00", "t0"): 1, ("G_00", "t1"): 1, ("G_00", "t2"): 0, ("G_01", "t0"): 1, ("G_01", "t1"): 0,
             ("G_01", "t2"): 0, ("F_00", "t0"): 1, ("F_00", "t1"): 1, ("F_00", "t2"): 1, ("F_01", "t0"): 0,
             ("F_01", "t1"): 1, ("F_01", "t2"): 0}

    def __init__(self):
        self.seen: dict[tuple[str, str], int] = {}

    def task_ids(self):
        return list(TASKS)

    def task_by_id(self, task_id):
        from cold_start.types import Task

        return Task(task_id=task_id, payload={}, metadata={})

    def register_prompt(self, arm_id, text):
        pass

    def run_arm(self, arm, task, runner, max_steps, artifact_subdir=None):
        key = (arm.arm_id, task.task_id)
        n = self.seen.get(key, 0)
        self.seen[key] = n + 1
        ok = self.FIRST[key] if n == 0 or task.task_id != "t0" else 1 - self.FIRST[key]
        return RunResult(success=bool(ok), reward=float(ok), steps=3, wallclock_s=1.0,
                         trace={"is_done": True, "error_tail": [], "verifier_message": "", "final_result": ""},
                         tokens={"cost_usd": 0.01})

    def reset(self, seed):
        pass

    def close(self):
        pass


def _prompts():
    out = {}
    for pool, arms in ARMS.items():
        for arm_id in arms:
            arm = mp.freeform_arm(arm_id, f"Guidance for {arm_id}. " * 8)
            out[arm_id] = mp.PoolArm(pool=pool, arm=arm, template="x", text=arm.prompt_guidance, sha256="s")
    return out


def test_collect_estimate_replay_contrast(tmp_path):
    logs, res, out = tmp_path / "logs", tmp_path / "res", tmp_path / "out"

    # ---- collect: 2 prompts x 2 pools x 3 tasks + every cell replicated once, two workers
    queue = mp.build_queue(ARMS, TASKS, n_replicates=12, pilot_arms=set(), seed=7)
    shares = collect.partition_remaining(queue, logs, 2, pilot=False)
    prompts = _prompts()
    for w in range(2):
        cfg = collect.WorkerConfig(worker=w, n_workers=2, log_dir=logs, budget_usd=100.0, assigned=shares[w])
        assert collect.run_worker(cfg, queue, prompts, _ScriptedAdapter()) == "done"
    attempts = collect.load_all(logs)
    assert gates.collection_gate(attempts, queue).passed
    assert len(emp.terminal_outcomes(attempts)) == len(queue) == 24

    # ---- estimate: frozen snapshot + reservoirs + manifest
    replay.run_estimate(logs, res, out)
    shas = replay.verify_reservoirs(res)
    manifest = res / replay.MANIFEST_FILE

    # ---- point replay (test emp) on the primary reservoirs
    point = [replay.make_emp_cell(p, "npmle", T, replay.load_reservoir(res / f"{p}_npmle.json"), 8)
             for p in replay.POOLS]
    rd.main(["--test", "emp", "--workers", "1", "--policies", f"{POLICY},{REFERENCE}", "--out-dir", str(out),
             "--skip-summary"], cells=point)
    rc.write_reservoir_stamp(out, "emp", manifest)

    # ---- prompt bootstrap (test emp_boot) from the snapshot, B = 3
    noise = replay.json.loads((res / "noise.json").read_text())
    boots = replay.bootstrap_reservoirs(replay.load_snapshot(res), noise, n_boot=3, seed=11)
    boot_cells = [replay.make_emp_cell(p, "npmle", T, r, 8, boot=b) for b, p, r in boots]
    rd.main(["--test", "emp_boot", "--workers", "1", "--policies", f"{POLICY},{REFERENCE}", "--out-dir", str(out),
             "--skip-summary"], cells=boot_cells)
    rc.write_reservoir_stamp(out, "emp_boot", manifest)

    # ---- K-grid -> flatness (sha-stamped), then force both cells informative (see module docstring)
    kg = replay.stamp_kgrid(replay.kgrid(point, workers=1, k_grid=(2, 4, 8, 16)), shas)
    describe.check_kgrid_snapshot(kg, shas)
    flat = describe.flatness(kg).assign(informative=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    flat.to_csv(out / "tables" / "emp_flatness.csv", index=False)

    # ---- the contrast table, through the registered decision path
    table = rc.prompt_bootstrap_contrast(POLICY, REFERENCE, horizons=(T,), mei=0.002, rule="noninferiority",
                                         out_dir=out, expected_n_boot=3, reservoir_manifest=manifest,
                                         paired_n_boot=500)
    row = table.iloc[0]
    assert list(table.columns) == list(rc.BOOT_COLUMNS)
    assert row["n_informative"] == 2 and row["n_boot"] == 3
    assert np.isfinite([row["delta"], row["lo"], row["hi"], row["paired_lo"], row["paired_hi"]]).all()
    assert row["lo"] <= row["hi"] and row["paired_lo"] <= row["delta"] <= row["paired_hi"]
    assert row["verdict"] in {"supported", "refuted", "inconclusive"}
    assert not bool(row["as_registered"])  # a smoke contrast, not one of the registrations
    written = pd.read_csv(out / "tables" / "emp_flatness.csv")
    assert set(written["reservoir_sha256"]) == {shas["emp_G_npmle"], shas["emp_F_npmle"]}
