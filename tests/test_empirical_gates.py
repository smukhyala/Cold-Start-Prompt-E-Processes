"""Gates G1-G3 as code, and the rehearsal's synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import gates  # noqa: E402
import rehearsal  # noqa: E402

from cold_start.growing.reservoirs import BetaReservoir  # noqa: E402


def _attempts(n_ok=100, n_missing=2, cost=0.03, anchor_successes=40, workers=2):
    rows = []
    for i in range(n_ok):
        rows.append({"pool": "G", "arm_id": f"G_{i % 5:02d}", "task_id": f"t{i}", "replicate": 0, "attempt": 1,
                     "status": "ok", "success": i % 2, "cost_usd": cost, "worker": i % workers})
    for i in range(n_missing):
        rows.append({"pool": "G", "arm_id": "G_00", "task_id": f"m{i}", "replicate": 0, "attempt": 3,
                     "status": "missing", "success": None, "cost_usd": 0.0, "worker": 0})
    for j in range(60):
        rows.append({"pool": "anchor", "arm_id": "anchor_baseline", "task_id": f"a{j}", "replicate": 0,
                     "attempt": 1, "status": "ok", "success": int(j < anchor_successes), "cost_usd": cost,
                     "worker": j % workers})
    return pd.DataFrame(rows)


def test_pilot_gate_passes_a_clean_pilot():
    g = gates.pilot_gate(_attempts(), n_workers=2, anchor_arm="anchor_baseline", anchor_rate=0.66)
    assert g.passed, g.report()


@pytest.mark.parametrize("kw, failing", [
    ({"cost": 0.09}, "cost_per_episode"),
    ({"n_missing": 20}, "missing_rate"),
    ({"anchor_successes": 20}, "anchor_drift"),
    ({"workers": 1}, "every_worker_produced"),
])
def test_pilot_gate_fails_each_check(kw, failing):
    g = gates.pilot_gate(_attempts(**kw), n_workers=2, anchor_arm="anchor_baseline", anchor_rate=0.66)
    assert not g.passed and not g.checks[failing]["passed"]


def test_collection_gate():
    assert gates.collection_gate(_attempts()).passed
    assert not gates.collection_gate(_attempts(n_missing=20)).passed


def test_rehearsal_gate():
    good = {"flat": {"sd_true": 0.03, "sd_npmle": 0.035, "sd_raw": 0.05},
            "wide": {"sd_true": 0.16, "sd_npmle": 0.155, "sd_raw": 0.165},
            "k_grid": [8, 12, 16, 24, 32, 48], "k_star_true": 24, "k_star_est": 32}
    assert gates.rehearsal_gate(good).passed
    bad = {**good, "k_star_est": 48}
    assert not gates.rehearsal_gate(bad).checks["k_star_within_one_step"]["passed"]
    flat_bad = {**good, "flat": {"sd_true": 0.03, "sd_npmle": 0.05, "sd_raw": 0.05}}
    assert not gates.rehearsal_gate(flat_bad).passed


def test_task_rates_are_the_logged_sixty():
    assert len(rehearsal.TASK_RATES) == 60
    assert sum(r == 1.0 for r in rehearsal.TASK_RATES) == 13


def test_level_solver_hits_the_prompt_mean():
    d = rehearsal.task_offsets(rehearsal.TASK_RATES)
    for mu in (0.3, 0.6, 0.9):
        a = rehearsal.solve_level(mu, d)
        assert np.mean(rehearsal.expit(a + d)) == pytest.approx(mu, abs=1e-9)


def test_synthesize_outcomes_matches_the_collector_schema():
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    out = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=30, seed=0)
    assert {"pool", "arm_id", "task_id", "replicate", "attempt", "status", "success"} <= set(out.columns)
    assert (out["replicate"] == 1).sum() == 30
    assert len(out[out["replicate"] == 0]) == 2 * 10 * 60
    again = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=30, seed=0)
    pd.testing.assert_frame_equal(out, again)
