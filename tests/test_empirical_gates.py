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
import make_pools as mp  # noqa: E402
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


def _queue(attempts, extra=0, pilot=True):
    """The queue the attempts came from (all pilot items), plus `extra` never-attempted items."""
    keys = list(dict.fromkeys(zip(attempts["pool"], attempts["arm_id"], attempts["task_id"], attempts["replicate"],
                                  strict=True)))
    keys += [("G", "G_09", f"never{i}", 0) for i in range(extra)]
    return [mp.QueueItem(index=i, pool=p, arm_id=a, task_id=t, replicate=int(r), pilot=pilot)
            for i, (p, a, t, r) in enumerate(keys)]


def _pilot(attempts, queue=None, relaunches=0, n_workers=2):
    return gates.pilot_gate(attempts, _queue(attempts) if queue is None else queue, n_workers=n_workers,
                            anchor_arm="anchor_baseline", anchor_rate=0.66, relaunches=relaunches)


def test_pilot_gate_passes_a_clean_pilot():
    g = _pilot(_attempts())
    assert g.passed, g.report()


@pytest.mark.parametrize("kw, failing", [
    ({"cost": 0.09}, "cost_per_episode"),
    ({"n_missing": 20}, "missing_rate"),
    ({"anchor_successes": 20}, "anchor_drift"),
    ({"workers": 1}, "every_worker_produced"),
])
def test_pilot_gate_fails_each_check(kw, failing):
    g = _pilot(_attempts(**kw))
    assert not g.passed and not g.checks[failing]["passed"]
    assert all(c["passed"] for k, c in g.checks.items() if k != failing), g.report()


def test_collection_gate():
    a = _attempts()
    assert gates.collection_gate(a, _queue(a)).passed
    a = _attempts(n_missing=20)
    assert not gates.collection_gate(a, _queue(a)).passed


def test_never_attempted_queue_items_count_as_missing():
    a = _attempts(n_missing=0)  # 160 terminal records, all ok
    g = gates.collection_gate(a, _queue(a, extra=5))
    assert g.passed and "5 with no terminal record" in g.checks["missing_rate"]["detail"]
    g = gates.collection_gate(a, _queue(a, extra=20))  # 20 / 180 = 11% never attempted
    assert not g.checks["missing_rate"]["passed"]
    assert "160 ok + 0 missing + 20 with no terminal record" in g.checks["missing_rate"]["detail"]
    g = _pilot(a, queue=_queue(a, extra=20))
    assert not g.checks["missing_rate"]["passed"]


def test_terminal_records_outside_the_queue_fail():
    a = _attempts()
    q = [item for item in _queue(a) if item.task_id != "t0"]
    assert not gates.collection_gate(a, q).checks["records_in_queue"]["passed"]


def test_pilot_scope_is_the_pilot_items_only():
    a = _attempts(n_missing=0)
    q = _queue(a) + [mp.QueueItem(index=10_000 + i, pool="F", arm_id="F_00", task_id=f"later{i}", replicate=0,
                                  pilot=False) for i in range(50)]
    assert _pilot(a, queue=q).passed  # 50 unattempted non-pilot items are not the pilot's business


def test_pilot_gate_requires_zero_watchdog_relaunches(tmp_path):
    a = _attempts()
    assert not _pilot(a, relaunches=1).checks["no_watchdog_relaunch"]["passed"]
    assert not _pilot(a, relaunches=None).checks["no_watchdog_relaunch"]["passed"]
    assert _pilot(a, relaunches=0).checks["no_watchdog_relaunch"]["passed"]


def test_count_relaunches_reads_the_watchdogs_own_lines(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import watchdog_empirical_pool as wd

    log = tmp_path / wd.RELAUNCH_LOG
    assert gates.count_relaunches(log) is None
    log.touch()
    assert gates.count_relaunches(log, mode="pilot") == 0
    wd._record_relaunch(tmp_path, 1, "stale_workers:3", "pilot")
    wd._record_relaunch(tmp_path, 2, "global_stall", "full")
    assert gates.RELAUNCH_LOG == wd.RELAUNCH_LOG
    assert gates.count_relaunches(log, mode="pilot") == 1
    assert gates.count_relaunches(log) == 2


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


def test_synthesize_outcomes_records_mu_true():
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    out = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=30, seed=0)
    assert "mu_true" in out.columns
    d = rehearsal.task_offsets(rehearsal.TASK_RATES)
    rep0 = out[out["replicate"] == 0]
    for (pool, arm), sub in rep0.groupby(["pool", "arm_id"]):
        values = sub["mu_true"].unique()
        assert len(values) == 1, "mu_true must be constant within an arm"
        mu_true = float(values[0])
        assert 0.0 <= mu_true <= 1.0
        a = rehearsal.solve_level(mu_true, d)
        assert np.mean(rehearsal.expit(a + d)) == pytest.approx(mu_true, abs=1e-9)
    # replicate-1 rows carry the same mu_true as their arm's replicate-0 rows.
    lookup = rep0.groupby(["pool", "arm_id"])["mu_true"].first()
    for row in out[out["replicate"] == 1].itertuples():
        assert row.mu_true == pytest.approx(lookup[(row.pool, row.arm_id)])


def test_realized_sd_matches_the_spread_of_the_drawn_true_rates():
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    out = rehearsal.synthesize_outcomes(truth, n_prompts=8, n_replicates=5, seed=1)
    for pool in ("G", "F"):
        got = rehearsal.realized_sd(out, pool)
        per_arm = out[out["pool"] == pool].groupby("arm_id")["mu_true"].first().to_numpy()
        assert got == pytest.approx(float(np.std(per_arm, ddof=0)))
    # A pool's realized spread need not equal (and, for a small n_prompts, should not
    # generally equal) the reservoir's population spread -- that gap is exactly why
    # `run_rehearsal` compares the NPMLE fit against `realized_sd`, not the population.
    assert rehearsal.realized_sd(out, "G") != rehearsal._sd(truth["G"])
