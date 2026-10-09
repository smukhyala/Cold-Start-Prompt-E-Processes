"""EXPLORATORY audit step E3: checks for the cross-pool descriptive table."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "growing_bandits" / "audit" / "cross_pool.py"
SPEC = importlib.util.spec_from_file_location("cross_pool", MODULE_PATH)
assert SPEC and SPEC.loader
cross_pool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cross_pool
SPEC.loader.exec_module(cross_pool)


def _frame(pool: str, rates: list[float], n_tasks: int = 20) -> pd.DataFrame:
    rows = []
    for i, rate in enumerate(rates):
        for j in range(n_tasks):
            hit = 1.0 if (j + 0.5) / n_tasks <= rate else 0.0
            rows.append({"pool": pool, "arm_id": f"a{i}", "task_id": f"task_e{j}", "replicate": 0,
                         "success": hit, "status": "ok"})
    return pd.DataFrame(rows)


def test_bernoulli_v_is_zero_when_every_task_is_deterministic() -> None:
    frame = _frame("p", [0.0, 0.0, 0.0])
    assert cross_pool.bernoulli_v(frame, "p") == 0.0


def test_bernoulli_v_matches_hand_value() -> None:
    frame = _frame("p", [0.0, 1.0])  # every task has p_t = 0.5
    assert cross_pool.bernoulli_v(frame, "p") == pytest.approx(0.25)


def test_variant_outcomes_drops_only_named_arms() -> None:
    frame = _frame("p", [0.2, 0.5, 0.8])
    out = cross_pool.variant_outcomes(frame, "p", "q", ("a1",))
    assert set(out["arm_id"]) == {"a0", "a2"}
    assert set(out["pool"]) == {"q"}
    assert len(frame) == 60


def test_reservoir_stats_tail_mass_and_best_atom() -> None:
    rates = [0.50] * 30 + [0.80] * 3
    frame = _frame("p", rates)
    scores = cross_pool.pool_scores(frame, "p")
    out = cross_pool.reservoir_stats(scores, 0.01)
    assert out["upper_tail_mass_15"] == pytest.approx(3 / 33, abs=0.03)
    assert out["upper_tail_mass_25"] <= out["upper_tail_mass_15"]
    assert out["best_atom_minus_level"] > 0.15
    assert out["npmle_sd"] <= out["raw_sd"] + 1e-9


def test_beta_calibration_interpolates_and_anchors_zero() -> None:
    cal = pd.DataFrame({
        "family": ["beta", "beta", "mixture"], "pool_id": ["b1", "b2", "m1"],
        "true_sd": [0.02, 0.10, 0.08], "upper_tail_mass": [0.0, 0.0, 0.05],
        "regret_range": [0.01, 0.05, 0.12], "horizon": [200, 200, 200]})
    mid = cross_pool.calibration_placement(cal, 0.06, 0.0)
    assert mid["beta_regret_range_T200"] == pytest.approx(0.03)
    assert np.isnan(mid["mixture_regret_range_T200"])
    low = cross_pool.calibration_placement(cal, 0.0, 0.0)
    assert low["beta_regret_range_T200"] == 0.0
    tail = cross_pool.calibration_placement(cal, 0.08, 0.05)
    assert tail["mixture_nearest_id"] == "m1"
    assert tail["mixture_regret_range_T200"] == pytest.approx(0.12)
    assert cross_pool.calibration_placement(cal, 0.5, 0.0)["beta_extrapolated"] is True


def test_association_uses_seven_independent_pools() -> None:
    labels = list(cross_pool.POOLS) + ["gitlab17_no_cookbook", "gitlab16_no_cookbook_no_explorer"]
    rows = []
    for k, lab in enumerate(labels):
        row = {"pool_variant": lab, "winner_gain_pp": float(k)}
        for col in cross_pool.PREDICTORS:
            row[col] = float(k)
        rows.append(row)
    assoc = cross_pool.association(pd.DataFrame(rows))
    sevens = assoc[assoc["subset"].str.startswith("7_pools")]
    assert set(sevens["n"]) == {7}
    assert set(assoc[assoc["subset"].str.startswith("all_9")]["n"]) == {9}


def test_committed_values_reproduce_for_G() -> None:
    path = cross_pool.MATRICES / "G.csv"
    if not path.exists():
        pytest.skip("matrix export not present")
    outcomes = cross_pool.load_outcomes()
    own = outcomes[outcomes["pool"] == "G"]
    v, _ = cross_pool.emp.within_cell_variance(own, ["G"])
    out = cross_pool.reservoir_stats(cross_pool.pool_scores(own, "G"), v)
    assert out["npmle_sd"] == pytest.approx(cross_pool.COMMITTED["G"], abs=0.0015)
