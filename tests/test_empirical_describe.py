"""Descriptive reads of the real pools, and the flatness guard Pre-registration 9 turns on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import describe  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402


def _kgrid():
    rows = []
    # G at T=50: flat (range 0.004); F at T=50: a U with range 0.05 and minimum at K=16.
    for K, g, f in [(4, 0.101, 0.15), (8, 0.100, 0.12), (16, 0.102, 0.10), (32, 0.104, 0.13)]:
        rows.append({"env_id": "emp_G_npmle", "pool": "G", "variant": "npmle", "horizon": 50, "K": K, "regret": g})
        rows.append({"env_id": "emp_F_npmle", "pool": "F", "variant": "npmle", "horizon": 50, "K": K, "regret": f})
    for T, K64, Kbest in [(500, 0.09, 0.07), (1000, 0.10, 0.06)]:
        for K, r in [(64, K64), (256, Kbest)]:
            rows.append({"env_id": "emp_F_npmle", "pool": "F", "variant": "npmle", "horizon": T, "K": K, "regret": r})
    return pd.DataFrame(rows)


def test_k_star_and_flatness():
    table = describe.k_star_table(_kgrid())
    g = table[(table["pool"] == "G") & (table["horizon"] == 50)].iloc[0]
    f = table[(table["pool"] == "F") & (table["horizon"] == 50)].iloc[0]
    assert g["k_star"] == 8 and g["regret_range"] == pytest.approx(0.004) and not g["informative"]
    assert f["k_star"] == 16 and f["regret_range"] == pytest.approx(0.05) and f["informative"]
    flat = describe.flatness(_kgrid())
    assert set(flat.columns) >= {"env_id", "pool", "variant", "horizon", "regret_range", "informative"}


def test_ties_go_to_the_smaller_k():
    kg = pd.DataFrame([{"env_id": "emp_G_npmle", "pool": "G", "variant": "npmle", "horizon": 50, "K": K,
                        "regret": 0.1} for K in (4, 8)])
    assert describe.k_star_table(kg).iloc[0]["k_star"] == 4


def test_cap64_cost_is_labelled_extrapolation():
    cost = describe.cap64_cost(_kgrid())
    row = cost[cost["horizon"] == 1000].iloc[0]
    assert row["cost"] == pytest.approx(0.04) and row["label"] == "extrapolation"


def test_cross_pool_prediction():
    kstar = pd.DataFrame([
        {"pool": "G", "variant": "npmle", "horizon": 50, "k_star": 16},
        {"pool": "F", "variant": "npmle", "horizon": 50, "k_star": 32},
    ])
    out = describe.cross_pool_prediction({"G": 0.7, "F": 0.5}, kstar).iloc[0]
    assert out["low_pool"] == "F" and bool(out["sign_agrees"])
    assert out["ratio_observed"] == pytest.approx(2.0)
    assert out["ratio_predicted"] == pytest.approx(np.exp(4 * 0.2))


def test_pool_summary():
    s = describe.pool_summary(EmpiricalReservoir([0.4, 0.6], [0.5, 0.5]))
    assert s["level"] == pytest.approx(0.5) and s["sd"] == pytest.approx(0.1)
    assert s["q99_minus_mean"] == pytest.approx(0.1)


def test_locate_ranks_against_the_corpus():
    pools = pd.DataFrame([{"env_id": "emp_G_npmle", "level": 0.5, "sd": 0.03, "q99_minus_mean": 0.05}])
    corpus = pd.DataFrame([{"env_id": f"c{i}", "level": 0.1 * i, "sd": 0.05 * i, "q99_minus_mean": 0.1 * i}
                           for i in range(1, 9)])
    out = describe.locate(pools, corpus).iloc[0]
    assert out["sd_pct_below"] == pytest.approx(0.0)
    assert out["level_pct_below"] == pytest.approx(4 / 8)
