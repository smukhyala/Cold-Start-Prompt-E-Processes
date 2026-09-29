"""Value-of-search calibration: synthetic pools of known spread, tabulated regret range and
policy gaps over K, so the pre-registration can pick tau_flat from a spread that is known to
be flat rather than assumed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))
import calibrate as cal  # noqa: E402


def test_pools_cover_the_registered_grid():
    pools = cal.calibration_pools()
    assert len(pools) == len(cal.SPREADS) + len(cal.TAIL_FRACS) * len(cal.TAIL_DELTAS)
    u = (np.arange(20001) + 0.5) / 20001
    sd = np.std(pools["beta_sd0.05"].sample_from_uniforms(u))
    assert abs(sd - 0.05) < 0.003


def test_calibrate_tiny_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "SPREADS", (0.02, 0.10))
    monkeypatch.setattr(cal, "TAIL_FRACS", ())
    monkeypatch.setattr(cal, "HORIZONS", (20,))
    monkeypatch.setattr(cal, "K_GRID", (2, 4, 8))
    out = cal.calibrate(workers=1, m=16, out_dir=tmp_path)
    assert set(out["pool_id"]) == {"beta_sd0.02", "beta_sd0.1"}
    wide = out.set_index("pool_id")["regret_range"]
    assert wide["beta_sd0.1"] > wide["beta_sd0.02"]


def test_calibrate_columns_and_seed_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "SPREADS", (0.02,))
    monkeypatch.setattr(cal, "TAIL_FRACS", (0.05,))
    monkeypatch.setattr(cal, "TAIL_DELTAS", (0.2,))
    monkeypatch.setattr(cal, "HORIZONS", (20,))
    monkeypatch.setattr(cal, "K_GRID", (2, 4, 8))
    out = cal.calibrate(workers=1, m=16, out_dir=tmp_path)
    required = {"pool_id", "family", "spread", "tail_frac", "tail_delta", "true_sd", "upper_tail_mass",
                "horizon", "regret_range", "k_star", "gap_fixed_K8", "gap_p3_star", "gap_always_search"}
    assert required <= set(out.columns)
    beta_row = out.set_index("pool_id").loc["beta_sd0.02"]
    assert beta_row["family"] == "beta" and np.isnan(beta_row["tail_frac"])
    tail_row = out.set_index("pool_id").loc["tail_f0.05_d0.2"]
    assert tail_row["family"] == "mixture"
    assert tail_row["tail_frac"] == pytest.approx(0.05) and tail_row["tail_delta"] == pytest.approx(0.2)
    assert (out["true_sd"] > 0).all()
    written_path = tmp_path / "calibration.csv"
    assert written_path.exists()
    written = pd.read_csv(written_path)
    assert set(written["pool_id"]) == set(out["pool_id"])


def test_seeds_are_disjoint_from_every_other_seed_range():
    assert cal.CAL_SEED_BASE == 50_000_000_000
    # every existing seed range (Pre-reg 9, heterogeneity, corpus, smoke) lies below 2.1e10.
    assert cal.CAL_SEED_BASE > 21_000_000_000


def test_tau_flat_from_calibration_interpolates_the_crossing_point():
    frame = pd.DataFrame([
        {"pool_id": "beta_sd0.01", "family": "beta", "true_sd": 0.01, "horizon": 200, "regret_range": 0.002},
        {"pool_id": "beta_sd0.02", "family": "beta", "true_sd": 0.02, "horizon": 200, "regret_range": 0.004},
        {"pool_id": "beta_sd0.035", "family": "beta", "true_sd": 0.035, "horizon": 200, "regret_range": 0.007},
        {"pool_id": "beta_sd0.05", "family": "beta", "true_sd": 0.05, "horizon": 200, "regret_range": 0.010},
        # a non-beta pool at a nearby true_sd must not be picked up by the interpolation.
        {"pool_id": "tail_f0.05_d0.2", "family": "mixture", "true_sd": 0.03, "horizon": 200, "regret_range": 100.0},
    ])
    tau = cal.tau_flat_from_calibration(frame, horizon=200, target=0.005)
    # crosses 0.005 linearly between (0.02, 0.004) and (0.035, 0.007)
    expected = 0.02 + (0.035 - 0.02) * (0.005 - 0.004) / (0.007 - 0.004)
    assert tau == pytest.approx(expected, abs=1e-9)


def test_tau_flat_from_calibration_raises_when_never_crossed():
    frame = pd.DataFrame([
        {"pool_id": "beta_sd0.01", "family": "beta", "true_sd": 0.01, "horizon": 200, "regret_range": 0.001},
        {"pool_id": "beta_sd0.02", "family": "beta", "true_sd": 0.02, "horizon": 200, "regret_range": 0.002},
    ])
    with pytest.raises(ValueError, match="never crosses"):
        cal.tau_flat_from_calibration(frame, horizon=200, target=0.005)
