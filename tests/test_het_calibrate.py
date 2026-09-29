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
import study as st  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402


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


def test_calibration_seed_range_lies_above_every_study_and_has_headroom():
    """The calibration grid's full seed range (every pool index x every horizon) sits
    strictly above the highest seed Pre-registration 9 or the heterogeneity study ever
    uses, and strictly below 2**53 (the float64/JSON-safe integer boundary), so no seed
    silently loses precision on its way through JSON."""
    import replay  # noqa: E402  (already on sys.path via `calibrate`'s own import)

    assert cal.CAL_SEED_BASE == 50_000_000_000
    n_pools = len(cal.SPREADS) + len(cal.TAIL_FRACS) * len(cal.TAIL_DELTAS)
    cal_min = cal._cal_seed(0, min(cal.HORIZONS))
    cal_max = cal._cal_seed(n_pools - 1, max(cal.HORIZONS))
    assert cal_min < cal_max

    prereg9_max = max(
        replay.seed_for(p, T, b, study=st.PREREG9)
        for p in st.PREREG9.pools for T in replay.ALL_HORIZONS for b in (None, replay.N_BOOT - 1)
    )
    het_max = max(
        replay.seed_for(p, T, b, study=st.HETEROGENEITY)
        for p in st.HETEROGENEITY.pools for T in replay.ALL_HORIZONS for b in (None, replay.N_BOOT - 1)
    )
    assert cal_min > prereg9_max
    assert cal_min > het_max
    assert cal_max < 2**53


def test_upper_tail_mass_matches_stage0s_at_or_above_weighted_median_convention():
    """Pins the spec section 6.4 definition on a small reservoir with known atoms: mass
    at or above the weighted median + 0.10, computed with >=, not `tail_prob`'s strict >."""
    res = EmpiricalReservoir([0.2, 0.4, 0.6, 0.8], [0.1, 0.4, 0.3, 0.2], validate=False)
    # cumulative weights [0.1, 0.5, 0.8, 1.0]: the weighted median (inf{x: F(x) >= 0.5}) is 0.4.
    assert res.quantile(0.5) == pytest.approx(0.4)
    # threshold = 0.4 + 0.10 = 0.5, sitting exactly on no atom; mass of atoms >= 0.5 is 0.6 and 0.8.
    assert cal._upper_tail_mass(res) == pytest.approx(0.3 + 0.2)

    # a threshold landing exactly on an atom must still be included (the >= convention).
    on_atom = EmpiricalReservoir([0.30, 0.40, 0.50], [0.2, 0.3, 0.5], validate=False)
    assert on_atom.quantile(0.5) == pytest.approx(0.40)
    assert cal._upper_tail_mass(on_atom) == pytest.approx(0.5)  # only the 0.50 atom (== 0.40 + 0.10)


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
