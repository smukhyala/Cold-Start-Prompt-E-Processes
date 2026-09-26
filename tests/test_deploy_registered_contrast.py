"""`registered_contrast.py` computes exactly the H1b' statistic DEPLOYMENT_PLAN.md registers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import registered_contrast as rc  # noqa: E402


def _write(root: Path, test: str, env: str, T: int, seed: int, shifts: dict[str, float], n: int = 40) -> None:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.12, 0.02, size=n)
    for policy, shift in shifts.items():
        path = root / "episodes" / test / f"{env}_T{T}_cap64" / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cell": f"{env}_T{T}_cap64", "family": "A", "policy": policy, "env_id": env, "horizon": T,
            "cap": 64, "base_seed": seed, "episode": np.arange(n),
            "regret_posterior_mean_shrunk": base + shift + rng.normal(0, 0.001, size=n),
        }).to_parquet(path, index=False)


@pytest.fixture
def run(tmp_path):
    """Six environments x five horizons. phi_k4 beats p3_star by 0.004 at T <= 200 and ties at T >= 500."""
    root = tmp_path / "deploy"
    for i in range(6):
        for T in (50, 100, 200, 500, 1000):
            shift = -0.004 - 0.0005 * (i % 2) if T <= 200 else 0.0
            _write(root, "robust", f"env{i}", T, 100 + 10 * i + T, {"phi_k4": shift, "p3_star": 0.0})
    return root


def test_primary_row_is_the_pooled_t_over_the_registered_horizons(run):
    out = rc.registered_contrast("phi_k4", "p3_star", test="robust", horizons=(50, 100, 200),
                                 mei=0.002, out_dir=run, n_boot=300)
    primary = out[out["row"] == "primary"].iloc[0]
    assert primary["n_cells"] == 18 and primary["n_envs"] == 6
    assert primary["delta"] == pytest.approx(-0.00425, abs=3e-4)
    assert primary["cluster_method"] == "env_mean_t" and primary["cluster_p"] < 0.05
    assert primary["mei"] == 0.002 and primary["verdict"] == "supported"


def test_secondary_rows_are_holm_corrected_among_the_registered_horizons(run):
    out = rc.registered_contrast("phi_k4", "p3_star", test="robust", horizons=(50, 100, 200),
                                 mei=0.002, out_dir=run, n_boot=300)
    sec = out[out["row"] == "secondary"].set_index("horizon")
    assert sorted(sec.index) == [50, 100, 200]
    assert (sec["p_holm"] >= sec["cluster_p"]).all() and (sec["p_holm"] <= 1.0).all()
    structural = out[out["row"] == "structural"].set_index("horizon")
    assert sorted(structural.index) == [500, 1000]
    assert structural["p_holm"].isna().all()
    assert structural["delta"].abs().max() < 1e-3


def test_verdict_rules(run):
    # Refuted: the policy is worse.
    for i in range(6):
        for T in (50, 100, 200):
            _write(run, "robust", f"env{i}", T, 100 + 10 * i + T, {"phi_k4": +0.003, "p3_star": 0.0})
    out = rc.registered_contrast("phi_k4", "p3_star", test="robust", horizons=(50, 100, 200),
                                 mei=0.002, out_dir=run, n_boot=300)
    assert out[out["row"] == "primary"].iloc[0]["verdict"] == "refuted"
    # Detectable but below the MEI: significant, small, and the interval excludes -MEI -> refuted.
    for i in range(6):
        for T in (50, 100, 200):
            _write(run, "robust", f"env{i}", T, 100 + 10 * i + T, {"phi_k4": -0.0004, "p3_star": 0.0})
    out = rc.registered_contrast("phi_k4", "p3_star", test="robust", horizons=(50, 100, 200),
                                 mei=0.002, out_dir=run, n_boot=300)
    p = out[out["row"] == "primary"].iloc[0]
    assert p["cluster_p"] < 0.05 and p["verdict"] == "refuted"


def test_refuses_a_missing_policy(run):
    with pytest.raises(FileNotFoundError, match="phi_k1"):
        rc.registered_contrast("phi_k1", "p3_star", test="robust", horizons=(50,), mei=0.002,
                               out_dir=run, n_boot=50)


def test_noninferiority_and_not_better_rules_on_the_paired_ci():
    # Non-inferiority: supported iff the paired upper bound is below +MEI.
    assert rc.verdict_by_rule("noninferiority", delta=0.0005, lo=-0.001, hi=0.0015, mei=0.002) == "supported"
    assert rc.verdict_by_rule("noninferiority", delta=0.003, lo=0.0021, hi=0.004, mei=0.002) == "refuted"
    assert rc.verdict_by_rule("noninferiority", delta=0.0015, lo=0.0, hi=0.003, mei=0.002) == "inconclusive"
    # Not-better: supported iff the paired lower bound is above -MEI.
    assert rc.verdict_by_rule("not_better", delta=0.001, lo=-0.001, hi=0.003, mei=0.002) == "supported"
    assert rc.verdict_by_rule("not_better", delta=-0.004, lo=-0.006, hi=-0.0025, mei=0.002) == "refuted"
    assert rc.verdict_by_rule("not_better", delta=-0.001, lo=-0.003, hi=0.001, mei=0.002) == "inconclusive"
    for name in ("capc_primary", "capc_level", "capc_phi"):
        assert name in rc.REGISTRATIONS and rc.REGISTRATIONS[name]["test"] == "capc"
    assert rc.REGISTRATIONS["capc_primary"]["rule"] == "noninferiority"
    assert rc.REGISTRATIONS["capc_level"]["rule"] == "not_better"


def test_paired_rule_registration_uses_the_paired_ci_below_cluster_min_envs(tmp_path):
    """Three environments: the cluster interval is NaN, and a paired-rule registration still decides."""
    root = tmp_path / "deploy"
    for i in range(3):
        for T in (50, 100):
            _write(root, "capc", f"m{i}", T, 500 + 10 * i + T, {"p3_star": 0.0, "fixed_K_star": 0.0004})
    out = rc.registered_contrast("p3_star", "fixed_K_star", test="capc", horizons=(50, 100), mei=0.002,
                                 out_dir=root, n_boot=300, rule="noninferiority")
    p = out[out["row"] == "primary"].iloc[0]
    assert np.isnan(p["cluster_p"]) and p["n_envs"] == 3
    assert p["delta"] == pytest.approx(-0.0004, abs=3e-4) and p["hi"] < 0.002
    assert p["verdict"] == "supported" and p["rule"] == "noninferiority"
