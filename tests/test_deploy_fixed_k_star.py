"""`fixed_K_star` (DEPLOYMENT_PLAN.md, Pre-registration 3): a per-horizon fixed K selected on validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import policy_table as pt  # noqa: E402
import select_fixed_k as sk  # noqa: E402

from cold_start.growing.search_policies import PowerSchedule  # noqa: E402

ENV = "beta_good_common"


def test_fixed_k_star_resolves_its_k_per_horizon_from_baseline_params():
    tuned = {"fixed_K_star": {"50": {"K": 20, "pooled_regret": 0.13}, "100": {"K": 32, "pooled_regret": 0.12}}}
    assert pt.resolve_params("fixed_K_star", 50, baseline_params=tuned) == {"K": 20}
    assert pt.resolve_params("fixed_K_star", 100, baseline_params=tuned) == {"K": 32}
    assert pt.baseline_is_tuned("fixed_K_star", 100, tuned) is True
    # A horizon the selection did not cover falls back to the placeholder and says so.
    pt._warned.clear()
    p = pt.resolve_params("fixed_K_star", 500, baseline_params=tuned)
    assert p["K"] == pt._PLACEHOLDER_FIXED_K and p["params_tuned"] is False
    assert pt.baseline_is_tuned("fixed_K_star", 500, tuned) is False
    assert pt.baseline_is_tuned("fixed_K_star", 100, None) is False


def test_fixed_k_star_is_a_cap_stamped_baseline_like_p3_star():
    tuned = {"meta": {"cap": 64}, "fixed_K_star": {"100": {"K": 32, "pooled_regret": 0.12}}}
    pt._warned.clear()
    p = pt.resolve_params("fixed_K_star", 100, cap=128, baseline_params=tuned)
    assert p["K"] == 32 and p["params_tuned"] is False and p["params_cap"] == 64
    assert pt.baseline_is_tuned("fixed_K_star", 100, tuned, cap=128) is False
    assert pt.resolve_params("fixed_K_star", 100, cap=64, baseline_params=tuned) == {"K": 32}


def test_fixed_k_star_builds_the_front_loaded_schedule():
    tuned = {"fixed_K_star": {"100": {"K": 32, "pooled_regret": 0.12}}}
    params = pt.resolve_params("fixed_K_star", 100, baseline_params=tuned)
    policy = pt.build_policy("fixed_K_star", params, horizon=100, n_replicates=8, table=None)
    assert isinstance(policy, PowerSchedule)
    assert policy.alpha == 0.0 and policy.c == 32.0


def test_migration_carries_the_block_under_by_cap():
    raw = {"meta": {"cap": 64}, "p3_star": {"100": {"alpha": 0.5, "c": 1.0}},
           "fixed_K_star": {"100": {"K": 32}}}
    migrated = pt.migrate_baseline_params(raw)
    assert migrated["by_cap"]["64"]["fixed_K_star"] == {"100": {"K": 32}}
    assert pt.unmigrate_baseline_params(migrated) == raw


# ---- the selector ----------------------------------------------------------------------


def test_candidate_grid_is_clipped_to_the_horizon_and_the_cap():
    assert sk.candidates(50) == (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48)
    assert sk.candidates(200)[-1] == 64 and len(sk.candidates(200)) == 17
    assert 3 <= min(sk.candidates(1000)) and max(sk.candidates(1000)) == 64


def test_selection_runs_on_the_validation_split_and_writes_the_block(tmp_path):
    out = tmp_path / "deploy"
    (out / "tables").mkdir(parents=True)
    json.dump({"meta": {"cap": 64, "select_split": "val"}, "p3_star": {}}, open(out / "baseline_params.json", "w"))
    frame = sk.select(
        env_ids=(ENV,), horizons=(50,), k_grid=(4, 12, 24), n_replicates=16, out_dir=out, workers=1,
    )
    assert set(frame["split"]) == {"val"} and set(frame["K"]) == {4, 12, 24}
    assert (frame["cap"] == 64).all()
    pooled = frame[frame["level"] == "pooled"]
    assert len(pooled) == 3 and pooled["is_argmin"].sum() == 1
    selected = json.load(open(out / "baseline_params.json"))
    block = selected["fixed_K_star"]["50"]
    assert block["K"] == int(pooled.loc[pooled["is_argmin"], "K"].item())
    assert block["pooled_regret"] == pytest.approx(pooled["regret"].min())
    assert selected["p3_star"] == {}, "the rest of the file is untouched"
    assert selected["meta"]["fixed_K_star"]["select_split"] == "val"
    assert selected["meta"]["fixed_K_star"]["k_grid"] == [4, 12, 24]
    assert (out / "tables" / "fixed_k_selection.csv").exists()
    assert pd.read_csv(out / "tables" / "fixed_k_selection.csv").shape[0] == len(frame)
