"""`bestmean_star` (DEPLOYMENT_PLAN.md, Pre-registration 5) and the generic rule selector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import policy_table as pt  # noqa: E402
import select_rule as sr  # noqa: E402

from cold_start.growing.search_policies import BestMeanGate  # noqa: E402

ENV = "beta_good_common"
TUNED = {"meta": {"cap": 64}, "bestmean_star": {"theta": 0.65, "alpha": 0.5, "c": 4.0, "pooled_regret": 0.12}}


def test_bestmean_star_resolves_builds_and_is_cap_stamped():
    for T in (50, 200):
        assert pt.resolve_params("bestmean_star", T, baseline_params=TUNED) == {"theta": 0.65, "alpha": 0.5, "c": 4.0}
    assert pt.baseline_is_tuned("bestmean_star", 100, TUNED) is True
    pt._warned.clear()
    p = pt.resolve_params("bestmean_star", 100, baseline_params=None)
    assert p["params_tuned"] is False and pt.baseline_is_tuned("bestmean_star", 100, None) is False
    p = pt.resolve_params("bestmean_star", 100, cap=128, baseline_params=TUNED)
    assert p["params_tuned"] is False and p["params_cap"] == 64
    policy = pt.build_policy("bestmean_star", pt.resolve_params("bestmean_star", 100, cap=64, baseline_params=TUNED),
                             horizon=100, n_replicates=8, table=None)
    assert isinstance(policy, BestMeanGate) and (policy.theta, policy.alpha, policy.c) == (0.65, 0.5, 4.0)


def test_rule_registry_lists_both_registered_rules_with_their_grids():
    assert set(sr.RULES) >= {"adaptive_K_star", "bestmean_star"}
    assert len(sr.RULES["adaptive_K_star"].candidates()) == 50
    bm = sr.RULES["bestmean_star"]
    assert len(bm.candidates()) == 48
    assert bm.kind == "bestmean_K" and set(bm.param_names) == {"theta", "alpha", "c"}


def test_selection_writes_the_named_block_and_table(tmp_path):
    out = tmp_path / "deploy"
    (out / "tables").mkdir(parents=True)
    json.dump({"meta": {"cap": 64}, "p3_star": {}}, open(out / "baseline_params.json", "w"))
    frame = sr.select("bestmean_star", env_ids=(ENV,), horizons=(50,),
                      candidates=({"theta": 0.6, "alpha": 0.5, "c": 3.0}, {"theta": 1.0, "alpha": 0.5, "c": 3.0}),
                      n_replicates=16, out_dir=out, workers=1)
    pooled = frame[frame["level"] == "pooled"]
    assert len(pooled) == 2 and pooled["is_argmin"].sum() == 1
    sel = json.load(open(out / "baseline_params.json"))
    assert set(sel["bestmean_star"]) == {"theta", "alpha", "c", "pooled_regret"}
    assert sel["meta"]["bestmean_star"]["n_candidates"] == 2
    assert list(sel) == ["meta", "p3_star", "bestmean_star"]
    assert (out / "tables" / "bestmean_selection.csv").exists()
    with pytest.raises(KeyError):
        sr.select("no_such_rule", env_ids=(ENV,), horizons=(50,), n_replicates=4, out_dir=out)
