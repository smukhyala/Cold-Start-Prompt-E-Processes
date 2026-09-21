"""`adaptive_K_star` (DEPLOYMENT_PLAN.md, Pre-registration 4): the tail-adaptive schedule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import policy_table as pt  # noqa: E402
import select_adaptive_k as sa  # noqa: E402

from cold_start.growing.search_policies import TailAdaptiveSchedule  # noqa: E402

ENV = "beta_good_common"
TUNED = {"meta": {"cap": 64}, "adaptive_K_star": {"alpha": 0.75, "c": 2.0, "b": 4.0, "pooled_regret": 0.12}}


def test_adaptive_k_star_resolves_one_triple_for_every_horizon():
    for T in (50, 100, 200, 1000):
        assert pt.resolve_params("adaptive_K_star", T, baseline_params=TUNED) == {"alpha": 0.75, "c": 2.0, "b": 4.0}
        assert pt.baseline_is_tuned("adaptive_K_star", T, TUNED) is True
    pt._warned.clear()
    p = pt.resolve_params("adaptive_K_star", 100, baseline_params=None)
    assert p["params_tuned"] is False and p["b"] == pt._PLACEHOLDER_ADAPTIVE_K["b"]
    assert pt.baseline_is_tuned("adaptive_K_star", 100, None) is False


def test_adaptive_k_star_is_cap_stamped_and_builds_the_rule():
    pt._warned.clear()
    p = pt.resolve_params("adaptive_K_star", 100, cap=128, baseline_params=TUNED)
    assert p["params_tuned"] is False and p["params_cap"] == 64
    params = pt.resolve_params("adaptive_K_star", 100, cap=64, baseline_params=TUNED)
    policy = pt.build_policy("adaptive_K_star", params, horizon=100, n_replicates=8, table=None)
    assert isinstance(policy, TailAdaptiveSchedule)
    assert (policy.alpha, policy.c, policy.b) == (0.75, 2.0, 4.0)


def test_migration_carries_the_block():
    raw = {"meta": {"cap": 64}, "p3_star": {}, "adaptive_K_star": {"alpha": 0.5, "c": 1.0, "b": 0.0}}
    migrated = pt.migrate_baseline_params(raw)
    assert migrated["by_cap"]["64"]["adaptive_K_star"] == raw["adaptive_K_star"]
    assert pt.unmigrate_baseline_params(migrated) == raw


def test_grid_is_the_registered_fifty():
    grid = sa.grid()
    assert len(grid) == 50
    assert {a for a, _, _ in grid} == {0.5, 0.75}
    assert {c for _, c, _ in grid} == {1, 2, 3, 4, 6}
    assert {b for _, _, b in grid} == {0, 1, 2, 4, 8}


def test_selection_picks_one_triple_on_validation_and_writes_the_block(tmp_path):
    out = tmp_path / "deploy"
    (out / "tables").mkdir(parents=True)
    json.dump({"meta": {"cap": 64}, "p3_star": {}}, open(out / "baseline_params.json", "w"))
    frame = sa.select(env_ids=(ENV,), horizons=(50,), candidates=((0.5, 1.0, 0.0), (0.5, 2.0, 2.0)),
                      n_replicates=16, out_dir=out, workers=1)
    assert set(frame["split"]) == {"val"} and len(frame[frame["level"] == "pooled"]) == 2
    assert frame[frame["level"] == "pooled"]["is_argmin"].sum() == 1
    sel = json.load(open(out / "baseline_params.json"))
    block = sel["adaptive_K_star"]
    assert set(block) == {"alpha", "c", "b", "pooled_regret"}
    assert (block["alpha"], block["c"], block["b"]) in {(0.5, 1.0, 0.0), (0.5, 2.0, 2.0)}
    assert sel["meta"]["adaptive_K_star"]["select_split"] == "val"
    assert list(sel) == ["meta", "p3_star", "adaptive_K_star"], "appended, existing order kept"
    assert (out / "tables" / "adaptive_k_selection.csv").exists()
