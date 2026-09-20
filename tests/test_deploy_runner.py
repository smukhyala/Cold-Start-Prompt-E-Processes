"""Tests for the deployment runner (`run_deployment.py`) and the policy table.

Structural tests walk `policy_table.POLICIES` against the trainer's `VARIANTS` and the
rules registry. Behavioural tests run the real runner end to end on `tmp_path`: the
baseline smoke path through a two-worker spawn pool (parquet + manifest + summary,
resume does no new work, a reference's paired difference against itself is exactly
zero) and the learned-policy path with a tiny artifact built here (search fraction
strictly inside (0, 1), snapshots logged, counters recorded). No simulator mocks.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
import warnings
from dataclasses import replace as rd_replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
for _p in (ROOT / "experiments" / "growing_bandits", DEPLOY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import policy_table as pt  # noqa: E402
import run_deployment as rd  # noqa: E402
import train_policies as tp  # noqa: E402

from cold_start.growing.deploy import feature_groups as fg  # noqa: E402
from cold_start.growing.deploy import rules  # noqa: E402
from cold_start.growing.deploy.artifacts import load_model, save_model  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.deploy.recommenders import RECOMMENDER_NAMES  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

CELLS_MOD = rd.import_cells_module()
HAVE_MODELS = all(
    pt.artifact_path(v).exists() for v in {pt.variant_of(p) for p in pt.POLICIES} - {None}
)

ENV_SPECS = {
    "beta_good_common": {"type": "beta", "params": {"a": 5.0, "b": 2.0}},
    "tail_b2.0_mu1.0_c1.0": {"type": "tail", "params": {"beta": 2.0, "mu_star": 1.0, "c": 1.0}},
}
#: Seeds for the explicit test cells: far above the corpus band and every split base.
TEST_SEED_BASE = 777_000_001

CLOCK = list(fg.FEATURE_SETS["clock"])


def _cells(horizon: int, n_replicates: int) -> list[CellSpec]:
    """An explicit two-cell list, independent of `cells.py`."""
    return [
        CellSpec(
            env_id=env_id,
            env_spec=spec,
            horizon=horizon,
            cap=64,
            base_seed=TEST_SEED_BASE + i,
            n_replicates=n_replicates,
        )
        for i, (env_id, spec) in enumerate(ENV_SPECS.items())
    ]


# ---- policy table -------------------------------------------------------------------------


def test_policy_table_is_well_formed_and_matches_the_trainer():
    names = list(pt.POLICIES)
    assert len(names) == len(set(names))
    for name, entry in pt.POLICIES.items():
        assert set(entry) == {"kind", "params", "group", "requires"}, name
        assert entry["kind"] in rules.KINDS, name
        assert entry["group"] in pt.GROUPS, name
        if entry["kind"] == "model":
            assert entry["params"]["artifact"] in tp.VARIANTS, (name, entry["params"]["artifact"])
            assert entry["group"] == "learned"
    for test, subset in pt.TEST_POLICIES.items():
        assert test in rd.TESTS
        assert set(subset) <= set(pt.POLICIES), test
        assert len(subset) == len(set(subset)), test
    assert set(pt.TEST_POLICIES) == set(rd.TESTS)
    # The plan's table, by the names the brief fixes.
    for required in (
        "always_search", "refine_after_init", "uniform", "fixed_K16",
        "power_a0.25", "power_a0.33", "power_a0.5", "power_a0.67", "p3_star", "cp0",
        "phi_k1", "phi_k4", "phi_k16", "phi_k16_tau05", "phi_k16_perstep", "phi_k16_guard",
        "phi_k16_quality", "phi_k16_clock", "phi_k16_cs", "phi_k16_all71", "phi_k16_hgb",
        "phi_k16_weighted", "phi_k16_noambig", "phi_k16_notrunc", "phi_k16_lucb",
        "rule_reservoir", "phi_reservoir_rule", "phi_k16_nopolicy", "phi_k16_famA_only",
        "phi_k16_famB_only", "phi_k16_noT1000", "phi_k16_noT200", "phi_sf_k16",
        "phi_sf_k16_noT1000",
    ):
        assert required in pt.POLICIES, required
    assert pt.POLICIES["phi_k16"]["params"]["artifact"] == "clock_quality_evidence_k16"
    assert pt.POLICIES["phi_k16_cs"]["params"]["artifact"] == "clock_quality_cs_k16"
    assert pt.POLICIES["phi_k16_perstep"]["params"]["per_step"] is True
    assert pt.POLICIES["phi_k16_guard"]["params"]["affordability_guard"] is True
    assert pt.POLICIES["phi_k16_tau05"]["params"]["tau"] == 0.5
    for ref in rd.REFERENCES:
        assert pt.POLICIES[ref]["group"] == "reference"
    assert set(pt.TEST_POLICIES["robust"]) == set(pt.TEST_POLICIES["cap"])


def test_policy_seed_is_keyed_on_the_study_name():
    assert pt.policy_seed("phi_k16") != pt.policy_seed("phi_k16_perstep")
    assert pt.policy_seed("uniform") == pt.policy_seed("uniform")


def test_resolve_params_uses_tuning_outputs_and_falls_back_to_placeholders(caplog):
    # Placeholders (no tuning outputs), logged once per policy and reason -- and stamped
    # into the params, so the manifest records that the constant was never tuned.
    pt._warned.clear()
    c_ph = rules.POLICY_SPECS["power_sqrt"]["c"]
    with caplog.at_level("WARNING", logger="deploy.policy_table"):
        p = pt.resolve_params("power_a0.5", 100)
        assert p == {"alpha": 0.5, "c": c_ph, "params_tuned": False, "params_fallback": {"c": c_ph}}
        assert pt.resolve_params("refine_after_init", 100) == {
            "K0": pt._PLACEHOLDER_K0, "params_tuned": False,
            "params_fallback": {"K0": pt._PLACEHOLDER_K0},
        }
        assert pt.resolve_params("p3_star", 100) == {
            **pt._PLACEHOLDER_P3_STAR, "params_tuned": False,
            "params_fallback": dict(pt._PLACEHOLDER_P3_STAR),
        }
        assert pt.resolve_params("rule_reservoir", 100) == {
            "tau": pt._PLACEHOLDER_RULE_TAU, "tau_source": "registered"
        }
        pt.resolve_params("power_a0.5", 200)
    assert sum("power_a0.5" in r.message for r in caplog.records) == 1

    # Tuned values, keyed exactly the way `tune_baselines.py` writes them. A tuned slot
    # carries NO `params_tuned` key: the unmarked case is "tuned", so a tuned item's
    # manifest record is byte-identical to the one earlier runs wrote.
    tuned = {
        "power": {str(float(1.0 / 3.0)): {"100": 2.25}, "0.5": {"100": 0.7, "200": 0.9}},
        "refine_after_init": {"100": 8},
        "p3_star": {"100": {"alpha": 2.0 / 3.0, "c": 0.4, "pooled_regret": 0.1}},
    }
    assert pt.resolve_params("power_a0.5", 100, baseline_params=tuned) == {"alpha": 0.5, "c": 0.7}
    assert pt.resolve_params("power_a0.5", 200, baseline_params=tuned) == {"alpha": 0.5, "c": 0.9}
    p = pt.resolve_params("power_a0.33", 100, baseline_params=tuned)
    assert p["c"] == 2.25 and abs(p["alpha"] - 1.0 / 3.0) < 1e-12
    assert pt.resolve_params("refine_after_init", 100, baseline_params=tuned) == {"K0": 8}
    assert pt.resolve_params("p3_star", 100, baseline_params=tuned) == {"alpha": 2.0 / 3.0, "c": 0.4}
    # A horizon the tuning did not cover falls back, per horizon, to the placeholder --
    # and says so, in the params and through `params_are_tuned` / `baseline_is_tuned`.
    p500 = pt.resolve_params("refine_after_init", 500, baseline_params=tuned)
    assert p500["K0"] == pt._PLACEHOLDER_K0 and p500["params_tuned"] is False
    assert pt.params_are_tuned(p500) is False
    assert pt.params_are_tuned(pt.resolve_params("refine_after_init", 100, baseline_params=tuned))
    for name, T, want in (("refine_after_init", 100, True), ("refine_after_init", 500, False),
                          ("p3_star", 100, True), ("p3_star", 500, False),
                          ("power_a0.5", 200, True), ("power_a0.5", 500, False),
                          ("always_search", 500, True), ("phi_k16", 500, True)):
        assert pt.baseline_is_tuned(name, T, tuned) is want, (name, T)
    assert pt.baseline_is_tuned("p3_star", 100, None) is False

    thresholds = {"reservoir_rule": {"tau_val": 2.5, "tau_off": None, "curve": {}}}
    assert pt.resolve_params("rule_reservoir", 100, thresholds=thresholds) == {
        "tau": 2.5, "tau_source": "tau_val"
    }


def test_untuned_provenance_never_reaches_the_policy_constructor():
    """`params_tuned` / `params_fallback` are manifest provenance, not constructor args."""
    table = CSTable.load_or_build(50, alpha=0.05)
    params = pt.resolve_params("p3_star", 100)  # no baseline_params -> placeholder
    assert params["params_tuned"] is False
    policy = pt.build_policy("p3_star", params, horizon=50, n_replicates=4, table=table)
    assert policy.name == "p3_star"
    assert policy.alpha == pt._PLACEHOLDER_P3_STAR["alpha"] and policy.c == pt._PLACEHOLDER_P3_STAR["c"]


def test_learned_tau_resolution_prefers_excl_heldout_for_horizon_holdouts(tmp_path, caplog):
    """A horizon-holdout model deploys `tau_val_excl_heldout` everywhere; others `tau_val`."""
    pipeline = _fit_clock_pipeline(n_rows=300)
    base = {"pipeline": pipeline, "features": CLOCK, "k": 16, "tau": 0.45}
    save_model(tmp_path / "clock_quality_evidence_k16.joblib", {**base, "meta": {"subset": {}}})
    save_model(
        tmp_path / "clock_quality_evidence_k16_noT1000.joblib",
        {**base, "meta": {"subset": {"exclude_horizons": [1000]}}},
    )
    thresholds = {
        "clock_quality_evidence_k16": {"tau_val": 0.6, "tau_val_excl_heldout": 0.1},
        "clock_quality_evidence_k16_noT1000": {"tau_val": 0.6, "tau_val_excl_heldout": 0.4},
    }
    kw = dict(models_dir=tmp_path, thresholds=thresholds)
    # Whatever the horizon (T=1000 included), the holdout model uses the excl-heldout tau.
    for T in (200, 1000):
        p = pt.resolve_params("phi_k16_noT1000", T, **kw)
        assert (p["tau"], p["tau_source"]) == (0.4, "tau_val_excl_heldout")
    # A non-holdout model never reads that key, even when M5 happens to write one.
    p = pt.resolve_params("phi_k16", 1000, **kw)
    assert (p["tau"], p["tau_source"]) == (0.6, "tau_val")
    # Key absent or None -> tau_val with a logged warning; no thresholds -> artifact tau.
    pt._warned.clear()
    with caplog.at_level("WARNING", logger="deploy.policy_table"):
        thresholds["clock_quality_evidence_k16_noT1000"]["tau_val_excl_heldout"] = None
        p = pt.resolve_params("phi_k16_noT1000", 200, **kw)
        assert (p["tau"], p["tau_source"]) == (0.6, "tau_val")
        p = pt.resolve_params("phi_k16_noT1000", 200, models_dir=tmp_path, thresholds=None)
        assert (p["tau"], p["tau_source"]) == (0.45, "artifact_tau")
    assert any("tau_val_excl_heldout" in r.message for r in caplog.records)
    # The fixed twin records its source too, and the source never reaches the constructor.
    p = pt.resolve_params("phi_k16_tau05", 200, **kw)
    assert (p["tau"], p["tau_source"]) == (0.5, "fixed")
    table = CSTable.load_or_build(50, alpha=0.05)
    policy = pt.build_policy("phi_k16_tau05", p, horizon=50, n_replicates=4, table=table)
    assert policy.tau == 0.5 and policy.name == "phi_k16_tau05"
    assert pt.heldout_horizons(load_model(tmp_path / "clock_quality_evidence_k16_noT1000.joblib")) == (1000,)


def test_build_cells_refuses_a_real_test_without_cells_module():
    with pytest.raises(RuntimeError, match="cells.py"):
        rd.build_cells("A", n_replicates=None, cells_mod=None)


@pytest.mark.skipif(CELLS_MOD is None, reason="cells.py (M5) not present")
def test_cell_grids_against_the_real_cells_module():
    expected = {"A": 40, "B": 40, "C": 15, "D": 19, "robust": 150, "cap": 24, "smoke": 2}
    for test, n_cells in expected.items():
        cells = rd.build_cells(test, n_replicates=None, cells_mod=CELLS_MOD)
        assert len(cells) == n_cells, test
        seeds = [c.base_seed for c in cells]
        assert len(set(seeds)) == len(seeds), test
        CELLS_MOD.assert_seed_disjointness(seeds)
        m_default = rd.DEFAULT_REPLICATES[test]
        for c in cells:
            assert c.n_replicates == (rd.T2000_REPLICATES if c.horizon == 2000 else m_default)
    # Smoke draws validation seeds; every real test draws test seeds.
    for c in rd.build_cells("smoke", n_replicates=None, cells_mod=CELLS_MOD):
        assert c.base_seed == CELLS_MOD.base_seed("val", c.env_id, c.horizon, c.cap)
    for c in rd.build_cells("A", n_replicates=None, cells_mod=CELLS_MOD):
        assert c.base_seed == CELLS_MOD.base_seed("test", c.env_id, c.horizon, c.cap)
    caps = {(c.horizon, c.cap) for c in rd.build_cells("cap", n_replicates=None, cells_mod=CELLS_MOD)}
    assert caps == {(200, 32), (200, 64), (200, 200), (1000, 32), (1000, 64), (1000, 1000)}
    # A manual cell list and a replicate override are honoured.
    cells = rd.build_cells(
        "A", n_replicates=7, cells_mod=CELLS_MOD,
        overrides=rd.parse_cell_overrides("beta_good_common:100:64"),
    )
    assert [(c.env_id, c.horizon, c.cap, c.n_replicates) for c in cells] == [
        ("beta_good_common", 100, 64, 7)
    ]


def test_build_work_logs_only_learned_and_rule_policies(tmp_path):
    cells = _cells(horizon=20, n_replicates=4)
    common = dict(
        out_dir=tmp_path, baseline_params=None, thresholds=None, models_dir=tmp_path,
        dynamics_grid=10, done={}, git_sha="test",
    )
    items = rd.build_work("smoke", cells, ["rule_reservoir", "cp0"], log_states=True, **common)
    by_policy = {(i.cell, i.policy): i for i in items}
    assert len(items) == 4
    for cell in cells:
        rule = by_policy[(rd.cell_name(cell), "rule_reservoir")]
        assert rule.log_states is not None and rule.snapshots_path is not None
        ref = by_policy[(rd.cell_name(cell), "cp0")]
        assert ref.log_states is None and ref.snapshots_path is None
        assert Path(rule.prefix_path).exists()
        assert np.load(rule.prefix_path).shape == (4, 20)
        assert rule.oracle_prior == ref.oracle_prior
    # `--log-policies` narrows logging; a name outside the loggable groups logs nothing.
    items = rd.build_work(
        "smoke", cells, ["rule_reservoir", "cp0"], log_states=True, log_policies={"cp0"}, **common
    )
    assert all(i.snapshots_path is None for i in items)
    # Skipping honours `done` only while the record still describes the item.
    items = rd.build_work("smoke", cells, ["rule_reservoir", "cp0"], log_states=False, **common)
    done = {}
    for item in items:
        if item.cell == rd.cell_name(cells[0]):
            Path(item.parquet_path).parent.mkdir(parents=True, exist_ok=True)
            Path(item.parquet_path).touch()
            done[(item.test, item.cell, item.policy)] = {
                "parquet": item.parquet_path, "params": item.params, "artifact_sha": None,
                "n_replicates": item.spec.n_replicates, "base_seed": item.spec.base_seed,
                "horizon": item.spec.horizon, "cap": item.spec.cap, "snapshots": None,
            }
    rest = {k: v for k, v in common.items() if k != "done"}
    items = rd.build_work("smoke", cells, ["rule_reservoir", "cp0"], log_states=False, done=done,
                          **rest)
    assert {i.cell for i in items} == {rd.cell_name(cells[1])}
    # ... a changed tau, replicate count, artifact or newly requested snapshots re-run it.
    key = ("smoke", rd.cell_name(cells[0]), "rule_reservoir")
    stale = {**done[key], "params": {**done[key]["params"], "tau": 9.0}}
    items = rd.build_work("smoke", cells, ["rule_reservoir"], log_states=False,
                          done={key: stale}, **rest)
    assert [(i.cell, i.policy) for i in items] == [(c, "rule_reservoir") for c in
                                                   (rd.cell_name(cells[0]), rd.cell_name(cells[1]))]
    item = items[0]
    assert rd.stale_reason(done[key], item) is None
    assert "params" in rd.stale_reason(stale, item)
    assert "n_replicates" in rd.stale_reason({**done[key], "n_replicates": 3}, item)
    assert "base_seed" in rd.stale_reason({**done[key], "base_seed": 1}, item)
    assert "artifact" in rd.stale_reason({**done[key], "artifact_sha": "old"},
                                         rd_replace(item, artifact_sha="new"))
    assert "snapshots" in rd.stale_reason(done[key], rd_replace(item, snapshots_path="x.pkl"))
    Path(item.parquet_path).unlink()
    assert "parquet" in rd.stale_reason(done[key], item)


@pytest.mark.parametrize("horizon", [50, 100, 1000])
def test_log_spec_times_are_visited_states(horizon):
    spec = CellSpec("e", ENV_SPECS["beta_good_common"], horizon, 64, 1, 200)
    ls = rd.log_spec_for(spec)
    assert ls.replicates == min(rd.LOG_REPLICATES, 200)
    assert list(ls.times) == sorted(set(ls.times))
    assert 1 <= len(ls.times) <= rd.LOG_TIMES
    assert ls.times[-1] == horizon
    assert all(spec.n_initial_arms <= t <= horizon for t in ls.times)


@pytest.mark.skipif(not HAVE_MODELS, reason="M4 model artifacts not present")
def test_every_learned_policy_builds_against_the_real_artifacts():
    table = CSTable.load_or_build(50, alpha=0.05)
    for name in pt.POLICIES:
        policy, seed = pt.resolve_policy(name, 50, n_replicates=4, table=table)
        assert policy.name == name
        assert seed == pt.policy_seed(name)
        if pt.POLICIES[name]["kind"] == "model":
            uses_log_e = pt.needs_pairwise_table(name)
            assert uses_log_e == any(c in fg.EVIDENCE_LOGE for c in policy.features)
            assert policy.counters()["n_nonfinite_rows"] == 0
    assert pt.needs_pairwise_table("phi_k16") is True
    assert pt.needs_pairwise_table("phi_k16_cs") is False
    assert pt.needs_pairwise_table("cp0") is False


# ---- runner: baselines through the spawn pool ----------------------------------------------


def _read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_smoke_baselines_resume_and_summary(tmp_path):
    out = tmp_path / "out"
    policies = ["always_search", "cp0", "power_a0.5"]
    cells = _cells(horizon=60, n_replicates=16)
    argv = [
        "--test", "smoke", "--n-replicates", "16", "--workers", "2",
        "--policies", ",".join(policies), "--out-dir", str(out), "--n-boot", "200",
        "--models-dir", str(tmp_path / "no-models"),
    ]
    result = rd.main(argv, cells=cells)
    assert result["n_items"] == 6 and result["n_run"] == 6
    assert result["n_skipped"] == 0 and result["n_failed"] == 0

    manifest = out / "manifest_smoke.jsonl"
    records = _read_manifest(manifest)
    assert len(records) == 6
    assert {(r["cell"], r["policy"]) for r in records} == {
        (rd.cell_name(c), p) for c in cells for p in policies
    }
    for rec in records:
        assert rec["n"] == 16 and rec["seconds"] > 0 and rec["sha"]
        assert Path(rec["parquet"]).exists() and Path(rec["dynamics"]).exists()
        assert rec["snapshots"] is None  # no --log-states, and baselines never log
        assert rec["counters"] is None  # baselines expose no counters

    # Parquet schema: the harness frame plus the runner's columns, one row per episode.
    frame = pd.read_parquet(out / "episodes" / "smoke" / rd.cell_name(cells[0]) / "cp0.parquet")
    assert len(frame) == 16
    for col in ("test", "cell", "family", "group", "policy", "env_id", "horizon", "cap",
                "base_seed", "replicate", "episode", "mu_star", "regret_disc", "search_frac"):
        assert col in frame.columns, col
    assert frame["family"].iloc[0] == "A" and frame["group"].iloc[0] == "reference"
    assert np.array_equal(frame["replicate"], frame["episode"])
    assert np.all(frame["regret_disc"] >= 0.0)
    for rec in RECOMMENDER_NAMES:
        assert f"regret_{rec}" in frame.columns
    dyn = np.load(out / "dynamics" / "smoke" / rd.cell_name(cells[0]) / "cp0.npz")
    assert dyn["t"].shape == (51,) and dyn["t"][-1] == 60

    # Summary: one row per (cell, policy, recommender); a reference vs itself is 0.
    summary = pd.read_csv(out / "summary_smoke.csv")
    assert len(summary) == 2 * 3 * len(RECOMMENDER_NAMES)
    own = summary[summary["policy"] == "cp0"]
    for col in ("d_regret_vs_cp0", "d_regret_vs_cp0_lo", "d_regret_vs_cp0_hi"):
        assert np.all(own[col] == 0.0), col
    assert np.all(own["d_regret_vs_cp0_win"] == 0.5)
    assert summary["d_regret_vs_p3_star"].isna().all()  # p3_star was not run
    a = summary[(summary["policy"] == "always_search") & (summary["recommender"] == "lcb")]
    assert np.all(a["k_final"] == 60.0) and np.all(a["search_frac"] == 1.0)  # T=60 < cap
    # Paired difference equals the difference of means, and the CI brackets it.
    row = a.iloc[0]
    ref = summary[(summary["policy"] == "cp0") & (summary["recommender"] == "lcb")
                  & (summary["cell"] == row["cell"])].iloc[0]
    assert abs(row["d_regret_vs_cp0"] - (row["regret"] - ref["regret"])) < 1e-12
    assert row["d_regret_vs_cp0_lo"] <= row["d_regret_vs_cp0"] <= row["d_regret_vs_cp0_hi"]

    # Resume: nothing left to do, the manifest is unchanged, the summary is rewritten.
    mtimes = {p: p.stat().st_mtime_ns for p in (out / "episodes").rglob("*.parquet")}
    again = rd.main(argv + ["--resume"], cells=cells)
    assert again["n_run"] == 0 and again["n_skipped"] == 6 and again["n_failed"] == 0
    assert len(_read_manifest(manifest)) == 6
    assert {p: p.stat().st_mtime_ns for p in (out / "episodes").rglob("*.parquet")} == mtimes
    assert (out / "summary_smoke.csv").exists()

    # Resume after one item's parquet vanished re-runs exactly that item.
    victim = out / "episodes" / "smoke" / rd.cell_name(cells[1]) / "power_a0.5.parquet"
    victim.unlink()
    third = rd.main(argv + ["--resume"], cells=cells)
    assert third["n_run"] == 1 and third["n_skipped"] == 5
    assert victim.exists()
    assert len(_read_manifest(manifest)) == 7  # appended, not rewritten

    # A later run of a policy subset without --resume redoes just that subset, appends
    # to the manifest (latest line per item wins) and the summary still covers all.
    subset = [a if a != ",".join(policies) else "cp0" for a in argv]
    fourth = rd.main(subset, cells=cells)
    assert fourth["n_run"] == 2 and fourth["n_skipped"] == 0
    records = _read_manifest(manifest)
    assert len(records) == 9
    assert len(rd.latest_records(records)) == 6
    assert len(pd.read_csv(out / "summary_smoke.csv")) == 2 * 3 * len(RECOMMENDER_NAMES)


def _planted_frame(policy: str, regret: list[float], cell: str = "c") -> pd.DataFrame:
    """A minimal episode frame with one recommender, as `summarize_cell` reads it."""
    m = len(regret)
    r = np.asarray(regret, dtype=np.float64)
    frame = {
        "test": "t", "cell": cell, "env_id": "e", "family": "A", "horizon": 10, "cap": 64,
        "base_seed": 1, "group": "x", "policy": policy, "episode": np.arange(m),
        "regret_disc": np.zeros(m), "k_final": np.full(m, 2.0), "search_frac": np.zeros(m),
        "cap_hit": np.zeros(m, dtype=bool), "t_cap_hit": np.full(m, -1), "n_eliminated_final": np.zeros(m),
        "herfindahl": np.ones(m), "n_singletons_final": np.zeros(m), "mu_star": np.ones(m),
        "mu_star_cap": np.ones(m), "best_discovered": np.ones(m), "n_demoted": np.zeros(m),
        "q_lcb": 1.0 - r, "n_rec_lcb": np.ones(m), "regret_lcb": r, "regret_sel_lcb": r,
        "regret_sup_lcb": r,
    }
    return pd.DataFrame(frame)


def test_summary_win_rate_is_the_share_of_episodes_the_policy_wins():
    """Planted: the policy has lower regret in 3 of 4 episodes -> win 0.75 vs cp0."""
    frames = {
        "cp0": _planted_frame("cp0", [0.5, 0.5, 0.5, 0.5]),
        "better": _planted_frame("better", [0.1, 0.2, 0.3, 0.9]),
        "tied": _planted_frame("tied", [0.1, 0.5, 0.5, 0.5]),
    }
    rows = {r["policy"]: r for r in rd.summarize_cell(frames, n_boot=50)}
    assert rows["better"]["d_regret_vs_cp0_win"] == 0.75
    assert rows["better"]["d_regret_vs_cp0"] == pytest.approx(np.mean([-0.4, -0.3, -0.2, 0.4]))
    assert rows["cp0"]["d_regret_vs_cp0_win"] == 0.5 and rows["cp0"]["d_regret_vs_cp0"] == 0.0
    # One strict win and three ties: 0.25 + 0.5 * 0.75.
    assert rows["tied"]["d_regret_vs_cp0_win"] == pytest.approx(0.625)
    assert np.isnan(rows["better"]["d_regret_vs_p3_star_win"])


# ---- runner: learned policy with a tiny artifact ---------------------------------------------


def _clock_row(t: int, horizon: int, k: int) -> list[float]:
    """The 11 CLOCK columns exactly as `features.extract_features` defines them."""
    remaining = horizon - t
    return [
        float(t), float(horizon), float(remaining), remaining / horizon, t / horizon, float(k),
        k / t if t else float(k), k / horizon, float(np.log(max(t, 1))), float(np.log(k)),
        k / np.sqrt(max(t, 1)),
    ]


def _fit_clock_pipeline(seed: int = 0, n_rows: int = 3000):
    """A logistic rule that wants to SEARCH early and stops as arms accumulate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_rows):
        horizon = int(rng.choice([40, 100, 200]))
        t = int(rng.integers(2, horizon))
        k = int(rng.integers(1, min(t, 64) + 1))
        row = _clock_row(t, horizon, k)
        score = row[3] - 0.15 * row[10] + 0.3 * rng.normal()
        rows.append(row)
        labels.append(score > 0.35)
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def test_learned_policy_smoke_logs_states_and_counters(tmp_path):
    models = tmp_path / "models"
    # `phi_k16_clock` deploys the `clock_k16` variant; give it a tiny CLOCK model with
    # k=4 so the commitment mechanism is exercised several times in T=60.
    save_model(
        models / "clock_k16.joblib",
        {"pipeline": _fit_clock_pipeline(), "features": CLOCK, "k": 4, "tau": 0.45,
         "meta": {"variant": "clock_k16"}},
    )
    artifact = load_model(models / "clock_k16.joblib")
    out = tmp_path / "out"
    cells = _cells(horizon=60, n_replicates=16)
    argv = [
        "--test", "smoke", "--n-replicates", "16", "--workers", "1", "--policies",
        "phi_k16_clock,cp0", "--models-dir", str(models), "--out-dir", str(out),
        "--log-states", "--n-boot", "100",
    ]
    result = rd.main(argv, cells=cells)
    assert result["n_run"] == 4 and result["n_failed"] == 0

    records = {(r["cell"], r["policy"]): r for r in _read_manifest(out / "manifest_smoke.jsonl")}
    for cell in cells:
        rec = records[(rd.cell_name(cell), "phi_k16_clock")]
        # No thresholds.json: the artifact's own tau is deployed, and recorded as such.
        assert rec["params"]["tau"] == artifact["tau"] == 0.45
        assert rec["params"]["tau_source"] == "artifact_tau"
        assert rec["params"]["artifact"] == str(models / "clock_k16.joblib")
        assert rec["artifact_sha"] == rd.file_sha256(models / "clock_k16.joblib")
        assert rec["counters"]["n_nonfinite_rows"] == 0
        assert rec["counters"]["n_decisions"] > 0 and rec["counters"]["n_committed_steps"] > 0
        assert rec["policy_seed"] == pt.policy_seed("phi_k16_clock")
        frame = pd.read_parquet(rec["parquet"])
        assert frame["group"].iloc[0] == "learned"
        assert 0.0 < float(frame["search_frac"].mean()) < 1.0
        assert np.all(frame["policy_seed"] == pt.policy_seed("phi_k16_clock"))
        # Snapshots: every logged time x every logged replicate, carrying the study name.
        ls = rd.log_spec_for(cell)
        with open(rec["snapshots"], "rb") as fh:
            snaps = pickle.load(fh)
        assert rec["n_snapshots"] == len(snaps) == len(ls.times) * ls.replicates
        assert {s.t for s in snaps} == set(ls.times)
        assert all(s.meta["policy"] == "phi_k16_clock" for s in snaps)
        assert all(s.meta["env_id"] == cell.env_id for s in snaps)
        # References never log states, even under --log-states.
        assert records[(rd.cell_name(cell), "cp0")]["snapshots"] is None

    summary = pd.read_csv(out / "summary_smoke.csv")
    learned = summary[(summary["policy"] == "phi_k16_clock")]
    assert len(learned) == 2 * len(RECOMMENDER_NAMES)
    assert learned["d_regret_vs_cp0"].notna().all()
    assert learned["d_regret_vs_p3_star"].isna().all()

    # thresholds.json lands with a different tau: --resume must redo the learned items
    # (and only those), and the manifest must carry the new tau and its source.
    (out / "thresholds.json").write_text(json.dumps(
        {"clock_k16": {"tau_val": 0.6, "tau_off": 0.45, "curve": {"0.6": 0.1}}}
    ))
    resumed = rd.main(argv + ["--resume"], cells=cells)
    assert resumed["n_run"] == 2 and resumed["n_skipped"] == 2 and resumed["n_failed"] == 0
    latest = rd.latest_records(_read_manifest(out / "manifest_smoke.jsonl"))
    for cell in cells:
        rec = latest[("smoke", rd.cell_name(cell), "phi_k16_clock")]
        assert (rec["params"]["tau"], rec["params"]["tau_source"]) == (0.6, "tau_val")
        assert pd.read_parquet(rec["parquet"])["search_frac"].mean() < frame["search_frac"].mean()
    # Nothing changed since: a further --resume does no work at all.
    again = rd.main(argv + ["--resume"], cells=cells)
    assert again["n_run"] == 0 and again["n_skipped"] == 4
    # A changed replicate count is a changed item for every policy.
    more = rd.main([a if a != "16" else "8" for a in argv] + ["--resume"], cells=cells)
    assert more["n_run"] == 4 and more["n_skipped"] == 0
    # A retrained artifact (new bytes) is a changed item for the learned policy only.
    save_model(
        models / "clock_k16.joblib",
        {"pipeline": _fit_clock_pipeline(seed=1), "features": CLOCK, "k": 4, "tau": 0.45,
         "meta": {"variant": "clock_k16"}},
    )
    retrained = rd.main([a if a != "16" else "8" for a in argv] + ["--resume"], cells=cells)
    assert retrained["n_run"] == 2 and retrained["n_skipped"] == 2


# ---- the orphan-parquet class (ledger rulings 17, 19, 22) ------------------------------------


def test_exclusions_are_durable_skips_that_only_force_policies_overrides(tmp_path):
    """Ruling 19's durable fix, end to end through the CLI.

    Ruling 16 dropped three policies from Test D's T=2000 cells on purpose; nothing on
    disk recorded that, so the next ``--resume`` scheduled all nine again and the
    workers that outlived the killed parent left nine orphan parquets (ruling 22). An
    exclusion line survives a resume, survives a plain re-run, and yields only to
    ``--force-policies``.
    """
    out = tmp_path / "out"
    cells = _cells(horizon=40, n_replicates=8)
    manifest = out / "manifest_smoke.jsonl"
    argv = [
        "--test", "smoke", "--n-replicates", "8", "--workers", "1",
        "--policies", "cp0,always_search", "--out-dir", str(out), "--skip-summary",
        "--models-dir", str(tmp_path / "no-models"),
    ]

    # A reason is mandatory in both directions: neither half is usable alone.
    with pytest.raises(SystemExit, match="exclusion-reason"):
        rd.main(argv + ["--exclude-policies", "always_search"], cells=cells)
    with pytest.raises(SystemExit, match="exclusion-reason"):
        rd.main(argv + ["--exclude-policies", "always_search", "--exclusion-reason", "  "],
                cells=cells)
    with pytest.raises(SystemExit, match="without --exclude-policies"):
        rd.main(argv + ["--exclusion-reason", "why"], cells=cells)
    assert not manifest.exists()

    reason = "costs hours per item through the exact chunked evaluator"
    first = rd.main(argv + ["--exclude-policies", "always_search",
                            "--exclusion-reason", reason], cells=cells)
    assert first["n_run"] == 2 and first["n_skipped"] == 2  # cp0 on both cells only
    records = _read_manifest(manifest)
    exclusions = [r for r in records if rd.record_kind(r) == rd.KIND_EXCLUSION]
    assert len(exclusions) == 2
    assert {r["cell"] for r in exclusions} == {rd.cell_name(c) for c in cells}
    for rec in exclusions:
        assert rec["policy"] == "always_search" and rec["reason"] == reason
        assert rec["sha"] and rec["at"]
    assert not list((out / "episodes" / "smoke").rglob("always_search.parquet"))

    # The exclusion is active for every later run, with or without --resume, and
    # `completed_records` counts it as done-with-no-file.
    active = rd.active_exclusions(_read_manifest(manifest))
    assert set(active) == {("smoke", rd.cell_name(c), "always_search") for c in cells}
    assert set(rd.completed_records(_read_manifest(manifest))) == {
        ("smoke", rd.cell_name(c), p) for c in cells for p in ("cp0", "always_search")
    }
    for extra in ([], ["--resume"]):
        again = rd.main(argv + extra, cells=cells)
        assert again["n_skipped"] >= 2
        assert not list((out / "episodes" / "smoke").rglob("always_search.parquet"))

    # --force-policies overrides it, and the completion supersedes the exclusion.
    forced = rd.main(argv + ["--resume", "--force-policies", "always_search"], cells=cells)
    assert forced["n_run"] == 2
    assert len(list((out / "episodes" / "smoke").rglob("always_search.parquet"))) == 2
    assert rd.active_exclusions(_read_manifest(manifest)) == {}
    # ... and the two cannot be asked for at once.
    with pytest.raises(SystemExit, match="both name"):
        rd.main(argv + ["--exclude-policies", "always_search", "--exclusion-reason", reason,
                        "--force-policies", "always_search"], cells=cells)


def _fork_quietly() -> int:
    """`os.fork` without CPython's multi-threaded-process DeprecationWarning.

    pytest is multi-threaded, so every fork from it warns; the fork is the point of the
    test below and the project keeps test output free of warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return os.fork()


def test_an_orphaned_worker_writes_nothing_into_episodes(tmp_path):
    """Close the write side: a worker whose pool parent is gone publishes no episode.

    Forks a child, forks a grandchild from it, kills the child, and lets the reparented
    grandchild try to publish a parquet exactly as `run_item` would. Nothing may land in
    ``episodes/`` -- the file that lands there is the one rulings 17 and 22 had to
    quarantine by hand.
    """
    final = tmp_path / "episodes" / "smoke" / "cell" / "phi_k16.parquet"
    final.parent.mkdir(parents=True)
    tmp = final.with_suffix(".parquet.tmp")
    tmp.write_bytes(b"episode bytes")
    marker = tmp_path / "outcome.txt"

    child = _fork_quietly()
    if child == 0:  # the "pool parent"
        parent_pid = os.getpid()
        if _fork_quietly() == 0:  # the worker
            try:
                rd._worker_init(logging.WARNING, parent_pid)
                deadline = time.time() + 10.0
                while os.getppid() == parent_pid and time.time() < deadline:
                    time.sleep(0.01)
                try:
                    rd._atomic_replace(tmp, final)
                    marker.write_text("WROTE")
                except rd.OrphanedWorker as exc:
                    marker.write_text(f"REFUSED {exc}")
            except BaseException as exc:  # noqa: BLE001 - reported through the marker
                marker.write_text(f"ERROR {type(exc).__name__}: {exc}")
            finally:
                os._exit(0)
        os._exit(0)  # the parent dies while the worker is still running
    os.waitpid(child, 0)

    deadline = time.time() + 15.0
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert marker.exists(), "the forked worker never reported"
    outcome = marker.read_text()
    assert outcome.startswith("REFUSED"), outcome
    assert not final.exists(), "an orphaned worker published an episode file"
    assert not tmp.exists(), "the orphaned worker left its temporary file behind"
    # The same call from a live parent still publishes normally.
    rd._PARENT_PID = None
    tmp.write_bytes(b"episode bytes")
    rd._atomic_replace(tmp, final)
    assert final.exists()


def test_worker_init_records_the_pool_parent_for_every_pooled_run(tmp_path):
    """The guard is only armed if `run_items` hands the pool its own pid."""
    import inspect

    source = inspect.getsource(rd.run_items)
    assert "initargs=(logging.getLogger().level or logging.INFO, os.getpid())" in source
    rd._worker_init(logging.WARNING, 4242)
    assert rd._PARENT_PID == 4242
    assert rd._parent_is_alive() is (os.getppid() == 4242)
    rd._worker_init(logging.WARNING, None)
    assert rd._PARENT_PID is None and rd._parent_is_alive() is True
