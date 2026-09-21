"""Per-cap constants (roadmap 3.3): stored under `by_cap`, read at the deployed cap, mismatches stamped."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import policy_table as pt  # noqa: E402

FLAT = {
    "meta": {"cap": 64, "tune_split": "tune"},
    "p3_star": {"200": {"alpha": 0.5, "c": 3.0}},
    "power": {"0.5": {"200": 3.0}},
    "refine_after_init": {"200": 8},
    "fixed_K_star": {"200": {"K": 64}},
}


def test_merge_cap_block_keeps_the_tuning_cap_flat_and_others_under_by_cap():
    out = pt.merge_cap_block(FLAT, 64, {"fixed_K_star": {"200": {"K": 48}}, "meta": {"fixed_K_star": {"x": 1}}})
    assert out["fixed_K_star"] == {"200": {"K": 48}} and out["meta"]["fixed_K_star"] == {"x": 1}
    assert "by_cap" not in out and out["meta"]["cap"] == 64
    out = pt.merge_cap_block(FLAT, 128, {"p3_star": {"200": {"alpha": 0.5, "c": 1.2}}, "meta": {"cap": 128}})
    assert out["p3_star"] == FLAT["p3_star"], "the cap-64 block is untouched"
    assert out["by_cap"]["128"]["p3_star"] == {"200": {"alpha": 0.5, "c": 1.2}}
    assert out["by_cap"]["128"]["meta"]["cap"] == 128
    # A second block at the same cap merges rather than replaces.
    out = pt.merge_cap_block(out, 128, {"fixed_K_star": {"200": {"K": 96}}})
    assert set(out["by_cap"]["128"]) == {"p3_star", "meta", "fixed_K_star"}
    assert list(out) == list(FLAT) + ["by_cap"], "appended; existing key order kept"


def test_resolution_reads_the_deployed_caps_block_and_stamps_a_missing_one():
    params = pt.merge_cap_block(FLAT, 128, {"p3_star": {"200": {"alpha": 0.5, "c": 1.2}},
                                            "fixed_K_star": {"200": {"K": 96}}, "meta": {"cap": 128}})
    assert pt.resolve_params("p3_star", 200, cap=128, baseline_params=params) == {"alpha": 0.5, "c": 1.2}
    assert pt.resolve_params("fixed_K_star", 200, cap=128, baseline_params=params) == {"K": 96}
    assert pt.resolve_params("p3_star", 200, cap=64, baseline_params=params) == {"alpha": 0.5, "c": 3.0}
    assert pt.baseline_is_tuned("p3_star", 200, params, cap=128) is True
    # No block for cap 32: the cap-64 constant deploys, stamped as mis-capped.
    pt._warned.clear()
    p = pt.resolve_params("p3_star", 200, cap=32, baseline_params=params)
    assert p["c"] == 3.0 and p["params_tuned"] is False and p["params_cap"] == 64
    assert pt.baseline_is_tuned("p3_star", 200, params, cap=32) is False


def test_tau_is_read_at_the_deployed_cap_and_stamped_otherwise():
    thresholds = {"clock_quality_evidence_k4": {"tau_val": 0.61, "tau_off": 0.6},
                  "by_cap": {"128": {"clock_quality_evidence_k4": {"tau_val": 0.55, "tau_off": 0.6}}}}
    tau, source, tau_cap = pt.tau_for_cap("phi_k4", "clock_quality_evidence_k4", thresholds, cap=128)
    assert (tau, source, tau_cap) == (0.55, "tau_val", 128)
    tau, source, tau_cap = pt.tau_for_cap("phi_k4", "clock_quality_evidence_k4", thresholds, cap=64)
    assert (tau, source, tau_cap) == (0.61, "tau_val", 64)
    pt._warned.clear()
    tau, source, tau_cap = pt.tau_for_cap("phi_k4", "clock_quality_evidence_k4", thresholds, cap=32)
    assert (tau, source, tau_cap) == (0.61, "tau_val", 64)
    tau, source, tau_cap = pt.tau_for_cap("phi_k4", "clock_quality_evidence_k4", thresholds, cap=None)
    assert (tau, source, tau_cap) == (0.61, "tau_val", 64)  # where it came from, whatever is deployed


def test_merge_thresholds_cap_block():
    flat = {"v": {"tau_val": 0.6, "fingerprint": "a"}}
    out = pt.merge_thresholds_block(flat, 64, {"v": {"tau_val": 0.62, "fingerprint": "a"}})
    assert out["v"]["tau_val"] == 0.62 and "by_cap" not in out
    out = pt.merge_thresholds_block(flat, 128, {"v": {"tau_val": 0.5, "fingerprint": "a"}})
    assert out["v"]["tau_val"] == 0.6 and out["by_cap"]["128"]["v"]["tau_val"] == 0.5


def test_write_baseline_params_keeps_key_order(tmp_path):
    path = tmp_path / "baseline_params.json"
    pt.write_baseline_params(path, FLAT)
    raw = path.read_text()
    assert raw == json.dumps(FLAT, indent=2) + "\n"
    assert list(json.loads(raw)) == list(FLAT)


def test_tune_baselines_merges_a_non_tuning_cap_under_by_cap(tmp_path):
    import tune_baselines as tb

    path = tmp_path / "baseline_params.json"
    pt.write_baseline_params(path, FLAT)
    run = {"power": {"0.5": {"200": 1.1}}, "refine_after_init": {"200": 4},
           "p3_star": {"200": {"alpha": 0.5, "c": 1.1}}, "meta": {"cap": 128, "tune_split": "tune"}}
    tb.write_params_for_cap(run, path, 128)
    out = json.loads(path.read_text())
    assert out["p3_star"] == FLAT["p3_star"] and out["meta"]["cap"] == 64
    assert out["by_cap"]["128"]["p3_star"] == run["p3_star"] and out["by_cap"]["128"]["meta"]["cap"] == 128
    assert list(out) == list(FLAT) + ["by_cap"]
    # At the tuning cap, or on a fresh file, the historical whole-file write.
    tb.write_params_for_cap(run | {"meta": {"cap": 64}}, path, 64)
    assert json.loads(path.read_text())["p3_star"] == run["p3_star"]
    fresh = tmp_path / "new.json"
    tb.write_params_for_cap(run, fresh, 128)
    assert json.loads(fresh.read_text())["meta"]["cap"] == 128 and "by_cap" not in json.loads(fresh.read_text())


def test_selectors_write_a_non_tuning_cap_under_by_cap(tmp_path):
    import select_fixed_k as sk
    import select_rule as sr

    out = tmp_path / "deploy"
    (out / "tables").mkdir(parents=True)
    pt.write_baseline_params(out / "baseline_params.json", FLAT)
    sk.select(env_ids=("beta_good_common",), horizons=(50,), k_grid=(4, 12), n_replicates=8,
              out_dir=out, workers=1, cap=32)
    sr.select("level_star", env_ids=("beta_good_common",), horizons=(50,),
              candidates=({"alpha": 0.5, "c": 2.0, "b": 0.0},), n_replicates=8, out_dir=out, workers=1, cap=32)
    got = json.loads((out / "baseline_params.json").read_text())
    assert got["fixed_K_star"] == FLAT["fixed_K_star"], "the cap-64 block is untouched"
    assert got["by_cap"]["32"]["fixed_K_star"]["50"]["K"] in (4, 12)
    assert got["by_cap"]["32"]["level_star"]["b"] == 0.0
    assert got["by_cap"]["32"]["meta"]["fixed_K_star"]["cap"] == 32
    assert (out / "tables" / "fixed_k_selection_cap32.csv").exists()
    assert (out / "tables" / "level_selection_cap32.csv").exists()
    # And the reader resolves them at cap 32, falling back (stamped) at cap 48.
    assert pt.resolve_params("fixed_K_star", 50, cap=32, baseline_params=got)["K"] in (4, 12)
    pt._warned.clear()
    assert pt.resolve_params("level_star", 50, cap=48, baseline_params=got).get("params_tuned") is False


def test_select_thresholds_merges_per_cap_without_disturbing_the_flat_entries():
    import select_thresholds as st

    protocol = {"split": "val", "n_replicates": 500, "envs": ["e"], "horizons": [50]}
    current = {"v": "fp1", "w": "fp2"}
    flat = {"v": {"tau_val": 0.6, "fingerprint": "fp1", **protocol}}
    new64 = {"w": {"tau_val": 0.7, "fingerprint": "fp2", **protocol}}
    out = st.merge_thresholds(flat, new64, 64, protocol, current)
    assert set(out) == {"v", "w"} and "by_cap" not in out
    new128 = {"v": {"tau_val": 0.5, "fingerprint": "fp1", **protocol}}
    out = st.merge_thresholds(out, new128, 128, protocol, current)
    assert out["v"]["tau_val"] == 0.6 and out["by_cap"]["128"]["v"]["tau_val"] == 0.5
    # A later cap-64 run keeps by_cap; a stale-fingerprint entry inside a cap block is dropped.
    out = st.merge_thresholds(out, {"v": {"tau_val": 0.65, "fingerprint": "fp1", **protocol}}, 64, protocol, current)
    assert out["v"]["tau_val"] == 0.65 and out["by_cap"]["128"]["v"]["tau_val"] == 0.5
    out = st.merge_thresholds(out, {}, 128, protocol, {"v": "fp9", "w": "fp2"})
    assert "v" not in out["by_cap"]["128"]
