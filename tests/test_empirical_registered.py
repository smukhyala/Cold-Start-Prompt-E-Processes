"""Pre-registration 9's decision rule: the prompt-bootstrap interval behind the flatness guard."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "deploy"))

import registered_contrast as rc  # noqa: E402

COL = "regret_posterior_mean_shrunk"


def _cell(root, test, env, T, seed, deltas, n=20):
    base = np.full(n, 0.1)
    for policy, shift in deltas.items():
        path = root / "episodes" / test / f"{env}_T{T}_cap{T}" / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"env_id": env, "horizon": T, "cap": T, "base_seed": seed, "episode": np.arange(n),
                      COL: base + shift}).to_parquet(path, index=False)


def _tree(root, point_delta, boot_deltas, informative, extra_cell_delta=None,
          policy="p3_star", reference="fixed_K_star", extra_boot_deltas=None):
    flat = []
    for pool in ("G", "F"):
        for T in (50, 100, 200):
            env = f"emp_{pool}_npmle"
            inf = (pool, T) in informative
            flat.append({"env_id": env, "pool": pool, "variant": "npmle", "horizon": T,
                         "regret_range": 0.05 if inf else 0.001, "informative": inf})
            d = point_delta if inf or extra_cell_delta is None else extra_cell_delta
            _cell(root, "emp", env, T, 1, {policy: d, reference: 0.0})
            boots_here = boot_deltas if inf or extra_boot_deltas is None else extra_boot_deltas
            for b, bd in enumerate(boots_here):
                _cell(root, "emp_boot", f"{env}_b{b:03d}", T, 100 + b, {policy: bd, reference: 0.0})
    (root / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(root / "tables" / "emp_flatness.csv", index=False)


def _run(root, expected_n_boot, policy="p3_star", reference="fixed_K_star", rule="noninferiority"):
    return rc.prompt_bootstrap_contrast(policy, reference, horizons=(50, 100, 200), mei=0.002,
                                        rule=rule, out_dir=root, expected_n_boot=expected_n_boot).iloc[0]


def test_registrations_are_written_down():
    expected = {
        "emp_primary": {"policy": "p3_star", "reference": "fixed_K_star", "rule": "noninferiority"},
        "emp_level": {"policy": "level_star", "reference": "p3_star", "rule": "not_better"},
        "emp_phi": {"policy": "phi_k4", "reference": "p3_star", "rule": "not_better"},
    }
    for name, exp in expected.items():
        reg = rc.REGISTRATIONS[name]
        assert reg["test"] == "emp" and reg["horizons"] == (50, 100, 200) and reg["mei"] == 0.002
        assert reg["interval"] == "prompt_bootstrap"
        assert reg["n_boot"] == 200
        assert reg["policy"] == exp["policy"] and reg["reference"] == exp["reference"]
        assert reg["rule"] == exp["rule"]


def test_uninformative_when_fewer_than_two_cells_move(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 200)})
    row = _run(tmp_path, expected_n_boot=5)
    assert row["verdict"] == "uninformative" and row["n_informative"] == 1


def test_supported_when_the_bootstrap_interval_sits_below_mei(tmp_path):
    boots = list(np.linspace(-0.001, 0.001, 40))
    _tree(tmp_path, 0.0, boots, informative={("F", 100), ("F", 200)})
    row = _run(tmp_path, expected_n_boot=40)
    assert row["verdict"] == "supported"
    assert row["lo"] == pytest.approx(np.percentile(boots, 2.5))
    assert row["hi"] == pytest.approx(np.percentile(boots, 97.5))
    assert row["n_boot"] == 40


def test_refuted_when_the_interval_sits_above_mei(tmp_path):
    _tree(tmp_path, 0.01, list(np.linspace(0.008, 0.012, 40)), informative={("G", 50), ("F", 50)})
    assert _run(tmp_path, expected_n_boot=40)["verdict"] == "refuted"


def test_only_informative_cells_enter_the_point_estimate(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 10, informative={("F", 100), ("F", 200)}, extra_cell_delta=0.5)
    assert _run(tmp_path, expected_n_boot=10)["delta"] == pytest.approx(0.0)


def test_noninformative_cells_wild_bootstrap_deltas_do_not_affect_the_interval(tmp_path):
    boots = list(np.linspace(-0.0005, 0.0005, 40))
    wild = list(np.linspace(-5.0, 5.0, 40))
    _tree(tmp_path, 0.0, boots, informative={("F", 100), ("F", 200)}, extra_boot_deltas=wild)
    row = _run(tmp_path, expected_n_boot=40)
    assert row["verdict"] == "supported"
    assert row["lo"] == pytest.approx(np.percentile(boots, 2.5))
    assert row["hi"] == pytest.approx(np.percentile(boots, 97.5))


def test_a_missing_bootstrap_cell_is_an_error(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 3, informative={("F", 100), ("F", 200)})
    victim = tmp_path / "episodes" / "emp_boot" / "emp_F_npmle_b001_T200_cap200"
    for f in victim.iterdir():
        f.unlink()
    victim.rmdir()
    with pytest.raises(ValueError, match="b001"):
        _run(tmp_path, expected_n_boot=3)


def test_a_tail_missing_replicate_is_an_error(tmp_path):
    """B is fixed at `expected_n_boot`; a boot run that died before writing the last replicate
    must not silently shrink B -- it has to raise."""
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    for T in (100, 200):
        victim = tmp_path / "episodes" / "emp_boot" / f"emp_F_npmle_b004_T{T}_cap{T}"
        for f in victim.iterdir():
            f.unlink()
        victim.rmdir()
    with pytest.raises(ValueError, match="b004"):
        _run(tmp_path, expected_n_boot=5)


def test_missing_bootstrap_directory_is_an_error(tmp_path):
    """No `emp_boot` tree at all must raise, not fall through to an inconclusive verdict."""
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    shutil.rmtree(tmp_path / "episodes" / "emp_boot")
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, expected_n_boot=5)


def test_bootstrap_tree_with_no_informative_data_is_an_error(tmp_path):
    """An `emp_boot` tree that holds cells but none for the informative keys must raise,
    not silently compute an empty-interval "inconclusive" verdict."""
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    for T in (100, 200):
        for b in range(5):
            victim = tmp_path / "episodes" / "emp_boot" / f"emp_F_npmle_b{b:03d}_T{T}_cap{T}"
            for f in victim.iterdir():
                f.unlink()
            victim.rmdir()
    with pytest.raises(ValueError):
        _run(tmp_path, expected_n_boot=5)


def test_not_better_rule_through_the_bootstrap_path_refuted(tmp_path):
    boots = list(np.linspace(-0.012, -0.008, 40))
    _tree(tmp_path, -0.01, boots, informative={("F", 100), ("F", 200)},
          policy="level_star", reference="p3_star")
    row = _run(tmp_path, expected_n_boot=40, policy="level_star", reference="p3_star", rule="not_better")
    assert row["verdict"] == "refuted"


def test_not_better_rule_through_the_bootstrap_path_supported(tmp_path):
    boots = list(np.linspace(-0.0005, 0.0005, 40))
    _tree(tmp_path, 0.0, boots, informative={("F", 100), ("F", 200)},
          policy="level_star", reference="p3_star")
    row = _run(tmp_path, expected_n_boot=40, policy="level_star", reference="p3_star", rule="not_better")
    assert row["verdict"] == "supported"


def test_duplicate_emp_cell_for_the_same_key_is_an_error(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    dup_dir = tmp_path / "episodes" / "emp" / "emp_F_npmle_T100_cap100_dup"
    dup_dir.mkdir(parents=True)
    for policy_name in ("p3_star", "fixed_K_star"):
        pd.DataFrame({"env_id": "emp_F_npmle", "horizon": 100, "cap": 100, "base_seed": 1,
                      "episode": np.arange(20), COL: np.full(20, 0.1)}).to_parquet(
            dup_dir / f"{policy_name}.parquet", index=False)
    with pytest.raises(ValueError, match="two cells"):
        _run(tmp_path, expected_n_boot=5)


# ---- I8(a): the episode-paired CI beside the interval of record; I7: one data snapshot --------


def test_paired_ci_is_reported_beside_the_bootstrap_interval(tmp_path):
    from cold_start.growing.deploy import stats

    boots = list(np.linspace(-0.001, 0.001, 20))
    _tree(tmp_path, 0.0, boots, informative={("F", 100), ("F", 200)})
    row = _run(tmp_path, expected_n_boot=20)
    diffs, _, _ = rc._diffs(tmp_path, "emp", "p3_star", "fixed_K_star", (100, 200))
    diffs = {c: d for c, d in diffs.items() if c.startswith("emp_F_npmle")}
    expected = stats.stratified_pooled(diffs, n_boot=10_000, seed=rc.ad._seed("registered", "p3_star",
                                                                               "fixed_K_star", "emp"))
    assert row["paired_lo"] == pytest.approx(expected["lo"]) and row["paired_hi"] == pytest.approx(expected["hi"])
    assert row["lo"] == pytest.approx(np.percentile(boots, 2.5))  # the verdict still reads the bootstrap
    assert list(rc.BOOT_COLUMNS).index("paired_lo") == list(rc.BOOT_COLUMNS).index("hi") + 1


def _snapshot(root, shas):
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"reservoirs": shas}))
    flat = pd.read_csv(root / "tables" / "emp_flatness.csv")
    flat["reservoir_sha256"] = flat["env_id"].map(lambda e: shas.get(e.removeprefix("emp_"), "?"))
    flat.to_csv(root / "tables" / "emp_flatness.csv", index=False)
    for test in ("emp", "emp_boot"):
        rc.write_reservoir_stamp(root, test, manifest)
    return manifest


def _run_checked(root, manifest):
    return rc.prompt_bootstrap_contrast("p3_star", "fixed_K_star", horizons=(50, 100, 200), mei=0.002,
                                        rule="noninferiority", out_dir=root, expected_n_boot=5,
                                        reservoir_manifest=manifest).iloc[0]


def test_matching_snapshot_passes(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    manifest = _snapshot(tmp_path, {"G_npmle": "g", "F_npmle": "f"})
    assert _run_checked(tmp_path, manifest)["verdict"] == "supported"


def test_flatness_from_another_reservoir_is_refused(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    manifest = _snapshot(tmp_path, {"G_npmle": "g", "F_npmle": "f"})
    manifest.write_text(json.dumps({"reservoirs": {"G_npmle": "g", "F_npmle": "f-new"}}))
    with pytest.raises(ValueError, match="emp_F_npmle"):
        _run_checked(tmp_path, manifest)


def test_episodes_from_another_manifest_are_refused(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    manifest = _snapshot(tmp_path, {"G_npmle": "g", "F_npmle": "f"})
    (tmp_path / "tables" / "emp_boot_reservoir_stamp.json").write_text('{"manifest_sha256": "old"}')
    with pytest.raises(ValueError, match="emp_boot"):
        _run_checked(tmp_path, manifest)
    (tmp_path / "tables" / "emp_reservoir_stamp.json").unlink()
    with pytest.raises(ValueError, match="emp_reservoir_stamp"):
        _run_checked(tmp_path, manifest)


def test_flatness_without_shas_is_refused_when_a_manifest_is_given(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 100), ("F", 200)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"reservoirs": {}}')
    with pytest.raises(ValueError, match="reservoir_sha256"):
        _run_checked(tmp_path, manifest)
