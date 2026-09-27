"""Replay cells: unique ids, disjoint seeds, CRN across variants, and the runner's plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402

RES = EmpiricalReservoir([0.3, 0.6, 0.9], [0.5, 0.3, 0.2], label="t")


def test_env_ids():
    assert replay.env_id("G", "npmle") == "emp_G_npmle"
    assert replay.env_id("F", "npmle", boot=7) == "emp_F_npmle_b007"


def test_variants_share_seeds_and_everything_else_differs():
    seeds = {(p, T): replay.seed_for(p, T) for p in replay.POOLS for T in replay.ALL_HORIZONS}
    assert len(set(seeds.values())) == len(seeds)
    cells = [replay.make_emp_cell(p, v, T, RES, 10) for p in replay.POOLS for v in replay.VARIANTS
             for T in replay.ALL_HORIZONS]
    assert len({rd.cell_name(c) for c in cells}) == len(cells)
    by_pool_T = {}
    for c in cells:
        by_pool_T.setdefault((c.env_id.split("_")[1], c.horizon), set()).add(c.base_seed)
    assert all(len(s) == 1 for s in by_pool_T.values())


def test_boot_env_ids_and_seeds_unique():
    cells = [replay.make_emp_cell(p, "npmle", T, RES, 10, boot=b) for b in range(replay.N_BOOT)
             for p in replay.POOLS for T in replay.PRIMARY_HORIZONS]
    assert len({rd.cell_name(c) for c in cells}) == len(cells)
    assert len({c.base_seed for c in cells}) == len(cells)
    point = {replay.seed_for(p, T) for p in replay.POOLS for T in replay.ALL_HORIZONS}
    assert not point & {c.base_seed for c in cells}


def test_seeds_are_disjoint_from_the_study():
    import cells as study_cells

    seeds = [replay.seed_for(p, T, b) for p in replay.POOLS for T in replay.ALL_HORIZONS
             for b in (None, 0, replay.N_BOOT - 1)]
    assert min(seeds) >= replay.EMP_SEED_BASE
    study_cells.assert_seed_disjointness(seeds)


def test_cells_are_uncapped():
    c = replay.make_emp_cell("G", "npmle", 200, RES, 10)
    assert c.cap == 200 and c.horizon == 200 and c.env_spec["type"] == "empirical"


def test_the_runner_knows_the_tests():
    assert "emp" in rd.TESTS and "emp_boot" in rd.TESTS
    assert rd.seeds_may_repeat("emp")
    with pytest.raises(RuntimeError, match="replay.py"):
        rd.build_cells("emp", n_replicates=None, cells_mod=rd.import_cells_module())


def _outcomes(rng, n_arms=30, n_tasks=60, n_reps=60):
    rows = []
    for pool in ("G", "F"):
        mus = rng.uniform(0.4, 0.8, n_arms)
        for i, mu in enumerate(mus):
            for t in range(n_tasks):
                rows.append({"pool": pool, "arm_id": f"{pool}_{i:02d}", "task_id": f"t{t}", "replicate": 0,
                             "attempt": 1, "status": "ok", "success": int(rng.random() < mu)})
        for k in range(n_reps):
            i, t = int(rng.integers(n_arms)), int(rng.integers(n_tasks))
            rows.append({"pool": pool, "arm_id": f"{pool}_{i:02d}", "task_id": f"t{t}", "replicate": 1,
                         "attempt": 1, "status": "ok", "success": int(rng.random() < mus[i])})
    return pd.DataFrame(rows).drop_duplicates(["arm_id", "task_id", "replicate"])


def test_estimate_builds_every_variant():
    outcomes = _outcomes(np.random.default_rng(0))
    reservoirs, noise, fits = replay.estimate(outcomes)
    assert set(reservoirs) == {(p, v) for p in replay.POOLS for v in replay.VARIANTS}
    assert set(noise["v"]) == {"G", "F"} and noise["n_pairs"] > 0
    assert fits["selected"].sum() == 2


def test_bootstrap_resamples_prompts_and_labels_replicates():
    outcomes = _outcomes(np.random.default_rng(1))
    _, noise, _ = replay.estimate(outcomes)
    boots = replay.bootstrap_reservoirs(outcomes, noise, n_boot=3, seed=5)
    assert [(b, p) for b, p, _ in boots] == [(b, p) for b in range(3) for p in replay.POOLS]
    assert boots[0][2].label == "G_npmle_b000"
    again = replay.bootstrap_reservoirs(outcomes, noise, n_boot=3, seed=5)
    assert all(np.array_equal(a[2].weights, b[2].weights) for a, b in zip(boots, again, strict=True))


def test_kgrid_runs_on_empirical_cells():
    cells = [replay.make_emp_cell("G", "npmle", 20, RES, 16)]
    frame = replay.kgrid(cells, workers=1, k_grid=(2, 4, 8))
    assert sorted(frame["K"]) == [2, 4, 8]
    assert set(frame["pool"]) == {"G"} and set(frame["variant"]) == {"npmle"}
    assert np.all(np.isfinite(frame["regret"]))


def test_point_replay_writes_paired_episodes(tmp_path):
    cell = replay.make_emp_cell("G", "npmle", 20, RES, 8)
    rd.main(["--test", "emp", "--workers", "1", "--policies", "always_search", "--out-dir", str(tmp_path),
             "--skip-summary"], cells=[cell])
    path = tmp_path / "episodes" / "emp" / rd.cell_name(cell) / "always_search.parquet"
    frame = pd.read_parquet(path)
    assert len(frame) == 8 and int(frame["base_seed"].iloc[0]) == cell.base_seed


# ---- Fix round 1: cap-T constants preflight (Important #1) ----------------------------


def _complete_baseline_params(cap: int) -> dict:
    c = str(int(cap))
    return {"by_cap": {c: {
        "p3_star": {c: {"alpha": 0.5, "c": 1.0}},
        "fixed_K_star": {c: {"K": 16}},
        "level_star": {"alpha": 0.5, "c": 1.0, "b": 0.1},
    }}}


def _complete_thresholds(cap: int) -> dict:
    return {"by_cap": {str(int(cap)): {replay.PHI_K4_VARIANT: {"tau_val": 0.5}}}}


def test_missing_cap_constants_is_empty_when_every_block_is_present():
    assert replay.missing_cap_constants(
        _complete_baseline_params(50), _complete_thresholds(50), [50]
    ) == []


def test_missing_cap_constants_names_a_missing_cap_block():
    problems = replay.missing_cap_constants({"by_cap": {}}, _complete_thresholds(50), [50])
    assert any("50" in p and "baseline_params" in p for p in problems)


def test_missing_cap_constants_names_an_incomplete_level_star_block():
    baseline_params = _complete_baseline_params(50)
    del baseline_params["by_cap"]["50"]["level_star"]["b"]
    problems = replay.missing_cap_constants(baseline_params, _complete_thresholds(50), [50])
    assert any("level_star" in p for p in problems)


def test_missing_cap_constants_names_a_missing_p3_star_entry():
    baseline_params = _complete_baseline_params(50)
    del baseline_params["by_cap"]["50"]["p3_star"]["50"]
    problems = replay.missing_cap_constants(baseline_params, _complete_thresholds(50), [50])
    assert any("p3_star" in p for p in problems)


def test_missing_cap_constants_names_a_missing_threshold():
    problems = replay.missing_cap_constants(_complete_baseline_params(50), {"by_cap": {}}, [50])
    assert any(replay.PHI_K4_VARIANT in p and "thresholds" in p for p in problems)


def test_missing_cap_constants_checks_every_horizon_independently():
    baseline_params = _complete_baseline_params(50)
    thresholds = _complete_thresholds(50)
    problems = replay.missing_cap_constants(baseline_params, thresholds, [50, 100])
    assert problems and all("100" in p for p in problems)
    assert not any("50" in p for p in problems)


def test_require_cap_constants_exits_when_incomplete(tmp_path):
    replay.pt.write_baseline_params(tmp_path / "baseline_params.json", _complete_baseline_params(50))
    (tmp_path / "thresholds.json").write_text(json.dumps(_complete_thresholds(50)))
    with pytest.raises(SystemExit, match="point"):
        replay._require_cap_constants(tmp_path, [50, 100], "point")


def test_require_cap_constants_passes_when_complete(tmp_path):
    replay.pt.write_baseline_params(tmp_path / "baseline_params.json", _complete_baseline_params(50))
    (tmp_path / "thresholds.json").write_text(json.dumps(_complete_thresholds(50)))
    replay._require_cap_constants(tmp_path, [50], "point")  # must not raise


# ---- Fix round 1: comparator cache invalidation on re-estimate (Important #2) ----------


def test_purge_stale_emp_comparators_removes_only_emp_prefixed_files(tmp_path):
    comparators = tmp_path / "comparators"
    comparators.mkdir()
    emp_files = [
        comparators / "emp_G_npmle_T50_cap50_seed400010050_M1000.npy",
        comparators / "emp_G_npmle_T50_cap50_seed400010050_M1000.json",
        comparators / "emp_F_npmle_b005_T50_cap50_seed420020050_M250.npy",
    ]
    other_files = [
        comparators / "beta_good_common_T50_cap64_seed10262460_M2000.npy",
        comparators / "beta_good_common_T50_cap64_seed10262460_M2000.json",
        comparators / "readme.txt",
    ]
    for f in emp_files + other_files:
        f.write_bytes(b"x")
    n = replay.purge_stale_emp_comparators(tmp_path)
    assert n == len(emp_files)
    assert all(not f.exists() for f in emp_files)
    assert all(f.exists() for f in other_files)


def test_purge_stale_emp_comparators_on_a_missing_comparators_dir_is_a_noop(tmp_path):
    assert replay.purge_stale_emp_comparators(tmp_path / "nope") == 0


# ---- Fix round 1: kgrid prebuilds CS tables before spawning workers (folded minor) -----


def test_kgrid_prebuilds_cs_tables_before_spawning_workers(monkeypatch):
    calls: list[tuple[int, float]] = []
    original = replay.CSTable.load_or_build.__func__

    def spy(cls, horizon, alpha=0.05, **kwargs):
        calls.append((int(horizon), float(alpha)))
        return original(cls, horizon, alpha, **kwargs)

    monkeypatch.setattr(replay.CSTable, "load_or_build", classmethod(spy))
    cells = [replay.make_emp_cell("G", "npmle", 21, RES, 4)]
    replay.kgrid(cells, workers=1, k_grid=(2, 4))
    assert (21, 0.05) in calls
