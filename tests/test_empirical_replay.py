"""Replay cells: unique ids, disjoint seeds, CRN across variants, and the runner's plumbing."""

from __future__ import annotations

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
