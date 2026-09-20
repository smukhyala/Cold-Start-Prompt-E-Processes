"""The K-matched control (NEXT-STEPS 2.4): is Φ a learned rule, or only a K_final chooser?

The control -- `always_search` capped at a learned policy's own realised K_final -- is only
a control if it runs on the SAME episodes as the learned policy, so a matched cell must
carry the Test-A (or Test-C) cell's `base_seed` with only the cap overridden. That is the
property these tests pin down first; the grid and the analysis follow from it.
"""

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

import analyze_capmatch as ac  # noqa: E402
import cells  # noqa: E402
import run_deployment as rd  # noqa: E402

ENV = "beta_good_common"


# ---- the matched cell --------------------------------------------------------------------


def test_matched_cell_keeps_the_seed_cap_cell_seed_and_overrides_only_the_cap():
    base = cells.make_cell("test", ENV, 200, 64, 2000)
    matched = cells.make_matched_cell("test", ENV, 200, 15, 2000, seed_cap=64)
    assert matched.base_seed == base.base_seed, "CRN pairing with the deployed episodes"
    assert matched.cap == 15 and base.cap == 64
    assert (matched.env_id, matched.horizon, matched.n_replicates) == (ENV, 200, 2000)
    # It is NOT a cell of the study's grid: its (env, T, cap) has no cell_id of its own.
    with pytest.raises(ValueError):
        cells.cell_id(ENV, 200, 15)


def test_matched_cell_refuses_a_cap_the_harness_cannot_run():
    with pytest.raises(ValueError):
        cells.make_matched_cell("test", ENV, 200, 1, 2000)
    with pytest.raises(ValueError):
        cells.make_matched_cell("test", ENV, 200, 201, 2000)  # above the horizon


# ---- the grid ---------------------------------------------------------------------------


def _cells_table(rows):
    return pd.DataFrame(rows, columns=["test", "policy", "env_id", "horizon", "cap", "k_final"])


def test_matched_grid_rounds_k_final_and_skips_cells_already_at_the_cap(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    _cells_table([
        ("A", "phi_k4", ENV, 50, 64, 14.5), ("A", "phi_k4", ENV, 100, 64, 25.9),
        ("A", "phi_k4", ENV, 200, 64, 63.9), ("A", "phi_k4", ENV, 500, 64, 64.0),
        ("A", "phi_k16", ENV, 50, 64, 18.0),
    ]).to_csv(tables / "cells_A_primary.csv", index=False)
    _cells_table([
        ("C", "phi_k4", "mix_a", 50, 64, 30.4), ("C", "phi_k4", "mix_a", 200, 64, 64.0),
    ]).to_csv(tables / "cells_C_primary.csv", index=False)
    grid = rd.matched_grid("phi_k4", tables_dir=tables, horizons=(50, 100, 200))
    # round(k_final); the T=500 row is outside the horizons; the K=64 rows ARE Test A/C.
    assert grid == [
        ("A", ENV, 50, 14), ("A", ENV, 100, 26), ("C", "mix_a", 50, 30),
    ]
    skipped = rd.matched_grid("phi_k4", tables_dir=tables, horizons=(50, 100, 200), include_at_cap=True)
    assert ("A", ENV, 200, 64) in skipped and ("C", "mix_a", 200, 64) in skipped


def test_matched_grid_needs_a_policy_that_ran(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    _cells_table([("A", "phi_k4", ENV, 50, 64, 14.5)]).to_csv(tables / "cells_A_primary.csv", index=False)
    _cells_table([]).to_csv(tables / "cells_C_primary.csv", index=False)
    with pytest.raises(RuntimeError, match="phi_k1"):
        rd.matched_grid("phi_k1", tables_dir=tables, horizons=(50,))


def test_build_cells_for_capmatch_uses_matched_cells(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    _cells_table([("A", "phi_k4", ENV, 50, 64, 14.5), ("A", "phi_k4", ENV, 200, 64, 64.0)]).to_csv(
        tables / "cells_A_primary.csv", index=False)
    _cells_table([]).to_csv(tables / "cells_C_primary.csv", index=False)
    specs = rd.build_cells("capmatch", n_replicates=None, cells_mod=cells,
                           match="phi_k4", tables_dir=tables)
    assert [rd.cell_name(s) for s in specs] == [f"{ENV}_T50_cap14"]
    assert specs[0].base_seed == cells.base_seed("test", ENV, 50, 64)
    assert specs[0].n_replicates == rd.DEFAULT_REPLICATES["capmatch"] == 2000


def test_capmatch_requires_match():
    with pytest.raises(RuntimeError, match="--match"):
        rd.build_cells("capmatch", n_replicates=None, cells_mod=cells)


# ---- the analysis -------------------------------------------------------------------------


def _frame(policy: str, cell: str, env: str, fam: str, T: int, cap: int, seed: int, n: int,
           shift: float, k_final: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.12, 0.02, size=n)  # the shared (CRN) episode component
    return pd.DataFrame({
        "test": "x", "cell": cell, "family": fam, "group": "g", "policy": policy, "env_id": env,
        "horizon": T, "cap": cap, "base_seed": seed, "episode": np.arange(n), "replicate": np.arange(n),
        "regret_posterior_mean_shrunk": base + shift + rng.normal(0, 0.002, size=n),
        "k_final": np.full(n, k_final), "search_frac": np.full(n, 0.3),
    })


def _write_run(root: Path, test: str, cell: str, env: str, fam: str, T: int, cap: int, seed: int,
               policies: dict[str, tuple[float, float]], n: int = 64) -> None:
    for policy, (shift, k_final) in policies.items():
        path = root / "episodes" / test / cell / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _frame(policy, cell, env, fam, T, cap, seed, n, shift, k_final).to_parquet(path, index=False)


@pytest.fixture
def run(tmp_path):
    """Four Test-A environments at T=50 and T=100, matched for phi_k4; one cell sits at K=64
    and so is read from Test A itself. Φ is 0.003 better than its K-matched control in
    every cell except the at-cap one, where it is worse by 0.001."""
    root = tmp_path / "deploy"
    tables = root / "tables"
    tables.mkdir(parents=True)
    rows = []
    envs = ["e1", "e2", "e3", "e4"]
    for i, env in enumerate(envs):
        for T in (50, 100):
            seed = 1000 + 10 * i + T
            at_cap = (env == "e4" and T == 100)
            K = 64 if at_cap else 10 + i
            rows.append(("A", "phi_k4", env, T, 64, float(K) + 0.2))
            if at_cap:
                _write_run(root, "A", f"{env}_T{T}_cap64", env, "A", T, 64, seed,
                           {"phi_k4": (+0.001, 64.0), "always_search": (0.0, 64.0), "uniform": (0.002, 64.0)})
            else:
                _write_run(root, "capmatch", f"{env}_T{T}_cap{K}", env, "A", T, K, seed,
                           {"phi_k4": (-0.003, K - 0.5), "always_search": (0.0, float(K)), "uniform": (0.002, K - 1.0)})
    _cells_table(rows).to_csv(tables / "cells_A_primary.csv", index=False)
    _cells_table([]).to_csv(tables / "cells_C_primary.csv", index=False)
    return root


def test_capmatch_contrasts_pair_by_episode_and_source_at_cap_cells_from_test_a(run):
    out = ac.capmatch_contrasts("phi_k4", out_dir=run, horizons=(50, 100), n_boot=200)
    assert set(out["reference"]) == {"always_search", "uniform"}
    cell_rows = out[(out["level"] == "cell") & (out["reference"] == "always_search")]
    assert len(cell_rows) == 8
    at_cap = cell_rows[(cell_rows["env_id"] == "e4") & (cell_rows["horizon"] == 100)].iloc[0]
    assert at_cap["source_test"] == "A" and at_cap["cap"] == 64 and at_cap["delta"] == pytest.approx(0.001, abs=5e-4)
    others = cell_rows[~((cell_rows["env_id"] == "e4") & (cell_rows["horizon"] == 100))]
    assert (others["source_test"] == "capmatch").all()
    np.testing.assert_allclose(others["delta"], -0.003, atol=5e-4)
    # Paired: the CRN component cancels, so the paired CI is far narrower than the spread of regret.
    assert (others["hi"] - others["lo"]).max() < 0.004
    # The control really did stop at K, and the learned policy's in-cell K is carried beside it.
    assert (others["k_final_ref"] == others["cap"]).all()
    assert (others["k_final"] < others["cap"]).all()


def test_capmatch_pooled_rows_use_the_environment_mean_t(run):
    out = ac.capmatch_contrasts("phi_k4", out_dir=run, horizons=(50, 100), n_boot=200)
    vs = out[out["reference"] == "always_search"]
    pooled = vs[(vs["level"] == "pooled") & (vs["test"] == "A")].iloc[0]
    assert pooled["n_cells"] == 8 and pooled["n_envs"] == 4
    assert pooled["delta"] == pytest.approx(vs[vs["level"] == "cell"]["delta"].mean())
    assert pooled["cluster_method"] == "env_mean_t"
    assert np.isfinite(pooled["cluster_lo"]) and np.isfinite(pooled["cluster_p"])
    horizon = vs[(vs["level"] == "horizon") & (vs["test"] == "A")]
    assert sorted(horizon["horizon"]) == [50, 100]


def test_capmatch_refuses_a_missing_control(run):
    (run / "episodes" / "capmatch" / "e1_T50_cap10" / "always_search.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="always_search"):
        ac.capmatch_contrasts("phi_k4", out_dir=run, horizons=(50, 100), n_boot=50)
