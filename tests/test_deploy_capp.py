"""`--test capp`: the CRN-paired cap sweep (roadmap 3.3, stage B)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import cells  # noqa: E402
import policy_table as pt  # noqa: E402
import run_deployment as rd  # noqa: E402


def test_capp_grid_is_the_registered_cap_ladder_on_the_eight_main_environments():
    grid = rd._cell_grid("capp", cells)
    assert {e for e, *_ in grid} == set(cells.MAIN_ENVS)
    at_200 = sorted({cap for e, T, cap, _ in grid if T == 200})
    at_1000 = sorted({cap for e, T, cap, _ in grid if T == 1000})
    assert at_200 == [32, 48, 64, 96, 128, 160, 200]
    assert at_1000 == [32, 48, 64, 96, 128, 192, 256, 384, 512, 1000]
    assert len(grid) == 8 * (7 + 10)
    assert {T for _, T, _, _ in grid} == {200, 1000}


def test_capp_cells_share_the_cap_64_seed_within_env_and_horizon():
    specs = rd.build_cells("capp", n_replicates=None, cells_mod=cells)
    assert len(specs) == 136 and all(s.n_replicates == rd.DEFAULT_REPLICATES["capp"] == 2000 for s in specs)
    by_key: dict[tuple[str, int], set[int]] = {}
    for s in specs:
        by_key.setdefault((s.env_id, s.horizon), set()).add(s.base_seed)
    # One seed per (env, T), whatever the cap -- the pairing -- and it is Test A's.
    assert all(len(seeds) == 1 for seeds in by_key.values())
    assert by_key[("beta_good_common", 200)] == {cells.base_seed("test", "beta_good_common", 200, 64)}
    # Cell names still differ across caps, so nothing collides on disk.
    names = [rd.cell_name(s) for s in specs]
    assert len(set(names)) == len(names)
    assert "beta_good_common_T1000_cap512" in names and "beta_good_common_T1000_cap1000" in names


def test_capp_policies_and_registry():
    assert "capp" in rd.TESTS and "capp" in pt.TEST_POLICIES
    assert set(pt.TEST_POLICIES["capp"]) == {
        "always_search", "cp0", "refine_after_init", "p3_star", "fixed_K_star", "level_star", "phi_k4", "phi_k16",
    }


def test_main_accepts_shared_seeds_for_capp_only(monkeypatch):
    """The runner refuses cells that share a base_seed -- except the paired sweep, where sharing IS the design."""
    assert rd.seeds_may_repeat("capp") is True
    assert rd.seeds_may_repeat("A") is False and rd.seeds_may_repeat("cap") is False
