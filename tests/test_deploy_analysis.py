"""Behaviour tests for the M7 analysis (`analyze_deployment.py`, `ood.py`,
`reservoir_analysis.py`, `make_deploy_figures.py`).

The statistical machinery is checked on a synthetic episode frame with *planted*
effects (3 cells x 4 policies x 64+ episodes, CRN-style shared episode noise), so the
expected answers are known: the paired delta must recover the planted offset inside
its CI, the pooled block must weight cells equally, the pre-registered table must hold
exactly three rows per stratum. The on-policy diagnostics are checked on real
snapshots from a short `harness.run_cell` with a tiny learned policy (no simulator
mocks): the parity gate must pass on them, the decision lookup must reproduce the
harness's recorded actions, and the reservoir quadrature must match the Beta closed
form. Figures are rendered on the synthetic tables under the Agg backend.
"""

from __future__ import annotations

import json
import pickle
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
for _p in (ROOT / "experiments" / "growing_bandits", DEPLOY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_deployment as ad  # noqa: E402
import make_deploy_figures as mf  # noqa: E402
import ood  # noqa: E402
import reservoir_analysis as ra  # noqa: E402

from cold_start.growing.deploy import feature_groups as fg  # noqa: E402
from cold_start.growing.deploy import stats  # noqa: E402
from cold_start.growing.deploy.artifacts import load_model, save_model  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec, LogSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.pairwise_table import get_pairwise_table  # noqa: E402
from cold_start.growing.deploy.recommenders import (  # noqa: E402
    PRIMARY_RECOMMENDER,
    RECOMMENDER_NAMES,
)
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.deploy.transforms import beta_excess_mean  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

POLICIES = ("cp0", "p3_star", "phi_k16", "phi_k16_quality")
GROUPS = {"cp0": "reference", "p3_star": "reference", "phi_k16": "learned", "phi_k16_quality": "learned"}
#: Planted mean regret offsets vs cp0 (negative = better).
PLANTED = {"cp0": 0.0, "p3_star": -0.01, "phi_k16": -0.03, "phi_k16_quality": 0.02}
CELLS = (
    ("beta_good_common", "A", 100, 64),
    ("beta_skewed", "A", 200, 64),
    ("tail_b2.0_mu1.0_c1.0", "B", 100, 128),
)
N_BOOT = 2000


# ---- synthetic episode frames -------------------------------------------------------------


def _episode_frame(cell: str, env: str, family: str, horizon: int, policy: str, n: int, seed: int) -> pd.DataFrame:
    """One policy's per-episode frame in the runner's schema, with a planted regret offset.

    Episode noise is shared across policies of a cell (the same `seed`), as CRN makes
    it in the real harness, plus a small policy-specific component.
    """
    shared = np.random.default_rng(seed)
    own = np.random.default_rng([seed, zlib.crc32(policy.encode())])
    base = 0.10 + 0.04 * shared.standard_normal(n)
    mu_star = np.clip(0.9 + 0.03 * shared.standard_normal(n), 0.5, 1.0)
    regret = np.clip(base + PLANTED[policy] + 0.005 * own.standard_normal(n), 0.0, 1.0)
    disc = np.clip(0.5 * regret + 0.002 * own.standard_normal(n), 0.0, None)
    cols: dict = {
        "test": "synthetic", "cell": cell, "family": family, "group": GROUPS[policy], "policy": policy,
        "env_id": env, "horizon": horizon, "cap": 64, "base_seed": seed, "alpha": 0.05,
        "n_initial_arms": 2, "policy_seed": zlib.crc32(policy.encode()),
        "episode": np.arange(n), "mu_star": mu_star, "mu_star_cap": mu_star,
        "best_discovered": mu_star - disc, "regret_disc": disc,
        "k_final": own.integers(4, 30, n), "search_frac": own.uniform(0.05, 0.4, n),
        "cap_hit": own.uniform(size=n) < 0.1, "t_cap_hit": -1,
        "n_eliminated_final": own.integers(0, 5, n), "herfindahl": own.uniform(0.1, 0.5, n),
        "n_singletons_final": own.integers(0, 6, n), "n_demoted": own.integers(0, 3, n),
    }
    cols["t_cap_hit"] = np.where(cols["cap_hit"], own.integers(20, horizon, n), -1)
    for i, rec in enumerate(RECOMMENDER_NAMES):
        r = np.clip(regret + 0.002 * i, 0.0, 1.0)
        cols[f"q_{rec}"] = mu_star - r
        cols[f"n_rec_{rec}"] = own.integers(5, 40, n)
        cols[f"regret_{rec}"] = r
        cols[f"regret_sel_{rec}"] = r - disc
        cols[f"regret_sup_{rec}"] = 1.0 - (mu_star - r)
    cols["replicate"] = cols["episode"]
    return pd.DataFrame(cols)


def synthetic_cells(policies=POLICIES) -> list[ad.Cell]:
    out: list[ad.Cell] = []
    for i, (env, fam, T, n) in enumerate(CELLS):
        name = f"{env}_T{T}_cap64"
        seed = 10_000 + i
        frames = {p: _episode_frame(name, env, fam, T, p, n, seed) for p in policies}
        out.append(ad.Cell(name=name, env_id=env, family=fam, horizon=T, cap=64, base_seed=seed, n=n,
                           frames=frames, groups={p: GROUPS[p] for p in policies}))
    return out


@pytest.fixture(scope="module")
def cells() -> list[ad.Cell]:
    return synthetic_cells()


@pytest.fixture(scope="module")
def per_cell(cells) -> dict[str, ad.CellStats]:
    return {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, N_BOOT) for c in cells}


# ---- paired deltas and pooling ----------------------------------------------------------------


def test_paired_delta_recovers_the_planted_effect(cells, per_cell):
    strata = ad.strata_of(cells)
    pooled = next(s for s in strata if s.level == "pooled")
    for policy, planted in PLANTED.items():
        a = ad.aggregate(per_cell, pooled, policy, "d_regret_vs_cp0")
        assert a["n_cells"] == 3 and a["n_episodes"] == 64 + 64 + 128
        assert a["lo"] <= planted <= a["hi"], (policy, planted, a)
        assert abs(a["mean"] - planted) < 0.004, (policy, a["mean"])
        # The paired CI is far tighter than the marginal spread (0.04 sd of shared noise).
        assert a["hi"] - a["lo"] < 0.01
    # Sign convention: negative delta = policy better; win rate = share of episodes it wins.
    a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
    assert a["win"] > 0.95
    a = ad.aggregate(per_cell, pooled, "phi_k16_quality", "d_regret_vs_cp0")
    assert a["win"] < 0.05
    # A reference against itself is exactly zero with a zero-width interval.
    a = ad.aggregate(per_cell, pooled, "cp0", "d_regret_vs_cp0")
    assert a["mean"] == 0.0 and a["lo"] == 0.0 and a["hi"] == 0.0 and a["win"] == 0.5


def test_cell_engine_agrees_with_stats_paired_bootstrap(cells, per_cell):
    """The multinomial-weight engine is the same percentile bootstrap as `stats`."""
    c = cells[0]
    d = (c.frames["phi_k16"][f"regret_{PRIMARY_RECOMMENDER}"] - c.frames["cp0"][f"regret_{PRIMARY_RECOMMENDER}"]).to_numpy()
    ref = stats.paired_bootstrap(d, n_boot=N_BOOT, seed=3)
    s = ad.Stratum("cell", c.family, str(c.horizon), c.name, (c.name,))
    a = ad.aggregate(per_cell, s, "phi_k16", "d_regret_vs_cp0")
    assert a["mean"] == pytest.approx(ref["mean"], abs=1e-12)
    assert a["se"] == pytest.approx(ref["se_paired"], rel=1e-9)
    assert a["lo"] == pytest.approx(ref["lo"], abs=0.5 * ref["se_paired"])
    assert a["hi"] == pytest.approx(ref["hi"], abs=0.5 * ref["se_paired"])


def test_pooled_block_weights_cells_equally(cells, per_cell):
    strata = ad.strata_of(cells)
    pooled = next(s for s in strata if s.level == "pooled")
    cell_means = [per_cell[c.name].mean[("phi_k16_quality", "regret")] for c in cells]
    a = ad.aggregate(per_cell, pooled, "phi_k16_quality", "regret")
    assert a["mean"] == pytest.approx(float(np.mean(cell_means)), abs=1e-12)
    # ... and not the episode-weighted mean, which the 128-episode cell would dominate.
    episode_weighted = np.concatenate(
        [c.frames["phi_k16_quality"][f"regret_{PRIMARY_RECOMMENDER}"].to_numpy() for c in cells]
    ).mean()
    assert abs(a["mean"] - episode_weighted) > 1e-6
    # `stats.stratified_pooled` is the reference construction for the same quantity.
    diffs = {c.name: (c.frames["phi_k16"][f"regret_{PRIMARY_RECOMMENDER}"] - c.frames["cp0"][f"regret_{PRIMARY_RECOMMENDER}"]).to_numpy() for c in cells}
    ref = stats.stratified_pooled(diffs, n_boot=N_BOOT, seed=1)
    a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
    assert a["mean"] == pytest.approx(ref["mean"], abs=1e-12)
    assert a["lo"] == pytest.approx(ref["lo"], abs=ref["se"])
    assert a["hi"] == pytest.approx(ref["hi"], abs=ref["se"])


def test_strata_cover_family_horizon_and_pooled(cells):
    strata = ad.strata_of(cells)
    levels = [s.level for s in strata]
    assert levels.count("cell") == 3
    assert {(s.family, s.horizon) for s in strata if s.level == "family_horizon"} == {("A", "100"), ("A", "200"), ("B", "100")}
    assert {(s.family, s.horizon) for s in strata if s.level == "family"} == {("A", "all"), ("B", "all")}
    assert {(s.family, s.horizon) for s in strata if s.level == "horizon"} == {("all", "100"), ("all", "200")}
    assert levels.count("pooled") == 1


def test_main_table_rows_and_columns(cells, per_cell):
    strata = ad.strata_of(cells)
    full = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    assert len(full) == len(strata) * len(POLICIES)
    for col in ("regret", "regret_lo", "regret_hi", "q", "regret_disc", "regret_sel", "k_final",
                "search_frac", "cap_hit_frac", "n_rec", "n_singletons_final",
                "d_regret_vs_cp0", "d_regret_vs_cp0_lo", "d_regret_vs_cp0_hi", "d_regret_vs_cp0_win",
                "d_regret_vs_p3_star", "d_regret_vs_p3_star_cluster_lo"):
        assert col in full.columns, col
    pooled = full[(full["level"] == "pooled") & (full["policy"] == "phi_k16")].iloc[0]
    assert pooled["n_cells"] == 3 and pooled["n_envs"] == 3
    # R_T = R_disc + R_sel holds for the table means.
    assert pooled["regret"] == pytest.approx(pooled["regret_disc"] + pooled["regret_sel"], abs=1e-9)
    # The cluster-over-environments CI exists for the pooled block (3 environments) and
    # brackets the planted effect; it is absent at the cell level (one environment).
    assert pooled["d_regret_vs_cp0_cluster_lo"] <= PLANTED["phi_k16"] <= pooled["d_regret_vs_cp0_cluster_hi"]
    cell_rows = full[full["level"] == "cell"]
    assert set(cell_rows["cell"]) == {c.name for c in cells}
    assert cell_rows["d_regret_vs_cp0_cluster_lo"].isna().all()


def test_primary_contrasts_has_exactly_three_rows_per_stratum(cells):
    strata = ad.strata_of(cells)
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells, strata, N_BOOT)
    non_cell = [s for s in strata if s.level != "cell"]
    assert len(pc) == 3 * len(non_cell)
    for _, grp in pc.groupby(["level", "family", "horizon"]):
        assert list(grp["hypothesis"]) == ["H1a", "H1b", "H2"]
        assert list(grp["reference"]) == ["cp0", "p3_star", "phi_k16_quality"]
        assert (grp["policy"] == "phi_k16").all() and grp["pre_registered"].all()
        assert (grp["status"] == "ok").all()
    pooled = pc[pc["level"] == "pooled"].set_index("hypothesis")
    assert pooled.loc["H1a", "lo"] <= PLANTED["phi_k16"] <= pooled.loc["H1a", "hi"]
    assert pooled.loc["H1b", "lo"] <= PLANTED["phi_k16"] - PLANTED["p3_star"] <= pooled.loc["H1b", "hi"]
    assert pooled.loc["H2", "lo"] <= PLANTED["phi_k16"] - PLANTED["phi_k16_quality"] <= pooled.loc["H2", "hi"]
    assert np.isfinite(pooled.loc["H1a", "cluster_lo"]) and pooled.loc["H1a", "n_envs"] == 3


def _ragged_cells() -> list[ad.Cell]:
    """The three synthetic cells with `phi_k16` withdrawn from the (A, 200) one.

    The shape of Test D after Ruling 16: a policy that ran in the test, but not in every
    cell of it.
    """
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    dropped = next(c for c in cells if c.family == "A" and c.horizon == 200)
    del dropped.frames["phi_k16"]
    del dropped.groups["phi_k16"]
    return cells


def test_primary_contrasts_keep_three_rows_when_a_policy_is_missing():
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    strata = ad.strata_of(cells)
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells, strata, 200)
    pooled = pc[pc["level"] == "pooled"].set_index("hypothesis")
    assert len(pooled) == 3
    # phi_k16_quality was never deployed in this test: "not evaluated", not a null result.
    assert pooled.loc["H2", "status"] == "not_evaluated:phi_k16_quality"
    assert np.isnan(pooled.loc["H2", "delta"]) and not pooled.loc["H2", "evaluated"]
    assert pooled.loc["H1a", "status"] == "ok" and pooled.loc["H1a", "evaluated"]


def test_missing_label_names_the_absent_policy_not_the_reference():
    """M1: the label must name the policy that is actually absent from the stratum."""
    cells = _ragged_cells()
    strata = ad.strata_of(cells)
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells, strata, 200)
    # In the (A, 200) stratum cp0 and p3_star both ran; phi_k16 did not. Blaming the
    # reference there is the defect: every one of the three rows names phi_k16.
    gone = pc[(pc["level"] == "family_horizon") & (pc["family"] == "A") & (pc["horizon"] == "200")]
    assert len(gone) == 3
    assert set(gone[gone["hypothesis"] != "H2"]["status"]) == {"missing:phi_k16"}
    assert (~gone["evaluated"]).all()
    # Both absent (H2's reference was never deployed either) -> the "never deployed"
    # label wins and names it.
    assert gone[gone["hypothesis"] == "H2"]["status"].iloc[0] == "not_evaluated:phi_k16_quality"
    # Strata where it did run are untouched.
    assert (pc[(pc["level"] == "family_horizon") & (pc["horizon"] == "100")
               & (pc["hypothesis"] == "H1a")]["status"] == "ok").all()


def test_levels_are_pooled_over_the_common_cell_set(caplog):
    """B4: a level column must mean the same cells in every row of its stratum."""
    cells = _ragged_cells()
    with caplog.at_level("WARNING", logger="deploy.analyze"):
        strata = ad.strata_of(cells)
    assert any("ragged" in r.message for r in caplog.records)
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 200) for c in cells}
    pooled = next(s for s in strata if s.level == "pooled")
    assert len(pooled.cells) == 3 and len(pooled.common) == 2  # the (A, 200) cell is out
    common = set(pooled.common)

    # p3_star ran in all three cells but is levelled over the two common ones, so its
    # regret is comparable with phi_k16's; its own coverage is still reported.
    a = ad.aggregate(per_cell, pooled, "p3_star", "regret")
    b = ad.aggregate(per_cell, pooled, "phi_k16", "regret")
    assert a["n_cells"] == b["n_cells"] == 2
    assert a["n_cells_policy"] == 3 and b["n_cells_policy"] == 2
    assert not a["common_support"] and not b["common_support"]
    expected = float(np.mean([per_cell[n].mean[("p3_star", "regret")] for n in common]))
    assert a["mean"] == pytest.approx(expected, abs=1e-12)
    # The three-cell mean -- what a silent drop would have produced -- is a different number.
    all_cells = float(np.mean([per_cell[c.name].mean[("p3_star", "regret")] for c in cells]))
    assert abs(a["mean"] - all_cells) > 1e-6

    # The descriptive means travel with it, so no column of the row is over other cells.
    desc = ad.descriptive_for(per_cell, pooled, "p3_star")
    k_common = float(np.mean([per_cell[n].descriptive["p3_star"]["k_final"] for n in common]))
    assert desc["k_final"] == pytest.approx(k_common, abs=1e-12)

    # A paired contrast is already on the intersection; it says so rather than dropping
    # silently, and the planted effect still comes back.
    d = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
    assert d["n_cells"] == 2 and d["n_cells_policy"] == 2
    assert d["lo"] <= PLANTED["phi_k16"] <= d["hi"]
    main = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    row = main[(main["level"] == "pooled") & (main["policy"] == "p3_star")].iloc[0]
    assert row["n_cells"] == 2 and row["n_cells_policy"] == 3 and not row["common_support"]


def test_untuned_reference_cells_are_refused_and_flagged(caplog):
    """B1: an untuned placeholder baseline must not be pooled into a contrast against it."""
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    placeholder = next(c for c in cells if c.family == "B")  # the (B, 100) cell
    placeholder.untuned = frozenset({"p3_star"})
    strata = ad.strata_of(cells)
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 200) for c in cells}

    pooled = next(s for s in strata if s.level == "pooled")
    a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_p3_star")
    assert a["n_cells"] == 2 and a["n_cells_excluded_untuned"] == 1 and a["ref_tuned"]
    kept = [c.name for c in cells if c is not placeholder]
    expected = float(np.mean([per_cell[n].mean[("phi_k16", "d_regret_vs_p3_star")] for n in kept]))
    assert a["mean"] == pytest.approx(expected, abs=1e-12)
    # Against the other reference, nothing is excluded.
    assert ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")["n_cells"] == 3

    # The stratum that holds only untuned cells keeps them -- a descriptive row, flagged.
    only = next(s for s in strata if s.level == "family" and s.family == "B")
    b = ad.aggregate(per_cell, only, "phi_k16", "d_regret_vs_p3_star")
    assert b["n_cells"] == 1 and b["n_cells_excluded_untuned"] == 0 and not b["ref_tuned"]

    # The pre-registered path (H1b: phi_k16 vs p3_star) refuses the same cell, and the
    # flag reaches the table a reader opens.
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells, strata, 200)
    h1b = pc[(pc["level"] == "pooled") & (pc["hypothesis"] == "H1b")].iloc[0]
    assert h1b["n_cells"] == 2 and h1b["n_cells_excluded_untuned"] == 1 and h1b["status"] == "ok"
    h1b_b = pc[(pc["level"] == "family") & (pc["family"] == "B") & (pc["hypothesis"] == "H1b")].iloc[0]
    assert h1b_b["status"] == "untuned_reference" and not h1b_b["ref_tuned"]
    main = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    row = main[(main["level"] == "pooled") & (main["policy"] == "phi_k16")].iloc[0]
    assert row["d_regret_vs_p3_star_n_excluded_untuned"] == 1
    assert row["d_regret_vs_p3_star_n_cells"] == 2 and row["d_regret_vs_cp0_n_cells"] == 3
    # p3_star's own row says its constants were placeholders where it ran untuned.
    fam_b = main[(main["level"] == "family") & (main["family"] == "B") & (main["policy"] == "p3_star")].iloc[0]
    assert not fam_b["params_tuned"]
    assert main[(main["level"] == "family") & (main["family"] == "A")
                & (main["policy"] == "p3_star")].iloc[0]["params_tuned"]


def _cells_across_envs(n_envs: int, policies=("cp0", "p3_star", "phi_k16")) -> list[ad.Cell]:
    """`n_envs` one-cell environments, so a stratum's cluster bootstrap has n_envs = n."""
    out: list[ad.Cell] = []
    for i in range(n_envs):
        env = f"env{i}"
        name = f"{env}_T100_cap64"
        seed = 5_000 + 37 * i
        frames = {p: _episode_frame(name, env, "A", 100, p, 64, seed) for p in policies}
        out.append(ad.Cell(name=name, env_id=env, family="A", horizon=100, cap=64, base_seed=seed,
                           n=64, frames=frames, groups={p: GROUPS[p] for p in policies}))
    return out


def test_cluster_ci_is_flagged_degenerate_at_small_n_envs():
    """F1: below `CLUSTER_MIN_ENVS` the percentile cluster CI IS [min env, max env]."""
    assert ad.CLUSTER_MIN_ENVS == 4
    for n_envs in (2, 3):
        cells = _cells_across_envs(n_envs)
        per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, N_BOOT) for c in cells}
        pooled = next(s for s in ad.strata_of(cells) if s.level == "pooled")
        a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
        env_means = [per_cell[c.name].mean[("phi_k16", "d_regret_vs_cp0")] for c in cells]
        assert a["n_envs"] == n_envs and a["cluster_degenerate"] is True
        # The arithmetic the flag is about: the interval is the range of the env means.
        assert a["cluster_lo"] == pytest.approx(min(env_means), abs=1e-12)
        assert a["cluster_hi"] == pytest.approx(max(env_means), abs=1e-12)

    cells = _cells_across_envs(5)
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, N_BOOT) for c in cells}
    strata = ad.strata_of(cells)
    pooled = next(s for s in strata if s.level == "pooled")
    a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
    env_means = [per_cell[c.name].mean[("phi_k16", "d_regret_vs_cp0")] for c in cells]
    assert a["n_envs"] == 5 and a["cluster_degenerate"] is False
    assert a["cluster_lo"] > min(env_means) and a["cluster_hi"] < max(env_means)
    # A row with no cluster interval at all is not "degenerate", it is empty.
    cell_level = next(s for s in strata if s.level == "cell")
    assert not ad.aggregate(per_cell, cell_level, "phi_k16", "d_regret_vs_cp0")["cluster_degenerate"]

    # Every table that carries cluster bounds carries the flag beside them.
    cells3 = _cells_across_envs(3)
    per3 = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 400) for c in cells3}
    strata3 = ad.strata_of(cells3)
    main = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells3, per3, strata3)
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells3, strata3, 400)
    sc = ad.secondary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells3, per3, strata3)
    for ref in ("cp0", "p3_star"):
        assert f"d_regret_vs_{ref}_cluster_degenerate" in main.columns
    pooled_row = main[(main["level"] == "pooled") & (main["policy"] == "phi_k16")].iloc[0]
    assert pooled_row["d_regret_vs_cp0_cluster_degenerate"]
    pooled_pc = pc[(pc["level"] == "pooled") & pc["evaluated"]]
    assert len(pooled_pc) == 2 and pooled_pc["cluster_degenerate"].all()  # H1a, H1b (H2 absent)
    assert sc[sc["level"] == "pooled"]["cluster_degenerate"].all()
    # A row with no contrast at all (H2 here) has no interval, so it is not "degenerate".
    assert not pc[~pc["evaluated"]]["cluster_degenerate"].any()


def test_mark_untuned_baselines_reads_the_json_and_the_manifest(caplog):
    """The flag comes from `baseline_params.json` *or* the manifest's stamp -- either alone."""
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    by_horizon = {c.horizon for c in cells}
    assert by_horizon == {100, 200}
    # Tuned at T=100 only: every T=200 cell deployed p3_star on the placeholder.
    baseline_params = {
        "power": {}, "refine_after_init": {"100": 8},
        "p3_star": {"100": {"alpha": 0.5, "c": 2.0, "pooled_regret": 0.1}},
    }
    with caplog.at_level("WARNING", logger="deploy.analyze"):
        ad.mark_untuned_baselines(cells, {}, baseline_params)
    assert any("UNTUNED" in r.message for r in caplog.records)
    for c in cells:
        assert c.untuned == (frozenset({"p3_star"}) if c.horizon == 200 else frozenset())

    # With everything tuned, only the manifest's own stamp can flag a cell.
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    full = {"power": {}, "refine_after_init": {},
            "p3_star": {str(T): {"alpha": 0.5, "c": 2.0} for T in by_horizon}}
    target = cells[0]
    manifest = {("synthetic", target.name, "p3_star"):
                {"params": {"alpha": 0.5, "c": 1.0, "params_tuned": False}}}
    ad.mark_untuned_baselines(cells, manifest, full)
    assert cells[0].untuned == frozenset({"p3_star"})
    assert all(c.untuned == frozenset() for c in cells[1:])


def test_secondary_contrasts_exclude_the_pre_registered_pairs(cells, per_cell):
    strata = ad.strata_of(cells)
    sec = ad.secondary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    pairs = set(zip(sec["policy"], sec["reference"], strict=False))
    for _, p, r, _ in ad.PRIMARY_HYPOTHESES:
        assert (p, r) not in pairs
    assert ("phi_k16_quality", "phi_k16") in pairs and ("p3_star", "cp0") in pairs
    assert (~sec["pre_registered"]).all() and (sec["level"] != "cell").all()
    row = sec[(sec["policy"] == "phi_k16_quality") & (sec["reference"] == "phi_k16") & (sec["level"] == "pooled")].iloc[0]
    assert row["lo"] <= PLANTED["phi_k16_quality"] - PLANTED["phi_k16"] <= row["hi"]


def test_decomposition_table_sums(cells, per_cell):
    strata = ad.strata_of(cells)
    dec = ad.decomposition_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    assert len(dec) == len(strata) * len(POLICIES)
    assert np.allclose(dec["regret"], dec["regret_disc"] + dec["regret_sel"], atol=1e-9)
    assert np.allclose(dec["d_regret_vs_cp0"], dec["d_disc_vs_cp0"] + dec["d_sel_vs_cp0"], atol=1e-9)
    assert (dec["regret_disc_lo"] <= dec["regret_disc"]).all() and (dec["regret_disc"] <= dec["regret_disc_hi"]).all()


def test_recommender_independent_quantities_share_one_resample(cells):
    a = ad.compute_cell_stats(cells[0], RECOMMENDER_NAMES[0], 300)
    b = ad.compute_cell_stats(cells[0], RECOMMENDER_NAMES[1], 300)
    for q in ("regret_disc", "d_disc_vs_cp0"):
        assert np.array_equal(a.boot[("phi_k16", q)], b.boot[("phi_k16", q)])
    assert not np.array_equal(a.boot[("phi_k16", "regret")], b.boot[("phi_k16", "regret")])


def test_tau_curves_keep_heldout_horizon_rows_apart(cells, per_cell):
    strata = ad.strata_of(cells)
    full = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    ts = pd.DataFrame({
        "variant": "clock_quality_evidence_k16_noT1000", "tau": [0.5] * 4, "split": "val",
        "env": ["e1", "e2", "e1", "e2"], "T": [200, 200, 1000, 1000],
        "heldout_T": [False, False, True, True], "mean_regret": [0.10, 0.12, 0.30, 0.32],
        "mean_search_frac": 0.2,
    })
    tau = ad.tau_curves("synthetic", full, ts, {})
    val = tau[~tau["posthoc"]]
    assert len(val) == 2 and set(val["heldout_T"]) == {False, True}
    assert val[~val["heldout_T"]]["regret"].iloc[0] == pytest.approx(0.11)
    assert val[val["heldout_T"]]["regret"].iloc[0] == pytest.approx(0.31)


def test_recommender_sensitivity_and_cap_table(cells, per_cell):
    strata = ad.strata_of(cells)
    tabs = []
    for rec in RECOMMENDER_NAMES[:2]:
        pc = {c.name: ad.compute_cell_stats(c, rec, 200) for c in cells}
        full = ad.main_table("synthetic", rec, cells, pc, strata)
        tabs.append(full[full["level"] == "cell"])
    ranks, kendall = ad.recommender_sensitivity("synthetic", pd.concat(tabs, ignore_index=True))
    assert set(ranks["rank"].unique()) == {1.0, 2.0, 3.0, 4.0}
    best = ranks[(ranks["recommender"] == PRIMARY_RECOMMENDER) & (ranks["rank"] == 1.0)]
    assert (best["policy"] == "phi_k16").all()
    # The synthetic recommenders differ by a constant, so their rankings agree exactly.
    assert np.allclose(kendall["kendall_tau"], 1.0)
    assert (kendall["level"] == "pooled").sum() == 1

    manifest = {("synthetic", cells[0].name, "phi_k16"): {"counters": {"n_decisions": 40, "n_search_decided": 10,
                                                                        "n_committed_steps": 100, "n_guard_vetoes": 0,
                                                                        "n_cap_demoted": 2}}}
    cap = ad.cap_demotion_table("synthetic", cells, manifest)
    assert len(cap) == 3 * len(POLICIES)
    row = cap[(cap["cell"] == cells[0].name) & (cap["policy"] == "phi_k16")].iloc[0]
    assert row["frac_decisions_search"] == pytest.approx(0.25)
    assert np.isnan(cap[(cap["policy"] == "cp0")]["frac_decisions_search"]).all()
    assert (cap["t_cap_hit_q50"].dropna() >= 20).all()


def test_table_names():
    assert ad.table_name("main", "A", PRIMARY_RECOMMENDER) == "main_A_primary.csv"
    assert ad.table_name("main", "A", "lcb") == "main_A_lcb.csv"
    assert ad.table_name("offline_vs_deployed", "A") == "offline_vs_deployed.csv"
    assert ad.table_name("offline_vs_deployed", "smoke") == "offline_vs_deployed_smoke.csv"
    assert ad.table_name("onpolicy_parity", "A") == "onpolicy_parity.csv"


# ---- OOD statistics -------------------------------------------------------------------------------


def _gaussian_frame(rng: np.random.Generator, n: int, features: tuple[str, ...], shift: float = 0.0) -> pd.DataFrame:
    X = rng.standard_normal((n, len(features))) * np.linspace(0.5, 3.0, len(features)) + shift * np.linspace(0.5, 3.0, len(features))
    df = pd.DataFrame(X, columns=list(features))
    df["p_search_model"] = rng.uniform(size=n)
    return df


def test_ood_fraction_is_zero_in_distribution_and_one_when_shifted():
    rng = np.random.default_rng(0)
    features = tuple(fg.CLOCK[:6]) + tuple(fg.QUALITY[:4])
    corpus = _gaussian_frame(rng, 4000, features)
    same = _gaussian_frame(rng, 500, features)
    shifted = _gaussian_frame(rng, 500, features, shift=10.0)  # +10 SD on every feature

    r_same = ood.ood_for_policy(same, corpus, "same_horizon", features, p_corpus=rng.uniform(size=4000), seed=1)
    r_shift = ood.ood_for_policy(shifted, corpus, "same_horizon", features, p_corpus=rng.uniform(size=4000), seed=1)
    # Primary columns are clock-free (the four QUALITY columns here); the all-feature
    # versions are kept alongside and labelled as clock-confounded.
    for r in (r_same, r_shift):
        assert r.summary["knn_applicable"] is True and r.summary["n_features_nonclock"] == 4
        assert r.summary["clock_confounded"] is True
    assert r_same.summary["ood_frac"] < 0.10 and r_same.summary["ood_frac_all"] < 0.10
    assert r_shift.summary["ood_frac"] > 0.99 and r_shift.summary["ood_frac_all"] > 0.99
    assert r_same.summary["max_frac_outside"] < 0.05 and r_shift.summary["mean_frac_outside"] > 0.99
    assert abs(r_same.summary["max_abs_std_shift"]) < 0.3 and r_shift.summary["max_abs_std_shift"] > 9.0
    assert r_same.summary["max_ks"] < 0.15 and r_shift.summary["max_ks"] > 0.99
    assert 0.4 < r_same.summary["domain_auc"] < 0.6 and r_shift.summary["domain_auc"] > 0.99
    assert 0.4 < r_same.summary["domain_auc_all"] < 0.6 and r_shift.summary["domain_auc_all"] > 0.99
    assert 0.4 < r_same.summary["domain_auc_CLOCK"] < 0.6 and r_shift.summary["domain_auc_QUALITY"] > 0.99
    assert np.isnan(r_same.summary["domain_auc_HISTORY"])  # no HISTORY column in the list
    assert len(r_same.features) == len(features)
    assert set(r_same.features["group"]) == {"CLOCK", "QUALITY"}
    hist = r_same.score_hist
    assert len(hist) == 20 and hist["frac_on"].sum() == pytest.approx(1.0) and hist["frac_corpus"].sum() == pytest.approx(1.0)
    flags = ood.flag_cells(pd.DataFrame([{"cell": "c", "policy": "p", **r_same.summary},
                                         {"cell": "c", "policy": "q", **r_shift.summary}]))
    assert list(flags["policy"]) == ["q"] and (flags["threshold"] == 0.10).all()


def test_clock_only_policy_gets_coverage_rows_but_no_knn_fraction():
    """A clock-only feature list is the logging schedule itself: kNN is not applicable."""
    rng = np.random.default_rng(5)
    features = tuple(fg.CLOCK)
    corpus = _gaussian_frame(rng, 2000, features)
    on = _gaussian_frame(rng, 200, features, shift=10.0)
    r = ood.ood_for_policy(on, corpus, "same_horizon", features, p_corpus=np.empty(0), seed=0)
    assert r.summary["knn_applicable"] is False and r.summary["n_features_nonclock"] == 0
    assert np.isnan(r.summary["ood_frac"]) and np.isnan(r.summary["domain_auc"])
    # The clock-confounded all-feature numbers are still reported, labelled as such.
    assert r.summary["ood_frac_all"] > 0.99 and r.summary["clock_confounded"] is True
    assert len(r.features) == len(fg.CLOCK) and (r.features["group"] == "CLOCK").all()
    assert r.features["frac_outside"].min() > 0.99
    # ... and such a policy never flags, because the flag is the clock-free fraction.
    flags = ood.flag_cells(pd.DataFrame([{"cell": "c", "policy": "phi_k16_clock", **r.summary}]))
    assert flags.empty


def test_constant_corpus_columns_do_not_break_standardization():
    rng = np.random.default_rng(2)
    features = ("f_T", "f_t", "f_K")
    corpus = _gaussian_frame(rng, 1000, features)
    corpus["f_T"] = 100.0  # constant within a horizon, as in the real corpus
    on = _gaussian_frame(rng, 100, features)
    on["f_T"] = 100.0
    st = ood.Standardizer.fit(corpus.loc[:, list(features)].to_numpy(), features)
    assert st.usable.tolist() == [False, True, True]
    r = ood.ood_for_policy(on, corpus, "same_horizon", features, p_corpus=np.empty(0), seed=0)
    # Clock-only list: the primary kNN is not applicable; the all-feature (clock-confounded)
    # kNN drops the constant column and still yields a finite fraction.
    assert r.summary["knn_applicable"] is False and np.isnan(r.summary["ood_frac"])
    assert r.summary["n_features_standardizable_all"] == 2 and np.isfinite(r.summary["ood_frac_all"])
    row = r.features.set_index("feature").loc["f_T"]
    assert np.isnan(row["std_mean_shift"]) and row["frac_outside"] == 0.0


def test_parity_table_counts_failures_at_the_m1_tolerances():
    columns = ["f_t", "f_log_e_pair"]
    item = ood.SnapshotItem("t", "c", "p", "e", "A", 100, 64, {}, "", "model", None, 0.5)
    scalar = np.array([[1.0, 0.0], [2.0, 5.0]])
    vec = np.array([[1.0 + 5e-6, 0.0], [2.0 + 3e-5, 5.0 + 5e-4]])
    tab = ood.parity_table(scalar, vec, columns, item).set_index("column")
    assert tab.loc["f_t", "n_fail"] == 1 and tab.loc["f_t", "max_abs_diff"] == pytest.approx(3e-5)
    assert tab.loc["f_log_e_pair", "n_fail"] == 0  # 5e-4 is inside the float32 log-e tolerance
    assert ood.parity_failures(tab.reset_index()) == ["f_t"]
    summary = ood.parity_summary(tab.reset_index())
    assert summary.set_index("column").loc["f_t", "passed"] is np.False_ or not summary.set_index("column").loc["f_t", "passed"]


# ---- real on-policy snapshots: parity gate, decisions, reservoir diagnostics -------------------


CLOCK = list(fg.FEATURE_SETS["clock"])


def _clock_row(t: int, horizon: int, k: int) -> list[float]:
    remaining = horizon - t
    return [t, horizon, remaining, remaining / horizon, t / horizon, k, k / t, k / horizon,
            np.log(max(t, 1)), np.log(k), k / np.sqrt(max(t, 1))]


def _fit_clock_pipeline(seed: int = 0, n_rows: int = 2000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_rows):
        horizon = int(rng.choice([40, 60, 100]))
        t = int(rng.integers(2, horizon))
        k = int(rng.integers(1, min(t, 64) + 1))
        row = _clock_row(t, horizon, k)
        rows.append(row)
        labels.append(row[3] - 0.15 * row[10] + 0.3 * rng.normal() > 0.35)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(
        np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    )


@pytest.fixture(scope="module")
def logged_run(tmp_path_factory):
    """A short real cell under a tiny learned CLOCK policy, with snapshots logged."""
    tmp = tmp_path_factory.mktemp("logged")
    T, M = 40, 12
    save_model(tmp / "clock_k16.joblib", {"pipeline": _fit_clock_pipeline(), "features": CLOCK, "k": 4,
                                          "tau": 0.45, "meta": {"variant": "clock_k16"}})
    env_spec = {"type": "beta", "params": {"a": 5.0, "b": 2.0}}
    spec = CellSpec(env_id="beta_good_common", env_spec=env_spec, horizon=T, cap=16, base_seed=777_000_101, n_replicates=M)
    table = CSTable.load_or_build(T)
    policy = make_policy("model", horizon=T, n_replicates=M, table=table,
                         params={"artifact": load_model(tmp / "clock_k16.joblib"), "tau": 0.45})
    policy.name = "phi_k16_clock"
    times = tuple(sorted({max(2, round(g * T / 8)) for g in range(1, 9)}))
    res = run_cell(spec, policy, table=table, dynamics_grid=0, log_states=LogSpec(times=times, replicates=M), policy_seed=1)
    snaps_path = tmp / "phi_k16_clock.pkl"
    with open(snaps_path, "wb") as fh:
        pickle.dump(res.snapshots, fh)
    item = ood.SnapshotItem(test="t", cell="beta_good_common_T40_cap16", policy="phi_k16_clock", env_id="beta_good_common",
                            family="A", horizon=T, cap=16, env_spec=env_spec, snapshots_path=str(snaps_path),
                            kind="model", artifact_path=str(tmp / "clock_k16.joblib"), tau=0.45)
    pair_cache = tmp_path_factory.mktemp("pairwise")
    result = ood.snapshot_pass(item, table=table, pairwise_table=get_pairwise_table(T, cache_dir=pair_cache))
    return {"item": item, "result": result, "episode": res, "spec": spec, "times": times}


def test_snapshot_pass_parity_gate_on_real_on_policy_states(logged_run):
    r = logged_run["result"]
    assert len(r.features) == len(logged_run["times"]) * logged_run["spec"].n_replicates
    assert set(r.parity["column"]) == set(fg.ALL_DEPLOYABLE)
    assert ood.parity_failures(r.parity) == [], r.parity.sort_values("max_abs_diff").tail(5)
    assert (r.parity["n_rows"] == len(r.features)).all()
    assert r.parity.set_index("column").loc["f_log_e_pair", "max_abs_diff"] < ood.PARITY_LOGE_ATOL
    assert r.policy_features == tuple(CLOCK)
    # The model's own P(SEARCH) is recomputed on the scalar features and is a probability.
    p = r.features["p_search_model"].to_numpy()
    assert np.isfinite(p).all() and (0.0 <= p).all() and (p <= 1.0).all() and p.std() > 0.0
    assert (r.features["est_I_hat"] >= 0.0).all() and (r.features["tau"] == 0.45).all()
    for col in ood.ORACLE_COLUMNS:
        assert col in r.features.columns


def test_decision_lookup_matches_the_harness_history(logged_run):
    """The action at a snapshot's time is the next entry of a later snapshot's history."""
    r = logged_run["result"]
    T = logged_run["spec"].horizon
    snaps = ood.load_snapshots(logged_run["item"].snapshots_path)
    seqs = ood.decision_sequences(snaps)
    final = {int(s.meta["replicate"]): s for s in snaps if s.t == T}
    feats = r.features
    assert feats.loc[feats["t"] == T, "decision_search"].isna().all()  # no step at t = T
    earlier = feats[feats["t"] < T]
    assert earlier["decision_search"].notna().all()
    for _, row in earlier.iterrows():
        hist = final[int(row["replicate"])].meta["history"].decisions
        n0 = T - hist.size
        assert row["decision_search"] == float(hist[int(row["t"]) - n0])
        assert ood.decision_at(seqs, int(row["replicate"]), int(row["t"])) == row["decision_search"]
    # The logged actions are the post-cap actions, whose per-episode share is search_frac.
    total = np.array([seqs[m][1].mean() for m in sorted(seqs)])
    assert np.allclose(total, logged_run["episode"].search_frac)


def test_reservoir_diagnostics_on_real_states(logged_run):
    r = logged_run["result"]
    reservoir = build_reservoir(logged_run["item"].env_spec)
    feats = ra.with_oracle_columns(r.features, reservoir)
    # Oracle I_t at c equals the Beta closed form E[(X - c)_+] for a Beta reservoir.
    c = feats["oracle_best_true_mu"].to_numpy()
    closed = beta_excess_mean(5.0, 2.0, c)
    assert np.allclose(feats["oracle_I_t"], closed, atol=2e-4)
    assert (feats["oracle_I_t"] >= 0).all() and (feats["oracle_p_t"] <= 1).all()
    meta = {"test": "t", "cell": "c", "policy": "phi_k16_clock", "env_id": "beta_good_common", "family": "A", "horizon": 40, "cap": 16}
    offline = pd.DataFrame({"scope": ["T=40", "all"], "score": ["oracle_I_t", "est_I_hat"],
                            "kind": ["oracle", "observable"], "n_rows": [1, 1],
                            "auc_raw": [0.9, 0.6], "auc_oriented": [0.9, 0.6], "sign": [1.0, 1.0]})
    row = ra.reservoir_row(feats, meta, offline)
    assert row["n_decisions"] == int(feats["decision_search"].notna().sum())
    assert 0.0 < row["search_rate"] < 1.0
    for key in ("auc_oracle_I_for_decision", "auc_model_p_for_decision", "spearman_est_I_hat_oracle_I"):
        assert np.isfinite(row[key]), key
    assert -1.0 <= row["corr_decision_oracle_I"] <= 1.0
    assert row["offline_label_auc_oracle_I_t"] == 0.9 and row["offline_label_auc_est_I_hat"] == 0.6
    assert np.isnan(row["offline_label_auc_oracle_p_new_beats_best_true"])
    bins = ra.decision_by_oracle_bins(feats, meta)
    assert 1 <= len(bins) <= ra.N_I_BINS and bins["n"].sum() == row["n_decisions"]
    assert (bins["I_lo"] <= bins["I_hi"]).all()


def test_oracle_tail_integral_matches_closed_form_for_beta():
    reservoir = build_reservoir({"type": "beta", "params": {"a": 2.0, "b": 5.0}})
    c = np.linspace(0.0, 1.0, 11)
    got = ra.oracle_tail_integral(reservoir, c, n_grid=2000)
    assert np.allclose(got, beta_excess_mean(2.0, 5.0, c), atol=1e-4)
    assert got[0] == pytest.approx(reservoir.mean(), abs=1e-4) and got[-1] == 0.0


# ---- CLI end to end on synthetic parquet -------------------------------------------------------------


def _write_synthetic_run(out: Path, cells: list[ad.Cell]) -> None:
    for c in cells:
        for policy, frame in c.frames.items():
            path = out / "episodes" / "smoke" / c.name / f"{policy}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            dyn = out / "dynamics" / "smoke" / c.name / f"{policy}.npz"
            dyn.parent.mkdir(parents=True, exist_ok=True)
            grid = np.linspace(0, c.horizon, 11)
            np.savez(dyn, t=grid, K_t=np.linspace(2, 20, 11), best_discovered=np.linspace(0.7, 0.9, 11),
                     best_posterior_mean=np.linspace(0.6, 0.85, 11), q_primary=np.linspace(0.6, 0.88, 11),
                     n_eliminated=np.linspace(0, 3, 11), search_rate=np.full(11, 0.2), herfindahl=np.full(11, 0.3))
    (out / "manifest_smoke.jsonl").write_text("")


def test_cli_refuses_main_tables_without_the_parity_gate(tmp_path, cells):
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    result = ad.main(["--test", "smoke", "--out-dir", str(out), "--n-boot", "200", "--workers", "1"])
    assert result["gate_ok"] is False and result["n_states"] == 0
    assert not (out / "tables" / "main_smoke_primary.csv").exists()

    result = ad.main(["--test", "smoke", "--out-dir", str(out), "--n-boot", "200", "--workers", "1",
                      "--allow-unverified", "--recommender", PRIMARY_RECOMMENDER])
    tables = out / "tables"
    for name in ("main_smoke_primary.csv", "cells_smoke_primary.csv", "primary_contrasts_smoke_primary.csv",
                 "secondary_contrasts_smoke_primary.csv", "decomposition_smoke_primary.csv",
                 "offline_vs_deployed_smoke.csv", "surrogate_validity_smoke.csv", "tau_curves_smoke.csv",
                 "recommender_sensitivity_smoke.csv", "recommender_kendall_smoke.csv",
                 "dynamics_smoke.csv", "cap_demotion_smoke.csv"):
        assert (tables / name).exists(), name
    main = pd.read_csv(tables / "main_smoke_primary.csv")
    assert set(main["level"]) == {"family_horizon", "family", "horizon", "pooled"}
    pc = pd.read_csv(tables / "primary_contrasts_smoke_primary.csv")
    assert len(pc) == 3 * main.groupby(["level", "family", "horizon"]).ngroups
    dyn = pd.read_csv(tables / "dynamics_smoke.csv")
    assert set(dyn["metric"]) == {"K_t", "best_discovered", "best_posterior_mean", "q_primary", "n_eliminated", "search_rate", "herfindahl"}
    assert dyn["t_frac"].between(0, 1).all()


def test_cli_refuses_main_tables_when_a_parity_column_fails(tmp_path, cells, monkeypatch):
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    tables = out / "tables"
    tables.mkdir()
    argv = ["--test", "smoke", "--out-dir", str(out), "--n-boot", "100", "--workers", "1",
            "--recommender", PRIMARY_RECOMMENDER]

    # (a) --skip-snapshots reuses a parity table on disk: a failing column refuses.
    pd.DataFrame({"column": ["f_t", "f_K"], "n_rows": [5000, 5000], "max_abs_diff": [0.0, 0.3],
                  "n_fail": [0, 7], "n_items": [1, 1], "rtol": 1e-5, "atol": 1e-6,
                  "passed": [True, False]}).to_csv(tables / "onpolicy_parity_smoke.csv", index=False)
    result = ad.main(argv + ["--skip-snapshots"])
    assert result["gate_ok"] is False
    assert not (tables / "main_smoke_primary.csv").exists()

    # (b) A fresh diagnostics pass whose scoring reports a failing column: refuses too,
    # and writes the parity tables so the failure is on record.
    (out / "snapshots").mkdir()
    (out / "snapshots" / "x.pkl").write_bytes(pickle.dumps([]))
    cell = cells[0]
    (out / "manifest_smoke.jsonl").write_text(json.dumps({
        "test": "smoke", "cell": cell.name, "policy": "phi_k16", "group": "learned",
        "snapshots": str(out / "snapshots" / "x.pkl"), "parquet": str(out / "episodes" / "smoke" / cell.name / "phi_k16.parquet"),
        "params": {"artifact": "unused.joblib", "tau": 0.6}, "counters": {},
    }) + "\n")
    item = ood.SnapshotItem("smoke", cell.name, "phi_k16", cell.env_id, cell.family, cell.horizon, 64, {},
                            str(out / "snapshots" / "x.pkl"), "model", "unused.joblib", 0.6)
    scalar = np.zeros((1200, len(fg.ALL_DEPLOYABLE)))
    vec = scalar.copy()
    vec[:, list(fg.ALL_DEPLOYABLE).index("f_K")] = 1.0  # one column off by 1.0 on every row
    parity = ood.parity_table(scalar, vec, list(fg.ALL_DEPLOYABLE), item)
    meta = {"test": "smoke", "cell": cell.name, "policy": "phi_k16", "env_id": cell.env_id,
            "family": cell.family, "horizon": cell.horizon, "cap": 64}

    def fake_diag(d):
        return {"cell": d.item.cell, "policy": d.item.policy, "n_states": 1200, "seconds": 0.0,
                "parity": parity, "ood_summary": {**meta, "ood_frac": 0.0}, "ood_features": pd.DataFrame(),
                "score_hist": pd.DataFrame(), "reservoir_row": dict(meta), "reservoir_bins": pd.DataFrame()}

    monkeypatch.setattr(ad, "run_diag_item", fake_diag)
    monkeypatch.setattr(ad, "prebuild_tables", lambda cells: None)
    (tables / "onpolicy_parity_smoke.csv").unlink()
    result = ad.main(argv + ["--no-oof"])
    assert result["gate_ok"] is False and result["n_states"] == 1200
    assert not (tables / "main_smoke_primary.csv").exists()
    written = pd.read_csv(tables / "onpolicy_parity_smoke.csv").set_index("column")
    assert bool(written.loc["f_K", "passed"]) is False and written.loc["f_K", "n_fail"] == 1200
    assert written.drop(index="f_K")["passed"].all()
    # The same failing table on disk refuses again under --skip-snapshots, and
    # --allow-unverified does not override a *failed* gate (only an absent one).
    assert ad.main(argv + ["--skip-snapshots", "--allow-unverified"])["gate_ok"] is False
    assert not (tables / "main_smoke_primary.csv").exists()


def test_cli_single_recommender_keeps_cross_recommender_tables(tmp_path, cells):
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    tables = out / "tables"
    base = ["--test", "smoke", "--out-dir", str(out), "--n-boot", "100", "--workers", "1", "--allow-unverified"]
    ad.main(base)
    kendall_all = pd.read_csv(tables / "recommender_kendall_smoke.csv")
    ovd_all = pd.read_csv(tables / "offline_vs_deployed_smoke.csv")
    assert len(kendall_all) > 0 and set(ovd_all["recommender"]) == set(RECOMMENDER_NAMES)
    # A single-recommender re-run must rebuild the cross-recommender tables from every
    # per-recommender table on disk, not shrink them to one recommender.
    ad.main(base + ["--recommender", "lcb"])
    kendall_one = pd.read_csv(tables / "recommender_kendall_smoke.csv")
    ovd_one = pd.read_csv(tables / "offline_vs_deployed_smoke.csv")
    ranks = pd.read_csv(tables / "recommender_sensitivity_smoke.csv")
    assert len(kendall_one) == len(kendall_all) and set(ovd_one["recommender"]) == set(RECOMMENDER_NAMES)
    assert set(ranks["recommender"]) == set(RECOMMENDER_NAMES)
    pd.testing.assert_frame_equal(kendall_one, kendall_all)


# ---- figures ---------------------------------------------------------------------------------------------


def test_figures_render_on_synthetic_tables(tmp_path, cells, per_cell):
    strata = ad.strata_of(cells)
    full = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    main = full[full["level"] != "cell"]
    dec = ad.decomposition_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    out = tmp_path / "figures"
    policies = list(POLICIES)

    frame = mf.fig_regret_vs_T(main, policies, out, "regret")
    assert len(frame) == 3 * len(POLICIES)
    mf.fig_decomposition(dec, policies, out, "decomp")

    offline = pd.DataFrame({"variant": ["clock_quality_evidence_k16", "clock_quality_k16"], "grouping": "meta_env",
                            "auc": [0.7, 0.65], "bal_acc_tau_off": [0.62, 0.6], "tau_off": [0.6, 0.6],
                            "bal_acc_05": [0.6, 0.58], "k": [16, 16], "feature_set": ["cqe", "cq"], "n_features": [62, 35]})
    ovd, validity = ad.offline_vs_deployed("synthetic", main, offline)
    assert set(ovd["policy"]) >= {"phi_k16", "phi_k16_quality"}
    assert ovd.loc[ovd["policy"] == "phi_k16", "oof_auc_env"].iloc[0] == 0.7
    mf.fig_offline_vs_deployed(ovd, validity, out, "ovd")

    ts = pd.DataFrame({"variant": "clock_quality_evidence_k16", "tau": np.repeat([0.3, 0.5, 0.7], 2), "split": "val",
                       "env": ["e1", "e2"] * 3, "T": 100, "mean_regret": [0.1, 0.12, 0.09, 0.11, 0.095, 0.115],
                       "mean_search_frac": 0.2})
    manifest = {("synthetic", cells[0].name, "phi_k16"): {"params": {"tau": 0.6}}}
    tau = ad.tau_curves("synthetic", main, ts, manifest)
    assert (tau[~tau["posthoc"]]["n_cells"] == 2).all() and tau[tau["posthoc"]]["tau"].iloc[0] == 0.6
    mf.fig_tau_curves(tau, out, "tau")

    dyn_rows = []
    for c in cells:
        for p in POLICIES:
            for m, _ in mf.DYNAMICS_METRICS:
                dyn_rows.append(pd.DataFrame({"test": "s", "cell": c.name, "family": c.family, "horizon": c.horizon, "policy": p,
                                              "group": GROUPS[p], "t": np.linspace(0, c.horizon, 6), "t_frac": np.linspace(0, 1, 6),
                                              "metric": m, "value": np.linspace(0, 1, 6)}))
    mf.fig_dynamics(pd.concat(dyn_rows, ignore_index=True), policies, "A", out, "dyn")

    feats = pd.DataFrame({"cell": np.repeat([c.name for c in cells], 3), "policy": "phi_k16",
                          "feature": ["f_t", "f_K", "f_best_mean"] * 3, "frac_outside": np.linspace(0, 0.3, 9)})
    summ = pd.DataFrame({"cell": [c.name for c in cells] * 2, "policy": ["phi_k16"] * 3 + ["phi_k16_quality"] * 3,
                         "ood_frac": np.linspace(0, 0.6, 6)})
    mf.fig_ood_heatmap(feats, summ, "phi_k16", out, "ood")

    cap = full[full["level"] == "cell"].copy()
    cap["cap"] = 32
    cap2 = cap.copy()
    cap2["cap"] = 64
    mf.fig_cap_sweep(pd.concat([cap, cap2], ignore_index=True), policies, out, "cap")

    bins = pd.DataFrame({"cell": np.repeat([c.name for c in cells], 10), "family": np.repeat([c.family for c in cells], 10),
                         "policy": "phi_k16", "bin": list(range(10)) * 3, "I_lo": 0.0, "I_hi": 0.1, "I_mean": 0.05,
                         "n": 10, "search_rate": np.tile(np.linspace(0, 1, 10), 3), "model_p_mean": 0.5})
    res = pd.DataFrame({"cell": [c.name for c in cells], "family": [c.family for c in cells], "horizon": 100,
                        "policy": "phi_k16", "auc_oracle_I_for_decision": [0.6, 0.7, np.nan], "search_rate": 0.2})
    mf.fig_reservoir(bins, res, ["phi_k16"], out, "res")

    for stem in ("regret", "decomp", "ovd", "tau", "dyn", "ood", "cap", "res"):
        for ext in ("png", "pdf", "csv"):
            assert (out / f"{stem}.{ext}").exists(), (stem, ext)


def test_direct_labels_merge_ties_instead_of_fanning_them_into_a_ranking():
    """F2: four policies with the same regret must not be drawn as a ranked column."""
    y = 0.054819
    items = [(1000.0, y, "always search (P0)", "#d55181"), (1000.0, y, "P3*", "#199e70"),
             (1000.0, y, "Φ16 (P9)", "#3987e5"), (1000.0, y, "Φ16 quality-only (P7)", "#c98500"),
             (1000.0, 0.070, "cp0", "#d95926")]
    tols = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    groups = mf._tie_groups(items, tols)
    assert sorted(len(g) for g in groups) == [1, 4]

    fig, ax = mf.plt.subplots()
    ax.plot([100, 1000], [0.09, y])
    mf._direct_labels(ax, items, tols)
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 2, texts  # one merged label for the tie, one for cp0
    tied = next(t for t in texts if "P3*" in t)
    for name in ("always search (P0)", "P3*", "Φ16 (P9)", "Φ16 quality-only (P7)"):
        assert name in tied
    assert tied.count("\n") == 3 and tied.count("= ") == 3  # a = b = c = d, stacked
    assert "cp0" not in tied
    # Every member still gets its own marker at its own true y: no invented offsets.
    ties_at_y = [ln for ln in ax.lines if list(ln.get_ydata()) == [y]]
    assert len(ties_at_y) == 4
    mf.plt.close(fig)

    # A separation the CI can resolve is still labelled separately.
    apart = [(1000.0, 0.05, "a", "#fff"), (1000.0, 0.06, "b", "#fff")]
    assert [len(g) for g in mf._tie_groups(apart, [1e-4, 1e-4])] == [1, 1]
    # ... and with no tolerance given, only exact ties merge (the smoke figure's case).
    assert [len(g) for g in mf._tie_groups(apart, [0.0, 0.0])] == [1, 1]
    same = [(1.0, 0.062776, "a", "#fff"), (1.0, 0.062776, "b", "#fff")]
    assert [len(g) for g in mf._tie_groups(same, [0.0, 0.0])] == [2]


def test_decomposition_caption_matches_the_rendered_brightness():
    """F3: the caption said R_sel was the 'light' segment; it is drawn at half the alpha."""
    assert mf.DISC_ALPHA > mf.SEL_ALPHA
    caption = mf.DECOMP_CAPTION
    assert "solid" not in caption and "(light)" not in caption
    assert caption.index("R_disc") < caption.index("bright") < caption.index("R_sel")
    assert "dim" in caption[caption.index("R_sel"):]
    disc_key, sel_key = mf.DECOMP_KEY
    assert disc_key.startswith("R_disc") and "bright" in disc_key
    assert sel_key.startswith("R_sel") and "dim" in sel_key


def test_cap_sweep_marks_the_training_support_boundary(tmp_path, cells, per_cell):
    """M3: two of the cap sweep's four points are extrapolation; the panel must say so."""
    strata = ad.strata_of(cells)
    full = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    base = full[full["level"] == "cell"].copy()
    frames = []
    for c in (32, mf.TRAINING_CAP, 128):
        g = base.copy()
        g["cap"] = c
        frames.append(g)
    out = tmp_path / "figures"
    mf.fig_cap_sweep(pd.concat(frames, ignore_index=True), ["phi_k16", "p3_star"], out, "cap")
    for ext in ("png", "pdf", "csv"):
        assert (out / f"cap.{ext}").exists()

    # The boundary itself: a shaded out-of-support region, a hairline at the cap, and a
    # label -- and nothing at all when the sweep stays inside the training support.
    fig, ax = mf.plt.subplots()
    ax.plot([32, mf.TRAINING_CAP, 128], [0.10, 0.09, 0.11])
    mf._log_x(ax, [32, mf.TRAINING_CAP, 128])
    n_patches, n_lines = len(ax.patches), len(ax.lines)
    assert mf._mark_training_support(ax, annotate=True) is True
    assert len(ax.patches) == n_patches + 1 and len(ax.lines) == n_lines + 1
    assert float(ax.lines[-1].get_xdata()[0]) == float(mf.TRAINING_CAP)
    assert ax.get_xlim()[1] > mf.TRAINING_CAP
    assert any(str(mf.TRAINING_CAP) in t.get_text() for t in ax.texts)
    mf.plt.close(fig)

    fig, ax = mf.plt.subplots()
    ax.plot([16, 32, mf.TRAINING_CAP], [0.10, 0.09, 0.11])
    mf._log_x(ax, [16, 32, mf.TRAINING_CAP])
    ax.set_xlim(16, mf.TRAINING_CAP)
    assert mf._mark_training_support(ax) is False
    assert not ax.patches and not ax.texts
    mf.plt.close(fig)


def test_figures_skip_when_no_requested_policy_is_present(tmp_path, cells, per_cell):
    """No requested policy in the table: a warning and no file, never an empty legend crash."""
    strata = ad.strata_of(cells)
    full = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    dec = ad.decomposition_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    out = tmp_path / "figures"
    absent = ["not_a_policy"]
    dyn = pd.DataFrame({"test": "s", "cell": cells[0].name, "family": "A", "horizon": 100, "policy": "cp0",
                        "group": "reference", "t": [0.0, 50.0], "t_frac": [0.0, 0.5], "metric": "K_t", "value": [2.0, 5.0]})
    bins = pd.DataFrame({"cell": [cells[0].name], "family": ["A"], "policy": ["phi_k16"], "bin": [0], "I_lo": 0.0,
                         "I_hi": 0.1, "I_mean": 0.05, "n": 10, "search_rate": 0.2, "model_p_mean": 0.5})
    res = pd.DataFrame({"cell": [cells[0].name], "family": ["A"], "horizon": 100, "policy": ["phi_k16"],
                        "auc_oracle_I_for_decision": [0.6], "search_rate": [0.2]})
    cap = full[full["level"] == "cell"].assign(cap=32)
    mf.fig_regret_vs_T(full, absent, out, "regret")
    mf.fig_decomposition(dec, absent, out, "decomp")
    mf.fig_dynamics(dyn, absent, "A", out, "dyn")
    mf.fig_cap_sweep(cap, absent, out, "cap")
    mf.fig_reservoir(bins, res, absent, out, "res")
    assert not out.exists() or not any(out.iterdir())


def test_decomposition_tolerates_a_degenerate_bootstrap(tmp_path, cells, per_cell):
    """A CI bound a float epsilon on the wrong side of the mean must not be a negative error bar."""
    strata = ad.strata_of(cells)
    dec = ad.decomposition_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    dec = dec.copy()
    dec["regret_lo"] = dec["regret"] + 1e-12
    dec["regret_hi"] = dec["regret"] - 1e-12
    out = tmp_path / "figures"
    frame = mf.fig_decomposition(dec, list(POLICIES), out, "decomp")
    assert len(frame) and (out / "decomp.png").exists()
