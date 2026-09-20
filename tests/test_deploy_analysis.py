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
import run_deployment as rd  # noqa: E402

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
    # Three environments is below `CLUSTER_MIN_ENVS`, so no interval is emitted at all:
    # the range ships under its own names and brackets the planted effect (ruling 21).
    assert np.isnan(pooled["d_regret_vs_cp0_cluster_lo"])
    assert np.isnan(pooled["d_regret_vs_cp0_cluster_hi"])
    assert pooled["d_regret_vs_cp0_cluster_degenerate"]
    assert (pooled["d_regret_vs_cp0_cluster_min_env"] <= PLANTED["phi_k16"]
            <= pooled["d_regret_vs_cp0_cluster_max_env"])
    cell_rows = full[full["level"] == "cell"]
    assert set(cell_rows["cell"]) == {c.name for c in cells}
    for col in ("cluster_lo", "cluster_hi", "cluster_min_env", "cluster_max_env"):
        assert cell_rows[f"d_regret_vs_cp0_{col}"].isna().all(), col
    assert not full["stratum_empty_common_support"].any()


def _cap_sweep_cells() -> list[ad.Cell]:
    """One environment and horizon at four caps -- the shape of the cap sweep."""
    out: list[ad.Cell] = []
    for i, cap in enumerate((32, 64, 128, 200)):
        name = f"beta_good_common_T200_cap{cap}"
        seed = 30_000 + i
        frames = {p: _episode_frame(name, "beta_good_common", "A", 200, p, 64, seed)
                  for p in POLICIES}
        out.append(ad.Cell(name=name, env_id="beta_good_common", family="A", horizon=200,
                           cap=cap, base_seed=seed, n=64, frames=frames,
                           groups={p: GROUPS[p] for p in POLICIES}))
    return out


def test_a_test_with_one_cap_keeps_the_strata_it_always_had(cells):
    """Adding `cap` to `Stratum` must not move a single-cap test's rows."""
    strata = ad.strata_of(cells)
    assert {s.cap for s in strata if s.level != "cell"} == {"all"}
    assert {s.cap for s in strata if s.level == "cell"} == {"64"}
    assert [(s.level, s.family, s.horizon) for s in strata] == [
        (s.level, s.family, s.horizon) for s in strata
    ]
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 200) for c in cells}
    main = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    non_cell = main[main["level"] != "cell"]
    assert set(non_cell["cap"]) == {"all"}
    assert set(main[main["level"] == "cell"]["cap"]) == {64}


def test_strata_split_by_cap_when_a_test_varies_it():
    """4.5: `main_cap_primary.csv` pooled four caps into one family_horizon row.

    The constants are tuned at one cap, so the four caps are four different comparators;
    pooling them left a NaN `cap` column on a row that mixed all of them.
    """
    cells = _cap_sweep_cells()
    strata = ad.strata_of(cells)
    fh = [s for s in strata if s.level == "family_horizon"]
    assert len(fh) == 4
    assert {s.cap for s in fh} == {"32", "64", "128", "200"}
    for s in fh:
        assert len(s.cells) == 1 and s.cells[0].endswith(f"cap{s.cap}")
    # The levels that pool horizons or families still pool caps, and say so.
    assert {s.cap for s in strata if s.level == "pooled"} == {"all"}

    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 200) for c in cells}
    main = ad.main_table("cap", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    rows = main[(main["level"] == "family_horizon") & (main["policy"] == "phi_k16")]
    assert len(rows) == 4 and set(rows["cap"]) == {"32", "64", "128", "200"}
    assert (rows["n_cells"] == 1).all()
    cov = ad.strata_coverage("cap", cells, strata)
    assert set(cov[cov["level"] == "family_horizon"]["cap"]) == {"32", "64", "128", "200"}


def test_mark_untuned_baselines_flags_a_cell_deployed_at_the_wrong_cap(caplog):
    """The false assertion 4.5 is about: params_tuned = True at a cap never tuned."""
    cells = _cap_sweep_cells()
    baseline = {
        "meta": {"cap": 64},
        "power": {"0.5": {"200": 0.7}},
        "p3_star": {"200": {"alpha": 0.5, "c": 0.4}},
        "refine_after_init": {"200": 8},
    }
    # Manifest lines as the shipped ones are: the constants, no provenance at all.
    manifest = {
        ("cap", c.name, "p3_star"): {"params": {"alpha": 0.5, "c": 0.4}} for c in cells
    }
    with caplog.at_level("WARNING", logger="deploy.analyze"):
        marks = ad.mark_untuned_baselines(cells, manifest, baseline)
    for c in cells:
        if c.cap == 64:
            assert "p3_star" not in marks[c.name], c.name
        else:
            assert "p3_star" in marks[c.name], c.name
    # ... and the contrast against it is then refused rather than reported as tuned.
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 200) for c in cells}
    strata = ad.strata_of(cells)
    main = ad.main_table("cap", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    wrong_cap = main[(main["level"] == "family_horizon") & (main["cap"] != "64")
                     & (main["policy"] == "phi_k16")]
    assert len(wrong_cap) == 3
    assert not wrong_cap["d_regret_vs_p3_star_ref_tuned"].any()
    at_64 = main[(main["level"] == "family_horizon") & (main["cap"] == "64")
                 & (main["policy"] == "phi_k16")]
    assert at_64["d_regret_vs_p3_star_ref_tuned"].all()


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
    # 3 environments: degenerate, so the range -- not an interval -- is what ships.
    assert pooled.loc["H1a", "n_envs"] == 3 and pooled.loc["H1a", "cluster_degenerate"]
    assert np.isnan(pooled.loc["H1a", "cluster_lo"])
    assert np.isfinite(pooled.loc["H1a", "cluster_min_env"])


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


def _disjoint_cells() -> list[ad.Cell]:
    """Two cells of one stratum whose policy sets do not overlap at all.

    `common_cells` keeps only the cells carrying every policy deployed anywhere in the
    stratum, so disjoint coverage empties the intersection -- the landmine ruling 23
    stress-tested, and exactly what adding `cap` as a stratum dimension creates.
    """
    cells = [c for c in synthetic_cells(policies=("cp0", "p3_star", "phi_k16")) if c.family == "A"]
    first, second = cells[0], cells[1]
    for policy in ("p3_star", "phi_k16"):
        del first.frames[policy]
        del first.groups[policy]
    del second.frames["cp0"]
    del second.groups["cp0"]
    return cells


def test_an_empty_stratum_is_a_visible_row_and_a_non_zero_exit(tmp_path, caplog):
    """Ruling 23: the stratum used to vanish from the CSV with only a log warning."""
    cells = _disjoint_cells()
    strata = ad.strata_of(cells)
    empty = [s for s in strata if ad.stratum_is_empty(s)]
    assert empty, "the fixture must actually empty a stratum"
    assert all(s.level != "cell" for s in empty)
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 200) for c in cells}
    with caplog.at_level("ERROR", logger="deploy.analyze"):
        main = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)
    assert any("EMPTY common support" in r.message for r in caplog.records)
    dec = ad.decomposition_table("synthetic", PRIMARY_RECOMMENDER, cells, per_cell, strata)

    for frame in (main, dec):
        flagged = frame[frame["stratum_empty_common_support"]]
        assert len(flagged) > 0
        # Every stratum that emptied is present in the table, not missing from it.
        assert {(r.level, r.family, r.horizon) for r in flagged.itertuples()} == {
            (s.level, s.family, s.horizon) for s in empty
        }
        # ... and carries no number a reader could quote: only the zero support counts.
        numeric = flagged.select_dtypes(include="number")
        counts = [c for c in numeric.columns
                  if c in ad.EMPTY_ROW_COUNTS or c.endswith(("_n_cells", "_n_excluded_untuned"))]
        # The level's own support is zero -- that is what "empty" means. The counts that
        # survive say where the policy DID run (n_cells_policy) and how many cells the
        # paired contrast had; neither is a quotable result.
        level_support = [c for c in ("n_cells", "n_envs", "n_episodes") if c in numeric.columns]
        assert (numeric[level_support] == 0).all().all()
        assert (numeric[counts] >= 0).all().all()
        rest = numeric.drop(columns=counts)
        assert rest.isna().all().all(), rest.columns[~rest.isna().all()].tolist()
        # The populated strata are untouched.
        assert not frame[~frame["stratum_empty_common_support"]]["regret"].isna().all()


def test_strata_coverage_reports_the_dropped_cells_and_active_exclusions():
    cells = _ragged_cells()
    strata = ad.strata_of(cells)
    exclusions = {
        ("synthetic", cells[0].name, "phi_k16_quality"): rd.exclusion_record(
            "synthetic", cells[0].name, "phi_k16_quality", "hours through the exact evaluator", "s"
        )
    }
    cov = ad.strata_coverage("synthetic", cells, strata, exclusions)
    assert len(cov) == len(strata)
    assert set(cov["level"]) == {s.level for s in strata}
    pooled = cov[cov["level"] == "pooled"].iloc[0]
    # phi_k16 is missing from the (A, 200) cell, so the pooled common support drops it.
    assert pooled["n_cells"] == 3 and pooled["n_common"] == 2
    assert pooled["n_cells_dropped"] == 1
    assert pooled["policies_ragged"] == "phi_k16"
    assert not pooled["stratum_empty_common_support"]
    # The deliberate exclusion is visible beside the stratum it touches, with its reason.
    assert pooled["n_active_exclusions"] == 1
    assert "phi_k16_quality" in pooled["active_exclusions"]
    assert "hours through the exact evaluator" in pooled["active_exclusions"]
    other = cov[(cov["level"] == "cell") & (cov["cell"] != cells[0].name)]
    assert (other["n_active_exclusions"] == 0).all()


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


def test_a_degenerate_cluster_range_ships_under_its_own_column_names():
    """Ruling 21: below `CLUSTER_MIN_ENVS` no interval is emitted, only a named range.

    Disclosure was not enough -- a careful reader still quoted one of these as a 95% CI
    (ledger correction C7). `cluster_lo`/`cluster_hi` are now NaN there and the range
    ships as `cluster_min_env`/`cluster_max_env`, so the wrong reading is unavailable
    rather than flagged.
    """
    assert ad.CLUSTER_MIN_ENVS == 4
    for n_envs in (2, 3):
        cells = _cells_across_envs(n_envs)
        per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, N_BOOT) for c in cells}
        pooled = next(s for s in ad.strata_of(cells) if s.level == "pooled")
        a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
        env_means = [per_cell[c.name].mean[("phi_k16", "d_regret_vs_cp0")] for c in cells]
        assert a["n_envs"] == n_envs and a["cluster_degenerate"] is True
        assert np.isnan(a["cluster_lo"]) and np.isnan(a["cluster_hi"])
        # The range is the range of the environment means, and says so in its name.
        assert a["cluster_min_env"] == pytest.approx(min(env_means), abs=1e-12)
        assert a["cluster_max_env"] == pytest.approx(max(env_means), abs=1e-12)

    cells = _cells_across_envs(5)
    per_cell = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, N_BOOT) for c in cells}
    strata = ad.strata_of(cells)
    pooled = next(s for s in strata if s.level == "pooled")
    a = ad.aggregate(per_cell, pooled, "phi_k16", "d_regret_vs_cp0")
    env_means = [per_cell[c.name].mean[("phi_k16", "d_regret_vs_cp0")] for c in cells]
    assert a["n_envs"] == 5 and a["cluster_degenerate"] is False
    assert a["cluster_lo"] > min(env_means) and a["cluster_hi"] < max(env_means)
    assert np.isnan(a["cluster_min_env"]) and np.isnan(a["cluster_max_env"])
    # A row with no cluster interval at all is not "degenerate", it is empty.
    cell_level = next(s for s in strata if s.level == "cell")
    empty = ad.aggregate(per_cell, cell_level, "phi_k16", "d_regret_vs_cp0")
    assert not empty["cluster_degenerate"]
    assert np.isnan(empty["cluster_min_env"]) and np.isnan(empty["cluster_max_env"])

    # Every table that carries cluster bounds carries the flag and the range beside them.
    cells3 = _cells_across_envs(3)
    per3 = {c.name: ad.compute_cell_stats(c, PRIMARY_RECOMMENDER, 400) for c in cells3}
    strata3 = ad.strata_of(cells3)
    main = ad.main_table("synthetic", PRIMARY_RECOMMENDER, cells3, per3, strata3)
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells3, strata3, 400)
    sc = ad.secondary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells3, per3, strata3)
    for ref in ("cp0", "p3_star"):
        for key in ad.CLUSTER_KEYS:
            assert f"d_regret_vs_{ref}_{key}" in main.columns, key
    pooled_row = main[(main["level"] == "pooled") & (main["policy"] == "phi_k16")].iloc[0]
    assert pooled_row["d_regret_vs_cp0_cluster_degenerate"]
    assert np.isnan(pooled_row["d_regret_vs_cp0_cluster_lo"])
    assert np.isfinite(pooled_row["d_regret_vs_cp0_cluster_min_env"])
    pooled_pc = pc[(pc["level"] == "pooled") & pc["evaluated"]]
    assert len(pooled_pc) == 2 and pooled_pc["cluster_degenerate"].all()  # H1a, H1b (H2 absent)
    assert pooled_pc["cluster_lo"].isna().all()
    assert np.isfinite(pooled_pc["cluster_min_env"]).all()
    assert sc[sc["level"] == "pooled"]["cluster_degenerate"].all()
    # A row with no contrast at all (H2 here) has no interval, so it is not "degenerate".
    assert not pc[~pc["evaluated"]]["cluster_degenerate"].any()
    # No shipped table may carry a bound without the degeneracy flag beside it.
    for frame in (pc, sc):
        assert set(ad.CLUSTER_KEYS) <= set(frame.columns)


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

    # With everything tuned, the manifest's own stamp still flags a cell.
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    full = {"power": {}, "refine_after_init": {},
            "p3_star": {str(T): {"alpha": 0.5, "c": 2.0} for T in by_horizon}}
    target = cells[0]
    manifest = {("synthetic", target.name, "p3_star"):
                {"params": {"alpha": 0.5, "c": 1.0, "params_tuned": False}}}
    ad.mark_untuned_baselines(cells, manifest, full)
    assert cells[0].untuned == frozenset({"p3_star"})
    assert all(c.untuned == frozenset() for c in cells[1:])

    # ... and so does a manifest that records constants the table no longer resolves:
    # the horizon was tuned after those cells ran (the smoke run's situation), so the
    # JSON looks complete while the episodes came from the placeholder.
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    stale = {("synthetic", cells[0].name, "p3_star"): {"params": {"alpha": 0.5, "c": 1.0}}}
    with caplog.at_level("WARNING", logger="deploy.analyze"):
        caplog.clear()
        ad.mark_untuned_baselines(cells, stale, full)
    assert cells[0].untuned == frozenset({"p3_star"})
    assert all(c.untuned == frozenset() for c in cells[1:])
    assert any("not the tuned comparator" in r.message for r in caplog.records)
    # The same constants the table resolves today are not stale.
    fresh = {("synthetic", cells[0].name, "p3_star"): {"params": {"alpha": 0.5, "c": 2.0}}}
    ad.mark_untuned_baselines(cells, fresh, full)
    assert all(c.untuned == frozenset() for c in cells)


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


def _manifest_line(out: Path, cell: ad.Cell, policy: str, **extra) -> dict:
    """A completion record in the runner's schema for one synthetic (cell, policy)."""
    return {
        "test": "smoke", "cell": cell.name, "policy": policy, "group": GROUPS[policy],
        "n": cell.n, "sha": "synthetic", "env_id": cell.env_id, "family": cell.family,
        "horizon": cell.horizon, "cap": cell.cap, "base_seed": cell.base_seed,
        "params": {}, "artifact_sha": None, "counters": None, "snapshots": None,
        "parquet": str(out / "episodes" / "smoke" / cell.name / f"{policy}.parquet"),
        **extra,
    }


def _write_synthetic_run(out: Path, cells: list[ad.Cell]) -> None:
    """The episode tree *and* the manifest that claims it, as a real run leaves them.

    Both, always: `ad.load_cells` reconciles one against the other, so a fixture that
    wrote only the parquet files would be indistinguishable from the orphan batches of
    rulings 17 and 22.
    """
    lines: list[dict] = []
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
            lines.append(_manifest_line(out, c, policy))
    (out / "manifest_smoke.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )


def _smoke_manifest(out: Path) -> dict:
    return rd.latest_records(rd.read_manifest(out / "manifest_smoke.jsonl"))


def test_reconciler_is_clean_on_a_tree_that_matches_its_manifest(tmp_path, cells):
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    orphans, holes = ad.reconcile_episodes(out, "smoke", _smoke_manifest(out))
    assert orphans == [] and holes == []
    assert len(ad.episode_files_on_disk(out, "smoke")) == len(cells) * len(POLICIES)
    assert len(ad.load_cells(out, "smoke", _smoke_manifest(out))) == len(cells)


def test_load_cells_refuses_an_orphan_parquet_that_no_manifest_line_claims(tmp_path, cells):
    """Rulings 17 and 22: a killed --resume leaves complete files with no manifest line.

    The orphan must not enter a number -- before this, `load_cells` globbed the tree and
    nine such files would have lifted Test D's common support from 16 policies to 19.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    victim = cells[0]
    orphan = out / "episodes" / "smoke" / victim.name / "orphan_policy.parquet"
    victim.frames["phi_k16"].to_parquet(orphan, index=False)

    orphans, holes = ad.reconcile_episodes(out, "smoke", _smoke_manifest(out))
    assert orphans == [(victim.name, "orphan_policy")] and holes == []
    with pytest.raises(SystemExit) as excinfo:
        ad.load_cells(out, "smoke", _smoke_manifest(out))
    assert f"{victim.name}/orphan_policy.parquet" in str(excinfo.value)
    # The escape hatch reads it, and says so on every table it then writes.
    loaded = ad.load_cells(out, "smoke", _smoke_manifest(out), accept_unmanifested=True)
    assert "orphan_policy" in {p for c in loaded for p in c.policies}


def test_load_cells_refuses_a_manifest_hole_whose_parquet_is_gone(tmp_path, cells):
    """The quarantine case: the tree no longer reproduces the manifest."""
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    victim = cells[1]
    (out / "episodes" / "smoke" / victim.name / "phi_k16.parquet").unlink()
    orphans, holes = ad.reconcile_episodes(out, "smoke", _smoke_manifest(out))
    assert orphans == [] and holes == [(victim.name, "phi_k16")]
    with pytest.raises(SystemExit, match="1 manifest hole"):
        ad.load_cells(out, "smoke", _smoke_manifest(out))


def test_an_excluded_item_with_a_file_on_disk_is_an_orphan(tmp_path, cells):
    """Ruling 19's durable fix: an exclusion line supersedes a completion.

    An item recorded as deliberately not run has no business having episodes, so a file
    that reappears under it is exactly the orphan class -- not a legitimate completion.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    victim = cells[0]
    with open(out / "manifest_smoke.jsonl", "a") as fh:
        fh.write(json.dumps(rd.exclusion_record(
            "smoke", victim.name, "phi_k16", "costs hours through the exact evaluator", "sha0"
        )) + "\n")
    orphans, holes = ad.reconcile_episodes(out, "smoke", _smoke_manifest(out))
    assert orphans == [(victim.name, "phi_k16")] and holes == []

    # Remove the file and the exclusion is consistent with the tree: no orphan, no hole
    # (an exclusion is done-with-no-file, so it is never a hole either).
    (out / "episodes" / "smoke" / victim.name / "phi_k16.parquet").unlink()
    assert ad.reconcile_episodes(out, "smoke", _smoke_manifest(out)) == ([], [])


def test_reconciler_ignores_quarantine_directories_and_partial_writes(tmp_path, cells):
    """The nine quarantined orphans of ruling 22 live beside the tree, not in it.

    A ``*.parquet.tmp`` from a worker killed mid-write is not an episode either, so
    neither may be reported -- otherwise the reconciler cries wolf on a clean tree and
    its refusal gets routed around.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    victim = cells[0]
    src = out / "episodes" / "smoke" / victim.name / "phi_k16.parquet"

    quarantine = out / "episodes_quarantine" / "orphaned_20260919" / victim.name
    quarantine.mkdir(parents=True)
    (quarantine / "phi_k16.parquet").write_bytes(src.read_bytes())
    inside = out / "episodes" / "smoke" / "quarantine_20260919"
    inside.mkdir()
    (inside / "phi_k16.parquet").write_bytes(src.read_bytes())
    (out / "episodes" / "smoke" / victim.name / "phi_k16_cs.parquet.tmp").write_bytes(b"partial")

    assert ad.reconcile_episodes(out, "smoke", _smoke_manifest(out)) == ([], [])
    loaded = ad.load_cells(out, "smoke", _smoke_manifest(out))
    assert {c.name for c in loaded} == {c.name for c in cells}


def test_cli_always_reconciles_against_the_manifest(tmp_path, cells, monkeypatch):
    """`manifest=None` is legal for fixtures; the CLI must never take it.

    This is the load-bearing half of the fix. `load_cells(out_dir, test)` still works
    for a hand-built tree with no manifest, so nothing stops a future edit from dropping
    the argument at the one call site that matters -- which is precisely the state the
    code was in when rulings 17 and 22 happened. The test pins the call site itself:
    every `main()` invocation must pass a manifest, and an orphan must reach a
    `SystemExit` through the CLI, not just through `load_cells`.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    argv = ["--test", "smoke", "--out-dir", str(out), "--n-boot", "100", "--workers", "1",
            "--allow-unverified", "--recommender", PRIMARY_RECOMMENDER]

    seen: list[dict] = []
    real_load_cells = ad.load_cells

    def spy(out_dir, test, manifest=None, **kwargs):
        seen.append({"manifest": manifest, "kwargs": kwargs})
        return real_load_cells(out_dir, test, manifest, **kwargs)

    monkeypatch.setattr(ad, "load_cells", spy)
    ad.main(argv)
    assert len(seen) == 1
    assert seen[0]["manifest"] is not None, "the CLI passed manifest=None to load_cells"
    assert set(seen[0]["manifest"]) == {
        ("smoke", c.name, p) for c in cells for p in POLICIES
    }
    assert seen[0]["kwargs"] == {"accept_unmanifested": False, "allow_mixed_sim": False}

    # And the refusal is reachable from the CLI: an orphan stops the run.
    cells[0].frames["phi_k16"].to_parquet(
        out / "episodes" / "smoke" / cells[0].name / "orphan_policy.parquet", index=False
    )
    with pytest.raises(SystemExit, match="orphan"):
        ad.main(argv)


def test_accept_unmanifested_stamps_every_table_it_writes(tmp_path, cells):
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    cells[0].frames["phi_k16"].to_parquet(
        out / "episodes" / "smoke" / cells[0].name / "orphan_policy.parquet", index=False
    )
    argv = ["--test", "smoke", "--out-dir", str(out), "--n-boot", "100", "--workers", "1",
            "--allow-unverified", "--recommender", PRIMARY_RECOMMENDER]
    try:
        result = ad.main(argv + ["--accept-unmanifested"])
        written = [Path(p) for p in result["written"]]
        assert written
        for path in written:
            frame = pd.read_csv(path)
            assert "accepted_unmanifested" in frame.columns, path.name
            assert bool(frame["accepted_unmanifested"].all())
    finally:
        ad._ACCEPT_UNMANIFESTED = False


def test_load_cells_refuses_a_test_spanning_two_simulation_surfaces(tmp_path, cells):
    """`--resume` must not be able to keep episodes from a different code version.

    ``manifest_A`` really is 1360 items at one sha plus 80 at another; the repo sha is
    not a substitute for a surface fingerprint (it moves for every unrelated commit and
    nothing downstream reads it), so the analysis is where a mixed set has to stop.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    manifest = out / "manifest_smoke.jsonl"

    # Every line unstamped, as the whole shipped tree is: no refusal, nothing to compare.
    assert ad.sim_shas_in(_smoke_manifest(out), "smoke") == (set(), len(cells) * len(POLICIES))
    assert len(ad.load_cells(out, "smoke", _smoke_manifest(out))) == len(cells)

    # One surface: fine, and only one.
    lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    manifest.write_text("".join(
        json.dumps({**line, "sim_sha": "aaaaaaaaaaaa"}) + "\n" for line in lines
    ))
    assert ad.sim_shas_in(_smoke_manifest(out), "smoke") == ({"aaaaaaaaaaaa"}, 0)
    assert len(ad.load_cells(out, "smoke", _smoke_manifest(out))) == len(cells)

    # Two: refused by name, and only --allow-mixed-sim gets past it.
    manifest.write_text("".join(
        json.dumps({**line, "sim_sha": "aaaaaaaaaaaa" if i else "bbbbbbbbbbbb"}) + "\n"
        for i, line in enumerate(lines)
    ))
    assert ad.sim_shas_in(_smoke_manifest(out), "smoke")[0] == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    with pytest.raises(SystemExit, match="2 simulation surfaces"):
        ad.load_cells(out, "smoke", _smoke_manifest(out))
    assert len(ad.load_cells(out, "smoke", _smoke_manifest(out), allow_mixed_sim=True)) == len(cells)

    # An exclusion line carries no sim_sha and must not count as a second surface.
    with open(manifest, "w") as fh:
        for line in lines:
            fh.write(json.dumps({**line, "sim_sha": "aaaaaaaaaaaa"}) + "\n")
        fh.write(json.dumps(rd.exclusion_record(
            "smoke", cells[0].name, "phi_k16", "deliberately dropped", "sha0")) + "\n")
    assert ad.sim_shas_in(_smoke_manifest(out), "smoke")[0] == {"aaaaaaaaaaaa"}


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
                 "dynamics_smoke.csv", "cap_demotion_smoke.csv", "strata_coverage_smoke.csv"):
        assert (tables / name).exists(), name
    # The five cross-recommender tables need --recommender all and nothing backfills
    # them from disk, so a single-recommender run does not produce them (4.4).
    for base_name in ad.CROSS_RECOMMENDER_TABLES:
        assert not (tables / ad.table_name(base_name, "smoke")).exists(), base_name
    assert result["cross_recommender"] is False
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
    # Appended, not written over: the latest line per item wins, so this supersedes the
    # fixture's own phi_k16 record while every other item keeps its completion line (the
    # tree and the manifest must still reconcile -- `ad.reconcile_episodes`).
    with open(out / "manifest_smoke.jsonl", "a") as fh:
        fh.write(json.dumps(_manifest_line(
            out, cell, "phi_k16", snapshots=str(out / "snapshots" / "x.pkl"),
            params={"artifact": "unused.joblib", "tau": 0.6}, counters={},
        )) + "\n")
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


def test_cli_exits_non_zero_on_an_empty_stratum_but_writes_it_first(tmp_path):
    """Ruling 23 through the CLI: the tables are written, then the run fails.

    Written first on purpose -- a reader has to be able to see WHICH stratum emptied,
    which is what `strata_coverage_<test>.csv` is for.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, _disjoint_cells())
    argv = ["--test", "smoke", "--out-dir", str(out), "--n-boot", "100", "--workers", "1",
            "--allow-unverified", "--recommender", PRIMARY_RECOMMENDER]
    with pytest.raises(SystemExit, match="EMPTY common support"):
        ad.main(argv)
    tables = out / "tables"
    cov = pd.read_csv(tables / "strata_coverage_smoke.csv")
    assert cov["stratum_empty_common_support"].any()
    voided = cov[cov["stratum_empty_common_support"]]
    assert (voided["n_common"] == 0).all() and (voided["level"] != "cell").all()
    assert voided["policies_ragged"].str.len().gt(0).all()
    main = pd.read_csv(tables / "main_smoke_primary.csv")
    assert main["stratum_empty_common_support"].any()
    assert main[main["stratum_empty_common_support"]]["regret"].isna().all()

    result = ad.main(argv + ["--allow-empty-strata"])
    assert result["empty_strata"], result
    assert all("cell" not in name for name in result["empty_strata"])


def test_a_single_recommender_run_does_not_touch_the_cross_recommender_tables(tmp_path, cells):
    """4.4: the disk backfill is gone; these five tables need --recommender all.

    `cross_recommender_frames` used to complete any recommender not computed in the run
    by reading its main/cells table off disk with no freshness check at all. Immediately
    after the ruling-22 quarantine, `--test D --recommender primary` would have
    recomputed the primary recommender on the corrected 16-policy episode set while
    silently backfilling four recommenders computed on the contaminated 19-policy one.
    A single-recommender run must now leave the five tables exactly as it found them.
    """
    out = tmp_path / "deploy"
    _write_synthetic_run(out, cells)
    tables = out / "tables"
    base = ["--test", "smoke", "--out-dir", str(out), "--n-boot", "100", "--workers", "1",
            "--allow-unverified"]
    full = ad.main(base)
    assert full["cross_recommender"] is True
    names = [ad.table_name(base_name, "smoke") for base_name in ad.CROSS_RECOMMENDER_TABLES]
    before = {name: (tables / name).read_bytes() for name in names}
    kendall_all = pd.read_csv(tables / "recommender_kendall_smoke.csv")
    ovd_all = pd.read_csv(tables / "offline_vs_deployed_smoke.csv")
    assert len(kendall_all) > 0 and set(ovd_all["recommender"]) == set(RECOMMENDER_NAMES)

    one = ad.main(base + ["--recommender", "lcb"])
    assert one["cross_recommender"] is False
    assert not any(Path(w).name in set(names) for w in one["written"])
    assert {name: (tables / name).read_bytes() for name in names} == before

    # ... and a later --recommender all run rebuilds them from that run alone.
    again = ad.main(base)
    assert again["cross_recommender"] is True
    pd.testing.assert_frame_equal(
        pd.read_csv(tables / "recommender_kendall_smoke.csv"), kendall_all
    )


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
