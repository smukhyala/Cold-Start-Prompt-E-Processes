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


def test_primary_contrasts_keep_three_rows_when_a_policy_is_missing():
    cells = synthetic_cells(policies=("cp0", "p3_star", "phi_k16"))
    strata = ad.strata_of(cells)
    pc = ad.primary_contrasts("synthetic", PRIMARY_RECOMMENDER, cells, strata, 200)
    pooled = pc[pc["level"] == "pooled"].set_index("hypothesis")
    assert len(pooled) == 3
    assert pooled.loc["H2", "status"] == "missing:phi_k16_quality" and np.isnan(pooled.loc["H2", "delta"])
    assert pooled.loc["H1a", "status"] == "ok"


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
    assert r_same.summary["ood_frac"] < 0.10
    assert r_shift.summary["ood_frac"] > 0.99
    assert r_same.summary["max_frac_outside"] < 0.05 and r_shift.summary["mean_frac_outside"] > 0.99
    assert abs(r_same.summary["max_abs_std_shift"]) < 0.3 and r_shift.summary["max_abs_std_shift"] > 9.0
    assert r_same.summary["max_ks"] < 0.15 and r_shift.summary["max_ks"] > 0.99
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
    assert r.summary["n_features_standardizable"] == 2 and np.isfinite(r.summary["ood_frac"])
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
