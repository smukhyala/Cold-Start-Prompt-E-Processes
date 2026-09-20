"""Tests for the deployment study's cell/seed definitions and the nested tuning scripts.

`cells.py` is the single source of truth for which episodes every run script sees, so
the properties checked here are the ones the study's validity rests on: the cell
enumeration is injective and stable, the three seed splits never overlap, and the
startup guard really does recognise a corpus seed and an old-benchmark seed. The
tuning and threshold-selection logic is exercised end to end on a tiny grid through
the modules' own functions (real simulator, no mocks), including resume and the
horizon-holdout bookkeeping.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import cells  # noqa: E402
import label_states  # noqa: E402
import select_thresholds as st  # noqa: E402
import tune_baselines as tb  # noqa: E402

from cold_start.growing.deploy.artifacts import save_model  # noqa: E402
from cold_start.growing.deploy.feature_groups import CLOCK  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.search_policies import PowerSchedule  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

PLAN_MAIN_ENVS = {
    "beta_good_common",
    "beta_rare_excellent",
    "beta_mostly_mediocre",
    "beta_skewed",
    "tail_b0.5_mu1.0_c1.0",
    "tail_b2.0_mu1.0_c1.0",
    "tail_b8.0_mu1.0_c1.0",
    "tail_b1.0_mu0.9_c2.0",
}

TINY_ENVS = ("beta_good_common", "tail_b2.0_mu1.0_c1.0")
TINY_T, TINY_M, TINY_CAP = 50, 32, 64


def _all_cells():
    for env_id in cells.ENV_ORDER:
        for horizon in cells.HORIZONS + cells.EXTRA_HORIZONS:
            for cap in cells.CAPS:
                yield env_id, horizon, cap
        for horizon in cells.HORIZONS:
            for cap in cells.EXTRA_CAPS:
                yield env_id, horizon, cap


# ---- cells -----------------------------------------------------------------------------


def test_environment_sets_match_label_states():
    assert set(cells.MAIN_ENVS) == PLAN_MAIN_ENVS
    assert len(cells.MAIN_ENVS) == 8
    by_id = dict(label_states.FAMILY_A + label_states.FAMILY_B + label_states.FAMILY_C)
    for env_id, spec in cells.MAIN_ENVS.items():
        assert spec == by_id[env_id]
    assert set(cells.HELDOUT_ENVS) == {name for name, _ in label_states.FAMILY_C}
    assert len(cells.ALL_CORPUS_ENVS) == 30
    assert set(cells.ALL_CORPUS_ENVS) <= set(by_id)
    assert not (set(cells.ALL_CORPUS_ENVS) & set(cells.HELDOUT_ENVS))
    assert cells.HORIZONS == (50, 100, 200, 500, 1000)


def test_env_selectors():
    assert cells.env_ids_for("main") == tuple(cells.MAIN_ENVS)
    assert set(cells.env_ids_for("all")) == set(cells.ALL_CORPUS_ENVS) | set(cells.HELDOUT_ENVS)
    assert cells.env_ids_for("beta_skewed, mix_broad_low_narrow_high") == (
        "beta_skewed",
        "mix_broad_low_narrow_high",
    )
    with pytest.raises(KeyError):
        cells.env_ids_for("no_such_env")


def test_cell_id_is_injective_and_stable():
    ids = [cells.cell_id(e, T, c) for e, T, c in _all_cells()]
    assert len(ids) == len(set(ids)) == cells.N_CELLS
    assert sorted(ids) == list(range(cells.N_CELLS))
    # Pinned values: the enumeration must never move under an existing run.
    first = cells.ENV_ORDER[0]
    assert first == "beta_good_common"
    assert cells.cell_id(first, 50, 32) == 0
    assert cells.cell_id(first, 50, "T") == 2
    assert cells.cell_id(first, 50, 50) == 2
    assert cells.cell_id(first, 100, 32) == 3
    assert cells.cell_id("beta_good_common", 1000, 64) == 13
    assert cells.cell_id(first, 2000, 32) == cells.N_MAIN_GRID_CELLS
    assert cells.base_seed("test", "beta_good_common", 1000, 64) == 10_013_000
    # Same cell, whichever spelling of the cap.
    assert cells.cell_id("beta_skewed", 200, "T") == cells.cell_id("beta_skewed", 200, 200)
    with pytest.raises(KeyError):
        cells.cell_id("no_such_env", 50, 64)
    with pytest.raises(ValueError):
        cells.cell_id(first, 75, 64)
    with pytest.raises(ValueError):
        cells.cell_id(first, 50, 48)


def test_base_seeds_disjoint_across_splits():
    per_split = {
        split: {cells.base_seed(split, e, T, c) for e, T, c in _all_cells()}
        for split in cells.SPLIT_BASE
    }
    for split, seeds in per_split.items():
        assert len(seeds) == cells.N_CELLS, split
        assert min(seeds) == cells.SPLIT_BASE[split]
    splits = list(per_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            assert not (per_split[a] & per_split[b]), (a, b)
    with pytest.raises(KeyError):
        cells.base_seed("train", "beta_skewed", 50, 64)


def test_make_cell_resolves_cap_and_seed():
    spec = cells.make_cell("val", "mix_broad_low_narrow_high", 200, "T", 16)
    assert isinstance(spec, CellSpec)
    assert spec.cap == 200
    assert spec.env_spec == cells.HELDOUT_ENVS["mix_broad_low_narrow_high"]
    assert spec.base_seed == cells.base_seed("val", "mix_broad_low_narrow_high", 200, 200)
    assert spec.n_replicates == 16
    with pytest.raises(KeyError):
        cells.make_cell("val", "no_such_env", 200, 64, 16)


def test_seed_disjointness_guard():
    corpus_seed = 20260910 + 1_000_003 * 0 + 31 * 50
    assert corpus_seed in cells.corpus_seed_set()
    with pytest.raises(ValueError, match="corpus"):
        cells.assert_seed_disjointness([corpus_seed])
    # A labelling batch of a detached trajectory is a corpus seed too.
    with pytest.raises(ValueError, match="corpus"):
        cells.assert_seed_disjointness([corpus_seed + 7919 * 3 + 1_000_003 * 5])
    bench_seed = 4242 + zlib.crc32(b"sqrt_t") % 10000
    with pytest.raises(ValueError, match="benchmark"):
        cells.assert_seed_disjointness([bench_seed])
    # The plan's ranges: corpus [20.26M, 343.4M], old benchmark [4.5k, 14.1k].
    assert 20_260_000 < min(cells.corpus_seed_set()) < max(cells.corpus_seed_set()) < 343_400_000
    bench = cells.old_benchmark_seed_set()
    assert 4_500 < min(bench) < max(bench) < 14_200
    study = [
        cells.base_seed(split, env_id, horizon, cap)
        for split in cells.SPLIT_BASE
        for env_id in list(cells.MAIN_ENVS) + list(cells.HELDOUT_ENVS)
        for horizon in cells.HORIZONS
        for cap in cells.CAPS
    ]
    cells.assert_seed_disjointness(study)
    cells.assert_seed_disjointness([])


def test_extra_caps_block_adds_cap_128_without_moving_existing_ids():
    """`EXTRA_CAPS=(128,)` is a third block, appended after `EXTRA_HORIZONS`, so
    `run_deployment.py --test cap` can use cap=128 without disturbing any id (and
    therefore any `base_seed`) a prior run already consumed.

    The pins below were captured with `cells.cell_id` *before* `EXTRA_CAPS` existed
    (main grid + `EXTRA_HORIZONS` only) and must still hold afterwards.
    """
    assert cells.cell_id("beta_good_common", 50, 32) == 0
    assert cells.cell_id("beta_good_common", 1000, "T") == 14
    assert cells.cell_id("tail_b8.0_mu1.0_c1.0", 200, 64) == 472
    assert cells.cell_id("tail_b2.0_mu1.0_c1.0", 1000, 32) == 372
    assert cells.cell_id("beta_rare_excellent", 200, "T") == 53
    assert cells.cell_id(cells.ENV_ORDER[-1], 1000, "T") == 494
    assert cells.cell_id("beta_good_common", 2000, 32) == 495
    assert cells.cell_id(cells.ENV_ORDER[-1], 2000, "T") == 593
    assert cells.N_MAIN_GRID_CELLS == 495

    assert cells.EXTRA_CAPS == (128,)
    lo = cells.N_MAIN_GRID_CELLS + cells.N_EXTRA_HORIZONS_CELLS

    # cap=128 exists at every corpus+heldout env x the main HORIZONS, and is unique.
    new_ids = {
        cells.cell_id(env_id, horizon, 128)
        for env_id in cells.ENV_ORDER
        for horizon in cells.HORIZONS
    }
    n_extra_caps = len(cells.ENV_ORDER) * len(cells.HORIZONS) * len(cells.EXTRA_CAPS)
    assert len(new_ids) == n_extra_caps == cells.N_EXTRA_CAPS_CELLS
    assert min(new_ids) == lo
    assert max(new_ids) == lo + n_extra_caps - 1 == cells.N_CELLS - 1

    # Disjoint from every pre-existing (main grid + EXTRA_HORIZONS) id.
    existing_ids = {
        cells.cell_id(env_id, horizon, cap)
        for env_id in cells.ENV_ORDER
        for horizon in cells.HORIZONS + cells.EXTRA_HORIZONS
        for cap in cells.CAPS
    }
    assert len(existing_ids) == lo
    assert not (existing_ids & new_ids)

    # cap=128 is only wired up at the main HORIZONS, not EXTRA_HORIZONS.
    with pytest.raises(ValueError):
        cells.cell_id("beta_good_common", 2000, 128)

    # Every new base_seed (any split, any env, either target horizon) is disjoint from
    # every pre-existing split seed and from the corpus.
    new_seeds = {
        cells.base_seed(split, env_id, horizon, 128)
        for split in cells.SPLIT_BASE
        for env_id in cells.ENV_ORDER
        for horizon in cells.HORIZONS
    }
    existing_seeds = {
        cells.base_seed(split, env_id, horizon, cap)
        for split in cells.SPLIT_BASE
        for env_id in cells.ENV_ORDER
        for horizon in cells.HORIZONS + cells.EXTRA_HORIZONS
        for cap in cells.CAPS
    }
    assert not (new_seeds & existing_seeds)
    assert max(new_seeds) < cells.CORPUS_SEED_MIN
    cells.assert_seed_disjointness(new_seeds)  # real check against the corpus seed set

    # The four run_deployment.py --test cap envs, at both requested horizons.
    for env_id in (
        "beta_good_common",
        "beta_rare_excellent",
        "tail_b2.0_mu1.0_c1.0",
        "tail_b8.0_mu1.0_c1.0",
    ):
        for horizon in (200, 1000):
            spec = cells.make_cell("test", env_id, horizon, 128, 16)
            assert spec.cap == 128
            assert spec.base_seed == cells.base_seed("test", env_id, horizon, 128)


# ---- tune_baselines -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_configs() -> list[tb.ScheduleConfig]:
    return tb.tuning_grid((0.5, 1.0 / 3.0), [0.5, 1.0, 2.0], (2, 4))


@pytest.fixture(scope="module")
def tiny_tuning(tiny_configs) -> pd.DataFrame:
    items = tb.build_items("tune", TINY_ENVS, [TINY_T], TINY_CAP, TINY_M, tiny_configs, chunk=3)
    rows: list[dict] = []
    for item_rows in tb.run_items(items, workers=1):
        rows += item_rows
    return pd.DataFrame(rows)


def test_tuning_grid_and_items(tiny_configs):
    assert len(tiny_configs) == 2 * 3 + 2
    assert len({cfg.key() for cfg in tiny_configs}) == len(tiny_configs)
    assert len({cfg.name() for cfg in tiny_configs}) == len(tiny_configs)
    grid = tb.c_grid()
    assert len(grid) == 40 and grid[0] == 0.25 and grid[-1] == 16.0
    assert np.allclose(np.diff(np.log(grid)), np.log(grid[1] / grid[0]))
    items = tb.build_items(
        "tune", TINY_ENVS, [TINY_T, 100], TINY_CAP, TINY_M, tiny_configs, chunk=3
    )
    assert [it.horizon for it in items][:3] == [100, 100, 100]  # longest first
    assert sum(len(it.configs) for it in items) == 2 * 2 * len(tiny_configs)
    assert all(len(it.configs) <= 3 for it in items)
    assert items[0].spec.base_seed == cells.base_seed("tune", items[0].env_id, 100, TINY_CAP)


def test_tuning_rows_match_a_direct_run(tiny_tuning, tiny_configs):
    df = tiny_tuning
    assert len(df) == len(TINY_ENVS) * len(tiny_configs)
    required = {"env", "T", "alpha", "c", "mean_regret", "mean_q", "se"}
    assert required <= set(df.columns)
    assert (df["split"] == "tune").all() and (df["n_replicates"] == TINY_M).all()
    assert df["mean_regret"].between(0.0, 1.0).all() and df["se"].gt(0.0).all()
    # A row is exactly what `run_cell` reports for that configuration on that cell.
    row = df[(df["policy"] == tb.POWER) & (df["env"] == TINY_ENVS[0])].iloc[0]
    spec = cells.make_cell("tune", TINY_ENVS[0], TINY_T, TINY_CAP, TINY_M)
    res = run_cell(
        spec,
        PowerSchedule(alpha=float(row["alpha"]), c=float(row["c"])),
        table=CSTable.load_or_build(TINY_T, alpha=0.05),
        dynamics_grid=0,
    )
    assert row["mean_regret"] == pytest.approx(float(res.regret(PRIMARY_RECOMMENDER).mean()))
    assert row["mean_q"] == pytest.approx(float(res.q[PRIMARY_RECOMMENDER].mean()))
    assert row["base_seed"] == spec.base_seed
    # refine_after_init rows sit on the same axes: alpha 0, c = K0.
    refine = df[df["policy"] == tb.REFINE]
    assert set(refine["K0"]) == {2, 4} and (refine["alpha"] == 0.0).all()
    assert (refine["c"] == refine["K0"]).all()


def test_selection_and_resume(tiny_tuning, tiny_configs, tmp_path):
    df = tiny_tuning
    params = tb.select_schedule_params(df, TINY_ENVS)
    assert set(params) == {tb.POWER, tb.REFINE}
    assert set(params[tb.POWER]) == {str(0.5), str(1.0 / 3.0)}
    for per_t in params[tb.POWER].values():
        assert set(per_t) == {str(TINY_T)} and per_t[str(TINY_T)] in (0.5, 1.0, 2.0)
    assert params[tb.REFINE][str(TINY_T)] in (2, 4)
    # The chosen c is the pooled (equal-weight) minimiser, computed by hand.
    power = df[df["policy"] == tb.POWER].assign(alpha_r=df["alpha"].round(6))
    for alpha_key, per_t in params[tb.POWER].items():
        pooled = (
            power[power["alpha_r"] == round(float(alpha_key), 6)]
            .groupby("c")["mean_regret"].mean()
        )
        assert per_t[str(TINY_T)] == pytest.approx(float(pooled.idxmin()))

    oracle = tb.oracle_tuned(df)
    assert len(oracle) == len(TINY_ENVS) * 2
    assert oracle.groupby(["env", "T"])["best_over_alpha"].sum().eq(1).all()
    for _, row in oracle.iterrows():
        grp = power[(power["env"] == row["env"]) & (power["alpha_r"] == round(row["alpha"], 6))]
        assert row["mean_regret"] == pytest.approx(float(grp["mean_regret"].min()))
        assert row["c_oracle"] == float(grp.loc[grp["mean_regret"].idxmin(), "c"])

    # Validation on the other split: every tuned (alpha, c*) and K0* per horizon.
    sel_items = tb.build_selection_items(params, TINY_ENVS, TINY_CAP, TINY_M)
    assert len(sel_items) == len(TINY_ENVS)
    assert all(it.split == "val" and len(it.configs) == 3 for it in sel_items)
    rows: list[dict] = []
    for item_rows in tb.run_items(sel_items, workers=1):
        rows += item_rows
    selection = pd.DataFrame(rows)
    assert (selection["base_seed"] != df["base_seed"].iloc[0]).all()
    p3 = tb.select_p3_star(selection, TINY_ENVS, params)
    assert set(p3) == {str(TINY_T)}
    assert p3[str(TINY_T)]["alpha"] in (0.5, 1.0 / 3.0)
    assert p3[str(TINY_T)]["c"] == params[tb.POWER][str(p3[str(TINY_T)]["alpha"])][str(TINY_T)]

    # Resume: after a CSV round trip nothing is left to do, and a new M is new work.
    path = tmp_path / "schedule_tuning.csv"
    tb.write_csv_atomic(df, path)
    back = tb.read_csv(path)
    assert tb.done_keys(back) == tb.done_keys(df)
    done = tb.done_keys(back)
    assert tb.build_items("tune", TINY_ENVS, [TINY_T], TINY_CAP, TINY_M, tiny_configs, done) == []
    fresh = tb.build_items("tune", TINY_ENVS, [TINY_T], TINY_CAP, 2 * TINY_M, tiny_configs, done)
    assert len(fresh) == 2
    allowed = tb.grid_keys([TINY_T], tiny_configs[:2])
    restricted = tb.restrict(back, "tune", TINY_ENVS, TINY_CAP, TINY_M, allowed)
    assert len(restricted) == 2 * len(TINY_ENVS)
    tb.write_json_atomic(params, tmp_path / "baseline_params.json")
    written = json.loads((tmp_path / "baseline_params.json").read_text())
    assert written[tb.REFINE] == params[tb.REFINE]


def _selection_row(T: int, alpha: float, c: float, env: str, regret: float) -> dict:
    return {
        "split": "val", "env": env, "T": T, "cap": TINY_CAP, "n_replicates": TINY_M,
        "policy": tb.POWER, "alpha": alpha, "c": c, "K0": -1,
        "mean_regret": regret, "mean_q": 1.0 - regret, "se": 0.01,
    }


def test_stale_validation_rows_cannot_win_p3_star():
    """A c tuned for T' must not compete at T just because an earlier run left rows there.

    c*(0.5, 50) = 1.0 and c*(0.5, 100) = 2.0; planted (T=50, c=2.0) rows with zero regret
    would win T=50 if the restriction forgot the horizon.
    """
    params = {
        tb.POWER: {"0.5": {"50": 1.0, "100": 2.0}},
        tb.REFINE: {"50": 4, "100": 4},
    }
    envs = ["e1", "e2"]
    rows = []
    for env in envs:
        rows.append(_selection_row(50, 0.5, 1.0, env, 0.10))
        rows.append(_selection_row(100, 0.5, 2.0, env, 0.05))
        rows.append(_selection_row(50, 0.5, 2.0, env, 0.0))  # stale: c*(0.5, 100) at T=50
    df = pd.DataFrame(rows)

    allowed = tb.selected_keys(params)
    assert (50, tb.POWER, 0.5, 1.0) in allowed and (100, tb.POWER, 0.5, 2.0) in allowed
    assert (50, tb.POWER, 0.5, 2.0) not in allowed
    restricted = tb.restrict(df, "val", envs, TINY_CAP, TINY_M, allowed)
    survivors = sorted(set(zip(restricted["T"], restricted["c"], strict=True)))
    assert survivors == [(50, 1.0), (100, 2.0)]
    expected = {"50": 1.0, "100": 2.0}
    assert {t: v["c"] for t, v in tb.select_p3_star(restricted, envs).items()} == expected
    # The guard inside select_p3_star holds on the unrestricted frame too.
    assert {t: v["c"] for t, v in tb.select_p3_star(df, envs, params).items()} == expected
    assert tb.select_p3_star(df, envs, params)["50"]["pooled_regret"] == pytest.approx(0.10)


def test_test_split_is_refused_by_default(tmp_path):
    common = ["--n-replicates", "4", "--workers", "1", "--envs", TINY_ENVS[0],
              "--horizons", str(TINY_T), "--out", str(tmp_path)]
    with pytest.raises(tb.TestSplitRefused):
        tb.main(["--split", "test", *common])
    with pytest.raises(tb.TestSplitRefused):
        tb.main(["--select-split", "test", *common])
    with pytest.raises(tb.TestSplitRefused):
        st.main(["--split", "test", "--models", str(tmp_path), *common])
    assert list(tmp_path.iterdir()) == []  # refused before anything was written
    assert tb.check_splits(("tune", "val"), False) == ""
    assert tb.check_splits(("tune", "test"), True) == tb.TESTSPLIT_SUFFIX
    assert tb.output_path(tmp_path, "thresholds", ".json", tb.TESTSPLIT_SUFFIX).name == (
        "thresholds_TESTSPLIT.json"
    )
    # Opted in: the tuning script runs, and every output carries the suffix.
    tb.main(["--split", "test", "--allow-test-split", "--n-c", "2", "--alphas", "0.5",
             "--k0s", "2", *common])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "baseline_params_TESTSPLIT.json",
        "schedule_oracle_tuned_TESTSPLIT.csv",
        "schedule_selection_TESTSPLIT.csv",
        "schedule_tuning_TESTSPLIT.csv",
    ]


# ---- select_thresholds ----------------------------------------------------------------------


def _clock_row(t: int, horizon: int, k: int) -> list[float]:
    remaining = horizon - t
    return [
        float(t), float(horizon), float(remaining), remaining / horizon, t / horizon, float(k),
        k / t if t else float(k), k / horizon, float(np.log(max(t, 1))), float(np.log(k)),
        k / np.sqrt(max(t, 1)),
    ]


@pytest.fixture(scope="module")
def clock_models(tmp_path_factory) -> dict[str, Path]:
    """Two tiny CLOCK artifacts, one marked as a T=100 horizon holdout."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(2000):
        horizon = int(rng.choice([50, 100]))
        t = int(rng.integers(2, horizon))
        k = int(rng.integers(2, 20))
        X.append(_clock_row(t, horizon, k))
        y.append(int(k / np.sqrt(t) < 1.0))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    pipe.fit(np.array(X), np.array(y))
    out = tmp_path_factory.mktemp("models")
    paths = {}
    for name, subset in (("tiny_clock", {}), ("tiny_clock_noT100", {"exclude_horizons": [100]})):
        paths[name] = save_model(
            out / f"{name}.joblib",
            {"pipeline": pipe, "features": list(CLOCK), "k": 4, "tau": 0.55,
             "meta": {"variant": name, "subset": subset}},
        )
    return paths


def _retrain(path: Path, k: int = 2) -> None:
    """Overwrite the artifact at `path` with a different model under the same name."""
    artifact = st.load_model(path)
    save_model(path, {**artifact, "k": k, "tau": 0.45, "meta": {**artifact["meta"], "k": k}})


def test_threshold_selection_end_to_end(clock_models, tmp_path):
    variants = st.list_variants(next(iter(clock_models.values())).parent)
    assert set(variants) == set(clock_models)
    artifacts = {name: st.load_model(path) for name, path in variants.items()}
    assert st.heldout_horizons(artifacts["tiny_clock_noT100"]) == (100,)
    assert not st.uses_log_e(artifacts["tiny_clock"])

    taus = [0.3, 0.7]
    grids = st.tau_grids(variants, taus)
    assert grids == {name: (0.3, 0.7) for name in variants}
    horizons = [TINY_T, 100]
    items = st.build_items(variants, TINY_ENVS[:1], horizons, grids, TINY_CAP, 16)
    assert len(items) == 2 * 2 and items[0].horizon == 100
    fingerprints = {name: st.artifact_fingerprint(path) for name, path in variants.items()}
    assert all(it.fingerprint == fingerprints[it.variant] for it in items)
    rows: list[dict] = []
    for item_rows in st.run_items(items, workers=1):
        rows += item_rows
    df = pd.DataFrame(rows)
    assert len(df) == 2 * 2 * len(taus)
    required = {
        "variant", "tau", "env", "T", "mean_regret", "mean_q", "mean_search_frac", "cap_hit_frac",
    }
    assert required <= set(df.columns) and set(st.COUNTER_NAMES) <= set(df.columns)
    assert (df["n_nonfinite_rows"] == 0).all() and (df["n_decisions"] > 0).all()
    assert (df["k"] == 4).all() and (df["tau_off"] == 0.55).all()
    assert (df["split"] == "val").all()
    assert all(row["fingerprint"] == fingerprints[row["variant"]] for _, row in df.iterrows())
    marked = df[df["variant"] == "tiny_clock_noT100"]
    assert marked["heldout_T"].eq(marked["T"] == 100).all()
    assert not df.loc[df["variant"] == "tiny_clock", "heldout_T"].any()
    # Higher tau searches less.
    by = df.groupby(["variant", "env", "T"])
    for _, grp in by:
        g = grp.sort_values("tau")
        assert g["mean_search_frac"].iloc[0] >= g["mean_search_frac"].iloc[-1]

    thresholds = st.build_thresholds(
        df, artifacts, TINY_ENVS[:1], horizons, grids, "val", 16, fingerprints
    )
    assert set(thresholds) == set(variants)
    plain, holdout = thresholds["tiny_clock"], thresholds["tiny_clock_noT100"]
    assert plain["tau_off"] == 0.55 and plain["k"] == 4 and plain["kind"] == "model"
    assert plain["tau_val"] in taus and plain["fingerprint"] == fingerprints["tiny_clock"]
    assert set(plain["curve"]) == {"0.3", "0.7"} and "heldout_horizons" not in plain
    expected = df[df["variant"] == "tiny_clock"].groupby("tau")["mean_regret"].mean()
    assert plain["curve"]["0.3"] == pytest.approx(float(expected[0.3]))
    assert plain["tau_val"] == pytest.approx(float(expected.idxmin()))
    assert holdout["heldout_horizons"] == [100]
    kept = df[(df["variant"] == "tiny_clock_noT100") & (df["T"] == TINY_T)]
    expected_excl = kept.groupby("tau")["mean_regret"].mean()
    assert holdout["curve_excl_heldout"]["0.7"] == pytest.approx(float(expected_excl[0.7]))
    assert holdout["tau_val_excl_heldout"] == pytest.approx(float(expected_excl.idxmin()))

    # Resume: a CSV round trip leaves nothing to do; one missing tau is one item of one tau.
    path = tmp_path / "threshold_selection.csv"
    tb.write_csv_atomic(df, path)
    back = tb.read_csv(path)
    env1 = TINY_ENVS[:1]
    assert st.build_items(variants, env1, horizons, grids, TINY_CAP, 16, st.done_keys(back)) == []
    partial = st.build_items(
        variants, env1, horizons, grids, TINY_CAP, 16, st.done_keys(back.iloc[1:])
    )
    assert len(partial) == 1 and len(partial[0].taus) == 1
    # An incomplete curve yields no threshold rather than one chosen on a subset of cells.
    incomplete = st.build_thresholds(back.iloc[1:], artifacts, env1, horizons, grids, "val", 16)
    assert set(incomplete) == {"tiny_clock_noT100"}


def test_retrained_artifact_is_new_work(clock_models, tmp_path):
    """Same name, different model: the old rows must neither count as done nor be selected on."""
    name = "tiny_clock"
    path = clock_models[name]
    variants = {name: path}
    grids = st.tau_grids(variants, [0.3, 0.7])
    old_fp = st.artifact_fingerprint(path)
    items = st.build_items(variants, TINY_ENVS[:1], [TINY_T], grids, TINY_CAP, 8)
    rows: list[dict] = []
    for item_rows in st.run_items(items, workers=1):
        rows += item_rows
    old = pd.DataFrame(rows)
    assert (old["fingerprint"] == old_fp).all() and (old["k"] == 4).all()
    done = st.done_keys(old)
    assert st.build_items(variants, TINY_ENVS[:1], [TINY_T], grids, TINY_CAP, 8, done) == []

    _retrain(path, k=2)
    new_fp = st.artifact_fingerprint(path)
    assert new_fp != old_fp
    fingerprints = {name: new_fp}
    # Resume sees the whole grid as undone ...
    redo = st.build_items(
        variants, TINY_ENVS[:1], [TINY_T], grids, TINY_CAP, 8, done, fingerprints=fingerprints
    )
    assert len(redo) == 1 and redo[0].taus == (0.3, 0.7) and redo[0].fingerprint == new_fp
    # ... the old rows are dropped from the CSV ...
    kept, n_stale = st.drop_stale_rows(old, fingerprints)
    assert kept is None and n_stale == len(old)
    # ... and never reach the JSON, even if they were still on disk.
    artifacts = {name: st.load_model(path)}
    assert artifacts[name]["k"] == 2
    stale_only = st.build_thresholds(
        old, artifacts, TINY_ENVS[:1], [TINY_T], grids, "val", 8, fingerprints
    )
    assert stale_only == {}
    rows = []
    for item_rows in st.run_items(redo, workers=1):
        rows += item_rows
    new = pd.DataFrame(rows)
    assert (new["k"] == 2).all() and (new["fingerprint"] == new_fp).all()
    both = pd.concat([old, new], ignore_index=True)
    fresh = st.build_thresholds(
        both, artifacts, TINY_ENVS[:1], [TINY_T], grids, "val", 8, fingerprints
    )
    assert fresh[name]["fingerprint"] == new_fp and fresh[name]["k"] == 2
    assert fresh[name]["curve"] == pytest.approx(
        {"0.3": float(new[new["tau"] == 0.3]["mean_regret"].mean()),
         "0.7": float(new[new["tau"] == 0.7]["mean_regret"].mean())}
    )
    # A run planned against one file must not silently continue on another.
    _retrain(path, k=3)
    with pytest.raises(RuntimeError, match="changed under the run"):
        st.run_item(redo[0])
    # A CSV from before fingerprints existed is entirely unverifiable.
    kept, n_stale = st.drop_stale_rows(old.drop(columns="fingerprint"), fingerprints)
    assert kept is None and n_stale == len(old)


def test_reservoir_rule_threshold_selection():
    """The hand rule is selected on its own grid and written without tau_off / k."""
    variants: dict[str, Path | None] = {st.RULE_VARIANT: None}
    grids = st.tau_grids(variants, [0.3, 0.7], [0.25, 4.0])
    assert grids == {st.RULE_VARIANT: (0.25, 4.0)}
    fp = st.artifact_fingerprint(None)
    assert fp.startswith("rule:") and fp == st.artifact_fingerprint(None)
    items = st.build_items(variants, TINY_ENVS, [TINY_T], grids, TINY_CAP, 16)
    assert len(items) == len(TINY_ENVS) and items[0].artifact_path is None
    rows: list[dict] = []
    for item_rows in st.run_items(items, workers=1):
        rows += item_rows
    df = pd.DataFrame(rows)
    assert len(df) == len(TINY_ENVS) * 2
    assert (df["variant"] == st.RULE_VARIANT).all() and (df["fingerprint"] == fp).all()
    assert (df["k"] == -1).all() and df["tau_off"].isna().all()
    assert (df[list(st.COUNTER_NAMES)] == -1).all().all()
    assert (df["mean_n_demoted"] >= 0).all()  # the rule exposes last_decision
    # A larger width multiplier searches less.
    for _, grp in df.groupby("env"):
        g = grp.sort_values("tau")
        assert g["mean_search_frac"].iloc[0] >= g["mean_search_frac"].iloc[-1]
    thresholds = st.build_thresholds(
        df, {st.RULE_VARIANT: None}, TINY_ENVS, [TINY_T], grids, "val", 16, {st.RULE_VARIANT: fp}
    )
    entry = thresholds[st.RULE_VARIANT]
    assert entry["kind"] == "rule" and "tau_off" not in entry and "k" not in entry
    assert set(entry["curve"]) == {"0.25", "4"} and entry["tau_val"] in (0.25, 4.0)
    expected = df.groupby("tau")["mean_regret"].mean()
    assert entry["tau_val"] == pytest.approx(float(expected.idxmin()))
    assert entry["taus"] == [0.25, 4.0]
