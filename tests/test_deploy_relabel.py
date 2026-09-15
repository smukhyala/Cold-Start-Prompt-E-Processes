"""Tests for M9 on-policy relabelling (`relabel_onpolicy.py`) and its seed band.

Everything runs the real simulator, harness and labeller on a tiny grid (T=50, two
episodes, two snapshot times, 64 labelling replicates) against a small learned model
saved under the ``clock_quality_evidence_k16`` name, so `policy_table` resolves it as
``phi_k16`` exactly as the study does. The training step needs the oracle corpus and
is skipped without it.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
for _p in (ROOT / "experiments" / "growing_bandits", DEPLOY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cells  # noqa: E402
import relabel_onpolicy as ro  # noqa: E402

from cold_start.growing.deploy import feature_groups as fg  # noqa: E402
from cold_start.growing.deploy.artifacts import load_model, save_model  # noqa: E402
from cold_start.growing.deploy.model_policy import ModelPolicy  # noqa: E402
from cold_start.growing.deploy.pairwise_table import get_pairwise_table  # noqa: E402
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.labeling import Snapshot, materialize  # noqa: E402
from cold_start.growing.schema import validate_columns  # noqa: E402
from cold_start.growing.search_policies import DecisionContext  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

HAVE_CORPUS = bool(list((ROOT / "data" / "oracle_labels").glob("part-*.parquet")))
needs_corpus = pytest.mark.skipif(not HAVE_CORPUS, reason="oracle corpus not present")

#: T=40 is not a study horizon (`cells.cell_id` refuses it); 50 is the smallest that is.
TINY_T = 50
TINY_M = 2
TINY_TIMES = 2
TINY_MAX_REPLICATES = 64
TINY_ENVS = ("beta_good_common", "tail_b2.0_mu1.0_c1.0")
#: CLOCK plus the pairwise e-process column, so the continuation exercises the cached
#: log-e table path that the real ``phi_k16`` uses.
TINY_FEATURES = list(fg.CLOCK) + list(fg.EVIDENCE_LOGE)


def _clock_row(t: int, horizon: int, k: int) -> list[float]:
    remaining = horizon - t
    return [
        float(t), float(horizon), float(remaining), remaining / horizon, t / horizon, float(k),
        k / t if t else float(k), k / horizon, float(np.log(max(t, 1))), float(np.log(k)),
        k / np.sqrt(max(t, 1)),
    ]


def _fit_tiny_pipeline(seed: int = 0, n_rows: int = 3000):
    """A logistic rule that searches early and stops as arms accumulate; log-e is noise."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_rows):
        horizon = int(rng.choice([50, 100, 200]))
        t = int(rng.integers(2, horizon))
        k = int(rng.integers(1, min(t, 64) + 1))
        row = _clock_row(t, horizon, k) + [float(rng.normal())]
        score = row[3] - 0.15 * row[10] + 0.3 * rng.normal()
        rows.append(row)
        labels.append(score > 0.35)
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("m9")
    models = tmp / "models"
    save_model(
        models / f"{ro.PHI_VARIANT}.joblib",
        {
            "pipeline": _fit_tiny_pipeline(),
            "features": TINY_FEATURES,
            "k": 16,
            "tau": 0.5,
            "meta": {"variant": ro.PHI_VARIANT, "subset": {}},
        },
    )
    return {
        "out": tmp / "out",
        "models": models,
        "thresholds": tmp / "thresholds.json",  # absent: the artifact's tau is deployed
        "pairwise": tmp / "pairwise_tables",
        "table": CSTable.load_or_build(TINY_T, alpha=0.05),
    }


@pytest.fixture(scope="module")
def harvested(workspace) -> list[dict]:
    return ro.harvest(
        workspace["out"],
        models_dir=workspace["models"],
        thresholds_path=workspace["thresholds"],
        envs=TINY_ENVS,
        horizons=[TINY_T],
        n_replicates=TINY_M,
        n_times=TINY_TIMES,
        pairwise_cache_dir=workspace["pairwise"],
    )


@pytest.fixture(scope="module")
def labelled(workspace, harvested) -> dict:
    return ro.label(
        workspace["out"],
        models_dir=workspace["models"],
        thresholds_path=workspace["thresholds"],
        workers=2,
        max_replicates=TINY_MAX_REPLICATES,
        limit_states=2,
        pairwise_cache_dir=workspace["pairwise"],
        write_diagnostics=False,
    )


# ---- seed band ------------------------------------------------------------------------


def test_onpolicy_seed_band_sits_between_test_and_corpus():
    top = (cells.N_CELLS - 1) * cells.CELL_STRIDE
    assert cells.SPLIT_BASE["onpolicy"] == 15_000_000
    assert cells.SPLIT_BASE["onpolicy"] > cells.SPLIT_BASE["test"] + top
    assert cells.SPLIT_BASE["onpolicy"] + top < 20_262_460
    assert cells.CORPUS_SEED_MIN == min(cells.corpus_seed_set()) == 20_262_460
    # The seeds the harvest AND the labelling batches consume, for the whole M9 grid.
    specs = ro.onpolicy_cells(
        cells.MAIN_ENVS, cells.HORIZONS, cap=64, n_replicates=32, alpha=0.05
    )
    assert len(specs) == 40
    assert all(s.base_seed >= 15_000_000 for s in specs)
    seeds = set()
    for s in specs:
        seeds |= ro.consumed_seeds(s)
    assert len(seeds) == 40 * (1 + 32 * (1 + cells.CORPUS_MAX_BATCHES)) - 40
    cells.assert_seed_disjointness(seeds)
    # A real corpus seed must still be rejected by the same guard.
    with pytest.raises(ValueError, match="corpus"):
        cells.assert_seed_disjointness([cells.CORPUS_SEED + 31 * 50])


def test_onpolicy_policies_and_variants_are_registered():
    import policy_table as pt
    import train_policies as tp

    phi = pt.POLICIES["phi_k16"]
    assert set(pt.ONPOLICY_POLICIES) == set(ro.ONPOLICY_VARIANT_NAMES) == {ro.VARIANT_UNION, ro.VARIANT_ONLY}
    for name in ro.ONPOLICY_VARIANT_NAMES:
        # The policy-table twin: phi_k16's entry with the artifact swapped.
        entry = pt.POLICIES[name]
        assert set(entry) == set(phi)
        assert entry["kind"] == "model" and entry["group"] == "learned"
        assert {k: v for k, v in entry["params"].items() if k != "artifact"} == {
            k: v for k, v in phi["params"].items() if k != "artifact"
        }
        assert entry["params"]["artifact"] == name
        assert entry["requires"] == [f"artifact:{name}", "thresholds"]
        assert pt.variant_of(name) == name and name in tp.VARIANTS
        assert name in pt.TEST_POLICIES["A"]
        for test in ("B", "C", "D", "robust", "cap", "smoke"):
            assert name not in pt.TEST_POLICIES[test], test
        # The trainer's registration: the primary configuration plus the on-policy marker.
        spec = tp.VARIANTS[name]
        assert spec["feature_set"] == "clock_quality_evidence" and spec["k"] == 16
        assert spec["estimator"] == "logit" and spec["row_filter"] == "decided"
        assert spec["subset"] == {"onpolicy": name.rsplit("_", 1)[-1]}
        assert tp.requires_onpolicy_rows(spec)
    assert ro.META_POLICY == tp.ONPOLICY_META_POLICY
    assert set(pt.CORPUS_POLICIES) == set(pt.ALL_POLICIES) - set(pt.ONPOLICY_POLICIES)
    assert not any(tp.requires_onpolicy_rows(tp.VARIANTS[pt.variant_of(p)])
                   for p in pt.CORPUS_POLICIES if pt.variant_of(p) is not None)


def test_corpus_trainer_skips_the_onpolicy_variants():
    import train_policies as tp

    trainable, skipped = tp.corpus_trainable(list(tp.VARIANTS))
    assert skipped == [ro.VARIANT_UNION, ro.VARIANT_ONLY]
    assert trainable + skipped == list(tp.VARIANTS)
    assert tp.corpus_trainable(list(tp.QUICK_VARIANTS)) == (list(tp.QUICK_VARIANTS), [])
    # The subset itself: undefined on a corpus-only frame, and the on-policy rows for "only".
    corpus_like = pd.DataFrame({"meta_policy": ["sqrt", "bracket", "cbrt"], "meta_env": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="requires on-policy rows"):
        tp.subset_mask(corpus_like, {"onpolicy": "union"})
    mixed = pd.DataFrame({"meta_policy": ["sqrt", ro.META_POLICY, "cbrt", ro.META_POLICY]})
    assert tp.subset_mask(mixed, {"onpolicy": "union"}).tolist() == [True, True, True, True]
    assert tp.subset_mask(mixed, {"onpolicy": "only"}).tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match="unknown onpolicy subset"):
        tp.subset_mask(mixed, {"onpolicy": "all"})


def test_snapshot_times_follow_the_corpus_schedule_and_can_add_pre_cap_times():
    spec = cells.make_cell("onpolicy", "beta_good_common", 1000, 64, 32)
    times = ro.snapshot_times(spec, 8)
    assert times == tuple(sorted(set(times)))
    assert 1 <= len(times) <= 8 and all(3 <= t <= 999 for t in times)
    assert times == ro.snapshot_times(spec, 8)  # seeded per cell
    with_early = ro.snapshot_times(spec, 8, early_times=3)
    assert set(times) <= set(with_early)
    extra = sorted(set(with_early) - set(times))
    assert 1 <= len(extra) <= 3 and all(3 <= t < 62 for t in extra)
    # At T=50 the pre-cap window is the whole horizon; times still stay inside [3, T-1].
    short = cells.make_cell("onpolicy", "beta_good_common", 50, 64, 32)
    assert all(3 <= t <= 49 for t in ro.snapshot_times(short, 8, early_times=3))


# ---- branch reset ---------------------------------------------------------------------


def test_branch_reset_policy_resets_exactly_when_the_state_changes(workspace):
    table = workspace["table"]
    artifact = load_model(workspace["models"] / f"{ro.PHI_VARIANT}.joblib")
    pairwise = get_pairwise_table(TINY_T, cache_dir=workspace["pairwise"])
    built: list[ModelPolicy] = []

    def make_inner(n_replicates: int) -> ModelPolicy:
        policy = make_policy(
            "model", horizon=TINY_T, n_replicates=n_replicates, table=table,
            params={"artifact": artifact, "tau": 0.5, "k": 16}, pairwise=pairwise,
        )
        built.append(policy)
        return policy

    snap = Snapshot(
        n=np.array([3, 2]), successes=np.array([2, 1]), mu=np.array([0.6, 0.4]),
        t=5, horizon=TINY_T, n_draws=2, base_seed=15_000_001,
    )
    branch_s = materialize(snap, 4, table)
    branch_r = materialize(snap, 4, table)
    wrapper = ro._BranchResetPolicy(make_inner)
    ctx = DecisionContext(t=5, horizon=TINY_T)

    # Three consecutive calls on the SEARCH branch: one reset, a standing commitment.
    for _ in range(3):
        action = wrapper.should_search(branch_s, ctx)
    assert action.shape == (4,) and action.dtype == bool
    assert wrapper.n_resets == 1 and len(built) == 1
    assert wrapper.inner.n_decisions == 4 and wrapper.inner.n_committed_steps == 8
    assert np.all(wrapper.inner.commit_left == 16 - 3)

    # The REFINE branch is a different object: counters cleared, a fresh decision.
    wrapper.should_search(branch_r, ctx)
    assert wrapper.n_resets == 2 and len(built) == 1  # same M: reset, not rebuilt
    assert wrapper.inner.n_decisions == 4 and wrapper.inner.n_committed_steps == 0
    assert np.all(wrapper.inner.commit_left == 16 - 1)

    # Back to the SEARCH branch object: reset again; same object twice: no reset.
    wrapper.should_search(branch_s, ctx)
    wrapper.should_search(branch_s, ctx)
    assert wrapper.n_resets == 3

    # A batch of a different size needs a differently sized inner policy.
    bigger = materialize(snap, 8, table)
    wrapper.should_search(bigger, ctx)
    assert wrapper.n_resets == 4 and len(built) == 2 and wrapper.inner.M == 8


# ---- harvest --------------------------------------------------------------------------


def test_harvest_detaches_on_policy_snapshots_on_fresh_seeds(harvested, workspace):
    assert [r["cell"] for r in harvested] == [f"{e}_T{TINY_T}" for e in TINY_ENVS]
    _, snap_dir, manifest = ro.harvest_paths(workspace["out"])
    assert manifest.exists()
    for rec in harvested:
        spec = cells.make_cell("onpolicy", rec["env_id"], TINY_T, 64, TINY_M)
        assert rec["base_seed"] == spec.base_seed >= 15_000_000
        assert rec["policy"] == "phi_k16"
        assert rec["params"]["tau"] == 0.5 and rec["params"]["tau_source"] == "artifact_tau"
        assert rec["params"]["artifact"] == str(workspace["models"] / f"{ro.PHI_VARIANT}.joblib")
        assert rec["params"]["k"] is None  # phi_k16 takes the artifact's k (16)
        assert 1 <= len(rec["times"]) <= TINY_TIMES
        assert all(3 <= t <= TINY_T - 1 for t in rec["times"])
        assert rec["n_at_cap"] == 0
        assert rec["n_snapshots"] == len(rec["times"]) * TINY_M
        with open(rec["pickle"], "rb") as fh:
            snaps = pickle.load(fh)
        assert len(snaps) == rec["n_snapshots"]
        allowed = ro.consumed_seeds(spec)
        for snap in snaps:
            assert snap.k < 64
            assert int(snap.n.sum()) == snap.t
            assert snap.t in rec["times"]
            history = snap.meta["history"]
            assert len(history.decisions) == snap.t - 2
            assert len(history.best_mean_trace) == snap.t - 2
            assert snap.meta["policy"] == "phi_k16" and snap.meta["allocation"] == "lucb"
            assert snap.base_seed in allowed
            assert snap.base_seed == spec.base_seed + 7919 * snap.meta["replicate"]
        cells.assert_seed_disjointness([s.base_seed for s in snaps])
    # Resume: nothing re-runs, the same records come back.
    again = ro.harvest(
        workspace["out"],
        models_dir=workspace["models"],
        thresholds_path=workspace["thresholds"],
        envs=TINY_ENVS,
        horizons=[TINY_T],
        n_replicates=TINY_M,
        n_times=TINY_TIMES,
        pairwise_cache_dir=workspace["pairwise"],
    )
    assert [r["finished_at"] for r in again] == [r["finished_at"] for r in harvested]


# ---- label ----------------------------------------------------------------------------


def test_label_writes_corpus_schema_rows_with_finite_k16_labels(labelled, workspace):
    assert labelled["n_failed"] == 0 and labelled["n_run"] == 2
    labels_dir = labelled["labels_dir"]
    parts = sorted(labels_dir.glob("part-*.parquet"))
    assert len(parts) == 2
    records = ro.latest_by(ro.read_manifest(labels_dir / ro.LABEL_MANIFEST), "part")
    assert set(records) == {f"onpolicy_{e}_T{TINY_T}_c00" for e in TINY_ENVS}
    frame = ro.load_onpolicy_rows(labels_dir)
    assert len(frame) == 4  # 2 cells x limit_states=2
    validate_columns(list(frame.columns))
    assert np.isfinite(frame["label_A_k16"].to_numpy(dtype=np.float64)).all()
    assert np.array_equal(frame["label_A"].to_numpy(), frame["label_A_k16"].to_numpy())
    assert np.all(frame["label_M_k16"] == TINY_MAX_REPLICATES)
    assert np.all(frame["label_se_k16"] >= 0.0)
    assert set(frame["meta_policy"]) == {"phi_k16_onpolicy"}
    assert set(frame["meta_allocation"]) == {"lucb"}
    assert set(frame["meta_shard"]) == {f"onpolicy_{e}_T{TINY_T}" for e in TINY_ENVS}
    assert set(frame["meta_family"]) == {"A", "B"}
    assert np.all(frame["meta_horizon"] == TINY_T)
    assert np.all(frame["meta_n_undefined_in_shard"] == 0)
    assert {"label_diag_search_p_new_plausible", "label_diag_refine_p_eliminated_one"} <= set(frame.columns)
    assert not {"label_A_k1", "label_A_k4", "label_se_k1", "label_se_k4"} & set(frame.columns)
    # Every deployable column the trainer could ask for is present, as a corpus row has it.
    assert set(fg.ALL_DEPLOYABLE) <= set(frame.columns)
    assert all(np.isfinite(frame[c].to_numpy(dtype=np.float64)).all() for c in fg.ALL_DEPLOYABLE)
    for rec in records.values():
        assert rec["n_rows"] == 2 and rec["n_undefined"] == 0
        assert rec["params"]["tau"] == 0.5 and rec["commit_steps"] == 16
        assert Path(rec["parquet"]).exists()
    # Resume: everything is complete, nothing runs.
    again = ro.label(
        workspace["out"],
        models_dir=workspace["models"],
        thresholds_path=workspace["thresholds"],
        workers=1,
        max_replicates=TINY_MAX_REPLICATES,
        limit_states=2,
        pairwise_cache_dir=workspace["pairwise"],
        write_diagnostics=False,
    )
    assert again["n_run"] == 0 and again["n_parts"] == 2


def test_label_refuses_states_harvested_under_a_different_policy(labelled, workspace, tmp_path):
    # A retrained artifact under the same name changes the continuation: refuse.
    models = tmp_path / "models"
    save_model(
        models / f"{ro.PHI_VARIANT}.joblib",
        {"pipeline": _fit_tiny_pipeline(seed=1), "features": TINY_FEATURES, "k": 16,
         "tau": 0.5, "meta": {"variant": ro.PHI_VARIANT, "subset": {}}},
    )
    with pytest.raises(RuntimeError, match="Re-harvest"):
        ro.label(
            workspace["out"],
            models_dir=models,
            thresholds_path=workspace["thresholds"],
            workers=1,
            max_replicates=TINY_MAX_REPLICATES,
            limit_states=2,
            pairwise_cache_dir=workspace["pairwise"],
            write_diagnostics=False,
        )


def test_label_resume_is_keyed_to_the_harvest(workspace, tmp_path):
    """A re-harvest that changes the snapshot set must invalidate the chunks labelled
    from the old pickle, even when the chunk boundaries did not move."""
    out = tmp_path / "out"
    env = TINY_ENVS[0]
    common = dict(
        models_dir=workspace["models"],
        thresholds_path=workspace["thresholds"],
        pairwise_cache_dir=workspace["pairwise"],
    )
    first = ro.harvest(out, envs=(env,), horizons=[TINY_T], n_replicates=TINY_M, n_times=TINY_TIMES, **common)
    res = ro.label(out, workers=1, max_replicates=TINY_MAX_REPLICATES, limit_states=2,
                   write_diagnostics=False, **common)
    assert res["n_run"] == 1 and res["n_failed"] == 0
    labels_dir = res["labels_dir"]
    part = f"onpolicy_{env}_T{TINY_T}_c00"
    before = ro.latest_by(ro.read_manifest(labels_dir / ro.LABEL_MANIFEST), "part")[part]
    assert before["harvest_sha"] == first[0]["pickle_sha"]
    old_t = sorted(ro.load_onpolicy_rows(labels_dir)["f_t"].tolist())

    # Re-harvest with an extra pre-cap time: the resume check sees new times and reruns
    # the cell; the pickle now holds a different (larger) snapshot set at the same path.
    second = ro.harvest(out, envs=(env,), horizons=[TINY_T], n_replicates=TINY_M, n_times=TINY_TIMES,
                        early_times=1, **common)
    assert second[0]["times"] != first[0]["times"] and 3 in second[0]["times"]
    assert second[0]["pickle"] == first[0]["pickle"]
    assert second[0]["pickle_sha"] != first[0]["pickle_sha"]

    again = ro.label(out, workers=1, max_replicates=TINY_MAX_REPLICATES, limit_states=2,
                     write_diagnostics=False, **common)
    assert again["n_run"] == 1 and again["n_failed"] == 0  # the same chunk, re-labelled
    after = ro.latest_by(ro.read_manifest(labels_dir / ro.LABEL_MANIFEST), "part")[part]
    assert after["harvest_sha"] == second[0]["pickle_sha"] != before["harvest_sha"]
    new_t = sorted(ro.load_onpolicy_rows(labels_dir)["f_t"].tolist())
    assert new_t != old_t and new_t[0] == 3.0  # rows now come from the new snapshot set
    # And with nothing changed, nothing runs.
    third = ro.label(out, workers=1, max_replicates=TINY_MAX_REPLICATES, limit_states=2,
                     write_diagnostics=False, **common)
    assert third["n_run"] == 0


# ---- train ----------------------------------------------------------------------------


def _fabricate_onpolicy_parts(labels_dir: Path, n_rows: int = 300) -> int:
    """Corpus rows rewritten as on-policy rows (schema only; the labels are cp0's), so the
    on-policy-only model has both classes and several environments to fold over."""
    from corpus import load_corpus

    corpus = load_corpus(ROOT / "data" / "oracle_labels", max_shards=3)
    rows = corpus.iloc[:: max(1, len(corpus) // n_rows)].head(n_rows).copy()
    rows = rows.drop(columns=[c for c in rows.columns if c.startswith(("label_A_k1", "label_se_k1", "label_M_k1", "label_A_k4", "label_se_k4", "label_M_k4"))] + ["meta_trajectory"])
    rows["meta_shard"] = "onpolicy_" + rows["meta_env"].astype(str) + "_T" + rows["meta_horizon"].astype(int).astype(str)
    rows["meta_policy"] = ro.META_POLICY
    rows["meta_allocation"] = "lucb"
    rows.to_parquet(labels_dir / "part-onpolicy_fabricated_c00.parquet", index=False)
    return len(rows)


@needs_corpus
def test_train_saves_both_retrained_models_and_merges_the_trainer_tables(labelled, workspace):
    out = workspace["out"]
    labels_dir = labelled["labels_dir"]
    n_fab = _fabricate_onpolicy_parts(labels_dir)
    result = ro.train(
        out,
        corpus_dir=ROOT / "data" / "oracle_labels",
        labels_dir=labels_dir,
        models_dir=workspace["models"],
        n_splits_union=3,
        n_splits_only=2,
        n_jobs=1,
        max_shards=3,
    )
    assert result["n_onpolicy_rows"] == n_fab + 4
    onpol = ro.load_onpolicy_rows(labels_dir)
    for name, subset in ((ro.VARIANT_UNION, "union"), (ro.VARIANT_ONLY, "only")):
        path = out / "models" / f"{name}.joblib"
        art = load_model(path)
        assert art["k"] == 16
        assert list(art["features"]) == list(fg.FEATURE_SETS["clock_quality_evidence"])
        assert art["meta"]["variant"] == name
        assert art["meta"]["subset"] == {"onpolicy": subset}
        assert art["meta"]["n_onpolicy_rows"] == n_fab + 4
        assert art["meta"]["n_corpus_rows"] == result["n_corpus_rows"]
        assert 0.05 <= art["tau"] <= 0.95
        p = art["pipeline"].predict_proba(onpol[list(art["features"])].to_numpy(dtype=np.float64))[:, 1]
        assert p.shape == (len(onpol),) and np.all((p >= 0) & (p <= 1))
        # It deploys as a phi_k16-style policy: 16-round commitment from the artifact.
        policy = make_policy(
            "model", horizon=TINY_T, n_replicates=3, table=workspace["table"],
            params={"artifact": art, "tau": None, "k": None},
            pairwise=get_pairwise_table(TINY_T, cache_dir=workspace["pairwise"]),
        )
        assert policy.k == 16 and policy.tau == art["tau"]
    n_only = int(result["artifacts"][ro.VARIANT_ONLY]["meta"]["n_rows"])
    assert 0 < n_only <= n_fab + 4
    assert int(result["artifacts"][ro.VARIANT_UNION]["meta"]["n_rows"]) > n_only

    metrics = pd.read_csv(out / "offline_metrics.csv")
    by_variant = {v: set(g["grouping"]) for v, g in metrics.groupby("variant")}
    assert {"meta_env", "meta_horizon", "meta_policy", "meta_family", "within_regime",
            "onpolicy_rows_oof", "corpus_rows_oof"} <= by_variant[ro.VARIANT_UNION]
    assert {"meta_env", "excluded_rows"} <= by_variant[ro.VARIANT_ONLY]
    env_rows = metrics[(metrics["variant"] == ro.VARIANT_UNION) & (metrics["grouping"] == "meta_env")]
    assert len(env_rows) == 1 and np.isfinite(env_rows["auc"].iloc[0])
    assert json.loads(env_rows["subset"].iloc[0]) == {"onpolicy": "union"}
    onpol_rows = metrics[(metrics["variant"] == ro.VARIANT_UNION) & (metrics["grouping"] == "onpolicy_rows_oof")]
    assert int(onpol_rows["n_rows"].iloc[0]) <= n_fab + 4
    ex = metrics[(metrics["variant"] == ro.VARIANT_ONLY) & (metrics["grouping"] == "excluded_rows")]
    assert int(ex["n_rows"].iloc[0]) > 0 and np.isfinite(ex["auc"].iloc[0])

    variants = json.loads((out / "variants.json").read_text())
    assert variants[ro.VARIANT_UNION]["subset"] == {"onpolicy": "union"}
    assert variants[ro.VARIANT_ONLY]["features"] == list(fg.FEATURE_SETS["clock_quality_evidence"])

    shift = pd.read_csv(out / ro.MODEL_SHIFT_CSV)
    coef = shift[shift["section"] == "coef"].set_index("key")
    assert set(fg.FEATURE_SETS["clock_quality_evidence"]) | {"intercept"} <= set(coef.index)
    assert np.isfinite(coef.loc["f_remaining_frac", ["phi", "union", "only"]].to_numpy(dtype=float)).all()
    assert np.isnan(coef.loc["f_leader_mean", "phi"])  # the tiny phi does not read it
    agree = shift[shift["section"] == "agreement"].set_index("key")
    assert set(agree.index) == {"corpus_rows", "onpolicy_rows"}
    assert ((agree[["union", "only"]] >= 0) & (agree[["union", "only"]] <= 1)).all().all()
    assert int(agree.loc["onpolicy_rows", "n_rows"]) == n_fab + 4


@needs_corpus
def test_label_diagnostics_compare_on_policy_with_the_corpus(labelled, harvested, workspace):
    artifact = load_model(workspace["models"] / f"{ro.PHI_VARIANT}.joblib")
    frame = ro.write_label_diagnostics(
        workspace["out"],
        labels_dir=labelled["labels_dir"],
        corpus_dir=ROOT / "data" / "oracle_labels",
        artifact=artifact,
        tau=0.5,
        harvest_records=harvested,
        max_shards=3,
    )
    assert (workspace["out"] / ro.DIAGNOSTICS_CSV).exists()
    assert set(frame["scope"]) == {"onpolicy", "corpus_schedule", "corpus_all"}
    pooled = frame[frame["horizon"] == "all"].set_index("scope")
    assert len(pooled) == 3
    assert pooled.loc["corpus_all", "n_states"] >= pooled.loc["corpus_schedule", "n_states"] > 0
    assert 0.0 <= pooled.loc["corpus_schedule", "p_search_given_decided"] <= 1.0
    assert 0.0 <= pooled.loc["corpus_schedule", "phi_agree_frac"] <= 1.0
    # The fabricated parts of the training test may share this directory, so only the
    # real T=50 rows are pinned here.
    onpol = frame[frame["scope"] == "onpolicy"]
    assert {"all", str(TINY_T)} <= set(onpol["horizon"].astype(str))
    at_t = onpol[onpol["horizon"].astype(str) == str(TINY_T)].iloc[0]
    assert int(at_t["n_states"]) >= 4 and int(at_t["n_undefined"]) == 0
    assert int(onpol[onpol["horizon"] == "all"]["n_undefined"].iloc[0]) == 0
    assert 0.0 <= float(at_t["tie_frac"]) <= 1.0
    assert 0.0 <= float(at_t["phi_search_rate"]) <= 1.0
    for col in ("n_states", "n_decided", "tie_frac", "mean_abs_A", "mean_se", "phi_agree_frac", "phi_auc"):
        assert col in frame.columns
