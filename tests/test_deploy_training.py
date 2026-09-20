"""Tests for the deployment-study trainer (`experiments/growing_bandits/deploy/train_policies.py`).

Two kinds of test. The structural ones run without data and guard the hygiene rule
that no oracle / label / meta column can reach a deployable model: they walk the
`VARIANTS` table itself rather than trusting a docstring. The behavioural ones run the
trainer in `--quick` mode (first three shards, two folds, three variants) against the
real corpus and check determinism, the artifact contract and the written tables; they
skip when the corpus is not present.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import corpus  # noqa: E402
import train_policies as tp  # noqa: E402

from cold_start.growing.deploy import artifacts, transforms  # noqa: E402
from cold_start.growing.deploy import feature_groups as fg  # noqa: E402

HAVE_CORPUS = bool(glob.glob(str(ROOT / "data" / "oracle_labels" / "part-*.parquet")))
needs_corpus = pytest.mark.skipif(not HAVE_CORPUS, reason="oracle corpus not present")

BRIEF_VARIANTS = {
    "clock_k16",
    "clock_quality_k16",
    "clock_quality_cs_k16",
    "clock_quality_evidence_k16",
    "all71_k16",
    "clock_quality_evidence_k1",
    "clock_quality_evidence_k4",
    "clock_quality_evidence_k16_hgb",
    "clock_quality_evidence_k16_weighted",
    "clock_quality_evidence_k16_noambig",
    "clock_quality_evidence_k16_notrunc",
    "clock_quality_evidence_k16_lucb",
    "clock_quality_evidence_k16_nopolicy",
    "clock_quality_evidence_k16_famA_only",
    "clock_quality_evidence_k16_famB_only",
    "clock_quality_evidence_k16_noT1000",
    "clock_quality_evidence_k16_noT200",
    "clock_sf_quality_cs_k16",
    "clock_sf_quality_cs_k16_noT1000",
    "reservoir_rule_k16",
    "legacy_E_k16_weighted",
}


def _string_leaves(obj):
    """Every string anywhere in a nested dict/list/tuple, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _string_leaves(k)
            yield from _string_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _string_leaves(v)


# ---- structural: the variants table --------------------------------------------------


def test_variants_table_covers_the_brief():
    assert BRIEF_VARIANTS <= set(tp.VARIANTS)
    for name, v in tp.VARIANTS.items():
        assert v["estimator"] in tp.ESTIMATORS, name
        assert v["row_filter"] in tp.ROW_FILTERS, name
        assert v["k"] in (1, 4, 16), name
        assert v["feature_set"] in tp.FEATURE_SETS, name


def test_every_variant_feature_list_is_deployable():
    for name, v in tp.VARIANTS.items():
        feats = tp.feature_list(v)
        assert len(feats) == len(set(feats)), name
        fg.assert_deployable(list(feats))  # raises on oracle_/label_/meta_ or unknown columns


def test_variants_table_never_names_a_forbidden_column():
    # Structural: walk every string in the table (feature names, subset values, keys).
    # Subsets select rows by metadata VALUES ("aggressive", "A", 1000), never by column.
    leaves = list(_string_leaves(tp.VARIANTS))
    assert leaves
    bad = [s for s in leaves if s.startswith(fg.FORBIDDEN_PREFIXES)]
    assert bad == []
    assert tp.feature_list(tp.VARIANTS["reservoir_rule_k16"]) == transforms.RESERVOIR_RULE_FEATURES
    assert tp.feature_list(tp.VARIANTS["legacy_E_k16_weighted"]) == fg.ALL_DEPLOYABLE
    assert len(tp.feature_list(tp.VARIANTS["all71_k16"])) == 71


def test_feature_hygiene_classes_follow_the_brief():
    cols = list(fg.ALL_DEPLOYABLE) + ["oracle_mu_star", "label_A_k16", "meta_policy"]
    table = tp.feature_hygiene_table(cols, tp.VARIANTS).set_index("column")
    assert table.loc["f_t", "availability_class"] == "a"
    assert table.loc["f_T", "availability_class"] == "e"
    assert table.loc["f_remaining_frac", "availability_class"] == "e"
    assert table.loc["f_K", "availability_class"] == "a+d"
    assert table.loc["f_K_over_T", "availability_class"] == "e+d"
    assert table.loc["f_log_K", "availability_class"] == "a+d"
    assert table.loc["f_n_singletons", "availability_class"] == "a+d"
    assert table.loc["f_mean_n", "availability_class"] == "a+d"
    assert table.loc["est_beta_a", "availability_class"] == "b"
    assert table.loc["f_leader_mean", "availability_class"] == "a"
    assert table.loc["f_log_e_pair", "availability_class"] == "a"
    assert table.loc["f_new_arms_last_10", "availability_class"] == "b+d"
    assert table.loc["oracle_mu_star", "availability_class"] == "c"
    assert table.loc["label_A_k16", "availability_class"] == "c"
    assert table.loc["meta_policy", "availability_class"] == "e"
    # nothing oracle/label/meta is deployed anywhere; history only in the two 71-column models
    forbidden = table[table["group"].isin(["ORACLE", "LABEL", "META"])]
    assert (forbidden["deployed_in"] == "").all()
    assert set(table.loc["f_new_arms_last_10", "deployed_in"].split(",")) == {
        "all71_k16", "legacy_E_k16_weighted"
    }
    assert table.loc["f_log_e_pair", "group"] == "EVIDENCE_LOGE"
    assert table.loc["f_leader_lcb", "group"] == "EVIDENCE_CS"


# ---- reservoir rule ---------------------------------------------------------------------


def test_beta_excess_mean_matches_numeric_integration():
    from scipy.integrate import quad
    from scipy.stats import beta

    rng = np.random.default_rng(20260914)
    for _ in range(20):
        a, b = rng.uniform(0.5, 10.0, size=2)
        c = rng.uniform(0.05, 0.95)
        numeric, err = quad(lambda x, a=a, b=b: beta.sf(x, a, b), c, 1.0)
        closed = float(transforms.beta_excess_mean(a, b, c))
        assert abs(closed - numeric) < 1e-8 + 10 * err
        assert closed >= 0.0
    # vectorised and appended as the fifth column of the pipeline input
    X = np.column_stack(
        [rng.random(5), rng.random(5), rng.random(5), rng.random(5),
         rng.uniform(1, 5, 5), rng.uniform(1, 5, 5), rng.random(5)]
    )
    Z = transforms.reservoir_rule_features(X)
    assert Z.shape == (5, 5)
    assert np.array_equal(Z[:, :4], X[:, :4])
    assert np.allclose(Z[:, 4], transforms.beta_excess_mean(X[:, 4], X[:, 5], X[:, 6]))
    # the trainer must pickle the importable function, not a private copy of it
    assert tp.reservoir_rule_features is transforms.reservoir_rule_features
    assert transforms.reservoir_rule_features.__module__ == "cold_start.growing.deploy.transforms"


# ---- corpus masks ----------------------------------------------------------------------


def test_truncated_or_demoted_mask():
    df = pd.DataFrame(
        {"f_remaining_budget": [3.0, 100.0, 100.0, 16.0], "f_K": [10.0, 60.0, 10.0, 48.0]}
    )
    # row 0: 3 rounds left -> truncated at k=4 (and k=16); row 1: K=60 -> demoted at k=16 only
    # (K + 4 == 64 is not over the cap); row 3: exactly 16 rounds left and K + 16 == 64 -> kept.
    assert corpus.truncated_or_demoted_mask(df, 4).tolist() == [True, False, False, False]
    assert corpus.truncated_or_demoted_mask(df, 16).tolist() == [True, True, False, False]


def test_label_columns_and_row_masks():
    assert corpus.label_columns(16) == ("label_A_k16", "label_se_k16")
    with pytest.raises(ValueError):
        corpus.label_columns(8)
    df = pd.DataFrame(
        {
            "label_A_k4": [0.0, 0.002, -0.05, 0.01],
            "label_se_k4": [0.0, 0.005, 0.001, 0.01],
            "meta_allocation": ["lucb", "ucb", "lucb", "racing"],
        }
    )
    assert corpus.decided_mask(df, 4).tolist() == [False, True, True, True]
    assert corpus.ambiguous_mask(df, 4).tolist() == [False, True, False, False]
    assert corpus.lucb_only_mask(df).tolist() == [True, False, True, False]


def test_subset_mask_selects_by_metadata_values():
    df = pd.DataFrame(
        {
            "meta_policy": ["sqrt", "aggressive", "random", "cbrt"],
            "meta_family": ["A", "B", "B", "A"],
            "meta_horizon": [200.0, 1000.0, 50.0, 1000.0],
        }
    )
    assert tp.subset_mask(df, {}).tolist() == [True] * 4
    assert tp.subset_mask(df, {"exclude_policies": ["aggressive", "random"]}).tolist() == [
        True, False, False, True
    ]
    assert tp.subset_mask(df, {"families": ["A"]}).tolist() == [True, False, False, True]
    assert tp.subset_mask(df, {"exclude_horizons": [1000]}).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        tp.subset_mask(df, {"meta_policy": ["sqrt"]})


def test_tau_selection_and_balanced_accuracy():
    truth = np.array([True, True, True, False, False, False])
    score = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    assert tp.balanced_accuracy(truth, score > 0.5) == pytest.approx(0.5 * (2 / 3 + 2 / 3))
    tau, bal = tp.select_tau(truth, score)
    assert 0.05 <= tau <= 0.95
    assert bal == pytest.approx(5 / 6)  # tau in [0.3, 0.7) separates all but one
    assert tp.ess(np.ones(10)) == pytest.approx(10.0)


# ---- behavioural: quick run on the real corpus ----------------------------------------------


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    if not HAVE_CORPUS:
        pytest.skip("oracle corpus not present")
    out = tmp_path_factory.mktemp("deploy_quick")
    return tp.main(["--quick", "--out", str(out)])


@needs_corpus
def test_quick_run_writes_every_output(quick_run):
    out = quick_run["out_dir"]
    for f in ("offline_metrics.csv", "offline_diagnostics.csv", "feature_hygiene.csv",
              "reservoir_diagnostics_offline.csv", "variants.json"):
        assert (out / f).exists(), f
    for name in tp.QUICK_VARIANTS:
        assert (out / "models" / f"{name}.joblib").exists()

    metrics = pd.read_csv(out / "offline_metrics.csv")
    assert set(metrics["variant"]) == set(tp.QUICK_VARIANTS)
    assert {"meta_env", "meta_horizon", "meta_policy", "meta_family", "within_regime"} <= set(
        metrics["grouping"]
    )
    env = metrics[metrics["grouping"] == "meta_env"].set_index("variant")
    assert (env["n_splits"] == 2).all()
    assert env["auc"].between(0, 1).all()
    assert env["tau_off"].between(0.05, 0.95).all()
    assert (env["bal_acc_tau_off"] >= env["bal_acc_05"] - 1e-12).all()

    hygiene = pd.read_csv(out / "feature_hygiene.csv")
    assert len(hygiene) == 116
    assert set(hygiene["group"]) == {"CLOCK", "QUALITY", "EVIDENCE_CS", "EVIDENCE_LOGE",
                                     "HISTORY", "ORACLE", "LABEL", "META"}
    assert hygiene["group"].value_counts().to_dict() == {
        "CLOCK": 11, "QUALITY": 24, "EVIDENCE_CS": 26, "EVIDENCE_LOGE": 1, "HISTORY": 9,
        "ORACLE": 12, "LABEL": 25, "META": 8,
    }

    diag = pd.read_csv(out / "offline_diagnostics.csv")
    assert list(diag.columns) == ["diagnostic", "key", "value"]
    kinds = set(diag["diagnostic"])
    assert {"simpson_p_search", "simpson_slope_q3_minus_q1", "policy_coef", "policy_coef_sign",
            "metadata_only", "ess", "label_fraction"} <= kinds
    assert diag[(diag["diagnostic"] == "policy_coef_sign")]["value"].isin([-1.0, 0.0, 1.0]).all()
    assert {"k=1|all", "k=1|decided", "k=16|all", "k=16|decided"} <= set(
        diag[diag["diagnostic"] == "ess"]["key"]
    )
    frac = diag[diag["diagnostic"] == "label_fraction"].set_index("key")["value"]
    assert frac["k=1|truncated"] == 0.0
    assert 0.0 <= frac["k=16|tie"] <= 1.0

    res = pd.read_csv(out / "reservoir_diagnostics_offline.csv")
    assert {"oracle_I_t", "oracle_p_new_beats_best_true", "est_p_new_beats_incumbent",
            "est_I_hat"} <= set(res["score"])
    assert (res["auc_oriented"] >= 0.5).all()
    assert (res["auc_oriented"] == np.maximum(res["auc_raw"], 1 - res["auc_raw"])).all()

    variants = json.loads((out / "variants.json").read_text())
    assert BRIEF_VARIANTS <= set(variants)
    for name in BRIEF_VARIANTS:
        fg.assert_deployable(variants[name]["features"])


@needs_corpus
def test_training_is_deterministic(quick_run, tmp_path):
    second = tp.main(["--quick", "--out", str(tmp_path / "second")])
    a = artifacts.load_model(quick_run["out_dir"] / "models" / "clock_k16.joblib")
    b = artifacts.load_model(second["out_dir"] / "models" / "clock_k16.joblib")
    la = a["pipeline"].named_steps["logisticregression"]
    lb = b["pipeline"].named_steps["logisticregression"]
    assert np.array_equal(la.coef_, lb.coef_)
    assert np.array_equal(la.intercept_, lb.intercept_)
    assert a["tau"] == b["tau"]
    assert a["features"] == b["features"]
    m1 = pd.read_csv(quick_run["out_dir"] / "offline_metrics.csv")
    m2 = pd.read_csv(second["out_dir"] / "offline_metrics.csv")
    pd.testing.assert_frame_equal(m1, m2)


@needs_corpus
def test_saved_artifact_reproduces_in_sample_predictions(quick_run):
    df = corpus.load_corpus(max_shards=3)
    for name in tp.QUICK_VARIANTS:
        art = artifacts.load_model(quick_run["out_dir"] / "models" / f"{name}.joblib")
        assert art["k"] == tp.VARIANTS[name]["k"]
        assert 0.05 <= art["tau"] <= 0.95
        assert art["tau"] == art["meta"]["tau_off"]
        assert list(art["features"]) == list(tp.feature_list(tp.VARIANTS[name]))
        v = tp.VARIANTS[name]
        eligible, _ = tp.eligible_masks(df, v)
        assert art["meta"]["n_rows"] == int(eligible.sum())
        X = tp.design_matrix(df, tuple(art["features"]))[eligible]
        y = df[art["meta"]["label"]].to_numpy(dtype=np.float64)[eligible] > 0
        fresh = tp.make_estimator(v["estimator"], v["feature_set"]).fit(X, y)
        p_saved = art["pipeline"].predict_proba(X)[:, 1]
        p_fresh = fresh.predict_proba(X)[:, 1]
        assert np.allclose(p_saved, p_fresh, atol=1e-9)
        assert tp.auc(y, p_saved) > 0.5
        # the saved pipeline consumes exactly the declared columns, in the declared order
        assert art["pipeline"].n_features_in_ == len(art["features"])


# Runs in a child interpreter with `-I` (isolated: no cwd, no PYTHONPATH, no user site), so
# sys.path is the stdlib, the venv's site-packages and the editable `src/` only -- never
# `experiments/`. Reads one feature row from stdin and prints the artifact's P(SEARCH).
ARTIFACT_LOADER = r"""
import json, sys
src = sys.argv[2]
if src not in sys.path:
    sys.path.insert(0, src)
leaked = [p for p in sys.path if "experiments" in p]
assert not leaked, leaked
assert "train_policies" not in sys.modules
import numpy as np
from cold_start.growing.deploy.artifacts import load_model
art = load_model(sys.argv[1])
x = np.asarray([json.load(sys.stdin)], dtype=float)
p = art["pipeline"].predict_proba(x)[:, 1]
print(json.dumps({"p": float(p[0]), "n_features": len(art["features"]), "k": art["k"]}))
"""


@needs_corpus
def test_every_artifact_loads_in_a_bare_interpreter(quick_run):
    """Regression for the reservoir rule: its transform used to pickle as `train_policies.*`."""
    df = corpus.load_corpus(max_shards=1)
    for name in tp.QUICK_VARIANTS:
        path = quick_run["out_dir"] / "models" / f"{name}.joblib"
        art = artifacts.load_model(path)
        row = tp.design_matrix(df.iloc[:1], tuple(art["features"]))[0]
        expected = float(art["pipeline"].predict_proba(row[None, :])[0, 1])
        proc = subprocess.run(
            [sys.executable, "-I", "-c", ARTIFACT_LOADER, str(path), str(ROOT / "src")],
            input=json.dumps(row.tolist()),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{name}: {proc.stderr}"
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        assert out["n_features"] == len(art["features"])
        assert out["k"] == art["k"]
        assert np.isfinite(out["p"]) and 0.0 <= out["p"] <= 1.0
        assert out["p"] == pytest.approx(expected, abs=1e-12)
