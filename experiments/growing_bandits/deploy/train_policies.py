#!/usr/bin/env python
"""M4: train every learned SEARCH-vs-REFINE variant and write the offline diagnostics.

This reproduces the offline study (`fit_models.py`) with the four things audit C found
wrong put right, and then saves what the deployment harness actually loads:

* **Feature groups are explicit column lists** (`feature_groups.FEATURE_SETS`), never
  substring selectors. The ablation named "time + e-process" in the old script was the
  clock plus a single column.
* **The primary estimator is unweighted.** Precision weights concentrate on the
  near-indifferent states (mean |A| in the top weight quartile is 0.0012) and lower
  held-out AUC; they survive here only as the `_weighted` sensitivity and as the
  reproduction gate for the published 0.647 / 0.7405 numbers.
* **The threshold is selected, not assumed.** `tau_off` maximises out-of-fold balanced
  accuracy on a grid; the 0.5 figure is still reported because RESULTS.md used it.
* **Models are saved whole** (`artifacts.save_model`: scaler + estimator in one sklearn
  pipeline), so deployment never folds coefficients by hand.

Every split holds out whole environments (`GroupKFold` by `meta_env`); rows of one
trajectory are never separated. Metrics are reported pooled over the out-of-fold
predictions (the number that matters for a deployed threshold) and as a mean over folds
(the convention RESULTS.md used, kept so the reproduction gate compares like with like).

Hygiene: the only columns a variant may read are those in `feature_groups.ALL_DEPLOYABLE`;
`artifacts.save_model` re-asserts this on every artifact, and `tests/test_deploy_training.py`
walks the `VARIANTS` table structurally.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
# Own directory first: the reservoir-rule artifact pickles `train_policies.reservoir_rule_features`
# by reference, so any process that imports this module (under any name) can also unpickle it.
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from corpus import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ambiguous_mask,
    decided_mask,
    label_columns,
    load_corpus,
    lucb_only_mask,
    truncated_or_demoted_mask,
)
from fit_models import fit_with_weights, precision_weights  # noqa: E402

from cold_start.growing.deploy import artifacts  # noqa: E402
from cold_start.growing.deploy import feature_groups as fg  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "results" / "growing_bandits" / "deploy"

# ---- variants -------------------------------------------------------------------

# The reservoir-aware rule (plan P11). The pipeline derives `est_I_hat` from the last three
# columns, so every column the *artifact* declares is a corpus column; see
# `reservoir_rule_features`.
RESERVOIR_RULE_FEATURES: tuple[str, ...] = (
    "est_p_new_beats_incumbent",
    "f_remaining_frac",
    "f_leader_width",
    "f_log_K",
    "est_beta_a",
    "est_beta_b",
    "f_best_mean",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = dict(fg.FEATURE_SETS)
FEATURE_SETS["reservoir_rule"] = RESERVOIR_RULE_FEATURES

ROW_FILTERS = ("decided", "noambig", "notrunc", "lucb")
ESTIMATORS = ("logit", "logit_w", "hgb")

CQE = "clock_quality_evidence"


def _v(
    feature_set: str,
    k: int = 16,
    estimator: str = "logit",
    row_filter: str = "decided",
    subset: dict | None = None,
) -> dict:
    return {
        "feature_set": feature_set,
        "k": int(k),
        "estimator": estimator,
        "row_filter": row_filter,
        "subset": dict(subset or {}),
    }


# name -> {feature_set, k, estimator, row_filter, subset}. `subset` restricts the training
# universe by metadata VALUES (never by reading a meta column as a feature); the excluded
# rows become the transfer test reported under grouping `excluded_rows`.
VARIANTS: dict[str, dict] = {
    # feature-set ladder at k=16 (plan P8, P7, P9a, P9b/P6, P10)
    "clock_k16": _v("clock"),
    "clock_quality_k16": _v("clock_quality"),
    "clock_quality_cs_k16": _v("clock_quality_cs"),
    "clock_quality_evidence_k16": _v(CQE),
    "all71_k16": _v("all71"),
    # commitment horizons (P4, P5)
    "clock_quality_evidence_k1": _v(CQE, k=1),
    "clock_quality_evidence_k4": _v(CQE, k=4),
    # estimator / weighting / row-filter sensitivities (P12, audit C, register #2, #11)
    "clock_quality_evidence_k16_hgb": _v(CQE, estimator="hgb"),
    "clock_quality_evidence_k16_weighted": _v(CQE, estimator="logit_w"),
    "clock_quality_evidence_k16_noambig": _v(CQE, row_filter="noambig"),
    "clock_quality_evidence_k16_notrunc": _v(CQE, row_filter="notrunc"),
    "clock_quality_evidence_k16_lucb": _v(CQE, row_filter="lucb"),
    # generalization (Tests B/C/D)
    "clock_quality_evidence_k16_nopolicy": _v(
        CQE, subset={"exclude_policies": ["aggressive", "random", "epsilon"]}
    ),
    "clock_quality_evidence_k16_famA_only": _v(CQE, subset={"families": ["A"]}),
    "clock_quality_evidence_k16_famB_only": _v(CQE, subset={"families": ["B"]}),
    "clock_quality_evidence_k16_noT1000": _v(CQE, subset={"exclude_horizons": [1000]}),
    "clock_quality_evidence_k16_noT200": _v(CQE, subset={"exclude_horizons": [200]}),
    "clock_sf_quality_cs_k16": _v("clock_sf_quality_cs"),
    "clock_sf_quality_cs_k16_noT1000": _v(
        "clock_sf_quality_cs", subset={"exclude_horizons": [1000]}
    ),
    # reservoir-aware simple rule (P11)
    "reservoir_rule_k16": _v("reservoir_rule"),
    # the exact configuration behind RESULTS.md's 0.647 / 0.7405 (reproduction gate)
    "legacy_E_k16_weighted": _v("all71", estimator="logit_w"),
}

QUICK_VARIANTS = ("clock_k16", "clock_quality_evidence_k16", "reservoir_rule_k16")

# Published E_all_observable row of fit_results_label_A_k16.json (mean over env folds).
GATE_VARIANT = "legacy_E_k16_weighted"
GATE_BAL_ACC = 0.647
GATE_AUC = 0.7405
GATE_TOL = 0.01

TAU_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)
REMAINING_BANDS = (0.1, 0.25, 0.5, 0.75)
MIN_CELL_ROWS = 20
SIMPSON_COEFS = ("f_K", "f_K_over_t", "f_n_singletons")


def feature_list(variant: dict) -> tuple[str, ...]:
    return tuple(FEATURE_SETS[variant["feature_set"]])


# ---- reservoir rule -----------------------------------------------------------------


def beta_excess_mean(a, b, c) -> np.ndarray:
    """`E[(X - c)_+]` for `X ~ Beta(a, b)`: the expected improvement of one fresh draw over `c`.

    Closed form `(a/(a+b)) (1 - I_c(a+1, b)) - c (1 - I_c(a, b))`, the integral of the Beta
    survival function from `c` to 1. It is the observable analogue of the oracle
    `I_t = int_c^1 P(mu > x) dx` in the plan's reservoir diagnostics.
    """
    from scipy.special import betainc

    a = np.maximum(np.asarray(a, dtype=np.float64), 1e-12)
    b = np.maximum(np.asarray(b, dtype=np.float64), 1e-12)
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return (a / (a + b)) * (1.0 - betainc(a + 1.0, b, c)) - c * (1.0 - betainc(a, b, c))


def reservoir_rule_features(X: np.ndarray) -> np.ndarray:
    """Pipeline step: the 7 declared corpus columns -> the 5 inputs of the logistic rule.

    Input columns follow `RESERVOIR_RULE_FEATURES`; the output keeps the first four and
    appends `est_I_hat` computed from `(est_beta_a, est_beta_b, f_best_mean)`.
    """
    X = np.asarray(X, dtype=np.float64)
    i_hat = beta_excess_mean(X[:, 4], X[:, 5], X[:, 6])
    return np.column_stack([X[:, :4], i_hat])


def _importable_reservoir_rule_features():
    """The function object as seen from the importable module, never from `__main__`.

    A function defined in `__main__` pickles as `__main__.reservoir_rule_features`, which
    no other process can resolve. Resolving it through the module name keeps the saved
    artifact loadable wherever `experiments/growing_bandits/deploy` is on `sys.path`.
    """
    mod = importlib.import_module("train_policies")
    return mod.reservoir_rule_features


# ---- estimators --------------------------------------------------------------------


def make_estimator(estimator: str, feature_set: str):
    """Fresh unfitted estimator. Hyperparameters are the ones the published study used."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    if estimator == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        if feature_set == "reservoir_rule":
            raise ValueError("the reservoir rule is a logistic model by construction")
        return HistGradientBoostingClassifier(
            max_depth=6, max_iter=300, learning_rate=0.06, random_state=0
        )
    if estimator not in ("logit", "logit_w"):
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")
    steps = []
    if feature_set == "reservoir_rule":
        steps.append(FunctionTransformer(_importable_reservoir_rule_features()))
    steps.append(StandardScaler())
    steps.append(LogisticRegression(C=1.0, max_iter=2000))
    return make_pipeline(*steps)


def design_matrix(df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    """`fit_models._matrix` convention: float64, non-finite values become 0."""
    X = df.loc[:, list(features)].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---- row selection -------------------------------------------------------------------


def subset_mask(df: pd.DataFrame, subset: dict) -> np.ndarray:
    """The training universe of a variant: which rows exist at all before the row filter."""
    keep = np.ones(len(df), dtype=bool)
    for key, values in subset.items():
        if key == "exclude_policies":
            keep &= ~df["meta_policy"].astype(str).isin([str(v) for v in values]).to_numpy()
        elif key == "families":
            keep &= df["meta_family"].astype(str).isin([str(v) for v in values]).to_numpy()
        elif key == "exclude_horizons":
            horizons = df["meta_horizon"].to_numpy(dtype=np.float64)
            keep &= ~np.isin(horizons, np.asarray(values, dtype=np.float64))
        else:
            raise ValueError(f"unknown subset key {key!r}")
    return keep


def row_filter_mask(df: pd.DataFrame, row_filter: str, k: int) -> np.ndarray:
    """Which rows of the universe are trained on. Every filter starts from decided rows."""
    m = decided_mask(df, k)
    if row_filter == "decided":
        return m
    if row_filter == "noambig":
        return m & ~ambiguous_mask(df, k)
    if row_filter == "notrunc":
        return m & ~truncated_or_demoted_mask(df, k)
    if row_filter == "lucb":
        return m & lucb_only_mask(df)
    raise ValueError(f"unknown row filter {row_filter!r}; expected one of {ROW_FILTERS}")


def eligible_masks(df: pd.DataFrame, variant: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(eligible, universe)`: rows trained on, and rows the folds are assigned over."""
    universe = subset_mask(df, variant["subset"])
    eligible = universe & row_filter_mask(df, variant["row_filter"], variant["k"])
    return eligible, universe


# ---- cross-validation ------------------------------------------------------------------


def fold_ids_by_group(groups: np.ndarray, universe: np.ndarray, n_splits: int) -> np.ndarray:
    """Fold id per row from `GroupKFold` over the universe rows; -1 outside the universe.

    Folds are assigned over the whole universe (ties included) and only then filtered to
    eligible rows, which is what `fit_models.group_scores` did; assigning them over decided
    rows alone would change the environment partition and break the reproduction gate.
    """
    from sklearn.model_selection import GroupKFold

    ids = np.full(len(groups), -1, dtype=np.int64)
    idx = np.flatnonzero(universe)
    g = np.asarray(groups)[idx]
    n = int(min(n_splits, len(np.unique(g))))
    if n < 2:
        return ids
    dummy = np.zeros((len(idx), 1))
    for f, (_, te) in enumerate(GroupKFold(n_splits=n).split(dummy, groups=g)):
        ids[idx[te]] = f
    return ids


def _fit_predict_fold(estimator, feature_set, X, y, w, tr, te):
    model = make_estimator(estimator, feature_set)
    if w is None:
        model.fit(X[tr], y[tr])
    else:
        fit_with_weights(model, X[tr], y[tr], w[tr])
    return te, model.predict_proba(X[te])[:, 1]


def oof_scores(
    estimator: str,
    feature_set: str,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    fold: np.ndarray,
    eligible: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Out-of-fold P(SEARCH) for every eligible row (NaN where no fold could predict it)."""
    from joblib import Parallel, delayed

    score = np.full(len(y), np.nan)
    jobs = []
    for f in np.unique(fold[fold >= 0]):
        te = np.flatnonzero(eligible & (fold == f))
        tr = np.flatnonzero(eligible & (fold >= 0) & (fold != f))
        if len(te) == 0 or len(tr) == 0 or y[tr].all() or not y[tr].any():
            continue
        jobs.append(delayed(_fit_predict_fold)(estimator, feature_set, X, y, w, tr, te))
    if not jobs:
        return score
    n_jobs = 1 if estimator == "hgb" else int(min(n_jobs, len(jobs)))
    for te, s in Parallel(n_jobs=n_jobs)(jobs):
        score[te] = s
    return score


# ---- metrics ---------------------------------------------------------------------------


def balanced_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    if truth.all() or not truth.any():
        return float("nan")
    hit = pred == truth
    return 0.5 * (float(hit[truth].mean()) + float(hit[~truth].mean()))


def auc(truth: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if truth.all() or not truth.any():
        return float("nan")
    return float(roc_auc_score(truth, score))


def balanced_accuracy_curve(truth: np.ndarray, score: np.ndarray, taus=TAU_GRID) -> np.ndarray:
    """Balanced accuracy of `score > tau` for every tau, vectorized over the grid."""
    if truth.all() or not truth.any():
        return np.full(len(taus), np.nan)
    pred = score[:, None] > np.asarray(taus)[None, :]
    tpr = pred[truth].mean(axis=0)
    tnr = (~pred[~truth]).mean(axis=0)
    return 0.5 * (tpr + tnr)


def select_tau(truth: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """`(tau_off, balanced accuracy at tau_off)`; the first maximiser on the grid."""
    curve = balanced_accuracy_curve(truth, score)
    if not np.isfinite(curve).any():
        return float("nan"), float("nan")
    i = int(np.nanargmax(curve))
    return float(TAU_GRID[i]), float(curve[i])


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / np.sum(w**2)) if w.size else float("nan")


def pooled_and_fold_metrics(
    truth: np.ndarray, score: np.ndarray, fold: np.ndarray, w: np.ndarray | None
) -> dict:
    """Pooled OOF metrics plus the mean-over-folds convention of `fit_models.group_scores`."""
    ok = np.isfinite(score)
    out = {
        "n_rows": int(ok.sum()),
        "n_pos": int(truth[ok].sum()),
        "class_balance": float(truth[ok].mean()) if ok.any() else float("nan"),
        "auc": float("nan"),
        "auc_fold_mean": float("nan"),
        "bal_acc_05": float("nan"),
        "bal_acc_05_fold_mean": float("nan"),
        "tau_off": float("nan"),
        "bal_acc_tau_off": float("nan"),
        "ess": ess(w[ok]) if w is not None else float("nan"),
    }
    if not ok.any():
        return out
    t, s = truth[ok], score[ok]
    out["auc"] = auc(t, s)
    out["bal_acc_05"] = balanced_accuracy(t, s > 0.5)
    out["tau_off"], out["bal_acc_tau_off"] = select_tau(t, s)
    aucs, bals = [], []
    for f in np.unique(fold[ok]):
        m = ok & (fold == f)
        tf = truth[m]
        if tf.all() or not tf.any():
            continue
        aucs.append(auc(tf, score[m]))
        bals.append(balanced_accuracy(tf, score[m] > 0.5))
    if aucs:
        out["auc_fold_mean"] = float(np.mean(aucs))
        out["bal_acc_05_fold_mean"] = float(np.mean(bals))
    return out


def within_regime_auc(df: pd.DataFrame, truth: np.ndarray, score: np.ndarray) -> dict:
    """Mean OOF AUC over (horizon x allocation x policy x remaining-band) cells.

    Pooled AUC rewards a model for telling regimes apart (early states at long horizons
    search; late states refine). Ranking *within* a cell is what a deployed rule has to
    do, so this strips the between-regime component out.
    """
    ok = np.isfinite(score)
    band = np.digitize(df["f_remaining_frac"].to_numpy(dtype=np.float64), REMAINING_BANDS)
    cells = pd.DataFrame(
        {
            "T": df["meta_horizon"].to_numpy(),
            "alloc": df["meta_allocation"].to_numpy(),
            "policy": df["meta_policy"].to_numpy(),
            "band": band,
            "y": truth,
            "s": score,
        }
    )[ok]
    aucs, n_rows = [], 0
    for _, cell in cells.groupby(["T", "alloc", "policy", "band"], sort=True):
        yv = cell["y"].to_numpy()
        if len(yv) < MIN_CELL_ROWS or yv.all() or not yv.any():
            continue
        aucs.append(auc(yv, cell["s"].to_numpy()))
        n_rows += len(yv)
    return {
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_cells": len(aucs),
        "n_rows": int(n_rows),
    }


# ---- one variant ---------------------------------------------------------------------


def train_variant(
    name: str,
    variant: dict,
    df: pd.DataFrame,
    out_dir: Path,
    n_splits: int,
    n_jobs: int,
    provenance: dict,
) -> tuple[list[dict], dict]:
    """OOF metrics under every grouping, then a refit on all eligible rows saved as an artifact.

    Returns `(metric rows, artifact meta)`.
    """
    k = variant["k"]
    features = feature_list(variant)
    fg.assert_deployable(list(features))
    label, se_col = label_columns(k)
    eligible, universe = eligible_masks(df, variant)
    X = design_matrix(df, features)
    y = df[label].to_numpy(dtype=np.float64) > 0
    w = None
    if variant["estimator"] == "logit_w":
        # Weights are normalised over the whole corpus, ties included, exactly as the
        # published fit did; the regularised objective is not invariant to their scale.
        w = precision_weights(df[se_col].to_numpy(dtype=np.float64))

    base = {
        "variant": name,
        "k": k,
        "estimator": variant["estimator"],
        "feature_set": variant["feature_set"],
        "n_features": len(features),
        "row_filter": variant["row_filter"],
        "subset": json.dumps(variant["subset"], sort_keys=True),
    }
    rows: list[dict] = []
    metrics: dict[str, dict] = {}
    for grouping in ("meta_env", "meta_horizon", "meta_policy", "meta_family"):
        groups = df[grouping].to_numpy().astype(str)
        fold = fold_ids_by_group(groups, universe, n_splits)
        n_groups = int(len(np.unique(groups[universe])))
        n_folds = int(len(np.unique(fold[fold >= 0])))
        score = oof_scores(
            variant["estimator"], variant["feature_set"], X, y, w, fold, eligible, n_jobs
        )
        m = pooled_and_fold_metrics(y, score, fold, w)
        m.update({"n_groups": n_groups, "n_splits": n_folds, "n_cells": np.nan})
        metrics[grouping] = m
        rows.append({**base, "grouping": grouping, **m})
        if grouping == "meta_env":
            wr = within_regime_auc(df, y, score)
            metrics["within_regime"] = wr
            rows.append(
                {
                    **base,
                    "grouping": "within_regime",
                    "n_groups": n_groups,
                    "n_splits": n_folds,
                    **wr,
                }
            )

    tau_off = metrics["meta_env"]["tau_off"]
    if not np.isfinite(tau_off):
        tau_off = 0.5

    # Refit on every eligible row; this is the deployed object.
    model = make_estimator(variant["estimator"], variant["feature_set"])
    if w is None:
        model.fit(X[eligible], y[eligible])
    else:
        fit_with_weights(model, X[eligible], y[eligible], w[eligible])

    # Transfer: rows the subset excluded, under the same row filter, scored by the refit model.
    excluded = (~universe) & row_filter_mask(df, variant["row_filter"], k)
    if variant["subset"] and excluded.any():
        s_ex = model.predict_proba(X[excluded])[:, 1]
        y_ex = y[excluded]
        tr = {
            "n_rows": int(excluded.sum()),
            "n_pos": int(y_ex.sum()),
            "class_balance": float(y_ex.mean()),
            "auc": auc(y_ex, s_ex),
            "bal_acc_05": balanced_accuracy(y_ex, s_ex > 0.5),
            "tau_off": float(tau_off),
            "bal_acc_tau_off": balanced_accuracy(y_ex, s_ex > tau_off),
            "n_groups": 0,
            "n_splits": 0,
        }
        metrics["excluded_rows"] = tr
        rows.append({**base, "grouping": "excluded_rows", **tr})

    meta = {
        "variant": name,
        **{key: variant[key] for key in ("feature_set", "k", "estimator", "row_filter", "subset")},
        "features": list(features),
        "label": label,
        "se_column": se_col if w is not None else None,
        "n_rows": int(eligible.sum()),
        "n_pos": int(y[eligible].sum()),
        "class_balance": float(y[eligible].mean()) if eligible.any() else float("nan"),
        "ess": ess(w[eligible]) if w is not None else None,
        "tau_off": float(tau_off),
        "tau_grid": [float(TAU_GRID[0]), float(TAU_GRID[-1]), 0.01],
        "metrics": metrics,
        "n_splits": int(n_splits),
        **provenance,
    }
    if variant["feature_set"] == "reservoir_rule":
        meta["derived_inputs"] = ["est_p_new_beats_incumbent", "f_remaining_frac",
                                  "f_leader_width", "f_log_K", "est_I_hat"]
        meta["unpickle_requires"] = (
            "experiments/growing_bandits/deploy on sys.path (module `train_policies`)"
        )
    artifacts.save_model(
        out_dir / "models" / f"{name}.joblib",
        {"pipeline": model, "features": list(features), "k": k, "tau": float(tau_off),
         "meta": meta},
    )
    return rows, meta


# ---- offline diagnostics -------------------------------------------------------------------


def _diag(diagnostic: str, key: str, value) -> dict:
    return {"diagnostic": diagnostic, "key": key, "value": float(value)}


def simpson_diagnostics(df: pd.DataFrame) -> list[dict]:
    """Failure-mode #1: P(SEARCH | K/t tercile) pooled vs within each generating policy.

    Tercile cut points are taken on the horizon's pooled decided rows and reused within
    every policy, so "pooled" and "within" bin the same variable the same way; only the
    conditioning differs.
    """
    rows: list[dict] = []
    dec = decided_mask(df, 16)
    y = df["label_A_k16"].to_numpy(dtype=np.float64) > 0
    kt = df["f_K_over_t"].to_numpy(dtype=np.float64)
    T_all = df["meta_horizon"].to_numpy(dtype=np.float64)
    pol = df["meta_policy"].to_numpy().astype(str)
    for T in sorted(np.unique(T_all)):
        m = dec & (T_all == T)
        if m.sum() < 3:
            continue
        cuts = np.quantile(kt[m], [1 / 3, 2 / 3])
        terc = np.digitize(kt, cuts) + 1  # 1..3
        groups = [("pooled", m)] + [(p, m & (pol == p)) for p in sorted(np.unique(pol[m]))]
        for gname, gm in groups:
            p_by_q = {}
            for q in (1, 2, 3):
                mq = gm & (terc == q)
                key = f"T={int(T)}|{gname}|q{q}"
                rows.append(_diag("simpson_n", key, mq.sum()))
                p_by_q[q] = float(y[mq].mean()) if mq.any() else float("nan")
                rows.append(_diag("simpson_p_search", key, p_by_q[q]))
            rows.append(_diag("simpson_slope_q3_minus_q1", f"T={int(T)}|{gname}",
                              p_by_q[3] - p_by_q[1]))
    return rows


def per_policy_coefficient_signs(df: pd.DataFrame) -> list[dict]:
    """Standardised logistic coefficients on `f_K`, `f_K_over_t`, `f_n_singletons`, pooled vs per policy."""
    rows: list[dict] = []
    features = FEATURE_SETS[CQE]
    idx = {c: features.index(c) for c in SIMPSON_COEFS}
    dec = decided_mask(df, 16)
    X = design_matrix(df, features)
    y = df["label_A_k16"].to_numpy(dtype=np.float64) > 0
    pol = df["meta_policy"].to_numpy().astype(str)
    fits = [("pooled", dec)] + [(p, dec & (pol == p)) for p in sorted(np.unique(pol))]
    for gname, gm in fits:
        if gm.sum() < 10 or y[gm].all() or not y[gm].any():
            continue
        model = make_estimator("logit", CQE).fit(X[gm], y[gm])
        coef = model.named_steps["logisticregression"].coef_[0]
        rows.append(_diag("policy_fit_n", gname, gm.sum()))
        for c, i in idx.items():
            rows.append(_diag("policy_coef", f"{gname}|{c}", coef[i]))
            rows.append(_diag("policy_coef_sign", f"{gname}|{c}", np.sign(coef[i])))
    return rows


def metadata_only_classifier(df: pd.DataFrame, n_splits: int) -> list[dict]:
    """How much of the label a one-hot of (policy, horizon, allocation) predicts on its own.

    Anything a deployable model scores below this is explained by regime membership, not
    by the state; audit C measured 0.588 bal-acc / 0.624 AUC.
    """
    from sklearn.linear_model import LogisticRegression

    dec = decided_mask(df, 16)
    y = df["label_A_k16"].to_numpy(dtype=np.float64) > 0
    onehot = pd.get_dummies(
        pd.DataFrame(
            {
                "policy": df["meta_policy"].astype(str),
                "horizon": df["meta_horizon"].astype(int).astype(str),
                "allocation": df["meta_allocation"].astype(str),
            }
        ),
        dtype=np.float64,
    )
    X = onehot.to_numpy()
    universe = np.ones(len(df), dtype=bool)
    fold = fold_ids_by_group(df["meta_env"].to_numpy().astype(str), universe, n_splits)
    score = np.full(len(df), np.nan)
    for f in np.unique(fold[fold >= 0]):
        te = np.flatnonzero(dec & (fold == f))
        tr = np.flatnonzero(dec & (fold >= 0) & (fold != f))
        if len(te) == 0 or len(tr) == 0 or y[tr].all() or not y[tr].any():
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        score[te] = clf.predict_proba(X[te])[:, 1]
    m = pooled_and_fold_metrics(y, score, fold, None)
    rows = [_diag("metadata_only", key, m[key]) for key in
            ("auc", "auc_fold_mean", "bal_acc_05", "bal_acc_05_fold_mean", "tau_off",
             "bal_acc_tau_off", "n_rows")]
    rows.append(_diag("metadata_only", "n_onehot_columns", X.shape[1]))
    return rows


def weight_and_label_diagnostics(df: pd.DataFrame) -> list[dict]:
    """ESS of the precision weights and the tie / ambiguous / truncated / demoted fractions."""
    rows: list[dict] = []
    for k in (1, 4, 16):
        _, se_col = label_columns(k)
        w = precision_weights(df[se_col].to_numpy(dtype=np.float64))
        dec = decided_mask(df, k)
        amb = ambiguous_mask(df, k)
        trunc = df["f_remaining_budget"].to_numpy(dtype=np.float64) < k
        demoted = df["f_K"].to_numpy(dtype=np.float64) + k > 64
        rows.append(_diag("ess", f"k={k}|all", ess(w)))
        rows.append(_diag("ess", f"k={k}|decided", ess(w[dec])))
        rows.append(_diag("ess_frac", f"k={k}|all", ess(w) / len(w)))
        rows.append(_diag("ess_frac", f"k={k}|decided", ess(w[dec]) / max(dec.sum(), 1)))
        rows.append(_diag("label_fraction", f"k={k}|tie", (~dec).mean()))
        rows.append(_diag("label_fraction", f"k={k}|ambiguous", amb.mean()))
        rows.append(_diag("label_fraction", f"k={k}|ambiguous_among_decided",
                          amb[dec].mean() if dec.any() else np.nan))
        rows.append(_diag("label_fraction", f"k={k}|truncated", trunc.mean()))
        rows.append(_diag("label_fraction", f"k={k}|demoted", demoted.mean()))
        rows.append(_diag("label_fraction", f"k={k}|truncated_or_demoted",
                          truncated_or_demoted_mask(df, k).mean()))
        rows.append(_diag("label_fraction", f"k={k}|search_preferred_among_decided",
                          (df[f"label_A_k{k}"].to_numpy() > 0)[dec].mean() if dec.any() else np.nan))
        rows.append(_diag("label_n", f"k={k}|all", len(df)))
        rows.append(_diag("label_n", f"k={k}|decided", dec.sum()))
    return rows


# ---- feature hygiene --------------------------------------------------------------------------

CLOCK_METADATA = ("f_T", "f_remaining_budget", "f_remaining_frac", "f_t_over_T", "f_K_over_T")
POLICY_CONTAMINATED = ("f_K", "f_K_over_t", "f_K_over_T", "f_K_over_sqrt_t", "f_log_K",
                       "f_n_singletons", "f_mean_n")


def hygiene_group(column: str) -> str:
    if column in fg.CLOCK:
        return "CLOCK"
    if column in fg.QUALITY:
        return "QUALITY"
    if column in fg.EVIDENCE_CS:
        return "EVIDENCE_CS"
    if column in fg.EVIDENCE_LOGE:
        return "EVIDENCE_LOGE"
    if column in fg.HISTORY:
        return "HISTORY"
    if column.startswith("oracle_"):
        return "ORACLE"
    if column.startswith("label_"):
        return "LABEL"
    if column.startswith("meta_"):
        return "META"
    raise ValueError(f"column {column!r} belongs to no group")


def availability_class(column: str) -> str:
    """a observable | b derivable-from-own-history | c oracle | d policy-contaminated | e metadata.

    A column can carry two classes (`a+d`: observable but a function of the generating
    policy's search count); `+` joins them so the CSV field never needs quoting.
    """
    group = hygiene_group(column)
    if group in ("ORACLE", "LABEL"):
        return "c"
    if group == "META":
        return "e"
    if group == "HISTORY":
        return "b+d"
    if group == "CLOCK":
        cls = "e" if column in CLOCK_METADATA else "a"
    elif group == "QUALITY":
        cls = "b" if column.startswith("est_") else "a"
    else:
        cls = "a"
    if column in POLICY_CONTAMINATED:
        cls += "+d"
    return cls


def feature_hygiene_table(columns: list[str], variants: dict[str, dict]) -> pd.DataFrame:
    used: dict[str, list[str]] = {c: [] for c in columns}
    for name, variant in variants.items():
        for c in feature_list(variant):
            used[c].append(name)
    rows = []
    for c in columns:
        rows.append(
            {
                "column": c,
                "prefix": c.split("_", 1)[0] + "_",
                "group": hygiene_group(c),
                "availability_class": availability_class(c),
                "deployed_in": ",".join(used[c]),
            }
        )
    return pd.DataFrame(rows)


# ---- reservoir diagnostics -------------------------------------------------------------------


def oracle_reservoir_specs() -> dict[str, dict]:
    from label_states import FAMILY_A, FAMILY_B

    return dict(FAMILY_A + FAMILY_B)


def oracle_improvement_integral(df: pd.DataFrame, n_grid: int = 400) -> np.ndarray:
    """Oracle `I_t = int_c^1 P(mu > x) dx` at `c = oracle_best_true_mu`, per row. ANALYSIS ONLY.

    Midpoint quadrature of the true reservoir's tail on an `n_grid` grid per environment,
    accumulated from the top so every row's `c` is a single interpolation.
    """
    from cold_start.growing.reservoirs import build_reservoir

    specs = oracle_reservoir_specs()
    envs = df["meta_env"].to_numpy().astype(str)
    c = df["oracle_best_true_mu"].to_numpy(dtype=np.float64)
    out = np.full(len(df), np.nan)
    h = 1.0 / n_grid
    mids = (np.arange(n_grid) + 0.5) * h
    edges = np.arange(n_grid + 1) * h
    for env in np.unique(envs):
        res = build_reservoir(specs[env])
        tail = np.array([res.tail_prob(float(x)) for x in mids])
        # J(edge_m) = h * sum_{j >= m} tail_j ; J(1) = 0
        cum = np.concatenate([np.cumsum(tail[::-1])[::-1] * h, [0.0]])
        m = envs == env
        out[m] = np.interp(c[m], edges, cum)
    return out


def reservoir_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """AUC of oracle and observable reservoir quantities for the k=16 label (plan section 13a)."""
    dec = decided_mask(df, 16)
    y = df["label_A_k16"].to_numpy(dtype=np.float64) > 0
    scores = {}
    for c in ("oracle_p_new_beats_best_true", "oracle_p_new_beats_best_true_plus_0.01",
              "oracle_p_new_beats_best_true_plus_0.05", "oracle_p_new_beats_best_true_plus_0.1"):
        scores[c] = df[c].to_numpy(dtype=np.float64)
    scores["oracle_I_t"] = oracle_improvement_integral(df)
    for c in ("est_p_new_beats_incumbent", "est_p_new_beats_incumbent_plus_0.01",
              "est_p_new_beats_incumbent_plus_0.05", "est_p_new_beats_incumbent_plus_0.1"):
        scores[c] = np.nan_to_num(df[c].to_numpy(dtype=np.float64))
    scores["est_I_hat"] = np.nan_to_num(beta_excess_mean(
        df["est_beta_a"].to_numpy(dtype=np.float64),
        df["est_beta_b"].to_numpy(dtype=np.float64),
        df["f_best_mean"].to_numpy(dtype=np.float64),
    ))
    T_all = df["meta_horizon"].to_numpy(dtype=np.float64)
    scopes = [("all", dec)] + [(f"T={int(T)}", dec & (T_all == T)) for T in sorted(np.unique(T_all))]
    rows = []
    for scope, m in scopes:
        for name, s in scores.items():
            a = auc(y[m], s[m]) if m.any() else float("nan")
            rows.append(
                {
                    "scope": scope,
                    "score": name,
                    "kind": "oracle" if name.startswith("oracle_") else "observable",
                    "n_rows": int(m.sum()),
                    "auc_raw": a,
                    "auc_oriented": max(a, 1.0 - a) if np.isfinite(a) else float("nan"),
                    "sign": (1.0 if a >= 0.5 else -1.0) if np.isfinite(a) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# ---- CLI -------------------------------------------------------------------------------------


def parse_variants(spec: str, quick: bool) -> list[str]:
    if spec == "all":
        return list(QUICK_VARIANTS) if quick else list(VARIANTS)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    return names


def check_reproduction_gate(meta: dict) -> tuple[bool, float, float]:
    env = meta["metrics"]["meta_env"]
    bal, a = env["bal_acc_05_fold_mean"], env["auc_fold_mean"]
    ok = abs(bal - GATE_BAL_ACC) <= GATE_TOL and abs(a - GATE_AUC) <= GATE_TOL
    return ok, bal, a


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--variants", type=str, default="all",
                    help="'all' or a comma-separated list of variant names")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="parallel folds (default: min(n_splits, cores); 1 under --quick)")
    ap.add_argument("--quick", action="store_true",
                    help="first 3 shards, n_splits=2, three variants, gate not asserted")
    args = ap.parse_args(argv)

    quick = bool(args.quick)
    n_splits = 2 if quick else int(args.n_splits)
    n_jobs = args.n_jobs if args.n_jobs is not None else (1 if quick else min(n_splits, os.cpu_count() or 1))
    names = parse_variants(args.variants, quick)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    t0 = time.time()
    df = load_corpus(args.data, max_shards=3 if quick else None)
    corpus_columns = [c for c in df.columns if c != "meta_trajectory"]
    fg.check_partition(corpus_columns)
    print(f"loaded {len(df)} rows x {len(corpus_columns)} columns from {args.data}"
          f" ({df['meta_env'].nunique()} envs, {df['meta_trajectory'].nunique()} trajectories)"
          f"  [{time.time() - t0:.1f}s]")
    provenance = {
        "corpus": str(args.data),
        "n_corpus_rows": int(len(df)),
        "quick": quick,
        "trained_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }

    variants_json = {name: {**VARIANTS[name], "features": list(feature_list(VARIANTS[name]))}
                     for name in VARIANTS}
    variants_json["_estimators"] = {
        "logit": "StandardScaler -> LogisticRegression(C=1.0, max_iter=2000), unweighted",
        "logit_w": "logit with fit_models.precision_weights(label_se_k{k}) sample weights",
        "hgb": "HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.06, random_state=0)",
        "reservoir_rule": "FunctionTransformer(est_I_hat) -> StandardScaler -> LogisticRegression",
    }
    (out_dir / "variants.json").write_text(json.dumps(variants_json, indent=2))

    metric_rows: list[dict] = []
    metas: dict[str, dict] = {}
    for name in names:
        t1 = time.time()
        rows, meta = train_variant(name, VARIANTS[name], df, out_dir, n_splits, n_jobs, provenance)
        metric_rows.extend(rows)
        metas[name] = meta
        env = meta["metrics"]["meta_env"]
        wr = meta["metrics"]["within_regime"]
        print(f"  {name:44s} k={meta['k']:<2d} n={meta['n_rows']:6d}  "
              f"AUC {env['auc']:.4f} (fold-mean {env['auc_fold_mean']:.4f})  "
              f"bal@0.5 {env['bal_acc_05']:.4f}  tau_off {meta['tau_off']:.2f} -> "
              f"{env['bal_acc_tau_off']:.4f}  within-regime {wr['auc']:.4f} "
              f"({wr['n_cells']} cells)  [{time.time() - t1:.0f}s]")
        if "excluded_rows" in meta["metrics"]:
            ex = meta["metrics"]["excluded_rows"]
            print(f"  {'':44s} transfer to excluded rows: AUC {ex['auc']:.4f}  "
                  f"bal@tau {ex['bal_acc_tau_off']:.4f}  n={ex['n_rows']}")

    columns = ["variant", "grouping", "k", "estimator", "feature_set", "n_features", "row_filter",
               "subset", "n_groups", "n_splits", "n_rows", "n_pos", "class_balance", "auc",
               "auc_fold_mean", "bal_acc_05", "bal_acc_05_fold_mean", "tau_off", "bal_acc_tau_off",
               "ess", "n_cells"]
    metrics_df = pd.DataFrame(metric_rows).reindex(columns=columns)
    metrics_df.to_csv(out_dir / "offline_metrics.csv", index=False)

    gate = None
    if GATE_VARIANT in metas:
        ok, bal, a = check_reproduction_gate(metas[GATE_VARIANT])
        gate = {"ok": bool(ok), "bal_acc_05_fold_mean": bal, "auc_fold_mean": a,
                "target_bal_acc": GATE_BAL_ACC, "target_auc": GATE_AUC, "tol": GATE_TOL,
                "asserted": not quick}
        print(f"\nREPRODUCTION GATE ({GATE_VARIANT}): bal-acc@0.5 fold-mean {bal:.4f} "
              f"(target {GATE_BAL_ACC} +/- {GATE_TOL}), AUC fold-mean {a:.4f} "
              f"(target {GATE_AUC} +/- {GATE_TOL}) -> {'PASS' if ok else 'FAIL'}"
              f"{'' if not quick else ' (not asserted under --quick)'}")
        (out_dir / "reproduction_gate.json").write_text(json.dumps(gate, indent=2))
        if not ok and not quick:
            raise AssertionError(
                f"reproduction gate failed: bal-acc {bal:.4f} vs {GATE_BAL_ACC}, "
                f"AUC {a:.4f} vs {GATE_AUC} (tol {GATE_TOL})"
            )

    t2 = time.time()
    diag_rows = []
    diag_rows.extend(simpson_diagnostics(df))
    diag_rows.extend(per_policy_coefficient_signs(df))
    diag_rows.extend(metadata_only_classifier(df, n_splits))
    diag_rows.extend(weight_and_label_diagnostics(df))
    pd.DataFrame(diag_rows, columns=["diagnostic", "key", "value"]).to_csv(
        out_dir / "offline_diagnostics.csv", index=False
    )
    feature_hygiene_table(corpus_columns, VARIANTS).to_csv(
        out_dir / "feature_hygiene.csv", index=False
    )
    reservoir_diagnostics(df).to_csv(out_dir / "reservoir_diagnostics_offline.csv", index=False)
    print(f"\ndiagnostics written [{time.time() - t2:.0f}s]; total {time.time() - t0:.0f}s -> {out_dir}")
    return {"metas": metas, "gate": gate, "out_dir": out_dir, "n_rows": int(len(df))}


if __name__ == "__main__":
    main()
