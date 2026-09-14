#!/usr/bin/env python
"""M6: fit the decision function, run the ablations, and test real generalization.

Three commitments here are what separate a result from a number:

* **The model ladder is climbed in order** -- time-only schedule, evidence threshold,
  evidence + search + budget, degree-2 polynomial, degree-3, then a gradient-boosted
  upper bound. The point is to find the SIMPLEST model that captures the signal. If the
  degree-2 polynomial nearly matches the boosted model, that is the headline result, not
  a disappointment.
* **Splits hold out whole groups, never rows.** Rows from one trajectory share arms, a
  reservoir draw sequence and a random stream; a random row split would let a near-copy
  of a training row sit in the test set and every model would look excellent.
* **Labels are weighted by their precision.** A Monte Carlo label with a large standard
  error is a weaker constraint than a tight one, and treating them equally lets noise in
  the hard states outvote signal in the easy ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.growing.schema import (  # noqa: E402
    PREFIX_ESTIMATE,
    PREFIX_FEATURE,
    PREFIX_ORACLE,
    design_matrix_columns,
)

# The six theoretically-motivated variables the brief names for the polynomial models.
POLY_VARS = {
    "z": "f_log_e_pair",            # leader/challenger evidence
    "u": "f_mean_width_plausible",  # unresolved uncertainty on the frontier
    "k": "f_log_K",                 # arms discovered
    "tau": "f_log_t",               # time spent
    "r": "est_p_new_beats_incumbent",  # estimated chance a fresh arm competes
    "h": "f_remaining_frac",        # budget left
}

ABLATIONS = {
    "A_time_only": lambda c: [x for x in c if x in ("f_t", "f_K", "f_log_t", "f_log_K",
                                                    "f_K_over_t", "f_K_over_T", "f_K_over_sqrt_t")],
    "B_evidence_only": lambda c: [x for x in c if "log_e" in x or "lcb" in x or "ucb" in x
                                  or "width" in x or "separated" in x],
    "C_time_plus_counts": lambda c: [x for x in c if x in ("f_t", "f_K", "f_log_t", "f_log_K",
                                                           "f_K_over_t", "f_K_over_T")
                                     or "mean" in x or "_n" in x or x.endswith("_n_frac")],
    "D_time_plus_eprocess": lambda c: [x for x in c if x in ("f_t", "f_K", "f_log_t", "f_log_K",
                                                             "f_K_over_t", "f_K_over_T")
                                       or "log_e" in x],
    "E_all_observable": lambda c: [x for x in c if x.startswith((PREFIX_FEATURE, PREFIX_ESTIMATE))],
    "F_plus_oracle": lambda c: [x for x in c if x.startswith((PREFIX_FEATURE, PREFIX_ESTIMATE,
                                                              PREFIX_ORACLE))],
}


def load_dataset(path: Path) -> dict:
    """Read every parquet shard into a column dict (pyarrow only; pandas is not needed)."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(path / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {path}")
    tbl = pq.read_table(files)
    return {name: np.asarray(tbl.column(name).to_pylist()) for name in tbl.column_names}


def _matrix(data: dict, cols: list[str]) -> np.ndarray:
    X = np.column_stack([np.asarray(data[c], dtype=np.float64) for c in cols])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] * 3 > X.shape[0]:
        print(f"  WARNING: {X.shape[1]} features for {X.shape[0]} rows -- "
              f"cross-validated scores here are not trustworthy")
    return X


def precision_weights(se: np.ndarray, floor_quantile: float = 0.25, max_ratio: float = 20.0) -> np.ndarray:
    """Inverse-variance weights that a handful of rows cannot run away with.

    A naive `1 / max(se, tiny)**2` is catastrophic here. Around 28% of states are exact
    ties -- every paired replicate recommends the same arm -- so their Monte Carlo
    standard error is exactly zero and their weight saturates whatever floor is used.
    Measured on the production corpus, that put **80.4% of all weight on the top 1% of
    rows**, and since a tie is scored as "not SEARCH" it dragged the apparent SEARCH
    rate from 41.0% down to 17.1%. Every model was effectively being fitted to a few
    hundred uninformative ties.

    Two guards fix it. The floor is a QUANTILE of the observed standard errors rather
    than a constant, so it scales with the corpus and encodes the fact that no label is
    infinitely precise. The cap is a multiple of the median weight, so no row can
    outvote a large block of its peers.
    """
    se = np.asarray(se, dtype=np.float64)
    positive = se[se > 0]
    floor = float(np.quantile(positive, floor_quantile)) if positive.size else 1e-4
    w = 1.0 / np.maximum(se, floor) ** 2
    w = np.minimum(w, max_ratio * float(np.median(w)))
    return w / max(w.mean(), 1e-12)


def fit_with_weights(model, X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Fit, routing sample weights to a Pipeline's final estimator.

    `Pipeline.fit` rejects a bare `sample_weight`; it has to be addressed to the step
    that consumes it. Silently dropping the weights would be worse than the error --
    every label would count equally regardless of its Monte Carlo precision.
    """
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        last = model.steps[-1][0]
        model.fit(X, y, **{f"{last}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def group_scores(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    model_fn,
    n_splits: int = 5,
    task: str = "sign",
) -> dict:
    """Grouped cross-validation.

    `task="sign"` fits sign(A_t) directly; `task="regress"` fits A_t and thresholds at
    zero. The distinction turned out to matter far more than any feature choice. Squared
    error on A_t is dominated by the few states with large |A_t| -- at k=16 the top 5% of
    rows carry 43% of the weighted sum of squares -- so a regression spends its capacity
    fitting magnitudes and is then thresholded for a decision it never optimised. Fitting
    the sign directly lifts held-out AUC from 0.694 to 0.751 at k=16 and 0.594 to 0.635 at
    k=1, and AUC is threshold-free, so this is genuine ranking improvement rather than
    recalibration.

    Two metrics are reported for every model. The precision-WEIGHTED one is what an
    inverse-variance argument suggests, but those weights concentrate on near-indifferent
    states: mean |A_t| in the highest-weight quartile is 0.0011 against 0.0687 in the
    lowest. It therefore grades a model mostly where the sign is both hardest to predict
    and least consequential for regret. The unweighted figure is the one to read for "can
    this decision be made"; the weighted one for "is the fit driven by precise labels".
    """
    from sklearn.model_selection import GroupKFold

    uniq = np.unique(groups)
    n_splits = int(min(n_splits, len(uniq)))
    if n_splits < 2:
        return {k: float("nan") for k in
                ("r2", "sign_acc", "balanced_acc", "balanced_acc_weighted", "auc")} | {"n_splits": 0}

    r2s, accs, bals, bals_w, aucs = [], [], [], [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        # Exact ties carry no sign information: both actions produced the identical
        # recommendation in every replicate. Training or scoring on them invents a
        # majority class out of indifference.
        trd, ted = tr[y[tr] != 0], te[y[te] != 0]
        if len(ted) < 50 or len(trd) < 50:
            continue
        truth = y[ted] > 0
        if truth.all() or not truth.any():
            continue

        model = model_fn()
        if task == "sign":
            fit_with_weights(model, X[trd], truth_of(y[trd]), w[trd])
            score = (model.predict_proba(X[ted])[:, 1]
                     if hasattr(model, "predict_proba") else model.predict(X[ted]))
            pred = score > 0.5
        else:
            fit_with_weights(model, X[trd], y[trd], w[trd])
            score = model.predict(X[ted])
            pred = score > 0

        resid = np.average((y[ted] - (score if task == "regress" else 0)) ** 2, weights=w[ted])
        var = np.average((y[ted] - np.average(y[ted], weights=w[ted])) ** 2, weights=w[ted])
        if task == "regress":
            r2s.append(1.0 - resid / var if var > 0 else float("nan"))

        hit = pred == truth
        accs.append(float(hit.mean()))
        bals.append(0.5 * (float(hit[truth].mean()) + float(hit[~truth].mean())))
        wd = w[ted]
        bals_w.append(0.5 * (float(np.average(hit[truth], weights=wd[truth]))
                             + float(np.average(hit[~truth], weights=wd[~truth]))))
        from sklearn.metrics import roc_auc_score

        aucs.append(float(roc_auc_score(truth, score)))

    return {
        "r2": float(np.nanmean(r2s)) if r2s else float("nan"),
        "sign_acc": float(np.mean(accs)) if accs else float("nan"),
        "balanced_acc": float(np.mean(bals)) if bals else float("nan"),
        "balanced_acc_weighted": float(np.mean(bals_w)) if bals_w else float("nan"),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_splits": n_splits,
    }


def truth_of(y: np.ndarray) -> np.ndarray:
    return np.asarray(y) > 0


def model_ladder(degree_cache: dict | None = None):
    """The ordered ladder. Each entry returns a fresh unfitted estimator."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    alphas = np.logspace(-3, 3, 13)

    def linear():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

    def poly(d):
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=d, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=alphas),
        )

    def boosted():
        # Deliberately NOT a candidate policy -- it exists to measure how much predictive
        # signal is available at all, so that the polynomial's shortfall is interpretable.
        return HistGradientBoostingRegressor(
            max_depth=6, max_iter=300, learning_rate=0.06, random_state=0
        )

    def logistic():
        from sklearn.linear_model import LogisticRegression

        # Plain LogisticRegression on standardised features. A CV sweep over C moves
        # held-out balanced accuracy by under 0.003 here and costs several minutes per
        # fit, which is not a trade worth making at 87k rows x 71 columns.
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))

    def boosted_clf():
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=6, max_iter=300, learning_rate=0.06, random_state=0
        )

    return {
        "M1_linear": linear,
        "M3_poly2": lambda: poly(2),
        "M4_poly3": lambda: poly(3),
        "FLEX_boosted": boosted,
        "SIGN_logistic": logistic,
        "SIGN_boosted": boosted_clf,
    }


def schedule_baselines(data: dict, y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> dict:
    """Model 0: does a pure K(t) growth schedule predict the oracle choice?

    The threshold `c` is tuned on TRAINING folds only and scored on held-out ones, the
    same protocol the learned models get. Tuning `c` on the full corpus and reporting
    that number would flatter the baseline against cross-validated competitors, which is
    exactly the comparison this section exists to make fair.
    """
    from sklearn.model_selection import GroupKFold

    t = np.asarray(data["f_t"], dtype=np.float64)
    K = np.asarray(data["f_K"], dtype=np.float64)
    decided = y != 0
    out: dict = {
        "_class_balance": {
            "frac_search_preferred": float(np.mean(y[decided] > 0)),
            "frac_exact_ties": float(np.mean(~decided)),
        }
    }

    def bal(truth: np.ndarray, pred: np.ndarray) -> float:
        if truth.all() or not truth.any():
            return float("nan")
        hit = pred == truth
        return 0.5 * (float(hit[truth].mean()) + float(hit[~truth].mean()))

    grid = np.linspace(0.05, 8.0, 160)
    for name, power in (("sqrt_t", 0.5), ("t^1/3", 1 / 3), ("t^2/3", 2 / 3)):
        target = np.maximum(t, 1) ** power
        scores, cs = [], []
        for tr, te in GroupKFold(n_splits=5).split(t.reshape(-1, 1), y, groups):
            trd, ted = tr[decided[tr]], te[decided[te]]
            if len(ted) < 50 or len(trd) < 50:
                continue
            best = (None, -1.0)
            for c in grid:
                b = bal(y[trd] > 0, K[trd] < c * target[trd])
                if np.isfinite(b) and b > best[1]:
                    best = (float(c), b)
            cs.append(best[0])
            scores.append(bal(y[ted] > 0, K[ted] < best[0] * target[ted]))
        out[f"K < c*{name}"] = {
            "balanced_acc": float(np.nanmean(scores)),
            "mean_c": float(np.nanmean(cs)),
        }
    out["always_refine"] = {"balanced_acc": 0.5}
    out["always_search"] = {"balanced_acc": 0.5}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default="data/oracle_labels")
    ap.add_argument("--label", type=str, default="label_A", help="which commitment horizon to fit")
    ap.add_argument("--se-col", type=str, default="label_se")
    ap.add_argument("--out", type=str, default="results/growing_bandits")
    ap.add_argument("--min-precision", type=float, default=0.0,
                    help="drop states whose |A| is below this multiple of their SE")
    args = ap.parse_args()

    data_dir = ROOT / args.data
    out_dir = ROOT / args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_dir)
    n = len(data["label_A"])
    print(f"loaded {n} labelled states, {len(data)} columns  <- {data_dir}")

    y = np.asarray(data[args.label], dtype=np.float64)
    se = np.asarray(data[args.se_col], dtype=np.float64)
    w = precision_weights(se)
    groups = np.asarray(data["meta_env"])

    all_cols = list(data.keys())
    observable = design_matrix_columns([c for c in all_cols if not c.startswith(PREFIX_ORACLE)])
    print(f"{len(observable)} deployable feature columns "
          f"({sum(c.startswith(PREFIX_FEATURE) for c in observable)} f_, "
          f"{sum(c.startswith(PREFIX_ESTIMATE) for c in observable)} est_)")

    results: dict = {"n_states": int(n), "label": args.label}

    print("\n=== MODEL 0: growth schedules (the baselines to beat) ===")
    results["model0_schedules"] = schedule_baselines(data, y, w, groups)
    bal = results["model0_schedules"].pop("_class_balance")
    print(f"  ({100*bal['frac_exact_ties']:.1f}% of states are exact ties and carry no sign "
          f"information -- excluded from every classification metric below)")
    print(f"  (of the rest, SEARCH is preferred in {100*bal['frac_search_preferred']:.1f}%)")
    print("  (c tuned on TRAIN folds only, scored on held-out folds -- same protocol "
          "as the learned models)")
    for k, v in results["model0_schedules"].items():
        c = f"   mean c={v['mean_c']:.2f}" if "mean_c" in v else ""
        print(f"  {k:22s} balanced {v['balanced_acc']:.4f}{c}")
    results["class_balance"] = bal

    print("\n=== MODEL LADDER on the six theoretical variables ===")
    poly_cols = [c for c in POLY_VARS.values() if c in data]
    missing = [k for k, c in POLY_VARS.items() if c not in data]
    if missing:
        print(f"  (missing variables, skipped: {missing})")
    Xp = _matrix(data, poly_cols)
    results["ladder_poly_vars"] = {}
    for name, fn in model_ladder().items():
        sc = group_scores(Xp, y, w, groups, fn, task="sign" if name.startswith("SIGN") else "regress")
        results["ladder_poly_vars"][name] = sc
        print(f"  {name:16s} R2 {sc['r2']:+.4f}   balanced {sc['balanced_acc']:.4f}"
              f"   (weighted {sc['balanced_acc_weighted']:.4f})   AUC {sc['auc']:.4f}")

    print("\n=== ABLATIONS: does e-process evidence beat a schedule? ===")
    print("  (ridge on each feature set -- model class held fixed so the comparison is "
          "about features)")
    results["ablations"] = {}
    for name, selector in ABLATIONS.items():
        cols = selector(all_cols)
        cols = [c for c in cols if c in data]
        if not cols:
            print(f"  {name:24s} (no columns matched)")
            continue
        if name != "F_plus_oracle":
            cols = design_matrix_columns([c for c in cols if not c.startswith(PREFIX_ORACLE)])
        X = _matrix(data, cols)
        sc = group_scores(X, y, w, groups, model_ladder()["SIGN_logistic"], task="sign")
        sc["n_cols"] = len(cols)
        results["ablations"][name] = sc
        print(f"  {name:24s} ({len(cols):3d} cols)  balanced {sc['balanced_acc']:.4f}   "
              f"AUC {sc['auc']:.4f}")

    a = results["ablations"].get("A_time_only", {}).get("balanced_acc", float("nan"))
    d = results["ablations"].get("D_time_plus_eprocess", {}).get("balanced_acc", float("nan"))
    if np.isfinite(a) and np.isfinite(d):
        print(f"\n  HEADLINE: e-process evidence adds {d - a:+.4f} balanced accuracy "
              f"over (t, K_t) alone")
        results["eprocess_value"] = float(d - a)

    print("\n=== GENERALIZATION: hold out whole groups, never rows ===")
    results["generalization"] = {}
    Xe = _matrix(data, observable)
    holdouts = {"by_environment": np.asarray(data["meta_env"])}
    if "meta_family" in data:
        holdouts["by_family"] = np.asarray(data["meta_family"])
    if "meta_horizon" in data:
        holdouts["by_horizon"] = np.asarray(data["meta_horizon"]).astype(str)
    if "meta_policy" in data:
        holdouts["by_behavioural_policy"] = np.asarray(data["meta_policy"])
    for name, g in holdouts.items():
        sc = group_scores(Xe, y, w, g, model_ladder()["SIGN_logistic"], task="sign")
        results["generalization"][name] = sc
        print(f"  {name:24s} balanced {sc['balanced_acc']:.4f}   AUC {sc['auc']:.4f}   "
              f"({sc['n_splits']} folds)")

    out_json = out_dir / "tables" / f"fit_results_{args.label}.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {out_json}")

    rows = []
    for section in ("ladder_poly_vars", "ablations", "generalization"):
        for name, sc in results.get(section, {}).items():
            rows.append({"section": section, "model": name, **{k: sc.get(k) for k in
                        ("r2", "sign_acc", "balanced_acc", "balanced_acc_weighted", "auc", "n_cols", "n_splits")}})
    csv_path = out_dir / "tables" / f"fit_results_{args.label}.csv"
    if rows:
        header = list(rows[0].keys())
        with csv_path.open("w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join("" if r.get(h) is None else str(r.get(h)) for h in header) + "\n")
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
