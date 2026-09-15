"""On-policy state shift and the on-policy feature-parity gate (register #5, #11).

A learned policy was fitted to corpus states harvested under *other* behavioural
policies and scored by the scalar `features.extract_features`. Deployed, it acts on
its own trajectories through the vectorized `features_vec`. Two things can therefore
go wrong that no offline metric sees, and this module measures both from the
`Snapshot`s the runner logs (`--log-states`: 24 normalized times x 128 replicates per
learned policy and cell):

1. **Parity.** For every logged snapshot the scalar extractor (exact
   `PairwiseEvidence`) and the vectorized extractor (cached float32 log-e table) are
   run on the *same* state and compared column by column at the M1 tolerances. This
   is the >= 1000-on-policy-state gate the plan requires before any deployed number
   is believed; `analyze_deployment.py` refuses to write the main tables if a column
   fails.
2. **Shift.** The on-policy rows are compared with corpus rows of the same horizon on
   the policy's own feature list: per-feature coverage of the corpus [0.5%, 99.5%]
   range, standardized mean shift and KS statistic; a kNN(5) distance in corpus-
   standardized space against the corpus's own kNN(5) 95th percentile ("OOD fraction");
   a domain classifier (corpus vs on-policy) per feature group; and the model's
   predicted-probability histogram on both.

Everything here is analysis-side. The scalar extractor is called with the cell's
reservoir so the `oracle_*` columns come out for the reservoir diagnostics
(`reservoir_analysis.py`); those columns are read only there, after the policy has
long since acted (register #4).
"""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cold_start.growing.deploy import feature_groups as fg  # noqa: E402
from cold_start.growing.deploy.artifacts import load_model  # noqa: E402
from cold_start.growing.deploy.features_vec import extract_features_vec  # noqa: E402
from cold_start.growing.deploy.history_vec import VecSearchHistory  # noqa: E402
from cold_start.growing.deploy.pairwise_table import get_pairwise_table  # noqa: E402
from cold_start.growing.deploy.transforms import (  # noqa: E402
    RESERVOIR_RULE_FEATURES,
    beta_excess_mean,
)
from cold_start.growing.evidence import PairwiseEvidence  # noqa: E402
from cold_start.growing.features import extract_features  # noqa: E402
from cold_start.growing.labeling import Snapshot, materialize  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

log = logging.getLogger("deploy.ood")

#: The M1 parity tolerances (`tests/test_deploy_feature_parity.py`); the cached pairwise
#: table stores its cover terms in float32, hence the looser `f_log_e_pair` bound.
PARITY_RTOL = 1e-5
PARITY_ATOL = 1e-6
PARITY_LOGE_ATOL = 1e-3
#: The plan's minimum number of on-policy states behind the gate.
PARITY_MIN_STATES = 1000

KNN_K = 5
KNN_SELF_QUANTILE = 0.95
COVERAGE_QUANTILES = (0.005, 0.995)
OOD_FLAG_THRESHOLD = 0.10
#: Corpus rows the domain classifier sees per (cell, policy); more buys nothing.
DOMAIN_CLF_MAX_CORPUS = 4000
DOMAIN_CLF_FOLDS = 3
SCORE_BIN_EDGES = np.round(np.linspace(0.0, 1.0, 21), 2)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "CLOCK": fg.CLOCK,
    "QUALITY": fg.QUALITY,
    "EVIDENCE": fg.EVIDENCE,
    "HISTORY": fg.HISTORY,
}

ORACLE_COLUMNS: tuple[str, ...] = (
    "oracle_best_true_mu",
    "oracle_mu_star",
    "oracle_gap_to_mu_star",
    "oracle_p_new_beats_best_true",
    "oracle_p_new_beats_best_true_plus_0.01",
    "oracle_p_new_beats_best_true_plus_0.05",
    "oracle_p_new_beats_best_true_plus_0.1",
    "oracle_p_within_0.01_of_mu_star",
    "oracle_p_within_0.05_of_mu_star",
    "oracle_p_within_0.1_of_mu_star",
    "oracle_true_gap_leader_challenger",
    "oracle_leader_is_truly_best",
)


def parity_atol(column: str) -> float:
    return PARITY_LOGE_ATOL if column == "f_log_e_pair" else PARITY_ATOL


# ---- snapshots ------------------------------------------------------------------------


def load_snapshots(path: str | Path) -> list[Snapshot]:
    with open(path, "rb") as fh:
        snaps = pickle.load(fh)
    if not isinstance(snaps, list):
        raise TypeError(f"{path}: expected a list of Snapshots, got {type(snaps).__name__}")
    return snaps


def decision_sequences(snaps: list[Snapshot]) -> dict[int, tuple[int, np.ndarray]]:
    """Per logged replicate, ``(n0, decisions)``: the SEARCH/REFINE action of every
    completed step, from that replicate's longest history.

    A snapshot at time ``t`` is taken *before* step ``t`` and its history covers steps
    ``n0 .. t-1``, so the action taken *at* the snapshot's own time is not in the
    snapshot -- it is entry ``t - n0`` of a later snapshot's history of the same
    replicate (the runner logs the same replicates at every time, up to ``T``).
    """
    out: dict[int, tuple[int, np.ndarray]] = {}
    for snap in snaps:
        rep = int(snap.meta["replicate"])
        dec = np.asarray(snap.meta["history"].decisions, dtype=bool)
        n0 = int(snap.t) - int(dec.size)
        if rep not in out or dec.size > out[rep][1].size:
            out[rep] = (n0, dec)
    return out


def decision_at(seqs: dict[int, tuple[int, np.ndarray]], replicate: int, t: int) -> float:
    """The action taken at step `t` (1.0 SEARCH, 0.0 REFINE) or NaN if it was not logged."""
    entry = seqs.get(int(replicate))
    if entry is None:
        return float("nan")
    n0, dec = entry
    i = int(t) - n0
    if i < 0 or i >= dec.size:
        return float("nan")
    return float(dec[i])


def _subsample(snaps: list[Snapshot], sample: int | None, seed: int) -> list[Snapshot]:
    """A deterministic subset of at most `sample` snapshots (``None``/0 keeps all)."""
    if not sample or sample >= len(snaps):
        return snaps
    idx = np.sort(np.random.default_rng(seed).choice(len(snaps), size=int(sample), replace=False))
    return [snaps[i] for i in idx]


# ---- the per-(cell, policy) pass -------------------------------------------------------


@dataclass(frozen=True)
class SnapshotItem:
    """One logged (test, cell, policy): what a worker needs to score its snapshots."""

    test: str
    cell: str
    policy: str
    env_id: str
    family: str
    horizon: int
    cap: int
    env_spec: dict
    snapshots_path: str
    kind: str  # `policy_table.POLICIES[name]["kind"]`
    artifact_path: str | None  # the deployed model, for its feature list and scores
    tau: float | None  # the deployed threshold (manifest `params["tau"]`)
    sample: int | None = None
    seed: int = 0


@dataclass
class SnapshotResult:
    """Scalar features (+ oracle columns, decisions, model score) and the parity rows."""

    features: pd.DataFrame
    parity: pd.DataFrame
    policy_features: tuple[str, ...]


def policy_feature_list(kind: str, artifact_path: str | None) -> tuple[str, ...]:
    """The columns the deployed rule actually reads, from the artifact itself."""
    if kind == "model":
        if artifact_path is None:
            raise ValueError("a model policy needs its artifact path")
        return tuple(load_model(artifact_path)["features"])
    if kind == "reservoir_rule":
        return tuple(RESERVOIR_RULE_FEATURES)
    raise ValueError(f"policy kind {kind!r} logs no snapshots")


def model_scores(artifact: dict, rows: pd.DataFrame) -> np.ndarray:
    """`P(SEARCH)` of the deployed pipeline on scalar feature rows (nan_to_num as the
    trainer's `design_matrix` and `ModelPolicy.predict_proba` both do)."""
    features = list(artifact["features"])
    X = np.nan_to_num(rows.loc[:, features].to_numpy(dtype=np.float64))
    pipeline = artifact["pipeline"]
    if getattr(pipeline, "feature_names_in_", None) is not None:
        X = pd.DataFrame(X, columns=features)
    return np.asarray(pipeline.predict_proba(X)[:, 1], dtype=np.float64)


def snapshot_pass(item: SnapshotItem, *, table=None, pairwise_table=None) -> SnapshotResult:
    """Score every logged snapshot of one (cell, policy) both ways and diff them.

    Scalar side: `features.extract_features(n, S, mu, t, T, table, PairwiseEvidence(),
    reservoir, history)` -- the corpus extractor, verbatim. Vectorized side:
    `materialize(snap, 1, table, 0)` + `VecSearchHistory.from_scalar` +
    `extract_features_vec(..., get_pairwise_table(T), hist)` -- the deployment path.
    """
    T = int(item.horizon)
    all_snaps = load_snapshots(item.snapshots_path)
    if not all_snaps:
        raise ValueError(f"{item.snapshots_path} holds no snapshots")
    # Decisions come from every logged snapshot (the longest history per replicate),
    # even when only a sample of states is scored.
    seqs = decision_sequences(all_snaps)
    snaps = _subsample(all_snaps, item.sample, item.seed)
    if table is None:
        table = CSTable.load_or_build(T)
    if pairwise_table is None:
        pairwise_table = get_pairwise_table(T)
    exact = PairwiseEvidence()
    reservoir = build_reservoir(item.env_spec)
    columns = list(fg.ALL_DEPLOYABLE)

    scalar_rows: list[dict] = []
    vec_rows = np.empty((len(snaps), len(columns)), dtype=np.float64)
    for i, snap in enumerate(snaps):
        hist = snap.meta["history"]
        scalar = extract_features(
            snap.n, snap.successes, snap.mu, snap.t, T, table, exact,
            reservoir=reservoir, history=hist,
        )
        state = materialize(snap, 1, table, 0)
        vhist = VecSearchHistory.from_scalar([hist], capacity=T)
        vec = extract_features_vec(state, snap.t, T, table, pairwise_table, vhist)
        for j, c in enumerate(columns):
            vec_rows[i, j] = float(vec[c][0])
        row = {
            "test": item.test,
            "cell": item.cell,
            "policy": item.policy,
            "env_id": item.env_id,
            "family": item.family,
            "horizon": T,
            "cap": int(item.cap),
            "replicate": int(snap.meta["replicate"]),
            "t": int(snap.t),
            "k": int(snap.k),
            "n_draws": int(snap.n_draws),
            "decision_search": decision_at(seqs, int(snap.meta["replicate"]), int(snap.t)),
        }
        row.update(scalar)
        scalar_rows.append(row)
    feats = pd.DataFrame(scalar_rows)

    scalar_mat = feats.loc[:, columns].to_numpy(dtype=np.float64)
    parity = parity_table(scalar_mat, vec_rows, columns, item)

    policy_features = policy_feature_list(item.kind, item.artifact_path)
    feats["p_search_model"] = np.nan
    if item.kind == "model":
        artifact = load_model(item.artifact_path)
        feats["p_search_model"] = model_scores(artifact, feats)
    feats["tau"] = float("nan") if item.tau is None else float(item.tau)
    feats["est_I_hat"] = beta_excess_mean(
        feats["est_beta_a"].to_numpy(dtype=np.float64),
        feats["est_beta_b"].to_numpy(dtype=np.float64),
        feats["f_best_mean"].to_numpy(dtype=np.float64),
    )
    return SnapshotResult(features=feats, parity=parity, policy_features=policy_features)


def parity_table(
    scalar: np.ndarray, vec: np.ndarray, columns: list[str], item: SnapshotItem
) -> pd.DataFrame:
    """Per column: max abs diff and the number of rows outside the M1 tolerance."""
    rows: list[dict] = []
    for j, c in enumerate(columns):
        s, v = scalar[:, j], vec[:, j]
        both_nan = np.isnan(s) & np.isnan(v)
        diff = np.abs(v - s)
        diff[both_nan] = 0.0
        atol = parity_atol(c)
        fail = ~both_nan & ~(diff <= atol + PARITY_RTOL * np.abs(s))
        rows.append(
            {
                "test": item.test,
                "cell": item.cell,
                "policy": item.policy,
                "horizon": int(item.horizon),
                "column": c,
                "n_rows": int(s.shape[0]),
                "max_abs_diff": float(np.nanmax(diff)) if s.shape[0] else float("nan"),
                "n_fail": int(fail.sum()),
                "rtol": PARITY_RTOL,
                "atol": atol,
            }
        )
    return pd.DataFrame(rows)


def parity_summary(parity: pd.DataFrame) -> pd.DataFrame:
    """Pool the per-item parity rows per column (max diff, total failures, total rows)."""
    if parity.empty:
        return parity
    g = parity.groupby("column", sort=False)
    out = pd.DataFrame(
        {
            "n_rows": g["n_rows"].sum(),
            "max_abs_diff": g["max_abs_diff"].max(),
            "n_fail": g["n_fail"].sum(),
            "n_items": g.size(),
            "rtol": g["rtol"].first(),
            "atol": g["atol"].first(),
        }
    ).reset_index()
    out["passed"] = out["n_fail"] == 0
    return out


def parity_failures(parity: pd.DataFrame) -> list[str]:
    """Columns with at least one failing row (empty list = the gate passes)."""
    if parity.empty:
        return []
    failing = parity.groupby("column")["n_fail"].sum()
    return sorted(str(c) for c, n in failing.items() if int(n) > 0)


# ---- corpus reference ------------------------------------------------------------------


def load_corpus_reference(data_dir: str | Path | None = None) -> pd.DataFrame:
    """The corpus rows a policy's states are compared against: deployable columns + meta."""
    from corpus import DEFAULT_DATA_DIR, load_corpus

    df = load_corpus(DEFAULT_DATA_DIR if data_dir is None else data_dir)
    keep = list(fg.ALL_DEPLOYABLE) + [
        "meta_env", "meta_family", "meta_policy", "meta_allocation", "meta_horizon",
        "meta_trajectory", "meta_state_index", "meta_shard",
    ]
    return df.loc[:, keep].reset_index(drop=True)


def corpus_rows_for(corpus: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, str]:
    """Corpus rows of `horizon`, or every row (with scope ``all_horizons``) when the
    corpus has none at that horizon (Test D's T=2000 extrapolation)."""
    same = corpus[corpus["meta_horizon"].to_numpy(dtype=np.float64) == float(horizon)]
    if len(same):
        return same, "same_horizon"
    return corpus, "all_horizons"


# ---- shift statistics ---------------------------------------------------------------------


@dataclass
class Standardizer:
    """Corpus mean/std per feature; constant corpus columns are excluded from distances."""

    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    usable: np.ndarray  # std > 0

    @classmethod
    def fit(cls, corpus_X: np.ndarray, features: tuple[str, ...]) -> Standardizer:
        X = np.nan_to_num(np.asarray(corpus_X, dtype=np.float64))
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        usable = std > 1e-12
        return cls(tuple(features), mean, std, usable)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        z = (X[:, self.usable] - self.mean[self.usable]) / self.std[self.usable]
        return z


def feature_shift(
    on: pd.DataFrame, corpus: pd.DataFrame, features: tuple[str, ...]
) -> pd.DataFrame:
    """Per feature: coverage of the corpus central range, standardized shift, KS."""
    lo_q, hi_q = COVERAGE_QUANTILES
    rows: list[dict] = []
    for c in features:
        x_on = np.nan_to_num(on[c].to_numpy(dtype=np.float64))
        x_c = np.nan_to_num(corpus[c].to_numpy(dtype=np.float64))
        lo, hi = np.quantile(x_c, [lo_q, hi_q])
        sd = float(x_c.std(ddof=0))
        shift = (float(x_on.mean()) - float(x_c.mean())) / sd if sd > 1e-12 else float("nan")
        if x_on.size and x_c.size:
            ks = float(ks_2samp(x_on, x_c, method="asymp").statistic)
        else:
            ks = float("nan")
        rows.append(
            {
                "feature": c,
                "group": fg_group_of(c),
                "n_on": int(x_on.size),
                "n_corpus": int(x_c.size),
                "corpus_lo": float(lo),
                "corpus_hi": float(hi),
                "frac_below": float(np.mean(x_on < lo)) if x_on.size else float("nan"),
                "frac_above": float(np.mean(x_on > hi)) if x_on.size else float("nan"),
                "frac_outside": float(np.mean((x_on < lo) | (x_on > hi))) if x_on.size else float("nan"),
                "mean_on": float(x_on.mean()) if x_on.size else float("nan"),
                "mean_corpus": float(x_c.mean()) if x_c.size else float("nan"),
                "sd_corpus": sd,
                "std_mean_shift": shift,
                "ks": ks,
            }
        )
    return pd.DataFrame(rows)


def fg_group_of(column: str) -> str:
    for name, cols in FEATURE_GROUPS.items():
        if column in cols:
            return name
    return "?"


def self_knn_threshold(
    corpus_Z: np.ndarray, k: int = KNN_K, q: float = KNN_SELF_QUANTILE
) -> float:
    """`q`-quantile of each corpus row's distance to its k-th nearest *other* corpus row."""
    from sklearn.neighbors import NearestNeighbors

    Z = np.asarray(corpus_Z, dtype=np.float64)
    if Z.shape[0] <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, _ = nn.kneighbors(Z)
    return float(np.quantile(dist[:, k], q))


def knn_ood(
    on_Z: np.ndarray, corpus_Z: np.ndarray, threshold: float, k: int = KNN_K
) -> dict:
    """Fraction of on-policy rows whose k-th-nearest-corpus distance exceeds `threshold`."""
    from sklearn.neighbors import NearestNeighbors

    Z = np.asarray(corpus_Z, dtype=np.float64)
    Q = np.asarray(on_Z, dtype=np.float64)
    if Z.shape[0] < k or Q.shape[0] == 0 or not np.isfinite(threshold):
        return {"ood_frac": float("nan"), "knn_dist_median": float("nan"), "knn_threshold": threshold}
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    dist, _ = nn.kneighbors(Q)
    d = dist[:, k - 1]
    return {
        "ood_frac": float(np.mean(d > threshold)),
        "knn_dist_median": float(np.median(d)),
        "knn_threshold": float(threshold),
    }


def domain_classifier_auc(on_X: np.ndarray, corpus_X: np.ndarray, seed: int = 0) -> float:
    """Cross-validated AUC of a logistic "is this row on-policy?" classifier.

    0.5 means the two sets are indistinguishable on these columns; 1.0 means
    perfectly separable. The corpus side is subsampled to `DOMAIN_CLF_MAX_CORPUS`
    rows so a cell's cost does not scale with the corpus.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    C = np.nan_to_num(np.asarray(corpus_X, dtype=np.float64))
    if C.shape[0] > DOMAIN_CLF_MAX_CORPUS:
        C = C[np.sort(rng.choice(C.shape[0], DOMAIN_CLF_MAX_CORPUS, replace=False))]
    Q = np.nan_to_num(np.asarray(on_X, dtype=np.float64))
    n_min = min(C.shape[0], Q.shape[0])
    if n_min < DOMAIN_CLF_FOLDS or C.shape[1] == 0:
        return float("nan")
    X = np.vstack([C, Q])
    y = np.concatenate([np.zeros(C.shape[0], dtype=np.int64), np.ones(Q.shape[0], dtype=np.int64)])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    cv = StratifiedKFold(n_splits=DOMAIN_CLF_FOLDS, shuffle=True, random_state=int(seed))
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def score_histogram(p_on: np.ndarray, p_corpus: np.ndarray) -> pd.DataFrame:
    """Shares of predicted P(SEARCH) per bin, on-policy vs corpus."""
    edges = SCORE_BIN_EDGES
    on = np.asarray(p_on, dtype=np.float64)
    on = on[np.isfinite(on)]
    cp = np.asarray(p_corpus, dtype=np.float64)
    cp = cp[np.isfinite(cp)]
    h_on = np.histogram(on, bins=edges)[0].astype(np.float64)
    h_cp = np.histogram(cp, bins=edges)[0].astype(np.float64)
    return pd.DataFrame(
        {
            "bin_lo": edges[:-1],
            "bin_hi": edges[1:],
            "frac_on": h_on / on.size if on.size else np.nan,
            "frac_corpus": h_cp / cp.size if cp.size else np.nan,
            "n_on": int(on.size),
            "n_corpus": int(cp.size),
        }
    )


@dataclass
class OODResult:
    summary: dict
    features: pd.DataFrame
    score_hist: pd.DataFrame


def ood_for_policy(
    on: pd.DataFrame,
    corpus_rows: pd.DataFrame,
    corpus_scope: str,
    policy_features: tuple[str, ...],
    *,
    standardizer: Standardizer | None = None,
    knn_threshold: float | None = None,
    p_corpus: np.ndarray | None = None,
    seed: int = 0,
) -> OODResult:
    """All shift statistics of one (cell, policy) against its corpus reference.

    `standardizer` and `knn_threshold` may be passed in when the caller caches them
    per (horizon, feature list) -- the corpus self-kNN is the expensive part.
    """
    feats = tuple(policy_features)
    if standardizer is None:
        standardizer = Standardizer.fit(corpus_rows.loc[:, list(feats)].to_numpy(), feats)
    corpus_Z = standardizer.transform(corpus_rows.loc[:, list(feats)].to_numpy())
    on_Z = standardizer.transform(on.loc[:, list(feats)].to_numpy())
    if knn_threshold is None:
        knn_threshold = self_knn_threshold(corpus_Z)
    knn = knn_ood(on_Z, corpus_Z, knn_threshold)
    shift = feature_shift(on, corpus_rows, feats)

    summary = {
        "n_on": int(len(on)),
        "n_corpus": int(len(corpus_rows)),
        "corpus_scope": corpus_scope,
        "n_features": len(feats),
        "n_features_standardizable": int(standardizer.usable.sum()),
        **knn,
        "max_frac_outside": float(shift["frac_outside"].max()) if len(shift) else float("nan"),
        "mean_frac_outside": float(shift["frac_outside"].mean()) if len(shift) else float("nan"),
        "mean_abs_std_shift": float(shift["std_mean_shift"].abs().mean()) if len(shift) else float("nan"),
        "max_abs_std_shift": float(shift["std_mean_shift"].abs().max()) if len(shift) else float("nan"),
        "max_ks": float(shift["ks"].max()) if len(shift) else float("nan"),
        "worst_feature_by_ks": (
            str(shift.loc[shift["ks"].idxmax(), "feature"]) if len(shift) and shift["ks"].notna().any() else ""
        ),
    }
    on_all = on.loc[:, list(feats)].to_numpy()
    corpus_all = corpus_rows.loc[:, list(feats)].to_numpy()
    summary["domain_auc_all"] = domain_classifier_auc(on_all, corpus_all, seed=seed)
    for name, cols in FEATURE_GROUPS.items():
        sub = [c for c in feats if c in cols]
        if not sub:
            summary[f"domain_auc_{name}"] = float("nan")
            continue
        summary[f"domain_auc_{name}"] = domain_classifier_auc(
            on.loc[:, sub].to_numpy(), corpus_rows.loc[:, sub].to_numpy(), seed=seed
        )
    p_on = on["p_search_model"].to_numpy(dtype=np.float64) if "p_search_model" in on else np.empty(0)
    hist = score_histogram(p_on, p_corpus if p_corpus is not None else np.empty(0))
    return OODResult(summary=summary, features=shift, score_hist=hist)


def flag_cells(ood_summary: pd.DataFrame, threshold: float = OOD_FLAG_THRESHOLD) -> pd.DataFrame:
    """Rows whose kNN OOD fraction exceeds `threshold` (the plan's prominent flag)."""
    if ood_summary.empty or "ood_frac" not in ood_summary:
        return pd.DataFrame(columns=list(ood_summary.columns) + ["threshold"])
    out = ood_summary[ood_summary["ood_frac"] > threshold].copy()
    out["threshold"] = threshold
    return out.reset_index(drop=True)


# ---- corpus OOF scores (optional; the trainer's own protocol) ----------------------------


def corpus_oof_scores(corpus_full: pd.DataFrame, variants: list[str], n_jobs: int = 1) -> pd.DataFrame:
    """Out-of-fold P(SEARCH) per corpus row for each variant, by `train_policies`'
    env-grouped 5-fold protocol (NaN outside the variant's eligible rows).

    Recomputed rather than read because the trainer does not persist its OOF
    scores; every function used is the trainer's own, so the folds and estimators are
    the ones behind `offline_metrics.csv`.
    """
    import train_policies as tp
    from corpus import label_columns

    out = pd.DataFrame(index=corpus_full.index)
    groups = corpus_full["meta_env"].to_numpy().astype(str)
    for name in variants:
        variant = tp.VARIANTS[name]
        features = tp.feature_list(variant)
        X = tp.design_matrix(corpus_full, features)
        label, se_col = label_columns(variant["k"])
        y = corpus_full[label].to_numpy(dtype=np.float64) > 0
        w = None
        if variant["estimator"] == "logit_w":
            w = tp.precision_weights(corpus_full[se_col].to_numpy(dtype=np.float64))
        eligible, universe = tp.eligible_masks(corpus_full, variant)
        fold = tp.fold_ids_by_group(groups, universe, 5)
        out[name] = tp.oof_scores(
            variant["estimator"], variant["feature_set"], X, y, w, fold, eligible, n_jobs
        )
    return out
