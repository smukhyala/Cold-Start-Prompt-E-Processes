"""What does `phi_k4` read? Feature importance on the states it visits, and the environment signal.

Section 12.4 says the k = 4 model's value is sizing K to the environment; §12.5 says it is
not the fraction of held arms near the best. This script asks the model directly, on the
robustness panel's T <= 200 cells -- the exact cells and episodes of H1b' (`cells.make_cell`
on the test split; the first `--replicates` episodes are, by CRN, the deployed ones):

1. **Harvest.** Build the deployed policy through `policy_table` (its own tau, its own
   artifact) and wrap its `predict_proba` so every decision it makes during `run_cell`
   is recorded with the feature matrix it saw. No re-implementation of the feature path.
2. **Importance, per horizon.** For each of its features: the standardized-logit
   contribution scale ``|coef_j| * sd_deployed(x_j) / scale_j``, and permutation
   importance on the decision -- shuffle the column across the horizon's rows and
   measure the mean |dP(SEARCH)| and the fraction of decisions that flip at tau.
3. **Environment signal.** For each feature, the Spearman correlation across the 30
   environments between its early-episode mean (rows with ``t <= early_frac * T``) and
   the K_final the deployed `phi_k4` reached in that environment
   (`cells_robust_primary.csv`). A feature that is both important to the decision and
   tracks the per-environment K is the candidate for the next null rule.

Writes ``tables/model_reads_importance.csv`` and ``tables/model_reads_env_signal.csv``.
Diagnostic only: nothing here is a test.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cells  # noqa: E402
import policy_table as pt  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy import feature_groups as fg  # noqa: E402
from cold_start.growing.deploy.harness import run_cell  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

log = logging.getLogger("deploy.model_reads")

HORIZONS: tuple[int, ...] = (50, 100, 200)


def group_of(column: str) -> str:
    for name, cols in (("CLOCK", fg.CLOCK), ("QUALITY", fg.QUALITY), ("EVIDENCE", fg.EVIDENCE), ("HISTORY", fg.HISTORY)):
        if column in cols:
            return name
    return "?"


def build(policy_name: str, horizon: int, n_replicates: int, out_dir: Path):
    baseline = pt.load_baseline_params(out_dir / "baseline_params.json")
    thresholds = pt.load_thresholds(out_dir / "thresholds.json")
    params = pt.resolve_params(policy_name, horizon, cap=rd.DEFAULT_CAP, baseline_params=baseline,
                               thresholds=thresholds, models_dir=pt.DEFAULT_MODELS_DIR)
    artifact = rd._artifact(params["artifact"])
    table = rd._table(horizon, 0.05)
    pairwise = rd._pairwise(horizon) if any(c in fg.EVIDENCE_LOGE for c in artifact["features"]) else None
    policy = pt.build_policy(policy_name, params, horizon=horizon, n_replicates=n_replicates,
                             table=table, pairwise=pairwise, artifact=artifact)
    return policy, artifact, table, params


class _Recorder:
    """Wraps a `ModelPolicy.predict_proba`; records (t, X, p, K_t) at every decision."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.original = policy.predict_proba
        self.records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

    def __call__(self, state, t, horizon):
        from cold_start.growing.deploy.features_vec import extract_features_vec, feature_matrix

        p = self.original(state, t, horizon)
        pol = self.policy
        feats = extract_features_vec(state, t, horizon, pol.table, pol.pairwise, pol.history,
                                     columns=pol.features)
        X = np.nan_to_num(np.asarray(feature_matrix(feats, pol.features), dtype=np.float64))
        self.records.append((int(t), X, np.asarray(p, dtype=np.float64), state.Kt.copy()))
        return p


def harvest(policy_name: str, *, env_ids, horizons, n_replicates: int, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """One row per (env, T, episode, decision time): the features the policy saw and its P(SEARCH)."""
    frames: list[pd.DataFrame] = []
    artifact = None
    for T in horizons:
        for env_id in env_ids:
            spec = cells.make_cell(rd.TEST_SPLIT, env_id, int(T), rd.DEFAULT_CAP, n_replicates)
            policy, artifact, table, _ = build(policy_name, int(T), n_replicates, out_dir)
            recorder = _Recorder(policy)
            policy.predict_proba = recorder
            run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
            features = list(policy.features)
            for t, X, p, Kt in recorder.records:
                frame = pd.DataFrame(X, columns=features)
                frame.insert(0, "env_id", env_id)
                frame.insert(1, "horizon", int(T))
                frame.insert(2, "t", t)
                frame.insert(3, "episode", np.arange(X.shape[0]))
                frame.insert(4, "K_t", Kt)
                frame["p_search"] = p
                frames.append(frame)
            log.info("%s T=%d: %d decision times, %d rows", env_id, T, len(recorder.records),
                     sum(r[1].shape[0] for r in recorder.records))
    return pd.concat(frames, ignore_index=True), artifact


def importance(rows: pd.DataFrame, artifact: dict, *, seed: int = 0) -> pd.DataFrame:
    features = list(artifact["features"])
    pipeline = artifact["pipeline"]
    tau = float(artifact["tau"])
    scaler = pipeline.named_steps["standardscaler"]
    logit = pipeline.named_steps["logisticregression"]
    coef = np.asarray(logit.coef_).ravel()
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for T, sub in rows.groupby("horizon"):
        X = sub.loc[:, features].to_numpy(dtype=np.float64)
        p0 = pipeline.predict_proba(X)[:, 1]
        d0 = p0 > tau
        sd = X.std(axis=0, ddof=1)
        for j, name in enumerate(features):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            p1 = pipeline.predict_proba(Xp)[:, 1]
            out.append({
                "horizon": int(T), "feature": name, "group": group_of(name), "coef": float(coef[j]),
                "sd_deployed": float(sd[j]), "train_scale": float(scaler.scale_[j]),
                "contribution_sd": float(abs(coef[j]) * sd[j] / scaler.scale_[j]),
                "perm_mean_abs_dp": float(np.mean(np.abs(p1 - p0))),
                "perm_flip_rate": float(np.mean((p1 > tau) != d0)),
                "n_rows": int(X.shape[0]), "search_rate": float(d0.mean()),
            })
    return pd.DataFrame(out)


def env_signal(rows: pd.DataFrame, artifact: dict, k_final: pd.DataFrame, *, early_frac: float) -> pd.DataFrame:
    """Spearman across environments of each feature's early mean with phi_k4's deployed K_final."""
    features = list(artifact["features"])
    out: list[dict] = []
    for T, sub in rows.groupby("horizon"):
        early = sub[sub["t"] <= early_frac * T]
        means = early.groupby("env_id")[features].mean()
        k = k_final[k_final["horizon"] == T].set_index("env_id")["k_final"]
        means = means.loc[means.index.intersection(k.index)]
        kk = k.loc[means.index].to_numpy()
        for name in features:
            x = means[name].to_numpy()
            if np.std(x) == 0:
                rho, pval = 0.0, 1.0
            else:
                rho, pval = spearmanr(x, kk)
            out.append({"horizon": int(T), "feature": name, "group": group_of(name),
                        "spearman_with_k_final": float(rho), "p": float(pval), "n_envs": int(len(kk)),
                        "early_frac": early_frac, "n_rows_early": int(len(early))})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="phi_k4")
    ap.add_argument("--replicates", type=int, default=16, help="episodes per cell (the first N deployed ones)")
    ap.add_argument("--early-frac", type=float, default=0.25)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    env_ids = tuple(cells.env_ids_for("corpus")) if hasattr(cells, "env_ids_for") else tuple(cells.ALL_CORPUS_ENVS)
    rows, artifact = harvest(args.policy, env_ids=env_ids, horizons=HORIZONS,
                             n_replicates=args.replicates, out_dir=args.out_dir)
    log.info("harvested %d decision rows over %d environments", len(rows), rows["env_id"].nunique())
    imp = importance(rows, artifact)
    k_final = pd.read_csv(args.out_dir / "tables" / "cells_robust_primary.csv")
    k_final = k_final[k_final["policy"] == args.policy][["env_id", "horizon", "k_final"]]
    sig = env_signal(rows, artifact, k_final, early_frac=args.early_frac)
    tables = args.out_dir / "tables"
    imp.to_csv(tables / "model_reads_importance.csv", index=False)
    sig.to_csv(tables / "model_reads_env_signal.csv", index=False)
    rows.to_parquet(args.out_dir / "model_reads_rows.parquet", index=False)

    merged = imp.merge(sig, on=["horizon", "feature", "group"])
    for T, sub in merged.groupby("horizon"):
        top_imp = sub.sort_values("perm_mean_abs_dp", ascending=False).head(8)
        top_sig = sub.reindex(sub["spearman_with_k_final"].abs().sort_values(ascending=False).index).head(8)
        log.info("T=%d  top by permutation importance:\n%s", T,
                 top_imp[["feature", "group", "coef", "contribution_sd", "perm_mean_abs_dp", "perm_flip_rate",
                          "spearman_with_k_final"]].round(4).to_string(index=False))
        log.info("T=%d  top by |Spearman with env K_final| (early rows):\n%s", T,
                 top_sig[["feature", "group", "spearman_with_k_final", "p", "perm_mean_abs_dp"]].round(4).to_string(index=False))
    grp = imp.groupby(["horizon", "group"])["perm_mean_abs_dp"].sum().unstack()
    log.info("permutation importance summed by group:\n%s", grp.round(4).to_string())
    log.info("wrote %s and %s", tables / "model_reads_importance.csv", tables / "model_reads_env_signal.csv")


if __name__ == "__main__":
    main()
