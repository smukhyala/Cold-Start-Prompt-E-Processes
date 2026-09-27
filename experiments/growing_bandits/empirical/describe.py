"""Descriptive reads of the real prompt pools (Pre-registration 9, item 4) and the flatness guard.

    .venv/bin/python experiments/growing_bandits/empirical/describe.py

reads ``tables/emp_kgrid.csv``, the reservoirs, the ``emp`` episodes and the corpus K*
envelope (``tables/k_star_envelope_all33.csv``), and writes
``tables/emp_{pool_location,kstar,kstar_location,flatness,cross_pool,cap64,rule_gaps}.csv``.

K* here is an in-sample argmin on the pool's own reservoir: a ceiling, never a deployable
policy. A cell is *informative* iff fixed-K regret moves by more than 5 x MEI = 0.01 over
the K-grid; Pre-registration 9 evaluates its contrasts on informative primary cells only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.reservoirs import Reservoir, build_reservoir  # noqa: E402

log = logging.getLogger("empirical.describe")

MEI = 0.002
FLATNESS_THRESHOLD = 5 * MEI
LEVEL_RULE_B = 4.0
STATS: tuple[str, ...] = ("level", "sd", "q99_minus_mean")


def pool_summary(res: Reservoir) -> dict:
    m = float(res.mean())
    q01, q99 = res.quantile(0.01), res.quantile(0.99)
    u = (np.arange(20_001) + 0.5) / 20_001
    draws = res.sample_from_uniforms(u)
    return {"level": m, "sd": float(np.std(draws)), "q99_minus_mean": float(q99 - m), "spread": float(q99 - q01),
            "validation_error": getattr(res, "validation_error", None)}


def k_star_table(kgrid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, pool, variant, T), sub in kgrid.groupby(["env_id", "pool", "variant", "horizon"]):
        sub = sub.sort_values("K")
        best = sub.iloc[int(np.argmin(sub["regret"].to_numpy()))]
        rng_ = float(sub["regret"].max() - sub["regret"].min())
        rows.append({"env_id": env, "pool": pool, "variant": variant, "horizon": int(T), "k_star": int(best["K"]),
                     "regret_at_k_star": float(best["regret"]), "regret_max": float(sub["regret"].max()),
                     "regret_range": rng_, "informative": rng_ > FLATNESS_THRESHOLD})
    return pd.DataFrame(rows)


def flatness(kgrid: pd.DataFrame) -> pd.DataFrame:
    return k_star_table(kgrid)[["env_id", "pool", "variant", "horizon", "regret_range", "informative"]]


def cross_pool_prediction(levels: dict[str, float], kstar: pd.DataFrame) -> pd.DataFrame:
    low, high = sorted(levels, key=lambda p: levels[p])
    rows = []
    prim = kstar[kstar["variant"] == "npmle"]
    for T in sorted(prim["horizon"].unique()):
        k = {r.pool: int(r.k_star) for r in prim[prim["horizon"] == T].itertuples()}
        if low not in k or high not in k:
            continue
        rows.append({"horizon": int(T), "low_pool": low, "high_pool": high, "k_star_low": k[low],
                     "k_star_high": k[high], "sign_agrees": k[low] > k[high], "tie": k[low] == k[high],
                     "ratio_observed": k[low] / k[high],
                     "ratio_predicted": float(np.exp(LEVEL_RULE_B * (levels[high] - levels[low])))})
    return pd.DataFrame(rows)


def cap64_cost(kgrid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, pool, variant, T), sub in kgrid[kgrid["horizon"].isin([500, 1000])].groupby(
            ["env_id", "pool", "variant", "horizon"]):
        at64 = sub[sub["K"] == 64]
        if at64.empty:
            continue
        rows.append({"env_id": env, "pool": pool, "variant": variant, "horizon": int(T),
                     "cost": float(at64["regret"].iloc[0] - sub["regret"].min()), "label": "extrapolation"})
    return pd.DataFrame(rows)


def rule_gaps(out_dir: Path, kstar: pd.DataFrame) -> pd.DataFrame:
    col = f"regret_{PRIMARY_RECOMMENDER}"
    rows = []
    for r in kstar.itertuples():
        cell = out_dir / "episodes" / "emp" / f"{r.env_id}_T{r.horizon}_cap{r.horizon}"
        for path in sorted(cell.glob("*.parquet")):
            regret = float(pd.read_parquet(path, columns=[col])[col].mean())
            rows.append({"env_id": r.env_id, "pool": r.pool, "variant": r.variant, "horizon": r.horizon,
                         "policy": path.stem, "regret": regret, "gap_to_k_star": regret - r.regret_at_k_star})
    return pd.DataFrame(rows)


def corpus_k_star(envelope: pd.DataFrame) -> pd.DataFrame:
    """One K*(T) per corpus environment and horizon from ``k_star_envelope_all33.csv`` (uncapped rows)."""
    env = envelope[(envelope["level"] == "env") & (envelope["cap"] == envelope["horizon"])]
    per = env.groupby(["env_id", "horizon"])["k_star"].agg(["nunique", "first"]).reset_index()
    bad = per[per["nunique"] != 1]
    if len(bad):
        raise ValueError(f"envelope has more than one k_star for {bad[['env_id', 'horizon']].to_dict('records')[:3]}")
    return per.rename(columns={"first": "k_star"})[["env_id", "horizon", "k_star"]]


def locate_k_star(kstar: pd.DataFrame, envelope: pd.DataFrame) -> pd.DataFrame:
    """Where each pool's K*(T) falls in the corpus environments' K*(T) distribution (spec section 6, item 4)."""
    corpus = corpus_k_star(envelope)
    rows = []
    for r in kstar.to_dict("records"):
        ks = corpus.loc[corpus["horizon"] == r["horizon"], "k_star"].to_numpy(dtype=float)
        k = float(r["k_star"])
        n = ks.size
        rows.append({"env_id": r["env_id"], "pool": r["pool"], "variant": r["variant"], "horizon": int(r["horizon"]),
                     "k_star": int(r["k_star"]), "corpus_n_envs": int(n),
                     "corpus_k_star_min": float(ks.min()) if n else np.nan,
                     "corpus_k_star_median": float(np.median(ks)) if n else np.nan,
                     "corpus_k_star_max": float(ks.max()) if n else np.nan,
                     "corpus_pct_below": float(np.mean(ks < k)) if n else np.nan,
                     "corpus_pct_at_or_below": float(np.mean(ks <= k)) if n else np.nan,
                     "inside_corpus_range": bool(n and ks.min() <= k <= ks.max())})
    return pd.DataFrame(rows)


def locate(pool_rows: pd.DataFrame, corpus_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in pool_rows.to_dict("records"):
        row = dict(r)
        for stat in STATS:
            row[f"{stat}_pct_below"] = float(np.mean(corpus_rows[stat].to_numpy() < r[stat]))
            row[f"{stat}_corpus_min"] = float(corpus_rows[stat].min())
            row[f"{stat}_corpus_max"] = float(corpus_rows[stat].max())
        out.append(row)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--envelope", type=Path, default=None,
                    help="corpus K* envelope (default: <out-dir>/tables/k_star_envelope_all33.csv)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tables = args.out_dir / "tables"
    kgrid = pd.read_csv(tables / "emp_kgrid.csv")

    import cells

    pools = []
    for p in replay.POOLS:
        for v in replay.VARIANTS:
            res = replay.load_reservoir(replay.RES_DIR / f"{p}_{v}.json")
            pools.append({"env_id": replay.env_id(p, v), "pool": p, "variant": v, **pool_summary(res)})
    pools = pd.DataFrame(pools)
    corpus = pd.DataFrame([{"env_id": e, **pool_summary(build_reservoir(spec))} for e, spec in cells.ALL_ENVS.items()])
    kstar = k_star_table(kgrid)
    levels = {r.pool: r.level for r in pools[pools["variant"] == "npmle"].itertuples()}

    locate(pools, corpus).to_csv(tables / "emp_pool_location.csv", index=False)
    kstar.to_csv(tables / "emp_kstar.csv", index=False)
    envelope = pd.read_csv(args.envelope or tables / "k_star_envelope_all33.csv")
    locate_k_star(kstar, envelope).to_csv(tables / "emp_kstar_location.csv", index=False)
    flatness(kgrid).to_csv(tables / "emp_flatness.csv", index=False)
    cross_pool_prediction(levels, kstar).to_csv(tables / "emp_cross_pool.csv", index=False)
    cap64_cost(kgrid).to_csv(tables / "emp_cap64.csv", index=False)
    rule_gaps(args.out_dir, kstar).to_csv(tables / "emp_rule_gaps.csv", index=False)
    prim = kstar[(kstar["variant"] == "npmle") & kstar["horizon"].isin(replay.PRIMARY_HORIZONS)]
    log.info("informative primary cells: %d of %d", int(prim["informative"].sum()), len(prim))


if __name__ == "__main__":
    main()
