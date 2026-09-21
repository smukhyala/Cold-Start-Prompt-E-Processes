"""Select `adaptive_K_star`'s (alpha, c, b) on the validation split (Pre-registration 4).

The environment-adaptive null model: SEARCH while ``K_t < c * T^alpha * (1 + b * (1 - q_t))``,
``q_t`` the fraction of held arms within 0.05 of the best held arm's posterior mean. One
triple for every horizon, the argmin of environment-equal-weight pooled regret over the
8 main environments x T in {50, 100, 200} on the validation split, cap 64, M = 2000, over
the registered 50-candidate grid. Writes ``tables/adaptive_k_selection.csv`` and
``baseline_params.json["adaptive_K_star"] = {alpha, c, b, pooled_regret}`` plus
``meta.adaptive_K_star``, appended to the file in its existing key order.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cells  # noqa: E402
import k_star_envelope as ks  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy.harness import CellSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

log = logging.getLogger("deploy.select_adaptive_k")

SPLIT = "val"
CAP = rd.DEFAULT_CAP
N_REPLICATES = 2000
HORIZONS: tuple[int, ...] = (50, 100, 200)
ALPHAS: tuple[float, ...] = (0.5, 0.75)
CS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0)
BS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
BLOCK = "adaptive_K_star"


def grid() -> tuple[tuple[float, float, float], ...]:
    return tuple((a, c, b) for a in ALPHAS for c in CS for b in BS)


@dataclass(frozen=True)
class Item:
    spec: CellSpec
    alpha: float
    c: float
    b: float


def run_item(item: Item) -> dict:
    spec = item.spec
    table = ks._table(spec.horizon, spec.alpha)
    policy = make_policy("adaptive_K", params={"alpha": item.alpha, "c": item.c, "b": item.b},
                         horizon=spec.horizon, n_replicates=spec.n_replicates, table=table)
    res = run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
    regret = res.regret(PRIMARY_RECOMMENDER)
    return {
        "level": "env", "split": SPLIT, "env_id": spec.env_id, "horizon": int(spec.horizon),
        "cap": int(spec.cap), "base_seed": int(spec.base_seed),
        "alpha": item.alpha, "c": item.c, "b": item.b, "n_envs": 1,
        "n_episodes": int(spec.n_replicates), "regret": float(regret.mean()),
        "regret_se": float(regret.std(ddof=1) / np.sqrt(len(regret))),
        "k_final": float(res.k_final.mean()), "search_frac": float(res.search_frac.mean()),
    }


def select(
    *,
    env_ids: tuple[str, ...] = cells.MAIN_ENV_IDS,
    horizons: tuple[int, ...] = HORIZONS,
    candidates: tuple[tuple[float, float, float], ...] | None = None,
    n_replicates: int = N_REPLICATES,
    out_dir: Path = rd.DEFAULT_OUT_DIR,
    workers: int = 1,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    candidates = tuple(candidates) if candidates is not None else grid()
    items = [
        Item(cells.make_cell(SPLIT, env_id, int(T), CAP, int(n_replicates)), a, c, b)
        for env_id in env_ids for T in horizons for (a, c, b) in candidates
    ]
    log.info("%d items: %d envs x %d horizons x %d candidates, M=%d, split=%s, cap=%d",
             len(items), len(env_ids), len(horizons), len(candidates), n_replicates, SPLIT, CAP)
    t0 = time.time()
    results: list[dict] = []
    if int(workers) <= 1:
        results = [run_item(it) for it in items]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(int(workers), initializer=ks._worker_init, initargs=(logging.INFO,)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_item, items, chunksize=1), 1):
                results.append(r)
                if i % 100 == 0 or i == len(items):
                    log.info("[%d/%d] elapsed %.0fs", i, len(items), time.time() - t0)
    env_rows = pd.DataFrame(results)
    pooled_rows = []
    for (a, c, b), sub in env_rows.groupby(["alpha", "c", "b"]):
        pooled_rows.append({
            "level": "pooled", "split": SPLIT, "env_id": "all", "horizon": "all", "cap": CAP,
            "base_seed": -1, "alpha": a, "c": c, "b": b, "n_envs": int(sub["env_id"].nunique()),
            "n_episodes": int(sub["n_episodes"].sum()), "regret": float(sub["regret"].mean()),
            "regret_se": float(np.sqrt((sub["regret_se"] ** 2).sum()) / len(sub)),
            "k_final": float(sub["k_final"].mean()), "search_frac": float(sub["search_frac"].mean()),
        })
    pooled = pd.DataFrame(pooled_rows).sort_values(["alpha", "c", "b"]).reset_index(drop=True)
    best = int(np.argmin(pooled["regret"].to_numpy()))
    pooled["is_argmin"] = False
    pooled.loc[best, "is_argmin"] = True
    env_rows["is_argmin"] = False
    frame = pd.concat([env_rows, pooled], ignore_index=True)
    win = pooled.loc[best]
    log.info("selected alpha=%.2f c=%.1f b=%.1f: pooled validation regret %.6f over %d cells (%.0fs)",
             win["alpha"], win["c"], win["b"], win["regret"], len(env_ids) * len(horizons), time.time() - t0)

    path = out_dir / "baseline_params.json"
    params = json.load(open(path)) if path.exists() else {}
    params[BLOCK] = {"alpha": float(win["alpha"]), "c": float(win["c"]), "b": float(win["b"]),
                     "pooled_regret": float(win["regret"])}
    params.setdefault("meta", {})[BLOCK] = {
        "select_split": SPLIT, "cap": CAP, "n_replicates": int(n_replicates),
        "horizons": [int(T) for T in horizons], "envs": list(env_ids),
        "grid": {"alpha": list(ALPHAS), "c": list(CS), "b": list(BS)}, "n_candidates": len(candidates),
        "pooling": "equal weight over envs of the cell mean regret", "band": 0.05,
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(params, fh, indent=2)
        fh.write("\n")
    tmp.replace(path)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "adaptive_k_selection.csv", index=False)
    return frame


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return select(n_replicates=args.replicates, out_dir=args.out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
