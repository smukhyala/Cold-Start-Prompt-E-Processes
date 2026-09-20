"""K*(env, T): the fixed-K regret envelope on the tune split (NEXT-STEPS 2.5).

The study's cap of 64 arms was inherited from the corpus (`CORPUS_MAX_LIVE_ARMS`), not
chosen. `DEPLOYMENT_RESULTS.md` section 9.2 records that no exact optimum was ever
computed, so nothing in the study is validated against a ceiling. This script measures
the achievable reference that stands in for one: for every (environment, horizon) it
runs `fixed_K` -- `PowerSchedule(alpha=0, c=K)`, the policy that recruits until it holds
K arms and then refines -- over a grid of K with the cap removed (``cap="T"``), and
reports the regret of each K and the per-environment and pooled argmin.

Two readings the output must NOT be given, stated here so the table cannot be quoted
without them:

* K* is a **tune-split in-sample argmin**. It is a design instrument for choosing a cap,
  never a test-split result, and it is not a policy the deployment tables compare
  against. Deployed `fixed_K16` (T=50: 0.135722 vs `phi_k4` 0.125865) shows a
  per-horizon-UNTUNED fixed K is not a tie with the learned policy.
* K* is **environment-specific** (the pilot's per-env argmins at T=1000 spanned
  128-400), so the per-env argmin is the ceiling and the pooled argmin is the baseline;
  they are separate rows and must be quoted separately.

Cells come from `cells.make_cell` with ``split="tune"``, so `base_seed` is the tune
split's and is disjoint from every test-split seed by construction; nothing here can
leak into a test-split number. A K that exceeds the horizon cannot be reached and is
not run.

Usage::

    .venv/bin/python experiments/growing_bandits/deploy/k_star_envelope.py --workers 12

writes ``results/growing_bandits/deploy/tables/k_star_envelope.csv``.
"""

from __future__ import annotations

import argparse
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

from cold_start.growing.deploy.harness import CellSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

log = logging.getLogger("k_star_envelope")

#: 14 points: dense where the pilot put K* for short horizons, sparse past the cap.
DEFAULT_K_GRID: tuple[int, ...] = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400)
DEFAULT_REPLICATES = 500
DEFAULT_OUT = ROOT / "results" / "growing_bandits" / "deploy" / "tables" / "k_star_envelope.csv"

FAMILY_OF_TYPE: dict[str, str] = {"beta": "A", "tail": "B", "mixture": "C"}

COLUMNS: tuple[str, ...] = (
    "level", "split", "env_id", "family", "horizon", "cap", "base_seed", "K",
    "n_envs", "n_episodes", "regret", "regret_se", "regret_disc", "regret_sel",
    "k_final", "search_frac", "k_star", "is_argmin",
)


@dataclass(frozen=True)
class Item:
    split: str
    spec: CellSpec
    K: int


def make_items(
    env_ids: tuple[str, ...], horizons: tuple[int, ...], k_grid: tuple[int, ...],
    n_replicates: int, split: str,
) -> list[Item]:
    items: list[Item] = []
    for env_id in env_ids:
        for T in horizons:
            spec = cells.make_cell(split, env_id, int(T), "T", int(n_replicates))
            for K in k_grid:
                if int(K) > int(T):
                    continue  # unreachable: the policy would just be always_search
                items.append(Item(split=split, spec=spec, K=int(K)))
    return items


_TABLES: dict[tuple[int, float], CSTable] = {}


def _table(horizon: int, alpha: float) -> CSTable:
    key = (int(horizon), float(alpha))
    if key not in _TABLES:
        _TABLES[key] = CSTable.load_or_build(int(horizon), float(alpha))
    return _TABLES[key]


def run_item(item: Item) -> dict:
    spec = item.spec
    table = _table(spec.horizon, spec.alpha)
    policy = make_policy(
        "fixed_K4", params={"K": item.K}, horizon=spec.horizon,
        n_replicates=spec.n_replicates, table=table,
    )
    res = run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
    regret = res.regret(PRIMARY_RECOMMENDER)
    return {
        "level": "env",
        "split": item.split,
        "env_id": spec.env_id,
        "family": FAMILY_OF_TYPE.get(str(spec.env_spec.get("type")), "?"),
        "horizon": int(spec.horizon),
        "cap": int(spec.cap),
        "base_seed": int(spec.base_seed),
        "K": int(item.K),
        "n_envs": 1,
        "n_episodes": int(spec.n_replicates),
        "regret": float(regret.mean()),
        "regret_se": float(regret.std(ddof=1) / np.sqrt(len(regret))) if len(regret) > 1 else float("nan"),
        "regret_disc": float(res.regret_disc.mean()),
        "regret_sel": float(res.regret_sel(PRIMARY_RECOMMENDER).mean()),
        "k_final": float(res.k_final.mean()),
        "search_frac": float(res.search_frac.mean()),
    }


def _worker_init(level: int) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _mark_argmin(frame: pd.DataFrame) -> pd.DataFrame:
    """`k_star` and `is_argmin` within every (level, env_id, horizon); ties go to the smaller K."""
    frame = frame.sort_values(["level", "env_id", "horizon", "K"]).reset_index(drop=True)
    frame["k_star"] = -1
    frame["is_argmin"] = False
    for _, idx in frame.groupby(["level", "env_id", "horizon"]).groups.items():
        sub = frame.loc[idx]
        best = sub.index[np.argmin(sub["regret"].to_numpy())]  # first minimum = smallest K
        frame.loc[idx, "k_star"] = int(frame.loc[best, "K"])
        frame.loc[best, "is_argmin"] = True
    return frame


def _pool_over_envs(env_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (T, K): the mean over environments. The pooled argmin is the *baseline* K."""
    rows: list[dict] = []
    for (T, K), sub in env_rows.groupby(["horizon", "K"]):
        rows.append({
            "level": "pooled",
            "split": sub["split"].iloc[0],
            "env_id": "all",
            "family": "all",
            "horizon": int(T),
            "cap": int(T),
            "base_seed": -1,
            "K": int(K),
            "n_envs": int(sub["env_id"].nunique()),
            "n_episodes": int(sub["n_episodes"].sum()),
            "regret": float(sub["regret"].mean()),
            # SE of a mean of independent cell means.
            "regret_se": float(np.sqrt((sub["regret_se"] ** 2).sum()) / len(sub)),
            "regret_disc": float(sub["regret_disc"].mean()),
            "regret_sel": float(sub["regret_sel"].mean()),
            "k_final": float(sub["k_final"].mean()),
            "search_frac": float(sub["search_frac"].mean()),
        })
    return pd.DataFrame(rows)


def run_envelope(
    env_ids: tuple[str, ...] = cells.MAIN_ENV_IDS,
    horizons: tuple[int, ...] = cells.HORIZONS,
    k_grid: tuple[int, ...] = DEFAULT_K_GRID,
    n_replicates: int = DEFAULT_REPLICATES,
    split: str = "tune",
    workers: int = 1,
) -> pd.DataFrame:
    items = make_items(tuple(env_ids), tuple(horizons), tuple(k_grid), n_replicates, split)
    log.info("%d items: %d envs x %d horizons x <=%d K, M=%d, split=%s",
             len(items), len(env_ids), len(horizons), len(k_grid), n_replicates, split)
    t0 = time.time()
    results: list[dict] = []
    if int(workers) <= 1:
        for i, item in enumerate(items, 1):
            results.append(run_item(item))
            log.info("[%d/%d] %s T=%d K=%d regret=%.6f", i, len(items), item.spec.env_id,
                     item.spec.horizon, item.K, results[-1]["regret"])
    else:
        ctx = mp.get_context("spawn")
        level = logging.getLogger().level or logging.INFO
        with ctx.Pool(int(workers), initializer=_worker_init, initargs=(level,)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_item, items, chunksize=1), 1):
                results.append(r)
                log.info("[%d/%d] %s T=%d K=%d regret=%.6f (elapsed %.0fs)", i, len(items),
                         r["env_id"], r["horizon"], r["K"], r["regret"], time.time() - t0)
    env_rows = pd.DataFrame(results)
    frame = pd.concat([env_rows, _pool_over_envs(env_rows)], ignore_index=True)
    frame = _mark_argmin(frame)
    log.info("done in %.0fs", time.time() - t0)
    return frame[list(COLUMNS)]


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", default=",".join(cells.MAIN_ENV_IDS),
                    help="comma-separated env ids (default: the 8 MAIN_ENV_IDS)")
    ap.add_argument("--horizons", default=",".join(str(T) for T in cells.HORIZONS))
    ap.add_argument("--k-grid", default=",".join(str(K) for K in DEFAULT_K_GRID))
    ap.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    ap.add_argument("--split", default="tune", choices=("tune", "validation"),
                    help="never 'test': the envelope is a design instrument")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frame = run_envelope(
        env_ids=tuple(args.envs.split(",")), horizons=_parse_ints(args.horizons),
        k_grid=_parse_ints(args.k_grid), n_replicates=args.replicates, split=args.split,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    log.info("wrote %s (%d rows)", args.out, len(frame))

    pooled = frame[(frame["level"] == "pooled") & frame["is_argmin"]]
    for _, r in pooled.iterrows():
        per_env = frame[(frame["level"] == "env") & (frame["horizon"] == r["horizon"]) & frame["is_argmin"]]
        log.info("T=%-5d pooled K*=%-4d regret=%.6f | per-env K* in [%d, %d]",
                 r["horizon"], r["K"], r["regret"], per_env["K"].min(), per_env["K"].max())
    return frame


if __name__ == "__main__":
    main()
