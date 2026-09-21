"""Select a registered rule's constants on the validation split (Pre-registrations 4 and 5).

One protocol for every "rule with a validation-selected block": the 8 main environments x
T in {50, 100, 200}, validation split, cap 64, M = 2000; one constant set for all horizons,
the argmin of environment-equal-weight pooled regret over the rule's registered grid.
Writes ``tables/<rule>_selection.csv`` (every candidate) and
``baseline_params.json[<block>] = {constants..., pooled_regret}`` plus ``meta.<block>``,
appended to the file in its existing key order (the audit anchor is a byte hash).

Registered rules (`RULES`): the grid IS the registration; adding one here is adding a
pre-registration, and DEPLOYMENT_PLAN.md must say so first.

Usage::

    .venv/bin/python experiments/growing_bandits/deploy/select_rule.py --rule bestmean_star --workers 12
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

log = logging.getLogger("deploy.select_rule")

SPLIT = "val"
CAP = rd.DEFAULT_CAP
N_REPLICATES = 2000
HORIZONS: tuple[int, ...] = (50, 100, 200)


@dataclass(frozen=True)
class Rule:
    """A registered rule: its `rules.py` kind, its `baseline_params.json` block, and its grid."""

    block: str
    kind: str
    param_names: tuple[str, ...]
    table: str
    grid: tuple[dict[str, float], ...]
    meta: dict

    def candidates(self) -> tuple[dict[str, float], ...]:
        return self.grid


def _grid(**axes: tuple[float, ...]) -> tuple[dict[str, float], ...]:
    names = list(axes)
    out: list[dict[str, float]] = []

    def rec(i: int, cur: dict[str, float]) -> None:
        if i == len(names):
            out.append(dict(cur))
            return
        for v in axes[names[i]]:
            cur[names[i]] = float(v)
            rec(i + 1, cur)
    rec(0, {})
    return tuple(out)


RULES: dict[str, Rule] = {
    # Pre-registration 4: K_target = c * T^alpha * (1 + b * (1 - q_t)).
    "adaptive_K_star": Rule(
        block="adaptive_K_star", kind="adaptive_K", param_names=("alpha", "c", "b"),
        table="adaptive_k_selection.csv",
        grid=_grid(alpha=(0.5, 0.75), c=(1.0, 2.0, 3.0, 4.0, 6.0), b=(0.0, 1.0, 2.0, 4.0, 8.0)),
        meta={"band": 0.05},
    ),
    # Pre-registration 5: SEARCH while best_mean < theta and K_t < c * T^alpha.
    "bestmean_star": Rule(
        block="bestmean_star", kind="bestmean_K", param_names=("theta", "alpha", "c"),
        table="bestmean_selection.csv",
        grid=tuple(
            {"theta": float(th), "alpha": float(a), "c": float(c)}
            for th in (0.55, 0.60, 0.625, 0.65, 0.675, 0.70)
            for (a, c) in ((0.5, 3.0), (0.5, 4.0), (0.5, 6.0), (0.5, 8.0),
                           (0.75, 1.0), (0.75, 1.5), (0.75, 2.0), (0.75, 3.0))
        ),
        meta={},
    ),
}


@dataclass(frozen=True)
class Item:
    spec: CellSpec
    kind: str
    params: tuple[tuple[str, float], ...]


def run_item(item: Item) -> dict:
    spec = item.spec
    params = dict(item.params)
    table = ks._table(spec.horizon, spec.alpha)
    policy = make_policy(item.kind, params=params, horizon=spec.horizon,
                         n_replicates=spec.n_replicates, table=table)
    res = run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
    regret = res.regret(PRIMARY_RECOMMENDER)
    return {
        "level": "env", "split": SPLIT, "env_id": spec.env_id, "horizon": int(spec.horizon),
        "cap": int(spec.cap), "base_seed": int(spec.base_seed), **params, "n_envs": 1,
        "n_episodes": int(spec.n_replicates), "regret": float(regret.mean()),
        "regret_se": float(regret.std(ddof=1) / np.sqrt(len(regret))),
        "k_final": float(res.k_final.mean()), "search_frac": float(res.search_frac.mean()),
    }


def select(
    rule_name: str,
    *,
    env_ids: tuple[str, ...] = cells.MAIN_ENV_IDS,
    horizons: tuple[int, ...] = HORIZONS,
    candidates: tuple[dict[str, float], ...] | None = None,
    n_replicates: int = N_REPLICATES,
    out_dir: Path = rd.DEFAULT_OUT_DIR,
    workers: int = 1,
) -> pd.DataFrame:
    rule = RULES[rule_name]
    out_dir = Path(out_dir)
    candidates = tuple(candidates) if candidates is not None else rule.candidates()
    names = list(rule.param_names)
    items = [
        Item(cells.make_cell(SPLIT, env_id, int(T), CAP, int(n_replicates)), rule.kind,
             tuple((k, float(cand[k])) for k in names))
        for env_id in env_ids for T in horizons for cand in candidates
    ]
    log.info("%s: %d items (%d envs x %d horizons x %d candidates), M=%d, split=%s, cap=%d",
             rule_name, len(items), len(env_ids), len(horizons), len(candidates), n_replicates, SPLIT, CAP)
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
    for key, sub in env_rows.groupby(names):
        key = key if isinstance(key, tuple) else (key,)
        pooled_rows.append({
            "level": "pooled", "split": SPLIT, "env_id": "all", "horizon": "all", "cap": CAP, "base_seed": -1,
            **dict(zip(names, key, strict=True)), "n_envs": int(sub["env_id"].nunique()),
            "n_episodes": int(sub["n_episodes"].sum()), "regret": float(sub["regret"].mean()),
            "regret_se": float(np.sqrt((sub["regret_se"] ** 2).sum()) / len(sub)),
            "k_final": float(sub["k_final"].mean()), "search_frac": float(sub["search_frac"].mean()),
        })
    pooled = pd.DataFrame(pooled_rows).sort_values(names).reset_index(drop=True)
    best = int(np.argmin(pooled["regret"].to_numpy()))
    pooled["is_argmin"] = False
    pooled.loc[best, "is_argmin"] = True
    env_rows["is_argmin"] = False
    frame = pd.concat([env_rows, pooled], ignore_index=True)
    win = pooled.loc[best]
    log.info("%s selected %s: pooled validation regret %.6f over %d cells (%.0fs)", rule_name,
             {k: float(win[k]) for k in names}, win["regret"], len(env_ids) * len(horizons), time.time() - t0)

    path = out_dir / "baseline_params.json"
    params = json.load(open(path)) if path.exists() else {}
    params[rule.block] = {**{k: float(win[k]) for k in names}, "pooled_regret": float(win["regret"])}
    params.setdefault("meta", {})[rule.block] = {
        "select_split": SPLIT, "cap": CAP, "n_replicates": int(n_replicates),
        "horizons": [int(T) for T in horizons], "envs": list(env_ids),
        "grid": [dict(c) for c in candidates], "n_candidates": len(candidates),
        "pooling": "equal weight over envs of the cell mean regret", **rule.meta,
    }
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(params, fh, indent=2)
        fh.write("\n")
    tmp.replace(path)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / rule.table, index=False)
    return frame


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", required=True, choices=sorted(RULES))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return select(args.rule, n_replicates=args.replicates, out_dir=args.out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
