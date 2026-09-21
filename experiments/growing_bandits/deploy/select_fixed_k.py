"""Select `fixed_K_star`'s K per horizon on the validation split (Pre-registration 3).

The null model H1b' must beat: recruit to K(T) arms immediately, then refine only. K(T)
is the argmin of the environment-equal-weight pooled regret over `K_GRID` (clipped to
[3, min(T, cap)]) on the 8 main environments, validation split, cap 64, M = 2000 -- the
protocol `p3_star` was selected under (`tune_baselines.py`), with at most 17 candidates
per horizon. Writes ``tables/fixed_k_selection.csv`` (every candidate) and the block
``baseline_params.json["fixed_K_star"][str(T)] = {"K", "pooled_regret"}`` plus
``meta.fixed_K_star`` recording the grid and split, touching nothing else in the file.

This is selection on deployed validation regret, so the winner carries the same
winner's curse as P3*; the plan states the bound. The selected K is never quoted from
here as a result -- the test-split deployment is.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cells  # noqa: E402
import k_star_envelope as ks  # noqa: E402
import policy_table as pt  # noqa: E402
import run_deployment as rd  # noqa: E402

log = logging.getLogger("deploy.select_fixed_k")

SPLIT = "val"
CAP = rd.DEFAULT_CAP
N_REPLICATES = 2000
K_GRID: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64)
BLOCK = "fixed_K_star"


def candidates(horizon: int, k_grid: tuple[int, ...] = K_GRID, cap: int = CAP) -> tuple[int, ...]:
    """`k_grid` clipped to what the harness can reach at `horizon` under `cap`."""
    hi = min(int(horizon), int(cap))
    return tuple(k for k in k_grid if 3 <= k <= hi)


def select(
    *,
    env_ids: tuple[str, ...] = cells.MAIN_ENV_IDS,
    horizons: tuple[int, ...] = cells.HORIZONS,
    k_grid: tuple[int, ...] = K_GRID,
    n_replicates: int = N_REPLICATES,
    out_dir: Path = rd.DEFAULT_OUT_DIR,
    workers: int = 1,
    cap: int = CAP,
) -> pd.DataFrame:
    """Select K per horizon at `cap`. At the file's tuning cap the block is written at the
    top level; at any other cap under ``by_cap["<cap>"]`` (`policy_table.merge_cap_block`)."""
    out_dir = Path(out_dir)
    cap = int(cap)
    items = []
    for env_id in env_ids:
        for T in horizons:
            spec = cells.make_cell(SPLIT, env_id, int(T), cap, int(n_replicates))
            for K in candidates(int(T), k_grid, cap):
                items.append(ks.Item(split=SPLIT, spec=spec, K=int(K)))
    log.info("%d candidates: %d envs x %d horizons, <=%d K each, M=%d, split=%s, cap=%d",
             len(items), len(env_ids), len(horizons), len(k_grid), n_replicates, SPLIT, cap)
    frame = ks.run_items(items, workers=workers)

    path = out_dir / "baseline_params.json"
    params = json.load(open(path)) if path.exists() else {"meta": {"cap": cap}}
    block = {}
    pooled = frame[(frame["level"] == "pooled") & frame["is_argmin"]]
    for _, r in pooled.sort_values("horizon").iterrows():
        block[str(int(r["horizon"]))] = {"K": int(r["K"]), "pooled_regret": float(r["regret"])}
        log.info("cap=%d T=%-5d K*=%-3d pooled validation regret %.6f", cap, r["horizon"], r["K"], r["regret"])
    existing_block = pt.baseline_params_for_cap(params, cap)[0].get(BLOCK, {})
    tuned_cap = pt.tuning_cap(params)
    if tuned_cap is not None and int(tuned_cap) != cap:
        existing_block = ((params.get(pt.BY_CAP_KEY) or {}).get(str(cap)) or {}).get(BLOCK, {})
    meta = {
        "select_split": SPLIT, "cap": cap, "n_replicates": int(n_replicates),
        "k_grid": [int(k) for k in k_grid], "envs": list(env_ids), "horizons": [int(T) for T in horizons],
        "pooling": "equal weight over envs of the cell mean regret",
    }
    params = pt.merge_cap_block(params, cap, {BLOCK: {**existing_block, **block}, "meta": {BLOCK: meta}})
    pt.write_baseline_params(path, params)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table = tables / ("fixed_k_selection.csv" if cap == CAP else f"fixed_k_selection_cap{cap}.csv")
    frame.to_csv(table, index=False)
    log.info("wrote %s and %s[%s] at cap %d", table, path.name, BLOCK, cap)
    return frame


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--horizons", default=",".join(str(T) for T in cells.HORIZONS))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    return select(horizons=horizons, n_replicates=args.replicates, out_dir=args.out_dir,
                  workers=args.workers, cap=args.cap)


if __name__ == "__main__":
    main()
