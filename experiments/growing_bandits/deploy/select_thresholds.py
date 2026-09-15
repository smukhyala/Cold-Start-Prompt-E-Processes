#!/usr/bin/env python
"""M5: select each learned policy's SEARCH threshold on validation seeds.

An artifact's `tau` is `tau_off`, the threshold that maximised out-of-fold balanced
accuracy on the corpus. That is a classification objective on off-policy states; the
deployment objective is regret on the policy's own trajectories, and the intercept of a
logistic model encodes the corpus class prior (failure-mode register #7). So every
variant is deployed at each `tau` in `TAUS` on the validation seeds of the main panel
(8 environments x 5 horizons), and `tau_val` is the `tau` with the lowest mean regret
pooled with equal weight over those 40 cells -- the same nested protocol the schedule
baselines get in `tune_baselines.py`. The whole `tau -> regret` curve is kept.

Horizon-holdout variants (`exclude_horizons` in the artifact's `meta.subset`) are still
selected on all five horizons, because that is how they are deployed; the threshold
chosen with their held-out horizon excluded is recorded alongside so Test D can
report the transfer at a threshold that never saw the transfer horizon.

One work item is `(variant, env, T)`: the worker loads the artifact once, computes the
cell's comparator prefix and oracle prior once, and runs every `tau` against them.
Pairwise log-e tables for every horizon in the run are built in the parent *before*
the pool is created (`pairwise_table.get_pairwise_table` saves through a fixed temp
name, so concurrent cache misses would race); workers then memory-map the cached file.
`ModelPolicy.counters()` are recorded per cell and a non-finite feature row aborts the
run: at deployment that is a parity bug, not a state to average over.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cells import (  # noqa: E402
    HORIZONS,
    assert_seed_disjointness,
    env_ids_for,
    make_cell,
)
from tune_baselines import (  # noqa: E402
    read_csv,
    summarize,
    write_csv_atomic,
    write_json_atomic,
)

from cold_start.growing.deploy.artifacts import load_model  # noqa: E402
from cold_start.growing.deploy.comparators import episode_reservoir_prefix  # noqa: E402
from cold_start.growing.deploy.feature_groups import EVIDENCE_LOGE  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.pairwise_table import get_pairwise_table  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.recommend import oracle_prior_from_reservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "results" / "growing_bandits" / "deploy"
DEFAULT_MODELS_DIR = DEFAULT_OUT_DIR / "models"

TAUS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_CAP = 64
SELECT_SPLIT = "val"
TAU_DECIMALS = 6

COUNTER_NAMES: tuple[str, ...] = (
    "n_decisions",
    "n_search_decided",
    "n_committed_steps",
    "n_guard_vetoes",
    "n_cap_demoted",
    "n_nonfinite_rows",
)


# ---- artifacts --------------------------------------------------------------------------


def list_variants(models_dir: Path) -> dict[str, Path]:
    """`{variant: path}` for every `<variant>.joblib` under `models_dir`, sorted by name."""
    paths = sorted(Path(models_dir).glob("*.joblib"))
    if not paths:
        raise FileNotFoundError(f"no model artifacts (*.joblib) under {models_dir}")
    return {p.stem: p for p in paths}


def uses_log_e(artifact: dict) -> bool:
    """Explicit column membership: does this model read the pairwise e-process feature?"""
    return any(col in EVIDENCE_LOGE for col in artifact["features"])


def heldout_horizons(artifact: dict) -> tuple[int, ...]:
    """Horizons the trainer excluded from this variant's rows (`meta.subset.exclude_horizons`)."""
    subset = dict(artifact.get("meta", {}).get("subset") or {})
    return tuple(int(h) for h in subset.get("exclude_horizons", ()))


# ---- work items ---------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdItem:
    """One `(variant, env, T)` cell and the thresholds still to run on it."""

    variant: str
    artifact_path: str
    split: str
    env_id: str
    horizon: int
    cap: int
    n_replicates: int
    taus: tuple[float, ...]

    @property
    def spec(self) -> CellSpec:
        return make_cell(self.split, self.env_id, self.horizon, self.cap, self.n_replicates)


def row_key(row) -> tuple:
    return (
        str(row["variant"]),
        str(row["split"]),
        str(row["env"]),
        int(row["T"]),
        int(row["cap"]),
        int(row["n_replicates"]),
        round(float(row["tau"]), TAU_DECIMALS),
    )


def done_keys(df: pd.DataFrame | None) -> set[tuple]:
    if df is None or len(df) == 0:
        return set()
    return {row_key(row) for _, row in df.iterrows()}


def build_items(
    variants: dict[str, Path],
    env_ids: Iterable[str],
    horizons: Iterable[int],
    taus: Iterable[float],
    cap: int,
    n_replicates: int,
    done: set[tuple] = frozenset(),
    split: str = SELECT_SPLIT,
) -> list[ThresholdItem]:
    """Items for every `(variant, env, T)` with a `tau` not yet in the CSV; longest T first."""
    env_ids, taus = list(env_ids), [float(t) for t in taus]
    items: list[ThresholdItem] = []
    for horizon in sorted(set(int(h) for h in horizons), reverse=True):
        for variant, path in variants.items():
            for env_id in env_ids:
                todo = tuple(
                    tau
                    for tau in taus
                    if (variant, split, env_id, horizon, int(cap), int(n_replicates),
                        round(tau, TAU_DECIMALS)) not in done
                )
                if todo:
                    items.append(
                        ThresholdItem(
                            variant, str(path), split, env_id, horizon, int(cap),
                            int(n_replicates), todo,
                        )
                    )
    return items


# ---- workers ----------------------------------------------------------------------------

_TABLES: dict[tuple[int, float], CSTable] = {}
_PAIRWISE: dict[int, object] = {}


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _table_for(horizon: int, alpha: float) -> CSTable:
    key = (int(horizon), float(alpha))
    if key not in _TABLES:
        _TABLES[key] = CSTable.load_or_build(int(horizon), alpha=float(alpha))
    return _TABLES[key]


def _pairwise_for(horizon: int):
    """Memory-mapped from the cache the parent built; kept per process, one table per T."""
    if horizon not in _PAIRWISE:
        _PAIRWISE[horizon] = get_pairwise_table(int(horizon))
    return _PAIRWISE[horizon]


def run_item(item: ThresholdItem) -> list[dict]:
    """Run the variant at every `tau` of the item on its cell; one row per `tau`."""
    spec = item.spec
    table = _table_for(spec.horizon, spec.alpha)
    artifact = load_model(item.artifact_path)
    pairwise = _pairwise_for(spec.horizon) if uses_log_e(artifact) else None
    reservoir = build_reservoir(spec.env_spec)
    prefix = episode_reservoir_prefix(reservoir, spec.base_seed, spec.n_replicates, spec.horizon)
    prior = oracle_prior_from_reservoir(reservoir)
    excluded = heldout_horizons(artifact)

    rows: list[dict] = []
    for tau in item.taus:
        policy = make_policy(
            "model",
            horizon=spec.horizon,
            n_replicates=spec.n_replicates,
            table=table,
            params={"artifact": artifact, "tau": float(tau)},
            pairwise=pairwise,
        )
        t0 = time.perf_counter()
        res = run_cell(
            spec,
            policy,
            table=table,
            reservoir=reservoir,
            dynamics_grid=0,
            comparator_prefix=prefix,
            oracle_prior=prior,
        )
        seconds = time.perf_counter() - t0
        counters = policy.counters()
        if counters["n_nonfinite_rows"] != 0:
            raise RuntimeError(
                f"{item.variant} tau={tau} env={item.env_id} T={spec.horizon}: "
                f"{counters['n_nonfinite_rows']} non-finite feature rows reached the model "
                "(feature parity bug; see model_policy.predict_proba)"
            )
        rows.append(
            {
                "variant": item.variant,
                "tau": float(tau),
                "env": item.env_id,
                "T": int(spec.horizon),
                "split": item.split,
                "cap": int(spec.cap),
                "base_seed": int(spec.base_seed),
                "n_replicates": int(spec.n_replicates),
                "k": int(artifact["k"]),
                "tau_off": float(artifact["tau"]) if artifact["tau"] is not None else float("nan"),
                "uses_log_e": bool(pairwise is not None),
                "heldout_T": bool(spec.horizon in excluded),
                "recommender": PRIMARY_RECOMMENDER,
                **summarize(res),
                **{name: int(counters[name]) for name in COUNTER_NAMES},
                "seconds": float(seconds),
            }
        )
    return rows


def run_items(items: list[ThresholdItem], workers: int) -> Iterator[list[dict]]:
    if workers <= 1:
        _worker_init()
        for item in items:
            yield run_item(item)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(workers), initializer=_worker_init) as pool:
        yield from pool.imap_unordered(run_item, items, chunksize=1)


# ---- selection ----------------------------------------------------------------------------


def _tau_key(tau: float) -> str:
    return f"{float(tau):g}"


def tau_curve(
    df: pd.DataFrame, env_ids: Iterable[str], horizons: Iterable[int], taus: Iterable[float]
) -> dict[str, float] | None:
    """``{tau: pooled regret}`` over `env_ids x horizons`; ``None`` unless every cell is present.

    Equal weight per cell. A `tau` missing a cell is not comparable to the others, and
    selecting among incomplete curves would favour whichever `tau` skipped the hardest
    cells, so an incomplete variant yields no curve at all.
    """
    env_ids, horizons = list(env_ids), [int(h) for h in horizons]
    n_cells = len(env_ids) * len(horizons)
    sub = df[df["env"].isin(env_ids) & df["T"].isin(horizons)]
    sub = sub.assign(tau_r=sub["tau"].round(TAU_DECIMALS))
    curve: dict[str, float] = {}
    for tau in taus:
        rows = sub[sub["tau_r"] == round(float(tau), TAU_DECIMALS)]
        if rows.groupby(["env", "T"]).ngroups != n_cells or len(rows) != n_cells:
            return None
        curve[_tau_key(tau)] = float(rows["mean_regret"].mean())
    return curve


def argmin_tau(curve: dict[str, float]) -> float:
    """Lowest pooled regret; ties go to the smaller `tau` (searches more)."""
    best = min(curve.items(), key=lambda kv: (kv[1], float(kv[0])))
    return float(best[0])


def build_thresholds(
    df: pd.DataFrame,
    artifacts: dict[str, dict],
    env_ids: Iterable[str],
    horizons: Iterable[int],
    taus: Iterable[float],
    split: str = SELECT_SPLIT,
    n_replicates: int | None = None,
) -> dict[str, dict]:
    """``{variant: {"tau_val", "tau_off", "curve", ...}}`` for every variant with a complete curve.

    `artifacts` maps variant -> loaded artifact (for `tau_off`, `k` and the holdout
    marker). Horizon-holdout variants also get `tau_val_excl_heldout` / `curve_excl_heldout`.
    """
    env_ids, horizons, taus = list(env_ids), [int(h) for h in horizons], [float(t) for t in taus]
    out: dict[str, dict] = {}
    if len(df) == 0:
        return out
    for variant, artifact in artifacts.items():
        sub = df[(df["variant"] == variant) & (df["split"] == split)]
        if n_replicates is not None:
            sub = sub[sub["n_replicates"] == int(n_replicates)]
        curve = tau_curve(sub, env_ids, horizons, taus)
        if curve is None:
            continue
        entry: dict = {
            "tau_val": argmin_tau(curve),
            "tau_off": None if artifact["tau"] is None else float(artifact["tau"]),
            "curve": curve,
            "k": int(artifact["k"]),
            "split": split,
            "n_replicates": int(sub["n_replicates"].iloc[0]),
            "envs": env_ids,
            "horizons": horizons,
            "n_cells": len(env_ids) * len(horizons),
            "recommender": PRIMARY_RECOMMENDER,
        }
        excluded = [h for h in heldout_horizons(artifact) if h in horizons]
        if excluded:
            kept = [h for h in horizons if h not in excluded]
            curve_excl = tau_curve(sub, env_ids, kept, taus)
            entry["heldout_horizons"] = excluded
            entry["curve_excl_heldout"] = curve_excl
            entry["tau_val_excl_heldout"] = None if curve_excl is None else argmin_tau(curve_excl)
        out[variant] = entry
    return out


# ---- CLI ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", type=str, default=str(DEFAULT_MODELS_DIR))
    ap.add_argument("--variants", type=str, default="",
                    help="comma list of variant names (default: all)")
    ap.add_argument("--split", default=SELECT_SPLIT, choices=["tune", "val", "test"])
    ap.add_argument("--n-replicates", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--envs", default="main",
                    help="main | heldout | corpus | all | comma list of ids")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS))
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--skip-loge", action="store_true",
                    help="skip variants that read f_log_e_pair (no pairwise tables needed)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args(argv)

    models_dir = Path(args.models)
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "threshold_selection.csv"
    json_path = out_dir / "thresholds.json"

    variants = list_variants(models_dir)
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        missing = [v for v in wanted if v not in variants]
        if missing:
            raise KeyError(f"unknown variants {missing}; available={sorted(variants)}")
        variants = {v: variants[v] for v in wanted}
    artifacts = {name: load_model(path) for name, path in variants.items()}
    if args.skip_loge:
        skipped = [name for name, art in artifacts.items() if uses_log_e(art)]
        variants = {name: path for name, path in variants.items() if name not in skipped}
        artifacts = {name: artifacts[name] for name in variants}
        print(f"--skip-loge: skipping {len(skipped)} variant(s): {skipped}")

    env_ids = env_ids_for(args.envs)
    horizons = [int(h) for h in args.horizons]
    taus = [float(t) for t in args.taus]
    seeds = [
        make_cell(args.split, env_id, horizon, args.cap, args.n_replicates).base_seed
        for env_id in env_ids
        for horizon in horizons
    ]
    assert_seed_disjointness(seeds)

    existing = read_csv(csv_path)
    items = build_items(variants, env_ids, horizons, taus, args.cap, args.n_replicates,
                        done_keys(existing), split=args.split)
    print(f"variants={len(variants)} envs={list(env_ids)} horizons={horizons} taus={taus} "
          f"M={args.n_replicates} cap={args.cap} workers={args.workers}")

    # Cell constants in the parent, before the pool exists: CS tables and -- when any
    # variant to run reads f_log_e_pair -- the pairwise tables of every horizon in the run.
    run_horizons = sorted({item.horizon for item in items})
    for horizon in run_horizons:
        CSTable.load_or_build(horizon, alpha=0.05)
    if any(uses_log_e(artifacts[item.variant]) for item in items):
        for horizon in run_horizons:
            t0 = time.time()
            get_pairwise_table(horizon)
            print(f"  pairwise table T={horizon} ready in {time.time() - t0:.1f}s")

    n_runs = sum(len(item.taus) for item in items)
    print(f"[select] {len(items)} items / {n_runs} runs to do, "
          f"{0 if existing is None else len(existing)} rows on disk")
    frames = [existing] if existing is not None else []
    df = existing if existing is not None else pd.DataFrame()
    start = time.time()
    done_runs = 0
    for i, rows in enumerate(run_items(items, args.workers), start=1):
        frames.append(pd.DataFrame(rows))
        done_runs += len(rows)
        df = pd.concat(frames, ignore_index=True)
        write_csv_atomic(df, csv_path)
        frames = [df]
        if i % 10 == 0 or i == len(items):
            el = time.time() - start
            eta = el / done_runs * (n_runs - done_runs) if done_runs else float("nan")
            print(f"[select]  {i}/{len(items)} items  {done_runs}/{n_runs} runs  "
                  f"{el:.0f}s elapsed  ETA {eta:.0f}s")

    thresholds = build_thresholds(
        df, artifacts, env_ids, horizons, taus, args.split, args.n_replicates
    )
    # Entries written by an earlier run under the same protocol (a `--variants` subset,
    # say) are kept; anything selected under a different split, M, panel or horizon set
    # is dropped rather than left for the deployment runner to read as current. An
    # incomplete variant is likewise dropped, never written from a subset of its cells.
    previous = json.loads(json_path.read_text()) if json_path.exists() else {}
    protocol = {"split": args.split, "n_replicates": int(args.n_replicates),
                "envs": list(env_ids), "horizons": horizons}
    merged = {
        name: entry for name, entry in previous.items()
        if all(entry.get(key) == value for key, value in protocol.items())
    }
    merged.update(thresholds)
    write_json_atomic(merged, json_path)
    for variant, entry in thresholds.items():
        extra = ""
        if "tau_val_excl_heldout" in entry:
            extra = f"  (excl. T={entry['heldout_horizons']}: tau={entry['tau_val_excl_heldout']})"
        print(f"  {variant:44s} tau_off={entry['tau_off']}  tau_val={entry['tau_val']}  "
              + " ".join(f"{k}:{v:.4f}" for k, v in entry["curve"].items()) + extra)
    incomplete = sorted(set(variants) - set(thresholds))
    if incomplete:
        print(f"  incomplete (no threshold written): {incomplete}")
    print(f"wrote {csv_path}\n      {json_path}")


if __name__ == "__main__":
    main()
