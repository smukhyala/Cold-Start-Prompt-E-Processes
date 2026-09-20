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

The hand rule `reservoir_rule` (plan P11, `rules.ReservoirRule`) gets the same
selection on its own grid `RULE_TAUS`: its `tau` scales the leader's CS width rather
than a probability, so the model grid would be meaningless for it. It is written under
the key ``"reservoir_rule"`` with the same layout, minus `tau_off` (it has no offline
threshold) and `k`.

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

Every row carries a fingerprint of the artifact it was produced with (sha256 of the
joblib bytes; the rule hashes its own source), and the resume key includes it: a
retrained model under the same name is new work, and its predecessor's rows are dropped
rather than folded into the new threshold. The test split is refused unless
`--allow-test-split` is given, and then the outputs are suffixed `_TESTSPLIT`.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
    check_splits,
    output_path,
    read_csv,
    summarize,
    write_csv_atomic,
    write_json_atomic,
)

from cold_start.growing.deploy import rules  # noqa: E402
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

#: P(SEARCH) thresholds for the learned models.
TAUS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
#: Width multipliers for the hand rule (`ReservoirRule`): a different scale entirely.
RULE_TAUS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
#: The hand rule's name in `variants`, the CSV and the JSON.
RULE_VARIANT = "reservoir_rule"
DEFAULT_CAP = 64
SELECT_SPLIT = "val"
TAU_DECIMALS = 6
FINGERPRINT_CHARS = 16

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


def is_rule(path: Path | None) -> bool:
    """The hand rule has no artifact: it is the entry whose path is ``None``."""
    return path is None


def artifact_fingerprint(path: Path | None) -> str:
    """Identity of what a row was produced with: the joblib bytes, or the rule's source.

    A retrained artifact keeps its name, so the name alone cannot tell a resumed run
    that its earlier rows belong to a different model. For the rule the analogue is its
    code: a changed `ReservoirRule` is a different policy.
    """
    if is_rule(path):
        src = inspect.getsource(rules.ReservoirRule).encode()
        return "rule:" + hashlib.sha256(src).hexdigest()[:FINGERPRINT_CHARS]
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:FINGERPRINT_CHARS]


def uses_log_e(artifact: dict | None) -> bool:
    """Explicit column membership: does this model read the pairwise e-process feature?"""
    if artifact is None:
        return False
    return any(col in EVIDENCE_LOGE for col in artifact["features"])


def heldout_horizons(artifact: dict | None) -> tuple[int, ...]:
    """Horizons the trainer excluded from this variant's rows (`meta.subset.exclude_horizons`)."""
    if artifact is None:
        return ()
    subset = dict(artifact.get("meta", {}).get("subset") or {})
    return tuple(int(h) for h in subset.get("exclude_horizons", ()))


def tau_grids(
    variants: Iterable[str], taus: Sequence[float] = TAUS, rule_taus: Sequence[float] = RULE_TAUS
) -> dict[str, tuple[float, ...]]:
    """Each variant's threshold grid: the model grid, or the rule's own."""
    return {
        name: tuple(float(t) for t in (rule_taus if name == RULE_VARIANT else taus))
        for name in variants
    }


# ---- work items ---------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdItem:
    """One `(variant, env, T)` cell and the thresholds still to run on it."""

    variant: str
    artifact_path: str | None
    fingerprint: str
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
        str(row["fingerprint"]),
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
    variants: Mapping[str, Path | None],
    env_ids: Iterable[str],
    horizons: Iterable[int],
    taus: Mapping[str, Sequence[float]],
    cap: int,
    n_replicates: int,
    done: set[tuple] = frozenset(),
    split: str = SELECT_SPLIT,
    fingerprints: Mapping[str, str] | None = None,
) -> list[ThresholdItem]:
    """Items for every `(variant, env, T)` with a `tau` not yet in the CSV; longest T first.

    `taus` maps each variant to its grid (`tau_grids`). Rows count as done only when
    their fingerprint matches the artifact on disk now (`fingerprints`, computed here
    when not given).
    """
    env_ids = list(env_ids)
    if fingerprints is None:
        fingerprints = {name: artifact_fingerprint(path) for name, path in variants.items()}
    items: list[ThresholdItem] = []
    for horizon in sorted(set(int(h) for h in horizons), reverse=True):
        for variant, path in variants.items():
            fp = fingerprints[variant]
            grid = [float(t) for t in taus[variant]]
            for env_id in env_ids:
                todo = tuple(
                    tau
                    for tau in grid
                    if (variant, fp, split, env_id, horizon, int(cap), int(n_replicates),
                        round(tau, TAU_DECIMALS)) not in done
                )
                if todo:
                    items.append(
                        ThresholdItem(
                            variant, None if path is None else str(path), fp, split, env_id,
                            horizon, int(cap), int(n_replicates), todo,
                        )
                    )
    return items


def drop_stale_rows(
    df: pd.DataFrame | None, fingerprints: Mapping[str, str]
) -> tuple[pd.DataFrame | None, int]:
    """Remove rows of this run's variants whose fingerprint is not the current one.

    Rows of variants outside `fingerprints` (a `--variants` subset run) are left alone.
    """
    if df is None or len(df) == 0:
        return df, 0
    if "fingerprint" not in df.columns:
        return None, len(df)  # written before fingerprints existed: nothing is verifiable
    stale = [
        str(row["variant"]) in fingerprints
        and str(row["fingerprint"]) != fingerprints[str(row["variant"])]
        for _, row in df.iterrows()
    ]
    n_stale = int(sum(stale))
    if n_stale == 0:
        return df, 0
    kept = df[[not s for s in stale]].reset_index(drop=True)
    return (kept if len(kept) else None), n_stale


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


def _load_checked(item: ThresholdItem) -> dict | None:
    """The item's artifact, verified to be the file the item was planned against."""
    path = None if item.artifact_path is None else Path(item.artifact_path)
    fp = artifact_fingerprint(path)
    if fp != item.fingerprint:
        raise RuntimeError(
            f"{item.variant}: artifact changed under the run (planned {item.fingerprint}, "
            f"on disk {fp}); restart so the stale rows are dropped"
        )
    return None if path is None else load_model(path)


def _build(item: ThresholdItem, artifact: dict | None, tau: float, table, pairwise):
    if artifact is None:
        return make_policy(
            RULE_VARIANT,
            horizon=item.horizon,
            n_replicates=item.n_replicates,
            table=table,
            params={"tau": float(tau)},
        )
    return make_policy(
        "model",
        horizon=item.horizon,
        n_replicates=item.n_replicates,
        table=table,
        params={"artifact": artifact, "tau": float(tau)},
        pairwise=pairwise,
    )


def run_item(item: ThresholdItem) -> list[dict]:
    """Run the variant at every `tau` of the item on its cell; one row per `tau`."""
    spec = item.spec
    table = _table_for(spec.horizon, spec.alpha)
    artifact = _load_checked(item)
    pairwise = _pairwise_for(spec.horizon) if uses_log_e(artifact) else None
    reservoir = build_reservoir(spec.env_spec)
    prefix = episode_reservoir_prefix(reservoir, spec.base_seed, spec.n_replicates, spec.horizon)
    prior = oracle_prior_from_reservoir(reservoir)
    excluded = heldout_horizons(artifact)

    rows: list[dict] = []
    for tau in item.taus:
        policy = _build(item, artifact, tau, table, pairwise)
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
        # The rule keeps no counters; -1 marks "not applicable", as the harness does
        # for `n_demoted` of a policy without `last_decision`.
        counters = policy.counters() if hasattr(policy, "counters") else {}
        if counters.get("n_nonfinite_rows", 0) != 0:
            raise RuntimeError(
                f"{item.variant} tau={tau} env={item.env_id} T={spec.horizon}: "
                f"{counters['n_nonfinite_rows']} non-finite feature rows reached the model "
                "(feature parity bug; see model_policy.predict_proba)"
            )
        if artifact is None:
            k, tau_off = -1, float("nan")
        else:
            k = int(artifact["k"])
            tau_off = float("nan") if artifact["tau"] is None else float(artifact["tau"])
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
                "fingerprint": item.fingerprint,
                "k": k,
                "tau_off": tau_off,
                "uses_log_e": bool(pairwise is not None),
                "heldout_T": bool(spec.horizon in excluded),
                "recommender": PRIMARY_RECOMMENDER,
                **summarize(res),
                **{name: int(counters.get(name, -1)) for name in COUNTER_NAMES},
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
    artifacts: Mapping[str, dict | None],
    env_ids: Iterable[str],
    horizons: Iterable[int],
    taus: Mapping[str, Sequence[float]],
    split: str = SELECT_SPLIT,
    n_replicates: int | None = None,
    fingerprints: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    """``{variant: {"tau_val", "curve", ...}}`` for every variant with a complete curve.

    `artifacts` maps variant -> loaded artifact (``None`` for the rule); `taus` maps
    variant -> its grid. With `fingerprints`, only rows produced by the artifact on disk
    now are used. Models also get `tau_off` and `k`; horizon-holdout variants also get
    `tau_val_excl_heldout` / `curve_excl_heldout`.
    """
    env_ids, horizons = list(env_ids), [int(h) for h in horizons]
    out: dict[str, dict] = {}
    if len(df) == 0:
        return out
    # Two concurrent runs could both append the same row; the last write wins.
    df = df.assign(_key=[row_key(row) for _, row in df.iterrows()])
    df = df.drop_duplicates("_key", keep="last").drop(columns="_key")
    for variant, artifact in artifacts.items():
        sub = df[(df["variant"] == variant) & (df["split"] == split)]
        if n_replicates is not None:
            sub = sub[sub["n_replicates"] == int(n_replicates)]
        if fingerprints is not None:
            sub = sub[sub["fingerprint"] == fingerprints[variant]]
        grid = [float(t) for t in taus[variant]]
        curve = tau_curve(sub, env_ids, horizons, grid)
        if curve is None:
            continue
        entry: dict = {
            "tau_val": argmin_tau(curve),
            "curve": curve,
            "kind": "rule" if artifact is None else "model",
            "taus": grid,
            "split": split,
            "n_replicates": int(sub["n_replicates"].iloc[0]),
            "envs": env_ids,
            "horizons": horizons,
            "n_cells": len(env_ids) * len(horizons),
            "recommender": PRIMARY_RECOMMENDER,
            "fingerprint": str(sub["fingerprint"].iloc[0]),
        }
        if artifact is not None:
            entry["tau_off"] = None if artifact["tau"] is None else float(artifact["tau"])
            entry["k"] = int(artifact["k"])
        excluded = [h for h in heldout_horizons(artifact) if h in horizons]
        if excluded:
            kept = [h for h in horizons if h not in excluded]
            curve_excl = tau_curve(sub, env_ids, kept, grid)
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
                    help=f"comma list of variant names, '{RULE_VARIANT}' included "
                         "(default: every artifact plus the rule)")
    ap.add_argument("--split", default=SELECT_SPLIT, choices=["tune", "val", "test"])
    ap.add_argument("--n-replicates", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--envs", default="main",
                    help="main | heldout | corpus | all | comma list of ids")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS),
                    help="P(SEARCH) grid for the learned models")
    ap.add_argument("--rule-taus", type=float, nargs="+", default=list(RULE_TAUS),
                    help=f"width-multiplier grid for {RULE_VARIANT}")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--skip-loge", action="store_true",
                    help="skip variants that read f_log_e_pair (no pairwise tables needed)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--allow-test-split", action="store_true",
                    help="permit --split test (outputs suffixed _TESTSPLIT)")
    args = ap.parse_args(argv)
    suffix = check_splits((args.split,), args.allow_test_split)

    models_dir = Path(args.models)
    if not models_dir.is_absolute():
        models_dir = ROOT / models_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_path(out_dir, "threshold_selection", ".csv", suffix)
    json_path = output_path(out_dir, "thresholds", ".json", suffix)

    variants: dict[str, Path | None] = dict(list_variants(models_dir))
    if RULE_VARIANT in variants:
        raise ValueError(f"{RULE_VARIANT!r} is the hand rule's name; rename that artifact")
    variants[RULE_VARIANT] = None
    # Every selectable variant's current identity, for validating kept JSON entries below.
    current = {name: artifact_fingerprint(path) for name, path in variants.items()}
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        missing = [v for v in wanted if v not in variants]
        if missing:
            raise KeyError(f"unknown variants {missing}; available={sorted(variants)}")
        variants = {v: variants[v] for v in wanted}
    artifacts = {name: (None if path is None else load_model(path))
                 for name, path in variants.items()}
    if args.skip_loge:
        skipped = [name for name, art in artifacts.items() if uses_log_e(art)]
        variants = {name: path for name, path in variants.items() if name not in skipped}
        artifacts = {name: artifacts[name] for name in variants}
        print(f"--skip-loge: skipping {len(skipped)} variant(s): {skipped}")
    fingerprints = {name: artifact_fingerprint(path) for name, path in variants.items()}
    grids = tau_grids(variants, args.taus, args.rule_taus)

    env_ids = env_ids_for(args.envs)
    horizons = [int(h) for h in args.horizons]
    seeds = [
        make_cell(args.split, env_id, horizon, args.cap, args.n_replicates).base_seed
        for env_id in env_ids
        for horizon in horizons
    ]
    assert_seed_disjointness(seeds)

    existing, n_stale = drop_stale_rows(read_csv(csv_path), fingerprints)
    if n_stale:
        print(f"dropped {n_stale} row(s) produced by artifacts that have since changed")
    items = build_items(variants, env_ids, horizons, grids, args.cap, args.n_replicates,
                        done_keys(existing), split=args.split, fingerprints=fingerprints)
    print(f"variants={len(variants)} envs={list(env_ids)} horizons={horizons} "
          f"taus={list(args.taus)} rule_taus={list(args.rule_taus)} "
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
    if n_stale and not items:
        # Nothing new to run, but the stale rows must not survive on disk either.
        if len(df):
            write_csv_atomic(df, csv_path)
        else:
            csv_path.unlink(missing_ok=True)
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
        df, artifacts, env_ids, horizons, grids, args.split, args.n_replicates, fingerprints
    )
    # Entries written by an earlier run under the same protocol (a `--variants` subset,
    # say) are kept only while the artifact they were selected for is still the one on
    # disk; anything selected under a different split, M, panel or horizon set, or for a
    # model since retrained, is dropped rather than left for the deployment runner to
    # read as current. An incomplete variant is likewise never written from a subset of
    # its cells.
    previous = json.loads(json_path.read_text()) if json_path.exists() else {}
    protocol = {"split": args.split, "n_replicates": int(args.n_replicates),
                "envs": list(env_ids), "horizons": horizons}
    merged = {
        name: entry for name, entry in previous.items()
        if all(entry.get(key) == value for key, value in protocol.items())
        and entry.get("fingerprint") == current.get(name)
    }
    merged.update(thresholds)
    write_json_atomic(merged, json_path)
    for variant, entry in thresholds.items():
        extra = ""
        if "tau_val_excl_heldout" in entry:
            extra = f"  (excl. T={entry['heldout_horizons']}: tau={entry['tau_val_excl_heldout']})"
        print(f"  {variant:44s} tau_off={entry.get('tau_off')}  tau_val={entry['tau_val']}  "
              + " ".join(f"{k}:{v:.4f}" for k, v in entry["curve"].items()) + extra)
    incomplete = sorted(set(variants) - set(thresholds))
    if incomplete:
        print(f"  incomplete (no threshold written): {incomplete}")
    print(f"wrote {csv_path}\n      {json_path}")


if __name__ == "__main__":
    main()
