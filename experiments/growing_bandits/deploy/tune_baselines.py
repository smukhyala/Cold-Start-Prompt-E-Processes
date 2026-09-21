#!/usr/bin/env python
"""M5: tune the schedule baselines' constants on tuning seeds; select P3* on validation seeds.

The learned policies get their threshold from validation episodes; a comparison against
a power schedule whose constant was set by hand (or on the test seeds) would not be a
fair one (failure-mode register #7). So the baselines get the same nested protocol:

1. **Tuning seeds.** For every alpha in `ALPHAS` and horizon `T`, run `K_t < c t^alpha`
   for `c` on a 40-point log grid over the main environments, plus `refine_after_init`
   for `K0` in `K0_GRID`; write every cell to `schedule_tuning.csv`. The constant per
   `(alpha, T)` is the `c` with the lowest mean regret pooled with equal weight over
   the environments -- family hidden, exactly as the learned models' threshold is chosen.
2. **Validation seeds.** Re-run each alpha at its tuned `c` (and the chosen `K0`) on
   fresh episodes; `P3*` per `T` is the alpha with the lowest pooled regret there
   (`schedule_selection.csv`). Selecting the winner on the seeds it was tuned on would
   bias its regret downward by the maximum of 40 noisy minima.
3. **Oracle-tuned schedule.** The best `c` per `(env, T, alpha)` on the tuning seeds is
   recorded in `schedule_oracle_tuned.csv` as a labelled, in-sample upper bound on what
   any single schedule could do; it is never used for selection.

Every cell shares its comparator prefix and oracle prior across the configurations it
hosts (`harness.run_cell` takes both), because for a baseline the prefix costs more
than the rollout. Work is parallelized over `(env, T, chunk of configs)` items with a
spawn-context `multiprocessing.Pool`; each worker memory-maps its own CS tables. Rows
already in the CSV are skipped, so an interrupted run resumes where it stopped.

The test split is refused (`TestSplitRefused`) unless `--allow-test-split` is given, and
then every output file is suffixed `_TESTSPLIT`, so a constant chosen on test episodes
can never be the one the deployment runner reads by default.
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

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import policy_table as pt  # noqa: E402
from cells import (  # noqa: E402
    HORIZONS,
    assert_seed_disjointness,
    cell_at_cap,
    env_ids_for,
)

from cold_start.growing.deploy.comparators import episode_reservoir_prefix  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec, EpisodeResult, run_cell  # noqa: E402
from cold_start.growing.deploy.recommenders import (  # noqa: E402
    PRIMARY_RECOMMENDER,
    RECOMMENDER_NAMES,
)
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.recommend import oracle_prior_from_reservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "results" / "growing_bandits" / "deploy"

ALPHAS: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0)
K0_GRID: tuple[int, ...] = (2, 4, 8)
C_MIN, C_MAX, N_C = 0.25, 16.0, 40
DEFAULT_CAP = 64
TUNE_SPLIT, SELECT_SPLIT = "tune", "val"

POWER, REFINE = "power", "refine_after_init"

#: Appended to every output name when a run was allowed to touch the test split.
TESTSPLIT_SUFFIX = "_TESTSPLIT"


class TestSplitRefused(ValueError):
    """Raised when a tuning/selection run names the test split without opting in."""


def check_splits(splits: Iterable[str], allow_test_split: bool) -> str:
    """Refuse the test split unless opted in; return the output-name suffix to use.

    Selecting a schedule constant or a threshold on the test episodes and then reporting
    those episodes would be the asymmetric tuning the register (#7) forbids. The opt-in
    exists for diagnostics only, and its outputs carry `TESTSPLIT_SUFFIX` so
    `run_deployment.py` (which reads the unsuffixed names) can never pick them up.
    """
    uses_test = any(split == "test" for split in splits)
    if uses_test and not allow_test_split:
        raise TestSplitRefused(
            "the test split is reserved for deployment runs; tuning or selecting on it "
            "requires --allow-test-split, and the outputs are then suffixed "
            f"{TESTSPLIT_SUFFIX!r}"
        )
    return TESTSPLIT_SUFFIX if uses_test else ""


def output_path(out_dir: Path, stem: str, ext: str, suffix: str) -> Path:
    return Path(out_dir) / f"{stem}{suffix}{ext}"

# Rounding used to key rows for resume and selection; the grid values are irrational
# (geomspace) and alpha = 1/3, so exact float equality after a CSV round trip is not
# something to rely on.
ALPHA_DECIMALS, C_DECIMALS = 6, 8


def c_grid(c_min: float = C_MIN, c_max: float = C_MAX, n: int = N_C) -> np.ndarray:
    # Rounded so the grid reads as intended (2.0, not 1.9999999999999998) in the JSON.
    return np.round(np.geomspace(float(c_min), float(c_max), int(n)), 10)


# ---- configurations ----------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleConfig:
    """One baseline: ``power`` (`K_t < c t^alpha`) or ``refine_after_init`` (`K0` arms)."""

    policy: str
    alpha: float
    c: float
    K0: int | None = None

    def key(self) -> tuple[str, float, float]:
        return (
            self.policy,
            round(float(self.alpha), ALPHA_DECIMALS),
            round(float(self.c), C_DECIMALS),
        )

    def name(self) -> str:
        if self.policy == REFINE:
            return f"{REFINE}_K{self.K0}"
        return f"{POWER}_a{self.alpha:.4f}_c{self.c:.6g}"

    def build(self, horizon: int, n_replicates: int, table):
        if self.policy == REFINE:
            return make_policy(
                "refine_after_init_K2",
                horizon=horizon,
                n_replicates=n_replicates,
                table=table,
                params={"K0": int(self.K0)},
            )
        return make_policy(
            "power_sqrt",
            horizon=horizon,
            n_replicates=n_replicates,
            table=table,
            params={"alpha": float(self.alpha), "c": float(self.c)},
        )


def power_config(alpha: float, c: float) -> ScheduleConfig:
    return ScheduleConfig(POWER, float(alpha), float(c), None)


def refine_config(K0: int) -> ScheduleConfig:
    """`PowerSchedule(alpha=0, c=K0)`: the `refine_after_init` kind, kept on the same axes."""
    return ScheduleConfig(REFINE, 0.0, float(K0), int(K0))


def tuning_grid(
    alphas: Iterable[float] = ALPHAS,
    cs: Iterable[float] = (),
    k0s: Iterable[int] = K0_GRID,
) -> list[ScheduleConfig]:
    cs = list(cs)
    if not cs:
        cs = list(c_grid())
    configs = [power_config(a, c) for a in alphas for c in cs]
    configs += [refine_config(k0) for k0 in k0s]
    return configs


# ---- work items ---------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One `(split, env, T, cap)` cell and the configurations to run on it."""

    split: str
    env_id: str
    horizon: int
    cap: int
    n_replicates: int
    configs: tuple[ScheduleConfig, ...]

    @property
    def spec(self) -> CellSpec:
        return cell_at_cap(self.split, self.env_id, self.horizon, self.cap, self.n_replicates)


def row_key(row) -> tuple:
    """Identity of one CSV row: cell x configuration (M included: a new M is new work)."""
    return (
        str(row["split"]),
        str(row["env"]),
        int(row["T"]),
        int(row["cap"]),
        int(row["n_replicates"]),
        str(row["policy"]),
        round(float(row["alpha"]), ALPHA_DECIMALS),
        round(float(row["c"]), C_DECIMALS),
    )


def done_keys(df: pd.DataFrame | None) -> set[tuple]:
    if df is None or len(df) == 0:
        return set()
    return {row_key(row) for _, row in df.iterrows()}


def build_items(
    split: str,
    env_ids: Iterable[str],
    horizons: Iterable[int],
    cap: int,
    n_replicates: int,
    configs: Iterable[ScheduleConfig],
    done: set[tuple] = frozenset(),
    chunk: int = 16,
) -> list[WorkItem]:
    """Items for every configuration not yet in the CSV, longest horizons first."""
    configs = list(configs)
    items: list[WorkItem] = []
    for horizon in sorted(set(int(h) for h in horizons), reverse=True):
        for env_id in env_ids:
            todo = [
                cfg
                for cfg in configs
                if (split, env_id, horizon, int(cap), int(n_replicates), *cfg.key()) not in done
            ]
            for start in range(0, len(todo), max(int(chunk), 1)):
                items.append(
                    WorkItem(
                        split,
                        env_id,
                        horizon,
                        int(cap),
                        int(n_replicates),
                        tuple(todo[start : start + chunk]),
                    )
                )
    return items


# ---- workers --------------------------------------------------------------------------

_TABLES: dict[tuple[int, float], CSTable] = {}


def _worker_init() -> None:
    # One thread per worker: the rollouts are many small numpy ops, and BLAS threads
    # would only fight the process pool for the same cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _table_for(horizon: int, alpha: float) -> CSTable:
    key = (int(horizon), float(alpha))
    if key not in _TABLES:
        _TABLES[key] = CSTable.load_or_build(int(horizon), alpha=float(alpha))
    return _TABLES[key]


def summarize(res: EpisodeResult) -> dict[str, float]:
    """Cell means of what the study reports, primary recommender first."""
    regret = res.regret(PRIMARY_RECOMMENDER)
    m = res.n_episodes
    out: dict[str, float] = {
        "mean_regret": float(regret.mean()),
        "se": float(regret.std(ddof=1) / np.sqrt(m)) if m > 1 else float("nan"),
        "mean_q": float(res.q[PRIMARY_RECOMMENDER].mean()),
        "mean_regret_disc": float(res.regret_disc.mean()),
        "mean_regret_sel": float(res.regret_sel(PRIMARY_RECOMMENDER).mean()),
        "mean_mu_star": float(res.mu_star.mean()),
        "mean_k_final": float(res.k_final.mean()),
        "mean_search_frac": float(res.search_frac.mean()),
        "cap_hit_frac": float(res.cap_hit.mean()),
        "mean_n_demoted": float(res.n_demoted.mean()),
    }
    for rec in RECOMMENDER_NAMES:
        out[f"mean_regret_{rec}"] = float(res.regret(rec).mean())
    return out


def run_item(item: WorkItem) -> list[dict]:
    """Run every configuration of the item on its cell; one row per configuration."""
    spec = item.spec
    table = _table_for(spec.horizon, spec.alpha)
    reservoir = build_reservoir(spec.env_spec)
    prefix = episode_reservoir_prefix(reservoir, spec.base_seed, spec.n_replicates, spec.horizon)
    prior = oracle_prior_from_reservoir(reservoir)

    rows: list[dict] = []
    for cfg in item.configs:
        policy = cfg.build(spec.horizon, spec.n_replicates, table)
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
        rows.append(
            {
                "split": item.split,
                "env": item.env_id,
                "T": int(spec.horizon),
                "cap": int(spec.cap),
                "base_seed": int(spec.base_seed),
                "n_replicates": int(spec.n_replicates),
                "policy": cfg.policy,
                "name": cfg.name(),
                "alpha": float(cfg.alpha),
                "c": float(cfg.c),
                "K0": int(cfg.K0) if cfg.K0 is not None else -1,
                "recommender": PRIMARY_RECOMMENDER,
                **summarize(res),
                "seconds": float(seconds),
            }
        )
    return rows


def run_items(items: list[WorkItem], workers: int) -> Iterator[list[dict]]:
    """Yield each item's rows as it completes; in-process when `workers <= 1`."""
    if workers <= 1:
        _worker_init()
        for item in items:
            yield run_item(item)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(workers), initializer=_worker_init) as pool:
        yield from pool.imap_unordered(run_item, items, chunksize=1)


# ---- selection ------------------------------------------------------------------------


def _pooled(df: pd.DataFrame, by: list[str], n_envs: int) -> pd.DataFrame:
    """Equal-weight mean over environments of the cell means; only complete groups count.

    A configuration missing an environment (an interrupted run, a failed cell) would
    otherwise be pooled over a different set of environments than its neighbours and
    could win on an easier subset.
    """
    g = df.groupby(by, sort=True)
    pooled = g["mean_regret"].mean().rename("pooled_regret").reset_index()
    pooled["n_envs"] = g["env"].nunique().to_numpy()
    pooled["pooled_q"] = g["mean_q"].mean().to_numpy()
    complete = pooled[pooled["n_envs"] == int(n_envs)].copy()
    return complete


def _alpha_key(alpha: float) -> str:
    return str(float(alpha))


def select_schedule_params(tuning: pd.DataFrame, env_ids: Iterable[str]) -> dict:
    """``{"power": {alpha: {T: c}}, "refine_after_init": {T: K0}}`` from the tuning rows.

    Per `(alpha, T)`, the grid point with the lowest equal-weight pooled regret; ties go
    to the smallest `c` (fewest arms). `K0` likewise per `T`.
    """
    env_ids = list(env_ids)
    df = tuning[tuning["env"].isin(env_ids)]
    df = df.assign(alpha_r=df["alpha"].round(ALPHA_DECIMALS), c_r=df["c"].round(C_DECIMALS))

    power: dict[str, dict[str, float]] = {}
    pw = _pooled(df[df["policy"] == POWER], ["alpha_r", "T", "c_r"], len(env_ids))
    for (alpha_r, horizon), grp in pw.groupby(["alpha_r", "T"], sort=True):
        grp = grp.sort_values(["pooled_regret", "c_r"], kind="stable")
        best = grp.iloc[0]
        # The unrounded grid value, so the JSON carries exactly what was run.
        c_exact = float(df.loc[(df["c_r"] == best["c_r"]) & (df["policy"] == POWER), "c"].iloc[0])
        alpha_exact = float(df.loc[df["alpha_r"] == alpha_r, "alpha"].iloc[0])
        power.setdefault(_alpha_key(alpha_exact), {})[str(int(horizon))] = c_exact

    refine: dict[str, int] = {}
    rf = _pooled(df[df["policy"] == REFINE], ["T", "c_r"], len(env_ids))
    for horizon, grp in rf.groupby("T", sort=True):
        grp = grp.sort_values(["pooled_regret", "c_r"], kind="stable")
        refine[str(int(horizon))] = int(round(float(grp.iloc[0]["c_r"])))
    return {POWER: power, REFINE: refine}


def oracle_tuned(tuning: pd.DataFrame) -> pd.DataFrame:
    """Best `c` per `(env, T, alpha)` on the tuning seeds: an in-sample bound, never a policy.

    `best_over_alpha` marks, within each `(env, T)`, the alpha whose oracle `c` is best.
    """
    df = tuning[tuning["policy"] == POWER]
    df = df.assign(alpha_r=df["alpha"].round(ALPHA_DECIMALS))
    rows = []
    for (env, horizon, _alpha_r), grp in df.groupby(["env", "T", "alpha_r"], sort=True):
        best = grp.sort_values(["mean_regret", "c"], kind="stable").iloc[0]
        rows.append(
            {
                "env": env,
                "T": int(horizon),
                "alpha": float(best["alpha"]),
                "c_oracle": float(best["c"]),
                "mean_regret": float(best["mean_regret"]),
                "se": float(best["se"]),
                "mean_q": float(best["mean_q"]),
                "n_grid": int(len(grp)),
                "split": str(best["split"]),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    idx = out.groupby(["env", "T"])["mean_regret"].idxmin()
    out["best_over_alpha"] = False
    out.loc[idx, "best_over_alpha"] = True
    return out


def selected_configs(params: dict) -> list[ScheduleConfig]:
    """The validation candidates: every `(alpha, c*(alpha, T))` and `K0*(T)` -- per horizon."""
    configs: list[tuple[int, ScheduleConfig]] = []
    for alpha_key, per_t in params[POWER].items():
        for horizon, c in per_t.items():
            configs.append((int(horizon), power_config(float(alpha_key), float(c))))
    for horizon, k0 in params[REFINE].items():
        configs.append((int(horizon), refine_config(int(k0))))
    return configs


def build_selection_items(
    params: dict,
    env_ids: Iterable[str],
    cap: int,
    n_replicates: int,
    done: set[tuple] = frozenset(),
    split: str = SELECT_SPLIT,
) -> list[WorkItem]:
    """Validation items: the tuned configurations of each horizon on that horizon's cells."""
    by_horizon: dict[int, list[ScheduleConfig]] = {}
    for horizon, cfg in selected_configs(params):
        by_horizon.setdefault(horizon, []).append(cfg)
    items: list[WorkItem] = []
    for horizon, configs in by_horizon.items():
        items += build_items(
            split, env_ids, [horizon], cap, n_replicates, configs, done, chunk=len(configs)
        )
    return items


def select_p3_star(
    selection: pd.DataFrame, env_ids: Iterable[str], params: dict | None = None
) -> dict[str, dict[str, float]]:
    """``{T: {"alpha": a, "c": c, "pooled_regret": r}}``: the best tuned power schedule per `T`.

    With `params`, only rows at the tuned ``c*(alpha, T)`` of their own horizon are
    candidates (the same guard `restrict` applies, kept here so the function is safe
    on any frame it is handed).
    """
    env_ids = list(env_ids)
    df = selection[(selection["env"].isin(env_ids)) & (selection["policy"] == POWER)]
    df = df.assign(alpha_r=df["alpha"].round(ALPHA_DECIMALS), c_r=df["c"].round(C_DECIMALS))
    if params is not None:
        tuned = {key[:1] + key[2:] for key in selected_keys(params) if key[1] == POWER}
        keep = [(int(row["T"]), *row_key(row)[6:]) in tuned for _, row in df.iterrows()]
        df = df[np.asarray(keep, dtype=bool)]
    pooled = _pooled(df, ["T", "alpha_r", "c_r"], len(env_ids))
    out: dict[str, dict[str, float]] = {}
    for horizon, grp in pooled.groupby("T", sort=True):
        best = grp.sort_values(["pooled_regret", "alpha_r"], kind="stable").iloc[0]
        exact = df[(df["alpha_r"] == best["alpha_r"]) & (df["c_r"] == best["c_r"])].iloc[0]
        out[str(int(horizon))] = {
            "alpha": float(exact["alpha"]),
            "c": float(exact["c"]),
            "pooled_regret": float(best["pooled_regret"]),
        }
    return out


# ---- I/O ----------------------------------------------------------------------------------


def read_csv(path: Path) -> pd.DataFrame | None:
    """Existing rows, floats parsed exactly so `alpha` and `c` keys survive the round trip."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def grid_keys(horizons: Iterable[int], configs: Iterable[ScheduleConfig]) -> set[tuple]:
    """``(T, policy, alpha, c)`` for every configuration at every horizon: the tuning grid."""
    configs = list(configs)
    return {(int(h), *cfg.key()) for h in horizons for cfg in configs}


def selected_keys(params: dict) -> set[tuple]:
    """``(T, policy, alpha, c)`` for the tuned configurations, each at its own horizon only."""
    return {(int(h), *cfg.key()) for h, cfg in selected_configs(params)}


def restrict(
    df: pd.DataFrame,
    split: str,
    env_ids: Iterable[str],
    cap: int,
    n_replicates: int,
    allowed: set[tuple],
) -> pd.DataFrame:
    """The rows of this run's configuration only: the CSV may also hold earlier runs' grids.

    `allowed` is a set of ``(T, policy, alpha, c)`` keys (`grid_keys` / `selected_keys`),
    so the horizon is part of the match: a validation row for ``c*(alpha, T')`` left by
    an earlier run must not compete at horizon ``T`` just because the same ``c`` was
    tuned there.
    """
    env_ids = set(env_ids)
    keep = [
        str(row["split"]) == split
        and str(row["env"]) in env_ids
        and int(row["cap"]) == int(cap)
        and int(row["n_replicates"]) == int(n_replicates)
        and (int(row["T"]), *row_key(row)[5:]) in allowed
        for _, row in df.iterrows()
    ]
    out = df[np.asarray(keep, dtype=bool)].reset_index(drop=True)
    # Two concurrent runs could both append the same row; the last write wins rather
    # than counting twice in the pooled mean.
    out = out.assign(_key=[row_key(row) for _, row in out.iterrows()])
    return out.drop_duplicates("_key", keep="last").drop(columns="_key").reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Whole-file rewrite through a temp name: a crash mid-write never leaves a torn CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_params_for_cap(params: dict, path: Path, cap: int) -> None:
    """Write this run's constants for `cap` without disturbing another cap's.

    A fresh file, or a run at the file's own tuning cap, is written as before (sorted
    keys, the historical form). A run at any other cap merges its blocks under
    ``by_cap["<cap>"]`` of the existing file (`policy_table.merge_cap_block`), so the
    cap-64 constants every shipped table was built from stay byte-identical and the
    reader (`baseline_params_for_cap`) picks the block that matches the deployed cap.
    """
    existing = json.loads(path.read_text()) if path.exists() else None
    tuning = pt.tuning_cap(existing) if existing else None
    if existing is None or tuning is None or int(tuning) == int(cap):
        write_json_atomic(params, path)
        return
    block = {k: v for k, v in params.items() if k in (*TUNED_BLOCKS_WRITTEN, "meta")}
    pt.write_baseline_params(path, pt.merge_cap_block(existing, int(cap), block))


#: The blocks a tuning run produces (the file's other keys belong to other selectors).
TUNED_BLOCKS_WRITTEN: tuple[str, ...] = ("power", "refine_after_init", "p3_star")


def write_json_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def collect(
    items: list[WorkItem],
    existing: pd.DataFrame | None,
    path: Path,
    workers: int,
    label: str,
) -> pd.DataFrame:
    """Run `items`, appending to `existing` and rewriting `path` after every item."""
    frames = [existing] if existing is not None else []
    n_runs = sum(len(it.configs) for it in items)
    n_disk = 0 if existing is None else len(existing)
    print(f"[{label}] {len(items)} items / {n_runs} runs to do, {n_disk} rows on disk")
    if not items:
        return existing if existing is not None else pd.DataFrame()
    start = time.time()
    done_runs = 0
    for i, rows in enumerate(run_items(items, workers), start=1):
        frames.append(pd.DataFrame(rows))
        done_runs += len(rows)
        df = pd.concat(frames, ignore_index=True)
        write_csv_atomic(df, path)
        if i % 10 == 0 or i == len(items):
            el = time.time() - start
            eta = el / done_runs * (n_runs - done_runs) if done_runs else float("nan")
            print(f"[{label}]  {i}/{len(items)} items  {done_runs}/{n_runs} runs  "
                  f"{el:.0f}s elapsed  ETA {eta:.0f}s")
        frames = [df]
    return frames[0]


# ---- CLI ------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default=TUNE_SPLIT, choices=["tune", "val", "test"],
                    help="seed split for the c grid (the plan's is 'tune')")
    ap.add_argument("--select-split", default=SELECT_SPLIT, choices=["tune", "val", "test"],
                    help="seed split on which P3* is chosen among the tuned alphas")
    ap.add_argument("--n-replicates", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--envs", default="main",
                    help="main | heldout | corpus | all | comma list of ids")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    ap.add_argument("--k0s", type=int, nargs="+", default=list(K0_GRID))
    ap.add_argument("--c-min", type=float, default=C_MIN)
    ap.add_argument("--c-max", type=float, default=C_MAX)
    ap.add_argument("--n-c", type=int, default=N_C, help="log-spaced grid points for c")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--chunk", type=int, default=16, help="configurations per work item")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--allow-test-split", action="store_true",
                    help="permit --split/--select-split test (outputs suffixed _TESTSPLIT)")
    args = ap.parse_args(argv)
    suffix = check_splits((args.split, args.select_split), args.allow_test_split)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = output_path(out_dir, "schedule_tuning", ".csv", suffix)
    selection_path = output_path(out_dir, "schedule_selection", ".csv", suffix)
    oracle_path = output_path(out_dir, "schedule_oracle_tuned", ".csv", suffix)
    params_path = output_path(out_dir, "baseline_params", ".json", suffix)

    env_ids = env_ids_for(args.envs)
    horizons = [int(h) for h in args.horizons]
    if args.split == args.select_split:
        print(f"WARNING: tuning and selection on the same split ({args.split}); "
              "P3* will be optimistic")
    seeds = [
        cell_at_cap(split, env_id, horizon, args.cap, args.n_replicates).base_seed
        for split in {args.split, args.select_split}
        for env_id in env_ids
        for horizon in horizons
    ]
    assert_seed_disjointness(seeds)
    for horizon in sorted(set(horizons)):
        CSTable.load_or_build(horizon, alpha=0.05)  # in the parent, so workers only mmap

    configs = tuning_grid(args.alphas, c_grid(args.c_min, args.c_max, args.n_c), args.k0s)
    print(f"envs={list(env_ids)} horizons={horizons} configs/cell={len(configs)} "
          f"M={args.n_replicates} cap={args.cap} workers={args.workers}")

    # 1. tuning grid
    existing = read_csv(tuning_path)
    items = build_items(args.split, env_ids, horizons, args.cap, args.n_replicates, configs,
                        done_keys(existing), chunk=args.chunk)
    tuning = collect(items, existing, tuning_path, args.workers, "tune")
    tuning = restrict(
        tuning, args.split, env_ids, args.cap, args.n_replicates, grid_keys(horizons, configs)
    )

    # 2. select c per (alpha, T) and K0 per T; record the oracle-tuned bound
    params = select_schedule_params(tuning, env_ids)
    write_csv_atomic(oracle_tuned(tuning), oracle_path)
    params["meta"] = {
        "tune_split": args.split,
        "select_split": args.select_split,
        "n_replicates": int(args.n_replicates),
        "cap": int(args.cap),
        "envs": list(env_ids),
        "horizons": horizons,
        "alphas": [float(a) for a in args.alphas],
        "k0_grid": [int(k) for k in args.k0s],
        "c_grid": [float(c) for c in c_grid(args.c_min, args.c_max, args.n_c)],
        "recommender": PRIMARY_RECOMMENDER,
        "pooling": "equal weight over envs of the cell mean regret",
    }
    write_params_for_cap(params, params_path, args.cap)
    for alpha_key, per_t in params[POWER].items():
        print(f"  power alpha={float(alpha_key):.4f}: "
              + ", ".join(f"T={t}: c={c:.3f}" for t, c in per_t.items()))
    print(f"  refine_after_init: {params[REFINE]}")

    # 3. validation: every tuned (alpha, c*) and K0*, P3* = argmin per T
    existing = read_csv(selection_path)
    items = build_selection_items(params, env_ids, args.cap, args.n_replicates, done_keys(existing),
                                  split=args.select_split)
    selection = collect(items, existing, selection_path, args.workers, "select")
    selection = restrict(
        selection, args.select_split, env_ids, args.cap, args.n_replicates, selected_keys(params)
    )
    params["p3_star"] = select_p3_star(selection, env_ids, params)
    write_params_for_cap(params, params_path, args.cap)
    for horizon, best in params["p3_star"].items():
        print(f"  P3* T={horizon}: alpha={best['alpha']:.4f} c={best['c']:.3f} "
              f"pooled regret={best['pooled_regret']:.4f}")
    print(f"wrote {tuning_path}\n      {selection_path}\n      {oracle_path}\n      {params_path}")


if __name__ == "__main__":
    main()
