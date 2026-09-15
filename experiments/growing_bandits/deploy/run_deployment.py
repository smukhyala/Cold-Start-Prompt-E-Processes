#!/usr/bin/env python
"""M6: run the deployment study's tests as resumable, parallel (test, cell, policy) items.

One work item is one policy on one cell -- ``M`` CRN-paired episodes through
`harness.run_cell` -- and it writes exactly one parquet file (the per-episode frame),
one ``.npz`` (the dynamics grid) and, for learned policies under ``--log-states``, one
pickle of on-policy `Snapshot`s. Items are independent, so they run in a
``spawn``-context `multiprocessing.Pool`; the only shared state is read-only and
memory-mapped (CS tables, pairwise log-e tables) and the parent builds all of it
*before* the pool exists, because the atomic cache writers use a fixed temporary
name and two workers missing the same cache would race.

What the parent does once per cell, rather than once per item, is the expensive
hidden-truth constant every policy in the cell shares: the reservoir prefix that
defines the comparator (`comparators.episode_reservoir_prefix`; a 60-step bisection
per draw for the mixture family) and the oracle prior. Both are computed once, the
prefix saved to disk, and every `run_cell` of the cell receives them.

The manifest (``manifest_<test>.jsonl``, one line per finished item, appended by the
parent as results arrive) is what makes a run resumable: ``--resume`` skips every
item already in it whose parquet still exists. It also records provenance -- the
resolved parameters (the tau or c actually deployed), the policy's diagnostic counters
(a non-zero ``n_nonfinite_rows`` fails the item: a non-finite feature at deployment is
a parity bug, not a normal state), the git sha and the wall time.

At the end ``summary_<test>.csv`` holds per (cell, policy, recommender) means and the
paired mean difference vs the two pre-registered references (``cp0``, ``p3_star``)
with `stats.paired_bootstrap` intervals. The full 10k-resample analysis is M7's; this
summary is the quick look that tells you whether a run is sane.

Cells come from `cells.py` (M5): environments, horizons and seeds live there and
nowhere else. Until it lands, ``--test smoke`` may run on a stand-in cell list with
seeds from a reserved range that no real test uses; every other test refuses.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import policy_table as pt  # noqa: E402

from cold_start.growing.deploy import stats  # noqa: E402
from cold_start.growing.deploy.artifacts import load_model  # noqa: E402
from cold_start.growing.deploy.comparators import episode_reservoir_prefix  # noqa: E402
from cold_start.growing.deploy.feature_groups import EVIDENCE_LOGE  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec, LogSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.pairwise_table import (  # noqa: E402
    MAX_TABULATED_N,
    get_pairwise_table,
)
from cold_start.growing.deploy.recommenders import RECOMMENDER_NAMES  # noqa: E402
from cold_start.growing.recommend import oracle_prior_from_reservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

log = logging.getLogger("deploy.run")

DEFAULT_OUT_DIR = ROOT / "results" / "growing_bandits" / "deploy"

TESTS: tuple[str, ...] = ("A", "B", "C", "D", "robust", "cap", "smoke")
REFERENCES: tuple[str, ...] = ("cp0", "p3_star")

#: Episodes per cell by test (DEPLOYMENT_PLAN.md "Environments / horizons / episodes").
DEFAULT_REPLICATES: dict[str, int] = {
    "A": 2000,
    "B": 2000,
    "C": 2000,
    "D": 2000,
    "robust": 500,
    "cap": 1000,
    "smoke": 100,
}
DEFAULT_CAP = 64
D_HORIZONS: tuple[int, ...] = (200, 1000)
#: Test D's extrapolation cells: three environments at T=2000, M=1000. One Beta and
#: two tail reservoirs, chosen so both families and both tail exponents of the cap
#: sweep are represented; the plan fixes the count, not the ids.
T2000_ENVS: tuple[str, ...] = ("beta_good_common", "tail_b2.0_mu1.0_c1.0", "tail_b8.0_mu1.0_c1.0")
T2000_HORIZON = 2000
T2000_REPLICATES = 1000
CAP_SWEEP_ENVS: tuple[str, ...] = (
    "beta_good_common",
    "beta_rare_excellent",
    "tail_b2.0_mu1.0_c1.0",
    "tail_b8.0_mu1.0_c1.0",
)
CAP_SWEEP_HORIZONS: tuple[int, ...] = (200, 1000)
CAP_SWEEP_CAPS: tuple[int | str, ...] = (32, 64, "T")
SMOKE_ENVS: tuple[str, ...] = ("beta_good_common", "tail_b2.0_mu1.0_c1.0")
SMOKE_HORIZON = 100

#: ``--log-states``: 24 normalized times x 128 replicates per learned policy (plan §10).
LOG_TIMES = 24
LOG_REPLICATES = 128
LOGGED_GROUPS: tuple[str, ...] = ("learned", "rule")

#: Stand-in seeds for a smoke run before `cells.py` exists: a range disjoint from the
#: corpus ``[20.26M, 343.4M]``, the old benchmark ``[4.5k, 14.1k]`` and every split
#: base in the plan (1M / 5M / 10M + cell_id * 1000).
STANDIN_SEED_BASE = 900_000_000
STANDIN_SEED_STRIDE = 1_000

FAMILY_OF_TYPE: dict[str, str] = {"beta": "A", "tail": "B", "mixture": "C"}

MAX_TASKS_PER_CHILD = 8


# ---- cells --------------------------------------------------------------------------------


def cell_name(spec: CellSpec) -> str:
    return f"{spec.env_id}_T{int(spec.horizon)}_cap{int(spec.cap)}"


def family_of(env_spec: dict) -> str:
    return FAMILY_OF_TYPE.get(str(env_spec.get("type")), "?")


def import_cells_module():
    """`cells.py` (M5) if it has landed, else ``None``."""
    try:
        import cells
    except ImportError:
        return None
    return cells


def _env_catalogue() -> dict[str, dict]:
    """Every corpus and held-out environment by id, from the label pipeline's lists."""
    from label_states import FAMILY_A, FAMILY_B, FAMILY_C

    return {env_id: spec for env_id, spec in FAMILY_A + FAMILY_B + FAMILY_C}


def standin_cell(env_id: str, horizon: int, cap: int, n_replicates: int, index: int) -> CellSpec:
    """A smoke-only cell with a reserved seed; see `STANDIN_SEED_BASE`."""
    catalogue = _env_catalogue()
    if env_id not in catalogue:
        raise KeyError(f"unknown environment {env_id!r}")
    return CellSpec(
        env_id=env_id,
        env_spec=catalogue[env_id],
        horizon=int(horizon),
        cap=int(cap),
        base_seed=STANDIN_SEED_BASE + STANDIN_SEED_STRIDE * int(index),
        n_replicates=int(n_replicates),
    )


def _cell_grid(test: str, cells_mod) -> list[tuple[str, int, int, int | None]]:
    """``(env_id, horizon, cap, n_replicates_override)`` tuples for `test`.

    ``None`` for the replicate count means the test's default applies; only Test D's
    T=2000 extrapolation cells fix their own.
    """
    if cells_mod is not None:
        main_envs = list(cells_mod.MAIN_ENVS)
        heldout = list(cells_mod.HELDOUT_ENVS)
        all_corpus = list(cells_mod.ALL_CORPUS_ENVS)
        horizons = tuple(int(h) for h in cells_mod.HORIZONS)
    else:
        main_envs = heldout = all_corpus = []
        horizons = ()

    grid: list[tuple[str, int, int, int | None]] = []
    if test in ("A", "B"):
        grid = [(e, T, DEFAULT_CAP, None) for e in main_envs for T in horizons]
    elif test == "C":
        grid = [(e, T, DEFAULT_CAP, None) for e in heldout for T in horizons]
    elif test == "D":
        grid = [(e, T, DEFAULT_CAP, None) for e in main_envs for T in D_HORIZONS]
        grid += [(e, T2000_HORIZON, DEFAULT_CAP, T2000_REPLICATES) for e in T2000_ENVS]
    elif test == "robust":
        grid = [(e, T, DEFAULT_CAP, None) for e in all_corpus for T in horizons]
    elif test == "cap":
        for e in CAP_SWEEP_ENVS:
            for T in CAP_SWEEP_HORIZONS:
                for cap in CAP_SWEEP_CAPS:
                    grid.append((e, T, T if cap == "T" else int(cap), None))
    elif test == "smoke":
        grid = [(e, SMOKE_HORIZON, DEFAULT_CAP, None) for e in SMOKE_ENVS]
    else:
        raise ValueError(f"unknown test {test!r}; tests={TESTS}")
    return grid


def parse_cell_overrides(spec: str) -> list[tuple[str, int, int, int | None]]:
    """``env:T:cap[,env:T:cap...]`` -> grid tuples (a manual cell list for any test)."""
    out: list[tuple[str, int, int, int | None]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"cell override {token!r} must be env:T:cap")
        out.append((parts[0], int(parts[1]), int(parts[2]), None))
    return out


def build_cells(
    test: str,
    *,
    n_replicates: int | None,
    cells_mod,
    overrides: list[tuple[str, int, int, int | None]] | None = None,
) -> list[CellSpec]:
    """The `CellSpec` list for `test`, seeded by `cells.make_cell` on the test split.

    Without `cells.py` only the smoke test may run, on stand-in seeds -- a real test
    on ad-hoc seeds would not be the pre-registered study.
    """
    grid = overrides if overrides is not None else _cell_grid(test, cells_mod)
    if cells_mod is None and test != "smoke":
        raise RuntimeError(
            f"test {test!r} needs experiments/growing_bandits/deploy/cells.py (M5) for its "
            "environments and seeds; only --test smoke may run on stand-in seeds"
        )
    if not grid:
        raise RuntimeError(f"no cells for test {test!r}")
    default_m = int(n_replicates) if n_replicates is not None else DEFAULT_REPLICATES[test]
    specs: list[CellSpec] = []
    for index, (env_id, horizon, cap, m_override) in enumerate(grid):
        m = default_m if (m_override is None or n_replicates is not None) else int(m_override)
        if cells_mod is not None:
            specs.append(cells_mod.make_cell("test", env_id, int(horizon), int(cap), m))
        else:
            log.warning(
                "cells.py absent: cell %s_T%d_cap%d uses stand-in seed %d (smoke only)",
                env_id, horizon, cap, STANDIN_SEED_BASE + STANDIN_SEED_STRIDE * index,
            )
            specs.append(standin_cell(env_id, horizon, cap, m, index))
    names = [cell_name(s) for s in specs]
    if len(set(names)) != len(names):
        raise RuntimeError(f"duplicate cells in test {test!r}: {names}")
    return specs


def log_spec_for(spec: CellSpec) -> LogSpec:
    """24 normalized times in ``[n_initial_arms, T]``, first 128 replicates (plan §10)."""
    n0, T = int(spec.n_initial_arms), int(spec.horizon)
    times = sorted({min(max(int(round(g * T / LOG_TIMES)), n0), T) for g in range(1, LOG_TIMES + 1)})
    return LogSpec(times=tuple(times), replicates=min(LOG_REPLICATES, int(spec.n_replicates)))


# ---- per-cell constants (parent process) --------------------------------------------------


def _save_atomic_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)


def prepare_cell_constants(spec: CellSpec, comparators_dir: Path) -> tuple[Path, tuple[float, float]]:
    """Compute (once) and cache the cell's comparator prefix and oracle prior.

    The prefix is ``(M, T)`` float64 -- 16 MB at M=2000, T=1000 -- so it goes to disk
    rather than through the pool's pipe with every item; workers `np.load` it. The
    cache key includes the seed and M, so a cell shared by two tests (A and B use the
    same seeds) is computed once for both.
    """
    comparators_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cell_name(spec)}_seed{int(spec.base_seed)}_M{int(spec.n_replicates)}"
    npy_path = comparators_dir / f"{stem}.npy"
    json_path = comparators_dir / f"{stem}.json"
    m, T = int(spec.n_replicates), int(spec.horizon)
    if npy_path.exists() and json_path.exists():
        prefix = np.load(npy_path, mmap_mode="r")
        with open(json_path) as fh:
            meta = json.load(fh)
        if prefix.shape == (m, T) and meta.get("oracle_prior") is not None:
            a, b = meta["oracle_prior"]
            return npy_path, (float(a), float(b))
    t0 = time.perf_counter()
    reservoir = build_reservoir(spec.env_spec)
    prefix = episode_reservoir_prefix(reservoir, spec.base_seed, m, T)
    prior = oracle_prior_from_reservoir(reservoir)
    _save_atomic_npy(npy_path, np.ascontiguousarray(prefix, dtype=np.float64))
    with open(json_path, "w") as fh:
        json.dump(
            {
                "cell": cell_name(spec),
                "base_seed": int(spec.base_seed),
                "n_replicates": m,
                "horizon": T,
                "oracle_prior": [float(prior[0]), float(prior[1])],
            },
            fh,
        )
    log.info("cell constants %s: %.1fs", stem, time.perf_counter() - t0)
    return npy_path, (float(prior[0]), float(prior[1]))


def prebuild_shared_tables(horizons: set[int], need_pairwise: set[int], alpha: float = 0.05) -> None:
    """Build every memory-mapped table the run needs, in the parent, before forking.

    Hard requirement (failure-mode review of M1): `CSTable.load_or_build` and
    `get_pairwise_table` save through a fixed ``.tmp`` name, so two workers missing the
    same cache would race. Horizons above `MAX_TABULATED_N` use the exact chunked
    evaluator and have nothing to cache.
    """
    for T in sorted(horizons):
        t0 = time.perf_counter()
        CSTable.load_or_build(int(T), alpha)
        log.info("CS table T=%d ready (%.1fs)", T, time.perf_counter() - t0)
    for T in sorted(need_pairwise):
        if int(T) > MAX_TABULATED_N:
            log.warning(
                "T=%d exceeds MAX_TABULATED_N=%d: f_log_e_pair uses the exact chunked "
                "evaluator (slow) in every worker", T, MAX_TABULATED_N,
            )
            continue
        t0 = time.perf_counter()
        get_pairwise_table(int(T))
        log.info("pairwise table T=%d ready (%.1fs)", T, time.perf_counter() - t0)


# ---- work items ------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """Everything a worker needs for one (test, cell, policy); plain data, picklable."""

    test: str
    cell: str
    policy: str
    group: str
    spec: CellSpec
    params: dict
    policy_seed: int
    prefix_path: str
    oracle_prior: tuple[float, float]
    dynamics_grid: int
    log_states: LogSpec | None
    parquet_path: str
    dynamics_path: str
    snapshots_path: str | None
    git_sha: str


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def output_paths(out_dir: Path, test: str, cell: str, policy: str) -> tuple[Path, Path, Path]:
    return (
        out_dir / "episodes" / test / cell / f"{policy}.parquet",
        out_dir / "dynamics" / test / cell / f"{policy}.npz",
        out_dir / "snapshots" / test / cell / f"{policy}.pkl",
    )


def _item_sort_key(item: WorkItem) -> tuple[int, int, str, str]:
    """Longest-first: big horizons and learned policies start early (LPT balancing)."""
    return (-int(item.spec.horizon), 0 if item.group in LOGGED_GROUPS else 1, item.cell, item.policy)


def build_work(
    test: str,
    cells: list[CellSpec],
    policies: list[str],
    *,
    out_dir: Path,
    baseline_params: dict | None,
    thresholds: dict | None,
    models_dir: Path,
    dynamics_grid: int,
    log_states: bool,
    done: set[tuple[str, str, str]],
    git_sha: str,
    artifacts: dict[str, dict] | None = None,
) -> list[WorkItem]:
    """Resolve every (cell, policy) into a `WorkItem`, skipping those in `done`.

    Parameters are resolved here, in the parent, so every placeholder warning is
    printed once and the manifest can carry the values actually deployed. `artifacts`
    (variant -> loaded model) saves re-reading each joblib once per cell.
    """
    comparators_dir = out_dir / "comparators"
    artifacts = artifacts or {}
    items: list[WorkItem] = []
    for spec in cells:
        name = cell_name(spec)
        pending = [p for p in policies if (test, name, p) not in done]
        if not pending:
            continue
        prefix_path, prior = prepare_cell_constants(spec, comparators_dir)
        for policy in pending:
            variant = pt.variant_of(policy)
            params = pt.resolve_params(
                policy,
                spec.horizon,
                baseline_params=baseline_params,
                thresholds=thresholds,
                models_dir=models_dir,
                artifact=artifacts.get(variant) if variant is not None else None,
            )
            group = pt.POLICIES[policy]["group"]
            parquet, dynamics, snapshots = output_paths(out_dir, test, name, policy)
            logged = log_states and group in LOGGED_GROUPS
            items.append(
                WorkItem(
                    test=test,
                    cell=name,
                    policy=policy,
                    group=group,
                    spec=spec,
                    params=params,
                    policy_seed=pt.policy_seed(policy),
                    prefix_path=str(prefix_path),
                    oracle_prior=prior,
                    dynamics_grid=int(dynamics_grid),
                    log_states=log_spec_for(spec) if logged else None,
                    parquet_path=str(parquet),
                    dynamics_path=str(dynamics),
                    snapshots_path=str(snapshots) if logged else None,
                    git_sha=git_sha,
                )
            )
    items.sort(key=_item_sort_key)
    return items


# ---- worker ----------------------------------------------------------------------------------

_TABLES: dict[tuple[int, float], CSTable] = {}
_PAIRWISE: dict[int, object] = {}
_ARTIFACTS: dict[str, dict] = {}
_THREAD_LIMITS = None


def _worker_init(log_level: int) -> None:
    """Per-process setup: logging (spawned children start with none) and one BLAS /
    OpenMP thread each, since the parallelism is across items."""
    global _THREAD_LIMITS
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s"
    )
    try:
        from threadpoolctl import threadpool_limits

        _THREAD_LIMITS = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - sklearn depends on threadpoolctl
        pass


def _table(horizon: int, alpha: float) -> CSTable:
    key = (int(horizon), float(alpha))
    if key not in _TABLES:
        _TABLES[key] = CSTable.load_or_build(int(horizon), float(alpha))
    return _TABLES[key]


def _pairwise(horizon: int):
    T = int(horizon)
    if T not in _PAIRWISE:
        _PAIRWISE[T] = get_pairwise_table(T)
    return _PAIRWISE[T]


def _artifact(path: str) -> dict:
    if path not in _ARTIFACTS:
        _ARTIFACTS[path] = load_model(path)
    return _ARTIFACTS[path]


def _atomic_replace(tmp: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)


def run_item(item: WorkItem) -> dict:
    """Run one (cell, policy), write its outputs, return its manifest line."""
    t0 = time.perf_counter()
    spec = item.spec
    table = _table(spec.horizon, spec.alpha)
    reservoir = build_reservoir(spec.env_spec)
    prefix = np.load(item.prefix_path)

    artifact = None
    pairwise = None
    if pt.POLICIES[item.policy]["kind"] == "model":
        artifact = _artifact(item.params["artifact"])
        if any(c in EVIDENCE_LOGE for c in artifact["features"]):
            pairwise = _pairwise(spec.horizon)
    # A fresh policy object per item: the harness resets and isolates one anyway, but
    # there is then no object whose state could carry across items at all.
    policy = pt.build_policy(
        item.policy,
        item.params,
        horizon=spec.horizon,
        n_replicates=spec.n_replicates,
        table=table,
        pairwise=pairwise,
        artifact=artifact,
    )
    res = run_cell(
        spec,
        policy,
        table=table,
        reservoir=reservoir,
        dynamics_grid=item.dynamics_grid,
        log_states=item.log_states,
        policy_seed=item.policy_seed,
        comparator_prefix=prefix,
        oracle_prior=item.oracle_prior,
    )

    counters = None
    counters_fn = getattr(policy, "counters", None)
    if counters_fn is not None:
        counters = {k: int(v) for k, v in counters_fn().items()}
        if counters.get("n_nonfinite_rows", 0) != 0:
            raise RuntimeError(
                f"{item.policy} on {item.cell}: {counters['n_nonfinite_rows']} non-finite "
                "feature rows at deployment (parity bug; register #5)"
            )

    frame = res.to_frame(item.policy, spec)
    frame.insert(0, "test", item.test)
    frame.insert(1, "cell", item.cell)
    frame.insert(2, "family", family_of(spec.env_spec))
    frame.insert(3, "group", item.group)
    frame["replicate"] = frame["episode"].to_numpy()

    parquet = Path(item.parquet_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    _atomic_replace(tmp, parquet)

    dynamics_written = False
    if res.dynamics is not None:
        dyn = Path(item.dynamics_path)
        dyn.parent.mkdir(parents=True, exist_ok=True)
        tmp = dyn.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **{k: np.asarray(v) for k, v in res.dynamics.items()})
        _atomic_replace(tmp, dyn)
        dynamics_written = True

    n_snapshots = 0
    if item.snapshots_path is not None:
        snap = Path(item.snapshots_path)
        snap.parent.mkdir(parents=True, exist_ok=True)
        tmp = snap.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(res.snapshots, fh, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_replace(tmp, snap)
        n_snapshots = len(res.snapshots)

    return {
        "test": item.test,
        "cell": item.cell,
        "policy": item.policy,
        "group": item.group,
        "n": int(res.n_episodes),
        "seconds": round(time.perf_counter() - t0, 3),
        "sha": item.git_sha,
        "env_id": spec.env_id,
        "family": family_of(spec.env_spec),
        "horizon": int(spec.horizon),
        "cap": int(spec.cap),
        "base_seed": int(spec.base_seed),
        "n_replicates": int(spec.n_replicates),
        "policy_seed": int(item.policy_seed),
        "params": item.params,
        "counters": counters,
        "search_frac_mean": float(np.mean(res.search_frac)),
        "parquet": str(parquet),
        "dynamics": str(item.dynamics_path) if dynamics_written else None,
        "snapshots": str(item.snapshots_path) if item.snapshots_path is not None else None,
        "n_snapshots": n_snapshots,
        "finished_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


def _run_item_safe(item: WorkItem) -> dict:
    """Never let one bad item take the pool down: failures come back as records."""
    try:
        rec = run_item(item)
        rec["ok"] = True
        return rec
    except Exception as exc:  # reported to the parent, which decides what to do
        return {
            "ok": False,
            "test": item.test,
            "cell": item.cell,
            "policy": item.policy,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---- manifest ----------------------------------------------------------------------------------


def manifest_path(out_dir: Path, test: str) -> Path:
    return out_dir / f"manifest_{test}.jsonl"


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def completed_items(records: list[dict]) -> set[tuple[str, str, str]]:
    """Items whose manifest line exists *and* whose parquet is still on disk."""
    done: set[tuple[str, str, str]] = set()
    for rec in records:
        if rec.get("parquet") and Path(rec["parquet"]).exists():
            done.add((rec["test"], rec["cell"], rec["policy"]))
    return done


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def run_items(items: list[WorkItem], *, workers: int, manifest: Path) -> tuple[int, list[dict]]:
    """Run every item, append each success to the manifest as it lands, return
    ``(n_failed, failure_records)``. ``workers <= 1`` runs inline (debuggable)."""
    n = len(items)
    n_done = 0
    failures: list[dict] = []
    t_start = time.time()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    def handle(rec: dict) -> None:
        nonlocal n_done
        n_done += 1
        elapsed = time.time() - t_start
        eta = elapsed * (n - n_done) / max(n_done, 1)
        if rec["ok"]:
            line = {k: v for k, v in rec.items() if k != "ok"}
            with open(manifest, "a") as fh:
                fh.write(json.dumps(line) + "\n")
            log.info(
                "[%d/%d] %s/%s  %.1fs  (elapsed %s, ETA %s)",
                n_done, n, rec["cell"], rec["policy"], rec["seconds"],
                _format_eta(elapsed), _format_eta(eta),
            )
        else:
            failures.append(rec)
            log.error(
                "[%d/%d] %s/%s FAILED: %s\n%s",
                n_done, n, rec["cell"], rec["policy"], rec["error"], rec.get("traceback", ""),
            )

    if workers <= 1:
        for item in items:
            handle(_run_item_safe(item))
        return len(failures), failures

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=int(workers),
        initializer=_worker_init,
        initargs=(logging.getLogger().level or logging.INFO,),
        maxtasksperchild=MAX_TASKS_PER_CHILD,
    ) as pool:
        for rec in pool.imap_unordered(_run_item_safe, items, chunksize=1):
            handle(rec)
    return len(failures), failures


# ---- summary ----------------------------------------------------------------------------------


_MEAN_COLUMNS: tuple[str, ...] = (
    "regret_disc",
    "k_final",
    "search_frac",
    "cap_hit",
    "n_eliminated_final",
    "herfindahl",
    "n_singletons_final",
    "mu_star",
    "mu_star_cap",
    "best_discovered",
)


def _paired_seed(*parts: str) -> int:
    return int(zlib.crc32("|".join(parts).encode("utf-8")))


def summarize_cell(
    frames: dict[str, pd.DataFrame],
    *,
    n_boot: int,
    references: tuple[str, ...] = REFERENCES,
) -> list[dict]:
    """Per (policy, recommender) rows for one cell, from its policies' episode frames.

    The paired difference vs each reference is over episodes aligned by ``episode``
    (the CRN pairing), and is exactly zero for a reference against itself -- which is
    how a reader can tell the alignment is right.
    """
    rows: list[dict] = []
    ref_frames = {r: frames[r].sort_values("episode").reset_index(drop=True) for r in references if r in frames}
    for policy, frame in frames.items():
        frame = frame.sort_values("episode").reset_index(drop=True)
        head = frame.iloc[0]
        base = {
            "test": head["test"],
            "cell": head["cell"],
            "env_id": head["env_id"],
            "family": head["family"],
            "horizon": int(head["horizon"]),
            "cap": int(head["cap"]),
            "base_seed": int(head["base_seed"]),
            "policy": policy,
            "group": head["group"],
            "n": int(len(frame)),
        }
        for col in _MEAN_COLUMNS:
            base[col] = float(frame[col].astype(np.float64).mean())
        demoted = frame["n_demoted"].to_numpy(dtype=np.float64)
        base["n_demoted"] = float(demoted.mean()) if bool(np.all(demoted >= 0)) else float("nan")
        hit = frame["cap_hit"].to_numpy(dtype=bool)
        t_hit = frame["t_cap_hit"].to_numpy(dtype=np.float64)
        base["t_cap_hit_given_hit"] = float(t_hit[hit].mean()) if hit.any() else float("nan")

        for rec in RECOMMENDER_NAMES:
            if f"regret_{rec}" not in frame.columns:
                continue
            row = dict(base)
            row["recommender"] = rec
            regret = frame[f"regret_{rec}"].to_numpy(dtype=np.float64)
            row["regret"] = float(regret.mean())
            row["q"] = float(frame[f"q_{rec}"].mean())
            row["regret_sel"] = float(frame[f"regret_sel_{rec}"].mean())
            row["regret_sup"] = float(frame[f"regret_sup_{rec}"].mean())
            row["n_rec"] = float(frame[f"n_rec_{rec}"].mean())
            for ref in references:
                prefix = f"d_regret_vs_{ref}"
                ref_frame = ref_frames.get(ref)
                if ref_frame is None or len(ref_frame) != len(frame):
                    for suffix in ("", "_lo", "_hi", "_win", "_se"):
                        row[prefix + suffix] = float("nan")
                    continue
                if not np.array_equal(ref_frame["episode"].to_numpy(), frame["episode"].to_numpy()):
                    raise RuntimeError(f"episode misalignment between {policy} and {ref} in {head['cell']}")
                if int(ref_frame["base_seed"].iloc[0]) != int(head["base_seed"]):
                    raise RuntimeError(f"{policy} and {ref} ran on different seeds in {head['cell']}")
                diff = regret - ref_frame[f"regret_{rec}"].to_numpy(dtype=np.float64)
                boot = stats.paired_bootstrap(
                    diff, n_boot=int(n_boot), seed=_paired_seed(head["cell"], policy, rec, ref)
                )
                row[prefix] = boot["mean"]
                row[prefix + "_lo"] = boot["lo"]
                row[prefix + "_hi"] = boot["hi"]
                row[prefix + "_win"] = boot["win_rate"]
                row[prefix + "_se"] = boot["se_paired"]
            rows.append(row)
    return rows


def write_summary(out_dir: Path, test: str, records: list[dict], *, n_boot: int) -> Path:
    """``summary_<test>.csv`` from every completed item in the manifest, cell by cell
    (so a Test-A run never holds 1,300 frames at once)."""
    by_cell: dict[str, dict[str, str]] = {}
    for rec in records:
        if rec.get("parquet") and Path(rec["parquet"]).exists():
            by_cell.setdefault(rec["cell"], {})[rec["policy"]] = rec["parquet"]
    rows: list[dict] = []
    t0 = time.perf_counter()
    for cell in sorted(by_cell):
        frames = {policy: pd.read_parquet(path) for policy, path in by_cell[cell].items()}
        rows.extend(summarize_cell(frames, n_boot=n_boot))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["cell", "policy", "recommender"]).reset_index(drop=True)
    path = out_dir / f"summary_{test}.csv"
    summary.to_csv(path, index=False)
    log.info("summary: %d rows over %d cells -> %s (%.1fs)", len(summary), len(by_cell), path,
             time.perf_counter() - t0)
    return path


# ---- CLI --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True, choices=TESTS)
    p.add_argument("--n-replicates", type=int, default=None, help="episodes per cell (default: the test's)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--policies", default=None, help="comma-separated subset of policy_table.POLICIES")
    p.add_argument("--cells", default=None, help="manual cell list env:T:cap,... (overrides the test's grid)")
    p.add_argument("--dynamics-grid", type=int, default=50)
    p.add_argument("--log-states", action="store_true", help="log on-policy snapshots for learned policies")
    p.add_argument("--resume", action="store_true", help="skip items already in the manifest")
    p.add_argument("--n-boot", type=int, default=2000, help="paired-bootstrap resamples in the summary")
    p.add_argument("--skip-summary", action="store_true")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--models-dir", type=Path, default=pt.DEFAULT_MODELS_DIR)
    p.add_argument("--baseline-params", type=Path, default=None,
                   help="baseline_params.json (default: <out-dir>/baseline_params.json)")
    p.add_argument("--thresholds", type=Path, default=None,
                   help="thresholds.json (default: <out-dir>/thresholds.json)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, cells: list[CellSpec] | None = None) -> dict:
    """Run one test end to end. `cells` (a function-level override for tests) replaces
    the grid from `cells.py`; the CLI ``--cells`` does the same by name."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    test = args.test
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    t_run = time.time()

    policies = list(pt.TEST_POLICIES[test])
    if args.policies:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
        pt.check_policies(policies)
    if len(set(policies)) != len(policies):
        raise SystemExit(f"duplicate policies in {policies}")
    missing = [
        p for p in policies
        if pt.variant_of(p) is not None and not pt.artifact_path(pt.variant_of(p), models_dir).exists()
    ]
    if missing:
        raise SystemExit(f"model artifacts missing under {models_dir} for policies {missing}")

    cells_mod = import_cells_module()
    if cells is None:
        overrides = parse_cell_overrides(args.cells) if args.cells else None
        cells = build_cells(test, n_replicates=args.n_replicates, cells_mod=cells_mod, overrides=overrides)
    elif args.n_replicates is not None:
        cells = [replace(c, n_replicates=int(args.n_replicates)) for c in cells]
    seeds = [int(c.base_seed) for c in cells]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"cells share a base_seed: {seeds}")
    if cells_mod is not None:
        cells_mod.assert_seed_disjointness(seeds)
        log.info("seed disjointness asserted for %d cells", len(cells))
    else:
        log.warning("cells.py absent: seed disjointness NOT asserted (stand-in seeds)")

    baseline_params = pt.load_baseline_params(args.baseline_params or out_dir / "baseline_params.json")
    thresholds = pt.load_thresholds(args.thresholds or out_dir / "thresholds.json")

    manifest = manifest_path(out_dir, test)
    if args.resume:
        prior = read_manifest(manifest)
        done = completed_items(prior)
        log.info("resume: %d items already complete in %s", len(done), manifest)
    else:
        prior = []
        done = set()
        if manifest.exists():
            log.warning("overwriting existing %s (use --resume to continue it)", manifest)
            manifest.unlink()

    artifacts = {
        v: load_model(pt.artifact_path(v, models_dir))
        for v in {pt.variant_of(p) for p in policies} - {None}
    }
    git_sha = _git_sha()
    items = build_work(
        test,
        cells,
        policies,
        out_dir=out_dir,
        baseline_params=baseline_params,
        thresholds=thresholds,
        models_dir=models_dir,
        dynamics_grid=args.dynamics_grid,
        log_states=args.log_states,
        done=done,
        git_sha=git_sha,
        artifacts=artifacts,
    )
    n_total = len(cells) * len(policies)
    log.info(
        "test %s: %d cells x %d policies = %d items, %d to run (%d workers, git %s)",
        test, len(cells), len(policies), n_total, len(items), args.workers, git_sha,
    )

    # Every shared, memory-mapped table is built here, before any worker exists.
    any_log_e = any(
        any(c in EVIDENCE_LOGE for c in art["features"]) for art in artifacts.values()
    )
    need_pairwise = {int(c.horizon) for c in cells} if any_log_e else set()
    if items:
        prebuild_shared_tables({int(c.horizon) for c in cells}, need_pairwise)

    n_failed, failures = run_items(items, workers=int(args.workers), manifest=manifest)

    records = read_manifest(manifest)
    summary_path = None
    if not args.skip_summary:
        summary_path = write_summary(out_dir, test, records, n_boot=int(args.n_boot))
    wall = time.time() - t_run
    log.info("test %s done: %d run, %d skipped, %d failed, %.1fs wall", test, len(items) - n_failed,
             len(done), n_failed, wall)
    result = {
        "test": test,
        "n_cells": len(cells),
        "n_policies": len(policies),
        "n_items": n_total,
        "n_run": len(items) - n_failed,
        "n_skipped": len(done),
        "n_failed": n_failed,
        "failures": failures,
        "manifest": str(manifest),
        "summary": str(summary_path) if summary_path else None,
        "wall_seconds": wall,
    }
    if n_failed:
        raise SystemExit(f"{n_failed} of {len(items)} items failed; see log above")
    return result


if __name__ == "__main__":
    main()
