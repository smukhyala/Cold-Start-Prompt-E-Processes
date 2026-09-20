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
parent as results arrive; never truncated, the latest line per item wins) is what
makes a run resumable: ``--resume`` skips an item only when its latest line still
describes the item about to run -- same resolved parameters (so a tau or schedule
constant that changed when M5's tuning outputs landed re-runs the cell), same
seed and replicate count, same model artifact bytes, and snapshots present if
they are requested now -- and its parquet still exists; anything else is re-run
with a loud log line. A run without ``--resume`` redoes what it was asked for
while leaving every other item's record in place. It also records provenance -- the
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
import functools
import hashlib
import importlib
import importlib.metadata
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

#: Smoke runs draw validation seeds: no effect size is ever quoted from test seeds
#: before the real run. Every other test uses the test split.
SMOKE_SPLIT = "val"
TEST_SPLIT = "test"

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
    """The `CellSpec` list for `test`, seeded by `cells.make_cell` on the test split
    (the validation split for the smoke test, see `SMOKE_SPLIT`).

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
    split = SMOKE_SPLIT if test == "smoke" else TEST_SPLIT
    specs: list[CellSpec] = []
    for index, (env_id, horizon, cap, m_override) in enumerate(grid):
        m = default_m if (m_override is None or n_replicates is not None) else int(m_override)
        if cells_mod is not None:
            specs.append(cells_mod.make_cell(split, env_id, int(horizon), int(cap), m))
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
    artifact_sha: str | None = None
    #: `sim_surface_sha` of the code that will produce this episode.
    sim_sha: str = ""


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: str | Path) -> str:
    """Content hash of a model artifact, so a retrained model invalidates its items."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Every module whose bytes can change an episode: the transitive closure, over the
#: project's own code, of `harness.run_cell`, `policy_table` (which resolves the
#: constants and builds the policy object) and `pairwise_table` (the log-e evidence the
#: runner hands to a policy). `test_sim_surface_covers_everything_reachable_from_run_cell`
#: asserts the list *equals* that closure, so a new module cannot join the simulation
#: surface without joining the fingerprint.
#:
#: The three package ``__init__.py`` shells are docstring-only and deliberately out of
#: scope; the same test holds them to that.
SIM_SURFACE_MODULES: tuple[str, ...] = (
    "cold_start.growing.deploy.artifacts",
    "cold_start.growing.deploy.comparators",
    "cold_start.growing.deploy.feature_groups",
    "cold_start.growing.deploy.features_vec",
    "cold_start.growing.deploy.harness",
    "cold_start.growing.deploy.history_vec",
    "cold_start.growing.deploy.model_policy",
    "cold_start.growing.deploy.pairwise_table",
    "cold_start.growing.deploy.recommenders",
    "cold_start.growing.deploy.rules",
    "cold_start.growing.allocation",
    "cold_start.growing.evidence",
    "cold_start.growing.features",
    "cold_start.growing.labeling",
    "cold_start.growing.recommend",
    "cold_start.growing.reservoirs",
    "cold_start.growing.rng",
    "cold_start.growing.schema",
    "cold_start.growing.search_policies",
    "cold_start.growing.simulator",
    "cold_start.growing.state",
    "cold_start.growing.tables",
    "cold_start.registry",
    "policy_table",
)

#: Third-party packages whose version is recorded beside `sim_surface_sha`. The
#: fingerprint hashes *source only*: a numpy or scipy upgrade changes an episode without
#: changing a byte of this repo, so the versions are stamped rather than pretended away.
SIM_SURFACE_PACKAGES: tuple[str, ...] = ("numpy", "scipy", "scikit-learn", "pandas", "pyarrow")


def sim_surface_files() -> dict[str, Path]:
    """``module name -> source file`` for `SIM_SURFACE_MODULES`."""
    out: dict[str, Path] = {}
    for name in SIM_SURFACE_MODULES:
        module = importlib.import_module(name)
        path = getattr(module, "__file__", None)
        if path is None:
            raise RuntimeError(f"{name} has no source file; it cannot be fingerprinted")
        out[name] = Path(path)
    return out


@functools.lru_cache(maxsize=1)
def sim_surface_sha() -> str:
    """A 12-hex fingerprint of the simulation surface's source.

    `stale_reason` checks the resolved params, the seed and replicate count, the
    horizon/cap and the model artifact's bytes -- but never the code, and the manifest's
    ``sha`` (the repo HEAD) is not a substitute: it moves for every unrelated commit and
    the analysis never reads it. The shipped episode sets are in fact mixed across code
    versions (``manifest_A`` is 1360 items at one sha plus 80 at another), which only a
    surface fingerprint can judge. With this stamped, a ``--resume`` cannot silently keep
    episodes produced by different code, and `analyze_deployment.load_cells` refuses a
    test that spans more than one surface.

    Keyed on module name, not path, so the value does not depend on where the checkout
    lives.
    """
    h = hashlib.sha256()
    for name, path in sorted(sim_surface_files().items()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(file_sha256(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def sim_surface_versions() -> dict[str, str]:
    """The resolved versions of the numerical stack `sim_surface_sha` cannot hash."""
    out = {"python": ".".join(str(v) for v in sys.version_info[:3])}
    for name in SIM_SURFACE_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


def stale_reason(record: dict, item: WorkItem, *, ignore_sim_sha: bool = False) -> str | None:
    """Why a manifest `record` no longer describes `item`, or ``None`` if it still does.

    The checks are the inputs that change an item's *result*: its resolved
    parameters (a tau or schedule constant that arrived with the tuning outputs),
    its seed and replicate count, the artifact's bytes, the simulation surface's source
    (`sim_surface_sha`), and whether snapshots were requested but never logged. Dynamics
    grid changes are not tracked.

    A record written before the surface was fingerprinted carries no ``sim_sha`` and is
    *not* called stale on that ground: the whole shipped tree predates the stamp, and
    turning 3,510 items stale would guarantee the flag gets routed around. Those records
    are surfaced instead, by `analyze_deployment.load_cells`, at analysis time.

    `ignore_sim_sha` (``--allow-stale-sim``) is the deliberate escape for the common
    benign case: `policy_table` is on the surface, so *registering a new policy* moves
    the fingerprint although no existing policy's episode can change. Without an escape
    that case costs a full re-run, and an unescapable guard is the kind that gets
    deleted. The item keeps the ``sim_sha`` it was produced under, so the analysis still
    sees -- and still refuses -- the mixed set.
    """
    if not record.get("parquet") or not Path(record["parquet"]).exists():
        return "parquet missing"
    if record.get("params") != item.params:
        return f"params changed {record.get('params')} -> {item.params}"
    if int(record.get("n_replicates", -1)) != int(item.spec.n_replicates):
        return f"n_replicates {record.get('n_replicates')} -> {item.spec.n_replicates}"
    if int(record.get("base_seed", -1)) != int(item.spec.base_seed):
        return f"base_seed {record.get('base_seed')} -> {item.spec.base_seed}"
    if int(record.get("horizon", -1)) != int(item.spec.horizon) or int(record.get("cap", -1)) != int(item.spec.cap):
        return "horizon/cap changed"
    if item.artifact_sha is not None and record.get("artifact_sha") != item.artifact_sha:
        return f"artifact bytes changed ({record.get('artifact_sha')} -> {item.artifact_sha})"
    recorded_sim = record.get("sim_sha")
    if not ignore_sim_sha and recorded_sim and item.sim_sha and recorded_sim != item.sim_sha:
        return f"simulation surface changed ({recorded_sim} -> {item.sim_sha})"
    if item.snapshots_path is not None and not record.get("snapshots"):
        return "snapshots requested but not logged before"
    return None


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
    done: dict[tuple[str, str, str], dict],
    git_sha: str,
    artifacts: dict[str, dict] | None = None,
    log_policies: set[str] | None = None,
    artifact_shas: dict[str, str] | None = None,
    excluded: dict[tuple[str, str, str], dict] | None = None,
    force_policies: set[str] | None = None,
    allow_stale_sim: bool = False,
) -> list[WorkItem]:
    """Resolve every (cell, policy) into a `WorkItem`, skipping those `done` describes.

    `done` maps an item key to its latest manifest record; the item is skipped only
    if `stale_reason` finds nothing changed, otherwise it is re-run and the reason
    logged. `excluded` (`active_exclusions`) is honoured on *every* run, not only
    ``--resume``: a deliberately dropped item stays dropped until ``--force-policies``
    names its policy, which is what nothing on disk recorded when ruling 19 rescheduled
    nine T=2000 log-e items that ruling 16 had dropped on purpose.
    Parameters are resolved here, in the parent, so every placeholder warning
    is printed once and the manifest can carry the values actually deployed.
    `artifacts` (variant -> loaded model) saves re-reading each joblib once per cell;
    `artifact_shas` (variant -> sha256 of the file) is stamped on every learned item.
    Snapshots are logged for the policies of `LOGGED_GROUPS` (or the `log_policies`
    subset of them).
    """
    comparators_dir = out_dir / "comparators"
    artifacts = artifacts or {}
    artifact_shas = artifact_shas or {}
    excluded = excluded or {}
    force_policies = force_policies or set()
    sim_sha = sim_surface_sha()
    items: list[WorkItem] = []
    for spec in cells:
        name = cell_name(spec)
        candidates: list[WorkItem] = []
        for policy in policies:
            exclusion = excluded.get((test, name, policy))
            if exclusion is not None:
                if policy not in force_policies:
                    log.info(
                        "%s/%s: skipped by an active manifest exclusion (%s); "
                        "--force-policies %s to run it anyway",
                        name, policy, exclusion.get("reason"), policy,
                    )
                    continue
                log.warning(
                    "%s/%s: --force-policies overrides the active exclusion (%s)",
                    name, policy, exclusion.get("reason"),
                )
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
            logged = log_states and group in LOGGED_GROUPS and (
                log_policies is None or policy in log_policies
            )
            item = WorkItem(
                test=test,
                cell=name,
                policy=policy,
                group=group,
                spec=spec,
                params=params,
                policy_seed=pt.policy_seed(policy),
                prefix_path="",  # filled once the cell is known to have work
                oracle_prior=(float("nan"), float("nan")),
                dynamics_grid=int(dynamics_grid),
                log_states=log_spec_for(spec) if logged else None,
                parquet_path=str(parquet),
                dynamics_path=str(dynamics),
                snapshots_path=str(snapshots) if logged else None,
                git_sha=git_sha,
                artifact_sha=artifact_shas.get(variant) if variant is not None else None,
                sim_sha=sim_sha,
            )
            record = done.get((test, name, policy))
            if record is not None:
                reason = stale_reason(record, item, ignore_sim_sha=allow_stale_sim)
                if reason is None:
                    continue
                log.warning("resume: re-running %s/%s: %s", name, policy, reason)
            candidates.append(item)
        if not candidates:
            continue
        prefix_path, prior = prepare_cell_constants(spec, comparators_dir)
        for item in candidates:
            items.append(replace(item, prefix_path=str(prefix_path), oracle_prior=prior))
    items.sort(key=_item_sort_key)
    return items


# ---- worker ----------------------------------------------------------------------------------

_TABLES: dict[tuple[int, float], CSTable] = {}
_PAIRWISE: dict[int, object] = {}
_ARTIFACTS: dict[str, dict] = {}
_THREAD_LIMITS = None
#: The pid of the process that started this worker's pool, or ``None`` when the item is
#: running inline in the parent. See `OrphanedWorker`.
_PARENT_PID: int | None = None


class OrphanedWorker(RuntimeError):
    """The pool parent died while this worker was still running an item.

    The parquet write happens in the worker and the manifest append in the parent
    (`run_items`), so a worker that outlives a killed parent writes a *complete*
    episode file that no manifest line will ever mention. Twice now (ledger rulings 17
    and 22) that left orphan T=2000 files behind a killed ``--resume``, and the next
    analysis would have silently pooled them: nine files lifted Test D's common support
    from 16 policies to 19, undoing a deliberate exclusion and moving a published
    headline. `analyze_deployment.reconcile_episodes` stops such a file from entering a
    number; this stops it from existing.

    The check is advisory. A worker already inside ``to_parquet`` when the parent dies
    still leaves a stray ``*.parquet.tmp``, which the reconciler ignores.
    """


def _worker_init(log_level: int, parent_pid: int | None = None) -> None:
    """Per-process setup: logging (spawned children start with none) and one BLAS /
    OpenMP thread each, since the parallelism is across items.

    `parent_pid` is the pool owner's pid, recorded so every output write can check that
    it is still this process's parent (`OrphanedWorker`).
    """
    global _THREAD_LIMITS, _PARENT_PID
    _PARENT_PID = None if parent_pid is None else int(parent_pid)
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


def _parent_is_alive() -> bool:
    """Whether the process that started this worker's pool is still this worker's parent.

    A killed parent leaves the worker reparented (to init, or to a subreaper), so
    ``os.getppid()`` no longer matches the pid recorded at `_worker_init`. Inline runs
    (``--workers 1``) never set it and are always "alive".
    """
    return _PARENT_PID is None or os.getppid() == _PARENT_PID


def _atomic_replace(tmp: Path, final: Path) -> None:
    """Publish `tmp` as `final`, unless this worker has been orphaned.

    The guard sits here rather than at the three call sites so no future output can be
    added without it. On a mismatch the temporary file is removed and nothing is
    published: an orphan that never lands is an orphan no analysis can pool.
    """
    if not _parent_is_alive():
        tmp.unlink(missing_ok=True)
        raise OrphanedWorker(
            f"parent {_PARENT_PID} is gone (ppid is now {os.getppid()}); "
            f"refusing to write {final}"
        )
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
        "sim_sha": item.sim_sha,
        "sim_versions": sim_surface_versions(),
        "env_id": spec.env_id,
        "family": family_of(spec.env_spec),
        "horizon": int(spec.horizon),
        "cap": int(spec.cap),
        "base_seed": int(spec.base_seed),
        "n_replicates": int(spec.n_replicates),
        "policy_seed": int(item.policy_seed),
        "params": item.params,
        "artifact_sha": item.artifact_sha,
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


#: Manifest record kinds. Every line written before exclusions existed carries no
#: ``kind`` at all and is a completion, so the field is always read through
#: `record_kind` rather than ``record["kind"]``.
KIND_COMPLETION = "completion"
KIND_EXCLUSION = "exclusion"


def record_kind(record: dict) -> str:
    """`KIND_COMPLETION` (the default for an unmarked line) or `KIND_EXCLUSION`."""
    return str(record.get("kind") or KIND_COMPLETION)


def exclusion_record(test: str, cell: str, policy: str, reason: str, sha: str) -> dict:
    """A manifest line recording that (cell, policy) is *deliberately* not run.

    Ruling 16 dropped three log-e policies from Test D's T=2000 cells because the exact
    chunked pairwise evaluator costs hours per item there. Nothing on disk said so, so
    the next ``--resume`` scheduled all nine again (ruling 19) and the workers that
    outlived the killed parent left nine orphan parquets behind (ruling 22). An
    exclusion line is the durable form of that decision: `latest_records` is
    latest-line-wins keyed on (test, cell, policy), so it supersedes any completion
    before it, `build_work` skips the item, and `analyze_deployment.reconcile_episodes`
    treats an excluded item that nonetheless has a file on disk as an orphan.

    `reason` is required and non-empty by the CLI, so an exclusion cannot quietly make a
    number look better: `analyze_deployment` surfaces every active one in
    ``strata_coverage_<test>.csv``.
    """
    reason = str(reason).strip()
    if not reason:
        raise ValueError("an exclusion needs a non-empty reason")
    return {
        "kind": KIND_EXCLUSION,
        "test": test,
        "cell": cell,
        "policy": policy,
        "reason": reason,
        "sha": sha,
        "at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


def append_records(manifest: Path, records: list[dict]) -> None:
    """Append manifest lines, the same way `run_items` appends completions."""
    if not records:
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def latest_records(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """The last manifest line per (test, cell, policy): a rerun supersedes its predecessor."""
    latest: dict[tuple[str, str, str], dict] = {}
    for rec in records:
        latest[(rec["test"], rec["cell"], rec["policy"])] = rec
    return latest


def active_exclusions(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """The items whose *latest* line is an exclusion, keyed like `latest_records`.

    A later completion (from ``--force-policies``) supersedes the exclusion, so an
    exclusion is active only while it is the last word on that item.
    """
    return {
        key: rec for key, rec in latest_records(records).items()
        if record_kind(rec) == KIND_EXCLUSION
    }


def completed_records(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Latest manifest line per item, for those whose parquet is still on disk.

    An active exclusion counts as done-with-no-file: ``--resume`` must not reschedule an
    item that was deliberately dropped just because no parquet exists for it.
    """
    out: dict[tuple[str, str, str], dict] = {}
    for key, rec in latest_records(records).items():
        if record_kind(rec) == KIND_EXCLUSION:
            out[key] = rec
        elif rec.get("parquet") and Path(rec["parquet"]).exists():
            out[key] = rec
    return out


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
        initargs=(logging.getLogger().level or logging.INFO, os.getpid()),
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

    The paired difference vs each reference is ``regret_policy - regret_ref`` over
    episodes aligned by ``episode`` (the CRN pairing): negative means the policy is
    better, and it is exactly zero for a reference against itself -- which is how a
    reader can tell the alignment is right. ``_win`` is the share of episodes the
    *policy* wins (lower regret; a tie counts half), i.e. `stats.paired_bootstrap`'s
    ``win_rate`` evaluated on ``-diff``.
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
                # Policy wins where diff < 0: the bootstrap's win_rate is P(diff > 0).
                row[prefix + "_win"] = boot["frac_negative"] + 0.5 * boot["frac_zero"]
                row[prefix + "_se"] = boot["se_paired"]
            rows.append(row)
    return rows


def write_summary(out_dir: Path, test: str, records: list[dict], *, n_boot: int) -> Path:
    """``summary_<test>.csv`` from every completed item in the manifest, cell by cell
    (so a Test-A run never holds 1,300 frames at once)."""
    by_cell: dict[str, dict[str, str]] = {}
    for rec in latest_records(records).values():
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


def _policy_list(spec: str | None) -> list[str]:
    """A comma-separated CLI policy list, empty when the option was not given."""
    if not spec:
        return []
    return [p.strip() for p in spec.split(",") if p.strip()]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True, choices=TESTS)
    p.add_argument("--n-replicates", type=int, default=None, help="episodes per cell (default: the test's)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--policies", default=None, help="comma-separated subset of policy_table.POLICIES")
    p.add_argument("--cells", default=None, help="manual cell list env:T:cap,... (overrides the test's grid)")
    p.add_argument("--dynamics-grid", type=int, default=50)
    p.add_argument("--log-states", action="store_true", help="log on-policy snapshots for learned policies")
    p.add_argument("--log-policies", default=None,
                   help="comma-separated subset of the learned/rule policies to log under --log-states "
                        "(default: all of them; ~7 GB of pickles for the whole Test-A table)")
    p.add_argument("--resume", action="store_true", help="skip items already in the manifest")
    p.add_argument("--exclude-policies", default=None,
                   help="comma-separated policies to record as DELIBERATELY not run on this "
                        "run's cells: one 'exclusion' manifest line each, which every later "
                        "run and the analysis honour. Requires --exclusion-reason.")
    p.add_argument("--exclusion-reason", default=None,
                   help="why the --exclude-policies items are dropped; recorded in the manifest "
                        "and surfaced in strata_coverage_<test>.csv")
    p.add_argument("--force-policies", default=None,
                   help="comma-separated policies to run even though an exclusion line is active")
    p.add_argument("--allow-stale-sim", action="store_true",
                   help="do not re-run an item just because sim_surface_sha() moved (e.g. a new "
                        "policy was registered); the item keeps the surface it was produced "
                        "under, so analyze_deployment still refuses the mixed set")
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
    git_sha = _git_sha()

    exclude_policies = _policy_list(args.exclude_policies)
    force_policies = set(_policy_list(args.force_policies))
    if exclude_policies:
        if not (args.exclusion_reason or "").strip():
            raise SystemExit("--exclude-policies requires a non-empty --exclusion-reason")
        pt.check_policies(exclude_policies)
        overlap = sorted(force_policies.intersection(exclude_policies))
        if overlap:
            raise SystemExit(f"--exclude-policies and --force-policies both name {overlap}")
        lines = [
            exclusion_record(test, cell_name(spec), policy, args.exclusion_reason, git_sha)
            for spec in cells
            for policy in exclude_policies
        ]
        append_records(manifest, lines)
        log.warning(
            "recorded %d exclusion lines in %s (%s): %s",
            len(lines), manifest.name, args.exclusion_reason, ", ".join(exclude_policies),
        )
    elif (args.exclusion_reason or "").strip():
        raise SystemExit("--exclusion-reason without --exclude-policies")
    if force_policies:
        pt.check_policies(sorted(force_policies))

    records_now = read_manifest(manifest)
    excluded = active_exclusions(records_now)
    if excluded:
        log.info("%d active exclusion(s) in %s", len(excluded), manifest.name)
    done: dict[tuple[str, str, str], dict] = {}
    if args.resume:
        done = completed_records(records_now)
        log.info("resume: %d items recorded complete in %s", len(done), manifest)
    elif manifest.exists():
        log.info("appending to existing %s; requested items are redone", manifest)

    log_policies = None
    if args.log_policies:
        log_policies = {p.strip() for p in args.log_policies.split(",") if p.strip()}
        pt.check_policies(sorted(log_policies))
    variants = {pt.variant_of(p) for p in policies} - {None}
    artifacts = {v: load_model(pt.artifact_path(v, models_dir)) for v in variants}
    artifact_shas = {v: file_sha256(pt.artifact_path(v, models_dir)) for v in variants}
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
        log_policies=log_policies,
        artifact_shas=artifact_shas,
        excluded=excluded,
        force_policies=force_policies,
        allow_stale_sim=bool(args.allow_stale_sim),
    )
    n_total = len(cells) * len(policies)
    n_skipped = n_total - len(items)
    log.info(
        "test %s: %d cells x %d policies = %d items, %d to run (%d workers, git %s, "
        "sim surface %s)",
        test, len(cells), len(policies), n_total, len(items), args.workers, git_sha,
        sim_surface_sha(),
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
             n_skipped, n_failed, wall)
    result = {
        "test": test,
        "n_cells": len(cells),
        "n_policies": len(policies),
        "n_items": n_total,
        "n_run": len(items) - n_failed,
        "n_skipped": n_skipped,
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
