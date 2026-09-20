#!/usr/bin/env python
"""M9: on-policy relabelling -- one policy-iteration step, the direct test of register #1.

The corpus labels are defined relative to the ``cp0`` continuation on states visited by
eight behavioural policies. The deployed learned policy visits its *own* states and
continues with *itself*, so nothing in the corpus certifies it there (failure-mode
register #1). This script asks the question directly, in three resumable steps:

* ``harvest`` -- run the deployed ``phi_k16`` (exactly as `policy_table` resolves it: the
  ``clock_quality_evidence_k16`` artifact at its validation threshold) on the eight Test-A
  environments x five horizons, on a fresh seed block (``cells.SPLIT_BASE["onpolicy"]``),
  and detach `Snapshot`s at eight times spread evenly in remaining budget. States at
  the live-arm cap are dropped, as the corpus dropped them: a forced SEARCH is
  undefined there -- and ``phi_k16`` is at the cap for most of every long-horizon
  episode, so ``--early-times`` can add pre-cap snapshot times (off by default).
* ``label`` -- oracle-label every snapshot at ``k = 16`` with **phi_k16 itself as the
  continuation policy** in both branches (`labeling.label_state`, oracle-prior
  recommender, ``target_se = 5e-4``, ``max_replicates = 2048``). The only things that
  differ from the corpus are therefore the state distribution and the continuation.
  Rows carry the corpus schema so the trainer can take them unchanged.
* ``train`` -- retrain the ``clock_quality_evidence`` / ``k=16`` / logit variant on the
  union (corpus + on-policy rows) and on the on-policy rows alone, through
  `train_policies.train_variant`, and write the coefficient / decision-agreement
  comparison between Phi and Phi'.

Three details of the labelling deserve stating. `label_state` runs the SEARCH branch to
the horizon and then the REFINE branch on one simulator, and materializes each batch at
its own replicate count; `_BranchResetPolicy` gives every branch a freshly reset
`ModelPolicy` sized for that batch (a `ModelPolicy` refuses a state whose ``M`` is not
its own, and its per-episode counters are cleared at every branch change as a matter of
isolation -- a standing commitment cannot in fact survive the horizon, since
``commit_left = min(k, T - t)``). `Simulator.step` never calls ``after_step``, so HISTORY
features would go stale inside a rollout; ``phi_k16`` reads none, which is asserted.
And the continuation's commitment cycle is phase-shifted relative to deployment: after
the 16 forced rounds the continuation makes a fresh decision at ``t + 16`` and commits
from there, whereas the deployed ``phi_k16`` decides at ``t = 2, 18, 34, ...`` and would
be at phase ``(t - 2) mod 16`` of a standing commitment when the label's window ends.
The corpus labels had no such effect because ``cp0`` is stateless; on-policy labels
are therefore "SEARCH/REFINE for 16 rounds, then phi_k16 restarted", not "then phi_k16
as it would have been". This is a property of the label, stated in the write-up.

Seeds: the on-policy block ``15_000_000 + cell_id * 1000`` sits above the test band and
below the corpus minimum; every seed the harvest and the labelling batches consume
(``base + 7919 m + 1_000_003 (b + 1)``) is checked against the corpus and the old
benchmark with `cells.assert_seed_disjointness` before any work starts.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cells  # noqa: E402
import policy_table as pt  # noqa: E402
from generate_states import snapshot_times_for  # noqa: E402
from label_states import _family_of, _write_chunk  # noqa: E402

from cold_start.growing.allocation import LUCB  # noqa: E402
from cold_start.growing.deploy.artifacts import load_model  # noqa: E402
from cold_start.growing.deploy.feature_groups import EVIDENCE_LOGE, HISTORY  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec, LogSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.pairwise_table import get_pairwise_table  # noqa: E402
from cold_start.growing.evidence import PairwiseEvidence  # noqa: E402
from cold_start.growing.features import extract_features  # noqa: E402
from cold_start.growing.labeling import Snapshot, label_state  # noqa: E402
from cold_start.growing.recommend import oracle_prior_from_reservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.schema import validate_columns  # noqa: E402
from cold_start.growing.search_policies import DecisionContext, SearchPolicy  # noqa: E402
from cold_start.growing.simulator import Simulator  # noqa: E402
from cold_start.growing.state import ForcedActionUnavailable, GrowingState  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

log = logging.getLogger("deploy.relabel")

DEFAULT_OUT_DIR = ROOT / "results" / "growing_bandits" / "deploy"
DEFAULT_CORPUS_DIR = ROOT / "data" / "oracle_labels"
LABELS_SUBDIR = "onpolicy_labels"
SNAPSHOTS_SUBDIR = "snapshots"
HARVEST_MANIFEST = "harvest_manifest.jsonl"
LABEL_MANIFEST = "manifest.jsonl"
DIAGNOSTICS_CSV = "onpolicy_label_diagnostics.csv"
MODEL_SHIFT_CSV = "onpolicy_model_shift.csv"

#: The deployed policy whose own states and continuation define the new labels.
SPLIT = "onpolicy"
PHI_POLICY = "phi_k16"
PHI_VARIANT = pt.variant_of(PHI_POLICY)
#: ``meta_policy`` of every on-policy row (the corpus has the eight behavioural names).
#: `train_policies.ONPOLICY_META_POLICY` is the same string: the trainer's `subset_mask`
#: selects on-policy rows by it. Spelled out here because the labelling workers must not
#: import the trainer (pandas and the corpus loader) just for one constant; `train`
#: asserts the two agree.
META_POLICY = "phi_k16_onpolicy"
META_ALLOCATION = "lucb"

DEFAULT_CAP = 64
DEFAULT_REPLICATES = 32
DEFAULT_SNAPSHOT_TIMES = 8
DEFAULT_ALPHA = 0.05
#: Salt for the per-cell snapshot-time generator (`generate_states.snapshot_times_for`).
SNAPSHOT_TIMES_SALT = 9

COMMIT_STEPS = 16
DEFAULT_TARGET_SE = 5e-4
DEFAULT_MAX_REPLICATES = 2048
DEFAULT_CHUNK_STATES = 64
DEFAULT_WORKERS = 12
MAX_TASKS_PER_CHILD = 4

#: The two retrained models: `train_policies.VARIANTS` entries whose ``subset`` carries the
#: ``onpolicy`` marker (``"union"`` / ``"only"``); the corpus-only trainer skips them and
#: `train` below reads their specs from that table rather than redefining them. Their
#: `policy_table.POLICIES` twins (`policy_table.ONPOLICY_POLICIES`) deploy on Test A.
VARIANT_UNION = "phi_k16_onpolicy_union"
VARIANT_ONLY = "phi_k16_onpolicy_only"
ONPOLICY_VARIANT_NAMES: tuple[str, ...] = (VARIANT_UNION, VARIANT_ONLY)
DEFAULT_SPLITS_UNION = 5
#: Eight on-policy environments -> four environment folds.
DEFAULT_SPLITS_ONLY = 4

#: Corpus rows generated by a deterministic growth schedule (`label_states.policy_by_name`):
#: the behavioural policies most like a deployed rule, the natural corpus comparison for
#: the on-policy label statistics. The all-policy corpus is reported alongside.
CORPUS_SCHEDULE_POLICIES: tuple[str, ...] = ("sqrt", "cbrt", "t23", "bracket")

# ---- seeds ---------------------------------------------------------------------------


def cell_key(env_id: str, horizon: int) -> str:
    return f"{env_id}_T{int(horizon)}"


def shard_id(env_id: str, horizon: int) -> str:
    return f"onpolicy_{cell_key(env_id, horizon)}"


def consumed_seeds(spec: CellSpec) -> set[int]:
    """Every `GrowingState.base_seed` this cell's harvest and labelling will touch.

    The harness detaches replicate ``m`` at ``base + 7919 m`` (the corpus's trajectory
    stride) and `labeling.label_state` materializes batch ``b`` of a snapshot at
    ``+ 1_000_003 (b + 1)``; both strides are the corpus's own, which is exactly why
    the whole set, not just the cell seed, has to be checked against the corpus.
    """
    seeds = {int(spec.base_seed)}
    for m in range(int(spec.n_replicates)):
        snapshot_seed = int(spec.base_seed) + cells.CORPUS_TRAJECTORY_STRIDE * m
        seeds.add(snapshot_seed)
        for b in range(cells.CORPUS_MAX_BATCHES):
            seeds.add(snapshot_seed + cells.CORPUS_BATCH_STRIDE * (b + 1))
    return seeds


def check_seeds(specs: Iterable[CellSpec]) -> None:
    """Every seed the cells will consume must be new to the corpus and the old benchmark."""
    seeds: set[int] = set()
    for spec in specs:
        seeds |= consumed_seeds(spec)
    cells.assert_seed_disjointness(seeds)


def onpolicy_cells(
    envs: Iterable[str], horizons: Iterable[int], *, cap: int, n_replicates: int, alpha: float
) -> list[CellSpec]:
    """The harvest grid on the on-policy split, seed-checked before anything runs."""
    specs = [
        cells.make_cell(SPLIT, env_id, int(T), int(cap), int(n_replicates), alpha=alpha)
        for env_id in envs
        for T in horizons
    ]
    check_seeds(specs)
    return specs


def harvested_cell(rec: dict, alpha: float) -> CellSpec:
    """Rebuild a harvest record's cell from the split, and refuse a record whose seed is
    not the one the split assigns (a manifest edited by hand, or a moved enumeration)."""
    spec = cells.make_cell(
        SPLIT, rec["env_id"], int(rec["horizon"]), int(rec["cap"]), int(rec["n_replicates"]), alpha=alpha
    )
    if int(spec.base_seed) != int(rec["base_seed"]):
        raise RuntimeError(
            f"harvest record {rec['cell']} carries seed {rec['base_seed']}, but the on-policy "
            f"split assigns {spec.base_seed}"
        )
    return spec


# ---- the deployed policy --------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_phi_artifact(artifact: dict) -> None:
    """The continuation runs inside `Simulator.run_to_horizon`, which never calls
    ``after_step``, so a model that reads HISTORY columns would see a frozen history.
    And the label is a 16-round commitment, so the policy's own commitment must match."""
    stale = [c for c in artifact["features"] if c in HISTORY]
    if stale:
        raise ValueError(
            f"{PHI_VARIANT} reads HISTORY features {stale}, which Simulator.step does not "
            "maintain; the on-policy continuation would be wrong"
        )
    if int(artifact["k"]) != COMMIT_STEPS:
        raise ValueError(f"{PHI_VARIANT} was trained at k={artifact['k']}, not {COMMIT_STEPS}")


def phi_params(
    horizon: int, *, models_dir: Path, thresholds: dict | None, artifact: dict
) -> dict:
    """`policy_table`'s own resolution of ``phi_k16`` at this horizon (artifact path, tau,
    tau source, k, mechanics), so the harvest deploys exactly what Test A deploys."""
    params = pt.resolve_params(
        PHI_POLICY, horizon, thresholds=thresholds, models_dir=models_dir, artifact=artifact
    )
    if params["per_step"] or params["affordability_guard"]:
        raise ValueError(f"{PHI_POLICY} is expected without per-step or guard mechanics")
    return params


def build_phi(
    params: dict, *, horizon: int, n_replicates: int, table, pairwise, artifact: dict
) -> SearchPolicy:
    return pt.build_policy(
        PHI_POLICY,
        params,
        horizon=horizon,
        n_replicates=n_replicates,
        table=table,
        pairwise=pairwise,
        artifact=artifact,
    )


def _pairwise_for(artifact: dict, horizon: int, cache_dir: Path | None):
    """The cached log-e grid, only if the model reads ``f_log_e_pair`` (explicit membership)."""
    if any(c in EVIDENCE_LOGE for c in artifact["features"]):
        return get_pairwise_table(int(horizon), cache_dir=cache_dir)
    return None


class _BranchResetPolicy(SearchPolicy):
    """A `ModelPolicy` continuation that starts afresh whenever the branch changes.

    `label_state` drives one simulator through the SEARCH branch to the horizon and
    then through the REFINE branch, and materializes each batch at its own replicate
    count. Two things follow. `ModelPolicy` is sized at construction and refuses a
    state of another ``M``, so the inner policy is rebuilt through `make_inner` when
    the replicate count changes. And `ModelPolicy` keeps per-episode state between
    steps (commitment counters, decision diagnostics, ``last_decision``); a standing
    commitment cannot actually outlive the SEARCH branch -- ``commit_left`` is
    ``min(k, T - t)`` and is zero at the horizon -- so resetting on every branch change
    is isolation rather than a fix for a known leak: each branch sees exactly the
    policy a fresh deployment would, and the diagnostics of one branch never mix
    into the other. The branch is identified by the `GrowingState` object itself; the
    wrapper keeps a reference to the last state it was asked about (holding it alive,
    so its identity cannot be recycled) and resets when a different object arrives.
    """

    name = "phi_k16_continuation"
    needs_evidence = False

    def __init__(self, make_inner: Callable[[int], SearchPolicy], rng=None) -> None:
        super().__init__(rng)
        self.make_inner = make_inner
        self.inner: SearchPolicy | None = None
        self._last_state: GrowingState | None = None
        self.n_resets = 0

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        if state is not self._last_state:
            if self.inner is None or getattr(self.inner, "M", state.M) != state.M:
                self.inner = self.make_inner(state.M)
            else:
                self.inner.reset()
            self._last_state = state
            self.n_resets += 1
        return self.inner.should_search(state, ctx)


# ---- harvest -------------------------------------------------------------------------


def harvest_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    """``(labels_dir, snapshots_dir, harvest_manifest)``."""
    labels_dir = Path(out_dir) / LABELS_SUBDIR
    return labels_dir, labels_dir / SNAPSHOTS_SUBDIR, labels_dir / HARVEST_MANIFEST


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


def _append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def latest_by(records: list[dict], key: str) -> dict[str, dict]:
    """The last manifest line per `key`: a rerun supersedes its predecessor."""
    latest: dict[str, dict] = {}
    for rec in records:
        latest[rec[key]] = rec
    return latest


def snapshot_times(spec: CellSpec, n_times: int, early_times: int = 0) -> tuple[int, ...]:
    """Times evenly spaced in remaining fraction, jittered per cell (as the corpus was).

    `early_times` adds that many log-spaced times in ``[3, cap - 2)``. With two warm-start
    arms ``K_t <= t``, so no episode can be at the cap before ``t = cap - 2`` and every
    such snapshot has a defined SEARCH counterfactual. It exists because the deployed
    ``phi_k16`` reaches the 64-arm cap in every T=1000 episode by ``t ~ 80`` (Test A),
    which leaves the evenly spaced schedule almost nothing to keep at long horizons.
    Off by default: the brief's schedule is the corpus's.
    """
    rng = np.random.default_rng([int(spec.base_seed), SNAPSHOT_TIMES_SALT])
    times = set(snapshot_times_for(int(spec.horizon), int(n_times), rng))
    if early_times > 0:
        lo, hi = 3, min(int(spec.cap) - 2, int(spec.horizon) - 1)
        if hi > lo:
            grid = np.geomspace(lo, hi, num=int(early_times) + 1)[:-1]
            times |= {int(t) for t in np.round(grid)}
    return tuple(sorted(times))


def harvest_cell(
    spec: CellSpec,
    *,
    params: dict,
    artifact: dict,
    table,
    pairwise,
    times: tuple[int, ...],
) -> tuple[list[Snapshot], int]:
    """Run ``phi_k16`` on one cell and return ``(kept snapshots, n_at_cap)``.

    Every replicate is detached at every time; a snapshot holding ``cap`` arms has no
    SEARCH counterfactual and is dropped here rather than labelled as ``A_t = 0``.
    """
    reservoir = build_reservoir(spec.env_spec)
    policy = build_phi(
        params,
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
        dynamics_grid=0,
        log_states=LogSpec(times=times, replicates=int(spec.n_replicates)),
        policy_seed=pt.policy_seed(PHI_POLICY),
    )
    counters = policy.counters()
    if counters["n_nonfinite_rows"] != 0:
        raise RuntimeError(
            f"{counters['n_nonfinite_rows']} non-finite feature rows while harvesting "
            f"{cell_key(spec.env_id, spec.horizon)} (parity bug; register #5)"
        )
    allowed = consumed_seeds(spec)
    kept: list[Snapshot] = []
    n_at_cap = 0
    for index, snap in enumerate(res.snapshots):
        if int(snap.base_seed) not in allowed:
            raise RuntimeError(
                f"snapshot seed {snap.base_seed} is outside the checked on-policy seed set"
            )
        if snap.k >= int(spec.cap):
            n_at_cap += 1
            continue
        snap.meta["state_index"] = index
        kept.append(snap)
    return kept, n_at_cap


def _write_pickle_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def harvest(
    out_dir: Path,
    *,
    models_dir: Path = pt.DEFAULT_MODELS_DIR,
    thresholds_path: Path = pt.DEFAULT_THRESHOLDS_PATH,
    envs: Iterable[str] = tuple(cells.MAIN_ENVS),
    horizons: Iterable[int] = cells.HORIZONS,
    n_replicates: int = DEFAULT_REPLICATES,
    n_times: int = DEFAULT_SNAPSHOT_TIMES,
    early_times: int = 0,
    cap: int = DEFAULT_CAP,
    alpha: float = DEFAULT_ALPHA,
    resume: bool = True,
    pairwise_cache_dir: Path | None = None,
) -> list[dict]:
    """Harvest every ``(env, T)`` cell into ``onpolicy_labels/snapshots/<env>_T<T>.pkl``.

    Runs inline: forty cells of 32 episodes are a couple of minutes, and the harvest is
    the one step that must not share cores with a labelling pool. Resumable through
    ``harvest_manifest.jsonl``: a cell is skipped only when its record still names the
    parameters and artifact bytes about to be used and its pickle exists.
    """
    out_dir = Path(out_dir)
    labels_dir, snap_dir, manifest = harvest_paths(out_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    specs = onpolicy_cells(envs, horizons, cap=cap, n_replicates=n_replicates, alpha=alpha)

    artifact_path = pt.artifact_path(PHI_VARIANT, models_dir)
    artifact = load_model(artifact_path)
    check_phi_artifact(artifact)
    artifact_sha = file_sha256(artifact_path)
    thresholds = pt.load_thresholds(thresholds_path)

    done = latest_by(read_manifest(manifest), "cell") if resume else {}
    records: list[dict] = []
    t_start = time.perf_counter()
    for i, spec in enumerate(specs, start=1):
        key = cell_key(spec.env_id, spec.horizon)
        params = phi_params(spec.horizon, models_dir=models_dir, thresholds=thresholds, artifact=artifact)
        times = snapshot_times(spec, n_times, early_times)
        pkl = snap_dir / f"{key}.pkl"
        rec = done.get(key)
        if (
            rec is not None
            and rec.get("params") == params
            and rec.get("artifact_sha") == artifact_sha
            and int(rec.get("base_seed", -1)) == int(spec.base_seed)
            and int(rec.get("n_replicates", -1)) == int(spec.n_replicates)
            and rec.get("times") == [int(t) for t in times]
            and Path(rec.get("pickle", "")).exists()
        ):
            records.append(rec)
            continue
        t0 = time.perf_counter()
        table = CSTable.load_or_build(spec.horizon, alpha)
        pairwise = _pairwise_for(artifact, spec.horizon, pairwise_cache_dir)
        kept, n_at_cap = harvest_cell(
            spec, params=params, artifact=artifact, table=table, pairwise=pairwise, times=times
        )
        _write_pickle_atomic(pkl, kept)
        rec = {
            "cell": key,
            "shard_id": shard_id(spec.env_id, spec.horizon),
            "env_id": spec.env_id,
            "family": _family_of(spec.env_id),
            "horizon": int(spec.horizon),
            "cap": int(spec.cap),
            "base_seed": int(spec.base_seed),
            "n_replicates": int(spec.n_replicates),
            "times": [int(t) for t in times],
            "n_times": int(n_times),
            "early_times": int(early_times),
            "policy": PHI_POLICY,
            "params": params,
            "artifact_sha": artifact_sha,
            "n_snapshots": len(kept),
            "n_at_cap": int(n_at_cap),
            "pickle": str(pkl),
            "pickle_sha": file_sha256(pkl),
            "seconds": round(time.perf_counter() - t0, 3),
            "finished_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        }
        _append_manifest(manifest, rec)
        records.append(rec)
        log.info(
            "[%d/%d] harvested %s: %d states kept, %d at cap, %.1fs (elapsed %.0fs)",
            i, len(specs), key, len(kept), n_at_cap, rec["seconds"], time.perf_counter() - t_start,
        )
    n_states = sum(int(r["n_snapshots"]) for r in records)
    n_cap = sum(int(r["n_at_cap"]) for r in records)
    log.info("harvest complete: %d cells, %d states (%d dropped at cap) -> %s", len(records), n_states, n_cap, snap_dir)
    return records


# ---- label ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelItem:
    """One chunk of one cell's snapshots; plain data, picklable, pure in its fields."""

    part: str
    shard_id: str
    env_id: str
    env_spec: dict
    horizon: int
    cap: int
    alpha: float
    pickle_path: str
    #: sha256 of the pickle bytes: the identity of the harvest these labels belong to.
    harvest_sha: str
    start: int
    stop: int
    n_at_cap: int
    params: dict
    artifact_path: str
    artifact_sha: str
    target_se: float
    max_replicates: int
    commit_steps: int
    pairwise_cache_dir: str | None

    @property
    def n_states(self) -> int:
        return int(self.stop - self.start)

    @property
    def weight(self) -> int:
        """Rollout cost scales with the horizon; used for the ETA, nothing else."""
        return self.n_states * int(self.horizon)


_TABLES: dict[tuple[int, float], CSTable] = {}
_PAIRWISE: dict[tuple[int, str | None], object] = {}
_ARTIFACTS: dict[str, dict] = {}


def _worker_init(log_level: int) -> None:
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s"
    )
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - sklearn depends on threadpoolctl
        pass


def _table(horizon: int, alpha: float) -> CSTable:
    key = (int(horizon), float(alpha))
    if key not in _TABLES:
        _TABLES[key] = CSTable.load_or_build(int(horizon), float(alpha))
    return _TABLES[key]


def _pairwise(artifact: dict, horizon: int, cache_dir: str | None):
    key = (int(horizon), cache_dir)
    if key not in _PAIRWISE:
        _PAIRWISE[key] = _pairwise_for(artifact, horizon, Path(cache_dir) if cache_dir else None)
    return _PAIRWISE[key]


def _artifact(path: str) -> dict:
    if path not in _ARTIFACTS:
        _ARTIFACTS[path] = load_model(path)
    return _ARTIFACTS[path]


def label_rows_for(
    snaps: list[Snapshot],
    *,
    horizon: int,
    cap: int,
    env_id: str,
    env_spec: dict,
    table,
    pairwise,
    artifact: dict,
    params: dict,
    target_se: float,
    max_replicates: int,
    commit_steps: int,
    shard: str,
    n_undefined_before: int = 0,
) -> tuple[list[dict], int]:
    """Feature + on-policy label rows for `snaps`; returns ``(rows, n_undefined)``.

    Features come from the SAME scalar call the corpus used (exact `PairwiseEvidence`,
    oracle columns from the true reservoir), so a row here is comparable column for
    column with a corpus row. The continuation in both branches is a fresh
    ``phi_k16`` per batch, wrapped in `_BranchResetPolicy`. ``meta_n_undefined_in_shard``
    is the corpus writer's running count, started at `n_undefined_before` (the states
    the harvest already dropped at the cap).
    """
    reservoir = build_reservoir(env_spec)
    prior = oracle_prior_from_reservoir(reservoir)
    scalar_pairwise = PairwiseEvidence()

    def make_inner(n_replicates: int) -> SearchPolicy:
        return build_phi(
            params,
            horizon=horizon,
            n_replicates=n_replicates,
            table=table,
            pairwise=pairwise,
            artifact=artifact,
        )

    def factory(offset: int) -> Simulator:
        return Simulator(
            table=table,
            reservoir=reservoir,
            allocation=LUCB(),
            search_policy=_BranchResetPolicy(make_inner),
            horizon=horizon,
            max_live_arms=cap,
        )

    rows: list[dict] = []
    n_undefined = int(n_undefined_before)
    for snap in snaps:
        if snap.k >= cap:
            n_undefined += 1
            continue
        row = extract_features(
            n=snap.n,
            successes=snap.successes,
            mu_true=snap.mu,
            t=snap.t,
            horizon=snap.horizon,
            table=table,
            pairwise=scalar_pairwise,
            reservoir=reservoir,
            history=snap.meta.get("history"),
        )
        try:
            lk = label_state(
                snap,
                factory,
                table,
                target_se=target_se,
                max_replicates=max_replicates,
                commit_steps=commit_steps,
                oracle_prior=prior,
                max_live_arms=cap,
            )
        except ForcedActionUnavailable:
            n_undefined += 1
            continue
        k = commit_steps
        row[f"label_A_k{k}"] = float(lk.advantage)
        row[f"label_se_k{k}"] = float(lk.se)
        row[f"label_M_k{k}"] = float(lk.n_replicates)
        row["label_A"] = float(lk.advantage)
        row["label_se"] = float(lk.se)
        row["label_se_unpaired"] = float(lk.se_unpaired)
        row["label_frac_identical"] = float(lk.frac_identical)
        row["label_M"] = float(lk.n_replicates)
        row["label_mean_search"] = float(lk.mean_search)
        row["label_mean_refine"] = float(lk.mean_refine)
        for dk, dv in lk.diagnostics.items():
            row[f"label_diag_{dk}"] = float(dv)
        row["meta_shard"] = shard
        row["meta_env"] = env_id
        row["meta_family"] = _family_of(env_id)
        row["meta_policy"] = META_POLICY
        row["meta_allocation"] = META_ALLOCATION
        row["meta_horizon"] = float(horizon)
        row["meta_state_index"] = float(snap.meta.get("state_index", -1))
        row["meta_n_undefined_in_shard"] = float(n_undefined)
        validate_columns(list(row))
        rows.append(row)
    return rows, n_undefined


def label_item(item: LabelItem) -> dict:
    """Label one chunk; returns the rows and the manifest record (parquet path unset)."""
    t0 = time.perf_counter()
    table = _table(item.horizon, item.alpha)
    artifact = _artifact(item.artifact_path)
    pairwise = _pairwise(artifact, item.horizon, item.pairwise_cache_dir)
    with open(item.pickle_path, "rb") as fh:
        snaps: list[Snapshot] = pickle.load(fh)
    chunk = snaps[item.start : item.stop]
    rows, n_undefined = label_rows_for(
        chunk,
        horizon=item.horizon,
        cap=item.cap,
        env_id=item.env_id,
        env_spec=item.env_spec,
        table=table,
        pairwise=pairwise,
        artifact=artifact,
        params=item.params,
        target_se=item.target_se,
        max_replicates=item.max_replicates,
        commit_steps=item.commit_steps,
        shard=item.shard_id,
        n_undefined_before=item.n_at_cap,
    )
    seconds = time.perf_counter() - t0
    return {
        "part": item.part,
        "shard_id": item.shard_id,
        "env_id": item.env_id,
        "horizon": int(item.horizon),
        "start": int(item.start),
        "stop": int(item.stop),
        "n_states": item.n_states,
        "n_rows": len(rows),
        "n_undefined": int(n_undefined - item.n_at_cap),
        "n_at_cap": int(item.n_at_cap),
        "seconds": round(seconds, 3),
        "seconds_per_state": round(seconds / max(item.n_states, 1), 3),
        "mean_M": float(np.mean([r["label_M"] for r in rows])) if rows else float("nan"),
        "params": item.params,
        "artifact_sha": item.artifact_sha,
        "harvest_sha": item.harvest_sha,
        "target_se": float(item.target_se),
        "max_replicates": int(item.max_replicates),
        "commit_steps": int(item.commit_steps),
        "rows": rows,
    }


def _label_item_safe(item: LabelItem) -> dict:
    try:
        rec = label_item(item)
        rec["ok"] = True
        return rec
    except Exception as exc:  # reported to the parent, which decides what to do
        return {
            "ok": False,
            "part": item.part,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def build_label_items(
    harvest_records: list[dict],
    *,
    chunk_states: int,
    limit_states: int | None,
    target_se: float,
    max_replicates: int,
    commit_steps: int,
    alpha: float,
    pairwise_cache_dir: Path | None,
) -> list[LabelItem]:
    """Chunk every harvested cell into `chunk_states`-state items, longest horizon first."""
    items: list[LabelItem] = []
    for rec in harvest_records:
        harvest_sha = file_sha256(rec["pickle"])
        if rec.get("pickle_sha") not in (None, harvest_sha):
            raise RuntimeError(
                f"snapshot pickle for {rec['cell']} changed since it was harvested "
                f"({rec['pickle_sha'][:12]} -> {harvest_sha[:12]}); re-harvest the cell"
            )
        n = int(rec["n_snapshots"])
        if limit_states is not None:
            n = min(n, int(limit_states))
        for c, start in enumerate(range(0, n, int(chunk_states))):
            stop = min(start + int(chunk_states), n)
            items.append(
                LabelItem(
                    part=f"{rec['shard_id']}_c{c:02d}",
                    shard_id=rec["shard_id"],
                    env_id=rec["env_id"],
                    env_spec=cells.ALL_ENVS[rec["env_id"]],
                    horizon=int(rec["horizon"]),
                    cap=int(rec["cap"]),
                    alpha=float(alpha),
                    pickle_path=rec["pickle"],
                    harvest_sha=harvest_sha,
                    start=start,
                    stop=stop,
                    n_at_cap=int(rec["n_at_cap"]),
                    params=dict(rec["params"]),
                    artifact_path=rec["params"]["artifact"],
                    artifact_sha=rec["artifact_sha"],
                    target_se=float(target_se),
                    max_replicates=int(max_replicates),
                    commit_steps=int(commit_steps),
                    pairwise_cache_dir=str(pairwise_cache_dir) if pairwise_cache_dir else None,
                )
            )
    items.sort(key=lambda it: (-it.horizon, it.part))
    return items


def _label_stale_reason(rec: dict, item: LabelItem, labels_dir: Path) -> str | None:
    """Why a manifest `rec` no longer describes `item`, or ``None`` if it still does.

    The harvest identity is checked by content: a re-harvest (different snapshot
    times, replicate count or policy) rewrites the cell's pickle, and a chunk of the old
    pickle must not stand in for the same slice of the new one even when the slice
    boundaries did not move.
    """
    if rec.get("n_rows", 0) > 0 and not (labels_dir / f"part-{item.part}.parquet").exists():
        return "parquet missing"
    if rec.get("harvest_sha") != item.harvest_sha:
        return "harvest changed (snapshot pickle differs)"
    if rec.get("params") != item.params or rec.get("artifact_sha") != item.artifact_sha:
        return "policy changed"
    for field in ("target_se", "max_replicates", "commit_steps", "start", "stop"):
        if rec.get(field) != getattr(item, field):
            return f"{field} changed"
    return None


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def label(
    out_dir: Path,
    *,
    models_dir: Path = pt.DEFAULT_MODELS_DIR,
    thresholds_path: Path = pt.DEFAULT_THRESHOLDS_PATH,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    workers: int = DEFAULT_WORKERS,
    target_se: float = DEFAULT_TARGET_SE,
    max_replicates: int = DEFAULT_MAX_REPLICATES,
    commit_steps: int = COMMIT_STEPS,
    chunk_states: int = DEFAULT_CHUNK_STATES,
    limit_states: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    resume: bool = True,
    pairwise_cache_dir: Path | None = None,
    write_diagnostics: bool = True,
) -> dict:
    """Label every harvested snapshot, chunk by chunk, through a spawn pool.

    Each chunk's parquet is written atomically by the parent (never by a worker) and
    its manifest line appended only afterwards, so a resumed run trusts a part only
    when the parquet is complete. The parameters the harvest recorded for ``phi_k16``
    are re-resolved here and must agree, or the states and the continuation would
    come from different policies. All memory-mapped tables are built in the parent
    before the pool exists (their atomic writers use a fixed temporary name).
    """
    out_dir = Path(out_dir)
    labels_dir, _, harvest_manifest = harvest_paths(out_dir)
    manifest = labels_dir / LABEL_MANIFEST
    harvest_records = list(latest_by(read_manifest(harvest_manifest), "cell").values())
    if not harvest_records:
        raise FileNotFoundError(f"no harvested cells in {harvest_manifest}; run `harvest` first")

    artifact_path = pt.artifact_path(PHI_VARIANT, models_dir)
    artifact = load_model(artifact_path)
    check_phi_artifact(artifact)
    artifact_sha = file_sha256(artifact_path)
    thresholds = pt.load_thresholds(thresholds_path)
    taus: set[float] = set()
    for rec in harvest_records:
        params = phi_params(int(rec["horizon"]), models_dir=models_dir, thresholds=thresholds, artifact=artifact)
        if rec["params"] != params or rec["artifact_sha"] != artifact_sha:
            raise RuntimeError(
                f"harvested cell {rec['cell']} used phi_k16 with {rec['params']} "
                f"(sha {rec['artifact_sha'][:12]}); now resolving to {params} "
                f"(sha {artifact_sha[:12]}). Re-harvest before labelling."
            )
        taus.add(float(params["tau"]))
    if len(taus) != 1:
        raise RuntimeError(f"phi_k16 resolved to several thresholds across horizons: {sorted(taus)}")
    tau = taus.pop()
    check_seeds(harvested_cell(rec, alpha) for rec in harvest_records)

    items = build_label_items(
        harvest_records,
        chunk_states=chunk_states,
        limit_states=limit_states,
        target_se=target_se,
        max_replicates=max_replicates,
        commit_steps=commit_steps,
        alpha=alpha,
        pairwise_cache_dir=pairwise_cache_dir,
    )
    done = latest_by(read_manifest(manifest), "part") if resume else {}
    todo: list[LabelItem] = []
    for item in items:
        rec = done.get(item.part)
        if rec is not None:
            reason = _label_stale_reason(rec, item, labels_dir)
            if reason is None:
                continue
            log.warning("resume: re-running %s: %s", item.part, reason)
        todo.append(item)
    log.info("%d parts total, %d already complete, %d to run (workers=%d)", len(items), len(items) - len(todo), len(todo), workers)

    if todo:
        for T in sorted({it.horizon for it in todo}):
            t0 = time.perf_counter()
            CSTable.load_or_build(T, alpha)
            _pairwise_for(artifact, T, pairwise_cache_dir)
            log.info("tables for T=%d ready (%.1fs)", T, time.perf_counter() - t0)

    total_weight = sum(it.weight for it in todo)
    done_weight = 0
    n_done = 0
    n_rows = 0
    failures: list[dict] = []
    t_start = time.perf_counter()
    labels_dir.mkdir(parents=True, exist_ok=True)

    def handle(rec: dict, item: LabelItem) -> None:
        nonlocal done_weight, n_done, n_rows
        n_done += 1
        done_weight += item.weight
        elapsed = time.perf_counter() - t_start
        if not rec["ok"]:
            failures.append(rec)
            log.error("[%d/%d] %s FAILED: %s\n%s", n_done, len(todo), rec["part"], rec["error"], rec.get("traceback", ""))
            return
        rows = rec.pop("rows")
        if rows:
            path = _write_chunk(rows, labels_dir, rec["part"])
            rec["parquet"] = str(path)
        else:
            rec["parquet"] = None
        rec["finished_at"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        line = {k: v for k, v in rec.items() if k != "ok"}
        _append_manifest(manifest, line)
        n_rows += rec["n_rows"]
        eta = elapsed * (total_weight - done_weight) / max(done_weight, 1)
        log.info(
            "[%d/%d] %s  %d rows  %.1fs (%.2fs/state, mean M %.0f)  rows so far %d  elapsed %s  ETA %s",
            n_done, len(todo), rec["part"], rec["n_rows"], rec["seconds"], rec["seconds_per_state"],
            rec["mean_M"], n_rows, _format_eta(elapsed), _format_eta(eta),
        )

    if workers <= 1:
        for item in todo:
            handle(_label_item_safe(item), item)
    elif todo:
        by_part = {it.part: it for it in todo}
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=int(workers),
            initializer=_worker_init,
            initargs=(logging.getLogger().level or logging.INFO,),
            maxtasksperchild=MAX_TASKS_PER_CHILD,
        ) as pool:
            for rec in pool.imap_unordered(_label_item_safe, todo, chunksize=1):
                handle(rec, by_part[rec["part"]])

    result = {
        "n_parts": len(items),
        "n_run": len(todo),
        "n_failed": len(failures),
        "failures": failures,
        "n_rows": n_rows,
        "labels_dir": labels_dir,
        "diagnostics": None,
    }
    finished = latest_by(read_manifest(manifest), "part")
    complete = not failures and all(it.part in finished for it in items)
    if write_diagnostics and complete:
        result["diagnostics"] = write_label_diagnostics(
            out_dir, labels_dir=labels_dir, corpus_dir=corpus_dir, artifact=artifact, tau=tau,
            harvest_records=harvest_records,
        )
    elif write_diagnostics:
        log.warning("diagnostics not written: %d failures, run incomplete", len(failures))
    return result


# ---- rows on disk --------------------------------------------------------------------


def load_onpolicy_rows(labels_dir: Path):
    """Every on-policy part as one DataFrame (sorted by file name, so stable)."""
    import pandas as pd

    files = sorted(Path(labels_dir).glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no on-policy parts under {labels_dir}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def _phi_scores(artifact: dict, df) -> np.ndarray:
    """``P(SEARCH)`` of a saved model on scalar-feature rows (the trainer's matrix convention)."""
    import train_policies as tp

    X = tp.design_matrix(df, tuple(artifact["features"]))
    return np.asarray(artifact["pipeline"].predict_proba(X)[:, 1], dtype=np.float64)


# ---- label diagnostics ---------------------------------------------------------------


def _label_stats(a: np.ndarray, se: np.ndarray, p: np.ndarray, tau: float) -> dict:
    """Sign / tie / magnitude / precision summary of k=16 labels, plus Phi's agreement."""
    import train_policies as tp

    decided = a != 0.0
    y = a > 0.0
    out = {
        "n_states": int(a.size),
        "n_decided": int(decided.sum()),
        "p_search_given_decided": float(y[decided].mean()) if decided.any() else float("nan"),
        "tie_frac": float((~decided).mean()) if a.size else float("nan"),
        "mean_abs_A": float(np.abs(a).mean()) if a.size else float("nan"),
        "mean_se": float(se.mean()) if se.size else float("nan"),
        "phi_search_rate": float((p > tau).mean()) if p.size else float("nan"),
        "phi_agree_frac": (
            float(((p[decided] > tau) == y[decided]).mean()) if decided.any() else float("nan")
        ),
        "phi_bal_acc": tp.balanced_accuracy(y[decided], p[decided] > tau) if decided.any() else float("nan"),
        "phi_auc": tp.auc(y[decided], p[decided]) if decided.any() else float("nan"),
    }
    return out


def write_label_diagnostics(
    out_dir: Path,
    *,
    labels_dir: Path,
    corpus_dir: Path,
    artifact: dict,
    tau: float,
    harvest_records: list[dict],
    max_shards: int | None = None,
):
    """``onpolicy_label_diagnostics.csv``: on-policy label statistics next to the corpus's.

    One row per ``(scope, horizon)``: the on-policy rows (``onpolicy``), the corpus's
    k=16 labels under the schedule policies (``corpus_schedule``: ``meta_policy`` in
    `CORPUS_SCHEDULE_POLICIES` = ``sqrt``, ``cbrt``, ``t23``, ``bracket`` -- the
    deterministic growth schedules; ``epsilon``, the randomised schedule, is excluded
    along with the Bernoulli and uniform policies) and under every policy
    (``corpus_all``), each per horizon and pooled. ``phi_*`` columns score the deployed ``phi_k16`` at the
    threshold it was harvested with (``tau``) against the label sign of the same rows:
    on the on-policy scope that is the fraction of its own states at which its
    decision agrees with the label its own continuation defines. The corpus's
    ``n_undefined`` is recovered from the per-shard running count the corpus writer
    stored (its maximum per shard), so it is a lower bound.
    """
    import pandas as pd
    from corpus import load_corpus

    onpol = load_onpolicy_rows(labels_dir)
    corpus = load_corpus(corpus_dir, max_shards=max_shards)
    n_at_cap: dict[int, int] = {}
    for rec in harvest_records:
        n_at_cap[int(rec["horizon"])] = n_at_cap.get(int(rec["horizon"]), 0) + int(rec["n_at_cap"])
    corpus_pol = corpus["meta_policy"].to_numpy().astype(str)
    scopes = [
        ("onpolicy", onpol, np.ones(len(onpol), dtype=bool)),
        ("corpus_schedule", corpus, np.isin(corpus_pol, CORPUS_SCHEDULE_POLICIES)),
        ("corpus_all", corpus, np.ones(len(corpus), dtype=bool)),
    ]
    rows: list[dict] = []
    for scope, df, mask in scopes:
        a = df["label_A_k16"].to_numpy(dtype=np.float64)
        se = df["label_se_k16"].to_numpy(dtype=np.float64)
        p = _phi_scores(artifact, df)
        T_all = df["meta_horizon"].to_numpy(dtype=np.float64)
        undefined_by_T: dict[float, int] = {}
        if scope == "onpolicy":
            undefined_by_T = {float(T): n for T, n in n_at_cap.items()}
        else:
            per_shard = (
                df.loc[mask, ["meta_shard", "meta_horizon", "meta_n_undefined_in_shard"]]
                .groupby(["meta_shard", "meta_horizon"], sort=False)["meta_n_undefined_in_shard"]
                .max()
            )
            for (_, T), n in per_shard.items():
                undefined_by_T[float(T)] = undefined_by_T.get(float(T), 0) + int(n)
        horizons = sorted(set(np.unique(T_all[mask]).tolist()) | set(undefined_by_T))
        for T in ["all", *horizons]:
            m = mask if T == "all" else (mask & (T_all == T))
            stats = _label_stats(a[m], se[m], p[m], tau)
            n_undef = sum(undefined_by_T.values()) if T == "all" else undefined_by_T.get(T, 0)
            rows.append({
                "scope": scope,
                "horizon": "all" if T == "all" else int(T),
                "n_undefined": int(n_undef),
                "tau": float(tau),
                **stats,
            })
    frame = pd.DataFrame(rows)
    path = Path(out_dir) / DIAGNOSTICS_CSV
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log.info("label diagnostics -> %s", path)
    return frame


# ---- train ---------------------------------------------------------------------------


def _metric_base(name: str, variant: dict) -> dict:
    """The identity columns `train_policies.train_variant` puts on every metric row."""
    import train_policies as tp

    return {
        "variant": name,
        "k": int(variant["k"]),
        "estimator": variant["estimator"],
        "feature_set": variant["feature_set"],
        "n_features": len(tp.feature_list(variant)),
        "row_filter": variant["row_filter"],
        "subset": json.dumps(variant["subset"], sort_keys=True),
    }


def _oof_scores_by_env(
    tp, frame, variant: dict, n_splits: int, n_jobs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(score, fold, y, eligible)``: the trainer's environment-fold OOF pass on `frame`."""
    features = tp.feature_list(variant)
    X = tp.design_matrix(frame, features)
    y = frame["label_A_k16"].to_numpy(dtype=np.float64) > 0
    eligible, universe = tp.eligible_masks(frame, variant)
    fold = tp.fold_ids_by_group(frame["meta_env"].to_numpy().astype(str), universe, n_splits)
    score = tp.oof_scores(variant["estimator"], variant["feature_set"], X, y, None, fold, eligible, n_jobs)
    return score, fold, y, eligible


def _coef_table(artifact: dict) -> dict[str, float]:
    """Standardized coefficients of a ``StandardScaler -> LogisticRegression`` pipeline."""
    steps = getattr(artifact["pipeline"], "named_steps", {})
    logit = steps.get("logisticregression")
    if logit is None:
        return {}
    coef = {f: float(c) for f, c in zip(artifact["features"], logit.coef_[0], strict=True)}
    coef["intercept"] = float(logit.intercept_[0])
    return coef


def model_shift_rows(phi: dict, union: dict, only: dict, frames: dict[str, object]) -> list[dict]:
    """Coefficient shift Phi -> Phi' and decision agreement at each model's ``tau_off``.

    ``section="coef"`` rows hold one feature (or the intercept) per row; ``"agreement"``
    rows hold the fraction of rows on which Phi' makes the same SEARCH/REFINE decision
    as Phi, per row set; ``"search_rate"`` the SEARCH share of each model there; and
    ``"tau"`` the thresholds used.
    """
    models = {"phi": phi, "union": union, "only": only}
    coefs = {k: _coef_table(v) for k, v in models.items()}
    names: list[str] = []
    for c in coefs.values():
        for f in c:
            if f not in names:
                names.append(f)
    rows: list[dict] = []
    for f in names:
        vals = {k: coefs[k].get(f, float("nan")) for k in models}
        rows.append({
            "section": "coef",
            "key": f,
            "n_rows": float("nan"),
            **vals,
            "delta_union": vals["union"] - vals["phi"],
            "delta_only": vals["only"] - vals["phi"],
        })
    rows.append({
        "section": "tau",
        "key": "tau_off",
        "n_rows": float("nan"),
        **{k: float(v["tau"]) for k, v in models.items()},
        "delta_union": float("nan"),
        "delta_only": float("nan"),
    })
    for set_name, frame in frames.items():
        decisions = {k: _phi_scores(v, frame) > float(v["tau"]) for k, v in models.items()}
        rows.append({
            "section": "agreement",
            "key": set_name,
            "n_rows": float(len(frame)),
            "phi": 1.0,
            "union": float((decisions["union"] == decisions["phi"]).mean()),
            "only": float((decisions["only"] == decisions["phi"]).mean()),
            "delta_union": float("nan"),
            "delta_only": float("nan"),
        })
        rates = {k: float(d.mean()) for k, d in decisions.items()}
        rows.append({
            "section": "search_rate",
            "key": set_name,
            "n_rows": float(len(frame)),
            **rates,
            "delta_union": rates["union"] - rates["phi"],
            "delta_only": rates["only"] - rates["phi"],
        })
    return rows


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def train(
    out_dir: Path,
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    labels_dir: Path | None = None,
    models_dir: Path | None = None,
    n_splits_union: int = DEFAULT_SPLITS_UNION,
    n_splits_only: int = DEFAULT_SPLITS_ONLY,
    n_jobs: int | None = None,
    max_shards: int | None = None,
) -> dict:
    """Phi' on the union and on the on-policy rows only; metrics merged into the trainer's tables.

    The two variants are the registered `train_policies.VARIANTS` entries carrying the
    ``onpolicy`` subset marker; both go through `train_policies.train_variant` on the
    SAME frame (corpus rows plus on-policy rows), whose `subset_mask` keeps every row
    for ``"union"`` and the on-policy rows for ``"only"`` -- so the trainer's own
    machinery does the environment folds, the tau selection, the refit, the artifact,
    and (for ``"only"``) the ``excluded_rows`` transfer to the corpus rows it never
    saw. One extra grouping is added for the union model: its environment-fold OOF
    scores restricted to the on-policy rows and to the corpus rows
    (``onpolicy_rows_oof``, ``corpus_rows_oof``), which is where a policy-iteration
    step should show first.
    """
    import pandas as pd
    import train_policies as tp
    from corpus import load_corpus

    if tp.ONPOLICY_META_POLICY != META_POLICY:
        raise RuntimeError(
            f"train_policies.ONPOLICY_META_POLICY={tp.ONPOLICY_META_POLICY!r} disagrees with "
            f"META_POLICY={META_POLICY!r}"
        )
    specs = {name: tp.VARIANTS[name] for name in ONPOLICY_VARIANT_NAMES}
    for name, spec in specs.items():
        if not tp.requires_onpolicy_rows(spec) or spec["k"] != COMMIT_STEPS:
            raise RuntimeError(f"{name} is not registered as an on-policy k={COMMIT_STEPS} variant: {spec}")

    out_dir = Path(out_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else harvest_paths(out_dir)[0]
    models_dir = Path(models_dir) if models_dir is not None else out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = int(n_jobs) if n_jobs is not None else min(n_splits_union, os.cpu_count() or 1)

    t0 = time.perf_counter()
    corpus = load_corpus(corpus_dir, max_shards=max_shards).drop(columns=["meta_trajectory"])
    onpol = load_onpolicy_rows(labels_dir)
    if (onpol["meta_policy"].astype(str) != META_POLICY).any():
        raise ValueError(f"on-policy rows must all carry meta_policy={META_POLICY!r}")
    union = pd.concat([corpus, onpol.reindex(columns=corpus.columns)], ignore_index=True)
    is_onpol = (union["meta_policy"].astype(str) == META_POLICY).to_numpy()
    log.info(
        "corpus %d rows + on-policy %d rows -> union %d rows [%.1fs]",
        len(corpus), len(onpol), len(union), time.perf_counter() - t0,
    )

    phi_path = pt.artifact_path(PHI_VARIANT, models_dir)
    phi = load_model(phi_path)
    provenance = {
        "corpus": str(corpus_dir),
        "onpolicy_labels": str(labels_dir),
        "n_corpus_rows": int(len(corpus)),
        "n_onpolicy_rows": int(len(onpol)),
        "phi_variant": PHI_VARIANT,
        "phi_artifact_sha": file_sha256(phi_path),
        "quick": max_shards is not None,
        "trained_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }

    metric_rows: list[dict] = []
    artifacts: dict[str, dict] = {}
    for name, n_splits in ((VARIANT_UNION, n_splits_union), (VARIANT_ONLY, n_splits_only)):
        t1 = time.perf_counter()
        rows, meta = tp.train_variant(name, specs[name], union, out_dir, n_splits, n_jobs, provenance)
        artifacts[name] = load_model(out_dir / "models" / f"{name}.joblib")
        metric_rows.extend(rows)
        env = meta["metrics"]["meta_env"]
        log.info(
            "%s: n=%d  AUC %.4f  bal@0.5 %.4f  tau_off %.2f -> %.4f  [%.0fs]",
            name, meta["n_rows"], env["auc"], env["bal_acc_05"], meta["tau_off"],
            env["bal_acc_tau_off"], time.perf_counter() - t1,
        )
        if "excluded_rows" in meta["metrics"]:
            ex = meta["metrics"]["excluded_rows"]
            log.info("%s: transfer to the corpus rows: AUC %.4f  bal@tau %.4f  n=%d",
                     name, ex["auc"], ex["bal_acc_tau_off"], ex["n_rows"])

    # Union model: where do its out-of-fold scores land on the two row populations?
    spec = specs[VARIANT_UNION]
    score, fold, y, _ = _oof_scores_by_env(tp, union, spec, n_splits_union, n_jobs)
    base = _metric_base(VARIANT_UNION, spec)
    n_groups = int(union["meta_env"].nunique())
    n_folds = int(len(np.unique(fold[fold >= 0])))
    for grouping, m in (("onpolicy_rows_oof", is_onpol), ("corpus_rows_oof", ~is_onpol)):
        metrics = tp.pooled_and_fold_metrics(y[m], score[m], fold[m], None)
        metric_rows.append({**base, "grouping": grouping, **metrics,
                            "n_groups": n_groups, "n_splits": n_folds, "n_cells": np.nan})

    metrics_path = out_dir / "offline_metrics.csv"
    fresh = pd.DataFrame(metric_rows)
    if metrics_path.exists():
        existing = list(pd.read_csv(metrics_path, nrows=0).columns)
        fresh = fresh.reindex(columns=existing + [c for c in fresh.columns if c not in existing])
    merged = tp.merge_metrics(metrics_path, fresh, list(ONPOLICY_VARIANT_NAMES))
    tmp = metrics_path.with_suffix(".csv.tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, metrics_path)

    variants_path = out_dir / "variants.json"
    variants_json = json.loads(variants_path.read_text()) if variants_path.exists() else {}
    for name, spec in specs.items():
        variants_json[name] = {**spec, "features": list(tp.feature_list(spec))}
    _write_json_atomic(variants_path, variants_json)

    shift = pd.DataFrame(model_shift_rows(
        phi, artifacts[VARIANT_UNION], artifacts[VARIANT_ONLY],
        {"corpus_rows": corpus, "onpolicy_rows": onpol},
    ))
    shift_path = out_dir / MODEL_SHIFT_CSV
    tmp = shift_path.with_suffix(".csv.tmp")
    shift.to_csv(tmp, index=False)
    os.replace(tmp, shift_path)
    log.info("metrics -> %s; variants -> %s; model shift -> %s [%.0fs total]",
             metrics_path, variants_path, shift_path, time.perf_counter() - t0)
    return {
        "artifacts": artifacts,
        "metrics": merged,
        "model_shift": shift,
        "n_corpus_rows": int(len(corpus)),
        "n_onpolicy_rows": int(len(onpol)),
    }


# ---- CLI -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--models-dir", type=Path, default=pt.DEFAULT_MODELS_DIR)
    p.add_argument("--thresholds", type=Path, default=pt.DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    h = sub.add_parser("harvest", help="on-policy states from the deployed phi_k16")
    h.add_argument("--envs", default="main", help="cells.env_ids_for selector or comma list")
    h.add_argument("--horizons", type=int, nargs="+", default=list(cells.HORIZONS))
    h.add_argument("--n-replicates", type=int, default=DEFAULT_REPLICATES)
    h.add_argument("--n-times", type=int, default=DEFAULT_SNAPSHOT_TIMES)
    h.add_argument("--early-times", type=int, default=0,
                   help="extra log-spaced snapshot times before t = cap - 2 (pre-cap states)")
    h.add_argument("--cap", type=int, default=DEFAULT_CAP)
    h.add_argument("--no-resume", action="store_true")

    lab = sub.add_parser("label", help="k=16 labels with phi_k16 as the continuation")
    lab.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    lab.add_argument("--target-se", type=float, default=DEFAULT_TARGET_SE)
    lab.add_argument("--max-replicates", type=int, default=DEFAULT_MAX_REPLICATES)
    lab.add_argument("--chunk-states", type=int, default=DEFAULT_CHUNK_STATES)
    lab.add_argument("--limit-states", type=int, default=None, help="per cell; smoke-test cap")
    lab.add_argument("--no-resume", action="store_true")
    lab.add_argument("--skip-diagnostics", action="store_true")

    tr = sub.add_parser("train", help="phi' on the union and on the on-policy rows only")
    tr.add_argument("--n-splits-union", type=int, default=DEFAULT_SPLITS_UNION)
    tr.add_argument("--n-splits-only", type=int, default=DEFAULT_SPLITS_ONLY)
    tr.add_argument("--n-jobs", type=int, default=None)
    tr.add_argument("--max-shards", type=int, default=None, help="corpus shards; smoke-test cap")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "harvest":
        records = harvest(
            args.out_dir,
            models_dir=args.models_dir,
            thresholds_path=args.thresholds,
            envs=cells.env_ids_for(args.envs),
            horizons=args.horizons,
            n_replicates=args.n_replicates,
            n_times=args.n_times,
            early_times=args.early_times,
            cap=args.cap,
            alpha=args.alpha,
            resume=not args.no_resume,
        )
        return {"records": records}
    if args.command == "label":
        return label(
            args.out_dir,
            models_dir=args.models_dir,
            thresholds_path=args.thresholds,
            corpus_dir=args.corpus,
            workers=args.workers,
            target_se=args.target_se,
            max_replicates=args.max_replicates,
            chunk_states=args.chunk_states,
            limit_states=args.limit_states,
            alpha=args.alpha,
            resume=not args.no_resume,
            write_diagnostics=not args.skip_diagnostics,
        )
    return train(
        args.out_dir,
        corpus_dir=args.corpus,
        models_dir=args.models_dir,
        n_splits_union=args.n_splits_union,
        n_splits_only=args.n_splits_only,
        n_jobs=args.n_jobs,
        max_shards=args.max_shards,
    )


if __name__ == "__main__":
    main()
