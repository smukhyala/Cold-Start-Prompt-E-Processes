"""Environments, horizons, caps and seeds of the deployment study -- the single source of truth.

Every run script (tuning, threshold selection, deployment) builds its `CellSpec`s here
so that three things hold by construction rather than by convention:

* **One seed per cell, shared by every policy.** `base_seed(split, env, T, cap)` is a
  pure function of the cell, so the CRN pairing in `harness.run_cell` extends across
  scripts: a policy tuned in `tune_baselines.py` and re-run in `run_deployment.py` on
  the same split sees the same episodes.
* **Splits never overlap.** Tuning (schedule constants), validation (thresholds, model
  selection), test and on-policy (M9 relabelling) seeds live in reserved bands
  `SPLIT_BASE[split] + cell_id * 1000` that are disjoint from each other, from the
  corpus (`[20.26M, 343.4M]`) and from the old benchmark (`[4.5k, 14.1k]`) --
  failure-mode register #9, #10.
* **The guard is computed, not remembered.** `assert_seed_disjointness` rebuilds the
  corpus seed set from `label_states.build_shards` with the production arguments of
  RUNBOOK.md section 1 and the old benchmark's `4242 + crc32(name) % 10000`, so a later
  change to either enumeration is caught at the next startup instead of trusted.

`cell_id` enumerates the plan's grid (33 environments sorted by id x `HORIZONS` x
`CAPS`) first, so those ids are stable, appends the Test-D extrapolation horizon
(`EXTRA_HORIZONS`) as a second block after it, and appends the cap-128 sweep
(`EXTRA_CAPS`, at the main `HORIZONS` only) as a third block after that -- each
appended block leaves every earlier id, and therefore every earlier `base_seed`,
untouched.
"""

from __future__ import annotations

import argparse
import functools
import sys
import zlib
from collections.abc import Iterable
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import label_states  # noqa: E402

from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

# ---- environments ----------------------------------------------------------------

#: Test A, in-distribution main panel (DEPLOYMENT_PLAN.md "Environments"): four Beta and
#: four tail reservoirs, none among the six degenerate tail combinations.
MAIN_ENV_IDS: tuple[str, ...] = (
    "beta_good_common",
    "beta_rare_excellent",
    "beta_mostly_mediocre",
    "beta_skewed",
    "tail_b0.5_mu1.0_c1.0",
    "tail_b2.0_mu1.0_c1.0",
    "tail_b8.0_mu1.0_c1.0",
    "tail_b1.0_mu0.9_c2.0",
)


def _constructible(spec: dict) -> bool:
    """`label_states._is_constructible`: the corpus dropped the reservoirs the validator rejects."""
    try:
        build_reservoir(spec)
    except Exception:
        return False
    return True


def _by_id(pairs: Iterable[tuple[str, dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for env_id, spec in pairs:
        if env_id in out:
            raise ValueError(f"duplicate environment id {env_id!r}")
        out[env_id] = spec
    return out


#: The 30 corpus environments (FAMILY_A + FAMILY_B minus the 6 degenerate tail combos).
ALL_CORPUS_ENVS: dict[str, dict] = {
    env_id: spec
    for env_id, spec in _by_id(label_states.FAMILY_A + label_states.FAMILY_B).items()
    if _constructible(spec)
}
#: The 3 mixture presets (FAMILY_C): never trained on, Test C's held-out family.
HELDOUT_ENVS: dict[str, dict] = _by_id(label_states.FAMILY_C)
#: The 8 Test-A environments, specs copied from `label_states` by id.
MAIN_ENVS: dict[str, dict] = {env_id: ALL_CORPUS_ENVS[env_id] for env_id in MAIN_ENV_IDS}
#: Every environment a cell may name, corpus and held-out alike.
ALL_ENVS: dict[str, dict] = {**ALL_CORPUS_ENVS, **HELDOUT_ENVS}

if len(ALL_CORPUS_ENVS) != 30 or len(HELDOUT_ENVS) != 3:
    raise RuntimeError(
        f"expected 30 corpus + 3 held-out environments; got {len(ALL_CORPUS_ENVS)} + "
        f"{len(HELDOUT_ENVS)} (label_states changed?)"
    )

ENV_SELECTORS: tuple[str, ...] = ("main", "heldout", "corpus", "all")


def env_ids_for(selector: str) -> tuple[str, ...]:
    """Resolve a CLI environment selector: a name in `ENV_SELECTORS` or a comma list of ids."""
    if selector == "main":
        return tuple(MAIN_ENVS)
    if selector == "heldout":
        return tuple(HELDOUT_ENVS)
    if selector == "corpus":
        return tuple(ALL_CORPUS_ENVS)
    if selector == "all":
        return tuple(ALL_ENVS)
    ids = tuple(part.strip() for part in selector.split(",") if part.strip())
    unknown = [env_id for env_id in ids if env_id not in ALL_ENVS]
    if not ids or unknown:
        raise KeyError(
            f"unknown environment selector {selector!r}: expected one of {ENV_SELECTORS} "
            f"or a comma list of ids; unknown ids={unknown}"
        )
    return ids


# ---- cells and seeds -------------------------------------------------------------

HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)
#: Test D's extrapolation horizon; enumerated after the main grid so main ids stay put.
EXTRA_HORIZONS: tuple[int, ...] = (2000,)
#: Live-arm caps of the cap sweep; ``"T"`` means ``cap == horizon`` (effectively no cap).
CAPS: tuple[int | str, ...] = (32, 64, "T")
#: The cap=128 sweep (``run_deployment.py --test cap``); enumerated after the
#: `EXTRA_HORIZONS` block, at the main `HORIZONS` only, so no earlier id moves.
EXTRA_CAPS: tuple[int | str, ...] = (128,)

#: ``onpolicy`` (M9): fresh episodes of the deployed learned policy, harvested for
#: on-policy relabelling; above the test band, below the corpus minimum (asserted below).
SPLIT_BASE: dict[str, int] = {
    "tune": 1_000_000,
    "val": 5_000_000,
    "test": 10_000_000,
    "onpolicy": 15_000_000,
}
CELL_STRIDE = 1_000
#: Smallest seed the corpus consumed (`corpus_seed_set`); every split band must end below it.
CORPUS_SEED_MIN = 20_262_460

#: Environment ids in enumeration order (sorted, so the order is a property of the ids).
ENV_ORDER: tuple[str, ...] = tuple(sorted(ALL_ENVS))
_ENV_INDEX: dict[str, int] = {env_id: i for i, env_id in enumerate(ENV_ORDER)}
N_MAIN_GRID_CELLS = len(ENV_ORDER) * len(HORIZONS) * len(CAPS)
#: Size of the `EXTRA_HORIZONS` block (env x `EXTRA_HORIZONS` x `CAPS`).
N_EXTRA_HORIZONS_CELLS = len(ENV_ORDER) * len(EXTRA_HORIZONS) * len(CAPS)
#: Size of the `EXTRA_CAPS` block (env x `HORIZONS` x `EXTRA_CAPS`).
N_EXTRA_CAPS_CELLS = len(ENV_ORDER) * len(HORIZONS) * len(EXTRA_CAPS)
N_CELLS = N_MAIN_GRID_CELLS + N_EXTRA_HORIZONS_CELLS + N_EXTRA_CAPS_CELLS

# The bands must not touch: the largest seed of one split must sit below the base of
# the next. Checked at import so a grid extension cannot silently break it.
_bases = sorted(SPLIT_BASE.values())
for _lo, _hi in zip(_bases, _bases[1:], strict=False):
    if _lo + (N_CELLS - 1) * CELL_STRIDE >= _hi:
        raise RuntimeError(
            f"seed bands overlap: {N_CELLS} cells x {CELL_STRIDE} from {_lo} reaches {_hi}"
        )
if _bases[-1] + (N_CELLS - 1) * CELL_STRIDE >= CORPUS_SEED_MIN:
    raise RuntimeError(
        f"the highest split band ({_bases[-1]} + {N_CELLS} x {CELL_STRIDE}) reaches the "
        f"corpus seed range starting at {CORPUS_SEED_MIN}"
    )


def resolve_cap(horizon: int, cap: int | str) -> int:
    """``"T"`` -> the horizon; an int passes through."""
    if isinstance(cap, str):
        if cap != "T":
            raise ValueError(f"cap must be an int or 'T'; got {cap!r}")
        return int(horizon)
    return int(cap)


def _cap_index(horizon: int, cap: int | str) -> int:
    """Index of `cap` within `CAPS` (main grid and `EXTRA_HORIZONS` block only).

    Raises `ValueError` for a cap outside `CAPS` -- including every `EXTRA_CAPS`
    value -- so `cell_id` can fall back to `_extra_cap_index` without ambiguity.
    """
    cap_int = resolve_cap(horizon, cap)
    for i, c in enumerate(CAPS):
        if c != "T" and cap_int == c:
            return i
    if cap_int == int(horizon):
        return CAPS.index("T")
    raise ValueError(f"cap {cap!r} is not one of {CAPS} at horizon {horizon}")


def _extra_cap_index(cap: int | str) -> int:
    """Index of `cap` within `EXTRA_CAPS`; these are fixed ints, no ``"T"`` alias."""
    if not isinstance(cap, str):
        cap_int = int(cap)
        for i, c in enumerate(EXTRA_CAPS):
            if c != "T" and cap_int == c:
                return i
    raise ValueError(f"cap {cap!r} is not one of EXTRA_CAPS={EXTRA_CAPS}")


def cell_id(env_id: str, horizon: int, cap: int | str) -> int:
    """Deterministic index of the cell ``(env, T, cap)``; injective over the study's grid.

    Main grid first (`ENV_ORDER` x `HORIZONS` x `CAPS`, env-major), then the
    `EXTRA_HORIZONS` block in the same layout, then the `EXTRA_CAPS` block
    (`ENV_ORDER` x `HORIZONS` x `EXTRA_CAPS`, env-major) -- `EXTRA_CAPS` is only
    defined at the main `HORIZONS`, not `EXTRA_HORIZONS`. Anything outside raises
    rather than silently sharing a seed with a real cell.
    """
    if env_id not in _ENV_INDEX:
        raise KeyError(f"unknown environment {env_id!r}; known={ENV_ORDER}")
    e = _ENV_INDEX[env_id]
    horizon = int(horizon)
    try:
        c = _cap_index(horizon, cap)
    except ValueError:
        c = None
    if c is not None:
        if horizon in HORIZONS:
            h = HORIZONS.index(horizon)
            return (e * len(HORIZONS) + h) * len(CAPS) + c
        if horizon in EXTRA_HORIZONS:
            h = EXTRA_HORIZONS.index(horizon)
            return N_MAIN_GRID_CELLS + (e * len(EXTRA_HORIZONS) + h) * len(CAPS) + c
        raise ValueError(
            f"horizon {horizon} is not in HORIZONS={HORIZONS} or EXTRA_HORIZONS={EXTRA_HORIZONS}"
        )
    ec = _extra_cap_index(cap)
    if horizon not in HORIZONS:
        raise ValueError(
            f"cap {cap!r} (EXTRA_CAPS) is only defined at HORIZONS={HORIZONS}; got horizon {horizon}"
        )
    h = HORIZONS.index(horizon)
    return (
        N_MAIN_GRID_CELLS
        + N_EXTRA_HORIZONS_CELLS
        + (e * len(HORIZONS) + h) * len(EXTRA_CAPS)
        + ec
    )


def base_seed(split: str, env_id: str, horizon: int, cap: int | str) -> int:
    """``SPLIT_BASE[split] + cell_id * CELL_STRIDE``: the CRN seed shared by every policy."""
    if split not in SPLIT_BASE:
        raise KeyError(f"unknown split {split!r}; expected one of {tuple(SPLIT_BASE)}")
    return SPLIT_BASE[split] + cell_id(env_id, horizon, cap) * CELL_STRIDE


def make_cell(
    split: str,
    env_id: str,
    horizon: int,
    cap: int | str,
    n_replicates: int,
    alpha: float = 0.05,
) -> CellSpec:
    """The harness `CellSpec` for one cell of one split (cap ``"T"`` resolved to the horizon)."""
    if env_id not in ALL_ENVS:
        raise KeyError(f"unknown environment {env_id!r}; known={ENV_ORDER}")
    return CellSpec(
        env_id=env_id,
        env_spec=ALL_ENVS[env_id],
        horizon=int(horizon),
        cap=resolve_cap(horizon, cap),
        base_seed=base_seed(split, env_id, horizon, cap),
        n_replicates=int(n_replicates),
        alpha=float(alpha),
    )


def make_matched_cell(
    split: str,
    env_id: str,
    horizon: int,
    cap: int,
    n_replicates: int,
    seed_cap: int | str = 64,
    alpha: float = 0.05,
) -> CellSpec:
    """A K-matched control cell (NEXT-STEPS 2.4): the ``(env, T, seed_cap)`` cell's seed, at `cap`.

    The control -- `always_search` capped at a learned policy's own realised K_final -- is a
    control only if it runs on the SAME episodes as the learned policy did, so the cell
    keeps `base_seed(split, env_id, horizon, seed_cap)` and overrides nothing but the cap.
    A cell of its own in the grid (the `EXTRA_CAPS` mechanism) would draw fresh episodes
    and throw the pairing away. Consequently a matched cell has no `cell_id`; it is never
    part of the study's grid and its name (``<env>_T<T>_cap<K>``) must not collide with
    one -- which it cannot, since a K that equals a grid cap IS that grid cell.

    `cap` must be reachable: at least `n_initial_arms + 1` (there must be a SEARCH to cap)
    and at most the horizon.
    """
    cap = int(cap)
    if not 3 <= cap <= int(horizon):
        raise ValueError(f"matched cap must be in [3, T={horizon}]; got {cap}")
    if env_id not in ALL_ENVS:
        raise KeyError(f"unknown environment {env_id!r}; known={ENV_ORDER}")
    return CellSpec(
        env_id=env_id,
        env_spec=ALL_ENVS[env_id],
        horizon=int(horizon),
        cap=cap,
        base_seed=base_seed(split, env_id, horizon, seed_cap),
        n_replicates=int(n_replicates),
        alpha=float(alpha),
    )


def cell_at_cap(
    split: str, env_id: str, horizon: int, cap: int | str, n_replicates: int, alpha: float = 0.05
) -> CellSpec:
    """The cell for tuning or selecting constants at `cap` on `split`.

    Caps of the study's grid (`CAPS`, `EXTRA_CAPS`) resolve through `make_cell` as always,
    so every existing tuning row keeps its seed. Any other cap -- the paired sweep's
    ladder (`run_deployment.CAPP_HORIZON_CAPS`) -- runs on the cap-64 cell's seed for
    that split (`make_matched_cell`): the constants selected at cap 48 are then selected
    on the same episodes as those at cap 64, which is the pairing the sweep deploys them
    under.
    """
    try:
        return make_cell(split, env_id, horizon, cap, n_replicates, alpha)
    except ValueError:
        return make_matched_cell(split, env_id, horizon, int(cap), n_replicates, seed_cap=64, alpha=alpha)


# ---- seed-disjointness guard --------------------------------------------------------

#: RUNBOOK.md section 1: the arguments the corpus was generated with.
CORPUS_SEED = 20260910
CORPUS_HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)
CORPUS_STATES_PER_HORIZON = 20_000
CORPUS_TRAJECTORIES = 8
CORPUS_SNAPSHOTS = 8
CORPUS_MAX_LIVE_ARMS = 64
#: `generate_states._detach`: trajectory `m` of a shard is seeded at `shard.seed + 7919 * m`.
CORPUS_TRAJECTORY_STRIDE = 7919
#: `labeling.label_state`: batch `b` materializes at `snapshot.base_seed + 1_000_003 * (b + 1)`
#: for `b` in `range(len(DEFAULT_BATCHES) + 8)` = 12 batches at most.
CORPUS_BATCH_STRIDE = 1_000_003
CORPUS_MAX_BATCHES = 12

OLD_BENCHMARK_SEED = 4242


def production_corpus_args() -> argparse.Namespace:
    """The `label_states.main` namespace for RUNBOOK section 1 (only what `build_shards` reads)."""
    return argparse.Namespace(
        horizons=list(CORPUS_HORIZONS),
        states_per_horizon=CORPUS_STATES_PER_HORIZON,
        trajectories=CORPUS_TRAJECTORIES,
        snapshots=CORPUS_SNAPSHOTS,
        max_live_arms=CORPUS_MAX_LIVE_ARMS,
        target_se=3e-4,
        max_replicates=4096,
        commit_steps=[1, 4, 16],
        seed=CORPUS_SEED,
        include_mixtures=False,
    )


@functools.lru_cache(maxsize=1)
def corpus_seed_set() -> frozenset[int]:
    """Every `GrowingState.base_seed` the corpus generation and labelling consumed.

    Shard seed ``s``, its eight detached trajectories ``s + 7919 m`` (which are the
    snapshot seeds), and each snapshot's labelling batches ``+ 1_000_003 (b + 1)``.
    """
    shards = label_states.build_shards(production_corpus_args())
    seeds: set[int] = set()
    for shard in shards:
        for m in range(CORPUS_TRAJECTORIES):
            snapshot_seed = int(shard.seed) + CORPUS_TRAJECTORY_STRIDE * m
            seeds.add(snapshot_seed)
            for b in range(CORPUS_MAX_BATCHES):
                seeds.add(snapshot_seed + CORPUS_BATCH_STRIDE * (b + 1))
    return frozenset(seeds)


@functools.lru_cache(maxsize=1)
def old_benchmark_seed_set() -> frozenset[int]:
    """`evaluate_policy.py`: ``4242 + crc32(name) % 10000`` per baseline (and its learned slot)."""
    import evaluate_policy
    import numpy as np

    names = list(evaluate_policy.baselines(np.random.default_rng(0))) + ["LEARNED_phi"]
    return frozenset(
        OLD_BENCHMARK_SEED + int(zlib.crc32(name.encode())) % 10_000 for name in names
    )


def assert_seed_disjointness(seeds: Iterable[int]) -> None:
    """Raise if any of `seeds` was consumed by the corpus or the old benchmark.

    Called at the start of every run script over every `base_seed` it will use; a
    collision would make a "held-out" episode one the labels were computed on.
    """
    wanted = {int(s) for s in seeds}
    corpus_hits = sorted(wanted & corpus_seed_set())
    bench_hits = sorted(wanted & old_benchmark_seed_set())
    if corpus_hits or bench_hits:
        raise ValueError(
            "seed collision: "
            f"{len(corpus_hits)} seed(s) in the corpus set (first: {corpus_hits[:5]}), "
            f"{len(bench_hits)} in the old benchmark set (first: {bench_hits[:5]})"
        )
