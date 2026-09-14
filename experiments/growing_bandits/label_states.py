#!/usr/bin/env python
"""M4: generate and oracle-label the full state pool, in parallel, resumably.

Shards are defined by (environment, horizon, behavioural policy, seed offset) and each
one *regenerates its own states from its seed* rather than receiving a pre-built pool.
That keeps the payload crossing the process boundary tiny, and -- more importantly --
makes every shard a pure function of its shard key, so a resumed run reproduces an
uninterrupted one exactly.

Three details are load-bearing on macOS, where the start method is `spawn`:

* the entry point must be a real file with a `__main__` guard, because spawn re-imports it;
* the confidence-sequence tables are memory-mapped in an `initializer` and shared through
  the page cache -- passing one as a task argument costs ~3.8 ms per task and a private
  copy per worker;
* workers never import pandas (52 MB each for nothing); parquet is written through pyarrow.

Measured throughput is ~200 ns per replicate-pull at M=2048, and effective parallel speedup
on this machine is ~5.5x rather than the core count, so the worker default is 9 rather than 14.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_states import GenSpec, harvest  # noqa: E402

from cold_start.growing.allocation import LUCB, UCBChallenger, WidthRacing  # noqa: E402
from cold_start.growing.evidence import PairwiseEvidence  # noqa: E402
from cold_start.growing.features import extract_features  # noqa: E402
from cold_start.growing.labeling import label_state_multi  # noqa: E402
from cold_start.growing.recommend import oracle_prior_from_reservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.search_policies import (  # noqa: E402
    BernoulliSearch,
    BracketExpansion,
    EpsilonSchedule,
    EvidenceGatedSchedule,
    PowerSchedule,
    UniformRandom,
)
from cold_start.growing.simulator import Simulator  # noqa: E402
from cold_start.growing.state import ForcedActionUnavailable  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

# ---- environment grid ---------------------------------------------------------

FAMILY_A = [
    ("beta_good_common", {"type": "beta", "params": {"a": 5.0, "b": 2.0}}),
    ("beta_moderately_rare", {"type": "beta", "params": {"a": 2.0, "b": 5.0}}),
    ("beta_rare_excellent", {"type": "beta", "params": {"a": 1.0, "b": 9.0}}),
    ("beta_mostly_mediocre", {"type": "beta", "params": {"a": 8.0, "b": 8.0}}),
    ("beta_skewed", {"type": "beta", "params": {"a": 0.5, "b": 3.0}}),
    ("beta_uniform", {"type": "beta", "params": {"a": 1.0, "b": 1.0}}),
]
FAMILY_B = [
    (
        f"tail_b{beta}_mu{mu_star}_c{c}",
        {"type": "tail", "params": {"beta": beta, "mu_star": mu_star, "c": c}},
    )
    for beta in (0.5, 1.0, 2.0, 4.0, 8.0)
    for mu_star in (0.8, 0.9, 1.0)
    for c in (1.0, 2.0)
]
# Family C is the held-out generalization set and is never used for training.
FAMILY_C = [
    (f"mix_{name}", {"type": "mixture", "params": {"preset": name}})
    for name in ("many_mediocre_rare_excellent", "bulk_half_tiny_cluster_high", "broad_low_narrow_high")
]


def policy_by_name(name: str, seed: int):
    rng = np.random.default_rng(seed)
    table = {
        "aggressive": lambda: BernoulliSearch(p=0.5, rng=rng),
        "conservative": lambda: BernoulliSearch(p=0.05, rng=rng),
        "sqrt": lambda: PowerSchedule(alpha=0.5, c=1.0, rng=rng),
        "cbrt": lambda: PowerSchedule(alpha=1.0 / 3.0, c=1.5, rng=rng),
        "t23": lambda: PowerSchedule(alpha=2.0 / 3.0, c=0.8, rng=rng),
        "random": lambda: UniformRandom(rng=rng),
        "epsilon": lambda: EpsilonSchedule(alpha=0.5, c=1.0, epsilon=0.15, rng=rng),
        "bracket": lambda: BracketExpansion(base_width=4, rng=rng),
    }
    if name not in table:
        raise ValueError(f"unknown policy {name!r}; expected one of {sorted(table)}")
    return table[name]()


ALLOCATIONS = {"lucb": LUCB, "ucb": UCBChallenger, "racing": WidthRacing}
BEHAVIOURAL = ["aggressive", "conservative", "sqrt", "cbrt", "t23", "random", "epsilon", "bracket"]


@dataclass(frozen=True)
class Shard:
    """A unit of work whose entire output is determined by these fields."""

    shard_id: str
    env_id: str
    horizon: int
    policy: str
    allocation: str
    seed: int
    trajectories: int
    snapshots: int
    max_live_arms: int
    target_se: float
    max_replicates: int
    commit_steps: tuple[int, ...]


_TABLES: dict[int, CSTable] = {}
_PAIRWISE: list = []


def _worker_init(alpha: float) -> None:
    """Per-worker setup. Runs once per process, never per task."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _PAIRWISE.append(PairwiseEvidence())
    _WORKER_ALPHA.append(alpha)


_WORKER_ALPHA: list[float] = []


def _table_for(horizon: int) -> CSTable:
    """Memory-mapped table, loaded once per (worker, horizon) and shared via page cache."""
    if horizon not in _TABLES:
        alpha = _WORKER_ALPHA[0] if _WORKER_ALPHA else 0.05
        _TABLES[horizon] = CSTable.load_or_build(horizon, alpha=alpha)
    return _TABLES[horizon]


def _env_spec(env_id: str) -> dict:
    for name, spec in FAMILY_A + FAMILY_B + FAMILY_C:
        if name == env_id:
            return spec
    raise KeyError(f"unknown environment {env_id!r}")


def _family_of(env_id: str) -> str:
    if env_id.startswith("beta_"):
        return "A"
    if env_id.startswith("tail_"):
        return "B"
    return "C"


def run_shard(shard: Shard) -> list[dict]:
    """Generate this shard's states and label them. Pure in `shard`."""
    if not _PAIRWISE:
        _worker_init(0.05)
    table = _table_for(shard.horizon)
    pairwise = _PAIRWISE[0]
    spec = _env_spec(shard.env_id)
    reservoir = build_reservoir(spec)
    prior = oracle_prior_from_reservoir(reservoir)

    policy = policy_by_name(shard.policy, shard.seed)
    allocation = ALLOCATIONS[shard.allocation]()

    snaps = harvest(
        reservoir,
        table,
        policy,
        allocation,
        GenSpec(
            horizon=shard.horizon,
            n_trajectories=shard.trajectories,
            snapshot_times=tuple(range(shard.snapshots)),
            max_live_arms=shard.max_live_arms,
            seed=shard.seed,
        ),
        shard.env_id,
        seed_offset=0,
    )

    def factory(offset: int) -> Simulator:
        return Simulator(
            table=table,
            reservoir=reservoir,
            allocation=LUCB(),
            search_policy=EvidenceGatedSchedule(alpha=0.5, c=1.0, min_pulls_per_arm=2),
            horizon=shard.horizon,
            max_live_arms=shard.max_live_arms,
        )

    rows: list[dict] = []
    n_undefined = 0
    for i, snap in enumerate(snaps):
        # A state already at the live-arm cap has no SEARCH counterfactual, so A_t is
        # undefined there. Skipping is the honest choice: labelling it zero would add a
        # maximally-confident fabricated row to the corpus.
        if snap.k >= shard.max_live_arms:
            n_undefined += 1
            continue
        row = extract_features(
            n=snap.n,
            successes=snap.successes,
            mu_true=snap.mu,
            t=snap.t,
            horizon=snap.horizon,
            table=table,
            pairwise=pairwise,
            reservoir=reservoir,
            history=snap.meta.get("history"),
        )
        try:
            labels = label_state_multi(
                snap,
                factory,
                table,
                commit_steps=shard.commit_steps,
                target_se=shard.target_se,
                max_replicates=shard.max_replicates,
                oracle_prior=prior,
                max_live_arms=shard.max_live_arms,
            )
        except ForcedActionUnavailable:
            n_undefined += 1
            continue
        canonical = labels[shard.commit_steps[0]]
        for k, lk in labels.items():
            row[f"label_A_k{k}"] = float(lk.advantage)
            row[f"label_se_k{k}"] = float(lk.se)
            row[f"label_M_k{k}"] = float(lk.n_replicates)
        row["label_A"] = float(canonical.advantage)
        row["label_se"] = float(canonical.se)
        row["label_se_unpaired"] = float(canonical.se_unpaired)
        row["label_frac_identical"] = float(canonical.frac_identical)
        row["label_M"] = float(canonical.n_replicates)
        row["label_mean_search"] = float(canonical.mean_search)
        row["label_mean_refine"] = float(canonical.mean_refine)
        for dk, dv in canonical.diagnostics.items():
            row[f"label_diag_{dk}"] = float(dv)
        row["meta_shard"] = shard.shard_id
        row["meta_env"] = shard.env_id
        row["meta_family"] = _family_of(shard.env_id)
        row["meta_policy"] = shard.policy
        row["meta_allocation"] = shard.allocation
        row["meta_horizon"] = float(shard.horizon)
        row["meta_state_index"] = float(i)
        row["meta_n_undefined_in_shard"] = float(n_undefined)
        rows.append(row)
    return rows


def _is_constructible(spec: dict) -> bool:
    """Drop environments the reservoir validator rejects as degenerate.

    Six of the thirty Family B parameter combinations put most of their mass at zero
    after clipping and raise `DegenerateReservoirError`. Left in the grid they claimed
    288 of 1560 shards, every one of which would have died at run time -- and because
    those shards are not evenly spread across horizons, their deaths would have skewed
    the surviving environment mixture on top of everything else.
    """
    try:
        build_reservoir(spec)
    except Exception:
        return False
    return True


def _coprime_stride(n: int) -> int:
    """A stride sharing no factor with `n`, so the walk is a full-period permutation."""
    if n <= 2:
        return 1
    from math import gcd

    for cand in (577, 373, 271, 181, 97, 61, 37, 19, 11, 7):
        if cand < n and gcd(cand, n) == 1:
            return cand
    stride = max(2, n // 2 - 1)
    while gcd(stride, n) != 1:
        stride -= 1
    return stride


def build_shards(args) -> list[Shard]:
    """Enumerate shards, honouring an explicit per-horizon quota.

    The quota is not cosmetic. Harvesting along trajectories without one makes a T=1000
    run yield roughly twenty times as many states as a T=50 run, so the long horizon
    silently becomes most of the corpus -- and it is also the region where the
    single-action advantage is smallest, so the compute buys the least signal.
    """
    envs = [e for e in list(FAMILY_A + FAMILY_B) if _is_constructible(e[1])]
    if args.include_mixtures:
        envs += [e for e in FAMILY_C if _is_constructible(e[1])]

    combos = [
        (env_id, policy, allocation)
        for env_id, _ in envs
        for policy in BEHAVIOURAL
        for allocation in ALLOCATIONS
    ]

    # Walk the cross product with a stride COPRIME to its length, not with +1.
    #
    # A +1 walk looks like it decorrelates the factors, and it does make the marginal
    # counts perfectly even -- which is exactly what makes the defect invisible. But
    # `combos` is enumerated environment-major, so 24 consecutive entries share an
    # environment; with fewer shards per horizon than combinations, each horizon then
    # takes a CONTIGUOUS window and therefore only ~13 of the 30 environments. Measured
    # on the production defaults: 69 of 180 environment x horizon cells were populated,
    # Beta reservoirs appeared at T=50 and T=1000 and nowhere else, and because Family B
    # is enumerated with the tail exponent outermost, beta became nearly monotone in the
    # horizon. The aliased effect was LARGER than the horizon effect it was aliased with
    # (0.22 vs 0.155 in P(A_t > 0)) and pointed the same way, so a fitted budget
    # coefficient would have silently absorbed the tail-exponent trend -- and the
    # hold-out-by-horizon test would have been meaningless.
    #
    # A coprime stride visits every combination before repeating any, so each horizon
    # gets a spread sample rather than a contiguous block.
    stride = _coprime_stride(len(combos))

    shards: list[Shard] = []
    for h_idx, horizon in enumerate(args.horizons):
        per_shard = args.trajectories * args.snapshots
        n_shards = max(1, args.states_per_horizon // max(per_shard, 1))
        rotation = (h_idx * 7919) % max(len(combos), 1)
        for s in range(n_shards):
            env_id, policy, allocation = combos[(s * stride + rotation) % len(combos)]
            shards.append(
                Shard(
                    shard_id=f"h{horizon}_s{s:05d}",
                    env_id=env_id,
                    horizon=horizon,
                    policy=policy,
                    allocation=allocation,
                    seed=args.seed + 1_000_003 * s + 31 * horizon,
                    trajectories=args.trajectories,
                    snapshots=args.snapshots,
                    max_live_arms=args.max_live_arms,
                    target_se=args.target_se,
                    max_replicates=args.max_replicates,
                    commit_steps=tuple(args.commit_steps),
                )
            )
    return shards


def _write_chunk(rows: list[dict], out_dir: Path, chunk_id: str) -> Path:
    """Atomic parquet write.

    A crash part-way through leaves a footer-less file, and pyarrow then fails to read
    the WHOLE directory -- so the visible file only ever appears complete.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    normalized = [{k: r.get(k) for k in keys} for r in rows]
    tbl = pa.Table.from_pylist(normalized)
    final = out_dir / f"part-{chunk_id}.parquet"
    tmp = out_dir / f"part-{chunk_id}.parquet.tmp"
    pq.write_table(tbl, tmp, compression="zstd", row_group_size=20_000)
    os.replace(tmp, final)
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    ap.add_argument("--states-per-horizon", type=int, default=20_000)
    ap.add_argument("--trajectories", type=int, default=8)
    ap.add_argument("--snapshots", type=int, default=8)
    ap.add_argument("--max-live-arms", type=int, default=64)
    ap.add_argument("--target-se", type=float, default=3e-4)
    ap.add_argument("--max-replicates", type=int, default=4096)
    ap.add_argument("--commit-steps", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--chunk-shards", type=int, default=8)
    ap.add_argument("--include-mixtures", action="store_true",
                    help="include Family C; normally held out for generalization")
    ap.add_argument("--out", type=str, default="data/oracle_labels")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--limit-shards", type=int, default=0, help="smoke-test cap")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    done: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["shard_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    shards = build_shards(args)
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    todo = [s for s in shards if s.shard_id not in done]

    print(f"{len(shards)} shards total, {len(done)} already complete, {len(todo)} to run")
    print(f"workers={args.workers}  horizons={args.horizons}  commit_steps={args.commit_steps}")
    if not todo:
        print("nothing to do")
        return

    # Build every table up front, in the parent: workers then memory-map a file that
    # already exists rather than racing to construct it.
    for horizon in sorted({s.horizon for s in todo}):
        t0 = time.time()
        CSTable.load_or_build(horizon, alpha=args.alpha)
        print(f"  CS table T={horizon} ready in {time.time()-t0:.1f}s")

    start = time.time()
    n_rows = 0
    buffer: list[dict] = []
    completed: list[str] = []
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(args.alpha,),
    ) as ex:
        futures = {ex.submit(run_shard, s): s for s in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            shard = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001 - one bad shard must not sink the run
                print(f"  shard {shard.shard_id} FAILED: {type(exc).__name__}: {exc}")
                continue
            buffer.extend(rows)
            completed.append(shard.shard_id)
            n_rows += len(rows)

            if len(completed) >= args.chunk_shards:
                _write_chunk(buffer, out_dir, completed[0])
                with manifest_path.open("a") as f:
                    for sid in completed:
                        f.write(json.dumps({"shard_id": sid}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                buffer, completed = [], []

            if i % 25 == 0 or i == len(todo):
                el = time.time() - start
                rate = i / max(el, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"  {i}/{len(todo)} shards  {n_rows} rows  "
                      f"{el/60:.1f}m elapsed  ETA {eta/60:.1f}m")

    if buffer:
        _write_chunk(buffer, out_dir, completed[0] if completed else "tail")
        with manifest_path.open("a") as f:
            for sid in completed:
                f.write(json.dumps({"shard_id": sid}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    print(f"\ndone: {n_rows} labelled states in {(time.time()-start)/60:.1f} minutes")
    print(f"      -> {out_dir}")


if __name__ == "__main__":
    main()
