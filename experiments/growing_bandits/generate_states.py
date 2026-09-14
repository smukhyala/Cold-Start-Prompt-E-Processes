#!/usr/bin/env python
"""Generate a pool of realistic bandit states from a mixture of behavioral policies.

State-distribution bias is the main threat to anything we fit later: a corpus
harvested from one policy only contains the states that policy visits, so a decision
function trained on it learns that policy's habits rather than the underlying
trade-off. Hence the deliberately wide mixture of behavioral rules.

The M replicate axis is reused here as M *independent trajectories*, so one vectorized
run yields M states per snapshot time rather than one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.growing.allocation import AllocationRule  # noqa: E402
from cold_start.growing.features import SearchHistory  # noqa: E402
from cold_start.growing.labeling import Snapshot  # noqa: E402
from cold_start.growing.search_policies import SearchPolicy  # noqa: E402
from cold_start.growing.simulator import Simulator, seed_initial_arms  # noqa: E402
from cold_start.growing.state import EMPTY_UID, GrowingState  # noqa: E402


@dataclass
class GenSpec:
    horizon: int
    n_trajectories: int
    snapshot_times: tuple[int, ...]
    max_live_arms: int = 32
    initial_arms: int = 2
    seed: int = 20260910


def snapshot_times_for(horizon: int, count: int, rng: np.random.Generator) -> tuple[int, ...]:
    """Snapshot times spread evenly in REMAINING BUDGET, the variable that matters.

    Two earlier attempts got this wrong in opposite directions, both silently. Pure log
    spacing from t=0 pushed three quarters of the corpus into the large-remaining region,
    where the single-action advantage is smallest. Interleaving a log-spaced-from-start
    half with a log-spaced-from-horizon half then piled snapshots at BOTH extremes and
    hollowed out the middle -- measured 51.8% of states at remaining > 0.75 and 23.2%
    below 0.10, but only 2.5% in the 0.25-0.50 band, which is precisely the transition
    region where the decision boundary should be most informative.

    Spacing evenly in `remaining / T` and converting back to `t` fixes both at once: it
    is the coordinate the oracle advantage actually scales with, so even coverage there
    is even coverage of the thing being modelled. A mild jitter keeps successive
    trajectories from sampling an identical lattice.
    """
    lo_t, hi_t = 3, max(4, horizon - 1)
    count = max(1, count)
    fracs = (np.arange(count) + 0.5) / count          # midpoints of equal bands in [0,1]
    fracs = fracs + rng.uniform(-0.4, 0.4, size=count) / count
    fracs = np.clip(fracs, 0.01, 0.99)
    times = np.round(horizon * (1.0 - fracs)).astype(int)
    times = np.clip(times, lo_t, hi_t)
    return tuple(int(t) for t in np.unique(times))


def harvest(
    reservoir,
    table,
    policy: SearchPolicy,
    allocation: AllocationRule,
    spec: GenSpec,
    env_id: str,
    seed_offset: int = 0,
) -> list[Snapshot]:
    """Run `n_trajectories` in parallel and snapshot every one at each chosen time."""
    rng = np.random.default_rng(spec.seed + seed_offset)
    times = snapshot_times_for(spec.horizon, len(spec.snapshot_times) or 4, rng)

    state = GrowingState(
        n_replicates=spec.n_trajectories,
        capacity=max(spec.initial_arms * 2, 8),
        horizon=spec.horizon,
        base_seed=spec.seed + seed_offset,
    )
    sim = Simulator(
        table=table,
        reservoir=reservoir,
        allocation=allocation,
        search_policy=policy,
        horizon=spec.horizon,
        max_live_arms=spec.max_live_arms,
    )
    seed_initial_arms(state, reservoir, spec.initial_arms, table)

    decisions: list[np.ndarray] = []
    best_trace: list[np.ndarray] = []
    out: list[Snapshot] = []

    for t in range(spec.initial_arms, spec.horizon):
        if t in times:
            out.extend(
                _detach(state, t, spec, env_id, policy, allocation, decisions, best_trace)
            )
        res = sim.step(state, t)
        decisions.append(res.searched.copy())
        best_trace.append(sim.best_posterior_mean(state).copy())
    return out


def _detach(
    state: GrowingState,
    t: int,
    spec: GenSpec,
    env_id: str,
    policy: SearchPolicy,
    allocation: AllocationRule,
    decisions: list[np.ndarray],
    best_trace: list[np.ndarray],
) -> list[Snapshot]:
    """Peel each replicate out of the batched state as a standalone Snapshot."""
    uid2 = state.view(state.uid)
    n2, s2, mu2 = state.view(state.n), state.view(state.S), state.view(state.mu)
    dec = np.array(decisions).T if decisions else np.zeros((state.M, 0), dtype=bool)
    trace = np.array(best_trace).T if best_trace else np.zeros((state.M, 0))

    snaps: list[Snapshot] = []
    for m in range(state.M):
        active = uid2[m] != EMPTY_UID
        if not active.any():
            continue
        snaps.append(
            Snapshot(
                n=n2[m, active].copy(),
                successes=s2[m, active].copy(),
                mu=mu2[m, active].copy(),
                t=int(t),
                horizon=spec.horizon,
                n_draws=int(state.n_draws[m]),
                base_seed=int(state.base_seed) + 7919 * m,
                meta={
                    "env_id": env_id,
                    "policy": policy.name,
                    "allocation": allocation.name,
                    "replicate": m,
                    "history": SearchHistory(
                        decisions=dec[m].copy(), best_mean_trace=trace[m].copy()
                    ),
                },
            )
        )
    return snaps
