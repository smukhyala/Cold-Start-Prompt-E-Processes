"""Tests for the state-pool generator and the production shard plan.

These are design-matrix tests rather than numerics tests. They exist because two
coverage defects got through code review and a passing test suite: a shard plan whose
marginal counts were perfectly even while environment was confounded with horizon, and
a snapshot schedule that covered both extremes of the budget while leaving the middle
nearly empty. Neither is visible in any single row -- only in the distribution -- so the
only thing that catches them is asserting on the distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parent.parent / "experiments" / "growing_bandits"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from generate_states import GenSpec, harvest, snapshot_times_for  # noqa: E402
from label_states import (  # noqa: E402
    ALLOCATIONS,
    BEHAVIOURAL,
    _coprime_stride,
    _is_constructible,
    build_shards,
)

from cold_start.growing.allocation import LUCB  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.search_policies import PowerSchedule  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402


def _args(**kw) -> argparse.Namespace:
    base = dict(
        horizons=[50, 100, 200, 500, 1000], states_per_horizon=20_000, trajectories=8,
        snapshots=8, max_live_arms=64, target_se=3e-4, max_replicates=4096,
        commit_steps=[1, 4, 16], include_mixtures=False, seed=20260910,
    )
    base.update(kw)
    return argparse.Namespace(**base)


# ---- snapshot schedule --------------------------------------------------------


@pytest.mark.parametrize("horizon", [50, 100, 200, 500, 1000])
def test_snapshot_times_are_in_range_and_unique(horizon: int):
    ts = snapshot_times_for(horizon, 8, np.random.default_rng(0))
    assert len(set(ts)) == len(ts)
    assert all(3 <= t < horizon for t in ts), ts


def test_snapshot_times_cover_every_remaining_budget_quartile():
    """The regression guard for a schedule that piles up at the extremes.

    An earlier version put 51.8% of snapshots above remaining 0.75 and 23.2% below 0.10
    while leaving only 2.5% in the 0.25-0.50 band -- the transition region where the
    decision boundary is most informative.
    """
    rem = []
    for horizon in (50, 100, 200, 500, 1000):
        for rep in range(40):
            for t in snapshot_times_for(horizon, 8, np.random.default_rng(rep)):
                rem.append((horizon - t) / horizon)
    rem = np.array(rem)
    shares = [float(np.mean((rem >= lo) & (rem < hi)))
              for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))]
    assert min(shares) > 0.15, f"a remaining-budget quartile is starved: {shares}"
    assert max(shares) < 0.40, f"a remaining-budget quartile dominates: {shares}"


def test_snapshot_times_vary_between_trajectories():
    a = snapshot_times_for(200, 8, np.random.default_rng(1))
    b = snapshot_times_for(200, 8, np.random.default_rng(2))
    assert a != b, "every trajectory sampled the identical lattice"


# ---- shard plan ---------------------------------------------------------------


def test_shard_plan_is_deterministic():
    assert build_shards(_args()) == build_shards(_args())


def test_every_environment_appears_at_every_horizon():
    """The regression guard for the environment-horizon confound.

    `combos` is enumerated environment-major, so 24 consecutive entries share an
    environment. Walking it with a +1 stride gave each horizon a contiguous window and
    therefore only ~13 of 30 environments -- 69 of 180 cells, with Beta reservoirs
    appearing at T=50 and T=1000 and nowhere else. Marginal counts were perfectly even
    throughout, which is exactly why it was invisible.
    """
    shards = build_shards(_args())
    envs = {s.env_id for s in shards}
    horizons = {s.horizon for s in shards}
    cells = {(s.env_id, s.horizon) for s in shards}
    assert len(cells) == len(envs) * len(horizons), (
        f"only {len(cells)} of {len(envs) * len(horizons)} environment x horizon cells "
        f"are populated"
    )


def test_policy_and_allocation_are_fully_crossed_with_horizon():
    shards = build_shards(_args())
    horizons = {s.horizon for s in shards}
    assert len({(s.policy, s.horizon) for s in shards}) == len(BEHAVIOURAL) * len(horizons)
    assert len({(s.allocation, s.horizon) for s in shards}) == len(ALLOCATIONS) * len(horizons)


def test_environment_coverage_is_roughly_balanced():
    counts: dict[str, int] = {}
    for s in build_shards(_args()):
        counts[s.env_id] = counts.get(s.env_id, 0) + 1
    lo, hi = min(counts.values()), max(counts.values())
    assert hi <= 2 * lo, f"environment counts range {lo}..{hi}"


def test_degenerate_environments_are_excluded():
    """Six Family B combinations raise on construction; shards targeting them would die."""
    for s in build_shards(_args()):
        assert _is_constructible_by_name(s.env_id), f"{s.env_id} cannot be constructed"


def _is_constructible_by_name(env_id: str) -> bool:
    from label_states import _env_spec

    return _is_constructible(_env_spec(env_id))


@pytest.mark.parametrize("n", [7, 30, 64, 720, 864, 1000])
def test_coprime_stride_is_a_full_period_permutation(n: int):
    """A stride sharing a factor with n would revisit a subset and starve the rest."""
    from math import gcd

    stride = _coprime_stride(n)
    assert gcd(stride, n) == 1, f"stride {stride} shares a factor with {n}"
    assert len({(i * stride) % n for i in range(n)}) == n


def test_shard_seeds_are_distinct():
    seeds = [s.seed for s in build_shards(_args())]
    assert len(set(seeds)) == len(seeds), "shards share seeds; their states would repeat"


# ---- harvest ------------------------------------------------------------------


def test_harvest_produces_valid_snapshots():
    horizon = 60
    table = CSTable.load_or_build(horizon, alpha=0.05)
    reservoir = build_reservoir({"type": "beta", "params": {"a": 5.0, "b": 2.0}})
    snaps = harvest(
        reservoir, table, PowerSchedule(alpha=0.5, c=1.0), LUCB(),
        GenSpec(horizon=horizon, n_trajectories=4, snapshot_times=tuple(range(4)),
                max_live_arms=16, seed=7),
        "test_env",
    )
    assert snaps, "harvest returned nothing"
    for s in snaps:
        s.validate()
        assert s.k >= 1
        assert int(s.n.sum()) == s.t, f"pulls {int(s.n.sum())} != t {s.t}"
        assert s.meta["env_id"] == "test_env"
        assert "history" in s.meta


def test_harvest_is_deterministic():
    horizon = 60
    table = CSTable.load_or_build(horizon, alpha=0.05)
    reservoir = build_reservoir({"type": "beta", "params": {"a": 5.0, "b": 2.0}})
    spec = GenSpec(horizon=horizon, n_trajectories=3, snapshot_times=tuple(range(3)),
                   max_live_arms=16, seed=7)
    a = harvest(reservoir, table, PowerSchedule(alpha=0.5, c=1.0), LUCB(), spec, "e")
    b = harvest(reservoir, table, PowerSchedule(alpha=0.5, c=1.0), LUCB(), spec, "e")
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=False):
        assert np.array_equal(x.n, y.n)
        assert np.array_equal(x.successes, y.successes)
