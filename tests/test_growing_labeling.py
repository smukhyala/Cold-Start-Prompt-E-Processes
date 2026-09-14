"""Tests for the oracle labeller -- the module that defines the response variable."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.allocation import LUCB
from cold_start.growing.labeling import (
    HIST_BINS,
    PairedOutcome,
    Snapshot,
    label_state,
    label_state_multi,
    materialize,
)
from cold_start.growing.recommend import oracle_prior_from_reservoir
from cold_start.growing.reservoirs import build_reservoir
from cold_start.growing.search_policies import EvidenceGatedSchedule
from cold_start.growing.simulator import Simulator
from cold_start.growing.tables import CSTable

HORIZON = 80


@pytest.fixture(scope="module")
def table():
    return CSTable.load_or_build(HORIZON, alpha=0.05)


@pytest.fixture(scope="module")
def reservoir():
    return build_reservoir({"type": "tail", "params": {"beta": 2.0, "mu_star": 1.0, "c": 1.0}})


@pytest.fixture(scope="module")
def factory(table, reservoir):
    def make(_offset: int) -> Simulator:
        return Simulator(
            table=table,
            reservoir=reservoir,
            allocation=LUCB(),
            search_policy=EvidenceGatedSchedule(alpha=0.5, c=1.0, min_pulls_per_arm=2),
            horizon=HORIZON,
            max_live_arms=16,
        )

    return make


def _snap(t: int = 40, seed: int = 11) -> Snapshot:
    return Snapshot(
        n=np.array([12, 8, 8]),
        successes=np.array([8, 3, 2]),
        mu=np.array([0.65, 0.35, 0.30]),
        t=t,
        horizon=HORIZON,
        n_draws=3,
        base_seed=seed,
    )


# ---- Snapshot validation ------------------------------------------------------


def test_snapshot_rejects_more_successes_than_pulls():
    with pytest.raises(ValueError):
        Snapshot(np.array([3]), np.array([5]), np.array([0.5]), 3, HORIZON, 1, 0).validate()


def test_snapshot_rejects_empty_arm_set():
    with pytest.raises(ValueError):
        Snapshot(np.array([]), np.array([]), np.array([]), 0, HORIZON, 0, 0).validate()


def test_snapshot_rejects_t_outside_the_horizon():
    with pytest.raises(ValueError):
        Snapshot(np.array([2]), np.array([1]), np.array([0.5]), HORIZON + 5, HORIZON, 1, 0).validate()


# ---- materialize round-trip ---------------------------------------------------


def test_materialize_reproduces_the_snapshot(table):
    """A replayed state must be indistinguishable from the state it came from."""
    snap = _snap()
    st = materialize(snap, 32, table)
    assert np.all(st.Kt == snap.k)
    n2, s2, mu2 = st.view(st.n), st.view(st.S), st.view(st.mu)
    for j in range(snap.k):
        assert np.all(n2[:, j] == snap.n[j]), f"arm {j} pulls not restored"
        assert np.all(s2[:, j] == snap.successes[j]), f"arm {j} successes not restored"
        assert np.allclose(mu2[:, j], snap.mu[j], atol=1e-6)
    assert np.all(st.n_draws == snap.n_draws), "reservoir draw counter not restored"
    assert st.t == snap.t


def test_materialize_refreshes_bounds_from_the_table(table):
    """Bounds must come from (n, S), not be left at the unpulled default of [0, 1]."""
    st = materialize(_snap(), 8, table)
    lo, hi = table.bounds(np.array([12]), np.array([8]))
    assert np.allclose(st.view(st.lcb)[:, 0], lo[0], atol=1e-6)
    assert np.allclose(st.view(st.ucb)[:, 0], hi[0], atol=1e-6)


def test_materialize_is_deterministic(table):
    a = materialize(_snap(), 16, table, seed_offset=5)
    b = materialize(_snap(), 16, table, seed_offset=5)
    assert np.array_equal(a.key, b.key)
    assert np.array_equal(a.tie, b.tie)


# ---- PairedOutcome algebra ----------------------------------------------------


def test_paired_outcome_matches_numpy():
    rng = np.random.default_rng(3)
    s, r = rng.random(500), rng.random(500)
    out = PairedOutcome()
    out.add(s[:250], r[:250])
    out.add(s[250:], r[250:])
    d = s - r
    assert out.count == 500
    assert out.advantage == pytest.approx(d.mean())
    assert out.se == pytest.approx(d.std(ddof=1) / np.sqrt(500))
    assert out.mean_search == pytest.approx(s.mean())
    assert out.sd_refine == pytest.approx(r.std(ddof=1))


def test_paired_se_beats_unpaired_when_branches_are_correlated():
    """The entire reason the two branches share a random stream."""
    rng = np.random.default_rng(4)
    common = rng.random(4000)
    s = common + 0.01 * rng.random(4000)
    r = common + 0.01 * rng.random(4000)
    out = PairedOutcome()
    out.add(s, r)
    assert out.se < out.se_unpaired / 5, f"paired {out.se:.6f} vs unpaired {out.se_unpaired:.6f}"


def test_identical_branches_give_zero_advantage_and_zero_se():
    out = PairedOutcome()
    v = np.linspace(0, 1, 100)
    out.add(v, v)
    assert out.advantage == 0.0
    assert out.se == 0.0
    assert out.frac_identical == 1.0


def test_histograms_count_every_replicate():
    out = PairedOutcome()
    rng = np.random.default_rng(5)
    out.add(rng.random(300), rng.random(300))
    assert out.hist_search.sum() == 300
    assert out.hist_refine.sum() == 300
    assert len(out.hist_search) == HIST_BINS


# ---- label_state --------------------------------------------------------------


def test_label_is_reproducible(factory, table):
    a = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512)
    b = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512)
    assert a.advantage == pytest.approx(b.advantage)
    assert a.n_replicates == b.n_replicates


def test_label_respects_the_replicate_cap(factory, table):
    r = label_state(_snap(), factory, table, target_se=0.0, max_replicates=512)
    assert r.n_replicates <= 512


def test_label_stops_early_once_precise_enough(factory, table):
    loose = label_state(_snap(), factory, table, target_se=1.0, max_replicates=4096)
    tight = label_state(_snap(), factory, table, target_se=1e-9, max_replicates=4096)
    assert loose.n_replicates < tight.n_replicates


def test_oracle_action_matches_the_sign(factory, table):
    r = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512)
    assert r.oracle_action == ("SEARCH" if r.advantage > 0 else "REFINE")


def test_confidence_interval_brackets_the_estimate(factory, table):
    r = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512)
    lo, hi = r.ci()
    assert lo <= r.advantage <= hi


def test_diagnostics_are_finite(factory, table):
    r = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512)
    assert r.diagnostics, "no marginal diagnostics recorded"
    bad = [k for k, v in r.diagnostics.items() if not np.isfinite(v)]
    assert not bad, f"non-finite diagnostics: {bad}"


def test_search_diagnostic_reports_a_plausible_new_arm(factory, table):
    r = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512)
    p = r.diagnostics["search_p_new_plausible"]
    assert 0.0 <= p <= 1.0
    mu_new = r.diagnostics["search_new_arm_mu_mean"]
    assert 0.0 <= mu_new <= 1.0


def test_oracle_prior_path_runs(factory, table, reservoir):
    prior = oracle_prior_from_reservoir(reservoir)
    r = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=512,
                    oracle_prior=prior)
    assert np.isfinite(r.advantage)
    assert np.isfinite(r.se)


# ---- commitment horizon -------------------------------------------------------


def test_commit_steps_forces_more_than_one_action(factory, table):
    """A longer commitment must move the label; otherwise the argument is ignored."""
    one = label_state(_snap(), factory, table, target_se=1e-4, max_replicates=2048,
                      commit_steps=1)
    many = label_state(_snap(), factory, table, target_se=1e-4, max_replicates=2048,
                       commit_steps=16)
    assert abs(many.advantage) > abs(one.advantage), (
        f"k=16 |A|={abs(many.advantage):.5f} did not exceed k=1 |A|={abs(one.advantage):.5f}"
    )


def test_commit_steps_past_the_horizon_is_safe(factory, table):
    snap = _snap(t=HORIZON - 2)
    r = label_state(snap, factory, table, target_se=1e-3, max_replicates=256,
                    commit_steps=50)
    assert np.isfinite(r.advantage)


def test_label_state_multi_returns_every_requested_horizon(factory, table):
    out = label_state_multi(_snap(), factory, table, commit_steps=(1, 4),
                            target_se=1e-3, max_replicates=256)
    assert set(out) == {1, 4}
    assert all(np.isfinite(v.advantage) for v in out.values())


def test_multi_k_agrees_with_single_k(factory, table):
    """label_state_multi must be exactly label_state per k, not an approximation."""
    multi = label_state_multi(_snap(), factory, table, commit_steps=(4,),
                              target_se=1e-3, max_replicates=256)
    single = label_state(_snap(), factory, table, target_se=1e-3, max_replicates=256,
                         commit_steps=4)
    assert multi[4].advantage == pytest.approx(single.advantage)


# ---- regression guards for mutations that a passing suite did not catch --------


def test_crn_coupling_between_the_two_branches_is_intact(factory, table, reservoir):
    """The two oracle branches must share ONE materialized base.

    This exists because the suite once passed while the branches were being built from
    independent seeds -- the pairing the whole labeller rests on was gone, and nothing
    failed. Measured: with the coupling intact the branches recommend the same arm in
    ~92% of replicates and the paired standard error is ~4.4x smaller than the unpaired
    one; with it broken those become 0% and 1.0x.
    """
    from cold_start.growing.recommend import oracle_prior_from_reservoir

    snap = Snapshot(
        n=np.array([10, 8, 8]), successes=np.array([7, 3, 3]),
        mu=np.array([0.70, 0.40, 0.38]), t=26, horizon=HORIZON, n_draws=3, base_seed=9,
    )
    r = label_state(
        snap, factory, table, target_se=1e-4, max_replicates=1024,
        oracle_prior=oracle_prior_from_reservoir(reservoir),
    )
    assert r.frac_identical > 0.5, (
        f"only {100*r.frac_identical:.1f}% of replicates agree across branches -- "
        f"the common random numbers are not shared"
    )
    assert r.se_unpaired > 3 * r.se, (
        f"pairing bought nothing: unpaired {r.se_unpaired:.6f} vs paired {r.se:.6f}"
    )


@pytest.mark.parametrize("k", [1, 2, 5, 9])
def test_commit_steps_are_applied_exactly(factory, table, k: int):
    """A clamped commitment loop once passed the k-sweep test silently.

    Asserting that |A| grows with k is too weak: capping k at 2 still grows it. The
    count of forced rounds actually applied is the thing to pin.
    """
    r = label_state(_snap(t=20), factory, table, target_se=1e-2, max_replicates=256,
                    commit_steps=k)
    assert r.commit_steps_applied == k, (
        f"asked for {k} forced rounds, applied {r.commit_steps_applied}"
    )


def test_commitment_is_truncated_only_by_the_horizon(factory, table):
    snap = _snap(t=HORIZON - 3)
    r = label_state(snap, factory, table, target_se=1e-2, max_replicates=256,
                    commit_steps=20)
    assert r.commit_steps_applied == 3, r.commit_steps_applied


def test_forced_search_at_the_arm_cap_is_refused_not_zeroed(table, reservoir):
    """The critical failure five independent reviewers found.

    At the live-arm cap a forced SEARCH has nowhere to put its arm. Silently demoting
    it to a REFINE made both branches identical, so the labeller emitted
    A_t = 0.000000 with SE = 0.000000 -- a fabricated label that looks maximally
    precise and stops the adaptive rule on its first batch. Refusing is correct: the
    question is undefined there, not answered by a zero.
    """
    from cold_start.growing.recommend import oracle_prior_from_reservoir
    from cold_start.growing.state import ForcedActionUnavailable

    cap = 6

    def capped(_offset: int) -> Simulator:
        return Simulator(
            table=table, reservoir=reservoir, allocation=LUCB(),
            search_policy=EvidenceGatedSchedule(alpha=0.5, c=1.0, min_pulls_per_arm=2),
            horizon=HORIZON, max_live_arms=cap,
        )

    at_cap = Snapshot(
        n=np.full(cap, 6), successes=np.full(cap, 3), mu=np.linspace(0.3, 0.7, cap),
        t=cap * 6, horizon=HORIZON, n_draws=cap, base_seed=5,
    )
    with pytest.raises(ForcedActionUnavailable):
        label_state(at_cap, capped, table, target_se=1e-3, max_replicates=256,
                    oracle_prior=oracle_prior_from_reservoir(reservoir),
                    max_live_arms=cap)

    below = Snapshot(
        n=np.full(3, 6), successes=np.full(3, 3), mu=np.linspace(0.3, 0.7, 3),
        t=18, horizon=HORIZON, n_draws=3, base_seed=5,
    )
    ok = label_state(below, capped, table, target_se=1e-3, max_replicates=256,
                     oracle_prior=oracle_prior_from_reservoir(reservoir),
                     max_live_arms=cap)
    assert np.isfinite(ok.advantage)
