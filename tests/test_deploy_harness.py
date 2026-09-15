"""Behaviour tests for the CRN-paired deployment harness.

Real simulator runs throughout (no mocks): T <= 60 and M <= 64 keep each cell in the
tens of milliseconds, and `CSTable.load_or_build(60)` is cached. The properties
asserted here are the ones the study's comparisons rest on -- common random numbers
across policies, an attainable comparator, exact budget accounting and an exact
regret decomposition -- so each is checked on the arrays the harness actually emits.
"""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.deploy.comparators import (
    episode_reservoir_prefix,
    expected_best_of_n,
    mu_star_episode,
)
from cold_start.growing.deploy.harness import (
    DYNAMICS_KEYS,
    CellSpec,
    EpisodeResult,
    LogSpec,
    run_cell,
)
from cold_start.growing.deploy.recommenders import (
    PRIMARY_RECOMMENDER,
    RECOMMENDER_NAMES,
    recommend_all_rules,
)
from cold_start.growing.reservoirs import build_reservoir
from cold_start.growing.search_policies import (
    BernoulliSearch,
    DecisionContext,
    PowerSchedule,
    UniformRandom,
)
from cold_start.growing.state import GrowingState
from cold_start.growing.tables import CSTable

HORIZON = 60
ENV_SPEC = {"type": "beta", "params": {"a": 5.0, "b": 2.0}}


@pytest.fixture(scope="module")
def table() -> CSTable:
    return CSTable.load_or_build(HORIZON, alpha=0.05)


def _spec(cap: int = 64, m: int = 64, seed: int = 20260914, horizon: int = HORIZON) -> CellSpec:
    return CellSpec(
        env_id="beta_good_common",
        env_spec=ENV_SPEC,
        horizon=horizon,
        cap=cap,
        base_seed=seed,
        n_replicates=m,
    )


def _check_invariants(res: EpisodeResult, spec: CellSpec) -> None:
    """The per-episode identities every result must satisfy, whatever the policy."""
    state = res.final_state
    assert state is not None
    assert np.all(state.view(state.n).sum(axis=1) == spec.horizon), "budget not conserved"
    assert np.all(res.regret_disc >= 0.0)
    assert np.all(res.best_discovered <= res.mu_star)
    assert np.all(res.best_discovered <= res.mu_star_cap)
    assert np.all(res.mu_star_cap <= res.mu_star)
    for rec in RECOMMENDER_NAMES:
        assert rec in res.q and rec in res.n_rec
        assert np.all(res.q[rec] <= res.best_discovered)
        assert np.allclose(res.regret(rec), res.regret_disc + res.regret_sel(rec), atol=1e-12)
        assert np.all(res.n_rec[rec] >= 1)
    assert np.all(res.k_final == spec.n_initial_arms + np.round(
        res.search_frac * (spec.horizon - spec.n_initial_arms)
    ).astype(int))


# ---- CRN pairing ------------------------------------------------------------------


def test_crn_identity_across_policies(table):
    """The k-th discovered arm is the same reservoir draw under every policy."""
    spec = _spec(cap=64)
    res_a = run_cell(spec, BernoulliSearch(p=1.0), table=table)
    res_b = run_cell(spec, PowerSchedule(alpha=0.5, c=1.0), table=table)

    mu_a = res_a.final_state.view(res_a.final_state.mu)
    mu_b = res_b.final_state.view(res_b.final_state.mu)
    k_min = np.minimum(res_a.k_final, res_b.k_final)
    assert np.all(k_min >= 2)
    assert np.any(res_a.k_final != res_b.k_final), "policies should differ in how far they search"
    for m in range(spec.n_replicates):
        assert np.array_equal(mu_a[m, : k_min[m]], mu_b[m, : k_min[m]])

    # The comparator is a per-episode constant shared by every policy in the cell.
    assert np.array_equal(res_a.mu_star, res_b.mu_star)
    prefix = episode_reservoir_prefix(
        build_reservoir(ENV_SPEC), spec.base_seed, spec.n_replicates, spec.horizon
    )
    assert np.array_equal(res_a.mu_star, prefix[:, : spec.horizon].max(axis=1))
    # ... and the discovered arms are literally rows of that prefix.
    for m in range(spec.n_replicates):
        assert np.array_equal(mu_a[m, : res_a.k_final[m]].astype(np.float64),
                              prefix[m, : res_a.k_final[m]])


def test_shared_cell_constants_reproduce_and_wrong_prefix_is_caught(table):
    """A cell's prefix and oracle prior can be computed once and shared across policies."""
    spec = _spec(cap=32, m=32)
    reservoir = build_reservoir(ENV_SPEC)
    prefix = episode_reservoir_prefix(reservoir, spec.base_seed, spec.n_replicates, spec.horizon)
    prior = (5.0, 2.0)
    policy = PowerSchedule(alpha=0.5, c=1.0)
    own = run_cell(spec, policy, table=table)
    shared = run_cell(
        spec, policy, table=table, reservoir=reservoir, comparator_prefix=prefix, oracle_prior=prior
    )
    assert np.array_equal(own.mu_star, shared.mu_star)
    assert np.array_equal(own.mu_star_cap, shared.mu_star_cap)
    for rec in RECOMMENDER_NAMES:
        assert np.array_equal(own.q[rec], shared.q[rec])

    wrong = episode_reservoir_prefix(reservoir, spec.base_seed + 1, spec.n_replicates, spec.horizon)
    with pytest.raises(RuntimeError, match="CRN pairing broken"):
        run_cell(spec, policy, table=table, comparator_prefix=wrong)
    with pytest.raises(ValueError, match="comparator prefix must be"):
        run_cell(spec, policy, table=table, comparator_prefix=prefix[:, :10])


def test_mu_star_cap_is_prefix_max_over_cap_draws():
    reservoir = build_reservoir(ENV_SPEC)
    prefix = episode_reservoir_prefix(reservoir, 7, 16, 60)
    assert prefix.shape == (16, 60)
    assert np.array_equal(mu_star_episode(reservoir, 7, 16, 60, cap=16), prefix[:, :16].max(axis=1))
    assert np.array_equal(mu_star_episode(reservoir, 7, 16, 60, cap=1000), prefix.max(axis=1))
    # float32 round trip, as the simulator stores mu
    assert np.array_equal(prefix, prefix.astype(np.float32).astype(np.float64))


def test_expected_best_of_n_matches_episode_comparator():
    """The cap-free MC reference agrees with the mean of the CRN comparator."""
    reservoir = build_reservoir(ENV_SPEC)
    ref = expected_best_of_n(reservoir, HORIZON, n_mc=50_000, seed=1)
    ep = mu_star_episode(reservoir, 99, 4000, HORIZON)
    assert abs(ref - ep.mean()) < 3.0 * ep.std(ddof=1) / np.sqrt(ep.size) + 1e-3
    assert expected_best_of_n(reservoir, 1, n_mc=50_000) == pytest.approx(reservoir.mean(), abs=5e-3)


# ---- invariants and cap accounting ---------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [BernoulliSearch(p=1.0), PowerSchedule(alpha=0.5, c=1.0), UniformRandom(), BernoulliSearch(p=0.05)],
    ids=["always_search", "sqrt_t", "uniform", "rare_search"],
)
def test_budget_and_regret_identities(table, policy):
    spec = _spec(cap=32)
    res = run_cell(spec, policy, table=table)
    _check_invariants(res, spec)
    assert res.n_episodes == spec.n_replicates
    # Baselines do not expose `last_decision`, so demotions are unknown, not zero.
    assert np.all(res.n_demoted == -1)


def test_always_search_hits_cap_16_at_t_16(table):
    spec = _spec(cap=16)
    res = run_cell(spec, BernoulliSearch(p=1.0), table=table)
    assert bool(res.cap_hit.all())
    assert np.all(res.t_cap_hit == 16)
    assert np.all(res.k_final == 16)
    assert np.all(res.search_frac == pytest.approx((16 - 2) / (HORIZON - 2)))
    _check_invariants(res, spec)


def test_cap_never_hit_records_minus_one(table):
    spec = _spec(cap=64)
    res = run_cell(spec, PowerSchedule(alpha=0.5, c=1.0), table=table)
    assert not res.cap_hit.any()
    assert np.all(res.t_cap_hit == -1)
    assert np.all(res.k_final < 64)


# ---- the policy hook contract --------------------------------------------------------


class _HookedAlwaysSearch(BernoulliSearch):
    """Always-search that exposes the full hook contract the model policy will use."""

    name = "hooked_always_search"

    def __init__(self) -> None:
        super().__init__(p=1.0)
        self.before: list[int] = []
        self.after: list[tuple[int, int, float]] = []

    def before_step(self, state: GrowingState, t: int) -> None:
        self.before.append(int(t))

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        decision = super().should_search(state, ctx)
        self.last_decision = decision.copy()  # the simulator masks `decision` in place
        return decision

    def after_step(self, state, res, t: int, best_pm: np.ndarray) -> None:
        self.after.append((int(t), int(res.searched.sum()), float(best_pm.mean())))


def test_hooks_are_called_in_order_and_demotions_are_counted(table):
    spec = _spec(cap=16, m=8)
    policy = _HookedAlwaysSearch()
    res = run_cell(spec, policy, table=table)
    assert policy.before == list(range(2, HORIZON))
    assert [t for t, _, _ in policy.after] == list(range(2, HORIZON))
    # Under always-search the cap binds from t=16 on: 60-16 wanted SEARCHes were demoted.
    assert np.all(res.n_demoted == HORIZON - 16)
    searched_per_step = [s for _, s, _ in policy.after]
    assert searched_per_step[: 16 - 2] == [8] * (16 - 2)
    assert searched_per_step[16 - 2 :] == [0] * (HORIZON - 16)
    assert all(0.0 <= pm <= 1.0 for _, _, pm in policy.after)


class _AliasingPolicy(BernoulliSearch):
    name = "aliasing"

    def should_search(self, state, ctx):
        decision = super().should_search(state, ctx)
        self.last_decision = decision  # no copy: the simulator will overwrite it
        return decision


def test_last_decision_alias_is_rejected(table):
    with pytest.raises(RuntimeError, match="store a copy"):
        run_cell(_spec(cap=16, m=4), _AliasingPolicy(p=1.0), table=table)


# ---- output shapes ----------------------------------------------------------------------


def test_to_frame_is_one_row_per_episode_with_documented_columns(table):
    spec = _spec(cap=32, m=16)
    res = run_cell(spec, PowerSchedule(alpha=0.5, c=1.0), table=table)
    df = res.to_frame("sqrt_t", spec)
    assert len(df) == spec.n_replicates
    assert list(df["episode"]) == list(range(spec.n_replicates))
    base = {
        "policy", "env_id", "horizon", "cap", "base_seed", "alpha", "n_initial_arms",
        "policy_seed", "episode", "mu_star", "mu_star_cap", "best_discovered",
        "regret_disc", "k_final", "search_frac", "cap_hit", "t_cap_hit",
        "n_eliminated_final", "herfindahl", "n_singletons_final", "n_demoted",
    }
    assert base <= set(df.columns)
    for rec in RECOMMENDER_NAMES:
        for prefix in ("q_", "n_rec_", "regret_", "regret_sel_", "regret_sup_"):
            assert f"{prefix}{rec}" in df.columns
    assert set(df["policy"]) == {"sqrt_t"}
    assert set(df["env_id"]) == {spec.env_id}
    assert np.allclose(df[f"regret_{PRIMARY_RECOMMENDER}"], res.regret())
    assert np.allclose(df[f"regret_sup_{PRIMARY_RECOMMENDER}"], 1.0 - res.q[PRIMARY_RECOMMENDER])
    assert df["cap_hit"].dtype == bool
    assert not df.isna().any().any()


def test_dynamics_grid_has_grid_plus_one_points_and_monotone_k(table):
    spec = _spec(cap=32, m=32)
    grid = 25
    res = run_cell(spec, UniformRandom(), table=table, dynamics_grid=grid)
    dyn = res.dynamics
    assert dyn is not None
    assert set(dyn) == set(DYNAMICS_KEYS)
    for key, arr in dyn.items():
        assert arr.shape == (grid + 1,), key
    assert np.all(np.diff(dyn["K_t"]) >= 0.0)
    assert np.all(np.diff(dyn["t"]) >= 0.0)
    assert dyn["t"][0] == spec.n_initial_arms and dyn["t"][-1] == spec.horizon
    assert dyn["K_t"][0] == spec.n_initial_arms
    assert dyn["K_t"][-1] == pytest.approx(res.k_final.mean())
    assert dyn["best_discovered"][-1] == pytest.approx(res.best_discovered.mean())
    assert dyn["q_primary"][-1] == pytest.approx(res.q[PRIMARY_RECOMMENDER].mean())
    assert dyn["herfindahl"][-1] == pytest.approx(res.herfindahl.mean())
    assert dyn["n_eliminated"][-1] == pytest.approx(res.n_eliminated_final.mean())
    # Best discovered can only improve; posterior best and quality stay in [0, 1].
    assert np.all(np.diff(dyn["best_discovered"]) >= -1e-12)
    for key in ("best_posterior_mean", "q_primary", "best_discovered"):
        assert np.all((dyn[key] >= 0.0) & (dyn[key] <= 1.0))
    # Search rate is NaN only before the first policy step; elsewhere a fraction.
    rate = dyn["search_rate"]
    assert np.isnan(rate[0])
    finite = rate[np.isfinite(rate)]
    assert finite.size >= grid - 2
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    # Over the whole run the interval rates average to the per-episode search fraction.
    assert np.nanmean(finite) == pytest.approx(res.search_frac.mean(), abs=0.05)


def test_dynamics_can_be_disabled(table):
    res = run_cell(_spec(m=4), PowerSchedule(), table=table, dynamics_grid=0)
    assert res.dynamics is None


def test_log_states_detaches_snapshots_with_consistent_history(table):
    spec = _spec(cap=32, m=16)
    times = (5, 20, 59)
    res = run_cell(
        spec, UniformRandom(), table=table, log_states=LogSpec(times=times, replicates=4)
    )
    assert len(res.snapshots) == len(times) * 4
    for snap in res.snapshots:
        snap.validate()
        assert int(snap.n.sum()) == snap.t
        assert snap.k == len(snap.mu)
        hist = snap.meta["history"]
        assert hist.decisions.shape == (snap.t - spec.n_initial_arms,)
        assert hist.best_mean_trace.shape == (snap.t - spec.n_initial_arms,)
        assert snap.n_draws == snap.k  # no arm is ever dropped, so draws == arms
        assert snap.meta["policy"] == "random_search"
        assert snap.meta["allocation"] == "lucb"
        assert snap.meta["env_id"] == spec.env_id
        assert snap.horizon == spec.horizon
    assert sorted({s.t for s in res.snapshots}) == list(times)
    assert sorted({s.meta["replicate"] for s in res.snapshots}) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="log_states.times"):
        run_cell(spec, UniformRandom(), table=table, log_states=LogSpec(times=(1,), replicates=1))


# ---- policy RNG isolation ----------------------------------------------------------------


def test_policy_seed_isolates_the_policy_rng(table):
    spec = _spec(cap=64, m=32)
    res_1 = run_cell(spec, UniformRandom(), table=table, policy_seed=1)
    res_2 = run_cell(spec, UniformRandom(), table=table, policy_seed=2)
    res_1b = run_cell(spec, UniformRandom(), table=table, policy_seed=1)
    assert res_1.policy_seed == 1 and res_2.policy_seed == 2
    assert not np.array_equal(res_1.k_final, res_2.k_final)
    assert np.array_equal(res_1.k_final, res_1b.k_final)
    assert np.array_equal(res_1.q[PRIMARY_RECOMMENDER], res_1b.q[PRIMARY_RECOMMENDER])
    # Same draws either way: the arm streams are the cell's, not the policy's.
    assert np.array_equal(res_1.mu_star, res_2.mu_star)


def test_default_policy_seed_is_stable_and_named(table):
    spec = _spec(cap=64, m=16)
    policy = UniformRandom()
    res_a = run_cell(spec, policy, table=table)
    res_b = run_cell(spec, UniformRandom(), table=table)
    assert res_a.policy_seed == res_b.policy_seed
    assert res_a.policy_seed is not None
    assert np.array_equal(res_a.k_final, res_b.k_final)
    # The harness replaced the shared `default_rng(0)` on the instance it was given.
    assert policy.rng.bit_generator.state != np.random.default_rng(0).bit_generator.state


# ---- recommenders ------------------------------------------------------------------------


def test_recommend_all_rules_keys_and_oracle_opt_out(table):
    spec = _spec(cap=32, m=8)
    res = run_cell(spec, PowerSchedule(alpha=0.5, c=1.0), table=table)
    state = res.final_state
    with_oracle = recommend_all_rules(state, (5.0, 2.0))
    assert tuple(with_oracle) == RECOMMENDER_NAMES
    without = recommend_all_rules(state, None)
    assert tuple(without) == tuple(n for n in RECOMMENDER_NAMES if n != "oracle_prior")
    assert PRIMARY_RECOMMENDER in without
    for name, rec in with_oracle.items():
        assert rec.cols.shape == (state.M,)
        assert np.array_equal(rec.mu, state.mu[state.row_off + rec.cols].astype(np.float64)), name
