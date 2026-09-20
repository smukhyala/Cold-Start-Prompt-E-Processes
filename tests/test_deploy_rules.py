"""Tests for the deployment policy registry, the P11 reservoir rule and RNG isolation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

from cold_start.growing.allocation import LUCB
from cold_start.growing.deploy import feature_groups as fg
from cold_start.growing.deploy.artifacts import save_model
from cold_start.growing.deploy.model_policy import ModelPolicy
from cold_start.growing.deploy.rules import (
    KINDS,
    POLICY_SPECS,
    ReservoirRule,
    expected_improvement_beta,
    fit_beta_moments_vec,
    make_policy,
    policy_seed,
    wrap_policy_seed,
)
from cold_start.growing.features import _fit_beta_moments
from cold_start.growing.reservoirs import build_reservoir
from cold_start.growing.search_policies import (
    BernoulliSearch,
    DecisionContext,
    EvidenceGatedSchedule,
    PowerSchedule,
    UniformRandom,
)
from cold_start.growing.simulator import Simulator, seed_initial_arms
from cold_start.growing.state import GrowingState
from cold_start.growing.tables import CSTable

T = 40
M = 16
N_INIT = 2
CLOCK = list(fg.FEATURE_SETS["clock"])
RESERVOIR = {"type": "beta", "params": {"a": 5.0, "b": 2.0}}


@pytest.fixture(scope="module")
def table():
    return CSTable.load_or_build(T, alpha=0.05)


@pytest.fixture(scope="module")
def artifact():
    """The smallest valid model artifact: a logistic fit on random CLOCK rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(CLOCK)))
    y = (X[:, 3] > 0).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    return {"pipeline": pipe, "features": CLOCK, "k": 4, "tau": 0.5, "meta": {}}


def _make(name: str, table, params=None, rng=None, **kw):
    return make_policy(
        name, horizon=T, n_replicates=M, table=table, params=params, rng=rng, **kw
    )


def _warm_state(table, seed: int = 11, steps: int = 0, policy=None) -> GrowingState:
    """Warm-started state, optionally advanced `steps` rounds under a coin-flip policy."""
    reservoir = build_reservoir(RESERVOIR)
    state = GrowingState(M, 8, T, base_seed=seed)
    seed_initial_arms(state, reservoir, N_INIT, table)
    if steps:
        sim = Simulator(
            table,
            reservoir,
            LUCB(),
            policy or UniformRandom(rng=np.random.default_rng(seed)),
            T,
            max_live_arms=64,
        )
        for t in range(N_INIT, N_INIT + steps):
            sim.step(state, t)
    return state


def _run(policy, table, *, seed: int = 4242, cap: int = 64, nan_mu: bool = False) -> tuple:
    reservoir = build_reservoir(RESERVOIR)
    state = GrowingState(M, 8, T, base_seed=seed)
    seed_initial_arms(state, reservoir, N_INIT, table)
    sim = Simulator(table, reservoir, LUCB(), policy, T, max_live_arms=cap)
    searched = []
    for t in range(N_INIT, T):
        if nan_mu:
            state.mu[:] = np.nan
        searched.append(sim.step(state, t).searched.copy())
    return state, np.array(searched)


# ---- registry ------------------------------------------------------------------


def test_every_registered_name_constructs_and_decides(table, artifact):
    state = _warm_state(table, steps=10)
    ctx = DecisionContext(t=N_INIT + 10, horizon=T)
    for name, spec in POLICY_SPECS.items():
        assert spec["kind"] in KINDS, name
        params = {"artifact": artifact} if spec["kind"] == "model" else None
        policy = _make(name, table, params=params)
        decision = policy.should_search(state, ctx)
        assert decision.dtype == bool and decision.shape == (M,), name


def test_registry_covers_every_kind():
    assert {spec["kind"] for spec in POLICY_SPECS.values()} == set(KINDS)


def test_unknown_names_are_rejected(table):
    with pytest.raises(KeyError):
        _make("no_such_policy", table)
    with pytest.raises(ValueError, match="artifact"):
        _make("model", table)


def test_params_override_registered_defaults(table):
    policy = _make("power_sqrt", table, params={"c": 2.5})
    assert policy.alpha == 0.5 and policy.c == 2.5
    assert _make("reservoir_rule", table, params={"tau": 0.2}).tau == 0.2
    assert _make("refine_after_init_K8", table).c == 8.0
    assert _make("fixed_K16", table).c == 16.0 and _make("fixed_K16", table).alpha == 0.0


def test_model_kind_from_path_and_dict(table, artifact, tmp_path):
    path = save_model(tmp_path / "clock.joblib", artifact)
    by_name = _make("model:clock.joblib", table, artifact_dir=tmp_path)
    by_param = _make("model", table, params={"artifact": str(path), "tau": 0.3, "k": 1})
    by_dict = _make("model", table, params={"artifact": artifact, "per_step": True})
    for policy in (by_name, by_param, by_dict):
        assert type(policy) is ModelPolicy
        assert policy.features == CLOCK
    assert by_name.k == 4 and by_name.tau == 0.5
    assert by_param.k == 1 and by_param.tau == 0.3
    assert by_dict.per_step and not by_dict.affordability_guard


def test_always_search_hits_the_cap(table):
    policy = _make("always_search", table)
    assert type(policy) is BernoulliSearch and policy.p == 1.0
    state, searched = _run(policy, table, cap=16)
    assert (state.Kt == 16).all()
    assert searched[: 16 - N_INIT].all(), "searches every step until the cap"
    assert not searched[16 - N_INIT :].any(), "then the simulator forces REFINE"


def test_cp0_matches_evidence_gated_schedule(table):
    """The registry's cp0 is the label continuation policy, decision for decision."""
    reference = EvidenceGatedSchedule(alpha=0.5, c=1.0, min_pulls_per_arm=2)
    policy = _make("cp0", table)
    for steps in (0, 5, 12, 25):
        state = _warm_state(table, seed=3 + steps, steps=steps)
        for t in (N_INIT + steps, T - 3, T - 1):
            ctx = DecisionContext(t=t, horizon=T)
            assert np.array_equal(policy.should_search(state, ctx), reference.should_search(state, ctx))


def test_refine_after_init_never_searches_past_k0(table):
    state, searched = _run(_make("refine_after_init_K4", table), table)
    assert (state.Kt == 4).all() and searched.sum() == M * (4 - N_INIT)
    state, searched = _run(_make("refine_after_init_K2", table), table)
    assert (state.Kt == 2).all() and not searched.any()


# ---- reservoir rule ------------------------------------------------------------


def test_beta_moments_match_scalar_fit(table):
    """Every fallback branch of `features._fit_beta_moments`, replicate by replicate."""
    rng = np.random.default_rng(5)
    state = GrowingState(6, 8, T, base_seed=1)
    arms = [1, 2, 3, 5, 8, 4]
    for j in range(max(arms)):
        where = np.array([k > j for k in arms])
        state.add_arms(np.full(6, 0.5, dtype=np.float32), np.full(6, j, dtype=np.int32), where=where)
    n2, s2 = state.view(state.n), state.view(state.S)
    for m, k in enumerate(arms):
        n2[m, :k] = rng.integers(1, 30, size=k)
        s2[m, :k] = rng.integers(0, n2[m, :k] + 1)
    # Row 5: four arms with identical statistics -> zero spread -> Beta(1,1) fallback.
    n2[5, :4] = 10
    s2[5, :4] = 6

    post = state.view(state.empirical_mean())
    a, b = fit_beta_moments_vec(post, state.active_mask())
    for m, k in enumerate(arms):
        a_ref, b_ref = _fit_beta_moments(post[m, :k])
        np.testing.assert_allclose((a[m], b[m]), (a_ref, b_ref), rtol=1e-10, atol=1e-12)
    assert (a[0], b[0]) == (1.0, 1.0) and (a[5], b[5]) == (1.0, 1.0)
    assert not (a[4] == 1.0 and b[4] == 1.0), "eight distinct arms should fit a real Beta"


@pytest.mark.parametrize("a,b,c", [(1.0, 1.0, 0.3), (5.0, 2.0, 0.8), (0.7, 3.0, 0.05), (8.0, 8.0, 0.5)])
def test_expected_improvement_closed_form(a, b, c):
    numeric, _ = quad(lambda x: beta_dist.sf(x, a, b), c, 1.0)
    closed = expected_improvement_beta(np.array([a]), np.array([b]), np.array([c]))[0]
    assert abs(closed - numeric) < 1e-9
    assert expected_improvement_beta(np.array([a]), np.array([b]), np.array([1.0]))[0] == 0.0


def test_reservoir_rule_searches_more_with_smaller_tau(table):
    state = _warm_state(table, steps=10)
    ctx = DecisionContext(t=N_INIT + 10, horizon=T)
    loose = ReservoirRule(tau=0.05).should_search(state, ctx)
    tight = ReservoirRule(tau=5.0).should_search(state, ctx)
    assert (loose | ~tight).all(), "a tighter tau can only remove SEARCH decisions"
    assert loose.sum() > tight.sum()

    _, searched_loose = _run(_make("reservoir_rule", table, params={"tau": 0.05}), table)
    _, searched_tight = _run(_make("reservoir_rule", table, params={"tau": 5.0}), table)
    assert searched_loose.sum() > searched_tight.sum()


def test_reservoir_rule_closed_form_decision(table):
    """SEARCH iff I_hat * (T - t) > tau * leader_width, with I_hat from the Beta fit."""
    state = _warm_state(table, steps=15)
    t = N_INIT + 15
    # The ratio I_hat * (T - t) / width sits in ~[0.05, 0.45] across these replicates.
    rule = ReservoirRule(tau=0.2)
    decision = rule.should_search(state, DecisionContext(t=t, horizon=T))

    post = state.view(state.empirical_mean())
    active = state.active_mask()
    masked = np.where(active, post, -np.inf)
    leader = masked.argmax(axis=1)
    lin = state.row_off + leader
    width = state.ucb[lin].astype(np.float64) - state.lcb[lin].astype(np.float64)
    a, b = fit_beta_moments_vec(post, active)
    i_hat = expected_improvement_beta(a, b, masked.max(axis=1))
    expected = i_hat * (T - t) > 0.2 * width
    assert np.array_equal(decision, expected)
    assert np.array_equal(rule.last_decision, decision)
    np.testing.assert_allclose(rule.last_i_hat, i_hat)
    assert decision.any() and not decision.all(), "state should be mixed for a useful test"


def test_reservoir_rule_never_reads_hidden_means(table):
    _, clean = _run(_make("reservoir_rule", table), table)
    _, blind = _run(_make("reservoir_rule", table), table, nan_mu=True)
    assert np.array_equal(clean, blind)


def test_reservoir_rule_forces_search_with_no_arms():
    state = GrowingState(4, 2, T, base_seed=0)
    assert ReservoirRule().should_search(state, DecisionContext(t=0, horizon=T)).all()


# ---- RNG isolation -------------------------------------------------------------


def test_wrap_policy_seed_is_reproducible_and_name_keyed(table):
    state = _warm_state(table)
    ctx = DecisionContext(t=N_INIT, horizon=T)

    def draws(policy):
        return np.array([policy.should_search(state, ctx) for _ in range(5)])

    same_a = draws(wrap_policy_seed(_make("uniform", table), 123, "uniform"))
    same_b = draws(wrap_policy_seed(_make("uniform", table), 123, "uniform"))
    other_name = draws(wrap_policy_seed(_make("uniform", table), 123, "uniform_b"))
    other_seed = draws(wrap_policy_seed(_make("uniform", table), 124, "uniform"))
    assert np.array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other_name)
    assert not np.array_equal(same_a, other_seed)
    assert policy_seed(123, "uniform") != policy_seed(123, "uniform_b")


def test_wrap_policy_seed_returns_the_same_object(table):
    policy = _make("power_sqrt", table)
    assert wrap_policy_seed(policy, 1, "power_sqrt") is policy
    assert type(policy) is PowerSchedule
