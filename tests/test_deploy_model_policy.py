"""Behaviour tests for the deployable model-backed SEARCH policy.

A tiny logistic model over the 11 CLOCK columns is fitted on fabricated rows and saved
through the real artifact contract; every test then runs a real `Simulator` on a
Beta(5,2) reservoir at T=40 and inspects what the policy wanted (`last_decision`),
what it believed (`last_proba`) and how it committed.
"""

from __future__ import annotations

import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from cold_start.growing.allocation import LUCB
from cold_start.growing.deploy import feature_groups as fg
from cold_start.growing.deploy.artifacts import load_model, save_model
from cold_start.growing.evidence import PairwiseEvidence
from cold_start.growing.features import extract_features
from cold_start.growing.labeling import Snapshot, materialize
from cold_start.growing.reservoirs import build_reservoir
from cold_start.growing.search_policies import DecisionContext
from cold_start.growing.simulator import Simulator, seed_initial_arms
from cold_start.growing.state import GrowingState
from cold_start.growing.tables import CSTable


def _install_features_vec_stub() -> bool:
    """Stand in for M1's `features_vec` only while it is absent from the tree.

    A per-replicate loop over the scalar `features.extract_features`, so the values are
    the corpus definitions by construction. Never reads `state.mu`.
    """
    try:
        import cold_start.growing.deploy.features_vec  # noqa: F401

        return False
    except ImportError:
        pass

    def extract_features_vec(state, t, horizon, table, pairwise, history=None, columns=None):
        pairwise = pairwise if pairwise is not None else PairwiseEvidence()
        n2, s2 = state.view(state.n), state.view(state.S)
        rows = []
        for m in range(state.M):
            k = int(state.Kt[m])
            hist = history.scalar(m) if history is not None else None
            rows.append(
                extract_features(
                    n2[m, :k], s2[m, :k], np.zeros(k), t, horizon, table, pairwise, history=hist
                )
            )
        keys = list(columns) if columns is not None else list(rows[0].keys())
        return {c: np.array([r[c] for r in rows], dtype=np.float64) for c in keys}

    def feature_matrix(feats, columns):
        return np.column_stack([np.asarray(feats[c], dtype=np.float64) for c in columns])

    mod = types.ModuleType("cold_start.growing.deploy.features_vec")
    mod.extract_features_vec = extract_features_vec
    mod.feature_matrix = feature_matrix
    sys.modules[mod.__name__] = mod
    return True


USING_STUB = _install_features_vec_stub()

from cold_start.growing.deploy.model_policy import ModelPolicy  # noqa: E402

T = 40
M = 8
N_INIT = 2
CLOCK = list(fg.FEATURE_SETS["clock"])
RESERVOIR = {"type": "beta", "params": {"a": 5.0, "b": 2.0}}


# ---- fixtures ------------------------------------------------------------------


def _clock_row(t: int, horizon: int, k: int) -> list[float]:
    """The 11 CLOCK columns exactly as `features.extract_features` defines them."""
    remaining = horizon - t
    return [
        float(t),
        float(horizon),
        float(remaining),
        remaining / horizon,
        t / horizon,
        float(k),
        k / t if t else float(k),
        k / horizon,
        float(np.log(max(t, 1))),
        float(np.log(k)),
        k / np.sqrt(max(t, 1)),
    ]


def _fit_clock_pipeline(seed: int = 0, n_rows: int = 3000):
    """Logistic model that wants to SEARCH early and stop as arms accumulate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for _ in range(n_rows):
        horizon = int(rng.choice([40, 100, 200]))
        t = int(rng.integers(2, horizon))
        k = int(rng.integers(1, min(t, 64) + 1))
        row = _clock_row(t, horizon, k)
        # remaining_frac minus a K/sqrt(t) penalty, with enough label noise that the
        # fitted probability falls off gradually rather than as a step.
        score = row[3] - 0.15 * row[10] + 0.3 * rng.normal()
        rows.append(row)
        labels.append(score > 0.35)
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


@pytest.fixture(scope="module")
def table():
    return CSTable.load_or_build(T, alpha=0.05)


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "clock_k4.joblib"
    save_model(
        path,
        {
            "pipeline": _fit_clock_pipeline(),
            "features": CLOCK,
            "k": 4,
            "tau": 0.5,
            "meta": {"variant": "test_clock"},
        },
    )
    return path


@pytest.fixture(scope="module")
def artifact(artifact_path):
    return load_model(artifact_path)


def _policy(artifact, table, **kw) -> ModelPolicy:
    return ModelPolicy(artifact, horizon=T, n_replicates=M, table=table, **kw)


# ---- driving the simulator -----------------------------------------------------


@dataclass
class Trace:
    decisions: np.ndarray  # (T - N_INIT, M) what the policy wanted
    proba: np.ndarray  # (T - N_INIT, M) P(SEARCH) where freshly decided, else NaN
    kt_before: np.ndarray  # (T - N_INIT, M) arm count at decision time
    searched: np.ndarray  # (T - N_INIT, M) what the simulator did
    state: GrowingState

    def step(self, t: int) -> int:
        return t - N_INIT


def _run(policy: ModelPolicy, table, *, seed: int = 4242, cap: int = 64, nan_mu: bool = False) -> Trace:
    reservoir = build_reservoir(RESERVOIR)
    state = GrowingState(M, 8, T, base_seed=seed)
    seed_initial_arms(state, reservoir, N_INIT, table)
    sim = Simulator(table, reservoir, LUCB(), policy, T, max_live_arms=cap)
    decisions, proba, kt, searched = [], [], [], []
    for t in range(N_INIT, T):
        if nan_mu:
            # Rewards come from `state.thresh`, so the trajectory is unchanged; only a
            # policy that peeks at the hidden means would notice.
            state.mu[:] = np.nan
        kt.append(state.Kt.copy())
        res = sim.step(state, t)
        policy.after_step(state, res, t, sim.best_posterior_mean(state))
        decisions.append(policy.last_decision.copy())
        proba.append(policy.last_proba.copy())
        searched.append(res.searched.copy())
    return Trace(np.array(decisions), np.array(proba), np.array(kt), np.array(searched), state)


def _scalar_proba(pipeline, state: GrowingState, t: int, table, pairwise) -> np.ndarray:
    """P(SEARCH) per replicate via the scalar corpus feature path."""
    n2, s2 = state.view(state.n), state.view(state.S)
    out = np.empty(state.M)
    for m in range(state.M):
        k = int(state.Kt[m])
        row = extract_features(n2[m, :k], s2[m, :k], np.zeros(k), t, T, table, pairwise)
        X = np.asarray([[row[c] for c in CLOCK]], dtype=np.float64)
        out[m] = pipeline.predict_proba(X)[0, 1]
    return out


# ---- tests ---------------------------------------------------------------------


def test_defaults_come_from_the_artifact(artifact, table):
    policy = _policy(artifact, table)
    assert policy.k == 4 and policy.tau == 0.5 and policy.features == CLOCK
    assert policy.history is None, "a CLOCK-only model must not build a history"
    assert policy.pairwise is None, "a model without f_log_e_pair must not build the table"
    assert _policy(artifact, table, tau=0.3, k=1).tau == 0.3
    assert _policy(artifact, table, tau=0.3, k=1).k == 1


def test_k1_equals_per_step_and_matches_predict_proba(artifact, table):
    """With k=1 every step is a fresh decision, identical to per-step use."""
    a = _run(_policy(artifact, table, k=1), table)
    b = _run(_policy(artifact, table, k=4, per_step=True), table)
    assert np.array_equal(a.decisions, b.decisions)
    assert np.allclose(a.proba, b.proba, equal_nan=True)
    assert np.isfinite(a.proba).all(), "k=1 must consult the model at every step"
    assert np.array_equal(a.decisions, a.proba > 0.5)

    # Replaying each state through the scalar corpus features reproduces `last_proba`.
    policy = _policy(artifact, table, k=1)
    reservoir = build_reservoir(RESERVOIR)
    state = GrowingState(M, 8, T, base_seed=4242)
    seed_initial_arms(state, reservoir, N_INIT, table)
    sim = Simulator(table, reservoir, LUCB(), policy, T, max_live_arms=64)
    pairwise = PairwiseEvidence()
    for t in range(N_INIT, T):
        expected = _scalar_proba(artifact["pipeline"], state, t, table, pairwise)
        sim.step(state, t)
        np.testing.assert_allclose(policy.last_proba, expected, rtol=1e-9, atol=1e-12)
    assert policy.n_decisions == M * (T - N_INIT)
    assert policy.n_committed_steps == 0


@pytest.mark.parametrize("horizon", [40, 41])
def test_k4_commitment_holds_and_truncates_at_horizon(artifact, table, horizon):
    """A decision at t is held for exactly min(4, T - t) steps."""
    policy = ModelPolicy(artifact, horizon=horizon, n_replicates=M, table=table, k=4)
    reservoir = build_reservoir(RESERVOIR)
    state = GrowingState(M, 8, horizon, base_seed=7)
    seed_initial_arms(state, reservoir, N_INIT, table)
    sim = Simulator(table, reservoir, LUCB(), policy, horizon, max_live_arms=64)
    decisions, proba, commit_left = [], [], []
    for t in range(N_INIT, horizon):
        sim.step(state, t)
        decisions.append(policy.last_decision.copy())
        proba.append(policy.last_proba.copy())
        commit_left.append(policy.commit_left.copy())
    decisions, proba, commit_left = np.array(decisions), np.array(proba), np.array(commit_left)

    fresh_steps = list(range(N_INIT, horizon, 4))
    for t in range(N_INIT, horizon):
        i = t - N_INIT
        if t in fresh_steps:
            assert np.isfinite(proba[i]).all(), f"t={t} should be a fresh decision"
            block = decisions[i : i + min(4, horizon - t)]
            assert (block == decisions[i]).all(), f"commitment broken inside block at t={t}"
            # `min(k, T - t)` rounds remain, one of which was just spent.
            assert (commit_left[i] == min(4, horizon - t) - 1).all()
        else:
            assert np.isnan(proba[i]).all(), f"t={t} should follow the commitment"
    assert (commit_left[-1] == 0).all(), "no commitment may outlive the horizon"
    last_fresh = fresh_steps[-1]
    assert (commit_left[last_fresh - N_INIT] == horizon - last_fresh - 1).all()
    assert policy.n_decisions == M * len(fresh_steps)
    assert policy.n_committed_steps == M * (horizon - N_INIT - len(fresh_steps))


def test_tau_monotonicity(artifact, table):
    """A lower threshold can only turn more model evaluations into SEARCH."""
    lo = _policy(artifact, table, k=1, tau=0.3)
    hi = _policy(artifact, table, k=1, tau=0.7)
    _run(lo, table)
    _run(hi, table)
    assert lo.n_search_decided >= hi.n_search_decided
    assert hi.n_search_decided > 0 and lo.n_search_decided < lo.n_decisions
    assert lo.n_search_decided > hi.n_search_decided, "the synthetic model is too steep to test"


def test_affordability_guard_never_searches_when_unaffordable(artifact, table):
    """`tau=0` wants to SEARCH every step; the guard must refuse once budget is short."""
    guarded = _policy(artifact, table, k=1, tau=0.0, affordability_guard=True)
    trace = _run(guarded, table)
    remaining = (T - np.arange(N_INIT, T))[:, None]
    unaffordable = remaining < 2 * (trace.kt_before + 1)
    assert unaffordable.any(), "the run must reach an unaffordable region"
    assert not (trace.decisions & unaffordable).any()
    assert (trace.decisions | unaffordable).all(), "affordable steps were wanted"
    assert guarded.n_guard_vetoes == int(unaffordable.sum())

    unguarded = _policy(artifact, table, k=1, tau=0.0)
    assert _run(unguarded, table).decisions.all()


def test_guard_can_veto_a_committed_search(artifact, table):
    """The guard is applied after the commitment, so a committed SEARCH can be refused."""
    policy = _policy(artifact, table, k=4, tau=0.0, affordability_guard=True)
    trace = _run(policy, table)
    remaining = (T - np.arange(N_INIT, T))[:, None]
    unaffordable = remaining < 2 * (trace.kt_before + 1)
    committed = np.isnan(trace.proba)
    assert (committed & unaffordable).any()
    assert not (trace.decisions & unaffordable).any()


def test_predict_proba_parity_on_materialized_snapshot(artifact, table):
    """The policy on a materialized state equals the pipeline on the scalar features."""
    snap = Snapshot(
        n=np.array([12, 7, 3, 1]),
        successes=np.array([9, 4, 2, 1]),
        mu=np.array([0.7, 0.6, 0.5, 0.8]),
        t=23,
        horizon=T,
        n_draws=4,
        base_seed=99,
    )
    state = materialize(snap, M, table)
    policy = _policy(artifact, table, k=1)
    decision = policy.should_search(state, DecisionContext(t=snap.t, horizon=T))

    row = extract_features(
        snap.n, snap.successes, snap.mu, snap.t, snap.horizon, table, PairwiseEvidence()
    )
    X = np.asarray([[row[c] for c in CLOCK]], dtype=np.float64)
    expected = artifact["pipeline"].predict_proba(X)[0, 1]
    np.testing.assert_allclose(policy.last_proba, np.full(M, expected), rtol=1e-9, atol=1e-12)
    assert np.array_equal(decision, np.full(M, expected > 0.5))
    assert np.array_equal(policy.last_decision, decision)


def test_hidden_means_are_never_read(artifact, table):
    clean = _run(_policy(artifact, table, k=4), table)
    blind = _run(_policy(artifact, table, k=4), table, nan_mu=True)
    assert np.array_equal(clean.decisions, blind.decisions)
    assert np.allclose(clean.proba, blind.proba, equal_nan=True)
    assert np.array_equal(clean.searched, blind.searched)


def test_after_step_counts_cap_demotions(artifact, table):
    policy = _policy(artifact, table, k=1, tau=0.0)
    trace = _run(policy, table, cap=6)
    assert trace.state.Kt.max() == 6
    demoted = int((trace.decisions & ~trace.searched).sum())
    assert demoted > 0 and policy.n_cap_demoted == demoted
    assert policy.counters()["n_cap_demoted"] == demoted


def test_history_is_built_only_for_history_models(artifact, table):
    hist_features = CLOCK + ["f_search_frac_last_10", "f_time_since_last_search"]
    rows = np.random.default_rng(1).normal(size=(60, len(hist_features)))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(rows, rows[:, 0] > 0)
    hist_artifact = {"pipeline": pipe, "features": hist_features, "k": 1, "tau": 0.5, "meta": {}}
    policy = _policy(hist_artifact, table)
    assert policy.history is not None
    _run(policy, table)
    assert policy.history.length == T - N_INIT


def test_dataframe_fitted_pipeline_is_fed_named_columns(artifact, table):
    """A pipeline fitted on a DataFrame must not warn at every step, nor drift."""
    import pandas as pd
    from sklearn.base import clone

    rng = np.random.default_rng(3)
    rows = [_clock_row(int(t), T, int(k)) for t, k in zip(rng.integers(2, T, 200), rng.integers(1, 20, 200), strict=True)]
    frame = pd.DataFrame(rows, columns=CLOCK)
    y = frame["f_remaining_frac"] > 0.5
    named = clone(artifact["pipeline"]).fit(frame, y)
    policy = _policy({"pipeline": named, "features": CLOCK, "k": 1, "tau": 0.5, "meta": {}}, table)
    assert policy._as_frame
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trace = _run(policy, table)
    assert np.isfinite(trace.proba).all()

    reordered = list(reversed(CLOCK))
    with pytest.raises(ValueError, match="fitted feature names"):
        _policy({"pipeline": named, "features": reordered, "k": 1, "tau": 0.5, "meta": {}}, table)


def test_rejects_bad_artifacts(artifact, table):
    with pytest.raises(ValueError):
        _policy({**artifact, "features": CLOCK + ["oracle_mu_star"]}, table)
    with pytest.raises(ValueError):
        _policy(artifact, table, k=0)
    with pytest.raises(ValueError):
        _policy(artifact, table, tau=float("nan"))
