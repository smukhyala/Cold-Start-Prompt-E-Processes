"""Parity gate: the vectorized deployment features must equal the scalar corpus features.

A learned policy is only as good as the agreement between the features it was
trained on (`features.extract_features`, one snapshot at a time) and the features it
is fed at deployment (`deploy.features_vec.extract_features_vec`, on the batched
state). A previous attempt drifted on one column and silently invalidated the whole
comparison, so this file checks every column, on states reached three different ways:

* harvested snapshots broadcast through `materialize` (identical rows),
* a live batched trajectory with per-row histories and differing arm counts,
* hand-built batches that hit the edge cases a harvest never visits (K = 1, ties,
  identical arms, unpulled arms, the live-arm cap).

Failure messages report the max abs diff per column so a drift is located, not just
detected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "growing_bandits"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from generate_states import GenSpec, harvest  # noqa: E402
from label_states import ALLOCATIONS, policy_by_name  # noqa: E402

from cold_start.growing.allocation import LUCB  # noqa: E402
from cold_start.growing.deploy import feature_groups as fg  # noqa: E402
from cold_start.growing.deploy.features_vec import (  # noqa: E402
    _beta_sf,
    extract_features_vec,
    feature_matrix,
)
from cold_start.growing.deploy.history_vec import VecSearchHistory  # noqa: E402
from cold_start.growing.deploy.pairwise_table import (  # noqa: E402
    CachedPairwiseTable,
    ChunkedPairwise,
    get_pairwise_table,
)
from cold_start.growing.evidence import PairwiseEvidence  # noqa: E402
from cold_start.growing.features import SearchHistory, extract_features  # noqa: E402
from cold_start.growing.labeling import Snapshot, materialize  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.search_policies import PowerSchedule  # noqa: E402
from cold_start.growing.simulator import Simulator, seed_initial_arms  # noqa: E402
from cold_start.growing.state import EMPTY_UID, GrowingState  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

ALL = fg.ALL_DEPLOYABLE
RTOL = 1e-5
ATOL = 1e-6
# The cached pairwise table stores its cover terms in float32.
LOGE_ATOL = 1e-3

HORIZONS = (100, 200)
POLICIES = ("sqrt", "aggressive", "bracket", "conservative")
ALLOCS = ("lucb", "ucb")
RESERVOIRS = {
    "beta_5_2": {"type": "beta", "params": {"a": 5.0, "b": 2.0}},
    "tail_b2_mu1_c1": {"type": "tail", "params": {"beta": 2.0, "mu_star": 1.0, "c": 1.0}},
}


def _atol(col: str) -> float:
    return LOGE_ATOL if col == "f_log_e_pair" else ATOL


def _diff_table(worst: dict[str, float]) -> str:
    lines = ["max abs diff per column (vectorized vs scalar):"]
    for col, d in sorted(worst.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {col:40s} {d:.3e}")
    return "\n".join(lines)


def _compare_rows(
    vec: dict[str, np.ndarray],
    scalar_rows: list[dict[str, float]],
    worst: dict[str, float],
    failures: list[str],
    where: str,
) -> None:
    """Check row m of `vec` against `scalar_rows[m]` for every column, accumulating."""
    assert set(vec) == set(ALL)
    M = len(scalar_rows)
    for col in ALL:
        v = vec[col]
        assert v.shape == (M,) and v.dtype == np.float64, (col, v.shape, v.dtype)
        s = np.array([row[col] for row in scalar_rows], dtype=np.float64)
        d = float(np.max(np.abs(v - s)))
        worst[col] = max(worst[col], d)
        if not np.allclose(v, s, rtol=RTOL, atol=_atol(col)):
            failures.append(f"{col} @ {where}: scalar={s.tolist()} vec={v.tolist()}")


def _scalar_features(state: GrowingState, m: int, t: int, T: int, table, hist) -> dict:
    """The corpus extractor on replicate `m`, detached exactly as `_detach` does."""
    active = state.view(state.uid)[m] != EMPTY_UID
    return extract_features(
        state.view(state.n)[m, active],
        state.view(state.S)[m, active],
        state.view(state.mu)[m, active],
        t,
        T,
        table,
        PairwiseEvidence(),
        history=hist,
    )


# ---- fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def pairwise_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("pairwise_tables")


@pytest.fixture(scope="module")
def tables() -> dict[int, CSTable]:
    return {T: CSTable.load_or_build(T) for T in HORIZONS}


@pytest.fixture(scope="module")
def pair_tables(pairwise_cache) -> dict[int, CachedPairwiseTable]:
    return {T: get_pairwise_table(T, cache_dir=pairwise_cache) for T in HORIZONS}


@pytest.fixture(scope="module")
def corpus(tables) -> list[tuple[int, Snapshot]]:
    """Harvested snapshots across horizons, policies, allocations and reservoirs."""
    out: list[tuple[int, Snapshot]] = []
    seed = 0
    for T in HORIZONS:
        for env_id, spec in RESERVOIRS.items():
            reservoir = build_reservoir(spec)
            for policy_name in POLICIES:
                for alloc_name in ALLOCS:
                    seed += 1
                    snaps = harvest(
                        reservoir,
                        tables[T],
                        policy_by_name(policy_name, seed),
                        ALLOCATIONS[alloc_name](),
                        GenSpec(
                            horizon=T,
                            n_trajectories=8,
                            snapshot_times=tuple(range(8)),
                            max_live_arms=64,
                            initial_arms=2,
                            seed=20260910 + seed,
                        ),
                        env_id,
                    )
                    out.extend((T, s) for s in snaps)
    return out


# ---- (a) materialize route -----------------------------------------------------


def test_corpus_is_large_and_diverse(corpus):
    assert len(corpus) >= 200
    ks = np.array([snap.k for _, snap in corpus])
    ts = np.array([snap.t / snap.horizon for _, snap in corpus])
    lengths = np.array([snap.meta["history"].decisions.size for _, snap in corpus])
    assert ks.min() == 2 and ks.max() >= 20, (ks.min(), ks.max())
    assert ts.min() < 0.15 and ts.max() > 0.85
    # Windows of 10/25/50 must be exercised both unsaturated and saturated.
    assert lengths.min() < 10 and lengths.max() > 50


def test_materialize_route_parity(corpus, tables, pair_tables):
    worst = {c: 0.0 for c in ALL}
    failures: list[str] = []
    for T, snap in corpus:
        table = tables[T]
        state = materialize(snap, 3, table, 0)
        hist_scalar = snap.meta["history"]
        hist = VecSearchHistory.from_scalar([hist_scalar] * 3, capacity=T)
        vec = extract_features_vec(state, snap.t, snap.horizon, table, pair_tables[T], history=hist)
        scalar = extract_features(
            snap.n,
            snap.successes,
            snap.mu,
            snap.t,
            snap.horizon,
            table,
            PairwiseEvidence(),
            history=hist_scalar,
        )
        _compare_rows(vec, [scalar] * 3, worst, failures, f"T={T} t={snap.t} k={snap.k}")
    assert not failures, (
        f"{len(failures)} column mismatches; first: {failures[:5]}\n{_diff_table(worst)}"
    )
    assert worst["f_log_e_pair"] < LOGE_ATOL, _diff_table(worst)


# ---- (b) live-loop route -------------------------------------------------------


def test_live_loop_history_parity(tables, pair_tables):
    """Per-row histories recorded step by step, checked at six times in one run.

    Replicates `harvest`'s bookkeeping (`decisions` / `best_trace` lists appended
    after every step) and detaches scalar histories from it exactly as `_detach` does.
    """
    T, M = 100, 16
    table = tables[T]
    reservoir = build_reservoir(RESERVOIRS["beta_5_2"])
    state = GrowingState(n_replicates=M, capacity=8, horizon=T, base_seed=777)
    sim = Simulator(
        table=table,
        reservoir=reservoir,
        allocation=LUCB(),
        search_policy=PowerSchedule(0.5, 1.0, rng=np.random.default_rng(3)),
        horizon=T,
        max_live_arms=64,
    )
    seed_initial_arms(state, reservoir, 2, table)
    vh = VecSearchHistory(M, capacity=T)
    decisions: list[np.ndarray] = []
    best_trace: list[np.ndarray] = []
    check_times = {2, 5, 17, 40, 73, 99}
    worst = {c: 0.0 for c in ALL}
    failures: list[str] = []
    n_checked = 0

    for t in range(2, T):
        if t in check_times:
            vec = extract_features_vec(state, t, T, table, pair_tables[T], history=vh)
            dec = np.array(decisions).T if decisions else np.zeros((M, 0), dtype=bool)
            trace = np.array(best_trace).T if best_trace else np.zeros((M, 0))
            scalar_rows = []
            for m in range(M):
                sh = SearchHistory(decisions=dec[m].copy(), best_mean_trace=trace[m].copy())
                detached = vh.scalar(m)
                assert np.array_equal(detached.decisions, sh.decisions)
                assert np.allclose(detached.best_mean_trace, sh.best_mean_trace, rtol=0, atol=0)
                scalar_rows.append(_scalar_features(state, m, t, T, table, sh))
            _compare_rows(vec, scalar_rows, worst, failures, f"live t={t}")
            n_checked += 1
        res = sim.step(state, t)
        decisions.append(res.searched.copy())
        best_trace.append(sim.best_posterior_mean(state).copy())
        vh.record(res.searched, sim.best_posterior_mean(state))

    assert n_checked == 6
    assert vh.length == T - 2
    assert not failures, (
        f"{len(failures)} column mismatches; first: {failures[:5]}\n{_diff_table(worst)}"
    )
    # The nine history columns are exact up to the float32 trace the simulator emits.
    for col in fg.HISTORY:
        assert worst[col] < 1e-6, (col, worst[col])


# ---- hand-built edge cases -----------------------------------------------------


def _hand_state(table, T: int, arms: list[tuple[list[int], list[int]]], seed: int) -> GrowingState:
    """One replicate per entry of `arms`, each `(n_list, S_list)`; bounds from `table`."""
    M = len(arms)
    kmax = max(len(n) for n, _ in arms)
    state = GrowingState(n_replicates=M, capacity=max(kmax, 2), horizon=T, base_seed=seed)
    rng = np.random.default_rng(seed)
    for j in range(kmax):
        where = np.array([len(n) > j for n, _ in arms])
        state.add_arms(
            rng.random(M).astype(np.float32), np.full(M, j, dtype=np.int32), where=where
        )
    n2, s2 = state.view(state.n), state.view(state.S)
    for m, (n, S) in enumerate(arms):
        n2[m, : len(n)] = n
        s2[m, : len(S)] = S
    idx = n2.astype(np.int64) * table.stride + s2.astype(np.int64)
    active = state.active_mask()
    state.view(state.lcb)[active] = np.asarray(table.lower_flat)[idx[active]]
    state.view(state.ucb)[active] = np.asarray(table.upper_flat)[idx[active]]
    return state


def test_edge_case_batch_parity(tables, pair_tables):
    """Mixed arm counts in one batch, including cases a harvest never produces."""
    T = 100
    table = tables[T]
    arms: list[tuple[list[int], list[int]]] = [
        ([7], [5]),  # K = 1: challenger is the leader, second = incumbent, no fit
        ([1], [0]),  # K = 1 singleton at the floor
        ([0, 0], [0, 0]),  # unpulled arms: bounds [0, 1], posts tied at 0.5
        ([5, 5, 5], [3, 3, 3]),  # identical arms: spread 0, sd 0, argmax ties
        ([4, 4], [2, 2]),  # K = 2 tie
        ([10, 3, 8], [9, 1, 4]),  # K = 3: Hill estimator switches on
        ([10, 10, 10, 10], [9, 8, 1, 0]),  # separated leader
        ([20, 1, 1, 1, 1, 1], [15, 1, 1, 0, 1, 0]),  # many singletons
    ]
    rng = np.random.default_rng(9)
    for K in (5, 9, 16, 30, 64):  # 64 is the live-arm cap
        n = rng.integers(1, 30, size=K)
        S = rng.integers(0, n + 1)
        arms.append((n.tolist(), S.tolist()))
    state = _hand_state(table, T, arms, seed=5)
    exact = PairwiseEvidence()
    worst = {c: 0.0 for c in ALL}
    failures: list[str] = []
    for t in (3, 60):
        for pairwise in (pair_tables[T], exact):
            vec = extract_features_vec(state, t, T, table, pairwise, history=None)
            rows = [_scalar_features(state, m, t, T, table, None) for m in range(len(arms))]
            _compare_rows(vec, rows, worst, failures, f"edge t={t}")
    assert not failures, (
        f"{len(failures)} column mismatches; first: {failures[:5]}\n{_diff_table(worst)}"
    )
    # With the exact pairwise object the only residual is float64 summation order
    # (~1e-16 on mean-derived columns): the float32 table is the sole 1e-7 source.
    vec = extract_features_vec(state, 60, T, table, exact)
    rows = [_scalar_features(state, m, 60, T, table, None) for m in range(len(arms))]
    for col in ALL:
        assert np.allclose(vec[col], [r[col] for r in rows], rtol=0, atol=1e-12), col


def test_beta_sf_matches_scipy_including_tiny_tails():
    """The fast survival function must reproduce `scipy.stats.beta.sf` everywhere.

    Covers the Beta(1,1) fallback, the moment-fit range `nu <= 64`, `x == 1`, and tails
    far below 1e-6 where naive `1 - cdf` would collapse to rounding noise.
    """
    from scipy.stats import beta

    rng = np.random.default_rng(21)
    a = np.concatenate([np.ones(500), rng.uniform(1.0, 64.0, 4000), rng.uniform(30.0, 64.0, 1500)])
    b = np.concatenate([np.ones(500), rng.uniform(1.0, 64.0, 4000), rng.uniform(1.0, 8.0, 1500)])
    x = np.concatenate([rng.uniform(0.0, 1.0, 500), rng.uniform(0.0, 1.0, 4000), rng.uniform(0.99, 1.0, 1500)])
    x[::97] = 1.0
    want = beta.sf(x, a, b)
    got = _beta_sf(x, a, b)
    assert np.max(np.abs(got - want)) < 1e-14
    tiny = want < 1e-6
    assert tiny.sum() > 100, "the test must actually exercise the tiny-tail branch"
    assert np.array_equal(got[tiny], want[tiny])
    # Fallback rows are exact too: sf(x; 1, 1) == 1 - x on both paths.
    assert np.array_equal(got[:500], want[:500])


def test_features_refuse_a_replicate_with_no_arms(tables, pair_tables):
    state = GrowingState(n_replicates=2, capacity=4, horizon=100, base_seed=1)
    with pytest.raises(ValueError, match="no arms"):
        extract_features_vec(state, 3, 100, tables[100], pair_tables[100])


# ---- (c) pairwise table --------------------------------------------------------


def _random_pairs(rng: np.random.Generator, max_n: int, count: int):
    n_l = rng.integers(0, max_n + 1, count)
    s_l = rng.integers(0, n_l + 1)
    n_c = rng.integers(0, max_n + 1, count)
    s_c = rng.integers(0, n_c + 1)
    return n_l, s_l, n_c, s_c


def test_pairwise_table_matches_exact_and_round_trips(tmp_path):
    exact = PairwiseEvidence()
    tab = get_pairwise_table(120, cache_dir=tmp_path)
    assert isinstance(tab, CachedPairwiseTable)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["pair_T120_G512_chal_left.npy", "pair_T120_G512_lead_right.npy"]

    pairs = _random_pairs(np.random.default_rng(5), 120, 500)
    fast = tab.log_e(*pairs)
    slow = exact.log_e(*pairs)
    assert fast.dtype == np.float64 and fast.shape == (500,)
    assert np.max(np.abs(fast - slow)) < 1e-3

    # Second call loads from disk (files untouched) and reproduces the same values.
    stamps = {p.name: os.stat(p).st_mtime_ns for p in tmp_path.iterdir()}
    again = get_pairwise_table(120, cache_dir=tmp_path)
    assert isinstance(again.lead_right, np.memmap) and isinstance(again.chal_left, np.memmap)
    assert {p.name: os.stat(p).st_mtime_ns for p in tmp_path.iterdir()} == stamps
    assert np.array_equal(again.log_e(*pairs), fast)


def test_pairwise_table_validates_every_query(tmp_path):
    tab = get_pairwise_table(30, cache_dir=tmp_path)
    ok = tab.log_e(np.array([30, 3]), np.array([30, 0]), np.array([0, 5]), np.array([0, 5]))
    assert np.all(np.isfinite(ok))
    with pytest.raises(ValueError, match="leader S"):
        tab.log_e([5], [6], [3], [1])
    with pytest.raises(ValueError, match="challenger n"):
        tab.log_e([5], [2], [31], [1])
    with pytest.raises(ValueError, match="leader n"):
        tab.log_e([-1], [0], [3], [1])


def test_chunked_fallback_for_large_horizons(tmp_path):
    tab = get_pairwise_table(1300, cache_dir=tmp_path)
    assert isinstance(tab, ChunkedPairwise)
    assert list(tmp_path.iterdir()) == []
    # 70 pairs span two chunks of 64; the last one sits at the max_n boundary.
    pairs = [a.copy() for a in _random_pairs(np.random.default_rng(6), 150, 70)]
    for arr, value in zip(pairs, (1300, 1200, 1300, 700), strict=True):
        arr[-1] = value
    assert np.allclose(tab.log_e(*pairs), PairwiseEvidence().log_e(*pairs), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="challenger n"):
        tab.log_e([5], [2], [1301], [1])


# ---- (d) column subsetting -----------------------------------------------------


class _NeverCalled:
    def log_e(self, *args):
        raise AssertionError("pairwise cover computed although f_log_e_pair was not requested")


def test_columns_subset_matches_full_computation(corpus, tables, pair_tables):
    T, snap = corpus[len(corpus) // 3]
    table = tables[T]
    state = materialize(snap, 4, table, 0)
    hist = VecSearchHistory.from_scalar([snap.meta["history"]] * 4, capacity=T)
    full = extract_features_vec(state, snap.t, T, table, pair_tables[T], history=hist)

    for name in ("clock", "clock_quality", "clock_quality_cs", "clock_quality_evidence", "all71"):
        cols = fg.FEATURE_SETS[name]
        pairwise = pair_tables[T] if "f_log_e_pair" in cols else _NeverCalled()
        sub = extract_features_vec(
            state, snap.t, T, table, pairwise, history=hist, columns=cols
        )
        assert tuple(sub) == tuple(cols)
        for c in cols:
            assert np.array_equal(sub[c], full[c]), c

    custom = ("est_quantile_0.9", "f_time_since_last_search", "est_beta_a", "f_t")
    sub = extract_features_vec(state, snap.t, T, table, _NeverCalled(), history=hist, columns=custom)
    assert tuple(sub) == custom
    for c in custom:
        assert np.array_equal(sub[c], full[c]), c

    with pytest.raises(ValueError, match="unknown feature columns"):
        extract_features_vec(state, snap.t, T, table, pair_tables[T], columns=("f_t", "oracle_mu_star"))


def test_feature_matrix_orders_columns(corpus, tables, pair_tables):
    T, snap = corpus[7]
    state = materialize(snap, 5, tables[T], 0)
    feats = extract_features_vec(state, snap.t, T, tables[T], pair_tables[T])
    cols = fg.FEATURE_SETS["clock_sf_quality_cs"]
    X = feature_matrix(feats, cols)
    assert X.shape == (5, len(cols)) and X.dtype == np.float64
    for j, c in enumerate(cols):
        assert np.array_equal(X[:, j], feats[c])
    with pytest.raises(KeyError, match="f_log_e_pair"):
        feature_matrix({"f_t": feats["f_t"]}, ("f_t", "f_log_e_pair"))
    with pytest.raises(ValueError):
        feature_matrix(feats, ())


# ---- (e) hygiene ----------------------------------------------------------------


def test_features_never_read_mu(corpus, tables, pair_tables):
    T, snap = corpus[-1]
    table = tables[T]
    state = materialize(snap, 3, table, 0)
    hist = VecSearchHistory.from_scalar([snap.meta["history"]] * 3, capacity=T)
    before = extract_features_vec(state, snap.t, T, table, pair_tables[T], history=hist)
    state.mu[:] = np.nan
    state.thresh[:] = 0
    after = extract_features_vec(state, snap.t, T, table, pair_tables[T], history=hist)
    for c in ALL:
        assert np.array_equal(before[c], after[c]), c


# ---- VecSearchHistory semantics --------------------------------------------------


def test_vec_history_matches_scalar_on_random_histories():
    rng = np.random.default_rng(11)
    for length in (0, 1, 2, 3, 9, 10, 11, 24, 25, 26, 49, 50, 51, 80):
        hists = []
        for _ in range(12):
            p = rng.choice([0.0, 0.05, 0.3, 1.0])
            dec = rng.random(length) < p
            trace = rng.random(length).astype(np.float32)
            hists.append(SearchHistory(decisions=dec, best_mean_trace=trace))
        vh = VecSearchHistory.from_scalar(hists, capacity=100)
        assert vh.length == length
        for w in (10, 25, 50):
            assert np.array_equal(vh.searched_in_last(w), [h.searched_in_last(w) for h in hists])
            assert np.array_equal(vh.search_fraction(w), [h.search_fraction(w) for h in hists])
            got = vh.improvement_over(w)
            want = np.array([h.improvement_over(w) for h in hists])
            assert np.allclose(got, want, rtol=0, atol=1e-7), (length, w)
        assert np.array_equal(
            vh.time_since_last_search(), [h.time_since_last_search() for h in hists]
        )
        for m, h in enumerate(hists):
            back = vh.scalar(m)
            assert np.array_equal(back.decisions, h.decisions)
            assert np.array_equal(back.best_mean_trace, h.best_mean_trace)


def test_vec_history_record_and_guards():
    vh = VecSearchHistory(3, capacity=2)
    assert vh.length == 0
    assert np.array_equal(vh.time_since_last_search(), [0, 0, 0])
    assert np.array_equal(vh.search_fraction(10), [0.0, 0.0, 0.0])
    vh.record(np.array([True, False, False]), np.array([0.5, 0.6, 0.7]))
    vh.record(np.array([False, False, True]), np.array([0.55, 0.6, 0.9]))
    assert vh.length == 2
    assert np.array_equal(vh.time_since_last_search(), [1, 2, 0])
    assert np.array_equal(vh.searched_in_last(1), [0, 0, 1])
    assert np.allclose(vh.improvement_over(10), [0.05, 0.0, 0.2])
    with pytest.raises(ValueError, match="full"):
        vh.record(np.zeros(3, dtype=bool), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        VecSearchHistory(3, capacity=4).record(np.zeros(2, dtype=bool), np.zeros(3))
    with pytest.raises(ValueError, match="one length"):
        VecSearchHistory.from_scalar(
            [SearchHistory(np.zeros(2, dtype=bool), np.zeros(2)), SearchHistory()], capacity=4
        )
    with pytest.raises(ValueError, match="capacity"):
        VecSearchHistory.from_scalar([SearchHistory(np.zeros(5, dtype=bool), np.zeros(5))], 4)
