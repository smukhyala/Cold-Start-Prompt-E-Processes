"""Tests for feature extraction and the oracle-leakage boundary."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.evidence import PairwiseEvidence
from cold_start.growing.features import SearchHistory, extract_features
from cold_start.growing.reservoirs import build_reservoir
from cold_start.growing.schema import (
    OracleLeakageError,
    design_matrix_columns,
    validate_columns,
)
from cold_start.growing.tables import CSTable

HORIZON = 200


@pytest.fixture(scope="module")
def table():
    return CSTable.load_or_build(HORIZON, alpha=0.05)


@pytest.fixture(scope="module")
def pairwise():
    return PairwiseEvidence()


@pytest.fixture(scope="module")
def reservoir():
    return build_reservoir({"type": "tail", "params": {"beta": 2.0, "mu_star": 0.9, "c": 1.0}})


def _row(table, pairwise, reservoir=None, history=None, **kw):
    args = dict(
        n=np.array([40, 25, 25, 3, 1]),
        successes=np.array([26, 9, 8, 2, 1]),
        mu_true=np.array([0.65, 0.30, 0.28, 0.55, 0.80]),
        t=94,
        horizon=HORIZON,
    )
    args.update(kw)
    return extract_features(
        table=table, pairwise=pairwise, reservoir=reservoir, history=history, **args
    )


def test_every_column_carries_a_known_prefix(table, pairwise, reservoir):
    validate_columns(list(_row(table, pairwise, reservoir).keys()))


def test_all_values_are_finite(table, pairwise, reservoir):
    """A NaN feature silently poisons a fit, so this is a hard invariant."""
    row = _row(table, pairwise, reservoir)
    bad = [c for c, v in row.items() if not np.isfinite(v)]
    assert not bad, f"non-finite features: {bad}"


def test_oracle_columns_cannot_reach_a_deployable_design_matrix(table, pairwise, reservoir):
    cols = list(_row(table, pairwise, reservoir).keys())
    with pytest.raises(OracleLeakageError):
        design_matrix_columns(cols)


def test_deployable_selection_keeps_only_observable_columns(table, pairwise, reservoir):
    cols = list(_row(table, pairwise, reservoir).keys())
    dep = design_matrix_columns([c for c in cols if not c.startswith("oracle_")])
    assert dep, "no deployable columns survived"
    assert not any(c.startswith("oracle_") for c in dep)


def test_ablation_f_can_opt_into_oracle_columns(table, pairwise, reservoir):
    cols = list(_row(table, pairwise, reservoir).keys())
    with_oracle = design_matrix_columns(cols, allow_oracle=True)
    assert any(c.startswith("oracle_") for c in with_oracle)


def test_omitting_the_reservoir_omits_every_oracle_column(table, pairwise):
    """A deployment-time caller has no reservoir, and must still get a full row."""
    row = _row(table, pairwise, reservoir=None)
    assert not any(c.startswith("oracle_") for c in row)
    assert design_matrix_columns(list(row.keys()))


def test_global_clock_features_are_arithmetically_right(table, pairwise):
    row = _row(table, pairwise, t=100, horizon=HORIZON)
    assert row["f_remaining_budget"] == 100.0
    assert row["f_remaining_frac"] == pytest.approx(0.5)
    assert row["f_K"] == 5.0
    assert row["f_K_over_t"] == pytest.approx(0.05)
    assert row["f_K_over_sqrt_t"] == pytest.approx(0.5)


def test_leader_and_challenger_are_distinct_arms(table, pairwise):
    row = _row(table, pairwise)
    assert row["f_ucb_chal_minus_lcb_lead"] == pytest.approx(-row["f_lcb_lead_minus_ucb_chal"])


def test_separation_flag_matches_the_bounds(table, pairwise):
    """A clearly dominant arm should read as separated; a tied one should not."""
    sep = _row(
        table,
        pairwise,
        n=np.array([120, 120]),
        successes=np.array([110, 20]),
        mu_true=np.array([0.9, 0.2]),
    )
    assert sep["f_is_separated"] == 1.0
    assert sep["f_log_e_pair"] > 3.0, f"log_e_pair={sep['f_log_e_pair']:.2f}"

    tied = _row(
        table,
        pairwise,
        n=np.array([20, 20]),
        successes=np.array([11, 10]),
        mu_true=np.array([0.55, 0.5]),
    )
    assert tied["f_is_separated"] == 0.0
    assert tied["f_log_e_pair"] < 3.0


def test_elimination_counts_track_the_plausible_set(table, pairwise):
    row = _row(
        table,
        pairwise,
        n=np.array([120, 120, 120]),
        successes=np.array([110, 20, 18]),
        mu_true=np.array([0.9, 0.2, 0.18]),
    )
    assert row["f_n_eliminated"] == 2.0
    assert row["f_n_plausible"] == 1.0
    assert row["f_frac_eliminated"] == pytest.approx(2 / 3)


def test_singleton_arms_are_counted(table, pairwise):
    assert _row(table, pairwise)["f_n_singletons"] == 1.0


def test_search_history_windows(table, pairwise):
    decisions = np.array([True] * 3 + [False] * 7)
    hist = SearchHistory(decisions=decisions, best_mean_trace=np.linspace(0.4, 0.6, 10))
    row = _row(table, pairwise, history=hist)
    assert row["f_search_frac_last_10"] == pytest.approx(0.3)
    assert row["f_time_since_last_search"] == 7.0
    assert row["f_best_mean_gain_last_10"] > 0.0


def test_history_defaults_are_total_when_absent(table, pairwise):
    row = _row(table, pairwise, history=None)
    assert row["f_search_frac_last_10"] == 0.0
    assert row["f_time_since_last_search"] == 0.0


def test_estimated_reservoir_never_reads_true_means(table, pairwise, reservoir):
    """The est_ block must be computable by a real algorithm.

    Inverting the true means while holding (n, S) fixed must not move a single
    est_ column -- otherwise it is an oracle wearing a deployable prefix.
    """
    base = _row(table, pairwise, reservoir)
    flipped = _row(table, pairwise, reservoir, mu_true=np.array([0.80, 0.55, 0.28, 0.30, 0.65]))
    for col in [c for c in base if c.startswith("est_") or c.startswith("f_")]:
        assert base[col] == pytest.approx(flipped[col]), f"{col} depends on true mu"


def test_oracle_tail_probabilities_decrease_with_delta(table, pairwise, reservoir):
    row = _row(table, pairwise, reservoir)
    seq = [
        row["oracle_p_new_beats_best_true"],
        row["oracle_p_new_beats_best_true_plus_0.01"],
        row["oracle_p_new_beats_best_true_plus_0.05"],
        row["oracle_p_new_beats_best_true_plus_0.1"],
    ]
    assert all(a >= b for a, b in zip(seq, seq[1:], strict=False)), seq


def test_single_arm_state_is_handled(table, pairwise, reservoir):
    row = _row(
        table,
        pairwise,
        reservoir,
        n=np.array([10]),
        successes=np.array([6]),
        mu_true=np.array([0.6]),
    )
    assert row["f_K"] == 1.0
    assert np.isfinite(row["f_log_e_pair"])


def test_empty_state_is_rejected(table, pairwise):
    with pytest.raises(ValueError):
        extract_features(
            n=np.array([]),
            successes=np.array([]),
            mu_true=np.array([]),
            t=0,
            horizon=HORIZON,
            table=table,
            pairwise=pairwise,
        )


def test_no_deployable_column_is_constant_across_a_diverse_corpus(table, pairwise, reservoir):
    """The structural guard for dead features.

    `est_frac_arms_above_incumbent` shipped as identically 0.0 -- it counted arms
    strictly greater than their own maximum. Per-row finiteness checks cannot catch
    that, and a constant column survives a fit silently: StandardScaler maps it to
    zero and ridge assigns it a harmless coefficient, so nothing ever complains. The
    only thing that catches it is varying the state and asserting the column moves.
    """
    rng = np.random.default_rng(17)
    rows = []
    for _ in range(120):
        k = int(rng.integers(1, 9))
        n = rng.integers(1, 60, size=k)
        successes = np.array([int(rng.integers(0, ni + 1)) for ni in n])
        mu = rng.random(k)
        t = int(max(n.sum(), 1))
        # Vary the decision history too: without it the search-history block is
        # constant by construction and the guard would only be testing half the row.
        n_hist = int(rng.integers(1, 60))
        hist = SearchHistory(
            decisions=rng.random(n_hist) < rng.random(),
            best_mean_trace=np.cumsum(rng.random(n_hist) * 0.02),
        )
        rows.append(
            extract_features(
                n=n, successes=successes, mu_true=mu, t=t, horizon=max(t + 5, HORIZON),
                table=table, pairwise=pairwise, reservoir=reservoir, history=hist,
            )
        )

    deployable = [c for c in rows[0] if c.startswith(("f_", "est_"))]
    constant = []
    for col in deployable:
        vals = np.array([r[col] for r in rows], dtype=float)
        if np.nanstd(vals) == 0.0:
            constant.append((col, float(vals[0])))

    # `f_T` is the only column the generator holds fixed.
    allowed = {"f_T"}
    offenders = [(c, v) for c, v in constant if c not in allowed]
    assert not offenders, (
        f"{len(offenders)} deployable column(s) never vary across 120 diverse states "
        f"-- they carry no information: {offenders}"
    )


def test_top_gap_measures_leader_isolation(table, pairwise):
    """A dominant leader gives a wide gap; a contested top gives a narrow one."""
    isolated = _row(
        table, pairwise,
        n=np.array([40, 40, 40]), successes=np.array([36, 10, 9]),
        mu_true=np.array([0.9, 0.25, 0.22]),
    )
    contested = _row(
        table, pairwise,
        n=np.array([40, 40, 40]), successes=np.array([28, 27, 27]),
        mu_true=np.array([0.70, 0.68, 0.67]),
    )
    assert isolated["est_top_gap"] > contested["est_top_gap"], (
        f"isolated {isolated['est_top_gap']:.4f} vs contested {contested['est_top_gap']:.4f}"
    )
    assert 0.0 <= contested["est_top_gap_normalized"] <= 1.0


def test_challenger_is_never_the_arm_it_is_compared_against(table, pairwise):
    """Separation features must describe a genuine pair.

    The challenger was selected by masking the EMPIRICAL leader while every separation
    feature compares against the CS leader. When the two differ -- a lightly-pulled arm
    with a high mean versus a well-measured one -- the CS leader was returned as its own
    challenger, forcing `lcb_lead_minus_ucb_chal <= 0` and making `log_e_pair` an arm
    tested against itself.
    """
    n = np.array([2, 60])
    successes = np.array([2, 39])
    post = (successes + 1) / (n + 2)
    lo, _ = table.bounds(n, successes)
    assert int(np.argmax(post)) != int(np.argmax(lo)), "fixture must have differing leaders"

    row = _row(table, pairwise, n=n, successes=successes, mu_true=np.array([0.9, 0.65]))
    # A self-comparison is exactly zero separation in both directions.
    assert row["f_lcb_lead_minus_ucb_chal"] != 0.0
    assert row["f_csleader_lcb"] != row["f_challenger_lcb"] or row["f_K"] == 1.0


def test_separation_features_use_the_same_pair_the_allocator_does(table, pairwise):
    """The feature row must describe the pair the simulator actually acts on."""
    from cold_start.growing.allocation import leader_and_challenger
    from cold_start.growing.state import GrowingState

    n = np.array([2, 60, 30])
    successes = np.array([2, 39, 12])
    st = GrowingState(1, 4, 200, base_seed=1)
    for j in range(3):
        st.add_arms(np.array([0.5], dtype=np.float32), np.array([j], dtype=np.int32))
        st.view(st.n)[:, j] = n[j]
        st.view(st.S)[:, j] = successes[j]
    lo, hi = table.bounds(n, successes)
    st.view(st.lcb)[0, :3] = lo
    st.view(st.ucb)[0, :3] = hi
    leader, challenger = leader_and_challenger(st)

    row = _row(table, pairwise, n=n, successes=successes, mu_true=np.array([0.9, 0.65, 0.4]))
    assert row["f_csleader_lcb"] == pytest.approx(float(lo[int(leader[0])]), abs=1e-6)
    assert row["f_challenger_ucb"] == pytest.approx(float(hi[int(challenger[0])]), abs=1e-6)
