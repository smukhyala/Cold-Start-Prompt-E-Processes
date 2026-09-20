"""Batched feature extraction that reproduces `features.extract_features` exactly.

A learned SEARCH/REFINE policy was fitted to rows produced by the scalar extractor,
one snapshot at a time. At deployment it must act on the batched `(M, Kmax)` state,
so every one of the 71 deployable columns is restated here as an `(M,)` array. The
rule is *reproduce, do not re-derive*: each block below mirrors the corresponding
lines of `extract_features`, including its fallbacks and edge cases, because a
previous deployment attempt that "re-implemented" one feature drifted from the
corpus (`r` was 0.333 there and 0.568 here) and silently invalidated the policy.
`tests/test_deploy_feature_parity.py` is the gate that keeps the two in step.

Hygiene: this module reads `state.n, state.S, state.lcb, state.ucb`, the active mask,
`state.Kt`, the clock, and the decision history -- never `state.mu`, `state.thresh`,
or a reservoir. Those are hidden truth; the harness computes its diagnostics from
them only after the policy has acted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import betainc, betaincc

from cold_start.growing.deploy import feature_groups as fg
from cold_start.growing.deploy.history_vec import VecSearchHistory
from cold_start.growing.features import DELTAS, WINDOWS
from cold_start.growing.state import GrowingState

# The three blocks that dominate the cost and are skipped when not requested: the
# pairwise e-process cover, the per-row sort of posterior means, and the SciPy
# incomplete-beta calls. Everything else is a handful of (M, K) numpy ops.
_LOGE_COLUMNS: frozenset[str] = frozenset(fg.EVIDENCE_LOGE)
_SORT_COLUMNS: frozenset[str] = frozenset(
    ("est_hill_tail_index", "est_quantile_0.5", "est_quantile_0.9", "est_quantile_0.99")
)
_BETA_COLUMNS: frozenset[str] = frozenset(
    (
        "est_beta_a",
        "est_beta_b",
        "est_beta_mean",
        "est_p_new_beats_incumbent",
        "est_p_new_beats_incumbent_plus_0.01",
        "est_p_new_beats_incumbent_plus_0.05",
        "est_p_new_beats_incumbent_plus_0.1",
    )
)
_QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.99)
# Below this tail mass `1 - betainc` starts losing relative precision to cancellation,
# so the exact complement is used there (see `_beta_sf`).
_SF_EXACT_BELOW = 1e-6


def extract_features_vec(
    state: GrowingState,
    t: int,
    horizon: int,
    table,
    pairwise,
    history: VecSearchHistory | None = None,
    columns: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """All (or the requested) deployable feature columns for every replicate.

    `table` is accepted for signature parity with the scalar extractor and the model
    policy that holds one; the bounds are read from `state.lcb / state.ucb`, which are
    exactly that table's values (refreshed by every `pull`). `pairwise` is anything
    with `.log_e(n_lead, S_lead, n_chal, S_chal)` -- the cached grid table or the exact
    `PairwiseEvidence`. Returns `(M,)` float64 arrays keyed by column name; with
    `columns` given, exactly those keys, in that order.
    """
    wanted = tuple(fg.ALL_DEPLOYABLE) if columns is None else tuple(columns)
    unknown = [c for c in wanted if c not in fg.ALL_DEPLOYABLE]
    if unknown:
        raise ValueError(f"unknown feature columns: {unknown}")
    need = set(wanted)

    M = state.M
    K = state.Kt.astype(np.int64)
    if bool((K < 1).any()):
        raise ValueError("cannot build features for a replicate with no arms")
    Kact = int(K.max())
    # Active slots are the contiguous columns 0..Kt-1, so everything past the widest
    # replicate is empty and is dropped before any (M, K) op.
    act = state.active_mask()[:, :Kact]
    if bool((act.sum(axis=1) != K).any()):
        raise ValueError("state.Kt disagrees with the active mask; the state is corrupt")
    rows = np.arange(M)
    Kf = K.astype(np.float64)

    n_int = state.view(state.n)[:, :Kact].astype(np.int64)
    S_int = state.view(state.S)[:, :Kact].astype(np.int64)
    n = n_int.astype(np.float64)
    S = S_int.astype(np.float64)
    lo = state.view(state.lcb)[:, :Kact].astype(np.float64)
    hi = state.view(state.ucb)[:, :Kact].astype(np.float64)
    width = hi - lo
    post = (S + 1.0) / (n + 2.0)

    out: dict[str, np.ndarray] = {}

    def const(value: float) -> np.ndarray:
        return np.full(M, float(value), dtype=np.float64)

    # ---- global clock and arm count ----
    remaining = horizon - t
    out["f_t"] = const(t)
    out["f_T"] = const(horizon)
    out["f_remaining_budget"] = const(remaining)
    out["f_remaining_frac"] = const(remaining / horizon if horizon else 0.0)
    out["f_t_over_T"] = const(t / horizon if horizon else 0.0)
    out["f_K"] = Kf.copy()
    out["f_K_over_t"] = Kf / t if t else Kf.copy()
    out["f_K_over_T"] = Kf / horizon if horizon else const(0.0)
    out["f_log_t"] = const(np.log(max(t, 1)))
    out["f_log_K"] = np.log(Kf)
    out["f_K_over_sqrt_t"] = Kf / np.sqrt(max(t, 1))

    # ---- leader, CS leader, challenger ----
    # Plain argmax on the masked rows: first index on ties, as numpy gives on the
    # compact scalar array. The allocation rules add tiebreak jitter; the corpus
    # features did not, so neither does this.
    post_m = np.where(act, post, -np.inf)
    lo_m = np.where(act, lo, -np.inf)
    emp_leader = np.argmax(post_m, axis=1)
    cs_leader = np.argmax(lo_m, axis=1)
    hi_m = np.where(act, hi, -np.inf)
    hi_m[rows, cs_leader] = -np.inf
    challenger = np.where(K > 1, np.argmax(hi_m, axis=1), cs_leader)

    t_floor = float(max(t, 1))
    for tag, idx in (("leader", emp_leader), ("csleader", cs_leader), ("challenger", challenger)):
        out[f"f_{tag}_n"] = n[rows, idx]
        out[f"f_{tag}_mean"] = post[rows, idx]
        out[f"f_{tag}_lcb"] = lo[rows, idx]
        out[f"f_{tag}_ucb"] = hi[rows, idx]
        out[f"f_{tag}_width"] = width[rows, idx]
        out[f"f_{tag}_n_frac"] = n[rows, idx] / t_floor

    # ---- separation ----
    lo_cs = lo[rows, cs_leader]
    hi_ch = hi[rows, challenger]
    out["f_empirical_gap"] = post[rows, cs_leader] - post[rows, challenger]
    out["f_lcb_lead_minus_ucb_chal"] = lo_cs - hi_ch
    out["f_ucb_chal_minus_lcb_lead"] = hi_ch - lo_cs
    out["f_is_separated"] = (lo_cs > hi_ch).astype(np.float64)
    if need & _LOGE_COLUMNS:
        pair = pairwise.log_e(
            n_int[rows, cs_leader],
            S_int[rows, cs_leader],
            n_int[rows, challenger],
            S_int[rows, challenger],
        )
        out["f_log_e_pair"] = np.where(K > 1, np.asarray(pair, dtype=np.float64), 0.0)

    # ---- plausible-winner frontier ----
    best_lcb = lo_m.max(axis=1)
    plausible = act & (hi >= best_lcb[:, None])
    n_plaus = plausible.sum(axis=1)
    n_plaus_f = n_plaus.astype(np.float64)
    out["f_n_plausible"] = n_plaus_f
    out["f_frac_plausible"] = n_plaus_f / Kf
    out["f_n_eliminated"] = Kf - n_plaus_f
    out["f_frac_eliminated"] = (Kf - n_plaus_f) / Kf
    has_plaus = n_plaus > 0
    out["f_max_width_plausible"] = np.where(
        has_plaus, np.where(plausible, width, -np.inf).max(axis=1), 0.0
    )
    out["f_mean_width_plausible"] = np.where(
        has_plaus,
        np.where(plausible, width, 0.0).sum(axis=1) / np.maximum(n_plaus_f, 1.0),
        0.0,
    )
    out["f_mean_width_all"] = np.where(act, width, 0.0).sum(axis=1) / Kf

    # ---- discovered-quality summary ----
    incumbent = post_m.max(axis=1)
    # Second-largest value: the max once one copy of the leader is removed, which is
    # `np.sort(post)[-2]` exactly, ties included, without a sort.
    post_rest = post_m.copy()
    post_rest[rows, emp_leader] = -np.inf
    second_raw = post_rest.max(axis=1)
    multi = K > 1
    mean_post = np.where(act, post, 0.0).sum(axis=1) / Kf
    dev2 = np.where(act, (post - mean_post[:, None]) ** 2, 0.0).sum(axis=1)
    var_post = dev2 / np.maximum(Kf - 1.0, 1.0)  # ddof=1; meaningful only for K > 1
    out["f_best_mean"] = incumbent
    out["f_second_best_mean"] = np.where(multi, second_raw, 0.0)
    out["f_mean_of_means"] = mean_post
    out["f_sd_of_means"] = np.where(multi, np.sqrt(var_post), 0.0)
    out["f_max_n"] = np.where(act, n, 0.0).max(axis=1)
    out["f_mean_n"] = np.where(act, n, 0.0).sum(axis=1) / Kf
    out["f_n_singletons"] = (act & (n <= 1)).sum(axis=1).astype(np.float64)

    # ---- search history ----
    hist = history if history is not None else VecSearchHistory(M, 0)
    if hist.M != M:
        raise ValueError(f"history has {hist.M} replicates but the state has {M}")
    for w in WINDOWS:
        out[f"f_new_arms_last_{w}"] = hist.searched_in_last(w).astype(np.float64)
        out[f"f_best_mean_gain_last_{w}"] = hist.improvement_over(w).astype(np.float64)
    for w in (10, 25):
        out[f"f_search_frac_last_{w}"] = hist.search_fraction(w).astype(np.float64)
    out["f_time_since_last_search"] = hist.time_since_last_search().astype(np.float64)

    # ---- estimated reservoir (deployable) ----
    second = np.where(multi, second_raw, incumbent)
    spread = incumbent - np.where(act, post, np.inf).min(axis=1)
    degenerate = spread <= 1e-9
    out["est_top_gap"] = incumbent - second
    out["est_top_gap_normalized"] = np.where(
        degenerate, 1.0, (incumbent - second) / np.where(degenerate, 1.0, spread)
    )
    out["est_frac_arms_within_5pct_of_best"] = (
        (act & (post >= (incumbent - 0.05)[:, None])).sum(axis=1) / Kf
    )

    if need & _BETA_COLUMNS:
        a_hat, b_hat = _fit_beta_moments_vec(mean_post, var_post, K)
        out["est_beta_a"] = a_hat
        out["est_beta_b"] = b_hat
        out["est_beta_mean"] = a_hat / (a_hat + b_hat)
        out["est_p_new_beats_incumbent"] = _beta_sf(incumbent, a_hat, b_hat)
        for d in DELTAS:
            out[f"est_p_new_beats_incumbent_plus_{d}"] = _beta_sf(
                np.minimum(incumbent + d, 1.0), a_hat, b_hat
            )

    if need & _SORT_COLUMNS:
        # Ascending sort with inactive slots pushed to the end; row m's active values
        # occupy positions 0..K[m]-1.
        asc = np.sort(np.where(act, post, np.inf), axis=1)
        out["est_hill_tail_index"] = _hill_tail_index_vec(asc, K)
        for q in _QUANTILES:
            out[f"est_quantile_{q}"] = _quantile_linear_vec(asc, K, q)

    return {c: np.asarray(out[c], dtype=np.float64) for c in wanted}


def _beta_sf(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`scipy.stats.beta.sf(x, a, b)` for arrays, at a usable cost.

    The scalar corpus called `beta.sf`, which is `scipy.special.betaincc`. Per element
    that is ~85 us for non-integer shapes in scipy 1.17 -- fine once per snapshot,
    but four calls per step at M=2048 cost more than every other column combined.
    `1 - betainc` is ~100x faster and equal to a few 1e-16 absolute; it only loses
    *relative* precision when the tail is tiny, so those rows are recomputed with the
    exact complement and stay bit-identical to the corpus.
    """
    p = 1.0 - betainc(a, b, x)
    small = p < _SF_EXACT_BELOW
    if bool(small.any()):
        p[small] = betaincc(a[small], b[small], x[small])
    return p


def _fit_beta_moments_vec(
    mean_post: np.ndarray, var_post: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`features._fit_beta_moments`, row-wise, with the identical Beta(1,1) fallbacks.

    `var_post` must be the ddof=1 variance of the active posterior means (its value
    is ignored where `K < 2`).
    """
    m = np.clip(mean_post, 1e-6, 1 - 1e-6)
    fallback = (K < 2) | (var_post <= 1e-12) | (var_post >= m * (1 - m))
    safe_v = np.where(fallback, 1.0, var_post)
    nu = m * (1 - m) / safe_v - 1.0
    # Capped by the number of arms behind the fit, as in the scalar reference.
    nu = np.clip(nu, 2.0, np.maximum(2.0, K.astype(np.float64)))
    a_hat = np.where(fallback, 1.0, m * nu)
    b_hat = np.where(fallback, 1.0, (1 - m) * nu)
    return a_hat, b_hat


def _hill_tail_index_vec(asc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """`features._hill_tail_index` on each row's descending posterior means.

    Every posterior mean is strictly positive, so the scalar's `sorted_desc > 0`
    filter keeps all K values and `x.size == K`.
    """
    M, Kact = asc.shape
    valid = K >= 3
    out = np.zeros(M, dtype=np.float64)
    if not bool(valid.any()):
        return out
    Kf = K.astype(np.float64)
    k = np.maximum(2, np.minimum(K - 1, np.floor(np.sqrt(Kf)).astype(np.int64) + 1))
    kmax = int(k.max())
    j = np.arange(kmax + 1)
    # Descending order statistic j of row m sits at ascending position K[m]-1-j.
    idx = np.clip(K[:, None] - 1 - j[None, :], 0, Kact - 1)
    top = np.take_along_axis(asc, idx, axis=1)  # (M, kmax + 1)
    top_k = top[np.arange(M), k]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(top[:, :kmax] / top_k[:, None])
    inside = j[None, :kmax] < k[:, None]
    val = np.where(inside, logs, 0.0).sum(axis=1) / k.astype(np.float64)
    return np.where(valid & np.isfinite(val), val, 0.0)


def _quantile_linear_vec(asc: np.ndarray, K: np.ndarray, q: float) -> np.ndarray:
    """`np.quantile(post, q)` (method="linear") on each row's K active values.

    Restates numpy's `_quantile`: virtual index `(K-1) q`, floor/next neighbours
    clipped to the last valid value, and its `_lerp`, which switches to
    `b - (b-a)(1-gamma)` at `gamma >= 0.5` for numerical symmetry. Reproducing the
    branch keeps the two paths bit-for-bit equal rather than merely close.
    """
    M = asc.shape[0]
    rows = np.arange(M)
    Kf = K.astype(np.float64)
    vi = (Kf - 1.0) * q
    prev = np.floor(vi)
    # At or past the last index (only K == 1 for q < 1) both neighbours are the last
    # value, so gamma is irrelevant there.
    above = vi >= Kf - 1.0
    prev_i = np.where(above, K - 1, prev.astype(np.int64))
    next_i = np.where(above, K - 1, prev_i + 1)
    gamma = vi - prev
    a = asc[rows, prev_i]
    b = asc[rows, next_i]
    diff = b - a
    res = a + diff * gamma
    return np.where(gamma >= 0.5, b - diff * (1.0 - gamma), res)


def feature_matrix(feats: dict[str, np.ndarray], columns: Sequence[str]) -> np.ndarray:
    """Stack named `(M,)` columns into the `(M, len(columns))` float64 design matrix."""
    cols = list(columns)
    if not cols:
        raise ValueError("feature_matrix needs at least one column")
    missing = [c for c in cols if c not in feats]
    if missing:
        raise KeyError(f"feature columns missing from the extracted dict: {missing}")
    return np.stack([np.asarray(feats[c], dtype=np.float64) for c in cols], axis=1)
