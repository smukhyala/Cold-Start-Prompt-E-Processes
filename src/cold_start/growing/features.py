"""Turn a saved bandit state into the feature row a decision function will see.

The contract that matters here is the prefix contract from `schema.py`. A column
named `f_*` or `est_*` is something a real algorithm could compute at decision time;
a column named `oracle_*` is truth we only have because this is a simulation. Keeping
them apart by construction -- rather than by remembering to -- is what stops the
fitted policy from quietly depending on knowledge it will not have when deployed.

The `est_*` block is the interesting half. It asks the same questions as the
`oracle_*` block ("how likely is a fresh arm to beat what I hold?") but answers them
using only the arms discovered so far. How much is lost in that translation is one of
the study's real findings, so both are recorded side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cold_start.growing.schema import (
    PREFIX_ESTIMATE,
    PREFIX_FEATURE,
    PREFIX_ORACLE,
    validate_columns,
)

DELTAS: tuple[float, ...] = (0.01, 0.05, 0.1)
EPSILONS: tuple[float, ...] = (0.01, 0.05, 0.1)
WINDOWS: tuple[int, ...] = (10, 25, 50)


@dataclass
class SearchHistory:
    """Per-step trajectory record, needed for the "recent behaviour" features.

    A policy that has just searched five times in a row is in a different situation
    from one that has not searched in fifty rounds, even with identical arm
    statistics -- so the decision history is part of the state, not metadata.
    """

    decisions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    best_mean_trace: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def searched_in_last(self, w: int) -> int:
        return int(self.decisions[-w:].sum()) if self.decisions.size else 0

    def search_fraction(self, w: int) -> float:
        tail = self.decisions[-w:]
        return float(tail.mean()) if tail.size else 0.0

    def time_since_last_search(self) -> int:
        """Rounds since the last SEARCH; the full length if there has never been one."""
        if not self.decisions.size:
            return 0
        idx = np.flatnonzero(self.decisions)
        return int(self.decisions.size - 1 - idx[-1]) if idx.size else int(self.decisions.size)

    def improvement_over(self, w: int) -> float:
        if self.best_mean_trace.size < 2:
            return 0.0
        past = self.best_mean_trace[max(0, self.best_mean_trace.size - w - 1)]
        return float(self.best_mean_trace[-1] - past)


def _hill_tail_index(sorted_desc: np.ndarray, k: int | None = None) -> float:
    """Hill estimator on the top order statistics of the discovered arm means.

    A crude read on how heavy the reservoir's upper tail looks *from the inside* --
    the deployable counterpart to knowing the true `beta`. Returns 0.0 when there is
    not enough spread to say anything, rather than a NaN that would poison a fit.
    """
    x = sorted_desc[sorted_desc > 0]
    if x.size < 3:
        return 0.0
    k = k or max(2, min(x.size - 1, int(np.sqrt(x.size)) + 1))
    top = x[: k + 1]
    if top[-1] <= 0:
        return 0.0
    logs = np.log(top[:-1] / top[-1])
    val = float(logs.mean())
    return val if np.isfinite(val) else 0.0


def _fit_beta_moments(vals: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Beta fit to the discovered arms' posterior means."""
    if vals.size < 2:
        return 1.0, 1.0
    m = float(np.clip(vals.mean(), 1e-6, 1 - 1e-6))
    v = float(vals.var(ddof=1))
    if v <= 1e-12 or v >= m * (1 - m):
        return 1.0, 1.0
    nu = m * (1 - m) / v - 1.0
    # Capped by the number of arms that informed the fit: a prior estimated from a
    # handful of survivorship-biased arms must not carry more weight than the
    # evidence behind it. See the matching note in `recommend.fit_empirical_bayes_prior`.
    nu = float(np.clip(nu, 2.0, max(2.0, float(vals.size))))
    return m * nu, (1 - m) * nu


def extract_features(
    n: np.ndarray,
    successes: np.ndarray,
    mu_true: np.ndarray,
    t: int,
    horizon: int,
    table,
    pairwise,
    reservoir=None,
    history: SearchHistory | None = None,
) -> dict[str, float]:
    """Build one feature row for a single state.

    `reservoir` is optional: when supplied its true tail probabilities are emitted as
    `oracle_*` columns for analysis. Omitting it simply omits those columns, which is
    what a deployment-time caller would do.
    """
    n = np.asarray(n, dtype=np.int64)
    successes = np.asarray(successes, dtype=np.int64)
    mu_true = np.asarray(mu_true, dtype=np.float64)
    k = int(n.size)
    if k == 0:
        raise ValueError("cannot build features for a state with no arms")

    lo, hi = table.bounds(n, successes)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    width = hi - lo
    post = (successes + 1.0) / (n + 2.0)

    out: dict[str, float] = {}
    p = PREFIX_FEATURE

    # ---- global clock and arm count ----
    remaining = horizon - t
    out[f"{p}t"] = float(t)
    out[f"{p}T"] = float(horizon)
    out[f"{p}remaining_budget"] = float(remaining)
    out[f"{p}remaining_frac"] = float(remaining / horizon) if horizon else 0.0
    out[f"{p}t_over_T"] = float(t / horizon) if horizon else 0.0
    out[f"{p}K"] = float(k)
    out[f"{p}K_over_t"] = float(k / t) if t else float(k)
    out[f"{p}K_over_T"] = float(k / horizon) if horizon else 0.0
    out[f"{p}log_t"] = float(np.log(max(t, 1)))
    out[f"{p}log_K"] = float(np.log(k))
    out[f"{p}K_over_sqrt_t"] = float(k / np.sqrt(max(t, 1)))

    # ---- leader, CS leader, challenger ----
    emp_leader = int(np.argmax(post))
    cs_leader = int(np.argmax(lo))
    # Mask the CS LEADER, because that is the arm every separation feature below
    # compares against -- and it is also the leader `allocation.leader_and_challenger`
    # uses, so the feature row describes the same pair the simulator acts on. Masking
    # the empirical leader instead lets the CS leader be returned as its own
    # challenger whenever the two differ, which makes `lcb_lead - ucb_chal` a
    # self-comparison (necessarily <= 0) and `log_e_pair` an arm tested against itself.
    masked_hi = hi.copy()
    masked_hi[cs_leader] = -np.inf
    challenger = int(np.argmax(masked_hi)) if k > 1 else cs_leader

    for tag, idx in (("leader", emp_leader), ("csleader", cs_leader), ("challenger", challenger)):
        out[f"{p}{tag}_n"] = float(n[idx])
        out[f"{p}{tag}_mean"] = float(post[idx])
        out[f"{p}{tag}_lcb"] = float(lo[idx])
        out[f"{p}{tag}_ucb"] = float(hi[idx])
        out[f"{p}{tag}_width"] = float(width[idx])
        out[f"{p}{tag}_n_frac"] = float(n[idx] / max(t, 1))

    # ---- separation ----
    out[f"{p}empirical_gap"] = float(post[cs_leader] - post[challenger])
    out[f"{p}lcb_lead_minus_ucb_chal"] = float(lo[cs_leader] - hi[challenger])
    out[f"{p}ucb_chal_minus_lcb_lead"] = float(hi[challenger] - lo[cs_leader])
    out[f"{p}is_separated"] = float(lo[cs_leader] > hi[challenger])
    if k > 1:
        pair = pairwise.log_e(
            np.array([n[cs_leader]]),
            np.array([successes[cs_leader]]),
            np.array([n[challenger]]),
            np.array([successes[challenger]]),
        )
        out[f"{p}log_e_pair"] = float(pair[0])
    else:
        out[f"{p}log_e_pair"] = 0.0

    # ---- plausible-winner frontier ----
    best_lcb = float(lo.max())
    plausible = hi >= best_lcb
    n_plaus = int(plausible.sum())
    out[f"{p}n_plausible"] = float(n_plaus)
    out[f"{p}frac_plausible"] = float(n_plaus / k)
    out[f"{p}n_eliminated"] = float(k - n_plaus)
    out[f"{p}frac_eliminated"] = float((k - n_plaus) / k)
    out[f"{p}max_width_plausible"] = float(width[plausible].max()) if n_plaus else 0.0
    out[f"{p}mean_width_plausible"] = float(width[plausible].mean()) if n_plaus else 0.0
    out[f"{p}mean_width_all"] = float(width.mean())

    # ---- discovered-quality summary ----
    out[f"{p}best_mean"] = float(post.max())
    out[f"{p}second_best_mean"] = float(np.sort(post)[-2]) if k > 1 else 0.0
    out[f"{p}mean_of_means"] = float(post.mean())
    out[f"{p}sd_of_means"] = float(post.std(ddof=1)) if k > 1 else 0.0
    out[f"{p}max_n"] = float(n.max())
    out[f"{p}mean_n"] = float(n.mean())
    out[f"{p}n_singletons"] = float((n <= 1).sum())

    # ---- search history ----
    hist = history or SearchHistory()
    for w in WINDOWS:
        out[f"{p}new_arms_last_{w}"] = float(hist.searched_in_last(w))
        out[f"{p}best_mean_gain_last_{w}"] = float(hist.improvement_over(w))
    for w in (10, 25):
        out[f"{p}search_frac_last_{w}"] = float(hist.search_fraction(w))
    out[f"{p}time_since_last_search"] = float(hist.time_since_last_search())

    # ---- estimated reservoir (deployable) ----
    e = PREFIX_ESTIMATE
    incumbent = float(post.max())
    # NOT `(post > incumbent).mean()`: nothing can exceed its own maximum, so that
    # expression is identically zero for every state and ships a dead regressor. The
    # informative deployable quantity is how ISOLATED the leader is among the arms we
    # already hold -- a wide gap means the incumbent is clearly the best of what we
    # have, a narrow one means the top is contested and more evidence may reorder it.
    second = float(np.sort(post)[-2]) if k > 1 else incumbent
    out[f"{e}top_gap"] = incumbent - second
    # With one arm the spread is zero and a 1e-9 floor turns this into ~1e8, a finite
    # outlier eight orders above every other value in the column. "Perfectly isolated"
    # is the meaningful reading there, so say 1.0.
    spread = incumbent - float(post.min())
    out[f"{e}top_gap_normalized"] = 1.0 if spread <= 1e-9 else (incumbent - second) / spread
    out[f"{e}frac_arms_within_5pct_of_best"] = float((post >= incumbent - 0.05).mean())
    a_hat, b_hat = _fit_beta_moments(post)
    out[f"{e}beta_a"] = float(a_hat)
    out[f"{e}beta_b"] = float(b_hat)
    out[f"{e}beta_mean"] = float(a_hat / (a_hat + b_hat))
    # Posterior-predictive P(a fresh draw beats the incumbent) under the fitted Beta.
    from scipy.stats import beta as _beta

    out[f"{e}p_new_beats_incumbent"] = float(_beta.sf(incumbent, a_hat, b_hat))
    for d in DELTAS:
        out[f"{e}p_new_beats_incumbent_plus_{d}"] = float(
            _beta.sf(min(incumbent + d, 1.0), a_hat, b_hat)
        )
    desc = np.sort(post)[::-1]
    out[f"{e}hill_tail_index"] = _hill_tail_index(desc)
    for q in (0.5, 0.9, 0.99):
        out[f"{e}quantile_{q}"] = float(np.quantile(post, q))

    # ---- oracle reservoir (analysis only) ----
    if reservoir is not None:
        o = PREFIX_ORACLE
        best_true = float(mu_true.max())
        mu_star = float(reservoir.essential_sup())
        out[f"{o}best_true_mu"] = best_true
        out[f"{o}mu_star"] = mu_star
        out[f"{o}gap_to_mu_star"] = mu_star - best_true
        out[f"{o}p_new_beats_best_true"] = float(reservoir.tail_prob(best_true))
        for d in DELTAS:
            out[f"{o}p_new_beats_best_true_plus_{d}"] = float(
                reservoir.tail_prob(min(best_true + d, 1.0))
            )
        for eps in EPSILONS:
            out[f"{o}p_within_{eps}_of_mu_star"] = float(reservoir.tail_prob(mu_star - eps))
        out[f"{o}true_gap_leader_challenger"] = float(
            mu_true[cs_leader] - mu_true[challenger]
        )
        out[f"{o}leader_is_truly_best"] = float(cs_leader == int(np.argmax(mu_true)))

    validate_columns(list(out.keys()))
    return out
