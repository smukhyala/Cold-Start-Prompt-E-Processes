"""CRN-paired evaluation harness: one cell, one policy, M paired episodes.

This replaces `experiments/growing_bandits/evaluate_policy.py`, which gave every policy
a different seed (so policies were compared on different arm draws), scored with an
oracle-prior recommender, measured regret against the unattainable ``essential_sup()``,
and kept only cell means. Four rules fix that here:

1. **Common random numbers.** Every policy in a cell ``(env, T, cap, base_seed)`` runs
   from the same `GrowingState(base_seed)`, so the ``k``-th arm any policy discovers
   is the same reservoir draw, and its rewards are the same coin flips. Paired
   differences between policies then cancel the episode's luck.
2. **Attainable comparator.** Regret is measured against `comparators.mu_star_episode`
   (best of the first ``T`` draws of the episode's own stream) so discovery regret is
   ``>= 0`` by construction and decomposes exactly: ``R_T = R_disc + R_sel``.
3. **Every recommender, same state.** The recommendation does not affect the
   trajectory, so all of `recommenders.RECOMMENDER_NAMES` are scored from the one
   final state, and per-episode results are kept -- the unit of independence is the
   episode, and paired bootstraps need it.
4. **Hygiene.** The policy only ever sees the `GrowingState` through
   `Simulator.step` and the hooks below. Everything that reads ``state.mu``, the
   reservoir, or the oracle prior lives in functions named ``_hidden_truth_*`` or in
   `comparators`, and runs only after the policy has acted (register #4, #5, #11).

Policy hook contract (duck-typed; every baseline works with none of them):

* ``policy.before_step(state, t)`` -- called before `Simulator.step` at time ``t``.
* ``policy.after_step(state, res, t, best_posterior_mean)`` -- called after the step
  with the `StepResult` and `Simulator.best_posterior_mean(state)`; a model policy
  appends its decision history here.
* ``policy.last_decision`` -- a bool ``(M,)`` array the policy sets inside
  ``should_search`` to the action it *wanted* before the simulator applied the
  live-arm cap. The harness reads it after each step and counts a demotion wherever
  the replicate was at the cap and wanted SEARCH. It never calls ``decide`` a second
  time to recover this (that would advance stochastic policies and commitment
  counters). Policies without the attribute get ``n_demoted = -1``.
  **Store a copy** (``self.last_decision = decision.copy()``): `Simulator.decide`
  masks the array ``should_search`` returns *in place*, so the same object would
  record the post-cap action and every demotion would vanish. The harness raises if
  ``last_decision`` aliases the array the simulator acted on.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cold_start.growing.allocation import LUCB, n_eliminated
from cold_start.growing.deploy.comparators import episode_reservoir_prefix, mu_star_from_prefix
from cold_start.growing.deploy.recommenders import (
    PRIMARY_RECOMMENDER,
    RECOMMENDER_NAMES,
    recommend_all_rules,
)
from cold_start.growing.features import SearchHistory
from cold_start.growing.labeling import Snapshot
from cold_start.growing.recommend import oracle_prior_from_reservoir, recommend
from cold_start.growing.reservoirs import build_reservoir
from cold_start.growing.search_policies import SearchPolicy
from cold_start.growing.simulator import Simulator, seed_initial_arms
from cold_start.growing.state import EMPTY_UID, GrowingState
from cold_start.growing.tables import CSTable

#: Snapshot seeds are offset per replicate exactly as `generate_states._detach` does,
#: so a logged on-policy state materializes the same way a corpus state would.
_SNAPSHOT_SEED_STRIDE = 7919

#: Keys of `EpisodeResult.dynamics`, each a ``(dynamics_grid + 1,)`` array.
DYNAMICS_KEYS: tuple[str, ...] = (
    "t",
    "K_t",
    "best_discovered",
    "best_posterior_mean",
    "q_primary",
    "n_eliminated",
    "search_rate",
    "herfindahl",
)


@dataclass(frozen=True)
class CellSpec:
    """One evaluation cell: environment x horizon x cap x seed, with M episodes."""

    env_id: str
    env_spec: dict
    horizon: int
    cap: int
    base_seed: int
    n_replicates: int
    alpha: float = 0.05
    n_initial_arms: int = 2


@dataclass(frozen=True)
class LogSpec:
    """Detach the first `replicates` episodes as `Snapshot`s at each of `times`."""

    times: tuple[int, ...]
    replicates: int


@dataclass
class EpisodeResult:
    """Everything the harness measures, per episode, as ``(M,)`` arrays.

    ``q[rec]`` is ``Q_T``: the TRUE mean of the arm recommender ``rec`` names at ``T``;
    ``n_rec[rec]`` is that arm's pull count. ``mu_star`` / ``mu_star_cap`` are the
    full-search and cap-feasible comparators (`comparators`), ``best_discovered`` the
    best true mean among the arms held at ``T``. Per-episode trajectory summaries:
    ``k_final`` (arms at ``T``), ``search_frac`` (SEARCH share of the ``T - n_initial``
    policy decisions), ``cap_hit`` / ``t_cap_hit`` (first clock time with
    ``K_t >= cap``; ``-1`` if never), ``n_eliminated_final`` (arms provably not best),
    ``herfindahl`` (pull concentration ``sum_a (n_a/T)^2``), ``n_singletons_final``
    (arms pulled exactly once), ``n_demoted`` (SEARCH decisions the cap turned into
    REFINE; ``-1`` when the policy does not expose ``last_decision``).

    Regrets are methods so the recommender is always explicit:

    * ``regret(rec) = mu_star - q[rec]`` -- primary, vs the full-search oracle;
    * ``regret_disc = mu_star - best_discovered >= 0`` -- recommender-independent;
    * ``regret_sel(rec) = best_discovered - q[rec]`` -- so ``regret == disc + sel``;
    * ``regret_sup(rec) = 1 - q[rec]`` -- the old benchmark's definition, continuity only.

    ``dynamics`` (see `DYNAMICS_KEYS`), ``snapshots`` (from `LogSpec`) and
    ``final_state`` are attached for diagnostics; `to_frame` omits them.
    """

    q: dict[str, np.ndarray]
    n_rec: dict[str, np.ndarray]
    mu_star: np.ndarray
    mu_star_cap: np.ndarray
    best_discovered: np.ndarray
    k_final: np.ndarray
    search_frac: np.ndarray
    cap_hit: np.ndarray
    t_cap_hit: np.ndarray
    n_eliminated_final: np.ndarray
    herfindahl: np.ndarray
    n_singletons_final: np.ndarray
    n_demoted: np.ndarray
    dynamics: dict[str, np.ndarray] | None = None
    snapshots: list[Snapshot] = field(default_factory=list)
    final_state: GrowingState | None = None
    policy_seed: int | None = None

    @property
    def n_episodes(self) -> int:
        return int(self.mu_star.shape[0])

    def regret(self, rec: str = PRIMARY_RECOMMENDER) -> np.ndarray:
        return self.mu_star - self.q[rec]

    @property
    def regret_disc(self) -> np.ndarray:
        return self.mu_star - self.best_discovered

    def regret_sel(self, rec: str = PRIMARY_RECOMMENDER) -> np.ndarray:
        return self.best_discovered - self.q[rec]

    def regret_sup(self, rec: str = PRIMARY_RECOMMENDER) -> np.ndarray:
        return 1.0 - self.q[rec]

    def to_frame(self, policy: str, spec: CellSpec) -> pd.DataFrame:
        """One row per episode, wide.

        Columns: ``policy, env_id, horizon, cap, base_seed, alpha, n_initial_arms,
        policy_seed, episode, mu_star, mu_star_cap, best_discovered, regret_disc,
        k_final, search_frac, cap_hit, t_cap_hit, n_eliminated_final, herfindahl,
        n_singletons_final, n_demoted`` and, for every recommender ``r`` in
        ``self.q``: ``q_r, n_rec_r, regret_r, regret_sel_r, regret_sup_r``.
        """
        m = self.n_episodes
        cols: dict[str, np.ndarray | str | int | float | None] = {
            "policy": policy,
            "env_id": spec.env_id,
            "horizon": int(spec.horizon),
            "cap": int(spec.cap),
            "base_seed": int(spec.base_seed),
            "alpha": float(spec.alpha),
            "n_initial_arms": int(spec.n_initial_arms),
            "policy_seed": self.policy_seed,
            "episode": np.arange(m, dtype=np.int64),
            "mu_star": self.mu_star,
            "mu_star_cap": self.mu_star_cap,
            "best_discovered": self.best_discovered,
            "regret_disc": self.regret_disc,
            "k_final": self.k_final,
            "search_frac": self.search_frac,
            "cap_hit": self.cap_hit,
            "t_cap_hit": self.t_cap_hit,
            "n_eliminated_final": self.n_eliminated_final,
            "herfindahl": self.herfindahl,
            "n_singletons_final": self.n_singletons_final,
            "n_demoted": self.n_demoted,
        }
        for rec in self.q:
            cols[f"q_{rec}"] = self.q[rec]
            cols[f"n_rec_{rec}"] = self.n_rec[rec]
            cols[f"regret_{rec}"] = self.regret(rec)
            cols[f"regret_sel_{rec}"] = self.regret_sel(rec)
            cols[f"regret_sup_{rec}"] = self.regret_sup(rec)
        return pd.DataFrame(cols)


# ---- policy-side plumbing --------------------------------------------------------


def default_policy_seed(policy: SearchPolicy) -> int:
    """``crc32(policy.name)``: the plan's per-policy seed component, process-stable."""
    name = getattr(policy, "name", None) or type(policy).__name__
    return int(zlib.crc32(str(name).encode()))


def _isolate_policy_rng(policy: SearchPolicy, spec: CellSpec, policy_seed: int | None) -> int | None:
    """Give the policy a fresh generator keyed by ``[base_seed, policy_seed]``.

    Every `SearchPolicy` built without an explicit ``rng`` holds
    ``np.random.default_rng(0)``: distinct objects, identical streams. Left alone, two
    stochastic policies in one cell would draw the same coins, and the same policy in
    two cells would too. Replacing the generator here makes the isolation the
    harness's responsibility rather than the caller's memory. With ``policy_seed=None``
    the plan's default ``crc32(policy_name)`` is used, so a caller that passes nothing
    still gets a stream that is fresh per policy per cell.
    """
    if not hasattr(policy, "rng"):
        return None
    seed = default_policy_seed(policy) if policy_seed is None else int(policy_seed)
    policy.rng = np.random.default_rng([int(spec.base_seed), seed])
    return seed


# ---- hidden-truth diagnostics (harness-side only) --------------------------------


def _hidden_truth_best_discovered(state: GrowingState) -> np.ndarray:
    """Max TRUE mean over the active arms of each replicate. Reads ``state.mu``."""
    mu = state.view(state.mu).astype(np.float64)
    return np.where(state.active_mask(), mu, -np.inf).max(axis=1)


def _pull_herfindahl(state: GrowingState) -> np.ndarray:
    """``sum_a (n_a / sum_b n_b)^2`` over active arms: 1 = all pulls on one arm."""
    n = state.view(state.n).astype(np.float64)
    total = n.sum(axis=1, keepdims=True)
    share = np.divide(n, total, out=np.zeros_like(n), where=total > 0.0)
    return np.where(state.active_mask(), share * share, 0.0).sum(axis=1)


def _assert_crn_pairing(state: GrowingState, prefix: np.ndarray, horizon: int) -> None:
    """Every discovered arm must be, bit for bit, its draw in the comparator prefix.

    This is the identity the whole paired design rests on, so it is checked on every
    run rather than trusted: a prefix passed in for the wrong seed, or a simulator
    change that re-keyed the reservoir stream, would otherwise make the comparator
    silently unrelated to the arms the policy actually saw.
    """
    if prefix.shape != (state.M, horizon):
        raise ValueError(f"comparator prefix must be ({state.M}, {horizon}); got {prefix.shape}")
    active = state.active_mask()
    k_cols = min(state.Kmax, horizon)
    if k_cols < state.Kmax and bool(active[:, k_cols:].any()):
        raise RuntimeError("a replicate holds more arms than the horizon allows draws")
    act = active[:, :k_cols]
    mu = state.view(state.mu).astype(np.float64)[:, :k_cols]
    if not np.array_equal(np.where(act, mu, 0.0), np.where(act, prefix[:, :k_cols], 0.0)):
        raise RuntimeError(
            "CRN pairing broken: the discovered arms are not the comparator prefix's draws "
            "(wrong base_seed for a precomputed prefix, or the reservoir stream changed)"
        )


def _detach_snapshots(
    state: GrowingState,
    t: int,
    spec: CellSpec,
    policy_name: str,
    decisions: list[np.ndarray],
    best_trace: list[np.ndarray],
    n_reps: int,
) -> list[Snapshot]:
    """Peel the first `n_reps` replicates out as standalone `Snapshot`s.

    Mirrors `generate_states._detach` (history = decisions and best-mean trace of
    steps ``n_initial_arms .. t-1``), so the scalar `features.extract_features` sees
    an on-policy state exactly as it sees a corpus state.
    """
    uid2 = state.view(state.uid)
    n2, s2, mu2 = state.view(state.n), state.view(state.S), state.view(state.mu)
    dec = np.array(decisions).T if decisions else np.zeros((state.M, 0), dtype=bool)
    trace = np.array(best_trace).T if best_trace else np.zeros((state.M, 0))

    snaps: list[Snapshot] = []
    for m in range(min(int(n_reps), state.M)):
        active = uid2[m] != EMPTY_UID
        snaps.append(
            Snapshot(
                n=n2[m, active].copy(),
                successes=s2[m, active].copy(),
                mu=mu2[m, active].copy(),
                t=int(t),
                horizon=int(spec.horizon),
                n_draws=int(state.n_draws[m]),
                base_seed=int(state.base_seed) + _SNAPSHOT_SEED_STRIDE * m,
                meta={
                    "env_id": spec.env_id,
                    "policy": policy_name,
                    "allocation": "lucb",
                    "replicate": m,
                    "cap": int(spec.cap),
                    "history": SearchHistory(
                        decisions=dec[m].copy(), best_mean_trace=trace[m].copy()
                    ),
                },
            )
        )
    return snaps


# ---- dynamics grid -----------------------------------------------------------------


def _grid_times(horizon: int, grid: int, first_visible: int) -> dict[int, list[int]]:
    """Map each loop time ``t`` to the grid slots recorded there.

    Slot ``g`` holds the state at ``t_g = round(g * T / grid)`` (half up), the visited
    state nearest normalized time ``g / grid``. The harness never sees the state
    before the warm start, so slots with ``t_g < n_initial_arms`` take the first
    visible state; when ``T < grid`` several slots share a time and repeat its values.
    """
    out: dict[int, list[int]] = {}
    for g in range(int(grid) + 1):
        t_g = int(np.floor(g * horizon / grid + 0.5))
        t_g = min(max(t_g, first_visible), horizon)
        out.setdefault(t_g, []).append(g)
    return out


class _Dynamics:
    """Cross-episode means of the trajectory statistics on the normalized-time grid.

    ``search_rate[g]`` is the SEARCH fraction over the steps that carried the state
    from slot ``g-1`` to slot ``g`` (``t_{g-1} <= t < t_g``); NaN where that interval
    holds no policy step (the first slot, and repeated slots when ``T < grid``).
    """

    def __init__(self, grid: int) -> None:
        size = int(grid) + 1
        self.arrays = {k: np.full(size, np.nan, dtype=np.float64) for k in DYNAMICS_KEYS}
        self._searched_since = 0.0
        self._steps_since = 0

    def note_step(self, searched: np.ndarray) -> None:
        self._searched_since += float(searched.sum())
        self._steps_since += 1

    def record(self, slots: list[int], state: GrowingState, t: int, sim: Simulator) -> None:
        a = self.arrays
        k_t = float(state.Kt.mean())
        best_disc = float(_hidden_truth_best_discovered(state).mean())
        best_pm = float(sim.best_posterior_mean(state).mean())
        q_primary = float(recommend(state, PRIMARY_RECOMMENDER).mu.mean())
        n_elim = float(n_eliminated(state).mean())
        hh = float(_pull_herfindahl(state).mean())
        if self._steps_since:
            rate = self._searched_since / (self._steps_since * state.M)
        else:
            rate = np.nan
        for i, g in enumerate(slots):
            a["t"][g] = t
            a["K_t"][g] = k_t
            a["best_discovered"][g] = best_disc
            a["best_posterior_mean"][g] = best_pm
            a["q_primary"][g] = q_primary
            a["n_eliminated"][g] = n_elim
            a["search_rate"][g] = rate if i == 0 else np.nan
            a["herfindahl"][g] = hh
        self._searched_since = 0.0
        self._steps_since = 0


# ---- the cell ----------------------------------------------------------------------


def run_cell(
    spec: CellSpec,
    policy: SearchPolicy,
    *,
    table=None,
    reservoir=None,
    dynamics_grid: int = 50,
    log_states: LogSpec | None = None,
    policy_seed: int | None = None,
    comparator_prefix: np.ndarray | None = None,
    oracle_prior: tuple[float, float] | None = None,
) -> EpisodeResult:
    """Run `policy` for all ``M`` episodes of `spec` and score every recommender.

    Protocol (identical for every policy, matching the corpus harness): warm start
    with ``n_initial_arms`` arms pulled once each, then one step per unit of budget
    from ``t = n_initial_arms`` to ``T`` under ``Simulator(LUCB(), policy, cap)``.

    Cell constants -- `table`, `reservoir`, `comparator_prefix`
    (`comparators.episode_reservoir_prefix(reservoir, base_seed, M, T)`) and
    `oracle_prior` (`recommend.oracle_prior_from_reservoir(reservoir)`) -- may be
    passed to share them across the policies of a cell; otherwise they are built
    here. Sharing matters for the mixture family, whose inverse CDF is a 60-step
    bisection: the prefix alone costs more than the rollout. A passed prefix is
    verified against the arms the run discovered (`_assert_crn_pairing`).

    `dynamics_grid` sets the number of normalized-time intervals for
    `EpisodeResult.dynamics` (``0`` disables it). `log_states` detaches on-policy
    `Snapshot`s at the listed times. `policy_seed` keys the policy's private
    generator (see `_isolate_policy_rng`).
    """
    horizon, m_reps = int(spec.horizon), int(spec.n_replicates)
    cap, n0 = int(spec.cap), int(spec.n_initial_arms)
    if m_reps < 1:
        raise ValueError(f"n_replicates must be >= 1; got {m_reps}")
    if not 1 <= n0 <= horizon:
        raise ValueError(f"need 1 <= n_initial_arms <= horizon; got {n0}, {horizon}")
    if cap < 1:
        raise ValueError(f"cap must be >= 1; got {cap}")
    if dynamics_grid < 0:
        raise ValueError(f"dynamics_grid must be >= 0; got {dynamics_grid}")
    if log_states is not None:
        bad = [t for t in log_states.times if not n0 <= int(t) <= horizon]
        if bad:
            raise ValueError(
                f"log_states.times must lie in [{n0}, {horizon}] (the states the loop "
                f"visits); got {bad}"
            )

    if reservoir is None:
        reservoir = build_reservoir(spec.env_spec)
    if table is None:
        table = CSTable.load_or_build(horizon, spec.alpha)
    table_horizon = getattr(table, "horizon", None)
    if table_horizon is not None and int(table_horizon) < horizon:
        raise ValueError(f"table covers n <= {table_horizon} but the horizon is {horizon}")

    used_seed = _isolate_policy_rng(policy, spec, policy_seed)
    policy_name = str(getattr(policy, "name", type(policy).__name__))

    state = GrowingState(m_reps, 8, horizon, base_seed=spec.base_seed)
    seed_initial_arms(state, reservoir, n0, table)
    sim = Simulator(table, reservoir, LUCB(), policy, horizon, max_live_arms=cap)

    before_step = getattr(policy, "before_step", None)
    after_step = getattr(policy, "after_step", None)
    keep_trace = log_states is not None
    need_best_pm = keep_trace or after_step is not None

    n_search = np.zeros(m_reps, dtype=np.int64)
    t_cap_hit = np.full(m_reps, -1, dtype=np.int64)
    n_demoted = np.zeros(m_reps, dtype=np.int64)
    demotions_known = True
    decisions: list[np.ndarray] = []
    best_trace: list[np.ndarray] = []
    snapshots: list[Snapshot] = []
    log_times = set(int(t) for t in log_states.times) if log_states is not None else set()
    dyn = _Dynamics(dynamics_grid) if dynamics_grid > 0 else None
    slots_at = _grid_times(horizon, dynamics_grid, n0) if dyn is not None else {}

    def note_cap(t: int) -> None:
        newly = (state.Kt >= cap) & (t_cap_hit < 0)
        t_cap_hit[newly] = t

    for t in range(n0, horizon):
        at_cap = state.Kt >= cap
        note_cap(t)
        if dyn is not None and t in slots_at:
            dyn.record(slots_at[t], state, t, sim)
        if t in log_times:
            snapshots.extend(
                _detach_snapshots(
                    state, t, spec, policy_name, decisions, best_trace, log_states.replicates
                )
            )

        if before_step is not None:
            before_step(state, t)
        res = sim.step(state, t)
        best_pm = sim.best_posterior_mean(state) if need_best_pm else None
        if after_step is not None:
            after_step(state, res, t, best_pm)

        n_search += res.searched
        if dyn is not None:
            dyn.note_step(res.searched)
        if keep_trace:
            decisions.append(res.searched.copy())
            best_trace.append(best_pm.copy())
        if demotions_known:
            wanted = getattr(policy, "last_decision", None)
            if wanted is None:
                demotions_known = False
            elif wanted is res.searched:
                raise RuntimeError(
                    f"{policy_name}.last_decision is the very array should_search "
                    "returned, which Simulator.decide has since masked in place; store "
                    "a copy so the pre-cap decision survives"
                )
            else:
                n_demoted += at_cap & np.asarray(wanted, dtype=bool)

    note_cap(horizon)
    if dyn is not None and horizon in slots_at:
        dyn.record(slots_at[horizon], state, horizon, sim)
    if horizon in log_times:
        snapshots.extend(
            _detach_snapshots(
                state, horizon, spec, policy_name, decisions, best_trace, log_states.replicates
            )
        )

    pulls = state.view(state.n).sum(axis=1)
    if not bool(np.all(pulls == horizon)):
        raise RuntimeError(
            f"budget not conserved: pull totals {np.unique(pulls)} != horizon {horizon}"
        )

    # ---- hidden truth: only now, and only here ---------------------------------
    if comparator_prefix is None:
        prefix = episode_reservoir_prefix(reservoir, spec.base_seed, m_reps, horizon)
    else:
        prefix = np.asarray(comparator_prefix, dtype=np.float64)
    _assert_crn_pairing(state, prefix, horizon)
    if oracle_prior is None:
        oracle_prior = oracle_prior_from_reservoir(reservoir)

    active = state.active_mask()
    n2 = state.view(state.n)
    best_discovered = _hidden_truth_best_discovered(state)
    recs = recommend_all_rules(state, oracle_prior)
    q = {name: recs[name].mu.astype(np.float64) for name in RECOMMENDER_NAMES}
    n_rec = {
        name: state.n[state.row_off + recs[name].cols].astype(np.int64)
        for name in RECOMMENDER_NAMES
    }
    if not demotions_known:
        n_demoted = np.full(m_reps, -1, dtype=np.int64)

    return EpisodeResult(
        q=q,
        n_rec=n_rec,
        mu_star=mu_star_from_prefix(prefix),
        mu_star_cap=mu_star_from_prefix(prefix, cap=cap),
        best_discovered=best_discovered,
        k_final=state.Kt.astype(np.int64),
        search_frac=n_search / float(max(horizon - n0, 1)),
        cap_hit=t_cap_hit >= 0,
        t_cap_hit=t_cap_hit,
        n_eliminated_final=n_eliminated(state).astype(np.int64),
        herfindahl=_pull_herfindahl(state),
        n_singletons_final=(active & (n2 == 1)).sum(axis=1).astype(np.int64),
        n_demoted=n_demoted,
        dynamics=dyn.arrays if dyn is not None else None,
        snapshots=snapshots,
        final_state=state,
        policy_seed=used_seed,
    )
