"""Oracle labelling: what is a fresh arm actually worth, here, right now?

For a saved state we clone it twice, force SEARCH in one copy and REFINE in the
other, run the *same* continuation policy to the horizon in both, and record the true
mean of the arm each branch ends up recommending. The difference

    A_t = E[mu_recommended | SEARCH] - E[mu_recommended | REFINE]

is the study's response variable: `A_t > 0` means searching was the better move.

Two design choices here are load-bearing and easy to get wrong.

**Precision is absolute, not a t-statistic.** The obvious reading of "spend more
rollouts near the boundary" -- keep adding batches while `|A|/SE < 3` -- never
terminates, because where `A_t` is genuinely ~0 no finite `M` makes that ratio large.
With `sd(mu_rec) = 0.06` and `A = 0.001` you would need `M ~ 32,000`, and every such
state (i.e. most of the interesting ones) would pin at the cap. We stop instead when
`SE(A) < target_se`, which is both the correct criterion for a regression label and
several times cheaper.

**A_t is a difference of recommended true means, not of regrets.** Algebraically
identical, since the `mu_star` terms cancel, but it avoids differencing two large
nearly-equal numbers and leaves the label well defined for reservoirs whose supremum
is not attained.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cold_start.growing.allocation import plausible_mask
from cold_start.growing.recommend import PRIMARY_RULE, recommend, recommend_with_oracle_prior
from cold_start.growing.simulator import Simulator
from cold_start.growing.state import AgrapaParams, ForcedActionUnavailable, GrowingState

DEFAULT_BATCHES: tuple[int, ...] = (256, 256, 512, 1024)
HIST_BINS = 16


@dataclass
class Snapshot:
    """A single bandit state, detached from any replicate axis.

    This is what gets stored in the state pool and later replayed: per-arm sufficient
    statistics plus the true means (simulation only), the clock, and how many
    reservoir draws have been consumed so a replay continues the same draw sequence.
    """

    n: np.ndarray
    successes: np.ndarray
    mu: np.ndarray
    t: int
    horizon: int
    n_draws: int
    base_seed: int
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(len(self.n))

    def validate(self) -> None:
        if not (len(self.n) == len(self.successes) == len(self.mu)):
            raise ValueError("snapshot arrays must have equal length")
        if self.k == 0:
            raise ValueError("snapshot has no arms; nothing to refine")
        if np.any(self.successes > self.n) or np.any(self.n < 0):
            raise ValueError("invalid (n, S): successes cannot exceed pulls")
        if not 0 <= self.t <= self.horizon:
            raise ValueError(f"t={self.t} outside [0, {self.horizon}]")
        # The safety property: no arm may have been pulled more times than the
        # confidence-sequence table has rows, or `materialize` indexes past its end and
        # dies with a raw IndexError deep inside a worker. A harvested snapshot also
        # satisfies the stronger `sum(n) == t` (every pull spends one budget unit, and
        # `tests/test_growing_generation.py` asserts it), but hand-built states used to
        # probe a specific configuration need not, so it is not enforced here.
        if int(np.max(self.n)) > self.horizon:
            raise ValueError(
                f"an arm has {int(np.max(self.n))} pulls but the horizon is "
                f"{self.horizon}; the confidence-sequence table only covers n <= T"
            )


def materialize(
    snap: Snapshot,
    n_replicates: int,
    table,
    seed_offset: int = 0,
    agrapa: AgrapaParams | None = None,
) -> GrowingState:
    """Broadcast a snapshot to `n_replicates` identical copies.

    `seed_offset` shifts the random stream so successive Monte Carlo batches are
    independent, while both branches within a batch share it -- that shared offset is
    exactly what couples the paired rollouts.
    """
    snap.validate()
    capacity = max(snap.k * 2, snap.k + 8)
    state = GrowingState(
        n_replicates=n_replicates,
        capacity=capacity,
        horizon=snap.horizon,
        base_seed=snap.base_seed + seed_offset,
        agrapa=agrapa,
    )
    for j in range(snap.k):
        state.add_arms(
            np.full(n_replicates, snap.mu[j], dtype=np.float32),
            np.full(n_replicates, j, dtype=np.int32),
        )
    # Install the observed sufficient statistics, then refresh bounds from the table.
    n2, s2 = state.view(state.n), state.view(state.S)
    for j in range(snap.k):
        n2[:, j] = int(snap.n[j])
        s2[:, j] = int(snap.successes[j])
    idx = state.view(state.n).astype(np.int64) * table.stride + state.view(state.S).astype(np.int64)
    active = state.active_mask()
    lcb2, ucb2 = state.view(state.lcb), state.view(state.ucb)
    lcb2[active] = np.asarray(table.lower_flat)[idx[active]]
    ucb2[active] = np.asarray(table.upper_flat)[idx[active]]

    state.n_draws[:] = snap.n_draws
    state.pulls_used[:] = snap.t
    state.t = snap.t
    return state


@dataclass
class PairedOutcome:
    """Accumulates the PAIRED difference d_i = mu_rec(SEARCH) - mu_rec(REFINE).

    Pairing is the whole reason the branches share a random stream, and estimating
    the standard error without it throws that away. Measured on this simulator, the
    paired variance is 0.3%-3% of the unpaired variance at long horizons, because
    95-99% of replicates recommend the *same* arm in both branches and contribute
    d_i = 0 exactly. Differencing first cancels the variance the two branches hold in
    common; differencing last leaves it in, and it then swamps a signal that is
    genuinely O(1/remaining budget).
    """

    total_d: float = 0.0
    total_d_sq: float = 0.0
    total_s: float = 0.0
    total_r: float = 0.0
    total_s_sq: float = 0.0
    total_r_sq: float = 0.0
    count: int = 0
    n_zero: int = 0
    hist_search: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, dtype=np.int64))
    hist_refine: np.ndarray = field(default_factory=lambda: np.zeros(HIST_BINS, dtype=np.int64))

    def add(self, mu_search: np.ndarray, mu_refine: np.ndarray) -> None:
        d = np.asarray(mu_search, dtype=np.float64) - np.asarray(mu_refine, dtype=np.float64)
        self.total_d += float(d.sum())
        self.total_d_sq += float(np.square(d).sum())
        self.total_s += float(mu_search.sum())
        self.total_r += float(mu_refine.sum())
        self.total_s_sq += float(np.square(mu_search, dtype=np.float64).sum())
        self.total_r_sq += float(np.square(mu_refine, dtype=np.float64).sum())
        self.count += int(d.size)
        self.n_zero += int((d == 0.0).sum())
        self.hist_search += np.histogram(mu_search, bins=HIST_BINS, range=(0.0, 1.0))[0]
        self.hist_refine += np.histogram(mu_refine, bins=HIST_BINS, range=(0.0, 1.0))[0]

    @staticmethod
    def _var(total: float, total_sq: float, count: int) -> float:
        if count < 2:
            return float("nan")
        mean = total / count
        return max(total_sq / count - mean**2, 0.0) * count / (count - 1)

    @property
    def advantage(self) -> float:
        return self.total_d / self.count if self.count else float("nan")

    @property
    def se(self) -> float:
        """Paired standard error of the mean difference."""
        v = self._var(self.total_d, self.total_d_sq, self.count)
        return float(np.sqrt(v / self.count)) if self.count else float("nan")

    @property
    def se_unpaired(self) -> float:
        """What the SE would have been without pairing; recorded to show CRN's value."""
        vs = self._var(self.total_s, self.total_s_sq, self.count)
        vr = self._var(self.total_r, self.total_r_sq, self.count)
        return float(np.sqrt(vs / self.count + vr / self.count)) if self.count else float("nan")

    @property
    def mean_search(self) -> float:
        return self.total_s / self.count if self.count else float("nan")

    @property
    def mean_refine(self) -> float:
        return self.total_r / self.count if self.count else float("nan")

    @property
    def sd_search(self) -> float:
        return float(np.sqrt(self._var(self.total_s, self.total_s_sq, self.count)))

    @property
    def sd_refine(self) -> float:
        return float(np.sqrt(self._var(self.total_r, self.total_r_sq, self.count)))

    @property
    def frac_identical(self) -> float:
        """Share of replicates where both branches recommended the same arm."""
        return self.n_zero / self.count if self.count else float("nan")


@dataclass
class LabelResult:
    """One labelled state: the oracle advantage and everything needed to trust it."""

    advantage: float
    se: float
    se_unpaired: float
    frac_identical: float
    n_replicates: int
    n_batches: int
    commit_steps_applied: int
    mean_search: float
    mean_refine: float
    sd_search: float
    sd_refine: float
    hist_search: np.ndarray
    hist_refine: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def oracle_action(self) -> str:
        return "SEARCH" if self.advantage > 0 else "REFINE"

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return (self.advantage - z * self.se, self.advantage + z * self.se)


def _marginal_diagnostics(
    before: GrowingState, after_search: GrowingState, after_refine: GrowingState
) -> dict[str, float]:
    """What each action bought on the very next step (spec section 11).

    Secondary targets: they do not define the label, but they are what explains *why*
    the oracle prefers one action, which is the part a fitted polynomial cannot tell us.
    """
    out: dict[str, float] = {}

    new_col = after_search.Kt - 1
    rows = np.arange(after_search.M)
    lin_new = after_search.row_off + new_col.astype(np.int64)
    plaus = plausible_mask(after_search)
    out["search_p_new_plausible"] = float(plaus[rows, new_col].mean())

    # Empty slots must be masked before any argmax. `(S + 1) / (n + 2)` scores an
    # unallocated slot at exactly 0.5, which beats every real arm in a low-mean
    # environment -- the diagnostic would then report that the new arm rarely leads
    # when in fact it was losing to a slot that holds nothing.
    act = after_search.active_mask()
    tie = after_search.view(after_search.tie)
    pm = np.where(act, after_search.view(after_search.empirical_mean()) + tie, -np.inf)
    out["search_p_new_is_leader"] = float((np.argmax(pm, axis=1) == new_col).mean())
    lcb_after = np.where(act, after_search.view(after_search.lcb) + tie, -np.inf)
    out["search_p_new_is_cs_leader"] = float((np.argmax(lcb_after, axis=1) == new_col).mean())

    best_before = np.where(before.active_mask(), before.view(before.mu), -np.inf).max(axis=1)
    best_after = np.where(
        after_search.active_mask(), after_search.view(after_search.mu), -np.inf
    ).max(axis=1)
    out["search_expected_best_mu_gain"] = float((best_after - best_before).mean())
    out["search_new_arm_mu_mean"] = float(after_search.mu[lin_new].mean())

    width_before = (before.view(before.ucb) - before.view(before.lcb))
    width_after = after_refine.view(after_refine.ucb) - after_refine.view(after_refine.lcb)
    act = before.active_mask()
    out["refine_mean_width_reduction"] = float(
        width_before[act].mean() - width_after[act].mean()
    )
    out["refine_n_plausible_before"] = float(plausible_mask(before).sum(axis=1).mean())
    out["refine_n_plausible_after"] = float(plausible_mask(after_refine).sum(axis=1).mean())
    out["refine_p_eliminated_one"] = float(
        (plausible_mask(after_refine).sum(axis=1) < plausible_mask(before).sum(axis=1)).mean()
    )
    return out


def label_state(
    snap: Snapshot,
    sim_factory: Callable[[int], Simulator],
    table,
    target_se: float = 0.005,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCHES,
    max_replicates: int = 4096,
    rule: str = PRIMARY_RULE,
    collect_diagnostics: bool = True,
    commit_steps: int = 1,
    oracle_prior: tuple[float, float] | None = None,
    max_live_arms: int | None = None,
) -> LabelResult:
    """Estimate A_t for one snapshot, adding batches until `SE(A) < target_se`.

    `sim_factory(seed_offset)` must return a simulator; both branches of a batch get
    the same offset so their reward and reservoir streams line up.

    `oracle_prior`, when given, recommends by posterior mean under a prior matched to
    the TRUE reservoir. That is the Bayes-optimal recommendation for expected simple
    regret -- the objective this study actually states -- so it is the right rule for
    an *oracle* label, and it is legitimate here precisely because a label is not a
    feature. The lower confidence bound is optimal for a risk-averse objective instead
    (it can never recommend a one-pull arm, which makes late search look worthless),
    and an empirical-Bayes fit is estimated from a survivorship-biased sample.

    `commit_steps` is how many rounds the forced action is held before the shared
    continuation policy resumes. At `commit_steps=1` this is exactly the brief's
    single-action definition. Larger values answer a related but different question --
    "is it worth committing to searching for a while?" -- and measurably carry far
    more signal when the remaining budget is large, because a good continuation
    policy absorbs a single forced action almost entirely.
    """
    snap.validate()
    if max_live_arms is not None and snap.k >= max_live_arms:
        raise ForcedActionUnavailable(
            f"state holds {snap.k} arms at a cap of {max_live_arms}: a forced SEARCH "
            f"cannot be applied, so A_t is undefined here"
        )
    paired = PairedOutcome()
    diagnostics: dict[str, float] = {}
    used = 0
    n_batches = 0

    for batch_index in range(len(batch_sizes) + 8):
        size = batch_sizes[min(batch_index, len(batch_sizes) - 1)]
        if used + size > max_replicates:
            size = max_replicates - used
        if size <= 0:
            break

        offset = 1_000_003 * (batch_index + 1)
        sim = sim_factory(offset)
        base = materialize(snap, size, table, seed_offset=offset)

        branch_s = base.clone()
        branch_r = base.clone()
        applied = 0
        for i in range(commit_steps):
            if snap.t + i >= snap.horizon:
                break
            # Only the FIRST forced step defines the label, so only it is strict.
            sim.step(branch_s, snap.t + i, force=True, strict=(i == 0))
            sim.step(branch_r, snap.t + i, force=False, strict=(i == 0))
            applied += 1
            if collect_diagnostics and n_batches == 0 and i == 0:
                diagnostics = _marginal_diagnostics(base, branch_s, branch_r)

        resume = min(snap.t + commit_steps, snap.horizon)
        sim.run_to_horizon(branch_s, resume)
        sim.run_to_horizon(branch_r, resume)

        if oracle_prior is not None:
            mu_s = recommend_with_oracle_prior(branch_s, oracle_prior).mu
            mu_r = recommend_with_oracle_prior(branch_r, oracle_prior).mu
        else:
            mu_s = recommend(branch_s, rule=rule).mu
            mu_r = recommend(branch_r, rule=rule).mu
        paired.add(mu_s, mu_r)

        used += size
        n_batches += 1

        if paired.se < target_se or used >= max_replicates:
            break

    return LabelResult(
        advantage=paired.advantage,
        se=paired.se,
        se_unpaired=paired.se_unpaired,
        frac_identical=paired.frac_identical,
        n_replicates=used,
        n_batches=n_batches,
        commit_steps_applied=applied,
        mean_search=paired.mean_search,
        mean_refine=paired.mean_refine,
        sd_search=paired.sd_search,
        sd_refine=paired.sd_refine,
        hist_search=paired.hist_search,
        hist_refine=paired.hist_refine,
        diagnostics=diagnostics,
    )


def label_state_multi(
    snap: Snapshot,
    sim_factory: Callable[[int], Simulator],
    table,
    commit_steps: tuple[int, ...] = (1, 4, 16),
    **kwargs,
) -> dict[int, LabelResult]:
    """Label one state at several commitment horizons.

    `commit_steps=1` remains the canonical label -- it is the quantity the research
    brief defines. The larger horizons are recorded alongside it because the pilot
    showed the single-action advantage is indistinguishable from zero in about three
    quarters of states: a good continuation policy simply undoes one forced action.
    How the advantage grows with commitment length is therefore not a nuisance
    parameter, it is a result about how much any single SEARCH/REFINE decision can
    possibly be worth.
    """
    return {
        k: label_state(snap, sim_factory, table, commit_steps=k, **kwargs)
        for k in commit_steps
    }
