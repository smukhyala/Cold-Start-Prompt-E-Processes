"""Vectorized bandit state: M Monte Carlo replicates x Kmax arm slots.

The whole study rests on advancing many rollouts at once. Rather than simulating
one trajectory at a time, every array here carries an axis over Monte Carlo
replicates, so one "step" advances all M of them with a fixed number of numpy ops.
Because numpy cost at these sizes is dominated by per-op dispatch overhead rather
than throughput, widening M is nearly free -- which is exactly the property the
adaptive-precision labeller needs.

Three performance rules are baked into the layout, each measured rather than
assumed (see the plan's D2):

* **Storage is 1-D, with 2-D views only for argmax.** Indexing a 2-D array with two
  index arrays runs ~3-4x slower than one flat gather, and `.ravel()` on a
  non-contiguous array silently *copies*, so writes through it vanish with no error.
* **Bounds are updated incrementally.** Exactly one arm per replicate changes per
  step, so we rewrite only the M touched entries (~2.9 us regardless of K) instead
  of re-gathering all M*K bounds (201 us at K=64, 2997 us at K=1000).
* **f32 scores, f64 accumulators.** argmax on f32 is ~2x faster; the betting
  recursion keeps f64 because its error accumulates over the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cold_start.growing import rng as crn


class ForcedActionUnavailable(RuntimeError):
    """Raised when a state cannot support the counterfactual a label is defined by.

    At the live-arm cap a forced SEARCH has nowhere to put the new arm. Silently
    demoting it to a REFINE makes both oracle branches identical, so the labeller
    records A_t = 0.000000 with SE = 0.000000 -- a fabricated label that looks
    maximally precise and terminates the adaptive stopping rule on its first batch.
    Such a state is not "indifferent": SEARCH is simply not an available action there,
    so the question is undefined and the state must be excluded, not answered with a
    zero.
    """


EMPTY_UID = np.int32(-1)
NEG_SENTINEL = np.float32(-1e30)

# Domain tag for reward streams (see cold_start.growing.rng).
_REWARD_DOMAIN = 0


class BoundTable(Protocol):
    """Precomputed anytime-valid confidence bounds indexed by (n, S).

    Kept structural so `state` does not import `evidence`: any object exposing flat
    lower/upper arrays addressed as ``flat[n * stride + S]`` will do.
    """

    lower_flat: np.ndarray
    upper_flat: np.ndarray
    stride: int


@dataclass
class AgrapaParams:
    """Settings for the paper-faithful upward-betting capital carried as a feature.

    Mirrors `cold_start.inference.upward_capital.UpwardCapitalEProcess` so the
    vectorized path can be checked against that scalar reference exactly.
    """

    m0: float = 0.5
    eps: float = 1e-9

    @property
    def cap(self) -> float:
        return 0.5 / max(self.m0, self.eps)


class GrowingState:
    """M replicates of a growing arm set, stored as flat arrays of length M*Kmax."""

    def __init__(
        self,
        n_replicates: int,
        capacity: int,
        horizon: int,
        base_seed: int,
        agrapa: AgrapaParams | None = None,
    ) -> None:
        if n_replicates < 1 or capacity < 1:
            raise ValueError("n_replicates and capacity must be >= 1")
        self.M = int(n_replicates)
        self.Kmax = int(capacity)
        self.T = int(horizon)
        self.base_seed = int(base_seed)
        self.agrapa = agrapa or AgrapaParams()

        size = self.M * self.Kmax
        self.n = np.zeros(size, dtype=np.int32)
        self.S = np.zeros(size, dtype=np.int32)
        self.mu = np.zeros(size, dtype=np.float32)
        self.uid = np.full(size, EMPTY_UID, dtype=np.int32)
        self.key = np.zeros(size, dtype=np.uint64)
        self.thresh = np.zeros(size, dtype=np.uint64)
        self.lcb = np.full(size, NEG_SENTINEL, dtype=np.float32)
        self.ucb = np.full(size, NEG_SENTINEL, dtype=np.float32)
        self.tie = np.zeros(size, dtype=np.float32)

        # Betting capital (paper-faithful aGRAPA), f64 -- feature only, not used for CS.
        self.log_e = np.zeros(size, dtype=np.float64)
        self.sum_sq_dev = np.zeros(size, dtype=np.float64)
        self.lam = np.zeros(size, dtype=np.float64)

        # Per-replicate bookkeeping.
        self.Kt = np.zeros(self.M, dtype=np.int32)
        self.n_draws = np.zeros(self.M, dtype=np.int32)
        self.pulls_used = np.zeros(self.M, dtype=np.int32)
        self.t = 0

        self._row_off = np.arange(self.M, dtype=np.int64) * self.Kmax
        self._rep_idx = np.arange(self.M, dtype=np.int64)

    # ---- views -----------------------------------------------------------------

    @property
    def row_off(self) -> np.ndarray:
        """Flat offset of each replicate's row; add a column index to address a slot."""
        return self._row_off

    def view(self, flat: np.ndarray) -> np.ndarray:
        """2-D (M, Kmax) view of a flat array. Use for argmax only; never `.ravel()` it."""
        return flat.reshape(self.M, self.Kmax)

    def active_mask(self) -> np.ndarray:
        return self.view(self.uid) != EMPTY_UID

    # ---- capacity --------------------------------------------------------------

    def ensure_capacity(self, needed: int) -> None:
        """Grow to at least `needed` slots per replicate by doubling.

        Reallocation copies every array, so callers should size generously up front
        rather than growing inside a step loop.
        """
        if needed <= self.Kmax:
            return
        new_cap = self.Kmax
        while new_cap < needed:
            new_cap *= 2

        def regrow(flat: np.ndarray, fill) -> np.ndarray:
            out = np.full(self.M * new_cap, fill, dtype=flat.dtype)
            out.reshape(self.M, new_cap)[:, : self.Kmax] = flat.reshape(self.M, self.Kmax)
            return out

        self.n = regrow(self.n, 0)
        self.S = regrow(self.S, 0)
        self.mu = regrow(self.mu, 0.0)
        self.uid = regrow(self.uid, EMPTY_UID)
        self.key = regrow(self.key, 0)
        self.thresh = regrow(self.thresh, 0)
        self.lcb = regrow(self.lcb, NEG_SENTINEL)
        self.ucb = regrow(self.ucb, NEG_SENTINEL)
        self.tie = regrow(self.tie, 0.0)
        self.log_e = regrow(self.log_e, 0.0)
        self.sum_sq_dev = regrow(self.sum_sq_dev, 0.0)
        self.lam = regrow(self.lam, 0.0)

        self.Kmax = new_cap
        self._row_off = np.arange(self.M, dtype=np.int64) * self.Kmax

    # ---- mutation --------------------------------------------------------------

    def add_arms(
        self,
        mu_new: np.ndarray,
        uid_new: np.ndarray,
        where: np.ndarray | None = None,
    ) -> np.ndarray:
        """Append one arm in each selected replicate. Returns the column used in each.

        `where` selects which replicates gain an arm. This matters because SEARCH and
        REFINE replicates diverge *within* a single vectorized step: some rows extend
        their arm set while others re-pull an existing arm, and both must happen in
        the same pass.

        `uid_new` is the arm's reservoir draw index -- the CRN key coordinate. It must
        NOT be the slot index: the same reservoir draw lands in different slots in the
        SEARCH and REFINE branches, and keying on the slot would silently break the
        pairing that makes the paired rollouts worth running.
        """
        if where is None:
            sel = np.arange(self.M, dtype=np.int64)
        else:
            sel = np.flatnonzero(np.asarray(where)).astype(np.int64)
        cols_all = self.Kt.astype(np.int64)
        if sel.size == 0:
            return cols_all

        self.ensure_capacity(int(self.Kt[sel].max()) + 1)
        cols = cols_all[sel]
        lin = self._row_off[sel] + cols
        mu_new = np.asarray(mu_new)[sel] if np.ndim(mu_new) and len(np.atleast_1d(mu_new)) == self.M else np.asarray(mu_new)
        uid_new = np.asarray(uid_new)[sel] if np.ndim(uid_new) and len(np.atleast_1d(uid_new)) == self.M else np.asarray(uid_new)
        rep_sel = sel

        self.mu[lin] = np.asarray(mu_new, dtype=np.float32)
        self.uid[lin] = np.asarray(uid_new, dtype=np.int32)
        self.key[lin] = crn.arm_key(self.base_seed, rep_sel, uid_new, domain=_REWARD_DOMAIN)
        self.thresh[lin] = crn.mu_to_threshold(mu_new)
        self.tie[lin] = crn.tiebreak_epsilon(self.base_seed, rep_sel, uid_new)
        self.n[lin] = 0
        self.S[lin] = 0
        self.log_e[lin] = 0.0
        self.sum_sq_dev[lin] = 0.0
        self.lam[lin] = 0.0
        # An unpulled arm is maximally uncertain: [0, 1].
        self.lcb[lin] = np.float32(0.0)
        self.ucb[lin] = np.float32(1.0)

        self.Kt[sel] += 1
        return cols_all

    def pull(self, cols: np.ndarray, table: BoundTable) -> np.ndarray:
        """Pull one arm per replicate and refresh only those arms' bounds.

        Returns the drawn rewards as a bool array of length M.
        """
        lin = self._row_off + np.asarray(cols, dtype=np.int64)

        counter = self.n[lin].astype(np.uint64)
        reward = crn.bernoulli(self.key[lin], counter, self.thresh[lin])

        # --- betting capital (aGRAPA); f64, mirrors the scalar reference exactly ---
        x = reward.astype(np.float64)
        lam = self.lam[lin]
        term = 1.0 + lam * (x - self.agrapa.m0)
        self.log_e[lin] += np.log(np.maximum(term, 1e-300))

        n_new = self.n[lin] + 1
        s_new = self.S[lin] + reward
        self.n[lin] = n_new
        self.S[lin] = s_new

        mu_hat = (0.5 + s_new.astype(np.float64)) / (1.0 + n_new)
        ssd = self.sum_sq_dev[lin] + (x - mu_hat) ** 2
        self.sum_sq_dev[lin] = ssd
        sigma2 = (0.25 + ssd) / (1.0 + n_new)
        raw = (mu_hat - self.agrapa.m0) / np.maximum(sigma2, self.agrapa.eps)
        self.lam[lin] = np.clip(raw, 0.0, self.agrapa.cap)

        # --- anytime-valid bounds: M-entry gather, cost independent of Kmax ---
        tab = n_new.astype(np.int64) * table.stride + s_new.astype(np.int64)
        self.lcb[lin] = table.lower_flat[tab]
        self.ucb[lin] = table.upper_flat[tab]

        self.pulls_used += 1
        return reward

    # ---- cloning ---------------------------------------------------------------

    def clone(self) -> GrowingState:
        """Deep copy. The two oracle branches must not share a single buffer."""
        other = GrowingState.__new__(GrowingState)
        other.__dict__.update(
            {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()
            }
        )
        return other

    def empirical_mean(self) -> np.ndarray:
        """Beta(1,1) posterior mean (S+1)/(n+2), flat. Shrinks 1-of-1 arms."""
        return (self.S + 1.0) / (self.n + 2.0)

    def __repr__(self) -> str:
        return (
            f"GrowingState(M={self.M}, Kmax={self.Kmax}, T={self.T}, t={self.t}, "
            f"K_t in [{int(self.Kt.min())}, {int(self.Kt.max())}])"
        )
