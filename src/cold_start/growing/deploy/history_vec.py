"""Batched decision history: `features.SearchHistory` for M replicates at once.

The corpus's history features were computed by `SearchHistory` on one trajectory at
a time. A deployed policy sees M trajectories in one `(M, Kmax)` state, so it needs
the same nine numbers as `(M,)` arrays -- and they must be the *same* numbers, because
the policy was fitted to them. Every method here therefore restates the scalar
definition on a preallocated `(M, capacity)` buffer rather than re-deriving it, and
`scalar(m)` hands back a genuine `SearchHistory` so parity can be asserted directly.

What is recorded and when (verified against `generate_states.harvest`): a snapshot
at time `t` is taken *before* step `t`, and the history at that moment holds, for
every completed step, the step's actual `StepResult.searched` (after the live-arm cap
has demoted any blocked SEARCH) and `sim.best_posterior_mean(state)` evaluated
*after* that step. A harness reproduces that by calling `record` once per step,
after `sim.step`, and extracting features before it.
"""

from __future__ import annotations

import numpy as np

from cold_start.growing.features import SearchHistory


class VecSearchHistory:
    """Per-replicate SEARCH decisions and best-posterior-mean trace, `(M, capacity)`."""

    def __init__(self, n_replicates: int, capacity: int) -> None:
        if n_replicates < 1:
            raise ValueError(f"n_replicates must be >= 1; got {n_replicates}")
        if capacity < 0:
            raise ValueError(f"capacity must be >= 0; got {capacity}")
        self.M = int(n_replicates)
        self.capacity = int(capacity)
        # Preallocated once: a step loop must not reallocate per step. The trace is
        # float64 so a difference of two entries is exact; the scalar path stores the
        # simulator's float32 values, and the two agree to float32 rounding.
        self._decisions = np.zeros((self.M, self.capacity), dtype=bool)
        self._trace = np.zeros((self.M, self.capacity), dtype=np.float64)
        self._length = 0

    # ---- construction ----------------------------------------------------------

    @classmethod
    def from_scalar(cls, hists: list[SearchHistory], capacity: int) -> VecSearchHistory:
        """Stack per-replicate scalar histories (all of one length) into a batch."""
        if not hists:
            raise ValueError("from_scalar needs at least one history")
        lengths = {int(h.decisions.size) for h in hists}
        if len(lengths) != 1:
            raise ValueError(f"histories must share one length; got {sorted(lengths)}")
        length = lengths.pop()
        for h in hists:
            if int(h.best_mean_trace.size) != length:
                raise ValueError("decisions and best_mean_trace must have equal length")
        if length > capacity:
            raise ValueError(f"histories of length {length} exceed capacity {capacity}")
        out = cls(len(hists), capacity)
        for m, h in enumerate(hists):
            out._decisions[m, :length] = np.asarray(h.decisions, dtype=bool)
            out._trace[m, :length] = np.asarray(h.best_mean_trace, dtype=np.float64)
        out._length = length
        return out

    # ---- recording -------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of completed steps recorded so far (the scalar `decisions.size`)."""
        return self._length

    def record(self, searched: np.ndarray, best_mean: np.ndarray) -> None:
        """Append one step for every replicate."""
        if self._length >= self.capacity:
            raise ValueError(
                f"history is full ({self.capacity} steps); size it to the horizon up front"
            )
        searched = np.asarray(searched, dtype=bool)
        best_mean = np.asarray(best_mean, dtype=np.float64)
        if searched.shape != (self.M,) or best_mean.shape != (self.M,):
            raise ValueError(
                f"expected two arrays of shape ({self.M},); got {searched.shape} and "
                f"{best_mean.shape}"
            )
        self._decisions[:, self._length] = searched
        self._trace[:, self._length] = best_mean
        self._length += 1

    # ---- the nine history features --------------------------------------------

    def searched_in_last(self, w: int) -> np.ndarray:
        """`decisions[-w:].sum()`; 0 for an empty history."""
        L = self._length
        start = max(0, L - int(w))
        return self._decisions[:, start:L].sum(axis=1).astype(np.int64)

    def search_fraction(self, w: int) -> np.ndarray:
        """`decisions[-w:].mean()`; 0.0 for an empty history."""
        L = self._length
        if L == 0:
            return np.zeros(self.M, dtype=np.float64)
        start = max(0, L - int(w))
        tail = self._decisions[:, start:L]
        return tail.sum(axis=1) / float(tail.shape[1])

    def time_since_last_search(self) -> np.ndarray:
        """Rounds since the last SEARCH; the full length if there has never been one.

        Empty history is 0 (not the length), exactly as in `SearchHistory`.
        """
        L = self._length
        if L == 0:
            return np.zeros(self.M, dtype=np.int64)
        dec = self._decisions[:, :L]
        ever = dec.any(axis=1)
        # argmax over the reversed row is the distance back to the latest True.
        since = np.argmax(dec[:, ::-1], axis=1).astype(np.int64)
        return np.where(ever, since, np.int64(L))

    def improvement_over(self, w: int) -> np.ndarray:
        """`trace[-1] - trace[max(0, len - w - 1)]`; 0.0 with fewer than two entries."""
        L = self._length
        if L < 2:
            return np.zeros(self.M, dtype=np.float64)
        past = max(0, L - int(w) - 1)
        return self._trace[:, L - 1] - self._trace[:, past]

    # ---- detaching -------------------------------------------------------------

    def scalar(self, m: int) -> SearchHistory:
        """Replicate `m` as the scalar object the corpus used (copies, never views)."""
        L = self._length
        return SearchHistory(
            decisions=self._decisions[m, :L].copy(),
            best_mean_trace=self._trace[m, :L].copy(),
        )
