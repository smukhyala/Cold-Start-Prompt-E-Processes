"""Deployable SEARCH policy backed by a saved sklearn pipeline.

`experiments/growing_bandits/evaluate_policy.LearnedPolicy` handles six hand-coded
variables and folds the scaler into the coefficients by hand -- which is exactly the
step the failure-mode register (#5) identifies as where the previous deployment went
wrong. This policy instead takes a whole model artifact (`deploy.artifacts`): the
fitted pipeline object, scaler included, plus the explicit column list it was trained
on. Features are produced by the vectorized parity layer (`deploy.features_vec`), the
pipeline is asked for `P(SEARCH)`, and the decision is `p > tau`.

Commitment semantics (register #6, made explicit here). The label a k-round model was
fitted on is "SEARCH now and keep searching for k rounds", so the deployed rule keeps
that promise: a fresh decision at step `t` is held for `min(k, T - t)` consecutive
steps (a commitment never outlives the horizon), and the model is consulted again
only when the counter runs out. `per_step=True` (plan variant P6') disables that and
re-evaluates every step, isolating the model from the commitment mechanism.
`affordability_guard=True` (P6g) adds cp0's `remaining >= 2 * (K_t + 1)` veto *after*
the commitment, so a committed SEARCH can still be refused when there is no budget
left to evaluate the arm it would recruit.

Hygiene: nothing here reads `state.mu`, `state.thresh` or a reservoir. The policy
exposes `last_decision` (the action it *wanted*, before the simulator's cap demotion)
and `last_proba` so the harness can count demotions and log probabilities without
re-running the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cold_start.growing.deploy.feature_groups import (
    EVIDENCE_LOGE,
    HISTORY,
    assert_deployable,
)
from cold_start.growing.deploy.features_vec import extract_features_vec, feature_matrix
from cold_start.growing.deploy.history_vec import VecSearchHistory
from cold_start.growing.search_policies import DecisionContext, SearchPolicy
from cold_start.growing.state import GrowingState

DEFAULT_TAU = 0.5


class ModelPolicy(SearchPolicy):
    """SEARCH iff `pipeline.predict_proba(features)[:, 1] > tau`, held for `k` rounds."""

    name = "model"
    needs_evidence = False

    def __init__(
        self,
        artifact: dict[str, Any],
        *,
        horizon: int,
        n_replicates: int,
        table,
        tau: float | None = None,
        k: int | None = None,
        affordability_guard: bool = False,
        per_step: bool = False,
        pairwise=None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.features: list[str] = list(artifact["features"])
        assert_deployable(self.features)
        self.pipeline = artifact["pipeline"]
        if not hasattr(self.pipeline, "predict_proba"):
            raise TypeError("model artifact pipeline must implement predict_proba")
        self.meta = dict(artifact.get("meta") or {})

        if tau is None:
            tau = artifact.get("tau")
        if tau is None:
            tau = DEFAULT_TAU
        self.tau = float(tau)
        if not np.isfinite(self.tau):
            raise ValueError(f"tau must be finite; got {tau}")

        if k is None:
            k = artifact["k"]
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"commitment horizon k must be >= 1; got {k}")

        self.horizon = int(horizon)
        self.M = int(n_replicates)
        if self.horizon < 1 or self.M < 1:
            raise ValueError("horizon and n_replicates must be >= 1")
        self.table = table
        self.affordability_guard = bool(affordability_guard)
        self.per_step = bool(per_step)

        # Only pay for what the model reads: the pairwise table (a multi-GB memmap at
        # long horizons) and the decision history are built only for models that use
        # them. Explicit column-name membership, never substring matching.
        self.uses_history = any(c in HISTORY for c in self.features)
        self.uses_log_e = any(c in EVIDENCE_LOGE for c in self.features)
        if self.uses_log_e and pairwise is None:
            from cold_start.growing.deploy.pairwise_table import get_pairwise_table

            pairwise = get_pairwise_table(self.horizon)
        self.pairwise = pairwise

        # If the pipeline was fitted on a DataFrame, sklearn remembers the column names
        # and warns on every ndarray it is later given. Feed it a frame with the same
        # names in that case -- and refuse an artifact whose feature list disagrees
        # with what the pipeline was actually fitted on.
        fitted_names = getattr(self.pipeline, "feature_names_in_", None)
        self._as_frame = fitted_names is not None
        if self._as_frame and list(fitted_names) != self.features:
            raise ValueError(
                "artifact 'features' does not match the pipeline's fitted feature names: "
                f"{self.features} vs {list(fitted_names)}"
            )

        self.reset()

    # ---- state -----------------------------------------------------------------

    def reset(self) -> None:
        """Clear commitments, history and counters so one object can run a fresh cell."""
        self.history = VecSearchHistory(self.M, self.horizon) if self.uses_history else None
        self.commit_left = np.zeros(self.M, dtype=np.int64)
        self.commit_action = np.zeros(self.M, dtype=bool)
        self.last_decision = np.zeros(self.M, dtype=bool)
        self.last_proba = np.full(self.M, np.nan, dtype=np.float64)
        # Diagnostics, all counted over (replicate, step) pairs.
        self.n_decisions = 0  # fresh model evaluations
        self.n_search_decided = 0  # ... of which p > tau
        self.n_committed_steps = 0  # steps that followed a standing commitment
        self.n_guard_vetoes = 0  # SEARCH actions refused by the affordability guard
        self.n_cap_demoted = 0  # wanted SEARCH, simulator forced REFINE (via after_step)
        self.n_nonfinite_rows = 0  # feature rows that needed nan_to_num before predict

    # ---- model -----------------------------------------------------------------

    def predict_proba(self, state: GrowingState, t: int, horizon: int) -> np.ndarray:
        """`P(SEARCH)` for every replicate from the vectorized parity features.

        `nan_to_num` mirrors the trainer's `_matrix` convention so the pipeline sees the
        same transformation at deployment as it did in training; the count of rows
        that needed it is kept as a diagnostic because a non-finite feature at
        deployment is a parity bug, not a normal state.
        """
        feats = extract_features_vec(
            state, t, horizon, self.table, self.pairwise, self.history, columns=self.features
        )
        X = np.asarray(feature_matrix(feats, self.features), dtype=np.float64)
        finite = np.isfinite(X).all(axis=1)
        if not finite.all():
            self.n_nonfinite_rows += int((~finite).sum())
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self._as_frame:
            import pandas as pd

            X = pd.DataFrame(X, columns=self.features)
        return np.asarray(self.pipeline.predict_proba(X)[:, 1], dtype=np.float64)

    # ---- decision --------------------------------------------------------------

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        t, horizon = int(ctx.t), int(ctx.horizon)
        if state.M != self.M:
            raise ValueError(f"policy was built for M={self.M} replicates, state has {state.M}")
        has_arms = state.Kt > 0
        if self.per_step:
            fresh = np.ones(self.M, dtype=bool)
        else:
            fresh = self.commit_left <= 0

        # All replicates start uncommitted at the same step and receive the same
        # `min(k, T - t)` window, so they re-decide in lockstep: `fresh` is all-or-
        # nothing in practice. Computing features for every row when any row needs a
        # decision therefore wastes nothing, and skipping the model entirely on
        # committed steps is where the k-fold saving actually comes from.
        decide = fresh & has_arms
        p = np.full(self.M, np.nan, dtype=np.float64)
        if decide.any():
            p_all = self.predict_proba(state, t, horizon)
            p[decide] = p_all[decide]
            model_action = p_all > self.tau
            self.commit_action[decide] = model_action[decide]
            self.commit_left[decide] = min(self.k, horizon - t)
            self.n_decisions += int(decide.sum())
            self.n_search_decided += int((model_action & decide).sum())

        action = self.commit_action.copy()
        self.n_committed_steps += int((~fresh & has_arms).sum())
        self.commit_left -= 1
        np.maximum(self.commit_left, 0, out=self.commit_left)

        if self.affordability_guard:
            # cp0's `_affordable`: never recruit an arm the remaining budget cannot
            # give at least two pulls, alongside the arms already held. Applied after
            # the commitment on purpose, so it can veto a committed SEARCH.
            affordable = ctx.remaining >= 2 * (state.Kt + 1)
            self.n_guard_vetoes += int((action & ~affordable).sum())
            action &= affordable
        # A replicate with no arms has nothing to refine. The model is never consulted
        # for it (the feature layer needs at least one arm) and no commitment is made.
        action |= ~has_arms

        self.last_decision = action.copy()
        self.last_proba = p
        return action

    # ---- harness hooks ---------------------------------------------------------

    def after_step(self, state: GrowingState, res, t: int, best_mean: np.ndarray) -> None:
        """Record what actually happened (post cap demotion), as the corpus did."""
        searched = np.asarray(res.searched, dtype=bool)
        self.n_cap_demoted += int((self.last_decision & ~searched).sum())
        if self.history is not None:
            self.history.record(searched, np.asarray(best_mean, dtype=np.float64))

    def counters(self) -> dict[str, int]:
        return {
            "n_decisions": self.n_decisions,
            "n_search_decided": self.n_search_decided,
            "n_committed_steps": self.n_committed_steps,
            "n_guard_vetoes": self.n_guard_vetoes,
            "n_cap_demoted": self.n_cap_demoted,
            "n_nonfinite_rows": self.n_nonfinite_rows,
        }

    def __repr__(self) -> str:
        return (
            f"ModelPolicy(k={self.k}, tau={self.tau:.3f}, n_features={len(self.features)}, "
            f"per_step={self.per_step}, guard={self.affordability_guard})"
        )
