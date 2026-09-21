"""Registry of every SEARCH policy in the deployment plan's policy table, by name.

The harness compares policies on common random numbers, so every policy in a cell
must be constructible from a name plus a parameter dict -- including the learned ones,
which need the cell's horizon, replicate count and CS table to build their feature
layer. `POLICY_SPECS` is the plan's table (DEPLOYMENT_PLAN.md "Policies") written as
data; `make_policy` turns a spec into a `SearchPolicy`.

Two things live here that are not simple lookups:

* `ReservoirRule` (plan P11, section 13): a hand rule that searches while the expected
  improvement a fresh arm would bring, integrated over the remaining budget, exceeds
  the leader's current uncertainty. It is the deployable reading of the oracle
  `I_t = INT_{mu_inc}^1 (1 - F(x)) dx`, with the reservoir CDF replaced by the same
  method-of-moments Beta fit `features.extract_features` records as `est_beta_a/b`.
* `wrap_policy_seed`: the policy-internal RNG is `default_rng([base_seed, crc32(name)])`,
  fresh per policy per cell, so a stochastic policy is reproducible and two policies
  never share a stream (the old benchmark seeded with the salted `hash(str)`).

Nothing here reads `state.mu`, `state.thresh` or a reservoir object.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import betainc

from cold_start.growing.deploy.artifacts import load_model
from cold_start.growing.deploy.model_policy import ModelPolicy
from cold_start.growing.search_policies import (
    BernoulliSearch,
    BestMeanGate,
    BracketExpansion,
    DecisionContext,
    EvidenceGatedSchedule,
    LevelScaledSchedule,
    PowerSchedule,
    SearchPolicy,
    TailAdaptiveSchedule,
    UniformRandom,
)
from cold_start.growing.state import GrowingState

# name -> {"kind": ..., **params}. Params here are the plan's defaults; `make_policy`
# lets a caller override any of them (tuned `c`, validation-selected `tau`, ...).
POLICY_SPECS: dict[str, dict[str, Any]] = {
    # P0
    "always_search": {"kind": "always_search"},
    # P1: search until K0 arms exist, then only refine. Warm start already gives 2.
    "refine_after_init_K2": {"kind": "refine_after_init", "K0": 2},
    "refine_after_init_K4": {"kind": "refine_after_init", "K0": 4},
    "refine_after_init_K8": {"kind": "refine_after_init", "K0": 8},
    # P2
    "uniform": {"kind": "uniform"},
    "fixed_K4": {"kind": "fixed_K", "K": 4},
    "fixed_K16": {"kind": "fixed_K", "K": 16},
    # P3: `c` is a placeholder pending tuning on the tuning seeds.
    "power_quarter": {"kind": "power", "alpha": 0.25, "c": 1.0},
    "power_cbrt": {"kind": "power", "alpha": 1.0 / 3.0, "c": 1.5},
    "power_sqrt": {"kind": "power", "alpha": 0.5, "c": 1.0},
    "power_t23": {"kind": "power", "alpha": 2.0 / 3.0, "c": 0.8},
    # Primary reference (H1a): the label continuation policy.
    "cp0": {"kind": "cp0", "alpha": 0.5, "c": 1.0, "min_pulls_per_arm": 2},
    "bracket": {"kind": "bracket", "base_width": 4},
    # P11 hand rule; `tau` selected on validation seeds.
    "reservoir_rule": {"kind": "reservoir_rule", "tau": 1.0},
    # Pre-registration 4's null model: K_target = c * T^alpha * (1 + b * (1 - q_t)).
    "adaptive_K": {"kind": "adaptive_K", "alpha": 0.5, "c": 2.0, "b": 0.0},
    # Pre-registration 5's null model: SEARCH while best_mean < theta and K_t < c * T^alpha.
    "bestmean_K": {"kind": "bestmean_K", "theta": 0.65, "alpha": 0.5, "c": 4.0},
    # Pre-registration 6's null model: K_target = c * T^alpha * exp(b * (0.5 - level_t)).
    "level_K": {"kind": "level_K", "alpha": 0.5, "c": 2.0, "b": 0.0},
    # Learned (P4-P10, P12): `artifact` (path or loaded dict) must come from `params`.
    "model": {"kind": "model"},
}

KINDS: tuple[str, ...] = (
    "always_search",
    "refine_after_init",
    "fixed_K",
    "uniform",
    "power",
    "cp0",
    "bracket",
    "adaptive_K",
    "bestmean_K",
    "level_K",
    "model",
    "reservoir_rule",
)


# ---- Beta method of moments, vectorized ----------------------------------------


def fit_beta_moments_vec(post: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-replicate twin of `features._fit_beta_moments`, with its exact fallbacks.

    `post` and `active` are `(M, Kmax)`. Every fallback returns Beta(1, 1): fewer than
    two arms, zero spread, or an over-dispersed population no Beta can match. The
    strength `nu` is capped by the arm count for the reason given there -- a prior
    fitted to a handful of survivorship-biased arms must not outweigh them.
    """
    post = np.asarray(post, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    k = active.sum(axis=1).astype(np.float64)
    k_safe = np.maximum(k, 1.0)
    vals = np.where(active, post, 0.0)
    m_raw = vals.sum(axis=1) / k_safe
    dev = np.where(active, (post - m_raw[:, None]) ** 2, 0.0)
    v = dev.sum(axis=1) / np.maximum(k - 1.0, 1.0)
    m = np.clip(m_raw, 1e-6, 1.0 - 1e-6)
    usable = (k >= 2.0) & (v > 1e-12) & (v < m * (1.0 - m))
    nu = m * (1.0 - m) / np.where(usable, v, 1.0) - 1.0
    nu = np.clip(nu, 2.0, np.maximum(2.0, k))
    a = np.where(usable, m * nu, 1.0)
    b = np.where(usable, (1.0 - m) * nu, 1.0)
    return a, b


def expected_improvement_beta(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """`E[(X - c)_+]` for `X ~ Beta(a, b)`, in closed form.

    `INT_c^1 (x - c) f(x) dx = (a / (a + b)) * (1 - I_c(a + 1, b)) - c * (1 - I_c(a, b))`,
    which equals `INT_c^1 (1 - F(x)) dx` by parts. `betainc` is the regularized
    incomplete Beta `I_c(a, b)`.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    mean = a / (a + b)
    tail_plus = 1.0 - betainc(a + 1.0, b, c)
    tail = 1.0 - betainc(a, b, c)
    return np.maximum(mean * tail_plus - c * tail, 0.0)


# ---- P11 hand rule -------------------------------------------------------------


class ReservoirRule(SearchPolicy):
    """SEARCH iff `I_hat * (T - t) > tau * leader_width`.

    Left side: how much a fresh arm is expected to improve on the incumbent, times the
    rounds left to exploit it. Right side: how uncertain we still are about the arm
    we hold, scaled by `tau`. Larger `tau` demands more expected gain per unit of
    leader uncertainty before searching, so it searches less.

    The incumbent is the empirical leader `argmax (S+1)/(n+2)` and `leader_width` is
    its CS width, matching `f_leader_mean` / `f_leader_width` in the corpus features.
    """

    name = "reservoir_rule"
    needs_evidence = False

    def __init__(self, tau: float = 1.0, rng: np.random.Generator | None = None) -> None:
        super().__init__(rng)
        self.tau = float(tau)
        if not np.isfinite(self.tau) or self.tau < 0.0:
            raise ValueError(f"tau must be a finite non-negative float; got {tau}")
        self.last_decision: np.ndarray | None = None
        self.last_i_hat: np.ndarray | None = None

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        active = state.active_mask()
        post = state.view(state.empirical_mean())
        masked = np.where(active, post, -np.inf)
        leader = np.argmax(masked, axis=1)
        lin = state.row_off + leader
        width = state.ucb[lin].astype(np.float64) - state.lcb[lin].astype(np.float64)
        has_arms = state.Kt > 0
        incumbent = np.where(has_arms, masked[np.arange(state.M), leader], 0.0)

        a, b = fit_beta_moments_vec(post, active)
        i_hat = expected_improvement_beta(a, b, incumbent)
        action = (i_hat * ctx.remaining > self.tau * width) | ~has_arms

        self.last_i_hat = i_hat
        self.last_decision = action.copy()
        return action


# ---- factory -------------------------------------------------------------------


def _resolve_artifact(artifact: Any, artifact_dir: Path | None) -> dict[str, Any]:
    if artifact is None:
        raise ValueError("a 'model' policy needs params['artifact'] (path or loaded dict)")
    if hasattr(artifact, "keys"):
        return artifact
    path = Path(artifact)
    if not path.is_absolute() and artifact_dir is not None:
        path = Path(artifact_dir) / path
    return load_model(path)


def make_policy(
    name: str,
    *,
    horizon: int,
    n_replicates: int,
    table,
    params: dict[str, Any] | None = None,
    artifact_dir: Path | None = None,
    pairwise=None,
    rng: np.random.Generator | None = None,
) -> SearchPolicy:
    """Build the named policy. `params` overrides the registered defaults.

    `name` is a `POLICY_SPECS` key, or `model:<path>` as shorthand for the learned kind
    with that artifact (relative paths resolve against `artifact_dir`).
    """
    if name.startswith("model:"):
        spec: dict[str, Any] = {"kind": "model", "artifact": name[len("model:") :]}
    elif name in POLICY_SPECS:
        spec = dict(POLICY_SPECS[name])
    else:
        raise KeyError(f"unknown policy {name!r}; registered={sorted(POLICY_SPECS)}")
    spec.update(params or {})
    kind = spec.pop("kind")
    if kind not in KINDS:
        raise KeyError(f"unknown policy kind {kind!r}; kinds={KINDS}")

    if kind == "always_search":
        return BernoulliSearch(p=1.0, rng=rng)
    if kind == "refine_after_init":
        return PowerSchedule(alpha=0.0, c=float(spec["K0"]), rng=rng)
    if kind == "fixed_K":
        return PowerSchedule(alpha=0.0, c=float(spec["K"]), rng=rng)
    if kind == "uniform":
        return UniformRandom(rng=rng)
    if kind == "power":
        return PowerSchedule(alpha=float(spec["alpha"]), c=float(spec["c"]), rng=rng)
    if kind == "cp0":
        return EvidenceGatedSchedule(
            alpha=float(spec.get("alpha", 0.5)),
            c=float(spec.get("c", 1.0)),
            min_pulls_per_arm=int(spec.get("min_pulls_per_arm", 2)),
            rng=rng,
        )
    if kind == "bracket":
        return BracketExpansion(base_width=int(spec.get("base_width", 4)), rng=rng)
    if kind == "reservoir_rule":
        return ReservoirRule(tau=float(spec.get("tau", 1.0)), rng=rng)
    if kind == "adaptive_K":
        return TailAdaptiveSchedule(
            alpha=float(spec["alpha"]), c=float(spec["c"]), b=float(spec.get("b", 0.0)), rng=rng
        )
    if kind == "bestmean_K":
        return BestMeanGate(theta=float(spec["theta"]), alpha=float(spec["alpha"]), c=float(spec["c"]), rng=rng)
    if kind == "level_K":
        return LevelScaledSchedule(
            alpha=float(spec["alpha"]), c=float(spec["c"]), b=float(spec.get("b", 0.0)), rng=rng
        )
    # kind == "model"
    artifact = _resolve_artifact(spec.get("artifact"), artifact_dir)
    return ModelPolicy(
        artifact,
        horizon=horizon,
        n_replicates=n_replicates,
        table=table,
        tau=spec.get("tau"),
        k=spec.get("k"),
        affordability_guard=bool(spec.get("affordability_guard", False)),
        per_step=bool(spec.get("per_step", False)),
        pairwise=pairwise,
        rng=rng,
    )


def policy_seed(base_seed: int, name: str) -> list[int]:
    """SeedSequence entropy for a policy's private stream: `[base_seed, crc32(name)]`."""
    return [int(base_seed), zlib.crc32(name.encode("utf-8"))]


def wrap_policy_seed(policy: SearchPolicy, base_seed: int, name: str) -> SearchPolicy:
    """Give `policy` its own reproducible RNG, keyed on the cell seed and its name.

    Every policy in a cell shares `base_seed` for the environment (that is the CRN
    pairing); their *internal* randomness must differ, or two coin-flip policies would
    flip the same coins. `crc32` rather than `hash()` because the latter is salted per
    process and would make the benchmark unreproducible.
    """
    policy.rng = np.random.default_rng(policy_seed(base_seed, name))
    return policy
