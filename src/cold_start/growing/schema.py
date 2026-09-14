"""Dataset column contract and config models for the growing-bandits study.

The single most important invariant in this study is that a *deployable* policy may
only see quantities a real algorithm could compute. Oracle knowledge -- true arm
means, true reservoir tail probabilities -- is available for analysis but must never
reach a model we intend to run for real.

Rather than trusting discipline, that boundary is enforced structurally: every
emitted column carries a mandatory prefix, and `design_matrix_columns` refuses to
hand back oracle columns unless an ablation explicitly asks for them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---- column prefixes ----------------------------------------------------------

PREFIX_META = "meta_"      # ids, seeds, provenance; never a feature
PREFIX_FEATURE = "f_"      # observable state; deployable
PREFIX_ESTIMATE = "est_"   # reservoir quantities estimated from observed arms; deployable
PREFIX_ORACLE = "oracle_"  # true mu / true reservoir; ANALYSIS ONLY
PREFIX_LABEL = "label_"    # targets and their precision

DEPLOYABLE_PREFIXES = (PREFIX_FEATURE, PREFIX_ESTIMATE)
ALL_PREFIXES = (PREFIX_META, PREFIX_FEATURE, PREFIX_ESTIMATE, PREFIX_ORACLE, PREFIX_LABEL)


class OracleLeakageError(RuntimeError):
    """Raised when oracle columns reach a design matrix that has not opted in."""


def validate_columns(columns: list[str]) -> None:
    """Every emitted column must carry a known prefix, so nothing escapes the contract."""
    bad = [c for c in columns if not c.startswith(ALL_PREFIXES)]
    if bad:
        raise ValueError(
            f"{len(bad)} column(s) lack a known prefix {ALL_PREFIXES}: {sorted(bad)[:10]}"
        )


def design_matrix_columns(
    columns: list[str],
    *,
    allow_oracle: bool = False,
    include_meta: bool = False,
) -> list[str]:
    """Select model inputs from a dataset's columns, refusing oracle leakage.

    `allow_oracle=True` is legitimate for exactly one thing: ablation F, which asks
    how much predictive headroom oracle reservoir knowledge would buy. It is never
    correct for a model that will be deployed by `evaluate_policy`.
    """
    oracle = [c for c in columns if c.startswith(PREFIX_ORACLE)]
    if oracle and not allow_oracle:
        raise OracleLeakageError(
            f"{len(oracle)} oracle column(s) reached a deployable design matrix "
            f"(e.g. {sorted(oracle)[:5]}). Pass allow_oracle=True only for ablation F."
        )
    keep = tuple(DEPLOYABLE_PREFIXES) + ((PREFIX_ORACLE,) if allow_oracle else ())
    if include_meta:
        keep = keep + (PREFIX_META,)
    return [c for c in columns if c.startswith(keep)]


def label_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith(PREFIX_LABEL)]


# ---- config models ------------------------------------------------------------


class ReservoirSpec(BaseModel):
    """Which reservoir family and parameters generate an environment."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvidenceSpec(BaseModel):
    alpha: float = 0.05
    prior_a: float = 1.0
    prior_b: float = 1.0
    pairwise_grid: int = 512
    agrapa_m0: float = 0.5


class GenerationSpec(BaseModel):
    """State-pool generation (spec section 7).

    `states_per_horizon` is a hard quota, not a target. Harvesting states along
    trajectories without one makes long horizons yield ~20x more states than short
    ones and silently dominate the corpus.
    """

    model_config = ConfigDict(extra="forbid")

    horizons: list[int] = Field(default_factory=lambda: [50, 100, 200, 500, 1000])
    states_per_horizon: int = 20_000
    max_live_arms: int = 64
    snapshots_per_trajectory: int = 8
    behavioral_policies: list[str] = Field(default_factory=list)
    allocation_rules: list[str] = Field(default_factory=lambda: ["ucb", "lucb", "racing"])
    reservoirs: list[ReservoirSpec] = Field(default_factory=list)
    seed: int = 20260910


class LabelingSpec(BaseModel):
    """Oracle rollout settings (spec sections 9 and 12).

    Precision is absolute, not a t-statistic: a rule like "add batches while
    |A|/SE < 3" never terminates where A is genuinely ~0, which is most states and
    exactly the ones the study needs.
    """

    model_config = ConfigDict(extra="forbid")

    target_se: float = 0.005
    batch_sizes: list[int] = Field(default_factory=lambda: [256, 256, 512, 1024])
    max_replicates: int = 4096
    continuation_policy: str = "cp0"
    recommendation_rule: str = "posterior_mean_oracle"
    histogram_bins: int = 16
    seed: int = 20260910


class GrowingConfig(BaseModel):
    """Top-level config, mirroring the repo's pydantic-YAML convention."""

    model_config = ConfigDict(extra="forbid")

    name: str
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    labeling: LabelingSpec = Field(default_factory=LabelingSpec)
    evidence: EvidenceSpec = Field(default_factory=EvidenceSpec)
    output_dir: str = "data/"
    n_workers: int = 9

    def hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()
