"""Explicit feature groups for the deployment study.

The offline study selected ablation columns by substring matching, which produced
sets that did not match their names (see docs/growing_bandits/DEPLOYMENT_PLAN.md,
audit C). Every group here is an explicit column list, classified by *what the
value is computed from* in `features.extract_features`:

- CLOCK     -- t, T, K and their ratios only.
- QUALITY   -- posterior means `(S+1)/(n+2)` and pull counts only; leader identity is
               `argmax(post)`. No confidence-sequence bound, no e-process anywhere in
               the computation.
- EVIDENCE  -- anything that reads the anytime-valid CS bounds `(lo, hi)` or the
               pairwise e-process, *including* features whose arm identity is chosen
               through those bounds (CS leader = argmax lcb, challenger = argmax ucb).
- HISTORY   -- the decision history (a behavioural-policy fingerprint in the corpus;
               on-policy it is the deployed rule's own past decisions).

The four groups partition the 71 deployable corpus columns exactly; `check_partition`
is asserted by `tests/test_deploy_feature_groups.py`.
"""

from __future__ import annotations

CLOCK: tuple[str, ...] = (
    "f_t",
    "f_T",
    "f_remaining_budget",
    "f_remaining_frac",
    "f_t_over_T",
    "f_K",
    "f_K_over_t",
    "f_K_over_T",
    "f_log_t",
    "f_log_K",
    "f_K_over_sqrt_t",
)

# Scale-free subset of CLOCK for the horizon-transfer test (raw t, T, remaining
# extrapolate outside the training support; these do not).
CLOCK_SF: tuple[str, ...] = (
    "f_t_over_T",
    "f_remaining_frac",
    "f_K_over_sqrt_t",
    "f_log_t",
    "f_log_K",
    "f_K_over_T",
)

QUALITY: tuple[str, ...] = (
    "f_leader_n",
    "f_leader_mean",
    "f_leader_n_frac",
    "f_best_mean",
    "f_second_best_mean",
    "f_mean_of_means",
    "f_sd_of_means",
    "f_max_n",
    "f_mean_n",
    "f_n_singletons",
    "est_top_gap",
    "est_top_gap_normalized",
    "est_frac_arms_within_5pct_of_best",
    "est_beta_a",
    "est_beta_b",
    "est_beta_mean",
    "est_p_new_beats_incumbent",
    "est_p_new_beats_incumbent_plus_0.01",
    "est_p_new_beats_incumbent_plus_0.05",
    "est_p_new_beats_incumbent_plus_0.1",
    "est_hill_tail_index",
    "est_quantile_0.5",
    "est_quantile_0.9",
    "est_quantile_0.99",
)

EVIDENCE_CS: tuple[str, ...] = (
    "f_leader_lcb",
    "f_leader_ucb",
    "f_leader_width",
    "f_csleader_n",
    "f_csleader_mean",
    "f_csleader_lcb",
    "f_csleader_ucb",
    "f_csleader_width",
    "f_csleader_n_frac",
    "f_challenger_n",
    "f_challenger_mean",
    "f_challenger_lcb",
    "f_challenger_ucb",
    "f_challenger_width",
    "f_challenger_n_frac",
    "f_empirical_gap",
    "f_lcb_lead_minus_ucb_chal",
    "f_ucb_chal_minus_lcb_lead",
    "f_is_separated",
    "f_n_plausible",
    "f_frac_plausible",
    "f_n_eliminated",
    "f_frac_eliminated",
    "f_max_width_plausible",
    "f_mean_width_plausible",
    "f_mean_width_all",
)

EVIDENCE_LOGE: tuple[str, ...] = ("f_log_e_pair",)

EVIDENCE: tuple[str, ...] = EVIDENCE_CS + EVIDENCE_LOGE

HISTORY: tuple[str, ...] = (
    "f_new_arms_last_10",
    "f_best_mean_gain_last_10",
    "f_new_arms_last_25",
    "f_best_mean_gain_last_25",
    "f_new_arms_last_50",
    "f_best_mean_gain_last_50",
    "f_search_frac_last_10",
    "f_search_frac_last_25",
    "f_time_since_last_search",
)

ALL_DEPLOYABLE: tuple[str, ...] = CLOCK + QUALITY + EVIDENCE + HISTORY

# Named feature sets used by the policy table in the deployment plan.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "clock": CLOCK,
    "clock_sf": CLOCK_SF,
    "clock_quality": CLOCK + QUALITY,
    "clock_quality_cs": CLOCK + QUALITY + EVIDENCE_CS,
    "clock_quality_evidence": CLOCK + QUALITY + EVIDENCE,
    "all71": ALL_DEPLOYABLE,
    "clock_sf_quality_cs": CLOCK_SF + QUALITY + EVIDENCE_CS,
}

FORBIDDEN_PREFIXES: tuple[str, ...] = ("oracle_", "label_", "meta_")


def check_partition(corpus_columns: list[str]) -> None:
    """Raise if the groups do not partition the corpus's deployable columns exactly."""
    deployable = sorted(c for c in corpus_columns if c.startswith(("f_", "est_")))
    groups = [CLOCK, QUALITY, EVIDENCE, HISTORY]
    union: list[str] = []
    for g in groups:
        union.extend(g)
    if len(union) != len(set(union)):
        raise ValueError("feature groups overlap")
    if sorted(union) != deployable:
        missing = sorted(set(deployable) - set(union))
        extra = sorted(set(union) - set(deployable))
        raise ValueError(f"groups != deployable columns; missing={missing} extra={extra}")


def assert_deployable(columns: list[str] | tuple[str, ...]) -> None:
    """Hard hygiene guard: no oracle/label/meta column may reach a deployed policy."""
    bad = [c for c in columns if c.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(f"forbidden columns in a deployable feature list: {bad}")
    unknown = [c for c in columns if c not in ALL_DEPLOYABLE]
    if unknown:
        raise ValueError(f"unknown feature columns: {unknown}")
