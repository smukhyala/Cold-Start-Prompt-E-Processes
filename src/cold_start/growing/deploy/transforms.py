"""Pipeline-internal feature transforms shared by the trainer and the deployed artifacts.

A saved model pickles any custom pipeline step *by reference* (module + name), so a
transform that lives in the training script would make its artifact loadable only
where that script is importable. Everything here is importable wherever `cold_start`
is, which is what makes an artifact self-contained.

Only deployable quantities are computed here: the inputs are corpus columns a real
algorithm can observe, and nothing reads a reservoir, a true mean or an `oracle_*` /
`label_*` / `meta_*` column.
"""

from __future__ import annotations

import numpy as np

# Column order the reservoir-aware rule (plan P11) declares in its artifact. The pipeline
# derives `est_I_hat` from the last three, so every declared column is a corpus column.
RESERVOIR_RULE_FEATURES: tuple[str, ...] = (
    "est_p_new_beats_incumbent",
    "f_remaining_frac",
    "f_leader_width",
    "f_log_K",
    "est_beta_a",
    "est_beta_b",
    "f_best_mean",
)

# What the logistic step of the reservoir rule actually sees, in order.
RESERVOIR_RULE_INPUTS: tuple[str, ...] = (
    "est_p_new_beats_incumbent",
    "f_remaining_frac",
    "f_leader_width",
    "f_log_K",
    "est_I_hat",
)


def beta_excess_mean(a, b, c) -> np.ndarray:
    """`E[(X - c)_+]` for `X ~ Beta(a, b)`: the expected improvement of one fresh draw over `c`.

    Closed form `(a/(a+b)) (1 - I_c(a+1, b)) - c (1 - I_c(a, b))`, the integral of the Beta
    survival function from `c` to 1. It is the observable analogue of the oracle
    `I_t = int_c^1 P(mu > x) dx` in the plan's reservoir diagnostics.
    """
    from scipy.special import betainc

    a = np.maximum(np.asarray(a, dtype=np.float64), 1e-12)
    b = np.maximum(np.asarray(b, dtype=np.float64), 1e-12)
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return (a / (a + b)) * (1.0 - betainc(a + 1.0, b, c)) - c * (1.0 - betainc(a, b, c))


def reservoir_rule_features(X: np.ndarray) -> np.ndarray:
    """Pipeline step: the 7 declared corpus columns -> the 5 inputs of the logistic rule.

    Input columns follow `RESERVOIR_RULE_FEATURES`; the output keeps the first four and
    appends `est_I_hat` computed from `(est_beta_a, est_beta_b, f_best_mean)`.
    """
    X = np.asarray(X, dtype=np.float64)
    i_hat = beta_excess_mean(X[:, 4], X[:, 5], X[:, 6])
    return np.column_stack([X[:, :4], i_hat])
