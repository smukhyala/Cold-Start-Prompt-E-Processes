from __future__ import annotations

import math

import numpy as np
import pytest

from cold_start.inference.global_null import GlobalNullEProcess
from cold_start.inference.hedged_capital import HedgedCapitalEProcess
from cold_start.inference.upward_capital import UpwardCapitalEProcess


def test_product_combine_sums_logs():
    a = HedgedCapitalEProcess(m0=0.5)
    b = HedgedCapitalEProcess(m0=0.5)
    a.update(1); a.update(1); a.update(0)
    b.update(0); b.update(1); b.update(1)
    g = GlobalNullEProcess(combine="product")
    assert abs(g.log_e([a, b]) - (a.log_e + b.log_e)) < 1e-12


def test_mixture_combine_is_logsumexp_minus_logK():
    a = HedgedCapitalEProcess(m0=0.5)
    b = HedgedCapitalEProcess(m0=0.5)
    a.update(1); a.update(1)
    b.update(0); b.update(0)
    g = GlobalNullEProcess(combine="mixture")
    expected = math.log((math.exp(a.log_e) + math.exp(b.log_e)) / 2)
    assert abs(g.log_e([a, b]) - expected) < 1e-9


def test_linear_mixture_uniform_matches_mixture():
    """With uniform weights w_a = 1/K, linear_mixture equals 'mixture'."""
    a = HedgedCapitalEProcess(m0=0.5)
    b = HedgedCapitalEProcess(m0=0.5)
    c = HedgedCapitalEProcess(m0=0.5)
    for x in (1, 0, 1, 1):
        a.update(x)
    for x in (0, 0, 1, 0):
        b.update(x)
    for x in (1, 1, 1, 0):
        c.update(x)

    g_mix = GlobalNullEProcess(combine="mixture")
    g_linear = GlobalNullEProcess(combine="linear_mixture")  # uniform default

    arms = {"a": a, "b": b, "c": c}
    assert abs(g_mix.log_e(arms) - g_linear.log_e(arms)) < 1e-9


def test_linear_mixture_respects_configured_weights():
    a = HedgedCapitalEProcess(m0=0.5)
    b = HedgedCapitalEProcess(m0=0.5)
    a.update(1); a.update(1); a.update(1)  # strong evidence
    b.update(0); b.update(0); b.update(0)  # against null

    weights = {"a": 0.9, "b": 0.1}
    g = GlobalNullEProcess(combine="linear_mixture", weights=weights)
    expected = math.log(0.9 * math.exp(a.log_e) + 0.1 * math.exp(b.log_e))
    assert abs(g.log_e({"a": a, "b": b}) - expected) < 1e-9


def test_linear_mixture_rejects_negative_weights():
    with pytest.raises(ValueError):
        GlobalNullEProcess(combine="linear_mixture", weights={"a": 1.2, "b": -0.2})


def test_linear_mixture_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError):
        GlobalNullEProcess(combine="linear_mixture", weights={"a": 0.4, "b": 0.4})


def test_weights_disallowed_for_product_and_mixture():
    with pytest.raises(ValueError):
        GlobalNullEProcess(combine="product", weights={"a": 1.0})
    with pytest.raises(ValueError):
        GlobalNullEProcess(combine="mixture", weights={"a": 1.0})


def test_linear_mixture_requires_mapping_at_call_time():
    a = HedgedCapitalEProcess(m0=0.5)
    a.update(1)
    g = GlobalNullEProcess(combine="linear_mixture")
    with pytest.raises(TypeError):
        g.log_e([a])  # iterable is not enough; need arm_id keys


def test_linear_mixture_valid_under_null_via_simulation():
    """Proposition 2: a convex combination of valid e-processes is itself a
    valid e-process. Ville bound P(sup_t E_t ≥ 1/α | H_0) ≤ α should hold
    under linear_mixture with the one-sided upward-capital construction."""
    alpha = 0.1
    T = 200
    n_runs = 1000
    rng = np.random.default_rng(99)
    sups = np.empty(n_runs)
    for i in range(n_runs):
        a = UpwardCapitalEProcess(m0=0.5)
        b = UpwardCapitalEProcess(m0=0.5)
        c = UpwardCapitalEProcess(m0=0.5)
        per_arm = {"a": a, "b": b, "c": c}
        g = GlobalNullEProcess(
            combine="linear_mixture",
            weights={"a": 0.5, "b": 0.25, "c": 0.25},
        )
        sup = 0.0
        for _ in range(T):
            # Pick an arm round-robin-ish (predictable allocation under H_0
            # is allowed; doesn't break Proposition 2).
            for aid in ("a", "b", "c"):
                x = float(rng.random() < 0.5)  # p = m_0 → boundary of H_0
                per_arm[aid].update(x)
            le = g.log_e(per_arm)
            if le > sup:
                sup = le
        sups[i] = sup
    rate = float(np.mean(sups >= math.log(1.0 / alpha)))
    assert rate <= alpha + 0.04, f"linear_mixture violated Ville bound: rate={rate:.4f}"
