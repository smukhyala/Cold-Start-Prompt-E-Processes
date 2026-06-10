from __future__ import annotations

import math

import numpy as np
import pytest

from cold_start.inference.upward_capital import UpwardCapitalEProcess


def _simulate_sup_log_e(n_runs: int, T: int, p: float, m0: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sups = np.empty(n_runs)
    for i in range(n_runs):
        ep = UpwardCapitalEProcess(m0=m0)
        sup = 0.0
        for _ in range(T):
            x = float(rng.random() < p)
            le = ep.update(x)
            if le > sup:
                sup = le
        sups[i] = sup
    return sups


def test_log_e_starts_at_zero():
    ep = UpwardCapitalEProcess(m0=0.5)
    assert ep.log_e == 0.0
    assert ep.n == 0


def test_first_step_log_e_is_zero_because_first_bet_is_zero():
    # aGRAPA chooses λ_1 from info through t=0, which is none — so λ_1 = 0
    # and the first multiplicative term is 1+0 = 1, leaving log_e at 0.
    ep = UpwardCapitalEProcess(m0=0.5)
    ep.update(1.0)
    assert ep.log_e == 0.0
    assert ep.n == 1


def test_log_e_is_finite_under_balanced_data():
    ep = UpwardCapitalEProcess(m0=0.5)
    le_seq = [ep.update(x) for x in (0, 1, 1, 0, 0, 1, 0, 1, 0, 1)]
    assert all(math.isfinite(v) for v in le_seq)
    assert ep.n == 10


def test_log_e_upper_alias_matches_log_e():
    ep = UpwardCapitalEProcess(m0=0.4)
    for x in (1, 1, 0, 1, 0):
        ep.update(x)
    assert ep.log_e_upper == ep.log_e


def test_type_I_error_controlled_under_null():
    """Ville's inequality: P(sup_t E_t ≥ 1/α | H_0) ≤ α.

    Under p = m_0 = 0.5 (the boundary of H_0: μ ≤ μ_0), at α = 0.1 the
    empirical false-rejection rate should be at most α plus a Monte Carlo
    slack of a few percentage points.
    """
    alpha = 0.1
    T = 300
    n_runs = 2000
    sups = _simulate_sup_log_e(n_runs, T, p=0.5, m0=0.5, seed=7)
    rate = float(np.mean(sups >= math.log(1.0 / alpha)))
    assert rate <= alpha + 0.03, f"rate={rate:.4f} exceeds alpha+slack"


def test_type_I_controlled_below_baseline():
    """Under p < m_0 (strict interior of H_0), false rejection should be
    even rarer than at the boundary."""
    alpha = 0.1
    T = 300
    n_runs = 1000
    sups = _simulate_sup_log_e(n_runs, T, p=0.3, m0=0.5, seed=21)
    rate = float(np.mean(sups >= math.log(1.0 / alpha)))
    assert rate <= alpha + 0.03, f"rate={rate:.4f} exceeds alpha+slack under p<m_0"


def test_has_power_under_alternative():
    """Under p = 0.75 and m_0 = 0.5, the one-sided e-process should detect
    the gap reliably by T = 300."""
    alpha = 0.05
    T = 300
    n_runs = 500
    sups = _simulate_sup_log_e(n_runs, T, p=0.75, m0=0.5, seed=13)
    rate = float(np.mean(sups >= math.log(1.0 / alpha)))
    assert rate >= 0.9, f"power={rate:.3f} too low"


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_rejects_out_of_range(bad: float):
    ep = UpwardCapitalEProcess(m0=0.5)
    with pytest.raises(ValueError):
        ep.update(bad)


def test_rejects_invalid_m0():
    with pytest.raises(ValueError):
        UpwardCapitalEProcess(m0=0.0)
    with pytest.raises(ValueError):
        UpwardCapitalEProcess(m0=1.0)
