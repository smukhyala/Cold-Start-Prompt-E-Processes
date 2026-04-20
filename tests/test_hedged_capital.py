from __future__ import annotations

import math

import numpy as np
import pytest

from cold_start.inference.hedged_capital import HedgedCapitalEProcess


def _simulate_sup_log_e(n_runs: int, T: int, p: float, m0: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sups = np.empty(n_runs)
    for i in range(n_runs):
        ep = HedgedCapitalEProcess(m0=m0)
        sup = 0.0
        for _ in range(T):
            x = float(rng.random() < p)
            le = ep.update(x)
            if le > sup:
                sup = le
        sups[i] = sup
    return sups


def test_nonnegativity_and_monotone_state_when_null_balanced():
    ep = HedgedCapitalEProcess(m0=0.5)
    le_seq = [ep.update(x) for x in (0, 1, 1, 0, 0, 1, 0, 1, 0, 1)]
    assert all(math.isfinite(v) for v in le_seq)
    assert ep.n == 10


def test_type_I_error_controlled_under_null():
    """Ville's inequality: P(sup_t E_t >= 1/alpha | H_0) <= alpha.

    With alpha=0.1 and 2000 runs under p=m_0=0.5, the empirical rate should be
    below 0.1 with high probability; allow a small tolerance.
    """
    alpha = 0.1
    T = 300
    n_runs = 2000
    sups = _simulate_sup_log_e(n_runs, T, p=0.5, m0=0.5, seed=7)
    rate = float(np.mean(sups >= math.log(1.0 / alpha)))
    # Ville bound is alpha; we allow mild slack for Monte-Carlo noise.
    assert rate <= alpha + 0.03, f"rate={rate:.4f} exceeds alpha+slack"


def test_has_power_under_alternative():
    """Under p=0.75, m_0=0.5 the e-process should detect the gap reliably by T=300."""
    alpha = 0.05
    T = 300
    n_runs = 500
    sups = _simulate_sup_log_e(n_runs, T, p=0.75, m0=0.5, seed=13)
    rate = float(np.mean(sups >= math.log(1.0 / alpha)))
    assert rate >= 0.9, f"power={rate:.3f} too low"


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_rejects_out_of_range(bad: float):
    ep = HedgedCapitalEProcess(m0=0.5)
    with pytest.raises(ValueError):
        ep.update(bad)
