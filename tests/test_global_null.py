from __future__ import annotations

import math

from cold_start.inference.global_null import GlobalNullEProcess
from cold_start.inference.hedged_capital import HedgedCapitalEProcess


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
