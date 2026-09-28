"""Amendment 1: G1's rehearsal at a stratified half (30) of the 60 logged Gmail tasks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import rehearsal  # noqa: E402


def test_stratified_half_keeps_every_other_element_from_index_zero():
    rates = tuple(range(60))  # already "sorted ascending"
    half = rehearsal.stratified_half(rates)
    assert half == tuple(range(0, 60, 2))
    assert len(half) == 30


def test_stratified_half_of_task_rates_spans_the_same_range():
    half = rehearsal.stratified_half(rehearsal.TASK_RATES)
    assert len(half) == 30
    assert half[0] == rehearsal.TASK_RATES[0]
    assert half[-1] == rehearsal.TASK_RATES[-2]  # 60 elements, every-other from index 0
    assert min(half) == min(rehearsal.TASK_RATES)
    assert max(half) == max(rehearsal.TASK_RATES)


def test_task_rates_itself_is_unchanged():
    """Amendment 1 must not touch the logged 60 rates -- only what's synthesized from them."""
    assert len(rehearsal.TASK_RATES) == 60
    assert sum(r == 1.0 for r in rehearsal.TASK_RATES) == 13


def test_synthesize_outcomes_defaults_to_all_sixty_tasks():
    from cold_start.growing.reservoirs import BetaReservoir

    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    out = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=5, seed=0)
    assert len(out[out["replicate"] == 0]) == 2 * 10 * 60  # unchanged default: the full logged bank


def test_synthesize_outcomes_honours_a_task_rates_override():
    from cold_start.growing.reservoirs import BetaReservoir

    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    half = rehearsal.stratified_half(rehearsal.TASK_RATES)
    out = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=5, seed=0, task_rates=half)
    assert len(out[out["replicate"] == 0]) == 2 * 10 * 30
    task_ids = set(out["task_id"])
    assert task_ids == {f"t{j:02d}" for j in range(30)}


def test_synthesize_outcomes_task_rates_override_is_deterministic_and_differs_from_the_default():
    from cold_start.growing.reservoirs import BetaReservoir

    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    half = rehearsal.stratified_half(rehearsal.TASK_RATES)
    a = rehearsal.synthesize_outcomes(truth, n_prompts=5, n_replicates=3, seed=1, task_rates=half)
    b = rehearsal.synthesize_outcomes(truth, n_prompts=5, n_replicates=3, seed=1, task_rates=half)
    pd.testing.assert_frame_equal(a, b)
    full = rehearsal.synthesize_outcomes(truth, n_prompts=5, n_replicates=3, seed=1)
    assert len(a[a["replicate"] == 0]) != len(full[full["replicate"] == 0])


class _FakeReservoir:
    def __init__(self, sd=0.1, mean=0.5):
        self._sd, self._mean = sd, mean

    def sd(self):
        return self._sd

    def mean(self):
        return self._mean


def _patch_heavy_pipeline(monkeypatch):
    """Stub out the expensive NPMLE/K*-envelope machinery `run_rehearsal` calls after
    synthesizing outcomes, so the wiring test below stays fast and makes no real fits."""
    from cold_start.growing.empirical_reservoir import EmpiricalReservoir

    fake_reservoirs = {(p, v): _FakeReservoir() for p in ("G", "F") for v in ("npmle", "raw")}
    # `make_emp_cell` needs a real `to_spec()`/`sd()`/`mean()`; only ("F", "npmle") is passed to it.
    fake_reservoirs[("F", "npmle")] = EmpiricalReservoir([0.5], [1.0], label="fake_npmle")
    monkeypatch.setattr(rehearsal.replay, "estimate", lambda outcomes: (fake_reservoirs, {}, None))
    monkeypatch.setattr(rehearsal.replay, "kgrid", lambda cells, workers: None)
    monkeypatch.setattr(rehearsal.describe, "k_star_table",
                        lambda df: pd.DataFrame({"variant": ["truth", "npmle"], "k_star": [24, 32]}))


def test_run_rehearsal_wires_n_tasks_into_synthesize_outcomes(monkeypatch):
    _patch_heavy_pipeline(monkeypatch)
    calls = []
    original = rehearsal.synthesize_outcomes

    def spy(*a, **kw):
        calls.append(kw.get("task_rates"))
        return original(*a, **kw)

    monkeypatch.setattr(rehearsal, "synthesize_outcomes", spy)

    rehearsal.run_rehearsal(seed=0, workers=1, m=5, n_tasks=30)
    assert calls[-1] == rehearsal.stratified_half(rehearsal.TASK_RATES)

    rehearsal.run_rehearsal(seed=0, workers=1, m=5, n_tasks=60)
    assert calls[-1] == rehearsal.TASK_RATES


def test_run_rehearsal_defaults_to_thirty_tasks(monkeypatch):
    _patch_heavy_pipeline(monkeypatch)
    calls = []
    original = rehearsal.synthesize_outcomes
    monkeypatch.setattr(rehearsal, "synthesize_outcomes",
                        lambda *a, **kw: (calls.append(kw.get("task_rates")), original(*a, **kw))[1])
    rehearsal.run_rehearsal(seed=0, workers=1, m=5)
    assert calls[-1] == rehearsal.stratified_half(rehearsal.TASK_RATES)


def test_run_rehearsal_refuses_an_unsupported_n_tasks(monkeypatch):
    _patch_heavy_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="30 or 60"):
        rehearsal.run_rehearsal(seed=0, workers=1, m=5, n_tasks=45)


def test_cli_forwards_n_tasks_and_defaults_to_thirty(monkeypatch, tmp_path):
    captured = {}

    def fake_run_rehearsal(seed=20260926, workers=12, m=1000, n_tasks=30):
        captured["n_tasks"] = n_tasks
        return {"gate": {}}

    fake_gate = SimpleNamespace(passed=True, report=lambda: "ok")
    monkeypatch.setattr(rehearsal, "run_rehearsal", fake_run_rehearsal)
    monkeypatch.setattr(rehearsal, "OUT", tmp_path / "rehearsal.json")
    monkeypatch.setattr(rehearsal.gates, "rehearsal_gate", lambda results: fake_gate)

    assert rehearsal.main(["--workers", "1"]) == 0
    assert captured["n_tasks"] == 30

    assert rehearsal.main(["--workers", "1", "--n-tasks", "60"]) == 0
    assert captured["n_tasks"] == 60
