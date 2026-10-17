# Empirical Reservoirs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure two real prompt pools on WebArena Gmail, estimate their noise-corrected reservoirs, and
re-run the growing-bandits registered contrasts on them (Pre-registration 9).

**Architecture:** A collector runs every (prompt, task) pair once on 8 parallel WebArena servers and writes
JSONL. A small estimation library turns the logs into discrete reservoirs (NPMLE primary; raw and
parametric sensitivity), wrapped in a new `EmpiricalReservoir` that the existing simulator, harness and
`run_deployment.py` consume unchanged. A replay driver builds cells, a descriptive module measures the
U-curves and the flatness guard, and `registered_contrast.py` gains a prompt-bootstrap interval.

**Tech Stack:** Python 3.13 (`.venv`), numpy, scipy, pandas/pyarrow, jinja2, PyYAML, anthropic SDK, the
existing webarena-infinity adapter (browser-use + `gpt-5.4-mini`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-empirical-reservoir-design.md`

## Global Constraints

- Agent settings, verbatim from the historical Gmail runs: `llm_provider: openai`, `llm_model: gpt-5.4-mini`,
  `llm_reasoning_effort: low`, `max_agent_steps: 30`, `timeout_s: 180`, `use_vision: false`, headless.
- WebArena app `apps/gmail`, suite `real-tasks`: exactly 60 tasks; every prompt runs every task once.
- Pools: G = 50 grid prompts, F = 50 Claude-written prompts, plus anchor `baseline`; 300 replicate cells;
  pool/queue seed **20260926**; pilot = 5 prompts per pool + anchor = 660 episodes.
- Budget: hard stop at **$260** of summed logged `cost_usd`.
- An agent timeout is a task failure (`success = 0`); harness exceptions and LLM API errors are
  `infra_error`, retried up to 2 times (3 attempts), then `missing` (never scored 0).
- Noise model: σ̂ᵢ² = v̂ / nᵢ, v̂ from replicate pairs; NPMLE on `np.linspace(0, 1, 401)`, tolerance 1e-8.
- Replay: cap = T; M = 1,000 (`emp`), M = 250 and B = 200 (`emp_boot`); primary horizons (50, 100, 200);
  all horizons (50, 100, 200, 500, 1000); seeds from `EMP_SEED_BASE = 400_000_000`.
- Pre-registration 9: MEI = 0.002; a cell is informative iff its K-grid regret range > 0.01; fewer than 2
  informative primary cells → verdict `uninformative`.
- **No constant is tuned on real data.** Cap-T constants come from the corpus only.
- **Pre-registration 9, the frozen pools and the queue are committed before any paid episode.**
- Run Python as `.venv/bin/python`, tests as `.venv/bin/pytest`. Commit only the files a task names
  (`git add <paths>`, never `git add -A`); end every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Bootstrap replicates colliding in the comparator cache** (`prepare_cell_constants` keys on cell name +
  seed): every replicate must have its own env id → pinned in Task 6 (`test_boot_env_ids_and_seeds_unique`).
- **The WebArena task bank changing under us** (the queue assumes 60 specific task ids): the collector must
  refuse to start on a mismatch → pinned in Task 4 (`test_bank_mismatch_refuses_to_start`).
- **Two terminal records for one (arm, task, replicate)** after a crash/resume: loading must fail loudly,
  never average them → pinned in Task 2 (`test_duplicate_terminal_records_raise`).
- **A prompt whose every episode went `missing`**: it must drop out of estimation (n = 0), not divide by
  zero → pinned in Task 2 (`test_prompt_with_no_ok_episodes_is_dropped`).
- **Free-form prompt text with braces, quotes, unicode or newlines** must survive YAML and Jinja byte-for-byte
  (the sha freeze depends on it) → pinned in Task 3 (`test_freeform_text_round_trips_exactly`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/cold_start/growing/empirical_reservoir.py` (new) | `EmpiricalReservoir`: discrete (atoms, weights) reservoir, registered as `"empirical"` |
| `src/cold_start/growing/empirical.py` (new) | Log loading, status constants, noise model, NPMLE, raw and parametric reservoirs |
| `configs/template_freeform.jinja` (new) | Renders `prompt_guidance` alone (pool F) |
| `experiments/growing_bandits/empirical/make_pools.py` (new) | Sample G, generate F with Claude, anchor, replicate list, queue, manifest |
| `src/cold_start/tasks/webarena.py` (modify) | `prompt_for`, `register_prompt`, `task_ids`, `task_by_id` |
| `experiments/growing_bandits/empirical/collect.py` (new) | Parallel resumable collector with budget stop |
| `scripts/watchdog_empirical_pool.py`, `scripts/run_empirical_pool.sh` (new) | Keep the collector alive unattended |
| `experiments/growing_bandits/empirical/replay.py` (new) | Estimation CLI, cell construction, K-grid, point and bootstrap replay |
| `experiments/growing_bandits/deploy/run_deployment.py`, `policy_table.py` (modify) | Register tests `emp`, `emp_boot` |
| `experiments/growing_bandits/empirical/describe.py` (new) | Pool location, K\*, flatness, cross-pool prediction, cap-64 cost, rule gaps |
| `experiments/growing_bandits/deploy/registered_contrast.py` (modify) | `emp_*` registrations and the prompt-bootstrap interval |
| `experiments/growing_bandits/empirical/gates.py`, `rehearsal.py` (new) | G1–G3 gate checks; the synthetic rehearsal |
| `docs/growing_bandits/DEPLOYMENT_PLAN.md`, `DEPLOYMENT_RESULTS.md` (modify) | Pre-registration 9; §13 |

Tests: `tests/test_empirical_reservoir.py`, `tests/test_empirical_estimation.py`,
`tests/test_empirical_pools.py`, `tests/test_empirical_collect.py`, `tests/test_empirical_watchdog.py`,
`tests/test_empirical_replay.py`, `tests/test_empirical_describe.py`,
`tests/test_empirical_registered.py`, `tests/test_empirical_gates.py`.

---

### Task 1: `EmpiricalReservoir`

**Files:**
- Create: `src/cold_start/growing/empirical_reservoir.py`
- Modify: `src/cold_start/cli/_bootstrap.py` (register the family for tests and CLIs)
- Modify: `experiments/growing_bandits/deploy/run_deployment.py:85` (import for workers), `:161` (`FAMILY_OF_TYPE`)
- Test: `tests/test_empirical_reservoir.py`

**Interfaces:**
- Consumes: `cold_start.growing.reservoirs.Reservoir`, `validate_reservoir`, `DegenerateReservoirError`,
  `build_reservoir`; `cold_start.registry.register`.
- Produces: `EmpiricalReservoir(atoms: Sequence[float], weights: Sequence[float], *, label: str = "",
  validate: bool = False)` with attributes `atoms: np.ndarray` (sorted, unique), `weights: np.ndarray`,
  `label: str`, `validation_error: str | None`, methods `mean() -> float`, `sd() -> float`,
  `essential_sup() -> float`, `to_spec() -> {"type": "empirical", "params": {"atoms", "weights", "label"}}`.
  `build_reservoir({"type": "empirical", ...})` works once the module is imported.

Why a separate module and not `reservoirs.py`: `tests/test_deploy_runner.py` pins the simulation-surface
fingerprint of every module reachable from `harness.run_cell`. `run_cell` reaches the reservoir only
through `build_reservoir`'s registry, so a module imported by `run_deployment.py` alone stays off the
surface and every pinned sha stays valid. The reservoir's full content travels in each cell's `env_spec`.

- [ ] **Step 1: Write the failing tests**

```python
"""`EmpiricalReservoir`: an exact discrete reservoir for real prompt pools."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.deploy.harness import CellSpec, run_cell
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER
from cold_start.growing.deploy.rules import make_policy
from cold_start.growing.empirical_reservoir import EmpiricalReservoir
from cold_start.growing.reservoirs import DegenerateReservoirError, build_reservoir
from cold_start.growing.tables import CSTable


def _three() -> EmpiricalReservoir:
    return EmpiricalReservoir([0.2, 0.5, 0.8], [0.25, 0.5, 0.25], label="three")


def test_inverse_cdf_is_the_left_continuous_step_inverse():
    res = _three()
    u = np.array([0.0, 0.25, 0.2500001, 0.75, 0.7500001, 1.0])
    assert res.sample_from_uniforms(u).tolist() == [0.2, 0.2, 0.5, 0.5, 0.8, 0.8]


def test_survival_is_the_mass_strictly_above():
    res = _three()
    x = np.array([-0.1, 0.1, 0.2, 0.49, 0.5, 0.8, 0.9])
    assert np.allclose(res._survival(x), [1.0, 1.0, 0.75, 0.75, 0.25, 0.0, 0.0])
    assert res.tail_prob(0.5) == pytest.approx(0.25)


def test_moments_and_sup():
    res = _three()
    assert res.mean() == pytest.approx(0.5)
    assert res.sd() == pytest.approx(np.sqrt(0.25 * 0.09 + 0.25 * 0.09))
    assert res.essential_sup() == pytest.approx(0.8)


def test_spec_round_trips_through_the_registry():
    res = _three()
    again = build_reservoir(res.to_spec())
    u = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(again.sample_from_uniforms(u), res.sample_from_uniforms(u))
    assert again.label == "three"


def test_duplicate_and_unsorted_atoms_merge():
    res = EmpiricalReservoir([0.5, 0.2, 0.5], [0.25, 0.5, 0.25])
    assert res.atoms.tolist() == [0.2, 0.5]
    assert res.weights.tolist() == [0.5, 0.5]


def test_zero_weight_atoms_are_dropped():
    res = EmpiricalReservoir([0.1, 0.9], [0.0, 1.0])
    assert res.atoms.tolist() == [0.9]


@pytest.mark.parametrize(
    "atoms, weights",
    [([], []), ([0.5], [0.5]), ([0.5, 0.6], [1.2, -0.2]), ([1.5], [1.0]), ([0.5, 0.6], [1.0])],
)
def test_rejects_malformed_input(atoms, weights):
    with pytest.raises(ValueError):
        EmpiricalReservoir(atoms, weights)


def test_a_flat_pool_is_recorded_not_raised():
    res = EmpiricalReservoir([0.6], [1.0])
    assert res.validation_error is not None and "spread" in res.validation_error
    with pytest.raises(DegenerateReservoirError):
        EmpiricalReservoir([0.6], [1.0], validate=True)


def test_runs_through_the_harness():
    res = _three()
    spec = CellSpec(env_id="emp_test", env_spec=res.to_spec(), horizon=20, cap=20, base_seed=1,
                    n_replicates=8)
    table = CSTable.load_or_build(20, 0.05)
    policy = make_policy("fixed_K4", params={"K": 4}, horizon=20, n_replicates=8, table=table)
    out = run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
    regret = out.regret(PRIMARY_RECOMMENDER)
    assert regret.shape == (8,)
    assert np.all(np.isfinite(regret)) and np.all(regret >= -1e-12)


def test_run_deployment_knows_the_family():
    import sys
    from pathlib import Path

    deploy = Path(__file__).resolve().parents[1] / "experiments" / "growing_bandits" / "deploy"
    sys.path.insert(0, str(deploy))
    import run_deployment as rd

    assert rd.family_of(_three().to_spec()) == "E"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_reservoir.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cold_start.growing.empirical_reservoir'`

- [ ] **Step 3: Write the implementation**

`src/cold_start/growing/empirical_reservoir.py`:

```python
"""Empirical reservoirs: a finite, weighted set of arm means estimated from real prompts.

The parametric families in `reservoirs.py` describe where arm means *might* come from.
This one describes where they *did* come from: a discrete distribution (atoms, weights)
estimated from logged WebArena outcomes by `cold_start.growing.empirical`. Everything the
simulator needs -- inverse-CDF sampling for the CRN coupling, the survival function for
the oracle columns -- is exact for a discrete distribution, so no numeric inversion is
involved and the paired-rollout coupling holds as it does for every other family.

A real pool may be flatter than any environment the corpus admitted (`MIN_SPREAD`), and
that is a finding rather than an error, so validation is *recorded* by default instead
of raised: `validation_error` holds the validator's message, or ``None``.

This module lives outside `reservoirs.py` on purpose: `harness.run_cell` reaches it only
through the registry, so it stays off the fingerprinted simulation surface
(`run_deployment.SIM_SURFACE_MODULES`) and no shipped episode's fingerprint moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cold_start.growing.reservoirs import DegenerateReservoirError, Reservoir, validate_reservoir
from cold_start.registry import register

_WEIGHT_TOL = 1e-6


@register("empirical", kind="reservoir")
class EmpiricalReservoir(Reservoir):
    """``mu`` takes the value ``atoms[k]`` with probability ``weights[k]``."""

    family = "empirical"

    def __init__(
        self,
        atoms: Sequence[float],
        weights: Sequence[float],
        *,
        label: str = "",
        validate: bool = False,
    ):
        a = np.asarray(atoms, dtype=float).ravel()
        w = np.asarray(weights, dtype=float).ravel()
        if a.size == 0:
            raise ValueError("an empirical reservoir needs at least one atom")
        if a.size != w.size:
            raise ValueError(f"atoms and weights must be the same length; got {a.size} and {w.size}")
        if not (np.all(np.isfinite(a)) and np.all((a >= 0.0) & (a <= 1.0))):
            raise ValueError("atoms must be finite and lie in [0, 1]")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        total = float(w.sum())
        if abs(total - 1.0) > _WEIGHT_TOL:
            raise ValueError(f"weights must sum to 1; got {total!r}")

        keep = w > 0.0
        a, w = a[keep], w[keep] / w[keep].sum()
        uniq, inverse = np.unique(a, return_inverse=True)
        merged = np.zeros(uniq.size, dtype=float)
        np.add.at(merged, inverse, w)
        self.atoms = uniq
        self.weights = merged
        self._cum = np.cumsum(merged)
        self._cum[-1] = 1.0
        self._cum0 = np.concatenate([[0.0], self._cum])
        self.label = str(label)

        self.validation_error: str | None = None
        try:
            validate_reservoir(self)
        except DegenerateReservoirError as exc:
            if validate:
                raise
            self.validation_error = str(exc)

    def _icdf(self, u: np.ndarray) -> np.ndarray:
        # inf{x : F(x) >= u}: the first atom whose cumulative weight reaches u.
        idx = np.searchsorted(self._cum, np.asarray(u, dtype=float), side="left")
        return self.atoms[np.minimum(idx, self.atoms.size - 1)]

    def _survival(self, x: np.ndarray) -> np.ndarray:
        n_at_or_below = np.searchsorted(self.atoms, np.asarray(x, dtype=float), side="right")
        return np.clip(1.0 - self._cum0[n_at_or_below], 0.0, 1.0)

    def essential_sup(self) -> float:
        return float(self.atoms[-1])

    def mean(self) -> float:
        return float(np.dot(self.atoms, self.weights))

    def sd(self) -> float:
        m = self.mean()
        return float(np.sqrt(np.dot(self.weights, (self.atoms - m) ** 2)))

    def params(self) -> dict[str, Any]:
        return {
            "atoms": [float(x) for x in self.atoms],
            "weights": [float(x) for x in self.weights],
            "label": self.label,
        }
```

In `src/cold_start/cli/_bootstrap.py`, after the task-source imports, add:

```python
# reservoirs registered outside cold_start.growing.reservoirs
from cold_start.growing import empirical_reservoir as _empirical_reservoir  # noqa: F401
```

In `experiments/growing_bandits/deploy/run_deployment.py`, after line 85
(`from cold_start.growing.reservoirs import build_reservoir  # noqa: E402`), add:

```python
import cold_start.growing.empirical_reservoir  # noqa: E402,F401  (registers "empirical" in every worker)
```

and change line 161 to:

```python
FAMILY_OF_TYPE: dict[str, str] = {"beta": "A", "tail": "B", "mixture": "C", "empirical": "E"}
```

- [ ] **Step 4: Run the tests, then the pinned-surface tests**

Run: `.venv/bin/pytest tests/test_empirical_reservoir.py tests/test_deploy_runner.py -v`
Expected: all PASS (in particular `test_sim_surface_covers_everything_reachable_from_run_cell`).

- [ ] **Step 5: Commit**

```bash
git add src/cold_start/growing/empirical_reservoir.py src/cold_start/cli/_bootstrap.py \
  experiments/growing_bandits/deploy/run_deployment.py tests/test_empirical_reservoir.py
git commit -m "growing/empirical: EmpiricalReservoir, an exact discrete reservoir for real prompt pools

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Estimation library (logs, noise model, NPMLE, raw, parametric)

**Files:**
- Create: `src/cold_start/growing/empirical.py`
- Test: `tests/test_empirical_estimation.py`

**Interfaces:**
- Consumes: `EmpiricalReservoir` (Task 1); `BetaReservoir`, `TailReservoir`, `MixtureReservoir`.
- Produces (every later task relies on these names):
  - constants `STATUS_OK = "ok"`, `STATUS_INFRA = "infra_error"`, `STATUS_MISSING = "missing"`,
    `TERMINAL_STATUSES`, `RECORD_COLUMNS`, `GRID` (401 points), `SCHEMA = "empirical_pool/1"`;
  - `load_attempts(paths: Iterable[str | Path]) -> pd.DataFrame` (every line, columns ⊇ `RECORD_COLUMNS`);
  - `terminal_outcomes(attempts: pd.DataFrame) -> pd.DataFrame` (status ∈ terminal; raises on duplicates);
  - `PromptScores(pool, arm_ids, successes, n)` with `.means`;
  - `prompt_scores(outcomes, pool) -> PromptScores` (replicate 0, status ok, arms with n ≥ 1);
  - `within_cell_variance(outcomes, pools=None) -> tuple[float, int]` (v̂, number of pairs);
  - `npmle(xbar, sigma2, grid=GRID, tol=1e-8, max_iter=100_000) -> tuple[np.ndarray, np.ndarray, float]`;
  - `npmle_reservoir(scores, v, label) -> EmpiricalReservoir`;
  - `raw_reservoir(scores, label) -> EmpiricalReservoir`;
  - `fit_parametric(scores, v, label) -> tuple[EmpiricalReservoir, pd.DataFrame]`
    (best-AIC family discretized on `GRID`; the frame lists every family's fit).

- [ ] **Step 1: Write the failing tests**

```python
"""The estimation library: logs -> per-prompt scores -> noise-corrected reservoirs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cold_start.growing import empirical as emp


def _rec(arm, task, success, replicate=0, status="ok", pool="G", attempt=1, cost=0.03):
    return {"schema": emp.SCHEMA, "pool": pool, "arm_id": arm, "task_id": task, "replicate": replicate,
            "attempt": attempt, "status": status, "success": success, "cost_usd": cost}


def _write(tmp_path, records, name="worker_0.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_load_and_terminal_keep_ok_and_missing_only(tmp_path):
    path = _write(tmp_path, [
        _rec("G_00", "t1", None, status="infra_error"),
        _rec("G_00", "t1", 1, attempt=2),
        _rec("G_00", "t2", None, status="missing", attempt=3),
    ])
    attempts = emp.load_attempts([path])
    assert len(attempts) == 3
    out = emp.terminal_outcomes(attempts)
    assert sorted(out["status"]) == ["missing", "ok"]


def test_duplicate_terminal_records_raise(tmp_path):
    path = _write(tmp_path, [_rec("G_00", "t1", 1), _rec("G_00", "t1", 0, attempt=2)])
    with pytest.raises(ValueError, match="duplicate"):
        emp.terminal_outcomes(emp.load_attempts([path]))


def test_empty_logs_give_an_empty_frame(tmp_path):
    path = _write(tmp_path, [])
    frame = emp.load_attempts([path])
    assert list(frame.columns)[: len(emp.RECORD_COLUMNS)] == list(emp.RECORD_COLUMNS)
    assert len(frame) == 0


def test_prompt_scores_use_replicate_zero_and_ok_only():
    out = pd.DataFrame([
        _rec("G_00", "t1", 1), _rec("G_00", "t2", 0), _rec("G_00", "t1", 0, replicate=1),
        _rec("G_01", "t1", 1), _rec("G_01", "t2", None, status="missing"),
        _rec("F_00", "t1", 1, pool="F"),
    ])
    s = emp.prompt_scores(out, "G")
    assert s.arm_ids == ("G_00", "G_01")
    assert s.successes.tolist() == [1.0, 1.0]
    assert s.n.tolist() == [2.0, 1.0]
    assert s.means.tolist() == [0.5, 1.0]


def test_prompt_with_no_ok_episodes_is_dropped():
    out = pd.DataFrame([_rec("G_00", "t1", 1), _rec("G_01", "t1", None, status="missing")])
    assert emp.prompt_scores(out, "G").arm_ids == ("G_00",)


def test_within_cell_variance_from_replicate_pairs():
    out = pd.DataFrame([
        _rec("a", "t1", 1), _rec("a", "t1", 1, replicate=1),
        _rec("a", "t2", 0), _rec("a", "t2", 0, replicate=1),
        _rec("a", "t3", 1), _rec("a", "t3", 0, replicate=1),
        _rec("a", "t4", 0), _rec("a", "t4", 1, replicate=1),
        _rec("a", "t5", 1),  # unpaired: ignored
    ])
    v, n_pairs = emp.within_cell_variance(out)
    assert n_pairs == 4
    assert v == pytest.approx(0.25)


def test_within_cell_variance_needs_pairs():
    with pytest.raises(ValueError, match="replicate"):
        emp.within_cell_variance(pd.DataFrame([_rec("a", "t1", 1)]))


def test_npmle_recovers_a_two_point_mixing_distribution():
    rng = np.random.default_rng(0)
    mu = np.where(rng.random(400) < 0.5, 0.4, 0.7)
    sigma2 = np.full(400, 0.03**2)
    xbar = mu + rng.normal(0.0, 0.03, 400)
    atoms, w, _ = emp.npmle(xbar, sigma2)
    mean = float(np.dot(atoms, w))
    sd = float(np.sqrt(np.dot(w, (atoms - mean) ** 2)))
    assert mean == pytest.approx(0.55, abs=0.01)
    assert sd == pytest.approx(0.15, abs=0.02)
    assert w[np.abs(atoms - 0.4) <= 0.05].sum() >= 0.4
    assert w[np.abs(atoms - 0.7) <= 0.05].sum() >= 0.4


def test_npmle_removes_noise_that_raw_keeps():
    rng = np.random.default_rng(1)
    n = np.full(200, 60.0)
    xbar = 0.6 + rng.normal(0.0, 0.05, 200)
    scores = emp.PromptScores("G", tuple(f"G_{i:02d}" for i in range(200)), xbar * n, n)
    v = 0.05**2 * 60.0
    corrected = emp.npmle_reservoir(scores, v, "G_npmle")
    raw = emp.raw_reservoir(scores, "G_raw")
    assert raw.sd() == pytest.approx(0.05, abs=0.01)
    assert corrected.sd() < 0.02


def test_npmle_increases_the_likelihood_over_uniform_weights():
    rng = np.random.default_rng(2)
    xbar = rng.uniform(0.3, 0.8, 60)
    sigma2 = rng.uniform(0.02, 0.06, 60) ** 2
    atoms, w, ll = emp.npmle(xbar, sigma2)
    lik = emp._likelihood_matrix(xbar, sigma2, atoms)
    uniform_ll = float(np.sum(np.log(lik @ np.full(atoms.size, 1.0 / atoms.size))))
    assert ll > uniform_ll


def test_npmle_rejects_bad_variances():
    with pytest.raises(ValueError):
        emp.npmle(np.array([0.5]), np.array([0.0]))


def test_fit_parametric_recovers_a_beta_pool():
    rng = np.random.default_rng(3)
    mu = rng.beta(8.0, 8.0, 300)
    n = np.full(300, 60.0)
    xbar = mu + rng.normal(0.0, 0.02, 300)
    scores = emp.PromptScores("G", tuple(f"G_{i:03d}" for i in range(300)), xbar * n, n)
    res, fits = emp.fit_parametric(scores, 0.02**2 * 60.0, "G_parametric")
    assert set(fits["family"]) == {"beta", "tail", "beta_mixture"}
    assert res.mean() == pytest.approx(0.5, abs=0.02)
    assert res.sd() == pytest.approx(np.sqrt(64.0 / (256.0 * 17.0)), abs=0.02)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_estimation.py -v`
Expected: FAIL with `ImportError: cannot import name 'empirical'`

- [ ] **Step 3: Write the implementation**

`src/cold_start/growing/empirical.py`:

```python
"""Real-prompt reservoirs from logged WebArena outcomes.

The collector (`experiments/growing_bandits/empirical/collect.py`) writes one JSON line per
attempt. This module turns those lines into reservoirs the simulator can run on:

1. **Scores.** A prompt's score is its success rate over the task bank, replicate 0 only,
   ``ok`` episodes only (``missing`` never counts as a failure).
2. **Noise.** The score is a sum of heterogeneous Bernoullis, one per task, so the binomial
   variance would over-correct: it assumes every task sits at the prompt's mean. The
   within-cell variance ``v`` is measured directly from replicate pairs, and prompt i's
   measurement variance is ``v / n_i``.
3. **Deconvolution.** The distribution of true prompt rates is the nonparametric maximum
   likelihood estimate (Kiefer-Wolfowitz) under ``xbar_i ~ N(mu_i, v / n_i)``, by EM on a
   fixed 401-point grid. No shape is assumed.

The raw reservoir (observed rates, equal weight) and a parametric fit are sensitivity
variants: raw over-states the spread by construction; parametric smooths the tail.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from cold_start.growing.empirical_reservoir import EmpiricalReservoir
from cold_start.growing.reservoirs import BetaReservoir, MixtureReservoir, Reservoir, TailReservoir

SCHEMA = "empirical_pool/1"
STATUS_OK = "ok"
STATUS_INFRA = "infra_error"
STATUS_MISSING = "missing"
TERMINAL_STATUSES: tuple[str, ...] = (STATUS_OK, STATUS_MISSING)
RECORD_COLUMNS: tuple[str, ...] = (
    "schema", "pool", "arm_id", "task_id", "replicate", "attempt", "status", "success", "cost_usd",
)
KEY: list[str] = ["arm_id", "task_id", "replicate"]

GRID = np.linspace(0.0, 1.0, 401)
NPMLE_TOL = 1e-8
NPMLE_MAX_ITER = 100_000
MIN_WEIGHT = 1e-6


# ---- logs -----------------------------------------------------------------------


def load_attempts(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Every attempt line from the collector's JSONL files, in file order."""
    rows: list[dict] = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    for col in RECORD_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.Series(dtype=object)
    extra = [c for c in frame.columns if c not in RECORD_COLUMNS]
    return frame[list(RECORD_COLUMNS) + extra]


def terminal_outcomes(attempts: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, task, replicate) that finished: ``ok`` or ``missing``.

    Two terminal rows for one key mean a resume ran an item twice; averaging them would
    silently weight that cell double, so it is an error.
    """
    out = attempts[attempts["status"].isin(TERMINAL_STATUSES)].copy()
    dup = out.duplicated(KEY, keep=False)
    if dup.any():
        keys = out.loc[dup, KEY].drop_duplicates().to_dict("records")
        raise ValueError(f"duplicate terminal records for {len(keys)} keys, e.g. {keys[:3]}")
    return out.reset_index(drop=True)


# ---- scores and noise -------------------------------------------------------------


@dataclass(frozen=True)
class PromptScores:
    pool: str
    arm_ids: tuple[str, ...]
    successes: np.ndarray
    n: np.ndarray

    @property
    def means(self) -> np.ndarray:
        return self.successes / self.n


def prompt_scores(outcomes: pd.DataFrame, pool: str) -> PromptScores:
    """Per-prompt successes and ``ok`` counts on replicate 0; prompts with no ``ok`` episode drop."""
    sub = outcomes[
        (outcomes["pool"] == pool) & (outcomes["replicate"] == 0) & (outcomes["status"] == STATUS_OK)
    ]
    grouped = sub.assign(success=sub["success"].astype(float)).groupby("arm_id")["success"]
    agg = grouped.agg(["sum", "count"]).sort_index()
    agg = agg[agg["count"] > 0]
    return PromptScores(
        pool=pool,
        arm_ids=tuple(str(a) for a in agg.index),
        successes=agg["sum"].to_numpy(dtype=float),
        n=agg["count"].to_numpy(dtype=float),
    )


def within_cell_variance(outcomes: pd.DataFrame, pools: Iterable[str] | None = None) -> tuple[float, int]:
    """``v = E[(x0 - x1)^2] / 2`` over (arm, task) cells run twice, both ``ok``."""
    ok = outcomes[outcomes["status"] == STATUS_OK]
    if pools is not None:
        ok = ok[ok["pool"].isin(list(pools))]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success",
                          aggfunc="first")
    if 0 not in wide.columns or 1 not in wide.columns:
        raise ValueError("no replicate pairs: within-cell variance needs replicate 0 and 1 of the same cell")
    pairs = wide[[0, 1]].dropna()
    if pairs.empty:
        raise ValueError("no replicate pairs: within-cell variance needs replicate 0 and 1 of the same cell")
    d = pairs[0].to_numpy(dtype=float) - pairs[1].to_numpy(dtype=float)
    return float(np.mean(d**2) / 2.0), int(len(pairs))


# ---- NPMLE --------------------------------------------------------------------------


def _likelihood_matrix(xbar: np.ndarray, sigma2: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return stats.norm.pdf(xbar[:, None], loc=grid[None, :], scale=np.sqrt(sigma2)[:, None])


def npmle(
    xbar: np.ndarray,
    sigma2: np.ndarray,
    grid: np.ndarray = GRID,
    tol: float = NPMLE_TOL,
    max_iter: int = NPMLE_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Kiefer-Wolfowitz NPMLE of the mixing distribution by EM on a fixed grid.

    Returns ``(grid, weights, log_likelihood)``. EM never decreases the likelihood; it stops
    when an iteration gains less than `tol`.
    """
    x = np.asarray(xbar, dtype=float).ravel()
    s2 = np.asarray(sigma2, dtype=float).ravel()
    if x.size == 0 or x.shape != s2.shape:
        raise ValueError("xbar and sigma2 must be non-empty and the same length")
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0.0):
        raise ValueError("every measurement variance must be finite and positive")
    lik = _likelihood_matrix(x, s2, grid)
    w = np.full(grid.size, 1.0 / grid.size)
    prev = -np.inf
    for _ in range(max_iter):
        mix = lik @ w
        ll = float(np.sum(np.log(mix)))
        if ll - prev < tol:
            return grid.copy(), w, ll
        prev = ll
        w = w * (lik.T @ (1.0 / mix)) / x.size
    raise RuntimeError(f"NPMLE did not converge in {max_iter} iterations")


def npmle_reservoir(scores: PromptScores, v: float, label: str) -> EmpiricalReservoir:
    atoms, w, _ = npmle(scores.means, v / scores.n)
    w = np.where(w < MIN_WEIGHT, 0.0, w)
    return EmpiricalReservoir(atoms, w / w.sum(), label=label)


def raw_reservoir(scores: PromptScores, label: str) -> EmpiricalReservoir:
    k = scores.means.size
    return EmpiricalReservoir(np.clip(scores.means, 0.0, 1.0), np.full(k, 1.0 / k), label=label)


# ---- parametric sensitivity -------------------------------------------------------


def grid_masses(res: Reservoir, grid: np.ndarray = GRID) -> np.ndarray:
    """The reservoir's probability of each grid cell (midpoint edges), from `_survival`."""
    mids = (grid[:-1] + grid[1:]) / 2.0
    edges = np.concatenate([[-1e-12], mids, [1.0 + 1e-12]])
    surv = np.asarray(res._survival(edges), dtype=float)
    mass = np.clip(surv[:-1] - surv[1:], 0.0, None)
    total = mass.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("reservoir puts no mass on [0, 1]")
    return mass / total


def _moments(scores: PromptScores, v: float) -> tuple[float, float]:
    m = float(np.clip(np.mean(scores.means), 0.02, 0.98))
    var = float(np.var(scores.means, ddof=1) - np.mean(v / scores.n))
    return m, max(var, 1e-4)


def _beta(theta: np.ndarray) -> Reservoir:
    return BetaReservoir(float(np.exp(theta[0])), float(np.exp(theta[1])), validate=False)


def _tail(theta: np.ndarray) -> Reservoir:
    return TailReservoir(float(np.exp(theta[0])), float(special.expit(theta[1])), float(np.exp(theta[2])),
                         validate=False)


def _beta_mixture(theta: np.ndarray) -> Reservoir:
    comps = [BetaReservoir(float(np.exp(theta[0])), float(np.exp(theta[1])), validate=False),
             BetaReservoir(float(np.exp(theta[2])), float(np.exp(theta[3])), validate=False)]
    w = float(special.expit(theta[4]))
    return MixtureReservoir(comps, [w, 1.0 - w], validate=False)


def _starts(scores: PromptScores, v: float) -> dict[str, np.ndarray]:
    m, var = _moments(scores, v)
    strength = max(m * (1.0 - m) / var - 1.0, 0.5)
    sd = np.sqrt(var)
    top = float(np.clip(np.max(scores.means) + 0.05, 0.05, 0.99))
    width = max(top - float(np.min(scores.means)), 0.05)
    lo_m, hi_m = float(np.clip(m - sd, 0.02, 0.98)), float(np.clip(m + sd, 0.02, 0.98))
    return {
        "beta": np.log([m * strength, (1.0 - m) * strength]),
        "tail": np.array([np.log(2.0), special.logit(top), np.log(1.0 / width**2)]),
        "beta_mixture": np.array([*np.log([lo_m * strength, (1.0 - lo_m) * strength]),
                                  *np.log([hi_m * strength, (1.0 - hi_m) * strength]), 0.0]),
    }


FAMILIES: dict[str, Callable[[np.ndarray], Reservoir]] = {
    "beta": _beta, "tail": _tail, "beta_mixture": _beta_mixture,
}


def fit_parametric(scores: PromptScores, v: float, label: str) -> tuple[EmpiricalReservoir, pd.DataFrame]:
    """Fit each family by maximum marginal likelihood; return the best-AIC fit on `GRID`."""
    lik = _likelihood_matrix(scores.means, v / scores.n, GRID)
    starts = _starts(scores, v)
    rows: list[dict] = []
    best: tuple[float, np.ndarray] | None = None

    for family, build in FAMILIES.items():
        def nll(theta: np.ndarray, build=build) -> float:
            try:
                mass = grid_masses(build(theta))
            except (ValueError, FloatingPointError, ZeroDivisionError):
                return 1e12
            mix = lik @ mass
            if not np.all(np.isfinite(mix)) or np.any(mix <= 0.0):
                return 1e12
            return float(-np.sum(np.log(mix)))

        fit = optimize.minimize(nll, starts[family], method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        k = starts[family].size
        aic = 2.0 * k + 2.0 * float(fit.fun)
        rows.append({"family": family, "n_params": k, "nll": float(fit.fun), "aic": aic,
                     "converged": bool(fit.success), "theta": json.dumps([float(t) for t in fit.x])})
        if best is None or aic < best[0]:
            best = (aic, grid_masses(build(fit.x)))

    table = pd.DataFrame(rows)
    table["selected"] = table["aic"] == table["aic"].min()
    assert best is not None
    return EmpiricalReservoir(GRID, best[1], label=label), table
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_estimation.py -v`
Expected: all PASS. If `test_fit_parametric_recovers_a_beta_pool` fails on the mixture's start (a
degenerate start returns 1e12 everywhere), print `fits` and widen that family's start before touching
the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/cold_start/growing/empirical.py tests/test_empirical_estimation.py
git commit -m "growing/empirical: noise model from replicate pairs, NPMLE, raw and parametric reservoirs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Pool builders (G sample, F generation, anchor, queue, manifest)

**Files:**
- Create: `configs/template_freeform.jinja`
- Create: `experiments/growing_bandits/empirical/make_pools.py`
- Test: `tests/test_empirical_pools.py`

**Interfaces:**
- Consumes: `cold_start.prompts.axes.load_axes`, `AXIS_NAMES`; `cold_start.prompts.template.render_prompt`;
  `cold_start.types.Arm`, `PromptVector`; `cold_start.tasks.webarena._import_webarena`, `_webarena_root`.
- Produces (Tasks 4, 10 rely on these):
  - `PoolArm(pool: str, arm: Arm, template: str, text: str, sha256: str)`;
  - `load_pool(path: Path, axes_path: Path) -> list[PoolArm]` — re-renders and **raises** if any
    rendered sha differs from the recorded one;
  - `QueueItem(index: int, pool: str, arm_id: str, task_id: str, replicate: int, pilot: bool)`;
  - `read_queue(path: Path) -> list[QueueItem]`;
  - `sample_grid_vectors(axes, n, seed) -> list[PromptVector]`;
  - `parse_freeform(text: str, n: int) -> list[str]`;
  - `build_queue(arms_by_pool: dict[str, list[str]], task_ids: list[str], *, n_replicates: int,
    pilot_arms: set[str], seed: int) -> list[QueueItem]`;
  - files under `data/empirical_pool/`: `pool_G.yaml`, `pool_F.yaml`, `pool_anchor.yaml`,
    `f_generation_raw.json`, `queue.jsonl`, `manifest.json`.
- Arm ids: `G_00`..`G_49`, `F_00`..`F_49`, `anchor_baseline` (namespaced, so a sampled grid vector equal
  to `baseline`'s cannot collide with the anchor).

- [ ] **Step 1: Write the failing tests**

```python
"""Pool construction: deterministic, frozen by hash, and exact about the text it froze."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import make_pools as mp  # noqa: E402

from cold_start.prompts.axes import load_axes  # noqa: E402

AXES = ROOT / "configs" / "axes.yaml"


def test_grid_has_2304_points_and_the_sample_is_deterministic():
    axes = load_axes(AXES)
    assert len(mp.all_grid_vectors(axes)) == 2304
    a = mp.sample_grid_vectors(axes, 50, seed=20260926)
    b = mp.sample_grid_vectors(axes, 50, seed=20260926)
    assert a == b and len(set(a)) == 50
    assert a != mp.sample_grid_vectors(axes, 50, seed=1)


def _texts(k, words=60):
    return [f"Prompt {i}: " + " ".join(["careful"] * words) for i in range(k)]


def test_parse_freeform_takes_the_first_n_valid():
    raw = "Here you go:\n" + json.dumps(_texts(3) + ["too short"] + _texts(2)) + "\nthanks"
    assert len(mp.parse_freeform(raw, 4)) == 4
    with pytest.raises(ValueError, match="valid"):
        mp.parse_freeform(json.dumps(_texts(2)), 4)


def test_parse_freeform_drops_duplicates():
    texts = _texts(2)
    assert len(mp.parse_freeform(json.dumps(texts + texts + _texts(1)), 2)) == 2
    with pytest.raises(ValueError):
        mp.parse_freeform(json.dumps(texts + texts), 3)


def test_generate_freeform_calls_the_client_once():
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(_texts(60)))])

    texts, raw = mp.generate_freeform(SimpleNamespace(messages=Messages()), n=50)
    assert len(texts) == 50 and len(calls) == 1
    assert calls[0]["model"] == mp.GENERATOR_MODEL and "temperature" not in calls[0]
    assert json.loads(raw["response_text"]) == _texts(60)


def test_freeform_text_round_trips_exactly(tmp_path):
    text = 'Use {{ braces }} and {% raw %}, "quotes", naïve unicode ✓,\n\ttabs and\nnewlines.  '
    path = tmp_path / "pool_F.yaml"
    mp.write_pool(path, "F", [mp.freeform_arm("F_00", text)], ROOT / "configs" / "template_freeform.jinja",
                  AXES, meta={"seed": 1})
    arms = mp.load_pool(path, AXES)
    assert arms[0].text == text
    assert arms[0].arm.prompt_guidance == text


def test_load_pool_refuses_a_changed_text(tmp_path):
    path = tmp_path / "pool_F.yaml"
    mp.write_pool(path, "F", [mp.freeform_arm("F_00", "Be precise. " * 10)],
                  ROOT / "configs" / "template_freeform.jinja", AXES, meta={})
    path.write_text(path.read_text().replace("Be precise.", "Be sloppy."))
    with pytest.raises(ValueError, match="sha256"):
        mp.load_pool(path, AXES)


def _queue():
    arms = {"G": [f"G_{i:02d}" for i in range(50)], "F": [f"F_{i:02d}" for i in range(50)],
            "anchor": ["anchor_baseline"]}
    tasks = [f"task_{i}" for i in range(60)]
    pilot = set(arms["G"][:5] + arms["F"][:5] + arms["anchor"])
    return mp.build_queue(arms, tasks, n_replicates=300, pilot_arms=pilot, seed=20260926), pilot


def test_queue_counts_order_and_pilot():
    q, pilot = _queue()
    assert len(q) == 6000 + 60 + 300
    assert [item.index for item in q] == list(range(len(q)))
    n_pilot = sum(item.pilot for item in q)
    assert n_pilot == 660
    assert all(item.pilot for item in q[:660]) and not any(item.pilot for item in q[660:])
    assert all(item.arm_id in pilot and item.replicate == 0 for item in q[:660])
    keys = {(i.arm_id, i.task_id, i.replicate) for i in q}
    assert len(keys) == len(q)


def test_replicates_repeat_existing_g_or_f_cells():
    q, _ = _queue()
    main = {(i.arm_id, i.task_id) for i in q if i.replicate == 0}
    reps = [i for i in q if i.replicate == 1]
    assert len(reps) == 300
    assert all((i.arm_id, i.task_id) in main and i.pool in ("G", "F") for i in reps)


def test_queue_is_deterministic_and_round_trips(tmp_path):
    a, _ = _queue()
    b, _ = _queue()
    assert a == b
    path = tmp_path / "queue.jsonl"
    mp.write_queue(path, a)
    assert mp.read_queue(path) == a
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_pools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_pools'`

- [ ] **Step 3: Write the implementation**

`configs/template_freeform.jinja` must render the guidance **byte-for-byte** (the sha freeze and
`test_freeform_text_round_trips_exactly` depend on it), so create it with no trailing newline —
`keep_trailing_newline=True` would otherwise append one:

```bash
printf '%s' '{{ arm.prompt_guidance }}' > configs/template_freeform.jinja
```

`experiments/growing_bandits/empirical/make_pools.py`:

```python
"""Build and freeze the real prompt pools (spec section 3.1) and the episode queue (3.2).

    .venv/bin/python experiments/growing_bandits/empirical/make_pools.py

writes, under ``data/empirical_pool/``: ``pool_G.yaml`` (50 grid prompts), ``pool_F.yaml``
(50 Claude-written prompts), ``pool_anchor.yaml`` (the hand-written ``baseline``),
``f_generation_raw.json`` (the generator's request and full response), ``queue.jsonl``
(6,360 episodes, pilot first) and ``manifest.json`` (sha256 of each file). It refuses to
overwrite an existing pool unless ``--force``: the pools are frozen by Pre-registration 9.

Every pool file records the sha256 of each arm's *rendered* prompt; `load_pool`
re-renders and refuses a mismatch, so the collector can only ever send the frozen text.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.prompts.axes import AXIS_NAMES, AxesSpec, load_axes  # noqa: E402
from cold_start.prompts.template import render_prompt  # noqa: E402
from cold_start.types import Arm, PromptVector  # noqa: E402

log = logging.getLogger("empirical.make_pools")

SEED = 20260926
N_PER_POOL = 50
N_REPLICATES = 300
PILOT_PER_POOL = 5
DATA_DIR = ROOT / "data" / "empirical_pool"
AXES_PATH = ROOT / "configs" / "axes.yaml"
GRID_TEMPLATE = ROOT / "configs" / "template.jinja"
FREEFORM_TEMPLATE = ROOT / "configs" / "template_freeform.jinja"
BASELINE_VECTOR = {"planning": 0, "verification": 0, "agency": 0, "expertise": 1, "format": 0, "goal": 0}
ANCHOR_ARM_ID = "anchor_baseline"

GENERATOR_MODEL = "claude-opus-4-7"
GENERATOR_MAX_TOKENS = 16_000
N_REQUEST = 60
MIN_WORDS, MAX_WORDS = 40, 250
GENERATION_PROMPT = """You are helping build a benchmark of system-prompt instructions for an AI agent that \
operates a web-based email client through a browser: reading, searching, starring, labeling, archiving, \
replying to, forwarding and composing emails.

Write {n} distinct instructions that could be appended to such an agent's system prompt to guide how it \
works. Make them genuinely different from one another: vary the strategy, tone, level of detail, emphasis \
(speed, caution, verification, planning, exploration, precision), structure (prose, bullet rules, \
numbered procedures) and persona. Write each as a real practitioner might, not as a caricature, and not \
deliberately bad.

Constraints for every instruction:
- 40 to 250 words.
- Addressed to the agent in the second person.
- Do not mention any specific email, person, label, or task.
- Plain text only: no Markdown headings, no code fences.

Return only a JSON array of {n} strings."""


# ---- prompts ------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolArm:
    pool: str
    arm: Arm
    template: str
    text: str
    sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def all_grid_vectors(axes: AxesSpec) -> list[PromptVector]:
    ranges = [range(axes[name].max + 1) for name in AXIS_NAMES]
    return [PromptVector(**dict(zip(AXIS_NAMES, combo, strict=True))) for combo in itertools.product(*ranges)]


def sample_grid_vectors(axes: AxesSpec, n: int, seed: int) -> list[PromptVector]:
    grid = all_grid_vectors(axes)
    idx = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [grid[int(i)] for i in idx]


def grid_arm(arm_id: str, vec: PromptVector) -> Arm:
    return Arm(arm_id=arm_id, name=arm_id, vector=vec, prompt_guidance="")


def freeform_arm(arm_id: str, text: str) -> Arm:
    # The vector is a placeholder the free-form template never reads.
    return Arm(arm_id=arm_id, name=arm_id, vector=PromptVector(**BASELINE_VECTOR), prompt_guidance=text)


def render(arm: Arm, template: Path, axes: AxesSpec) -> str:
    return render_prompt(arm.vector, axes, template, prompt_guidance=arm.prompt_guidance,
                         arm_id=arm.arm_id, arm_name=arm.name)


def parse_freeform(text: str, n: int) -> list[str]:
    """The first `n` valid, distinct instructions in the response's JSON array."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in the generator's response")
    items = json.loads(text[start : end + 1])
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item)
        norm = " ".join(s.split())
        if not MIN_WORDS <= len(norm.split()) <= MAX_WORDS or norm in seen:
            continue
        seen.add(norm)
        out.append(s)
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} valid distinct instructions; need {n}")


def generate_freeform(client, n: int = N_PER_POOL) -> tuple[list[str], dict]:
    """One generator call; returns the instructions and the full request/response record."""
    prompt = GENERATION_PROMPT.format(n=N_REQUEST)
    response = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=GENERATOR_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    raw = {"model": GENERATOR_MODEL, "max_tokens": GENERATOR_MAX_TOKENS, "prompt": prompt, "response_text": text}
    return parse_freeform(text, n), raw


# ---- pool files ---------------------------------------------------------------------


def write_pool(path: Path, pool: str, arms: list[Arm], template: Path, axes_path: Path, meta: dict) -> None:
    axes = load_axes(axes_path)
    entries = []
    for arm in arms:
        text = render(arm, template, axes)
        entries.append({
            "arm_id": arm.arm_id,
            "name": arm.name,
            "vector": arm.vector.as_dict(),
            "prompt_guidance": arm.prompt_guidance,
            "prompt_sha256": sha256_text(text),
        })
    doc = {"pool": pool, "template": str(Path(template).relative_to(ROOT)) if Path(template).is_relative_to(ROOT)
           else str(template), "meta": meta, "arms": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=10_000)


def load_pool(path: Path, axes_path: Path = AXES_PATH) -> list[PoolArm]:
    """Every arm of a pool file, re-rendered; raises if a rendered prompt's sha256 moved."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    template = Path(doc["template"])
    if not template.is_absolute():
        template = ROOT / template
    axes = load_axes(axes_path)
    out: list[PoolArm] = []
    for entry in doc["arms"]:
        arm = Arm(arm_id=entry["arm_id"], name=entry["name"], vector=PromptVector(**entry["vector"]),
                  prompt_guidance=entry["prompt_guidance"])
        text = render(arm, template, axes)
        sha = sha256_text(text)
        if sha != entry["prompt_sha256"]:
            raise ValueError(f"{path.name}:{arm.arm_id}: rendered prompt sha256 {sha} != frozen "
                             f"{entry['prompt_sha256']}")
        out.append(PoolArm(pool=doc["pool"], arm=arm, template=str(template), text=text, sha256=sha))
    return out


# ---- queue --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueItem:
    index: int
    pool: str
    arm_id: str
    task_id: str
    replicate: int
    pilot: bool


def build_queue(
    arms_by_pool: dict[str, list[str]],
    task_ids: list[str],
    *,
    n_replicates: int,
    pilot_arms: set[str],
    seed: int,
) -> list[QueueItem]:
    """Every (arm, task) once, plus `n_replicates` repeats of G/F cells; pilot first, each part shuffled."""
    rng = np.random.default_rng(seed)
    main = [(pool, arm, task) for pool in sorted(arms_by_pool) for arm in arms_by_pool[pool] for task in task_ids]
    eligible = [m for m in main if m[0] in ("G", "F")]
    rep_idx = sorted(int(i) for i in rng.choice(len(eligible), size=n_replicates, replace=False))
    items = [(p, a, t, 0) for p, a, t in main] + [(*eligible[i], 1) for i in rep_idx]
    pilot = [it for it in items if it[1] in pilot_arms and it[3] == 0]
    rest = [it for it in items if not (it[1] in pilot_arms and it[3] == 0)]
    ordered = [pilot[int(i)] for i in rng.permutation(len(pilot))] + [rest[int(i)] for i in rng.permutation(len(rest))]
    return [QueueItem(index=k, pool=p, arm_id=a, task_id=t, replicate=r, pilot=k < len(pilot))
            for k, (p, a, t, r) in enumerate(ordered)]


def write_queue(path: Path, queue: list[QueueItem]) -> None:
    with open(path, "w") as fh:
        for item in queue:
            fh.write(json.dumps(asdict(item)) + "\n")


def read_queue(path: Path) -> list[QueueItem]:
    with open(path) as fh:
        return [QueueItem(**json.loads(line)) for line in fh if line.strip()]


def gmail_task_ids() -> list[str]:
    from cold_start.tasks.webarena import _import_webarena, _webarena_root

    _, _, tasks_mod = _import_webarena()
    tasks = tasks_mod.load_tasks(str(_webarena_root() / "apps/gmail"), "real-tasks")
    return [str(t["id"]) for t in tasks]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing pool files")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out: Path = args.out
    pools = {p: out / f"pool_{p}.yaml" for p in ("G", "F", "anchor")}
    if any(p.exists() for p in pools.values()) and not args.force:
        raise SystemExit(f"pool files exist under {out}; they are frozen (pass --force to rebuild)")

    axes = load_axes(AXES_PATH)
    g_arms = [grid_arm(f"G_{i:02d}", v) for i, v in enumerate(sample_grid_vectors(axes, N_PER_POOL, args.seed))]
    write_pool(pools["G"], "G", g_arms, GRID_TEMPLATE, AXES_PATH, meta={"seed": args.seed, "grid_size": 2304})

    from dotenv import load_dotenv
    import anthropic

    load_dotenv(ROOT / ".env")
    texts, raw = generate_freeform(anthropic.Anthropic(), N_PER_POOL)
    (out / "f_generation_raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    f_arms = [freeform_arm(f"F_{i:02d}", t) for i, t in enumerate(texts)]
    write_pool(pools["F"], "F", f_arms, FREEFORM_TEMPLATE, AXES_PATH, meta={"generator": GENERATOR_MODEL})

    anchor = [grid_arm(ANCHOR_ARM_ID, PromptVector(**BASELINE_VECTOR))]
    write_pool(pools["anchor"], "anchor", anchor, GRID_TEMPLATE, AXES_PATH, meta={"historical_arm": "baseline"})

    task_ids = gmail_task_ids()
    if len(task_ids) != 60:
        raise SystemExit(f"expected 60 Gmail real-tasks, found {len(task_ids)}")
    arms_by_pool = {"G": [a.arm_id for a in g_arms], "F": [a.arm_id for a in f_arms], "anchor": [ANCHOR_ARM_ID]}
    pilot = set(arms_by_pool["G"][:PILOT_PER_POOL] + arms_by_pool["F"][:PILOT_PER_POOL] + [ANCHOR_ARM_ID])
    queue = build_queue(arms_by_pool, task_ids, n_replicates=N_REPLICATES, pilot_arms=pilot, seed=args.seed)
    write_queue(out / "queue.jsonl", queue)

    files = [*pools.values(), out / "f_generation_raw.json", out / "queue.jsonl"]
    manifest = {"seed": args.seed, "task_ids": task_ids, "files": {p.name: file_sha256(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d pool files, %d queue items (%d pilot) under %s",
             len(pools), len(queue), sum(q.pilot for q in queue), out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_pools.py tests/test_template_render.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/template_freeform.jinja experiments/growing_bandits/empirical/make_pools.py \
  tests/test_empirical_pools.py
git commit -m "growing/empirical: pool builders -- grid sample, Claude free-form pool, anchor, queue, hash freeze

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Adapter hooks and the collector

**Files:**
- Modify: `src/cold_start/tasks/webarena.py` (add `prompt_for`, `register_prompt`, `task_ids`, `task_by_id`;
  `run_arm` calls `prompt_for`)
- Create: `experiments/growing_bandits/empirical/collect.py`
- Test: `tests/test_empirical_collect.py`

**Interfaces:**
- Consumes: `make_pools.PoolArm`, `load_pool`, `QueueItem`, `read_queue` (Task 3); `empirical.STATUS_*`,
  `SCHEMA`, `load_attempts`, `terminal_outcomes` (Task 2); `cold_start.types.RunResult`, `Task`.
- Produces:
  - adapter methods `prompt_for(arm: Arm) -> str`, `register_prompt(arm_id: str, text: str) -> None`,
    `task_ids() -> list[str]`, `task_by_id(task_id: str) -> Task`;
  - `classify(result: RunResult | None, exc: BaseException | None) -> str`;
  - `WorkerConfig(worker, n_workers, log_dir, budget_usd, max_attempts=3, pilot_only=False, max_steps=30)`;
  - `run_worker(cfg, queue, prompts: dict[str, PoolArm], adapter) -> str` returning `"done"` or `"budget"`;
  - `spent_usd(log_dir: Path) -> float`; `check_bank(adapter, queue) -> None`;
  - log files `logs/empirical_pool/worker_<w>.jsonl`, status file `logs/empirical_pool/STATUS`
    (`running` | `done` | `budget` | `failed`).

- [ ] **Step 1: Write the failing tests**

```python
"""The collector: resumable, partitioned, budget-capped, and honest about infra failures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import collect  # noqa: E402
import make_pools as mp  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.types import RunResult, Task  # noqa: E402

TASKS = ["t0", "t1", "t2"]


class FakeAdapter:
    """Scripted outcomes: ``script[(arm_id, task_id)]`` is a list consumed one call at a time."""

    def __init__(self, script=None, default_cost=0.01, tasks=TASKS):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default_cost = default_cost
        self.tasks = tasks
        self.calls, self.prompts, self.resets = [], {}, 0

    def task_ids(self):
        return list(self.tasks)

    def task_by_id(self, task_id):
        return Task(task_id=task_id, payload={}, metadata={})

    def register_prompt(self, arm_id, text):
        self.prompts[arm_id] = text

    def run_arm(self, arm, task, runner, max_steps):
        self.calls.append((arm.arm_id, task.task_id))
        todo = self.script.get((arm.arm_id, task.task_id))
        out = todo.pop(0) if todo else True
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, RunResult):
            return out
        return RunResult(success=bool(out), reward=float(bool(out)), steps=3, wallclock_s=1.0,
                         trace={"is_done": True, "errors": []}, tokens={"cost_usd": self.default_cost})

    def close(self):
        pass

    def reset(self, seed):
        self.resets += 1


def _prompts():
    arms = [mp.freeform_arm("F_00", "Be precise. " * 10), mp.freeform_arm("F_01", "Be quick. " * 10)]
    return {a.arm_id: mp.PoolArm(pool="F", arm=a, template="x", text=a.prompt_guidance, sha256="s") for a in arms}


def _queue(pilot_first=0):
    items, k = [], 0
    for arm in ("F_00", "F_01"):
        for task in TASKS:
            items.append(mp.QueueItem(index=k, pool="F", arm_id=arm, task_id=task, replicate=0, pilot=k < pilot_first))
            k += 1
    return items


def _cfg(tmp_path, **kw):
    base = dict(worker=0, n_workers=1, log_dir=tmp_path, budget_usd=100.0)
    base.update(kw)
    return collect.WorkerConfig(**base)


def _records(tmp_path):
    return emp.load_attempts(sorted(tmp_path.glob("worker_*.jsonl")))


def test_runs_every_item_once_and_writes_the_schema(tmp_path):
    adapter = FakeAdapter()
    assert collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter) == "done"
    recs = _records(tmp_path)
    assert len(recs) == 6 and set(recs["status"]) == {"ok"}
    assert set(recs["schema"]) == {emp.SCHEMA}
    assert adapter.prompts.keys() == {"F_00", "F_01"}


def test_workers_partition_the_queue(tmp_path):
    q = _queue()
    a, b = FakeAdapter(), FakeAdapter()
    collect.run_worker(_cfg(tmp_path, worker=0, n_workers=2), q, _prompts(), a)
    collect.run_worker(_cfg(tmp_path, worker=1, n_workers=2), q, _prompts(), b)
    assert len(a.calls) == 3 and len(b.calls) == 3
    assert not set(a.calls) & set(b.calls)


def test_resume_skips_finished_items(tmp_path):
    q = _queue()
    collect.run_worker(_cfg(tmp_path), q[:4], _prompts(), FakeAdapter())
    again = FakeAdapter()
    collect.run_worker(_cfg(tmp_path), q, _prompts(), again)
    assert len(again.calls) == 2
    assert len(emp.terminal_outcomes(_records(tmp_path))) == 6


def test_infra_errors_retry_then_go_missing_never_zero(tmp_path):
    boom = [RuntimeError("browser crashed")] * 3
    adapter = FakeAdapter(script={("F_00", "t0"): boom})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)
    recs = _records(tmp_path)
    cell = recs[(recs["arm_id"] == "F_00") & (recs["task_id"] == "t0")]
    assert cell["status"].tolist() == ["infra_error", "infra_error", "missing"]
    assert cell["success"].isna().all()
    assert adapter.resets == 2


def test_an_infra_error_that_recovers_is_scored(tmp_path):
    adapter = FakeAdapter(script={("F_00", "t0"): [RuntimeError("503"), False]})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)
    out = emp.terminal_outcomes(_records(tmp_path))
    row = out[(out["arm_id"] == "F_00") & (out["task_id"] == "t0")].iloc[0]
    assert row["status"] == "ok" and row["success"] == 0 and row["attempt"] == 2


def test_classify():
    ok = RunResult(success=False, reward=0.0, steps=30, wallclock_s=180.0, trace={"timed_out": True})
    assert collect.classify(ok, None) == emp.STATUS_OK
    api = RunResult(success=False, reward=0.0, steps=2, wallclock_s=5.0,
                    trace={"is_done": False, "errors": ["RateLimitError: Error code: 429"]})
    assert collect.classify(api, None) == emp.STATUS_INFRA
    done_with_noise = RunResult(success=True, reward=1.0, steps=5, wallclock_s=5.0,
                                trace={"is_done": True, "errors": ["Error code: 429"]})
    assert collect.classify(done_with_noise, None) == emp.STATUS_OK
    assert collect.classify(None, RuntimeError("x")) == emp.STATUS_INFRA


def test_budget_stop(tmp_path):
    adapter = FakeAdapter(default_cost=0.1)
    status = collect.run_worker(_cfg(tmp_path, budget_usd=0.25), _queue(), _prompts(), adapter)
    assert status == "budget"
    assert len(adapter.calls) == 3
    assert collect.spent_usd(tmp_path) == pytest.approx(0.3)


def test_pilot_only(tmp_path):
    adapter = FakeAdapter()
    collect.run_worker(_cfg(tmp_path, pilot_only=True), _queue(pilot_first=2), _prompts(), adapter)
    assert len(adapter.calls) == 2


def test_bank_mismatch_refuses_to_start():
    with pytest.raises(RuntimeError, match="task bank"):
        collect.check_bank(FakeAdapter(tasks=["t0", "t1"]), _queue())


def test_adapter_prompt_hooks(tmp_path):
    from cold_start.tasks.webarena import WebArenaInfinityAdapter

    adapter = WebArenaInfinityAdapter(artifacts_dir=str(tmp_path))
    arm = mp.grid_arm("G_00", mp.PromptVector(**mp.BASELINE_VECTOR))
    rendered = adapter.prompt_for(arm)
    assert "agent" in rendered
    adapter.register_prompt("G_00", "PINNED")
    assert adapter.prompt_for(arm) == "PINNED"
    adapter._tasks = [{"id": "task_e1", "instruction": "x", "verify": "v"}]
    adapter._web_app_abs = "/tmp"
    assert adapter.task_ids() == ["task_e1"]
    assert adapter.task_by_id("task_e1").task_id == "task_e1"
    with pytest.raises(KeyError):
        adapter.task_by_id("nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_collect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect'`

- [ ] **Step 3: Adapter hooks**

In `src/cold_start/tasks/webarena.py`, replace the three prompt lines in `run_arm`

```python
        extension = self._prompt_cache.get(arm.arm_id)
        if extension is None:
            extension = render_arm_prompt(arm, self._axes, self._template_path)
            self._prompt_cache[arm.arm_id] = extension
```

with

```python
        extension = self.prompt_for(arm)
```

and add these methods after `sample_task`:

```python
    def prompt_for(self, arm: Arm) -> str:
        """The system-prompt extension sent for `arm`: pinned text if registered, else rendered."""
        extension = self._prompt_cache.get(arm.arm_id)
        if extension is None:
            extension = render_arm_prompt(arm, self._axes, self._template_path)
            self._prompt_cache[arm.arm_id] = extension
        return extension

    def register_prompt(self, arm_id: str, text: str) -> None:
        """Pin the exact extension for `arm_id`, bypassing this adapter's template.

        The empirical-pool collector renders each pool with its own template and freezes
        the text by sha256; pinning it here means the agent receives those bytes and no
        others, whichever template this adapter was built with.
        """
        self._prompt_cache[arm_id] = text

    def task_ids(self) -> list[str]:
        if self._tasks is None:
            raise RuntimeError("reset() must be called before task_ids()")
        return [str(raw["id"]) for raw in self._tasks]

    def task_by_id(self, task_id: str) -> Task:
        if self._tasks is None:
            raise RuntimeError("reset() must be called before task_by_id()")
        for idx, raw in enumerate(self._tasks):
            if str(raw["id"]) == task_id:
                return self.sample_task(idx + 1)
        raise KeyError(f"task {task_id!r} is not in the {self._task_suite} bank")
```

- [ ] **Step 4: The collector**

`experiments/growing_bandits/empirical/collect.py`:

```python
"""Collect the real prompt pools' outcomes on WebArena Gmail (spec section 3.3).

    .venv/bin/python experiments/growing_bandits/empirical/collect.py --workers 8 [--pilot]

Worker w owns queue items with ``index % n_workers == w``, runs them in queue order on its
own WebArena server (port 8001 + w), and appends one JSON line per *attempt* to
``logs/empirical_pool/worker_<w>.jsonl``. Resuming is automatic: items with a terminal
record (``ok`` or ``missing``) are skipped, and an item's earlier ``infra_error`` attempts
count toward its three.

An agent timeout is a task failure, scored 0, as every historical Gmail run scored it. A
harness exception, or an LLM API error in the agent's trace while the agent was not done,
is ``infra_error``: the worker restarts its server and browser and retries, and the third
failure writes ``missing``, which estimation excludes and never scores 0.

Every worker re-reads the summed ``cost_usd`` of all workers before each attempt and stops
at the budget, so the overshoot is bounded by the episodes already in flight (8 x ~$0.05).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import multiprocessing as mp_
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_pools  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.types import RunResult  # noqa: E402

log = logging.getLogger("empirical.collect")

LOG_DIR = ROOT / "logs" / "empirical_pool"
BASE_PORT = 8001
BUDGET_USD = 260.0
MAX_ATTEMPTS = 3
MAX_STEPS = 30
N_BANK_TASKS = 60
AGENT = {
    "web_app": "apps/gmail",
    "task_suite": "real-tasks",
    "use_vision": False,
    "headless": True,
    "timeout_s": 180,
    "llm_provider": "openai",
    "llm_model": "gpt-5.4-mini",
    "llm_reasoning_effort": "low",
}
API_ERROR_MARKERS: tuple[str, ...] = (
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailable",
    "Error code: 429", "Error code: 500", "Error code: 502", "Error code: 503", "Error code: 529", "overloaded",
)


@dataclass(frozen=True)
class WorkerConfig:
    worker: int
    n_workers: int
    log_dir: Path
    budget_usd: float
    max_attempts: int = MAX_ATTEMPTS
    pilot_only: bool = False
    max_steps: int = MAX_STEPS


def classify(result: RunResult | None, exc: BaseException | None) -> str:
    if exc is not None or result is None:
        return emp.STATUS_INFRA
    trace = result.trace or {}
    if trace.get("timed_out"):
        return emp.STATUS_OK
    errors = " ".join(str(e) for e in trace.get("errors") or [])
    if not result.success and not trace.get("is_done", False) and any(m in errors for m in API_ERROR_MARKERS):
        return emp.STATUS_INFRA
    return emp.STATUS_OK


def _log_files(log_dir: Path) -> list[Path]:
    return sorted(Path(log_dir).glob("worker_*.jsonl"))


def spent_usd(log_dir: Path) -> float:
    attempts = emp.load_attempts(_log_files(log_dir))
    if attempts.empty:
        return 0.0
    return float(attempts["cost_usd"].fillna(0.0).astype(float).sum())


def _progress(log_dir: Path) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str, int], int]]:
    attempts = emp.load_attempts(_log_files(log_dir))
    done = {(r.arm_id, r.task_id, int(r.replicate))
            for r in attempts.itertuples() if r.status in emp.TERMINAL_STATUSES}
    tries: dict[tuple[str, str, int], int] = {}
    for r in attempts.itertuples():
        key = (r.arm_id, r.task_id, int(r.replicate))
        tries[key] = max(tries.get(key, 0), int(r.attempt))
    return done, tries


def check_bank(adapter, queue: list[make_pools.QueueItem]) -> None:
    bank = set(adapter.task_ids())
    needed = {item.task_id for item in queue}
    if not needed <= bank:
        raise RuntimeError(f"task bank mismatch: {sorted(needed - bank)[:5]} are not in the adapter's bank")


def _record(item, attempt: int, status: str, result: RunResult | None, exc: BaseException | None,
            cfg: WorkerConfig, prompt: make_pools.PoolArm) -> dict:
    trace = (result.trace if result is not None else {}) or {}
    tokens = (result.tokens if result is not None else {}) or {}
    return {
        "schema": emp.SCHEMA,
        "pool": item.pool,
        "arm_id": item.arm_id,
        "task_id": item.task_id,
        "replicate": item.replicate,
        "attempt": attempt,
        "status": status,
        "success": None if status != emp.STATUS_OK or result is None else int(bool(result.success)),
        "cost_usd": float(tokens.get("cost_usd", 0.0) or 0.0),
        "steps": None if result is None else int(result.steps),
        "wallclock_s": None if result is None else float(result.wallclock_s),
        "timed_out": bool(trace.get("timed_out", False)),
        "errors": [str(e)[:300] for e in (trace.get("errors") or [])[:3]],
        "exception": None if exc is None else repr(exc)[:500],
        "tokens": tokens,
        "queue_index": item.index,
        "pilot": item.pilot,
        "worker": cfg.worker,
        "port": BASE_PORT + cfg.worker,
        "prompt_sha256": prompt.sha256,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def _append(path: Path, record: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _recover(adapter) -> None:
    try:
        adapter.close()
    finally:
        adapter.reset(seed=0)


def run_worker(cfg: WorkerConfig, queue: list[make_pools.QueueItem], prompts: dict[str, make_pools.PoolArm],
               adapter) -> str:
    """Run this worker's share of `queue`; returns ``"done"`` or ``"budget"``."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.log_dir / f"worker_{cfg.worker}.jsonl"
    for arm_id, prompt in prompts.items():
        adapter.register_prompt(arm_id, prompt.text)
    done, tries = _progress(cfg.log_dir)
    mine = [q for q in queue if q.index % cfg.n_workers == cfg.worker and (q.pilot or not cfg.pilot_only)]
    for item in mine:
        key = (item.arm_id, item.task_id, item.replicate)
        if key in done:
            continue
        attempt = tries.get(key, 0)
        while True:
            if spent_usd(cfg.log_dir) >= cfg.budget_usd:
                return "budget"
            attempt += 1
            prompt = prompts[item.arm_id]
            result, exc = None, None
            try:
                task = adapter.task_by_id(item.task_id)
                result = adapter.run_arm(prompt.arm, task, None, cfg.max_steps)
            except Exception as err:  # noqa: BLE001 -- any harness failure is infra, by definition
                exc = err
            status = classify(result, exc)
            if status == emp.STATUS_INFRA and attempt >= cfg.max_attempts:
                status = emp.STATUS_MISSING
            _append(path, _record(item, attempt, status, result, exc, cfg, prompt))
            if status != emp.STATUS_INFRA:
                break
            log.warning("worker %d: %s/%s attempt %d infra error; recovering", cfg.worker, item.arm_id,
                        item.task_id, attempt)
            _recover(adapter)
    return "done"


def load_prompts(data_dir: Path) -> dict[str, make_pools.PoolArm]:
    out: dict[str, make_pools.PoolArm] = {}
    for name in ("G", "F", "anchor"):
        for arm in make_pools.load_pool(data_dir / f"pool_{name}.yaml"):
            if arm.arm.arm_id in out:
                raise RuntimeError(f"duplicate arm id {arm.arm.arm_id}")
            out[arm.arm.arm_id] = arm
    return out


def _worker_main(worker: int, n_workers: int, data_dir: str, log_dir: str, budget: float, pilot: bool) -> str:
    from dotenv import load_dotenv

    import cold_start.cli._bootstrap  # noqa: F401
    from cold_start.tasks.webarena import WebArenaInfinityAdapter

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{worker} %(levelname)s %(message)s")
    data, logs = Path(data_dir), Path(log_dir)
    queue = make_pools.read_queue(data / "queue.jsonl")
    prompts = load_prompts(data)
    adapter = WebArenaInfinityAdapter(port=BASE_PORT + worker, artifacts_dir=str(logs / "artifacts" / f"w{worker}"),
                                      axes_path=str(ROOT / "configs" / "axes.yaml"),
                                      template_path=str(ROOT / "configs" / "template.jinja"), **AGENT)
    adapter.reset(seed=0)
    try:
        if len(adapter.task_ids()) != N_BANK_TASKS:
            raise RuntimeError(f"task bank has {len(adapter.task_ids())} tasks; expected {N_BANK_TASKS}")
        check_bank(adapter, queue)
        cfg = WorkerConfig(worker=worker, n_workers=n_workers, log_dir=logs, budget_usd=budget, pilot_only=pilot)
        return run_worker(cfg, queue, prompts, adapter)
    finally:
        adapter.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data", type=Path, default=make_pools.DATA_DIR)
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--budget", type=float, default=BUDGET_USD)
    ap.add_argument("--pilot", action="store_true", help="run only the pilot items")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.log_dir / "STATUS"
    status_path.write_text("running\n")
    load_prompts(args.data)  # fail fast on a moved hash, before any server starts
    ctx = mp_.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = [pool.apply_async(_worker_main, (w, args.workers, str(args.data), str(args.log_dir),
                                                   args.budget, args.pilot)) for w in range(args.workers)]
        outcomes = []
        for w, r in enumerate(results):
            try:
                outcomes.append(r.get())
            except Exception as err:  # noqa: BLE001
                log.error("worker %d failed: %r", w, err)
                outcomes.append("failed")
    final = "budget" if "budget" in outcomes else ("failed" if "failed" in outcomes else "done")
    status_path.write_text(final + "\n")
    log.info("collector finished: %s (spent $%.2f)", final, spent_usd(args.log_dir))
    return {"done": 0, "budget": 3, "failed": 1}[final]


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_collect.py tests/test_webarena_stub.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cold_start/tasks/webarena.py experiments/growing_bandits/empirical/collect.py \
  tests/test_empirical_collect.py
git commit -m "growing/empirical: resumable budget-capped collector; adapter prompt pinning and task lookup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Watchdog and launcher

**Files:**
- Create: `scripts/watchdog_empirical_pool.py`
- Create: `scripts/run_empirical_pool.sh`
- Test: `tests/test_empirical_watchdog.py`

**Interfaces:**
- Consumes: `logs/empirical_pool/STATUS` and `worker_*.jsonl` (Task 4), `collect.py` CLI.
- Produces: `decide(status: str | None, alive: bool, minutes_since_progress: float) -> str` returning one of
  `"finished"`, `"wait"`, `"relaunch"`, `"kill_and_relaunch"`; the launcher `scripts/run_empirical_pool.sh
  [--pilot]`.

- [ ] **Step 1: Write the failing tests**

```python
"""The watchdog's decision rule, isolated from processes and clocks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import watchdog_empirical_pool as wd  # noqa: E402


@pytest.mark.parametrize("status, alive, minutes, expected", [
    ("done", False, 0.0, "finished"),
    ("budget", False, 0.0, "finished"),
    ("running", True, 5.0, "wait"),
    ("running", True, wd.STALL_MINUTES + 1, "kill_and_relaunch"),
    ("running", False, 1.0, "relaunch"),
    ("failed", False, 1.0, "relaunch"),
    (None, False, 0.0, "relaunch"),
])
def test_decide(status, alive, minutes, expected):
    assert wd.decide(status, alive, minutes) == expected


def test_line_count_counts_every_worker(tmp_path):
    (tmp_path / "worker_0.jsonl").write_text("{}\n{}\n")
    (tmp_path / "worker_1.jsonl").write_text("{}\n")
    assert wd.line_count(tmp_path) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_watchdog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'watchdog_empirical_pool'`

- [ ] **Step 3: Write the implementation**

`scripts/watchdog_empirical_pool.py`:

```python
"""Keep the empirical-pool collector alive unattended.

    .venv/bin/python scripts/watchdog_empirical_pool.py [--pilot]

Every POLL_SECONDS: if STATUS says done/budget, exit. If the collector process is gone,
relaunch it (it resumes by itself). If it is alive but no log line has been written for
STALL_MINUTES, kill and relaunch it. Gives up after MAX_RELAUNCHES and writes
WATCHDOG_GAVE_UP beside the logs.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "empirical_pool"
COLLECT = ROOT / "experiments" / "growing_bandits" / "empirical" / "collect.py"
POLL_SECONDS = 120
STALL_MINUTES = 30.0
MAX_RELAUNCHES = 20


def decide(status: str | None, alive: bool, minutes_since_progress: float) -> str:
    if status in ("done", "budget"):
        return "finished"
    if alive and minutes_since_progress > STALL_MINUTES:
        return "kill_and_relaunch"
    if alive:
        return "wait"
    return "relaunch"


def line_count(log_dir: Path) -> int:
    total = 0
    for path in sorted(Path(log_dir).glob("worker_*.jsonl")):
        with open(path) as fh:
            total += sum(1 for _ in fh)
    return total


def _status(log_dir: Path) -> str | None:
    path = log_dir / "STATUS"
    return path.read_text().strip() if path.exists() else None


def _launch(extra: list[str], log_dir: Path) -> subprocess.Popen:
    out = open(log_dir / "collect.out", "a")  # noqa: SIM115 -- lives as long as the child
    return subprocess.Popen([sys.executable, str(COLLECT), *extra], stdout=out, stderr=subprocess.STDOUT,
                            cwd=ROOT, start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    extra = ["--workers", str(args.workers)] + (["--pilot"] if args.pilot else [])
    proc = _launch(extra, LOG_DIR)
    relaunches, last_lines, last_progress = 0, line_count(LOG_DIR), time.time()
    while True:
        time.sleep(POLL_SECONDS)
        lines = line_count(LOG_DIR)
        if lines != last_lines:
            last_lines, last_progress = lines, time.time()
        action = decide(_status(LOG_DIR), proc.poll() is None, (time.time() - last_progress) / 60.0)
        if action == "finished":
            print(f"collector finished: {_status(LOG_DIR)} ({lines} lines)", flush=True)
            return 0
        if action == "wait":
            continue
        if relaunches >= MAX_RELAUNCHES:
            (LOG_DIR / "WATCHDOG_GAVE_UP").write_text(f"{relaunches} relaunches\n")
            return 1
        if action == "kill_and_relaunch" and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        relaunches += 1
        print(f"relaunch {relaunches}: {action} at {lines} lines", flush=True)
        proc, last_progress = _launch(extra, LOG_DIR), time.time()


if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/run_empirical_pool.sh`:

```bash
#!/usr/bin/env bash
# Launch the empirical-pool collector under its watchdog, detached and awake.
#   scripts/run_empirical_pool.sh --pilot   # the 660-episode pilot (gate G2)
#   scripts/run_empirical_pool.sh           # everything left in the queue (resumes the pilot)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/empirical_pool
nohup caffeinate -dimsu .venv/bin/python scripts/watchdog_empirical_pool.py "$@" \
  >> logs/empirical_pool/watchdog.out 2>&1 &
echo "watchdog pid $! -- tail -f logs/empirical_pool/watchdog.out logs/empirical_pool/collect.out"
```

Then `chmod +x scripts/run_empirical_pool.sh`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_watchdog.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/watchdog_empirical_pool.py scripts/run_empirical_pool.sh tests/test_empirical_watchdog.py
git commit -m "growing/empirical: watchdog and detached launcher for the collector

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Replay — estimation CLI, cells, K-grid, point and bootstrap runs

**Files:**
- Create: `experiments/growing_bandits/empirical/replay.py`
- Modify: `experiments/growing_bandits/deploy/run_deployment.py` (`TESTS`, `DEFAULT_REPLICATES`,
  `seeds_may_repeat`, `_cell_grid`)
- Modify: `experiments/growing_bandits/deploy/policy_table.py` (`TEST_POLICIES`)
- Test: `tests/test_empirical_replay.py`

**Interfaces:**
- Consumes: Task 2's library; `EmpiricalReservoir`; `run_deployment.main(argv, *, cells=...)`;
  `k_star_envelope.Item`, `k_star_envelope.run_item`, `k_star_envelope.DEFAULT_K_GRID`; `CellSpec`.
- Produces:
  - constants `EMP_SEED_BASE = 400_000_000`, `EMP_CELL_STRIDE = 1_000`, `POOLS = ("G", "F")`,
    `VARIANTS = ("npmle", "raw", "parametric")`, `PRIMARY_HORIZONS = (50, 100, 200)`,
    `ALL_HORIZONS = (50, 100, 200, 500, 1000)`, `EMP_REPLICATES = 1000`, `BOOT_REPLICATES = 250`,
    `N_BOOT = 200`, `CONTRAST_POLICIES = ("p3_star", "fixed_K_star", "level_star", "phi_k4")`;
  - `env_id(pool, variant, boot=None) -> str`; `seed_for(pool, horizon, boot=None) -> int`;
  - `make_emp_cell(pool, variant, horizon, reservoir, n_replicates, boot=None) -> CellSpec`;
  - `noise_model(outcomes, *, per_pool=None, seed=0) -> dict` with keys `v` (`{"G": float, "F": float}`),
    `pooled`, `per_pool`, `se_diff`, `n_pairs`;
  - `estimate(outcomes) -> tuple[dict[tuple[str, str], EmpiricalReservoir], dict, pd.DataFrame]`
    (reservoirs by (pool, variant), the noise model, the parametric fit table);
  - `bootstrap_reservoirs(outcomes, noise, *, n_boot, seed) -> list[tuple[int, str, EmpiricalReservoir]]`;
  - `kgrid(cells, *, workers) -> pd.DataFrame` (columns of `k_star_envelope.run_item` plus `pool`, `variant`);
  - files `data/empirical_pool/reservoirs/<pool>_<variant>.json`, `noise.json`, `parametric_fits.csv`,
    `results/growing_bandits/deploy/tables/emp_kgrid.csv`, episodes under
    `results/growing_bandits/deploy/episodes/{emp,emp_boot}/`.
- CLI: `replay.py estimate | point | kgrid | boot [--workers N]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Replay cells: unique ids, disjoint seeds, CRN across variants, and the runner's plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402

RES = EmpiricalReservoir([0.3, 0.6, 0.9], [0.5, 0.3, 0.2], label="t")


def test_env_ids():
    assert replay.env_id("G", "npmle") == "emp_G_npmle"
    assert replay.env_id("F", "npmle", boot=7) == "emp_F_npmle_b007"


def test_variants_share_seeds_and_everything_else_differs():
    seeds = {(p, T): replay.seed_for(p, T) for p in replay.POOLS for T in replay.ALL_HORIZONS}
    assert len(set(seeds.values())) == len(seeds)
    cells = [replay.make_emp_cell(p, v, T, RES, 10) for p in replay.POOLS for v in replay.VARIANTS
             for T in replay.ALL_HORIZONS]
    assert len({rd.cell_name(c) for c in cells}) == len(cells)
    by_pool_T = {}
    for c in cells:
        by_pool_T.setdefault((c.env_id.split("_")[1], c.horizon), set()).add(c.base_seed)
    assert all(len(s) == 1 for s in by_pool_T.values())


def test_boot_env_ids_and_seeds_unique():
    cells = [replay.make_emp_cell(p, "npmle", T, RES, 10, boot=b) for b in range(replay.N_BOOT)
             for p in replay.POOLS for T in replay.PRIMARY_HORIZONS]
    assert len({rd.cell_name(c) for c in cells}) == len(cells)
    assert len({c.base_seed for c in cells}) == len(cells)
    point = {replay.seed_for(p, T) for p in replay.POOLS for T in replay.ALL_HORIZONS}
    assert not point & {c.base_seed for c in cells}


def test_seeds_are_disjoint_from_the_study():
    import cells as study_cells

    seeds = [replay.seed_for(p, T, b) for p in replay.POOLS for T in replay.ALL_HORIZONS
             for b in (None, 0, replay.N_BOOT - 1)]
    assert min(seeds) >= replay.EMP_SEED_BASE
    study_cells.assert_seed_disjointness(seeds)


def test_cells_are_uncapped():
    c = replay.make_emp_cell("G", "npmle", 200, RES, 10)
    assert c.cap == 200 and c.horizon == 200 and c.env_spec["type"] == "empirical"


def test_the_runner_knows_the_tests():
    assert "emp" in rd.TESTS and "emp_boot" in rd.TESTS
    assert rd.seeds_may_repeat("emp")
    with pytest.raises(RuntimeError, match="replay.py"):
        rd.build_cells("emp", n_replicates=None, cells_mod=rd.import_cells_module())


def _outcomes(rng, n_arms=30, n_tasks=60, n_reps=60):
    rows = []
    for pool in ("G", "F"):
        mus = rng.uniform(0.4, 0.8, n_arms)
        for i, mu in enumerate(mus):
            for t in range(n_tasks):
                rows.append({"pool": pool, "arm_id": f"{pool}_{i:02d}", "task_id": f"t{t}", "replicate": 0,
                             "attempt": 1, "status": "ok", "success": int(rng.random() < mu)})
        for k in range(n_reps):
            i, t = int(rng.integers(n_arms)), int(rng.integers(n_tasks))
            rows.append({"pool": pool, "arm_id": f"{pool}_{i:02d}", "task_id": f"t{t}", "replicate": 1,
                         "attempt": 1, "status": "ok", "success": int(rng.random() < mus[i])})
    return pd.DataFrame(rows).drop_duplicates(["arm_id", "task_id", "replicate"])


def test_estimate_builds_every_variant():
    outcomes = _outcomes(np.random.default_rng(0))
    reservoirs, noise, fits = replay.estimate(outcomes)
    assert set(reservoirs) == {(p, v) for p in replay.POOLS for v in replay.VARIANTS}
    assert set(noise["v"]) == {"G", "F"} and noise["n_pairs"] > 0
    assert fits["selected"].sum() == 2


def test_bootstrap_resamples_prompts_and_labels_replicates():
    outcomes = _outcomes(np.random.default_rng(1))
    _, noise, _ = replay.estimate(outcomes)
    boots = replay.bootstrap_reservoirs(outcomes, noise, n_boot=3, seed=5)
    assert [(b, p) for b, p, _ in boots] == [(b, p) for b in range(3) for p in replay.POOLS]
    assert boots[0][2].label == "G_npmle_b000"
    again = replay.bootstrap_reservoirs(outcomes, noise, n_boot=3, seed=5)
    assert all(np.array_equal(a[2].weights, b[2].weights) for a, b in zip(boots, again, strict=True))


def test_kgrid_runs_on_empirical_cells():
    cells = [replay.make_emp_cell("G", "npmle", 20, RES, 16)]
    frame = replay.kgrid(cells, workers=1, k_grid=(2, 4, 8))
    assert sorted(frame["K"]) == [2, 4, 8]
    assert set(frame["pool"]) == {"G"} and set(frame["variant"]) == {"npmle"}
    assert np.all(np.isfinite(frame["regret"]))


def test_point_replay_writes_paired_episodes(tmp_path):
    cell = replay.make_emp_cell("G", "npmle", 20, RES, 8)
    rd.main(["--test", "emp", "--workers", "1", "--policies", "always_search", "--out-dir", str(tmp_path),
             "--skip-summary"], cells=[cell])
    path = tmp_path / "episodes" / "emp" / rd.cell_name(cell) / "always_search.parquet"
    frame = pd.read_parquet(path)
    assert len(frame) == 8 and int(frame["base_seed"].iloc[0]) == cell.base_seed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'replay'`

- [ ] **Step 3: Register the tests in the runner**

In `experiments/growing_bandits/deploy/run_deployment.py`:

```python
TESTS: tuple[str, ...] = ("A", "B", "C", "D", "robust", "cap", "smoke", "capmatch", "capp", "capc",
                          "emp", "emp_boot")
```

add to `DEFAULT_REPLICATES`:

```python
    "emp": 1000,
    "emp_boot": 250,
```

replace `seeds_may_repeat`:

```python
def seeds_may_repeat(test: str) -> bool:
    """Whether cells of `test` may share a base_seed: the paired cap sweep, and the empirical
    replay, whose raw / npmle / parametric variants of one (pool, T) are CRN-paired by design."""
    return test in ("capp", "emp")
```

and in `_cell_grid`, before the final `else`:

```python
    elif test in ("emp", "emp_boot"):
        raise RuntimeError(
            f"--test {test} cells are built from the frozen real-prompt pools by "
            "experiments/growing_bandits/empirical/replay.py, never from cells.py"
        )
```

In `experiments/growing_bandits/deploy/policy_table.py`, add to `TEST_POLICIES`:

```python
    # Pre-registration 9: the real prompt pools, uncapped; cap-T constants selected on the corpus only.
    "emp": ("always_search", "p3_star", "fixed_K_star", "level_star", "phi_k4"),
    "emp_boot": ("p3_star", "fixed_K_star", "level_star", "phi_k4"),
```

- [ ] **Step 4: Write `replay.py`**

`experiments/growing_bandits/empirical/replay.py`:

```python
"""Replay the growing-bandit policies on the real prompt pools (spec sections 4-5).

    .venv/bin/python experiments/growing_bandits/empirical/replay.py estimate
    .venv/bin/python experiments/growing_bandits/empirical/replay.py point --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py boot  --workers 12

``estimate`` reads the collector's logs and writes the six reservoirs (G/F x npmle/raw/
parametric), the noise model and the parametric fit table under
``data/empirical_pool/reservoirs/``. ``point`` deploys `TEST_POLICIES["emp"]` on every
(pool, variant, T) at M = 1000; ``kgrid`` runs fixed-K over the K-grid on the same cells;
``boot`` re-estimates the NPMLE on B = 200 prompt resamples and deploys the four contrast
policies at the primary horizons, M = 250.

Seeds: every (pool, T) has one seed shared by its three variants (CRN across variants);
every bootstrap replicate has its own seeds and its own env id, so the per-cell
comparator cache (`run_deployment.prepare_cell_constants`) can never mix replicates.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import k_star_envelope as kse  # noqa: E402
import run_deployment as rd  # noqa: E402

import cold_start.growing.empirical_reservoir  # noqa: E402,F401  (registers "empirical" in workers)
from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

log = logging.getLogger("empirical.replay")

EMP_SEED_BASE = 400_000_000
EMP_CELL_STRIDE = 1_000
POOLS: tuple[str, ...] = ("G", "F")
VARIANTS: tuple[str, ...] = ("npmle", "raw", "parametric")
PRIMARY_HORIZONS: tuple[int, ...] = (50, 100, 200)
ALL_HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)
EMP_REPLICATES = 1000
BOOT_REPLICATES = 250
N_BOOT = 200
BOOT_SEED = 20260926
CONTRAST_POLICIES: tuple[str, ...] = ("p3_star", "fixed_K_star", "level_star", "phi_k4")
NOISE_BOOT = 1000

DATA_DIR = ROOT / "data" / "empirical_pool"
LOG_DIR = ROOT / "logs" / "empirical_pool"
RES_DIR = DATA_DIR / "reservoirs"


def env_id(pool: str, variant: str, boot: int | None = None) -> str:
    return f"emp_{pool}_{variant}" + ("" if boot is None else f"_b{boot:03d}")


def seed_for(pool: str, horizon: int, boot: int | None = None) -> int:
    b = 0 if boot is None else boot + 1
    index = b * 100 + POOLS.index(pool) * 10 + ALL_HORIZONS.index(int(horizon))
    return EMP_SEED_BASE + EMP_CELL_STRIDE * index


def make_emp_cell(pool: str, variant: str, horizon: int, reservoir: EmpiricalReservoir, n_replicates: int,
                  boot: int | None = None) -> CellSpec:
    return CellSpec(env_id=env_id(pool, variant, boot), env_spec=reservoir.to_spec(), horizon=int(horizon),
                    cap=int(horizon), base_seed=seed_for(pool, horizon, boot), n_replicates=int(n_replicates))


# ---- estimation ---------------------------------------------------------------------


def _pair_sq_diffs(outcomes: pd.DataFrame, pool: str) -> np.ndarray:
    ok = outcomes[(outcomes["status"] == emp.STATUS_OK) & (outcomes["pool"] == pool)]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success",
                          aggfunc="first")[[0, 1]].dropna()
    return (wide[0].to_numpy(float) - wide[1].to_numpy(float)) ** 2 / 2.0


def noise_model(outcomes: pd.DataFrame, *, per_pool: bool | None = None, seed: int = 0) -> dict:
    """v per pool; pooled unless the pools differ by more than the bootstrap SE of their difference.

    With `per_pool` given (the prompt bootstrap reuses the full-sample decision) the SE is not
    needed and is not computed.
    """
    pooled, n_pairs = emp.within_cell_variance(outcomes, POOLS)
    by_pool = {p: emp.within_cell_variance(outcomes, [p])[0] for p in POOLS}
    se_diff = float("nan")
    if per_pool is None:
        sq = {p: _pair_sq_diffs(outcomes, p) for p in POOLS}
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(NOISE_BOOT):
            v_b = [float(sq[p][rng.integers(0, sq[p].size, sq[p].size)].mean()) for p in POOLS]
            diffs.append(v_b[0] - v_b[1])
        se_diff = float(np.std(diffs, ddof=1))
        per_pool = abs(by_pool["G"] - by_pool["F"]) > se_diff
    v = dict(by_pool) if per_pool else {p: pooled for p in POOLS}
    return {"v": v, "pooled": pooled, "by_pool": by_pool, "per_pool": bool(per_pool), "se_diff": se_diff,
            "n_pairs": n_pairs}


def estimate(outcomes: pd.DataFrame) -> tuple[dict[tuple[str, str], EmpiricalReservoir], dict, pd.DataFrame]:
    noise = noise_model(outcomes)
    reservoirs: dict[tuple[str, str], EmpiricalReservoir] = {}
    fit_tables = []
    for pool in POOLS:
        scores = emp.prompt_scores(outcomes, pool)
        v = noise["v"][pool]
        reservoirs[(pool, "npmle")] = emp.npmle_reservoir(scores, v, f"{pool}_npmle")
        reservoirs[(pool, "raw")] = emp.raw_reservoir(scores, f"{pool}_raw")
        res, fits = emp.fit_parametric(scores, v, f"{pool}_parametric")
        reservoirs[(pool, "parametric")] = res
        fit_tables.append(fits.assign(pool=pool))
    return reservoirs, noise, pd.concat(fit_tables, ignore_index=True)


def _resample_pool(outcomes: pd.DataFrame, pool: str, rng: np.random.Generator) -> pd.DataFrame:
    sub = outcomes[outcomes["pool"] == pool]
    arms = sorted(sub["arm_id"].unique())
    picks = rng.integers(0, len(arms), len(arms))
    parts = [sub[sub["arm_id"] == arms[int(i)]].assign(arm_id=f"{arms[int(i)]}#{k}") for k, i in enumerate(picks)]
    return pd.concat(parts, ignore_index=True)


def bootstrap_reservoirs(outcomes: pd.DataFrame, noise: dict, *, n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> list[tuple[int, str, EmpiricalReservoir]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, str, EmpiricalReservoir]] = []
    for b in range(n_boot):
        resampled = pd.concat([_resample_pool(outcomes, p, rng) for p in POOLS], ignore_index=True)
        nb = noise_model(resampled, per_pool=noise["per_pool"], seed=b)
        for pool in POOLS:
            scores = emp.prompt_scores(resampled, pool)
            out.append((b, pool, emp.npmle_reservoir(scores, nb["v"][pool], f"{pool}_npmle_b{b:03d}")))
    return out


def save_reservoir(res: EmpiricalReservoir, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.to_spec()))


def load_reservoir(path: Path) -> EmpiricalReservoir:
    return build_reservoir(json.loads(path.read_text()))


# ---- runs -----------------------------------------------------------------------------


def _kgrid_item(item: kse.Item) -> dict:
    return kse.run_item(item)


def kgrid(cells: list[CellSpec], *, workers: int, k_grid: tuple[int, ...] = kse.DEFAULT_K_GRID) -> pd.DataFrame:
    items = [kse.Item(split="emp", spec=c, K=int(K)) for c in cells for K in k_grid if int(K) <= c.horizon]
    if workers <= 1:
        rows = [_kgrid_item(i) for i in items]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_kgrid_item, items, chunksize=1)
    frame = pd.DataFrame(rows)
    parts = frame["env_id"].str.split("_", expand=True)
    return frame.assign(pool=parts[1], variant=parts[2])


def point_cells() -> list[CellSpec]:
    return [make_emp_cell(p, v, T, load_reservoir(RES_DIR / f"{p}_{v}.json"), EMP_REPLICATES)
            for p in POOLS for v in VARIANTS for T in ALL_HORIZONS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["estimate", "point", "kgrid", "boot"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.stage == "estimate":
        outcomes = emp.terminal_outcomes(emp.load_attempts(sorted(LOG_DIR.glob("worker_*.jsonl"))))
        reservoirs, noise, fits = estimate(outcomes)
        for (pool, variant), res in reservoirs.items():
            save_reservoir(res, RES_DIR / f"{pool}_{variant}.json")
        (RES_DIR / "noise.json").write_text(json.dumps(noise, indent=2))
        fits.to_csv(RES_DIR / "parametric_fits.csv", index=False)
        for (pool, variant), res in reservoirs.items():
            log.info("%s %-10s mean %.4f sd %.4f atoms %d %s", pool, variant, res.mean(), res.sd(),
                     res.atoms.size, res.validation_error or "")
    elif args.stage == "point":
        rd.main(["--test", "emp", "--workers", str(args.workers), "--out-dir", str(args.out_dir)],
                cells=point_cells())
    elif args.stage == "kgrid":
        frame = kgrid(point_cells(), workers=args.workers)
        out = args.out_dir / "tables" / "emp_kgrid.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(frame))
    else:
        outcomes = emp.terminal_outcomes(emp.load_attempts(sorted(LOG_DIR.glob("worker_*.jsonl"))))
        noise = json.loads((RES_DIR / "noise.json").read_text())
        cells = []
        for b, pool, res in bootstrap_reservoirs(outcomes, noise):
            save_reservoir(res, RES_DIR / "boot" / f"{pool}_npmle_b{b:03d}.json")
            cells.extend(make_emp_cell(pool, "npmle", T, res, BOOT_REPLICATES, boot=b) for T in PRIMARY_HORIZONS)
        rd.main(["--test", "emp_boot", "--workers", str(args.workers), "--out-dir", str(args.out_dir),
                 "--skip-summary"], cells=cells)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_replay.py tests/test_deploy_runner.py tests/test_deploy_cells.py -v`
Expected: all PASS. `test_point_replay_writes_paired_episodes` exercises `run_deployment.main` end to end;
if it fails on a missing `baseline_params.json` under `tmp_path`, that is expected behaviour only for
policies that require it — `always_search` does not, so read the traceback before changing anything.

- [ ] **Step 6: Commit**

```bash
git add experiments/growing_bandits/empirical/replay.py experiments/growing_bandits/deploy/run_deployment.py \
  experiments/growing_bandits/deploy/policy_table.py tests/test_empirical_replay.py
git commit -m "growing/empirical: replay driver -- estimate, emp/emp_boot cells, K-grid, bootstrap reservoirs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Descriptive analysis and the flatness guard

**Files:**
- Create: `experiments/growing_bandits/empirical/describe.py`
- Test: `tests/test_empirical_describe.py`

**Interfaces:**
- Consumes: `emp_kgrid.csv` (Task 6), reservoirs (Task 6), `cells.ALL_ENVS` (the corpus),
  episodes under `episodes/emp/`.
- Produces:
  - `FLATNESS_THRESHOLD = 0.01` (5 × MEI);
  - `pool_summary(res) -> dict` (`level`, `sd`, `q99_minus_mean`, `spread`, `validation_error`);
  - `k_star_table(kgrid) -> pd.DataFrame` (`env_id`, `pool`, `variant`, `horizon`, `k_star`,
    `regret_at_k_star`, `regret_max`, `regret_range`, `informative`);
  - `flatness(kgrid) -> pd.DataFrame` — the rows of `k_star_table` Task 8 reads, written to
    `tables/emp_flatness.csv`;
  - `cross_pool_prediction(levels: dict[str, float], kstar: pd.DataFrame) -> pd.DataFrame`;
  - `cap64_cost(kgrid) -> pd.DataFrame`;
  - `rule_gaps(out_dir, kstar) -> pd.DataFrame`;
  - `locate(pool_rows, corpus_rows) -> pd.DataFrame`;
  - CLI writing `tables/emp_{pool_location,kstar,flatness,cross_pool,cap64,rule_gaps}.csv`.

- [ ] **Step 1: Write the failing tests**

```python
"""Descriptive reads of the real pools, and the flatness guard Pre-registration 9 turns on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import describe  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402


def _kgrid():
    rows = []
    # G at T=50: flat (range 0.004); F at T=50: a U with range 0.05 and minimum at K=16.
    for K, g, f in [(4, 0.101, 0.15), (8, 0.100, 0.12), (16, 0.102, 0.10), (32, 0.104, 0.13)]:
        rows.append({"env_id": "emp_G_npmle", "pool": "G", "variant": "npmle", "horizon": 50, "K": K, "regret": g})
        rows.append({"env_id": "emp_F_npmle", "pool": "F", "variant": "npmle", "horizon": 50, "K": K, "regret": f})
    for T, K64, Kbest in [(500, 0.09, 0.07), (1000, 0.10, 0.06)]:
        for K, r in [(64, K64), (256, Kbest)]:
            rows.append({"env_id": "emp_F_npmle", "pool": "F", "variant": "npmle", "horizon": T, "K": K, "regret": r})
    return pd.DataFrame(rows)


def test_k_star_and_flatness():
    table = describe.k_star_table(_kgrid())
    g = table[(table["pool"] == "G") & (table["horizon"] == 50)].iloc[0]
    f = table[(table["pool"] == "F") & (table["horizon"] == 50)].iloc[0]
    assert g["k_star"] == 8 and g["regret_range"] == pytest.approx(0.004) and not g["informative"]
    assert f["k_star"] == 16 and f["regret_range"] == pytest.approx(0.05) and f["informative"]
    flat = describe.flatness(_kgrid())
    assert set(flat.columns) >= {"env_id", "pool", "variant", "horizon", "regret_range", "informative"}


def test_ties_go_to_the_smaller_k():
    kg = pd.DataFrame([{"env_id": "emp_G_npmle", "pool": "G", "variant": "npmle", "horizon": 50, "K": K,
                        "regret": 0.1} for K in (4, 8)])
    assert describe.k_star_table(kg).iloc[0]["k_star"] == 4


def test_cap64_cost_is_labelled_extrapolation():
    cost = describe.cap64_cost(_kgrid())
    row = cost[cost["horizon"] == 1000].iloc[0]
    assert row["cost"] == pytest.approx(0.04) and row["label"] == "extrapolation"


def test_cross_pool_prediction():
    kstar = pd.DataFrame([
        {"pool": "G", "variant": "npmle", "horizon": 50, "k_star": 16},
        {"pool": "F", "variant": "npmle", "horizon": 50, "k_star": 32},
    ])
    out = describe.cross_pool_prediction({"G": 0.7, "F": 0.5}, kstar).iloc[0]
    assert out["low_pool"] == "F" and bool(out["sign_agrees"])
    assert out["ratio_observed"] == pytest.approx(2.0)
    assert out["ratio_predicted"] == pytest.approx(np.exp(4 * 0.2))


def test_pool_summary():
    s = describe.pool_summary(EmpiricalReservoir([0.4, 0.6], [0.5, 0.5]))
    assert s["level"] == pytest.approx(0.5) and s["sd"] == pytest.approx(0.1)
    assert s["q99_minus_mean"] == pytest.approx(0.1)


def test_locate_ranks_against_the_corpus():
    pools = pd.DataFrame([{"env_id": "emp_G_npmle", "level": 0.5, "sd": 0.03, "q99_minus_mean": 0.05}])
    corpus = pd.DataFrame([{"env_id": f"c{i}", "level": 0.1 * i, "sd": 0.05 * i, "q99_minus_mean": 0.1 * i}
                           for i in range(1, 9)])
    out = describe.locate(pools, corpus).iloc[0]
    assert out["sd_pct_below"] == pytest.approx(0.0)
    assert out["level_pct_below"] == pytest.approx(4 / 8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_describe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'describe'`

- [ ] **Step 3: Write the implementation**

`experiments/growing_bandits/empirical/describe.py`:

```python
"""Descriptive reads of the real prompt pools (Pre-registration 9, item 4) and the flatness guard.

    .venv/bin/python experiments/growing_bandits/empirical/describe.py

reads ``tables/emp_kgrid.csv``, the reservoirs and the ``emp`` episodes, and writes
``tables/emp_{pool_location,kstar,flatness,cross_pool,cap64,rule_gaps}.csv``.

K* here is an in-sample argmin on the pool's own reservoir: a ceiling, never a deployable
policy. A cell is *informative* iff fixed-K regret moves by more than 5 x MEI = 0.01 over
the K-grid; Pre-registration 9 evaluates its contrasts on informative primary cells only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.reservoirs import Reservoir, build_reservoir  # noqa: E402

log = logging.getLogger("empirical.describe")

MEI = 0.002
FLATNESS_THRESHOLD = 5 * MEI
LEVEL_RULE_B = 4.0
STATS: tuple[str, ...] = ("level", "sd", "q99_minus_mean")


def pool_summary(res: Reservoir) -> dict:
    m = float(res.mean())
    q01, q99 = res.quantile(0.01), res.quantile(0.99)
    u = (np.arange(20_001) + 0.5) / 20_001
    draws = res.sample_from_uniforms(u)
    return {"level": m, "sd": float(np.std(draws)), "q99_minus_mean": float(q99 - m), "spread": float(q99 - q01),
            "validation_error": getattr(res, "validation_error", None)}


def k_star_table(kgrid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, pool, variant, T), sub in kgrid.groupby(["env_id", "pool", "variant", "horizon"]):
        sub = sub.sort_values("K")
        best = sub.iloc[int(np.argmin(sub["regret"].to_numpy()))]
        rng_ = float(sub["regret"].max() - sub["regret"].min())
        rows.append({"env_id": env, "pool": pool, "variant": variant, "horizon": int(T), "k_star": int(best["K"]),
                     "regret_at_k_star": float(best["regret"]), "regret_max": float(sub["regret"].max()),
                     "regret_range": rng_, "informative": rng_ > FLATNESS_THRESHOLD})
    return pd.DataFrame(rows)


def flatness(kgrid: pd.DataFrame) -> pd.DataFrame:
    return k_star_table(kgrid)[["env_id", "pool", "variant", "horizon", "regret_range", "informative"]]


def cross_pool_prediction(levels: dict[str, float], kstar: pd.DataFrame) -> pd.DataFrame:
    low, high = sorted(levels, key=lambda p: levels[p])
    rows = []
    prim = kstar[kstar["variant"] == "npmle"]
    for T in sorted(prim["horizon"].unique()):
        k = {r.pool: int(r.k_star) for r in prim[prim["horizon"] == T].itertuples()}
        if low not in k or high not in k:
            continue
        rows.append({"horizon": int(T), "low_pool": low, "high_pool": high, "k_star_low": k[low],
                     "k_star_high": k[high], "sign_agrees": k[low] >= k[high],
                     "ratio_observed": k[low] / k[high],
                     "ratio_predicted": float(np.exp(LEVEL_RULE_B * (levels[high] - levels[low])))})
    return pd.DataFrame(rows)


def cap64_cost(kgrid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, pool, variant, T), sub in kgrid[kgrid["horizon"].isin([500, 1000])].groupby(
            ["env_id", "pool", "variant", "horizon"]):
        at64 = sub[sub["K"] == 64]
        if at64.empty:
            continue
        rows.append({"env_id": env, "pool": pool, "variant": variant, "horizon": int(T),
                     "cost": float(at64["regret"].iloc[0] - sub["regret"].min()), "label": "extrapolation"})
    return pd.DataFrame(rows)


def rule_gaps(out_dir: Path, kstar: pd.DataFrame) -> pd.DataFrame:
    col = f"regret_{PRIMARY_RECOMMENDER}"
    rows = []
    for r in kstar.itertuples():
        cell = out_dir / "episodes" / "emp" / f"{r.env_id}_T{r.horizon}_cap{r.horizon}"
        for path in sorted(cell.glob("*.parquet")):
            regret = float(pd.read_parquet(path, columns=[col])[col].mean())
            rows.append({"env_id": r.env_id, "pool": r.pool, "variant": r.variant, "horizon": r.horizon,
                         "policy": path.stem, "regret": regret, "gap_to_k_star": regret - r.regret_at_k_star})
    return pd.DataFrame(rows)


def locate(pool_rows: pd.DataFrame, corpus_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in pool_rows.to_dict("records"):
        row = dict(r)
        for stat in STATS:
            row[f"{stat}_pct_below"] = float(np.mean(corpus_rows[stat].to_numpy() < r[stat]))
            row[f"{stat}_corpus_min"] = float(corpus_rows[stat].min())
            row[f"{stat}_corpus_max"] = float(corpus_rows[stat].max())
        out.append(row)
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tables = args.out_dir / "tables"
    kgrid = pd.read_csv(tables / "emp_kgrid.csv")

    import cells

    pools = []
    for p in replay.POOLS:
        for v in replay.VARIANTS:
            res = replay.load_reservoir(replay.RES_DIR / f"{p}_{v}.json")
            pools.append({"env_id": replay.env_id(p, v), "pool": p, "variant": v, **pool_summary(res)})
    pools = pd.DataFrame(pools)
    corpus = pd.DataFrame([{"env_id": e, **pool_summary(build_reservoir(spec))} for e, spec in cells.ALL_ENVS.items()])
    kstar = k_star_table(kgrid)
    levels = {r.pool: r.level for r in pools[pools["variant"] == "npmle"].itertuples()}

    locate(pools, corpus).to_csv(tables / "emp_pool_location.csv", index=False)
    kstar.to_csv(tables / "emp_kstar.csv", index=False)
    flatness(kgrid).to_csv(tables / "emp_flatness.csv", index=False)
    cross_pool_prediction(levels, kstar).to_csv(tables / "emp_cross_pool.csv", index=False)
    cap64_cost(kgrid).to_csv(tables / "emp_cap64.csv", index=False)
    rule_gaps(args.out_dir, kstar).to_csv(tables / "emp_rule_gaps.csv", index=False)
    prim = kstar[(kstar["variant"] == "npmle") & kstar["horizon"].isin(replay.PRIMARY_HORIZONS)]
    log.info("informative primary cells: %d of %d", int(prim["informative"].sum()), len(prim))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_describe.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/growing_bandits/empirical/describe.py tests/test_empirical_describe.py
git commit -m "growing/empirical: descriptive reads -- pool location, K*, flatness guard, cross-pool, cap-64

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Pre-registration 9 contrasts with the prompt-bootstrap interval

**Files:**
- Modify: `experiments/growing_bandits/deploy/registered_contrast.py`
- Test: `tests/test_empirical_registered.py`

**Interfaces:**
- Consumes: `_diffs`, `verdict_by_rule`, `ALL_HORIZONS` (existing); `tables/emp_flatness.csv` (Task 7);
  episodes of tests `emp` and `emp_boot` (Task 6).
- Produces: `REGISTRATIONS["emp_primary" | "emp_level" | "emp_phi"]` (each with `"interval":
  "prompt_bootstrap"`), `FLATNESS_MIN_INFORMATIVE = 2`, `prompt_bootstrap_contrast(policy, reference, *,
  horizons, mei, rule, out_dir, flatness_path=None) -> pd.DataFrame` with columns `BOOT_COLUMNS`;
  verdicts `supported` / `refuted` / `inconclusive` / `uninformative`; `main()` dispatches on `"interval"`.

- [ ] **Step 1: Write the failing tests**

```python
"""Pre-registration 9's decision rule: the prompt-bootstrap interval behind the flatness guard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "deploy"))

import registered_contrast as rc  # noqa: E402

COL = "regret_posterior_mean_shrunk"


def _cell(root, test, env, T, seed, deltas, n=20):
    base = np.full(n, 0.1)
    for policy, shift in deltas.items():
        path = root / "episodes" / test / f"{env}_T{T}_cap{T}" / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"env_id": env, "horizon": T, "cap": T, "base_seed": seed, "episode": np.arange(n),
                      COL: base + shift}).to_parquet(path, index=False)


def _tree(root, point_delta, boot_deltas, informative, extra_cell_delta=None):
    flat = []
    for pool in ("G", "F"):
        for T in (50, 100, 200):
            env = f"emp_{pool}_npmle"
            inf = (pool, T) in informative
            flat.append({"env_id": env, "pool": pool, "variant": "npmle", "horizon": T,
                         "regret_range": 0.05 if inf else 0.001, "informative": inf})
            d = point_delta if inf or extra_cell_delta is None else extra_cell_delta
            _cell(root, "emp", env, T, 1, {"p3_star": d, "fixed_K_star": 0.0})
            for b, bd in enumerate(boot_deltas):
                _cell(root, "emp_boot", f"{env}_b{b:03d}", T, 100 + b, {"p3_star": bd, "fixed_K_star": 0.0})
    (root / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(root / "tables" / "emp_flatness.csv", index=False)


def _run(root):
    return rc.prompt_bootstrap_contrast("p3_star", "fixed_K_star", horizons=(50, 100, 200), mei=0.002,
                                        rule="noninferiority", out_dir=root).iloc[0]


def test_registrations_are_written_down():
    for name in ("emp_primary", "emp_level", "emp_phi"):
        reg = rc.REGISTRATIONS[name]
        assert reg["test"] == "emp" and reg["horizons"] == (50, 100, 200) and reg["mei"] == 0.002
        assert reg["interval"] == "prompt_bootstrap"
    assert rc.REGISTRATIONS["emp_primary"]["rule"] == "noninferiority"
    assert rc.REGISTRATIONS["emp_level"]["rule"] == "not_better"


def test_uninformative_when_fewer_than_two_cells_move(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 200)})
    row = _run(tmp_path)
    assert row["verdict"] == "uninformative" and row["n_informative"] == 1


def test_supported_when_the_bootstrap_interval_sits_below_mei(tmp_path):
    boots = list(np.linspace(-0.001, 0.001, 40))
    _tree(tmp_path, 0.0, boots, informative={("F", 100), ("F", 200)})
    row = _run(tmp_path)
    assert row["verdict"] == "supported"
    assert row["lo"] == pytest.approx(np.percentile(boots, 2.5))
    assert row["hi"] == pytest.approx(np.percentile(boots, 97.5))
    assert row["n_boot"] == 40


def test_refuted_when_the_interval_sits_above_mei(tmp_path):
    _tree(tmp_path, 0.01, list(np.linspace(0.008, 0.012, 40)), informative={("G", 50), ("F", 50)})
    assert _run(tmp_path)["verdict"] == "refuted"


def test_only_informative_cells_enter_the_point_estimate(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 10, informative={("F", 100), ("F", 200)}, extra_cell_delta=0.5)
    assert _run(tmp_path)["delta"] == pytest.approx(0.0)


def test_a_missing_bootstrap_cell_is_an_error(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 3, informative={("F", 100), ("F", 200)})
    victim = tmp_path / "episodes" / "emp_boot" / "emp_F_npmle_b001_T200_cap200"
    for f in victim.iterdir():
        f.unlink()
    victim.rmdir()
    with pytest.raises(ValueError, match="b001"):
        _run(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_registered.py -v`
Expected: FAIL with `KeyError: 'emp_primary'`

- [ ] **Step 3: Write the implementation**

In `experiments/growing_bandits/deploy/registered_contrast.py`, add `import re` to the imports, then append
to `REGISTRATIONS` (inside the dict, after `capc_phi`):

```python
    # Pre-registration 9: the real prompt pools, uncapped, informative primary cells only, on the
    # PROMPT-BOOTSTRAP interval (B = 200 NPMLE re-estimates; `prompt_bootstrap_contrast`).
    "emp_primary": {"policy": "p3_star", "reference": "fixed_K_star", "test": "emp",
                    "horizons": (50, 100, 200), "mei": 0.002, "out": "emp_primary.csv", "rule": "noninferiority",
                    "interval": "prompt_bootstrap"},
    "emp_level": {"policy": "level_star", "reference": "p3_star", "test": "emp",
                  "horizons": (50, 100, 200), "mei": 0.002, "out": "emp_level.csv", "rule": "not_better",
                  "interval": "prompt_bootstrap"},
    "emp_phi": {"policy": "phi_k4", "reference": "p3_star", "test": "emp",
                "horizons": (50, 100, 200), "mei": 0.002, "out": "emp_phi.csv", "rule": "not_better",
                "interval": "prompt_bootstrap"},
```

After `registered_contrast(...)`, add:

```python
#: Pre-registration 9's flatness guard: below this many informative primary cells, every
#: contrast's verdict is "uninformative" -- K barely matters on the real pools.
FLATNESS_MIN_INFORMATIVE = 2
BOOT_TEST = "emp_boot"
_BOOT_ENV = re.compile(r"^(?P<base>emp_[A-Za-z]+_npmle)_b(?P<b>\d{3})$")
BOOT_COLUMNS: tuple[str, ...] = (
    "row", "policy", "reference", "test", "horizon", "delta", "lo", "hi", "n_informative", "informative_cells",
    "n_boot", "mei", "rule", "verdict", "as_registered",
)


def _informative_cells(flatness_path: Path, horizons: tuple[int, ...]) -> set[tuple[str, int]]:
    flat = pd.read_csv(flatness_path)
    flat = flat[(flat["variant"] == "npmle") & flat["horizon"].isin(horizons)]
    return {(str(r.env_id), int(r.horizon)) for r in flat.itertuples() if bool(r.informative)}


def prompt_bootstrap_contrast(
    policy: str, reference: str, *, horizons: tuple[int, ...], mei: float, rule: str, out_dir: Path,
    flatness_path: Path | None = None,
) -> pd.DataFrame:
    """Pre-registration 9: the full-sample Δ on informative primary cells, with the 95% percentile
    interval of the same statistic over the prompt-bootstrap replicates (test ``emp_boot``)."""
    out_dir = Path(out_dir)
    horizons = tuple(int(h) for h in horizons)
    informative = _informative_cells(flatness_path or out_dir / "tables" / "emp_flatness.csv", horizons)
    as_registered = any(
        (policy, reference, horizons, float(mei), rule) == (r["policy"], r["reference"], tuple(r["horizons"]),
                                                             float(r["mei"]), r.get("rule"))
        for r in REGISTRATIONS.values() if r.get("interval") == "prompt_bootstrap"
    )
    row = {"row": "primary", "policy": policy, "reference": reference, "test": "emp", "horizon": "all",
           "n_informative": len(informative), "informative_cells": ";".join(f"{e}@{t}" for e, t in sorted(informative)),
           "mei": float(mei), "rule": rule, "as_registered": as_registered,
           "delta": np.nan, "lo": np.nan, "hi": np.nan, "n_boot": 0}
    if len(informative) < FLATNESS_MIN_INFORMATIVE:
        row["verdict"] = "uninformative"
        return pd.DataFrame([row])[list(BOOT_COLUMNS)]

    diffs, env_of, horizon_of = _diffs(out_dir, "emp", policy, reference, horizons)
    point = [float(d.mean()) for c, d in diffs.items() if (env_of[c], horizon_of[c]) in informative]
    if len(point) != len(informative):
        raise ValueError(f"emp holds {len(point)} of the {len(informative)} informative cells")

    bdiffs, benv, bhor = _diffs(out_dir, BOOT_TEST, policy, reference, horizons)
    by_boot: dict[int, dict[tuple[str, int], float]] = {}
    for c, d in bdiffs.items():
        m = _BOOT_ENV.match(benv[c])
        if m is None:
            raise ValueError(f"{c}: env id {benv[c]!r} is not a bootstrap replicate")
        key = (m.group("base"), bhor[c])
        if key in informative:
            by_boot.setdefault(int(m.group("b")), {})[key] = float(d.mean())
    n_boot = max(by_boot) + 1 if by_boot else 0
    stats_b = []
    for b in range(n_boot):
        cells_b = by_boot.get(b, {})
        missing = sorted(informative - set(cells_b))
        if missing:
            raise ValueError(f"bootstrap replicate b{b:03d} lacks informative cells {missing}")
        stats_b.append(float(np.mean([cells_b[k] for k in sorted(informative)])))

    delta = float(np.mean(point))
    lo, hi = (float(np.percentile(stats_b, 2.5)), float(np.percentile(stats_b, 97.5))) if stats_b else (np.nan, np.nan)
    row.update({"delta": delta, "lo": lo, "hi": hi, "n_boot": n_boot,
                "verdict": verdict_by_rule(rule, delta=delta, lo=lo, hi=hi, mei=mei)})
    return pd.DataFrame([row])[list(BOOT_COLUMNS)]
```

In `main()`, insert this block immediately **before** the existing line
`out = registered_contrast(args.policy, args.reference, test=args.test, horizons=horizons,` (everything from
that line to the end of `main()` stays exactly as it is):

```python
    if reg.get("interval") == "prompt_bootstrap":
        out = prompt_bootstrap_contrast(args.policy, args.reference, horizons=horizons, mei=args.mei,
                                        rule=reg["rule"], out_dir=args.out_dir)
        path = args.out or (args.out_dir / "tables" / reg["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        p = out.iloc[0]
        log.info("%s: %s vs %s on emp (informative %d): delta=%+.6f prompt-bootstrap [%+.6f, %+.6f] B=%d -> %s",
                 args.registration, args.policy, args.reference, p["n_informative"], p["delta"], p["lo"], p["hi"],
                 p["n_boot"], str(p["verdict"]).upper())
        log.info("wrote %s", path)
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_registered.py tests/test_deploy_registered_contrast.py -v`
Expected: all PASS (the existing registration tests must be untouched by the change).

- [ ] **Step 5: Commit**

```bash
git add experiments/growing_bandits/deploy/registered_contrast.py tests/test_empirical_registered.py
git commit -m "growing/deploy: Pre-registration 9 contrasts on the prompt-bootstrap interval, behind the flatness guard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Gates and the synthetic rehearsal

**Files:**
- Create: `experiments/growing_bandits/empirical/gates.py`
- Create: `experiments/growing_bandits/empirical/rehearsal.py`
- Test: `tests/test_empirical_gates.py`

**Interfaces:**
- Consumes: Task 2 (`load_attempts`, `terminal_outcomes`, statuses), Task 6 (`noise_model`, `kgrid`,
  `make_emp_cell`), Task 7 (`k_star_table`), `BetaReservoir`, `EmpiricalReservoir`.
- Produces:
  - `GateResult(name: str, passed: bool, checks: dict[str, dict])` with `.report() -> str`;
  - `pilot_gate(attempts, *, n_workers, anchor_arm, anchor_rate, max_cost=0.05, max_missing=0.05)`;
  - `collection_gate(attempts, *, max_missing=0.05)`;
  - `rehearsal_gate(results: dict)`;
  - `TASK_RATES` (the 60 logged Gmail per-task rates), `synthesize_outcomes(truth: dict[str, Reservoir], *,
    n_prompts, n_replicates, seed) -> pd.DataFrame`, `run_rehearsal(seed, workers, m) -> dict`,
    output `results/growing_bandits/empirical/rehearsal.json`.

- [ ] **Step 1: Write the failing tests**

```python
"""Gates G1-G3 as code, and the rehearsal's synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import gates  # noqa: E402
import rehearsal  # noqa: E402

from cold_start.growing.reservoirs import BetaReservoir  # noqa: E402


def _attempts(n_ok=100, n_missing=2, cost=0.03, anchor_successes=40, workers=2):
    rows = []
    for i in range(n_ok):
        rows.append({"pool": "G", "arm_id": f"G_{i % 5:02d}", "task_id": f"t{i}", "replicate": 0, "attempt": 1,
                     "status": "ok", "success": i % 2, "cost_usd": cost, "worker": i % workers})
    for i in range(n_missing):
        rows.append({"pool": "G", "arm_id": "G_00", "task_id": f"m{i}", "replicate": 0, "attempt": 3,
                     "status": "missing", "success": None, "cost_usd": 0.0, "worker": 0})
    for j in range(60):
        rows.append({"pool": "anchor", "arm_id": "anchor_baseline", "task_id": f"a{j}", "replicate": 0,
                     "attempt": 1, "status": "ok", "success": int(j < anchor_successes), "cost_usd": cost,
                     "worker": j % workers})
    return pd.DataFrame(rows)


def test_pilot_gate_passes_a_clean_pilot():
    g = gates.pilot_gate(_attempts(), n_workers=2, anchor_arm="anchor_baseline", anchor_rate=0.66)
    assert g.passed, g.report()


@pytest.mark.parametrize("kw, failing", [
    ({"cost": 0.09}, "cost_per_episode"),
    ({"n_missing": 20}, "missing_rate"),
    ({"anchor_successes": 20}, "anchor_drift"),
    ({"workers": 1}, "every_worker_produced"),
])
def test_pilot_gate_fails_each_check(kw, failing):
    g = gates.pilot_gate(_attempts(**kw), n_workers=2, anchor_arm="anchor_baseline", anchor_rate=0.66)
    assert not g.passed and not g.checks[failing]["passed"]


def test_collection_gate():
    assert gates.collection_gate(_attempts()).passed
    assert not gates.collection_gate(_attempts(n_missing=20)).passed


def test_rehearsal_gate():
    good = {"flat": {"sd_true": 0.03, "sd_npmle": 0.035, "sd_raw": 0.05},
            "wide": {"sd_true": 0.16, "sd_npmle": 0.155, "sd_raw": 0.165},
            "k_grid": [8, 12, 16, 24, 32, 48], "k_star_true": 24, "k_star_est": 32}
    assert gates.rehearsal_gate(good).passed
    bad = {**good, "k_star_est": 48}
    assert not gates.rehearsal_gate(bad).checks["k_star_within_one_step"]["passed"]
    flat_bad = {**good, "flat": {"sd_true": 0.03, "sd_npmle": 0.05, "sd_raw": 0.05}}
    assert not gates.rehearsal_gate(flat_bad).passed


def test_task_rates_are_the_logged_sixty():
    assert len(rehearsal.TASK_RATES) == 60
    assert sum(r == 1.0 for r in rehearsal.TASK_RATES) == 13


def test_level_solver_hits_the_prompt_mean():
    d = rehearsal.task_offsets(rehearsal.TASK_RATES)
    for mu in (0.3, 0.6, 0.9):
        a = rehearsal.solve_level(mu, d)
        assert np.mean(rehearsal.expit(a + d)) == pytest.approx(mu, abs=1e-9)


def test_synthesize_outcomes_matches_the_collector_schema():
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    out = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=30, seed=0)
    assert {"pool", "arm_id", "task_id", "replicate", "attempt", "status", "success"} <= set(out.columns)
    assert (out["replicate"] == 1).sum() == 30
    assert len(out[out["replicate"] == 0]) == 2 * 10 * 60
    again = rehearsal.synthesize_outcomes(truth, n_prompts=10, n_replicates=30, seed=0)
    pd.testing.assert_frame_equal(out, again)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_empirical_gates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gates'`

- [ ] **Step 3: Write `gates.py`**

```python
"""Gates G1-G3 of the empirical-pool study (spec section 8), as code, so a gate is a verdict.

    .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot
    .venv/bin/python experiments/growing_bandits/empirical/gates.py collection
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cold_start.growing import empirical as emp  # noqa: E402

MAX_COST_PER_EPISODE = 0.05
MAX_MISSING = 0.05
#: The historical anchor: the hand-written `baseline` arm on Gmail, gpt-5.4-mini, all logs.
ANCHOR_RATE = 0.66
N_WORKERS = 8


@dataclass
class GateResult:
    name: str
    checks: dict[str, dict] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks.values())

    def add(self, key: str, passed: bool, detail: str) -> None:
        self.checks[key] = {"passed": bool(passed), "detail": detail}

    def report(self) -> str:
        lines = [f"{self.name}: {'PASS' if self.passed else 'FAIL'}"]
        lines += [f"  [{'ok' if c['passed'] else 'XX'}] {k}: {c['detail']}" for k, c in self.checks.items()]
        return "\n".join(lines)


def _missing_rate(terminal: pd.DataFrame) -> float:
    return float((terminal["status"] == emp.STATUS_MISSING).mean()) if len(terminal) else 0.0


def pilot_gate(attempts: pd.DataFrame, *, n_workers: int, anchor_arm: str, anchor_rate: float,
               max_cost: float = MAX_COST_PER_EPISODE, max_missing: float = MAX_MISSING) -> GateResult:
    g = GateResult("G2 pilot")
    terminal = emp.terminal_outcomes(attempts)
    cost = float(attempts["cost_usd"].fillna(0.0).astype(float).sum()) / max(len(terminal), 1)
    g.add("cost_per_episode", cost <= max_cost, f"${cost:.4f} per terminal episode (limit ${max_cost})")
    miss = _missing_rate(terminal)
    g.add("missing_rate", miss <= max_missing, f"{miss:.2%} missing (limit {max_missing:.0%})")
    ok = terminal[terminal["status"] == emp.STATUS_OK]
    workers = set(int(w) for w in ok["worker"].dropna())
    g.add("every_worker_produced", workers >= set(range(n_workers)),
          f"workers with an ok episode: {sorted(workers)} of {n_workers}")
    anchor = ok[(ok["arm_id"] == anchor_arm) & (ok["replicate"] == 0)]
    n = len(anchor)
    k = int(anchor["success"].astype(int).sum())
    lo = stats.binom.ppf(0.025, n, anchor_rate) / max(n, 1)
    hi = stats.binom.ppf(0.975, n, anchor_rate) / max(n, 1)
    rate = k / max(n, 1)
    g.add("anchor_drift", n > 0 and lo <= rate <= hi,
          f"anchor {k}/{n} = {rate:.3f}; historical {anchor_rate} gives [{lo:.3f}, {hi:.3f}]")
    return g


def collection_gate(attempts: pd.DataFrame, *, max_missing: float = MAX_MISSING) -> GateResult:
    g = GateResult("G3 collection")
    miss = _missing_rate(emp.terminal_outcomes(attempts))
    g.add("missing_rate", miss <= max_missing, f"{miss:.2%} missing (limit {max_missing:.0%})")
    return g


def rehearsal_gate(results: dict) -> GateResult:
    g = GateResult("G1 rehearsal")
    for name in ("flat", "wide"):
        r = results[name]
        err = abs(r["sd_npmle"] - r["sd_true"])
        g.add(f"{name}_npmle_sd", err <= 0.01, f"NPMLE sd {r['sd_npmle']:.4f} vs true {r['sd_true']:.4f}")
    flat = results["flat"]
    g.add("flat_raw_inflated", flat["sd_raw"] >= 1.25 * flat["sd_true"],
          f"raw sd {flat['sd_raw']:.4f} vs true {flat['sd_true']:.4f} (needs >= 1.25x)")
    grid = list(results["k_grid"])
    step = abs(grid.index(results["k_star_est"]) - grid.index(results["k_star_true"]))
    g.add("k_star_within_one_step", step <= 1,
          f"K*(T=200) estimated {results['k_star_est']} vs true {results['k_star_true']} ({step} grid steps)")
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gate", choices=["pilot", "collection"])
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "empirical_pool")
    args = ap.parse_args(argv)
    attempts = emp.load_attempts(sorted(args.log_dir.glob("worker_*.jsonl")))
    if args.gate == "pilot":
        g = pilot_gate(attempts, n_workers=N_WORKERS, anchor_arm="anchor_baseline", anchor_rate=ANCHOR_RATE)
    else:
        g = collection_gate(attempts)
    print(g.report())
    return 0 if g.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write `rehearsal.py`**

```python
"""Gate G1: the whole estimation-and-replay pipeline on synthetic data with a known answer.

    .venv/bin/python experiments/growing_bandits/empirical/rehearsal.py --workers 12

Pool G's truth is flat (level 0.6, sd 0.03, the hand-written arms' measured spread); pool
F's is the corpus's `beta_good_common` (sd 0.16). Outcomes are generated per (prompt, task)
from ``p_ij = expit(a_i + d_j)``, where ``d_j`` are the 60 logged Gmail task difficulties
and ``a_i`` is solved so prompt i's mean over tasks is exactly its true rate. Passing means
the NPMLE recovers both spreads, the raw spread is visibly inflated on the flat pool, and
the pipeline locates K*(T = 200) on the wide pool to within one K-grid step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import describe  # noqa: E402
import gates  # noqa: E402
import k_star_envelope as kse  # noqa: E402
import replay  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402
from cold_start.growing.reservoirs import BetaReservoir, Reservoir  # noqa: E402

log = logging.getLogger("empirical.rehearsal")

#: Per-task success rates of the 12 hand-written arms on Gmail (1,631 logged episodes, 2026-09-26).
TASK_RATES: tuple[float, ...] = (
    *[0.0] * 6, *[0.04] * 3, 0.07, *[0.11] * 2, *[0.15] * 2, *[0.19] * 2, 0.26, 0.30, *[0.33] * 3, 0.37,
    *[0.41] * 3, *[0.52] * 5, *[0.63] * 2, 0.67, 0.87, *[0.93] * 4, *[0.96] * 8, 0.97, *[1.0] * 13,
)
expit = special.expit
OUT = ROOT / "results" / "growing_bandits" / "empirical" / "rehearsal.json"
REHEARSAL_T = 200


def task_offsets(rates) -> np.ndarray:
    d = special.logit(np.clip(np.asarray(rates, dtype=float), 0.02, 0.98))
    return d - d.mean()


def solve_level(mu: float, d: np.ndarray) -> float:
    return float(optimize.brentq(lambda a: float(np.mean(expit(a + d))) - mu, -30.0, 30.0, xtol=1e-12))


def synthesize_outcomes(truth: dict[str, Reservoir], *, n_prompts: int = 50, n_replicates: int = 300,
                        seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = task_offsets(TASK_RATES)
    rows, probs = [], {}
    for pool in sorted(truth):
        mus = truth[pool].sample(rng, n_prompts)
        for i, mu in enumerate(mus):
            p = expit(solve_level(float(np.clip(mu, 0.005, 0.995)), d) + d)
            arm = f"{pool}_{i:02d}"
            for j, pj in enumerate(p):
                probs[(pool, arm, f"t{j:02d}")] = pj
                rows.append({"pool": pool, "arm_id": arm, "task_id": f"t{j:02d}", "replicate": 0, "attempt": 1,
                             "status": emp.STATUS_OK, "success": int(rng.random() < pj), "cost_usd": 0.0})
    keys = sorted(probs)
    for k in rng.choice(len(keys), size=n_replicates, replace=False):
        pool, arm, task = keys[int(k)]
        rows.append({"pool": pool, "arm_id": arm, "task_id": task, "replicate": 1, "attempt": 1,
                     "status": emp.STATUS_OK, "success": int(rng.random() < probs[keys[int(k)]]), "cost_usd": 0.0})
    return pd.DataFrame(rows)


def _sd(res: Reservoir) -> float:
    u = (np.arange(20_001) + 0.5) / 20_001
    return float(np.std(res.sample_from_uniforms(u)))


def run_rehearsal(seed: int = 20260926, workers: int = 12, m: int = 1000) -> dict:
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    outcomes = synthesize_outcomes(truth, seed=seed)
    reservoirs, noise, _ = replay.estimate(outcomes)
    results: dict = {"noise": noise, "k_grid": [K for K in kse.DEFAULT_K_GRID if K <= REHEARSAL_T]}
    for name, pool in (("flat", "G"), ("wide", "F")):
        results[name] = {"sd_true": _sd(truth[pool]), "sd_npmle": reservoirs[(pool, "npmle")].sd(),
                         "sd_raw": reservoirs[(pool, "raw")].sd(), "mean_true": float(truth[pool].mean()),
                         "mean_npmle": reservoirs[(pool, "npmle")].mean()}
    true_wide = EmpiricalReservoir(emp.GRID, emp.grid_masses(truth["F"]), label="F_truth")
    cells = [replay.make_emp_cell("F", "truth", REHEARSAL_T, true_wide, m),
             replay.make_emp_cell("F", "npmle", REHEARSAL_T, reservoirs[("F", "npmle")], m)]
    kstar = describe.k_star_table(replay.kgrid(cells, workers=workers))
    results["k_star_true"] = int(kstar[kstar["variant"] == "truth"]["k_star"].iloc[0])
    results["k_star_est"] = int(kstar[kstar["variant"] == "npmle"]["k_star"].iloc[0])
    gate = gates.rehearsal_gate(results)
    results["gate"] = {"passed": gate.passed, "checks": gate.checks}
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--replicates", type=int, default=1000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = run_rehearsal(workers=args.workers, m=args.replicates)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    gate = gates.rehearsal_gate(results)
    print(gate.report())
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `make_emp_cell("F", "truth", ...)` uses `seed_for("F", 200)`, the same seed as the real F cell at
T = 200, which is correct here (CRN between truth and estimate) and harmless (the rehearsal writes no
episodes to the deployment tree).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_empirical_gates.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/growing_bandits/empirical/gates.py experiments/growing_bandits/empirical/rehearsal.py \
  tests/test_empirical_gates.py
git commit -m "growing/empirical: gates G1-G3 as code, and the synthetic rehearsal

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Gate G0 and G1 — corpus constants, rehearsal, frozen pools, Pre-registration 9

No new code. Every step here runs something and commits its output. **Nothing in this task spends on
WebArena**; the only paid call is the ~$1 Claude generation of pool F.

**Files:**
- Modify: `results/growing_bandits/deploy/baseline_params.json`, `results/growing_bandits/deploy/thresholds.json`
  (new `by_cap` blocks 50, 100, 500 only)
- Create: `data/empirical_pool/*` (pools, queue, manifest), `results/growing_bandits/empirical/rehearsal.json`
- Modify: `docs/growing_bandits/DEPLOYMENT_PLAN.md` (append Pre-registration 9)

- [ ] **Step 1: Full test suite green before anything runs**

Run: `.venv/bin/pytest -q`
Expected: all PASS.

- [ ] **Step 2: Select the missing cap-T constants on the corpus (Pre-registration 7's procedure)**

```bash
cp results/growing_bandits/deploy/baseline_params.json /tmp/baseline_params.before.json
cp results/growing_bandits/deploy/thresholds.json /tmp/thresholds.before.json
for X in 50 100 500; do
  .venv/bin/python experiments/growing_bandits/deploy/tune_baselines.py --cap $X --horizons $X --workers 12
  .venv/bin/python experiments/growing_bandits/deploy/select_fixed_k.py --cap $X --horizons $X --replicates 500 --workers 12
  .venv/bin/python experiments/growing_bandits/deploy/select_rule.py --rule level_star --cap $X --horizons $X --replicates 500 --workers 12
  .venv/bin/python experiments/growing_bandits/deploy/select_thresholds.py --cap $X --horizons $X --variants clock_quality_evidence_k4 --workers 12
done
```

Expected: each command logs its selection and exits 0.

- [ ] **Step 3: Verify the additions and that nothing else moved**

```bash
.venv/bin/python - <<'EOF'
import json
for name in ("baseline_params", "thresholds"):
    before = json.load(open(f"/tmp/{name}.before.json"))
    after = json.load(open(f"results/growing_bandits/deploy/{name}.json"))
    added = sorted(set(after["by_cap"]) - set(before["by_cap"]), key=int)
    assert added == ["50", "100", "500"], (name, added)
    for k, v in before.items():
        if k != "by_cap":
            assert after[k] == v, (name, k)
    for cap, block in before["by_cap"].items():
        assert after["by_cap"][cap] == block, (name, cap)
    print(name, "adds by_cap", added, "and changes nothing else")
bp = json.load(open("results/growing_bandits/deploy/baseline_params.json"))["by_cap"]
for cap in ("50", "100", "500"):
    assert {"p3_star", "fixed_K_star", "level_star"} <= set(bp[cap]), cap
EOF
```

Expected: two "adds by_cap ['50', '100', '500'] and changes nothing else" lines. If a script also rewrote
a shipped table, `git diff --stat` shows it: diff it cell by cell against HEAD and revert anything that is
not an addition before committing.

- [ ] **Step 4: Commit the constants**

```bash
git add results/growing_bandits/deploy/baseline_params.json results/growing_bandits/deploy/thresholds.json
git commit -m "growing/deploy: cap-T constants at caps 50, 100, 500, selected on the corpus (for Pre-registration 9)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Run the rehearsal (gate G1)**

Run: `.venv/bin/python experiments/growing_bandits/empirical/rehearsal.py --workers 12`
Expected: `G1 rehearsal: PASS` with four `[ok]` lines. **If it prints FAIL, stop the plan here** and report
the failing checks to Sanjay: the design cannot answer the question at this budget, and no money has been
spent.

- [ ] **Step 6: Commit the rehearsal**

```bash
git add results/growing_bandits/empirical/rehearsal.json
git commit -m "growing/empirical: gate G1 rehearsal passed on synthetic pools with known K*

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Build and freeze the pools**

Run: `.venv/bin/python experiments/growing_bandits/empirical/make_pools.py`
Expected: `wrote 3 pool files, 6360 queue items (660 pilot) under .../data/empirical_pool`.

Then read five F prompts and the G sample by eye:

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "experiments/growing_bandits/empirical")
import make_pools as mp
from pathlib import Path
F = mp.load_pool(Path("data/empirical_pool/pool_F.yaml"))
for a in F[:5]:
    print("----", a.arm.arm_id, len(a.text.split()), "words\n", a.text)
G = mp.load_pool(Path("data/empirical_pool/pool_G.yaml"))
print(len(G), "grid prompts; first vector", G[0].arm.vector)
EOF
```

Expected: five distinct, plausible instructions (none mentions a specific email or task). If any does,
stop and report to Sanjay rather than regenerating: regeneration is a pool change and must happen before
the pre-registration commit, with his agreement.

- [ ] **Step 8: Write Pre-registration 9 into `docs/growing_bandits/DEPLOYMENT_PLAN.md`**

Append this section verbatim, filling the six sha256 values from `data/empirical_pool/manifest.json`:

````markdown
## Pre-registration 9 — real prompt pools on WebArena Gmail (registered 2026-09-26, before any paid episode)

Every result in §1.0 and §12 was measured on synthetic reservoirs. This registers the first test on real
prompts. Design: `docs/superpowers/specs/2026-09-26-empirical-reservoir-design.md`.

### Design

- **Pools (frozen; sha256 in `data/empirical_pool/manifest.json`):** G = 50 prompts sampled from the
  2,304-point axis grid (seed 20260926); F = 50 instructions written by `claude-opus-4-7` from one fixed
  prompt; anchor = the hand-written `baseline`. `pool_G.yaml` `<sha>`, `pool_F.yaml` `<sha>`,
  `pool_anchor.yaml` `<sha>`, `f_generation_raw.json` `<sha>`, `queue.jsonl` `<sha>`.
- **Episodes:** every prompt × the 60 Gmail `real-tasks`, once; 300 random (prompt, task) cells a second
  time; `gpt-5.4-mini`, low effort, 30 steps, 180 s; one shuffled queue, pilot (5 + 5 + anchor) first.
  An agent timeout is a failure; an infrastructure error is retried twice then `missing`, never 0.
- **Reservoirs:** primary = the NPMLE of the true-rate distribution under x̄ᵢ ~ N(μᵢ, v̂/nᵢ), v̂ from the
  replicate pairs (pooled unless the pools differ by more than the bootstrap SE); sensitivity = raw rates
  and the best-AIC parametric fit.
- **Replay:** cap = T, M = 1,000, CRN seeds from 400,000,000; policies `always_search`, `p3_star`,
  `fixed_K_star`, `level_star`, `phi_k4` with cap-T constants selected **on the corpus only** (caps 50,
  100, 500 selected for this registration, before any episode, by Pre-registration 7's procedure); fixed K
  over `DEFAULT_K_GRID` for the U-curves.
- **Interval of record:** the prompt bootstrap — B = 200 resamples of each pool's prompts with their
  outcomes, v̂ and the NPMLE re-estimated, the four contrast policies redeployed at M = 250 on fresh seeds;
  95% percentile interval.
- **Flatness guard:** a primary cell (npmle, T ∈ {50, 100, 200}) is informative iff its fixed-K regret
  range exceeds 5 × MEI = 0.01. Fewer than 2 informative cells → every verdict is **uninformative: K
  barely matters on real Gmail prompts**. Otherwise contrasts pool the informative cells, equally weighted.

### The registered contrasts (MEI = 0.002)

1. **Primary (non-inferiority):** Δ = `p3_star` − `fixed_K_star`. Supported iff the bootstrap upper bound
   < +MEI; refuted iff the lower bound ≥ +MEI; else inconclusive. (`emp_primary`)
2. **Secondary (not-better):** Δ = `level_star` − `p3_star` (`emp_level`) and `phi_k4` − `p3_star`
   (`emp_phi`). Supported iff the lower bound > −MEI; refuted iff the upper bound ≤ −MEI; else
   inconclusive. Uncorrected.
3. **Secondary:** the lower-level pool has the larger K\*(T) at each primary T; reported with the ratio
   against exp(4 · (ℓ_high − ℓ_low)).
4. **Descriptive:** pool level / sd / q99 − mean against the 33 corpus environments; U-curves; each
   rule's gap to the pool's K\*; cap-64 cost at T ∈ {500, 1000} (extrapolation beyond 50 prompts).
5. **Sensitivity:** 1–2 on raw and parametric reservoirs, no verdicts.

### Gates

G1 rehearsal (passed, `results/growing_bandits/empirical/rehearsal.json`); G2 pilot: cost ≤ $0.05 per
episode, missing ≤ 5%, all 8 workers productive, anchor inside the Binomial(60, 0.66) 95% band; G3
missing ≤ 5%; hard budget stop $260.

### Run

```
scripts/run_empirical_pool.sh --pilot && .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot
scripts/run_empirical_pool.sh        && .venv/bin/python experiments/growing_bandits/empirical/gates.py collection
.venv/bin/python experiments/growing_bandits/empirical/replay.py estimate
.venv/bin/python experiments/growing_bandits/empirical/replay.py point --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --workers 12
.venv/bin/python experiments/growing_bandits/empirical/describe.py
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot --workers 12
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration emp_primary
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration emp_level
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration emp_phi
```
````

- [ ] **Step 9: Commit the frozen pools and the registration together (gate G0)**

```bash
git add data/empirical_pool/pool_G.yaml data/empirical_pool/pool_F.yaml data/empirical_pool/pool_anchor.yaml \
  data/empirical_pool/f_generation_raw.json data/empirical_pool/queue.jsonl data/empirical_pool/manifest.json \
  docs/growing_bandits/DEPLOYMENT_PLAN.md
git commit -m "growing/empirical: Pre-registration 9 -- real prompt pools on WebArena Gmail, pools frozen by hash

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -u origin empirical-reservoir
```

---

### Task 11: Gate G2 — the pilot (first paid run)

- [ ] **Step 1: Checkpoint with Sanjay**

Stop and ask: "G0 and G1 are committed (`<sha>`). The pilot spends ~$24 (660 episodes, ~1.1 h on 8
workers). Launch it?" Do not continue without an explicit yes.

- [ ] **Step 2: Launch**

Run: `scripts/run_empirical_pool.sh --pilot`
Then watch `logs/empirical_pool/collect.out` until `logs/empirical_pool/STATUS` reads `done`.

- [ ] **Step 3: Gate**

Run: `.venv/bin/python experiments/growing_bandits/empirical/gates.py pilot`
Expected: `G2 pilot: PASS`. On FAIL, report the failing lines to Sanjay and stop; on an `anchor_drift`
failure specifically, the agent or app has changed since June and the historical comparisons need his
decision before the full run.

- [ ] **Step 4: Report** the pilot's cost per episode, missing rate, anchor rate, and the ten pilot
prompts' raw rates to Sanjay. Do not commit logs (they are large and untracked by convention).

---

### Task 12: Gate G3, the full run, analysis and §13

- [ ] **Step 1: Checkpoint with Sanjay**

Ask: "Pilot passed. The full run spends ~$206 more (5,700 episodes, ~9.5 h; hard stop $260). Launch?"
Continue only on an explicit yes.

- [ ] **Step 2: Launch and gate**

Run: `scripts/run_empirical_pool.sh` (it resumes past the pilot), wait for `STATUS` = `done` (or `budget`,
which is itself a finding to report), then `.venv/bin/python experiments/growing_bandits/empirical/gates.py collection`.
Expected: `G3 collection: PASS`.

- [ ] **Step 3: Estimate, replay, describe**

```bash
.venv/bin/python experiments/growing_bandits/empirical/replay.py estimate
.venv/bin/python experiments/growing_bandits/empirical/replay.py point --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --workers 12
.venv/bin/python experiments/growing_bandits/empirical/describe.py
```

Expected: `estimate` logs six reservoirs (mean, sd, atoms, any validation error); `describe` logs
`informative primary cells: k of 6`.

Then confirm every deployed constant was selected at its own cap (Task 10 Step 2), since a fallback
would silently deploy cap-64 constants:

```bash
grep -c '"params_tuned": false' results/growing_bandits/deploy/manifest_emp.jsonl || true
```

Expected: `0`. Anything else: stop and report which policy/horizon fell back.

- [ ] **Step 4: Bootstrap and the registered contrasts**

```bash
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot --workers 12
for R in emp_primary emp_level emp_phi; do
  .venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration $R
done
```

Expected: three verdict lines. If `describe` found fewer than 2 informative cells, all three read
`UNINFORMATIVE` — that is the registered outcome, not a failure; the bootstrap stage still runs so the
record is complete.

- [ ] **Step 5: Sensitivity rows** (no verdicts): rerun contrasts 1–2 on the raw and parametric cells by
passing `--policy/--reference` with `--test emp` and non-registered horizons is not supported by the
bootstrap path, so compute them from the point episodes directly:

```bash
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "experiments/growing_bandits/deploy")
import pandas as pd, numpy as np, registered_contrast as rc
from pathlib import Path
out = Path("results/growing_bandits/deploy")
rows = []
for pol, ref in (("p3_star", "fixed_K_star"), ("level_star", "p3_star"), ("phi_k4", "p3_star")):
    diffs, env_of, hor = rc._diffs(out, "emp", pol, ref, (50, 100, 200))
    for variant in ("raw", "parametric", "npmle"):
        ds = [d.mean() for c, d in diffs.items() if env_of[c].endswith(variant)]
        rows.append({"policy": pol, "reference": ref, "variant": variant, "delta": float(np.mean(ds)), "n_cells": len(ds)})
pd.DataFrame(rows).to_csv(out / "tables" / "emp_sensitivity.csv", index=False)
print(pd.DataFrame(rows))
EOF
```

- [ ] **Step 6: Write §13 of `docs/growing_bandits/DEPLOYMENT_RESULTS.md`**

Title `## 13. Real prompt pools (Pre-registration 9)`. Subsections, each quoting its source table in the
document's convention `(table.csv, row keys)`:
13.1 what was collected (episodes, missing, cost, anchor);
13.2 where the real pools sit (`emp_pool_location.csv`) — level, sd, tail against the corpus;
13.3 the U-curves and K\* (`emp_kstar.csv`), and the flatness guard's count;
13.4 the three registered verdicts (`emp_primary.csv`, `emp_level.csv`, `emp_phi.csv`) with intervals;
13.5 the cross-pool prediction (`emp_cross_pool.csv`);
13.6 sensitivity and cap-64 extrapolation (`emp_sensitivity.csv`, `emp_cap64.csv`), labelled as such;
13.7 what this changes in §1.0 — one paragraph, and edit §1.0 in place if a claim's scope changed.

- [ ] **Step 7: Commit results and document**

```bash
git add data/empirical_pool/reservoirs results/growing_bandits/deploy/tables/emp_*.csv \
  docs/growing_bandits/DEPLOYMENT_RESULTS.md
git commit -m "growing/empirical: real prompt pools -- results (Pre-registration 9) and section 13

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 8: Open the PR** (base `main`, after PR #5 merges), summarizing the registered verdicts,
the pools' location on the simulation map, and the budget spent; end the body with
`🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
