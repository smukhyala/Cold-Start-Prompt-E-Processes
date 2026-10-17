# Prompt Heterogeneity (Pre-registration 10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure prompt heterogeneity across environment (Gmail, GitLab) × prompt content (stylistic G,
knowledge-bearing K), validate the instrument with known-effect anchors, and test whether the value of
adaptive prompt search rises with heterogeneity.

**Architecture:** A `Study` config generalizes the Pre-registration 9 pipeline (replay, describe, contrasts)
from one hard-coded pool pair to any set of cells, leaving Pre-reg 9 byte-reproducible. A new pure-statistics
module estimates variance components, split-half reliability and bootstrap intervals. New pool builders
freeze manual bundles, generate K prompts behind a leak guard, and build three collector profiles (gitlab,
gmail, bridge). A calibration simulation and a verdicts module turn everything into the pre-registered
classification and H1–H4.

**Tech Stack:** Python 3.13 (`.venv`), numpy, scipy, pandas, statsmodels (sensitivity fits only),
matplotlib, anthropic SDK, the existing collector/harness/`run_deployment.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-prompt-heterogeneity-design.md`

## Global Constraints

- Agent identical to Pre-registration 9: `gpt-5.4-mini`, `llm_reasoning_effort: low`, `max_agent_steps: 30`,
  `use_vision: false`, headless. Wall clock: **600 s** for profiles `gitlab` and `bridge`, **180 s** for `gmail`.
- G prompts byte-identical to `data/empirical_pool/pool_G.yaml` (verify rendered sha256 per arm).
- GitLab tasks: 60 of 140 `real-tasks`, 20/20/20 by difficulty, `select_tasks(..., seed=20260928)`, two
  stratified blocks of 30. Gmail tasks: the existing Amendment-1 30-task subset.
- Cells and prompt counts: GLG 50, GLK 40, GMK 40, GMG (existing Pre-reg 9 G data), GMB 20 (bridge).
  Replicates: 300 GLG, 240 GLK, 120 GMK. Anchors: `baseline` (both apps), `explorer`,
  `gitlab_oracle_operator` (GitLab).
- K generator `claude-opus-4-7`, no temperature, from a frozen manual bundle ≤ 25,000 words; leak guard:
  reject any shared 8-token sequence with a task instruction of the app, or a named task entity.
- Statistics: MEI 0.002; flat = regret range < 0.005 at T ∈ {50, 100, 200} **and** τ upper 95% < 0.03;
  meaningful = regret range ≥ 0.01 at ≥ 2 of 3 **and** bootstrap lower bound at T = 200 > 0.005; B = 200
  prompt bootstrap (replay); B = 2,000 two-way bootstrap (τ); 10,000 permutations (split-half).
- Seeds: Pre-reg 9 study keeps `EMP_SEED_BASE = 400_000_000`; this study uses `600_000_000`; calibration
  uses `700_000_000`.
- Pre-registration 10 committed before any paid episode; the pilot is judged on cost/infra/anchors only.
- Never launch the collector, WebArena, or any LLM call from tests; never write to `data/`, `logs/` or
  `results/` from tests. `.venv/bin/python`, `.venv/bin/pytest`. `git add <paths>` only (force-add
  `data/`/`results/` files in operational tasks, as the repo does). Commits end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Pre-reg 9 must stay reproducible after the Study refactor** — the default study reproduces every emp
  table byte-for-byte → pinned in Task 1 (`test_prereg9_study_is_the_default_and_unchanged`).
- **A cell with a missing (prompt, task) outcome** must not crash or bias the moment estimator; it is
  imputed additively and counted → pinned in Task 2 (`test_missing_cells_are_imputed_and_counted`).
- **A K prompt that paraphrases a task instruction** must be rejected before freezing → pinned in Task 4
  (`test_leak_guard_rejects_task_ngrams_and_entities`).
- **Old GitLab data has one run per cell** (no replicates): τ must still be estimable and the interaction
  and noise reported jointly, not as a fabricated split → pinned in Task 2 (`test_single_replicate_design`).
- **A bridge cell has no replicate pairs**: its noise must be borrowed explicitly (GMG's), never silently
  zero → pinned in Task 1 (`test_cell_without_replicates_borrows_declared_noise`).

---

## File Structure

| file | responsibility |
|---|---|
| `experiments/growing_bandits/empirical/study.py` (new) | `Study` dataclass; `PREREG9` and `HETEROGENEITY` instances; outcome loading per study |
| `replay.py`, `describe.py`, `deploy/registered_contrast.py` (modify) | take a `Study` instead of module constants; defaults = `PREREG9` |
| `src/cold_start/growing/heterogeneity.py` (new) | variance components, split-half, two-way bootstrap, discriminating tasks, timeout sensitivity |
| `experiments/growing_bandits/empirical/stage0.py` (new) | Gmail and old-GitLab reanalyses |
| `experiments/growing_bandits/empirical/calibrate.py` (new) | value-of-search calibration simulation |
| `experiments/growing_bandits/empirical/make_het_pools.py` (new) | manual bundles, K generation + leak guard, G/anchor reuse, per-profile queues |
| `collect.py`, `scripts/watchdog_empirical_pool.py`, `scripts/run_empirical_pool.sh` (modify) | `--profile` |
| `experiments/growing_bandits/empirical/gates.py` (modify) | pilot gate with anchor ordering; per-cell G3 |
| `experiments/growing_bandits/empirical/het_verdicts.py` (new) | classification, H1–H4, H3 contrasts |
| `experiments/growing_bandits/empirical/figures.py` (new) | the seven paper figures |
| `deploy/policy_table.py` (modify) | `fixed_K8`; `TEST_POLICIES["het"|"het_boot"|"cal"]` |

Refactor tasks (1, 6) specify exact interfaces and tests; the implementer reads the current file and
threads the new parameter through, keeping every existing test green. New modules (2, 3, 4, 5, 8) are given
in full.

---

### Task 1: `Study` config — generalize replay, describe and contrasts

**Files:**
- Create: `experiments/growing_bandits/empirical/study.py`
- Modify: `experiments/growing_bandits/empirical/replay.py`, `describe.py`,
  `experiments/growing_bandits/deploy/registered_contrast.py`, `deploy/run_deployment.py` (`TESTS`,
  `DEFAULT_REPLICATES`, `seeds_may_repeat`, `_cell_grid` for `het`/`het_boot`/`cal`),
  `deploy/policy_table.py` (`fixed_K8`, `TEST_POLICIES`)
- Test: `tests/test_het_study.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Study:
      name: str                      # "prereg9" | "het"
      test: str                      # "emp" | "het"
      boot_test: str                 # "emp_boot" | "het_boot"
      pools: tuple[str, ...]         # ("G", "F") | ("GMG", "GMK", "GMB", "GLG", "GLK")
      data_dir: Path
      res_dir: Path
      log_dirs: tuple[Path, ...]
      seed_base: int                 # 400_000_000 | 600_000_000
      table_prefix: str              # "emp" | "het"
      noise_mode: str                # "pairwise_decision" (Pre-reg 9) | "per_pool"
      borrowed_noise: Mapping[str, str]  # pool -> pool whose v it uses when it has no replicate pairs
      extra_outcomes: tuple[tuple[Path, str, str], ...]  # (snapshot file, source pool, relabel as)
  PREREG9: Study
  HETEROGENEITY: Study
  def load_study_outcomes(study: Study) -> pd.DataFrame   # terminal outcomes of every log dir + relabelled extras
  ```
  `HETEROGENEITY`: `data_dir=data/heterogeneity`, `res_dir=data/heterogeneity/reservoirs`,
  `log_dirs=(logs/heterogeneity/gitlab, logs/heterogeneity/gmail, logs/heterogeneity/bridge)`,
  `extra_outcomes=((data/empirical_pool/reservoirs/outcomes_snapshot.jsonl, "G", "GMG"),)`,
  `borrowed_noise={"GMB": "GMG"}`, `noise_mode="per_pool"`.
- Every replay/describe/contrast public function gains `study: Study = PREREG9` (keyword); every module
  constant it used (`POOLS`, `DATA_DIR`, `LOG_DIR`, `RES_DIR`, `EMP_SEED_BASE`, test ids, `emp_` prefixes,
  `_BOOT_ENV`, `EMP_RESERVOIR_MANIFEST`, table names `emp_*.csv`) is read from the study. `env_id` becomes
  `f"{study.test}_{pool}_{variant}"`; `seed_for` uses `study.seed_base` and `study.pools.index(pool)`; the
  boot env regex is `rf"^(?P<base>{study.test}_[A-Za-z]+_npmle)_b(?P<b>\d{{3}})$"`. CLIs gain
  `--study prereg9|het` (default prereg9).
- `noise_model(outcomes, study=...)`: `"pairwise_decision"` keeps Pre-reg 9's exact G-vs-F logic;
  `"per_pool"` returns each pool's own v from its replicate pairs, and for a pool with no pairs uses
  `v[study.borrowed_noise[pool]]`, raising `ValueError` if the pool has no pairs and no declared borrow.
- `policy_table.POLICIES["fixed_K8"] = {"kind": "fixed_K", "params": {"K": 8}, "group": "baseline", "requires": []}`;
  `TEST_POLICIES["het"] = TEST_POLICIES["het_boot"] = ("always_search", "p3_star", "fixed_K_star",
  "level_star", "phi_k4", "fixed_K8")`; `TEST_POLICIES["cal"] = ("always_search", "p3_star", "fixed_K8")`.
  `run_deployment`: add `"het"`, `"het_boot"`, `"cal"` to `TESTS` with `DEFAULT_REPLICATES` 1000 / 250 /
  1000, `seeds_may_repeat` true for `het` and `cal`, and `_cell_grid` raising like `emp`.

- [ ] **Step 1: Write the failing tests**

```python
"""The Study refactor: Pre-reg 9 unchanged by default; a second study with its own pools, ids and seeds."""

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
import study as st  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402

RES = EmpiricalReservoir([0.3, 0.6, 0.9], [0.5, 0.3, 0.2])


def test_prereg9_study_is_the_default_and_unchanged():
    s = st.PREREG9
    assert (s.test, s.boot_test, s.pools, s.seed_base, s.table_prefix) == ("emp", "emp_boot", ("G", "F"), 400_000_000, "emp")
    assert replay.env_id("G", "npmle") == "emp_G_npmle"
    assert replay.seed_for("F", 200) == 400_000_000 + 1_000 * (1 * replay.POOL_STRIDE + 200)


def test_het_study_ids_and_seeds_are_disjoint_from_prereg9():
    h = st.HETEROGENEITY
    assert replay.env_id("GLK", "npmle", study=h) == "het_GLK_npmle"
    het_seeds = {replay.seed_for(p, T, b, study=h) for p in h.pools for T in (50, 100, 200, 500, 1000)
                 for b in (None, 0, 199)}
    emp_seeds = {replay.seed_for(p, T, b) for p in ("G", "F") for T in (50, 100, 200, 500, 1000)
                 for b in (None, 0, 199)}
    assert not het_seeds & emp_seeds
    import cells
    cells.assert_seed_disjointness(sorted(het_seeds))


def _pairs(pool, v, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        x0 = int(rng.random() < 0.5)
        x1 = x0 if rng.random() > 2 * v else 1 - x0
        for rep, x in ((0, x0), (1, x1)):
            rows.append({"pool": pool, "arm_id": f"{pool}_{k % 10}", "task_id": f"t{k}", "replicate": rep,
                         "status": "ok", "success": x})
    return rows


def test_cell_without_replicates_borrows_declared_noise():
    rows = _pairs("GMG", 0.05) + [{"pool": "GMB", "arm_id": "GMB_0", "task_id": "t0", "replicate": 0,
                                   "status": "ok", "success": 1}]
    for pool in ("GMK", "GLG", "GLK"):
        rows += _pairs(pool, 0.08, seed=1)
    noise = replay.noise_model(pd.DataFrame(rows), study=st.HETEROGENEITY)
    assert noise["v"]["GMB"] == noise["v"]["GMG"]
    assert noise["borrowed"] == {"GMB": "GMG"}


def test_undeclared_pool_without_pairs_raises():
    rows = _pairs("GMG", 0.05) + [{"pool": "GLK", "arm_id": "a", "task_id": "t", "replicate": 0,
                                   "status": "ok", "success": 1}]
    rows += _pairs("GMK", 0.05) + _pairs("GLG", 0.05)
    with pytest.raises(ValueError, match="GLK"):
        replay.noise_model(pd.DataFrame(rows), study=st.HETEROGENEITY)


def test_runner_and_policy_table_know_the_new_tests():
    import policy_table as pt
    assert {"het", "het_boot", "cal"} <= set(rd.TESTS)
    assert pt.POLICIES["fixed_K8"]["params"] == {"K": 8}
    assert "fixed_K8" in pt.TEST_POLICIES["het"] and "fixed_K8" in pt.TEST_POLICIES["cal"]


def test_load_study_outcomes_relabels_extras(tmp_path):
    snap = tmp_path / "snap.jsonl"
    pd.DataFrame([{"pool": "G", "arm_id": "G_00", "task_id": "t1", "replicate": 0, "status": "ok", "success": 1},
                  {"pool": "F", "arm_id": "F_00", "task_id": "t1", "replicate": 0, "status": "ok", "success": 0}]
                 ).to_json(snap, orient="records", lines=True)
    s = st.Study(name="x", test="x", boot_test="x_boot", pools=("GMG",), data_dir=tmp_path, res_dir=tmp_path,
                 log_dirs=(tmp_path / "none",), seed_base=900_000, table_prefix="x", noise_mode="per_pool",
                 borrowed_noise={}, extra_outcomes=((snap, "G", "GMG"),))
    out = st.load_study_outcomes(s)
    assert list(out["pool"]) == ["GMG"] and list(out["arm_id"]) == ["GMG_G_00"]
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_het_study.py -v` → FAIL
  (`ModuleNotFoundError: study`).

- [ ] **Step 3: Implement.** Create `study.py` with the dataclass, the two instances and
  `load_study_outcomes` (reads every `worker_*.jsonl` under each log dir via
  `cold_start.growing.empirical.load_attempts` + `terminal_outcomes`, missing dirs skipped; appends each
  extra snapshot's rows whose `pool == source`, setting `pool = relabel` and `arm_id = f"{relabel}_{arm_id}"`).
  Then refactor `replay.py`, `describe.py`, `registered_contrast.py` as specified in Interfaces — module
  constants stay as the `PREREG9` values so existing imports keep working. Add the policy-table and runner
  entries.

- [ ] **Step 4: Verify** — `.venv/bin/pytest tests/test_het_study.py tests/test_empirical_*.py
  tests/test_deploy_runner.py tests/test_deploy_registered_contrast.py -q` → all PASS; then prove Pre-reg 9
  unchanged on real data without writing anything: run `describe.k_star_table` on
  `results/growing_bandits/deploy/tables/emp_kgrid.csv` and `registered_contrast.prompt_bootstrap_contrast`
  for `emp_primary` into a tmp out path, and diff against the committed `emp_kstar.csv` / `emp_primary.csv`
  (must be identical).

- [ ] **Step 5: Commit** — `git add` the six files + test; message "growing/heterogeneity: Study config --
  replay, describe and contrasts parameterized by study; Pre-reg 9 unchanged".

---

### Task 2: Heterogeneity statistics

**Files:**
- Create: `src/cold_start/growing/heterogeneity.py`
- Test: `tests/test_heterogeneity_stats.py`

**Interfaces:**
- Produces:
  - `success_matrix(outcomes, pool) -> tuple[np.ndarray, list[str], list[str], int]` — (prompts × tasks)
    replicate-0 `ok` success matrix, arm ids, task ids, number of imputed cells.
  - `variance_components(Y, noise_var=None) -> dict` with keys `tau`, `tau2`, `task_var`, `interaction_var`,
    `noise_var`, `ms_prompt`, `ms_task`, `ms_resid`, `n_prompts`, `n_tasks`, `tau2_truncated`.
  - `two_way_bootstrap(Y, stat, *, n_boot=2000, seed=0, rows=None, cols=None) -> np.ndarray`
  - `split_half(Y, task_ids, *, seed=0, n_perm=10_000) -> dict` (`r`, `r_sb`, `p_value`)
  - `prompt_effects(Y) -> np.ndarray` (shrunken prompt main effects)
  - `discriminating_tasks(Y, lo=0.2, hi=0.8) -> np.ndarray` (column mask)
  - `noise_from_pairs(outcomes, pool) -> tuple[float, int]`

- [ ] **Step 1: Write the failing tests**

```python
"""Variance components for the crossed prompt x task design, and model-free reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cold_start.growing import heterogeneity as het


def _simulate(I=50, J=60, tau=0.05, task_sd=0.3, inter_sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, tau, I)
    b = rng.normal(0, task_sd, J)
    ab = rng.normal(0, inter_sd, (I, J))
    p = np.clip(0.6 + a[:, None] + b[None, :] + ab, 0.01, 0.99)
    return (rng.random((I, J)) < p).astype(float), p


def test_tau_is_recovered_on_average():
    taus = [het.variance_components(_simulate(seed=s)[0])["tau"] for s in range(40)]
    assert np.mean(taus) == pytest.approx(0.05, abs=0.01)


def test_flat_pool_gives_tau_near_zero_and_truncation_is_reported():
    vc = [het.variance_components(_simulate(tau=0.0, seed=s)[0]) for s in range(40)]
    assert np.mean([v["tau"] for v in vc]) < 0.02
    assert any(v["tau2_truncated"] for v in vc)


def test_components_identities():
    Y, _ = _simulate(seed=3)
    v = het.variance_components(Y, noise_var=0.1)
    J, I = Y.shape[1], Y.shape[0]
    assert v["tau2"] == pytest.approx(max((v["ms_prompt"] - v["ms_resid"]) / J, 0.0))
    assert v["task_var"] == pytest.approx(max((v["ms_task"] - v["ms_resid"]) / I, 0.0))
    assert v["interaction_var"] == pytest.approx(max(v["ms_resid"] - 0.1, 0.0))


def test_single_replicate_design():
    Y, _ = _simulate(seed=4)
    v = het.variance_components(Y)
    assert v["noise_var"] is None and v["interaction_var"] is None
    assert v["tau"] >= 0.0


def test_missing_cells_are_imputed_and_counted():
    rows = [{"pool": "P", "arm_id": f"p{i}", "task_id": f"t{j}", "replicate": 0, "status": "ok",
             "success": int((i + j) % 3 == 0)} for i in range(5) for j in range(6) if (i, j) != (2, 3)]
    Y, arms, tasks, n_imp = het.success_matrix(pd.DataFrame(rows), "P")
    assert Y.shape == (5, 6) and n_imp == 1 and np.isfinite(Y).all()


def test_two_way_bootstrap_brackets_the_truth():
    Y, _ = _simulate(tau=0.06, seed=5)
    draws = het.two_way_bootstrap(Y, lambda m: het.variance_components(m)["tau"], n_boot=300, seed=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert lo < 0.06 < hi


def test_split_half_detects_real_differences_and_not_noise():
    Y, _ = _simulate(tau=0.10, seed=6)
    tasks = [f"task_{'emh'[j % 3]}{j}" for j in range(Y.shape[1])]
    real = het.split_half(Y, tasks, seed=0, n_perm=2000)
    assert real["r_sb"] > 0.5 and real["p_value"] < 0.01
    Yf, _ = _simulate(tau=0.0, seed=7)
    flat = het.split_half(Yf, tasks, seed=0, n_perm=2000)
    assert flat["p_value"] > 0.01


def test_discriminating_tasks_mask():
    Y = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1]], dtype=float)
    assert het.discriminating_tasks(Y).tolist() == [False, False, True, True]


def test_prompt_effects_shrink_toward_zero():
    Y, _ = _simulate(tau=0.0, seed=8)
    raw = Y.mean(axis=1) - Y.mean()
    assert np.abs(het.prompt_effects(Y)).sum() <= np.abs(raw).sum() + 1e-12
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Prompt heterogeneity in a crossed prompt x task design (Pre-registration 10, section 6).

    y_ij = mu + a_i + b_j + (ab)_ij + e_ij

With one replicate-0 observation per (prompt, task) cell, the two-way ANOVA mean squares give
unbiased moment estimates: E[MS_prompt] = s2_resid + J tau^2, E[MS_task] = s2_resid + I s2_task,
E[MS_resid] = s2_resid = s2_ab + s2_e. tau therefore needs no noise model and no NPMLE; the replicate
pairs only split s2_resid into interaction and execution noise. Negative moment estimates are truncated at
zero and flagged.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

STATUS_OK = "ok"


def success_matrix(outcomes: pd.DataFrame, pool: str) -> tuple[np.ndarray, list[str], list[str], int]:
    sub = outcomes[(outcomes["pool"] == pool) & (outcomes["replicate"] == 0) & (outcomes["status"] == STATUS_OK)]
    wide = sub.assign(success=sub["success"].astype(float)).pivot_table(
        index="arm_id", columns="task_id", values="success", aggfunc="first")
    Y = wide.to_numpy(dtype=float)
    missing = ~np.isfinite(Y)
    n_imp = int(missing.sum())
    if n_imp:
        grand = np.nanmean(Y)
        row = np.nanmean(Y, axis=1, keepdims=True)
        col = np.nanmean(Y, axis=0, keepdims=True)
        Y = np.where(missing, np.clip(row + col - grand, 0.0, 1.0), Y)
    return Y, [str(a) for a in wide.index], [str(t) for t in wide.columns], n_imp


def variance_components(Y: np.ndarray, noise_var: float | None = None) -> dict:
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    if I < 2 or J < 2:
        raise ValueError("variance components need at least 2 prompts and 2 tasks")
    grand = Y.mean()
    rm, cm = Y.mean(axis=1), Y.mean(axis=0)
    ms_prompt = J * np.sum((rm - grand) ** 2) / (I - 1)
    ms_task = I * np.sum((cm - grand) ** 2) / (J - 1)
    resid = Y - rm[:, None] - cm[None, :] + grand
    ms_resid = np.sum(resid**2) / ((I - 1) * (J - 1))
    tau2_raw = (ms_prompt - ms_resid) / J
    tau2 = max(tau2_raw, 0.0)
    out = {
        "tau2": float(tau2), "tau": float(np.sqrt(tau2)), "tau2_truncated": bool(tau2_raw < 0.0),
        "task_var": float(max((ms_task - ms_resid) / I, 0.0)),
        "ms_prompt": float(ms_prompt), "ms_task": float(ms_task), "ms_resid": float(ms_resid),
        "n_prompts": I, "n_tasks": J, "noise_var": None, "interaction_var": None,
    }
    if noise_var is not None:
        out["noise_var"] = float(noise_var)
        out["interaction_var"] = float(max(ms_resid - noise_var, 0.0))
    return out


def two_way_bootstrap(Y: np.ndarray, stat: Callable[[np.ndarray], float], *, n_boot: int = 2000, seed: int = 0,
                      rows: np.ndarray | None = None, cols: np.ndarray | None = None) -> np.ndarray:
    """Resample prompts (rows) and tasks (columns) independently with replacement.

    `rows` / `cols` let a caller share resampling indices across cells (paired contrasts): pass an
    (n_boot, I) / (n_boot, J) index array; otherwise indices are drawn here.
    """
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    rng = np.random.default_rng(seed)
    R = rows if rows is not None else rng.integers(0, I, (n_boot, I))
    C = cols if cols is not None else rng.integers(0, J, (n_boot, J))
    return np.array([stat(Y[np.ix_(R[b], C[b])]) for b in range(len(R))])


def _halves(task_ids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: dict[str, list[int]] = {}
    for j, t in enumerate(task_ids):
        key = t.split("_")[1][0] if "_" in t else "x"
        strata.setdefault(key, []).append(j)
    a, b = [], []
    for key in sorted(strata):
        idx = np.array(strata[key])
        rng.shuffle(idx)
        a += list(idx[: len(idx) // 2])
        b += list(idx[len(idx) // 2:])
    return np.array(sorted(a)), np.array(sorted(b))


def split_half(Y: np.ndarray, task_ids: list[str], *, seed: int = 0, n_perm: int = 10_000) -> dict:
    Y = np.asarray(Y, dtype=float)
    a, b = _halves(task_ids, seed)
    xa, xb = Y[:, a].mean(axis=1), Y[:, b].mean(axis=1)
    r = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 and xb.std() > 0 else 0.0
    rng = np.random.default_rng(seed + 1)
    null = np.array([np.corrcoef(xa, rng.permutation(xb))[0, 1] for _ in range(n_perm)])
    null = np.nan_to_num(null)
    p = float((1 + np.sum(null >= r)) / (1 + n_perm))
    r_sb = 2 * r / (1 + r) if r > -1 else float("nan")
    return {"r": r, "r_sb": float(r_sb), "p_value": p, "n_half_a": int(a.size), "n_half_b": int(b.size)}


def prompt_effects(Y: np.ndarray) -> np.ndarray:
    """Empirical-Bayes shrunken prompt main effects: (row mean - grand) * tau2 / (tau2 + MS_resid / J)."""
    v = variance_components(Y)
    raw = Y.mean(axis=1) - Y.mean()
    denom = v["tau2"] + v["ms_resid"] / v["n_tasks"]
    return raw * (v["tau2"] / denom if denom > 0 else 0.0)


def discriminating_tasks(Y: np.ndarray, lo: float = 0.2, hi: float = 0.8) -> np.ndarray:
    cm = np.asarray(Y, dtype=float).mean(axis=0)
    return (cm >= lo) & (cm <= hi)


def noise_from_pairs(outcomes: pd.DataFrame, pool: str) -> tuple[float, int]:
    ok = outcomes[(outcomes["pool"] == pool) & (outcomes["status"] == STATUS_OK)]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success", aggfunc="first")
    if 0 not in wide.columns or 1 not in wide.columns:
        return float("nan"), 0
    pairs = wide[[0, 1]].dropna()
    if pairs.empty:
        return float("nan"), 0
    d = pairs[0].to_numpy() - pairs[1].to_numpy()
    return float(np.mean(d**2) / 2.0), int(len(pairs))
```

- [ ] **Step 4: Verify** — `.venv/bin/pytest tests/test_heterogeneity_stats.py -v` → PASS; report runtime
  (the permutation and bootstrap tests should stay under ~20 s; lower `n_boot`/`n_perm` in tests only).

- [ ] **Step 5: Commit** — "growing/heterogeneity: variance components, split-half reliability, two-way bootstrap".

---

### Task 3: Stage 0 — free reanalyses

**Files:**
- Create: `experiments/growing_bandits/empirical/stage0.py`
- Test: `tests/test_het_stage0.py`

**Interfaces:**
- Consumes: Task 2; `cold_start.growing.empirical.npmle`.
- Produces: `analyze_cell(Y, task_ids, *, noise_var=None, n_boot=2000, seed=0) -> dict` (variance
  components + 95% two-way-bootstrap interval of τ + split-half + discriminating-task τ + upper-tail mass);
  `gitlab_paired_matrix(csv_path, exclude=()) -> tuple[np.ndarray, list[str], list[str]]`;
  CLI writing `results/growing_bandits/heterogeneity/stage0_gmail.csv` (G, F from the Pre-reg 9 snapshot,
  noise from its replicate pairs) and `stage0_gitlab_paired.csv` (all 18 arms; without
  `gitlab_oracle_operator` and `explorer`; the 12 generic arms).
- Upper-tail mass: NPMLE on row means with σ²ᵢ = MS_resid / J, share of mass ≥ (weighted median + 0.10).

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
import stage0  # noqa: E402


def test_analyze_cell_fields_and_interval():
    rng = np.random.default_rng(0)
    p = np.clip(0.6 + rng.normal(0, 0.08, 40)[:, None] + rng.normal(0, 0.3, 30)[None, :], 0.02, 0.98)
    Y = (rng.random(p.shape) < p).astype(float)
    tasks = [f"task_{'emh'[j % 3]}{j}" for j in range(30)]
    out = stage0.analyze_cell(Y, tasks, noise_var=0.1, n_boot=200, seed=1)
    for k in ("tau", "tau_lo", "tau_hi", "r_sb", "split_half_p", "tau_discriminating", "upper_tail_mass",
              "interaction_var", "task_var"):
        assert k in out
    assert out["tau_lo"] <= out["tau"] <= out["tau_hi"]


def test_gitlab_paired_matrix(tmp_path):
    rows = [{"arm_id": a, "task": f"task_e{j}", "timestep": j, "reward": 1.0, "success": int(j % 2 == 0),
             "runtime": 1, "cost": 0.01, "steps": 3, "log_path": ""} for a in ("x", "y", "oracle") for j in range(4)]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    Y, arms, tasks = stage0.gitlab_paired_matrix(path, exclude=("oracle",))
    assert Y.shape == (2, 4) and arms == ["x", "y"]
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** `stage0.py`: `analyze_cell` composes Task 2 functions
  (`two_way_bootstrap(..., stat=lambda m: variance_components(m)["tau"])`, percentiles 2.5/97.5; split-half;
  τ on `discriminating_tasks`; tail via `emp.npmle(row_means, np.full(I, ms_resid/J))`); `gitlab_paired_matrix`
  pivots `paired_results.csv` (`arm_id` × `task`, `success`), dropping excluded arms; `main()` writes both CSVs
  (creating `results/growing_bandits/heterogeneity/`), reading the Pre-reg 9 snapshot via
  `replay.load_snapshot_unchecked(replay.RES_DIR)`.
- [ ] **Step 4: Verify** tests pass; full suite once.
- [ ] **Step 5: Commit** — "growing/heterogeneity: Stage 0 reanalyses (Gmail pools, old GitLab paired run)".

---

### Task 4: Pool builders — manual bundles, K generation with leak guard, G/anchor reuse, per-profile queues

**Files:**
- Create: `experiments/growing_bandits/empirical/make_het_pools.py`
- Test: `tests/test_het_pools.py`

**Interfaces:**
- Consumes: `make_pools` (`PoolArm`, `write_pool`, `load_pool`, `QueueItem`, `write_queue`, `read_queue`,
  `select_tasks`, `parse_freeform`, `freeform_arm`, `grid_arm`, `sha256_text`, `file_sha256`,
  `FREEFORM_TEMPLATE`, `GRID_TEMPLATE`, `AXES_PATH`, `ROOT`).
- Produces:
  - `APPS = {"gitlab": {"web_app": "apps/gitlab-plan-and-track", "manual_globs": [...], "app_description": ...},
    "gmail": {"web_app": "apps/gmail", "manual_globs": [...], "app_description": None}}` with
    GitLab globs `user/project/issues/**/*.md`, `user/project/labels.md`, `user/project/milestones/**/*.md`,
    `user/project/issue_board.md`, `user/group/epics/**/*.md`, `user/group/iterations/**/*.md` under
    `webarena-infinity/apps/user-manuals/gitlab`, preceded by `apps/gitlab-plan-and-track/APP_DESCRIPTION.md`;
    Gmail globs `organize-and-manage/*.md`, `compose-and-send/*.md`, `settings-and-configuration/*.md` under
    `.../user-manuals/gmail`.
  - `build_bundle(app, max_words=25_000) -> str` — files in sorted path order, each prefixed
    `## <relative path>`, truncated at the word cap (whole-file granularity, last file cut at the cap).
  - `KNOWLEDGE_PROMPT` (below), `generate_knowledge(client, app, bundle, n=40) -> tuple[list[str], dict]`.
  - `leak_violations(text, task_texts, entities) -> list[str]`; `task_entities(app) -> set[str]`.
  - `build_profiles(seed=20260928) -> None` (CLI `make_het_pools.py`) writing under `data/heterogeneity/`:
    `bundles/{gitlab,gmail}.md`, `k_generation_{gitlab,gmail}.json`, `pools/{GLG,GLK,GMK,GMB,anchors_gitlab,anchors_gmail}.yaml`,
    `{gitlab,gmail,bridge}/queue.jsonl`, `manifest.json` (sha256 of every file, task subsets, seeds, and
    `bridge_source`: `{"GMB_00": "GLG_17", ...}` — each bridge arm's G source, needed for H1's pairing).
    Refuses to overwrite without `--force`.
- Pools: `GLG_00..49` = the 50 G arms re-id'd (template.jinja ignores the id; rendered sha must equal the
  Pre-reg 9 sha — assert it). `GMB_00..19` = 20 of them chosen by `np.random.default_rng(seed).choice(50, 20,
  replace=False)`. `GLK_00..39`, `GMK_00..39` from generation. Anchors: `GL_anchor_baseline`,
  `GL_anchor_explorer` (vectors from `configs/arms_initial.yaml`), `GL_anchor_oracle` (vector and guidance from
  `configs/arms_gitlab_strong.yaml`), all three rendered with `configs/template_gitlab.jinja` as in the old
  paired run; `GM_anchor_baseline` with `template.jinja` (as Pre-reg 9's anchor).
- Queues (`QueueItem.pool` = cell id; anchors `pool="anchor"`):
  - `gitlab`: pilot = (first 5 GLG + first 5 GLK + 3 anchors) × block A, shuffled; then the rest of block A
    shuffled; then block B shuffled; then 300 GLG + 240 GLK replicate cells (drawn without replacement from
    their main cells) shuffled. Blocks: `select_tasks(bank, 60, seed)` then block A = `select_tasks(sixty, 30,
    seed + 1)`, block B = the rest.
  - `gmail`: GMK 40 × 30 + `GM_anchor_baseline` × 30, shuffled; then 120 GMK replicate cells.
  - `bridge`: GMB 20 × 30, shuffled.

```python
KNOWLEDGE_PROMPT = """You are writing system-prompt extensions for an AI agent that operates the web \
application described in the documentation below, through a browser. Each extension should give the agent \
practical procedural knowledge about THIS application: where features live, how its workflows run, useful \
shortcuts or navigation routes, and how to confirm that a change actually took effect.

Write {n} distinct extensions. Vary their scope (one workflow vs. broad coverage), emphasis, structure \
(prose, rules, numbered procedures) and length. Each must be 40 to 250 words, addressed to the agent in the \
second person, plain text only, and must not invent features the documentation does not describe.

Return only a JSON array of {n} strings.

<documentation>
{bundle}
</documentation>"""
```

- Leak guard: tokens = lowercase words (`re.findall(r"[a-z0-9]+", text.lower())`); violation if any 8-token
  window of the prompt appears in any task instruction's token windows, or if the prompt contains (case-
  insensitive, word-boundary) any entity from `task_entities(app)` = quoted strings (`'…'`, `"…"`) and
  capitalized multi-word names extracted from the app's task instructions (`real-tasks.json`), minus a
  stoplist of UI words present in the bundle itself (an entity that appears in the manual bundle is not a
  leak). `generate_knowledge` requests `n + 10` (50), drops violators and duplicates, keeps the first 40,
  raises if fewer than 40 remain, and records every rejection with its reason in the returned dict.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
import make_het_pools as mh  # noqa: E402


def test_leak_guard_rejects_task_ngrams_and_entities():
    tasks = ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label."]
    ents = {"sarah chen", "q1 product roadmap"}
    clean = "Use the star icon beside a message, then confirm the label list shows your change. " * 3
    leaky_ngram = "Star Sarah Chen's Q1 product roadmap email and move it to the inbox."
    leaky_entity = "When a message from Sarah Chen arrives, archive it after reading."
    assert mh.leak_violations(clean, tasks, ents) == []
    assert any("8-gram" in v for v in mh.leak_violations(leaky_ngram, tasks, ents))
    assert any("entity" in v for v in mh.leak_violations(leaky_entity, tasks, ents))


def test_generate_knowledge_filters_and_records(monkeypatch):
    good = [f"Guidance {i}: open the sidebar, pick the feature, confirm the banner. " + "word " * 40 for i in range(45)]
    bad = ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label. " + "word " * 40]

    class Msg:
        def create(self, **kw):
            assert "temperature" not in kw
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(bad + good))],
                                   stop_reason="end_turn")

    monkeypatch.setattr(mh, "task_texts", lambda app: ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label."])
    monkeypatch.setattr(mh, "task_entities", lambda app: {"sarah chen"})
    texts, raw = mh.generate_knowledge(SimpleNamespace(messages=Msg()), "gmail", "bundle text", n=40)
    assert len(texts) == 40 and all("Sarah" not in t for t in texts)
    assert raw["rejected"] and raw["rejected"][0]["reason"]


def test_generate_knowledge_raises_when_too_few(monkeypatch):
    class Msg:
        def create(self, **kw):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(["too short"] * 60))],
                                   stop_reason="end_turn")
    monkeypatch.setattr(mh, "task_texts", lambda app: [])
    monkeypatch.setattr(mh, "task_entities", lambda app: set())
    with pytest.raises(ValueError):
        mh.generate_knowledge(SimpleNamespace(messages=Msg()), "gmail", "b", n=40)


def test_bundle_is_capped_and_deterministic():
    a = mh.build_bundle("gmail", max_words=3000)
    assert len(a.split()) <= 3000 + 200 and a == mh.build_bundle("gmail", max_words=3000)
    assert a.startswith("## ")


def test_gitlab_queue_blocks_and_pilot():
    arms = {"GLG": [f"GLG_{i:02d}" for i in range(50)], "GLK": [f"GLK_{i:02d}" for i in range(40)],
            "anchor": ["GL_anchor_baseline", "GL_anchor_explorer", "GL_anchor_oracle"]}
    block_a = [f"task_e{i}" for i in range(10)] + [f"task_m{i}" for i in range(10)] + [f"task_h{i}" for i in range(10)]
    block_b = [f"task_e{i}" for i in range(10, 20)] + [f"task_m{i}" for i in range(10, 20)] + [f"task_h{i}" for i in range(10, 20)]
    q = mh.gitlab_queue(arms, block_a, block_b, seed=1)
    n_pilot = sum(i.pilot for i in q)
    assert n_pilot == 13 * 30 and all(i.pilot for i in q[:n_pilot])
    main = [i for i in q if i.replicate == 0]
    assert len(main) == (50 + 40 + 3) * 60
    first_b = next(k for k, i in enumerate(q) if i.task_id in block_b)
    assert all(i.task_id in block_a for i in q[:first_b])
    reps = [i for i in q if i.replicate == 1]
    assert sum(i.pool == "GLG" for i in reps) == 300 and sum(i.pool == "GLK" for i in reps) == 240
    assert all(i.index == k for k, i in enumerate(q))
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** the module per Interfaces (`task_texts(app)` reads `real-tasks.json` instructions;
  `gitlab_queue`, `gmail_queue`, `bridge_queue` pure functions used by `build_profiles`).
- [ ] **Step 4: Verify** tests; full suite once.
- [ ] **Step 5: Commit** — "growing/heterogeneity: manual bundles, knowledge-pool generation with leak guard, profile queues".

---

### Task 5: Calibration simulation

**Files:**
- Create: `experiments/growing_bandits/empirical/calibrate.py`
- Test: `tests/test_het_calibrate.py`

**Interfaces:**
- Consumes: `replay.make_emp_cell`, `replay.kgrid`, `describe.k_star_table`, `run_deployment.main(cells=...)`,
  `EmpiricalReservoir`, `emp.grid_masses`, `BetaReservoir`, `MixtureReservoir`.
- Produces: `calibration_pools() -> dict[str, Reservoir]`; `calibrate(workers, m=1000, out_dir=...) -> pd.DataFrame`
  writing `results/growing_bandits/heterogeneity/calibration.csv` with columns `pool_id, family, spread,
  tail_frac, tail_delta, true_sd, upper_tail_mass, horizon, regret_range, k_star, gap_fixed_K8, gap_p3_star,
  gap_always_search`.

```python
LEVEL = 0.6
SPREADS = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15)
TAIL_FRACS = (0.01, 0.02, 0.05)
TAIL_DELTAS = (0.10, 0.20, 0.30)
BULK_SD = 0.02
CAL_SEED_BASE = 700_000_000


def _beta(mean: float, sd: float) -> BetaReservoir:
    s = mean * (1 - mean) / sd**2 - 1
    return BetaReservoir(mean * s, (1 - mean) * s, validate=False)


def calibration_pools() -> dict[str, Reservoir]:
    pools = {f"beta_sd{sd:g}": _beta(LEVEL, sd) for sd in SPREADS}
    for f in TAIL_FRACS:
        for d in TAIL_DELTAS:
            pools[f"tail_f{f:g}_d{d:g}"] = MixtureReservoir(
                [_beta(LEVEL, BULK_SD), _beta(LEVEL + d, BULK_SD)], [1 - f, f], validate=False)
    return pools
```

  Each pool is discretized to an `EmpiricalReservoir(emp.GRID, emp.grid_masses(pool))` so cells use the same
  family as the empirical study; cells use a dedicated `Study`-free seed scheme
  `CAL_SEED_BASE + 1_000 * (pool_index * 10_000 + T)` and env ids `cal_<pool_id>`; K-grid via `replay.kgrid`,
  policies via `run_deployment.main(["--test", "cal", ...], cells=...)` into `results/growing_bandits/heterogeneity/cal_run/`
  (its own out-dir, so the deployment tree is untouched). `gap_*` = mean regret of the policy minus
  `regret_at_k_star` (fixed_K8 read from the K-grid row K = 8).

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))
import calibrate as cal  # noqa: E402


def test_pools_cover_the_registered_grid():
    pools = cal.calibration_pools()
    assert len(pools) == len(cal.SPREADS) + len(cal.TAIL_FRACS) * len(cal.TAIL_DELTAS)
    u = (np.arange(20001) + 0.5) / 20001
    sd = np.std(pools["beta_sd0.05"].sample_from_uniforms(u))
    assert abs(sd - 0.05) < 0.003


def test_calibrate_tiny_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "SPREADS", (0.02, 0.10))
    monkeypatch.setattr(cal, "TAIL_FRACS", ())
    monkeypatch.setattr(cal, "HORIZONS", (20,))
    monkeypatch.setattr(cal, "K_GRID", (2, 4, 8))
    out = cal.calibrate(workers=1, m=16, out_dir=tmp_path)
    assert set(out["pool_id"]) == {"beta_sd0.02", "beta_sd0.1"}
    wide = out.set_index("pool_id")["regret_range"]
    assert wide["beta_sd0.1"] > wide["beta_sd0.02"]
```

- [ ] **Step 2–4:** verify failure, implement, verify (full suite once).
- [ ] **Step 5: Commit** — "growing/heterogeneity: value-of-search calibration simulation".

---

### Task 6: Collector profiles and gates

**Files:**
- Modify: `experiments/growing_bandits/empirical/collect.py`, `scripts/watchdog_empirical_pool.py`,
  `scripts/run_empirical_pool.sh`, `experiments/growing_bandits/empirical/gates.py`
- Test: `tests/test_het_collect_profiles.py`

**Interfaces:**
- `collect.PROFILES: dict[str, dict]`:
  `"prereg9"` (current defaults: `apps/gmail`, 180 s, data `data/empirical_pool`, logs `logs/empirical_pool`,
  bank 60, pools from `pool_G/F/anchor.yaml`); `"gitlab"` (`apps/gitlab-plan-and-track`, 600 s,
  `data/heterogeneity/gitlab` queue + `data/heterogeneity/pools/{GLG,GLK,anchors_gitlab}.yaml`,
  `logs/heterogeneity/gitlab`, bank 140); `"gmail"` (`apps/gmail`, 180 s, pools `GMK`, `anchors_gmail`,
  logs `logs/heterogeneity/gmail`, bank 60); `"bridge"` (`apps/gmail`, 600 s, pool `GMB`,
  logs `logs/heterogeneity/bridge`, bank 60). `--profile` (default `prereg9`) selects them; `--data` /
  `--log-dir` still override. `verify_queue` checks the profile's manifest entry
  (`data/heterogeneity/manifest.json["files"]["<profile>/queue.jsonl"]`).
- Records gain `profile`, `timeout_s`, and `ended_by` ∈ {`"clock"` if `timed_out`, `"steps"` if
  `steps >= max_steps` and not `is_done`, `"agent"` otherwise}.
- Watchdog: forwards `--profile` and derives `LOG_DIR` from the profile (so STATUS / lock / relaunch log
  follow it); launcher passes `"$@"` unchanged.
- Gates: `pilot_gate(..., max_cost=0.25, anchor_order=("GL_anchor_oracle", "GL_anchor_explorer"))` adds check
  `anchor_order`: success rate of the first anchor > the second on pilot items; `collection_gate` reports and
  checks missing ≤ 5% **per pool** (cell). Existing Pre-reg 9 gate behavior unchanged when the new arguments
  are omitted. Gate CLI gains `--profile`.

- [ ] **Step 1: Tests** (fakes only; no processes): profile table round-trip; `ended_by` classification on
  three fake `RunResult`s; watchdog `collector_args` includes `--profile gitlab` and its log dir is
  `logs/heterogeneity/gitlab`; `pilot_gate` fails `anchor_order` when explorer ≥ oracle and passes otherwise;
  `collection_gate` fails when one pool has 6% missing even if the total is 3%; the default profile reproduces
  the current `prereg9` paths exactly.
- [ ] **Step 2–4:** verify failure; implement by reading the current files and replacing the module constants
  with the selected profile (keep every existing test green); full suite once.
- [ ] **Step 5: Commit** — "growing/heterogeneity: collector profiles (gitlab, gmail, bridge), ended_by, per-cell gates".

---

### Task 7: Figures

**Files:**
- Create: `experiments/growing_bandits/empirical/figures.py`
- Test: `tests/test_het_figures.py`

**Interfaces:** `make_figures(tables_dir, out_dir) -> list[Path]` producing PNG + PDF for the seven spec §7
figures from the CSVs that Task 8 and Stage 0 write (`het_components.csv`, `het_classification.csv`,
`calibration.csv`, `het_kgrid.csv`, `het_policy_gaps.csv`, `het_portability.csv`, plus historical rows
from `stage0_*.csv`). Style: dark background (`#0f1115`), light text, no top/right spines, one accent colour
per pool family (G, F, K), empirical points labelled, flat/moderate/meaningful bands as light horizontal
fills; match fonts/sizes of `deploy/make_deploy_figures.py`. Figure 3 plots the calibration curve (true sd
on x, regret range at T = 200 on y; tail mixtures as a second marker set) with every empirical cell as a
labelled point with its τ interval as horizontal error bars.

- [ ] **Step 1: Test** with small synthetic CSVs in `tmp_path`: all seven files are produced, non-empty, and
  a missing optional table (e.g. no meaningful cell → empty `het_policy_gaps.csv`) yields an annotated empty
  panel rather than an exception.
- [ ] **Step 2–4; Step 5: Commit** — "growing/heterogeneity: paper figures".

---

### Task 8: Verdicts — classification, H1–H4, H3 contrasts

**Files:**
- Create: `experiments/growing_bandits/empirical/het_verdicts.py`
- Modify: `experiments/growing_bandits/deploy/registered_contrast.py` (add `het_*` registrations reusing the
  Study-parameterized `prompt_bootstrap_contrast`); `experiments/growing_bandits/empirical/replay.py` (new stage
  `boot_kgrid`: the K-grid at T = 200, M = 250, on every bootstrap reservoir of every cell, written to
  `<prefix>_boot_kgrid.csv`; the regret-range lower bound for §6.5 is the 2.5th percentile over replicates)
- Test: `tests/test_het_verdicts.py`

**Interfaces:**
- `classify_cell(regret_range: dict[int, float], rr_lo_T200: float, tau_hi: float) -> str` → `"flat" |
  "meaningful" | "moderate"`, exactly spec §6.5.
- `h_verdict(delta_draws: np.ndarray, *, refute_below: float) -> dict` → one-sided 95% lower bound
  (5th percentile) and upper bound (95th); `"supported"` iff lo > 0, `"refuted"` iff hi < refute_below,
  else `"inconclusive"` (H1: refute_below 0.01; H2: 0.01).
- `h1(outcomes_gitlab_GLG, outcomes_bridge, bridge_arm_map, *, n_boot, seed)` — paired by prompt over the 20
  bridge prompts; tasks resampled independently per app.
- `h2(Y_K, Y_G, *, n_boot, seed)` — shared task indices (same task set), independent prompt resampling.
- `h4(effects_gmail, effects_gitlab, *, n_boot, seed)` — Spearman ρ and bootstrap CI over the 50 prompts.
- `h3(classes: dict[str, str], contrasts: pd.DataFrame, mei=0.002) -> dict` — spec §6.6 verbatim:
  supported / refuted / untestable, with the cells that decided it.
- `registered_contrast.REGISTRATIONS` gains, with `"interval": "prompt_bootstrap"`, `"study": "het"`,
  `"n_boot": 200`, horizons (50, 100, 200), mei 0.002, and `"cells": "meaningful"` (the informative set is the
  meaningful cells from `het_classification.csv`, not the flatness guard) — plus the same three with
  `"cells": "flat"` for the H3 refutation check:
  `het_scale` (p3_star vs fixed_K8, rule `superior` — supported iff hi < −MEI; reversed iff lo > +MEI),
  `het_spread` (always_search vs p3_star, rule `inferior` — supported iff lo > +MEI; reversed iff hi < −MEI),
  `het_primary`/`het_level`/`het_phi` (Pre-reg 9's three, same rules). Add `"superior"`/`"inferior"` to
  `RULES` and `verdict_by_rule`.
- Per cell, `het_verdicts` calls `stage0.analyze_cell` (variance components, τ interval, split-half,
  discriminating-task τ, upper-tail mass) with the cell's noise from `heterogeneity.noise_from_pairs` (or the
  declared borrow), and a **timeout sensitivity** row (spec §6.8): the same analysis with `ended_by == "clock"`
  episodes set to missing (then imputed and counted) instead of 0, plus per-prompt timeout rates.
- CLI `het_verdicts.py` writes `results/growing_bandits/heterogeneity/{het_components.csv,
  het_classification.csv, het_hypotheses.csv, het_portability.csv, het_policy_gaps.csv,
  het_timeout_sensitivity.csv}`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
import het_verdicts as hv  # noqa: E402


@pytest.mark.parametrize("rr, lo, tau_hi, expected", [
    ({50: 0.004, 100: 0.004, 200: 0.004}, 0.001, 0.02, "flat"),
    ({50: 0.004, 100: 0.004, 200: 0.004}, 0.001, 0.05, "moderate"),
    ({50: 0.012, 100: 0.011, 200: 0.009}, 0.006, 0.08, "moderate"),
    ({50: 0.012, 100: 0.004, 200: 0.015}, 0.006, 0.08, "meaningful"),
    ({50: 0.012, 100: 0.011, 200: 0.015}, 0.004, 0.08, "moderate"),
])
def test_classify_cell(rr, lo, tau_hi, expected):
    assert hv.classify_cell(rr, lo, tau_hi) == expected


def test_h_verdict_rules():
    assert hv.h_verdict(np.linspace(0.01, 0.05, 101), refute_below=0.01)["verdict"] == "supported"
    assert hv.h_verdict(np.linspace(-0.02, 0.005, 101), refute_below=0.01)["verdict"] == "refuted"
    assert hv.h_verdict(np.linspace(-0.02, 0.03, 101), refute_below=0.01)["verdict"] == "inconclusive"


def _contrasts(rows):
    return pd.DataFrame(rows, columns=["cells", "registration", "verdict"])


def test_h3_supported_untestable_refuted():
    classes = {"GLK": "meaningful", "GLG": "flat", "GMG": "flat"}
    ok = _contrasts([("meaningful", "het_scale", "supported"), ("meaningful", "het_spread", "supported"),
                     ("flat", "het_scale", "inconclusive"), ("flat", "het_spread", "inconclusive")])
    assert hv.h3(classes, ok)["verdict"] == "supported"
    assert hv.h3({"GLG": "flat", "GMG": "moderate"}, ok)["verdict"] == "untestable"
    bad = _contrasts([("meaningful", "het_scale", "reversed"), ("meaningful", "het_spread", "supported"),
                      ("flat", "het_scale", "inconclusive"), ("flat", "het_spread", "inconclusive")])
    assert hv.h3(classes, bad)["verdict"] == "refuted"
    flat_diff = _contrasts([("meaningful", "het_scale", "supported"), ("meaningful", "het_spread", "supported"),
                            ("flat", "het_scale", "supported"), ("flat", "het_spread", "inconclusive")])
    assert hv.h3(classes, flat_diff)["verdict"] == "refuted"


def test_h2_shares_task_indices():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.3, 30)
    YG = (rng.random((50, 30)) < np.clip(0.6 + base, 0.02, 0.98)).astype(float)
    YK = (rng.random((40, 30)) < np.clip(0.6 + rng.normal(0, 0.12, 40)[:, None] + base, 0.02, 0.98)).astype(float)
    out = hv.h2(YK, YG, n_boot=300, seed=1)
    assert out["verdict"] == "supported"
```

- [ ] **Step 2–4:** verify failure; implement; full suite once.
- [ ] **Step 5: Commit** — "growing/heterogeneity: pre-registered classification, H1-H4 verdicts, H3 contrasts".

---

### Task 9 (in-session): Stage 0, calibration, pools, Pre-registration 10 — no WebArena spend

- [ ] Run `stage0.py` and `calibrate.py --workers 12`; commit their outputs (force-add under `results/`).
  **Sanity checks before continuing:** Stage 0 Gmail τ(G) ≈ 0.035 and τ(F) ≈ 0 within their intervals
  (consistency with §13); old GitLab τ without the two outliers ≈ 0.02; calibration regret range at T = 200
  for `beta_sd0.035` ≈ 0.009 (Gmail G's value). A large disagreement stops the plan.
- [ ] Run `make_het_pools.py` (two Claude calls, ~$2). Read 5 K prompts per app; verify zero leak-guard
  violations in the kept 40 and review the recorded rejections. Commit `data/heterogeneity/`.
- [ ] Write **Pre-registration 10** into `DEPLOYMENT_PLAN.md`: design (spec §4), frozen file hashes from
  `data/heterogeneity/manifest.json`, statistics (§6), classification (§6.5), hypotheses and decision rules
  (§6.3, §6.6), gates (§4.5), the Stage-0 and calibration results as context, and the run commands. Commit
  and push **before** any paid episode.

### Task 10 (in-session): collection with checkpoints

- [ ] **Checkpoint with Sanjay:** pilot cost (~390 episodes). Launch
  `scripts/run_empirical_pool.sh --profile gitlab --pilot --budget 60`; gate
  `gates.py pilot --profile gitlab`. Report cost/episode and the projected full cost.
- [ ] **Checkpoint:** GitLab full (`--profile gitlab --budget <set with Sanjay>`), then `--profile gmail`,
  then `--profile bridge`; G3 per profile.

### Task 11 (in-session): analysis, §14, PR

- [ ] `replay.py estimate|point|kgrid --study het`; `describe.py --study het`; `replay.py boot --study het`
  (all cells, T ∈ {50, 100, 200}) and `replay.py boot_kgrid --study het` (regret-range lower bound); `het_verdicts.py`; the
  `het_*` registrations; `figures.py`.
- [ ] Write §14 of `DEPLOYMENT_RESULTS.md` (collection; Stage 0; variance decomposition; classification; H1–H4;
  H3; figures; what it licenses per spec §8) and update §1.0; commit tables/figures; open the PR.
