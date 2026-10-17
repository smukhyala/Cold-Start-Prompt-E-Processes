"""Pre-registration 11: the GLK heterogeneity gate, its fake-positive diagnostics, and the Stage B thesis test.

    .venv/bin/python experiments/growing_bandits/empirical/glk_gate.py --study glk30 --tau-flat 0.019165515325428883
    .venv/bin/python experiments/growing_bandits/empirical/glk_gate.py --study glk   --tau-flat 0.019165515325428883

Runs after ``replay.py estimate / point / kgrid / boot / boot_kgrid`` and ``describe.py`` for the same study
(`study.GLK30`: the 40 GitLab procedural prompts on block A, every record re-scored at the 30-step budget --
the primary outcome; `study.GLK`: the same cell as collected -- secondary). Reads the replay tables under
``<out-dir>/tables/<prefix>_*.csv`` and the episodes under ``<out-dir>/episodes/<test>/``, and writes under
``--results`` (default ``results/growing_bandits/heterogeneity``):

* ``<prefix>_gate_classification.csv`` -- one row per task mix (``primary`` = block A equal weights;
  ``S1`` = bank-weighted, tiers e : m : h = 20 : 20 : 100; ``S2`` = discriminating tasks, pooled rate in
  [0.2, 0.8]): tau_main / tau_set with MLS intervals, split-half r_SB and p, raw and deconvolved sd, the
  regret range at T = 50 / 100 / 200 with its T = 200 bootstrap interval, the fixed-K8 and always-search
  gaps at T = 200, the drop-2 / drop-4 regret ranges at T = 200, the Pre-reg 10 class and the Pre-reg 11
  tier (``flat`` / ``moderate`` / ``meaningful`` / ``decision_relevant``), and ``detectable``.
* ``<prefix>_classification.csv`` -- the primary mix's Pre-reg 10 class in the shape
  `registered_contrast.load_classification` reads (``pool``, ``class``, ``manifest_sha256``).
* ``<prefix>_gate.csv`` -- ``gate_pass`` and its reason.
* ``<prefix>_diagnostics.csv`` -- diagnostics 1-7, long format (diagnostic, statistic, value, threshold, pass).
* ``<prefix>_prompt_table.csv`` -- one row per prompt: rate (this study's outcome and as collected), n, words,
  clock-ended share, mean steps, mean recorded errors.
* ``<prefix>_stage_b.csv`` -- C1 (``p3_star`` - ``fixed_K8``, rule ``superior``) and C2 (``always_search`` -
  ``p3_star``, rule ``inferior``) per primary T and pooled, on the prompt-bootstrap interval; C3 (share of the
  K-grid regret range ``p3_star`` captures) per T with its bootstrap 5th percentile; and one ``verdict`` row
  (``supported`` / ``partially_supported`` / ``contradicted`` / ``inconclusive``) with ``read`` = the gate.
* ``<prefix>_matrix_replay_T40.csv`` -- the model-free secondary: T = 40 on the recorded outcome matrix.

Every table carries `PROVENANCE_COLUMNS`. Stage B is always computed; its verdict is *read* only when the gate
passes (Pre-reg 11 "Gate").
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import calibrate  # noqa: E402
import describe  # noqa: E402
import het_verdicts as hv  # noqa: E402
import k_star_envelope as kse  # noqa: E402
import make_het_pools as mhp  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402
import stage0  # noqa: E402
import study as st  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing import heterogeneity as het  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402

log = logging.getLogger("empirical.glk_gate")

POOL = "GLK"
RESULTS = ROOT / "results" / "growing_bandits" / "heterogeneity"
PRIMARY_HORIZONS: tuple[int, ...] = (50, 100, 200)
GATE_T = 200
MIXES: tuple[str, ...] = ("primary", "S1", "S2")
#: S1: the 140-task bank's tier counts (20 easy, 20 medium, 100 hard).
TIER_WEIGHTS: dict[str, float] = {"e": 20.0, "m": 20.0, "h": 100.0}
DISCRIMINATING = (0.2, 0.8)

# ---- Pre-registration 11, "Classification" ---------------------------------------------------------
DR_RANGE_T200 = 0.02
DR_RR_LO_T200 = 0.01
DR_GAP_K8 = 0.005
DR_GAP_ALWAYS_SEARCH = 0.01
DR_DROP2_RANGE = 0.015
DR_SPLIT_HALF_P = 0.05
# ---- "Fake-positive diagnostics" --------------------------------------------------------------------
LOO_PROMPT_MAX = 0.30
LOO_TASK_MAX = 0.30
LENGTH_ADJ_MIN = 0.8
OVERLAP_NGRAM = 6
# ---- "Stage B" ---------------------------------------------------------------------------------------
MEI = 0.002
C3_POINT_MIN = 0.7
C3_LO_MIN = 0.5
C3_GAP_MAX = 0.005
C3_LO_PERCENTILE = 5.0
STAGE_B_MIN_HORIZONS = 2
MATRIX_T = 40
MATRIX_M = 2000
MATRIX_KS: tuple[int, ...] = (4, 8)

# ---- sizes of the gate's own replays (the primary mix's come from the registered replay tables) -----
N_BOOT = 200
POINT_M = replay.EMP_REPLICATES
BOOT_M = replay.BOOT_REPLICATES
#: The gate's own replay seeds: ``seed_base + GATE_SEED_OFFSET + 1_000 * index``, index < 3e6, so every seed
#: lies in [base + 5e9, base + 8e9) -- above the study's replay seeds (< base + 2.1e9) and below the next base.
GATE_SEED_OFFSET = 5_000_000_000
_MIX_STRIDE, _BOOT_STRIDE = 300_000, 1_000
_MIX_INDEX = {"primary": 0, "S1": 1, "S2": 2, "primary_drop2": 3, "primary_drop4": 4, "S1_drop2": 5,
              "S1_drop4": 6, "S2_drop2": 7, "S2_drop4": 8}
MATRIX_SEED_OFFSET = 9_000_000_000

PROVENANCE_COLUMNS: tuple[str, ...] = ("study", "score_cap", "task_universe", "manifest_sha256", "tau_flat",
                                       "calibration_sha256", "n_boot", "unregistered")


# ==== pure rules ======================================================================================


def tier_of(regret_range: Mapping[int, float], rr_lo_t200: float, rr_hi_t200: float, *, gap_k8: float,
            gap_always_search: float, drop2_range_t200: float, r_sb: float, split_half_p: float
            ) -> tuple[str, str, str]:
    """``(Pre-reg 10 class, Pre-reg 11 tier, reason)``. The tier is ``decision_relevant`` iff the class is
    ``meaningful`` and every decision-relevance condition holds; otherwise it is the class."""
    cls = hv.classify_cell(regret_range, rr_lo_t200, rr_hi_t200)
    if cls != "meaningful":
        return cls, cls, f"class {cls}"
    fails = []
    if not regret_range[GATE_T] >= DR_RANGE_T200:
        fails.append(f"range_T200 {regret_range[GATE_T]:.4f} < {DR_RANGE_T200}")
    if not rr_lo_t200 > DR_RR_LO_T200:
        fails.append(f"rr_lo_T200 {rr_lo_t200:.4f} <= {DR_RR_LO_T200}")
    if not gap_k8 >= DR_GAP_K8:
        fails.append(f"gap_fixed_K8 {gap_k8:.4f} < {DR_GAP_K8}")
    if not gap_always_search >= DR_GAP_ALWAYS_SEARCH:
        fails.append(f"gap_always_search {gap_always_search:.4f} < {DR_GAP_ALWAYS_SEARCH}")
    if not drop2_range_t200 >= DR_DROP2_RANGE:
        fails.append(f"drop2_range_T200 {drop2_range_t200:.4f} < {DR_DROP2_RANGE}")
    if not (np.isfinite(r_sb) and r_sb > 0.0 and split_half_p < DR_SPLIT_HALF_P):
        fails.append(f"split-half r_SB {r_sb:.3f} p {split_half_p:.3f} (needs > 0 at p < {DR_SPLIT_HALF_P})")
    if fails:
        return cls, cls, "meaningful but not decision-relevant: " + "; ".join(fails)
    return cls, "decision_relevant", "every decision-relevance condition holds"


def gate(tiers: Mapping[str, str]) -> tuple[bool, str]:
    """Stage B is read iff the primary mix is decision-relevant, or moderate with S1 decision-relevant."""
    p, s1 = tiers.get("primary"), tiers.get("S1")
    if p == "decision_relevant":
        return True, "primary mix decision-relevant"
    if p == "moderate" and s1 == "decision_relevant":
        return True, "primary mix moderate and S1 (bank-weighted) decision-relevant"
    return False, f"primary mix {p}, S1 {s1}: no decision-relevant heterogeneity"


def captured_share(regret_max: float, regret_min: float, regret_policy: float) -> float:
    """C3's statistic: the share of the K-grid regret range the policy captures (NaN on a zero range)."""
    span = regret_max - regret_min
    return float((regret_max - regret_policy) / span) if span > 0 else float("nan")


def c3_holds(captured: float, captured_lo: float, gap: float) -> bool:
    by_share = np.isfinite(captured) and captured >= C3_POINT_MIN and np.isfinite(captured_lo) and captured_lo >= C3_LO_MIN
    return bool(by_share or (np.isfinite(gap) and gap <= C3_GAP_MAX))


def stage_b_verdict(per_horizon: pd.DataFrame) -> tuple[str, str]:
    """Pre-reg 11 Stage B on one row per primary T with ``c1`` / ``c2`` (the contrast verdicts:
    ``supported`` / ``reversed`` / ``inconclusive``), ``c3`` (bool) and ``captured`` (float).
    Contradicted takes precedence, then supported, then the two partial readings."""
    c1 = per_horizon["c1"].astype(str)
    c2 = per_horizon["c2"].astype(str)
    c3 = per_horizon["c3"].astype(bool)
    captured = per_horizon["captured"].astype(float)
    if (c1 == "reversed").any() or (c2 == "reversed").any():
        return "contradicted", "C1 or C2 reversed beyond MEI at a primary horizon"
    # Amendment 1 (review finding 4): a horizon where C3 holds -- through its small-gap branch -- never counts
    # toward "captured < 0.5"; a tiny range makes the share meaningless, which is what the gap branch is for.
    if int(((captured < C3_LO_MIN) & ~c3).sum()) >= STAGE_B_MIN_HORIZONS:
        return "contradicted", f"p3_star captures < {C3_LO_MIN} of the range at >= {STAGE_B_MIN_HORIZONS} horizons"
    h1, h2 = c1 == "supported", c2 == "supported"
    k = STAGE_B_MIN_HORIZONS
    if int((h1 & h2 & c3).sum()) >= k:
        return "supported", f"C1, C2 and C3 all hold at >= {k} of {len(per_horizon)} horizons"
    if int((h2 & c3).sum()) >= k:
        return "partially_supported", "C2 and C3 hold but C1 fails (a small fixed K suffices: thin upper tail)"
    if int((h1 & h2).sum()) >= k:
        return "partially_supported", "C1 and C2 hold but C3 fails (the corpus-tuned constants do not transfer)"
    return "inconclusive", "no registered pattern holds at enough horizons"


# ==== task mixes and reservoirs =======================================================================


def tier_of_task(task_id: str) -> str:
    return str(task_id).removeprefix("task_")[0]


def mix_weights(tasks: list[str], Y: np.ndarray, mix: str) -> np.ndarray:
    """Per-task weights summing to 1 (zero weight = dropped): equal (primary), bank tier weights (S1), or
    equal on the discriminating tasks (S2)."""
    if mix == "primary":
        w = np.ones(len(tasks))
    elif mix == "S1":
        tiers = [tier_of_task(t) for t in tasks]
        counts = {t: tiers.count(t) for t in set(tiers)}
        w = np.array([TIER_WEIGHTS[t] / counts[t] for t in tiers])
    elif mix == "S2":
        w = het.discriminating_tasks(Y, *DISCRIMINATING).astype(float)
        if w.sum() == 0:
            raise ValueError("S2: no discriminating task")
    else:
        raise ValueError(f"unknown mix {mix!r}; mixes={MIXES}")
    return w / w.sum()


def weighted_scores(Y: np.ndarray, arms: list[str], w: np.ndarray) -> emp.PromptScores:
    """A prompt's score under the mix: sum_j w_j y_ij, with effective n = 1 / sum_j w_j^2 (so the NPMLE's
    per-prompt noise v / n is the variance of a w-weighted mean of cells with within-cell variance v)."""
    w = np.asarray(w, dtype=float)
    means = np.asarray(Y, dtype=float) @ w
    n_eff = 1.0 / float(np.sum(w ** 2))
    n = np.full(len(arms), n_eff)
    return emp.PromptScores(pool=POOL, arm_ids=tuple(arms), successes=means * n, n=n)


def drop_extremes(scores: emp.PromptScores, k: int) -> emp.PromptScores:
    """Pre-reg 11 diagnostic 1: ``k = 2`` drops the top and the bottom prompt; ``k = 4`` drops the *top four*
    (by raw rate; ties by arm id) -- the result must not rest on a few excellent prompts."""
    order = sorted(range(len(scores.arm_ids)), key=lambda i: (scores.means[i], scores.arm_ids[i]))
    if k == 2:
        drop = {order[0], order[-1]}
    elif k == 4:
        drop = set(order[-4:])
    else:
        raise ValueError(f"drop_extremes: k must be 2 or 4, got {k}")
    keep = [i for i in range(len(scores.arm_ids)) if i not in drop]
    return emp.PromptScores(pool=scores.pool, arm_ids=tuple(scores.arm_ids[i] for i in keep),
                            successes=scores.successes[keep], n=scores.n[keep])


def resample_scores(scores: emp.PromptScores, rng: np.random.Generator) -> emp.PromptScores:
    idx = rng.integers(0, len(scores.arm_ids), len(scores.arm_ids))
    return emp.PromptScores(pool=scores.pool, arm_ids=tuple(f"{scores.arm_ids[i]}#{k}" for k, i in enumerate(idx)),
                            successes=scores.successes[idx], n=scores.n[idx])


def gate_seed(study: st.Study, mix: str, boot: int | None, horizon: int) -> int:
    b1 = 0 if boot is None else boot + 1
    index = _MIX_INDEX[mix] * _MIX_STRIDE + b1 * _BOOT_STRIDE + int(horizon)
    return study.seed_base + GATE_SEED_OFFSET + 1_000 * index


def mix_cell(study: st.Study, mix: str, scores: emp.PromptScores, v: float, horizon: int, m: int,
             boot: int | None = None) -> CellSpec:
    res = emp.npmle_reservoir(scores, v, f"{POOL}_{mix}")
    suffix = "" if boot is None else f"_b{boot:03d}"
    return CellSpec(env_id=f"{study.test}_{POOL}{mix.replace('_', '')}_npmle{suffix}", env_spec=res.to_spec(),
                    horizon=int(horizon), cap=int(horizon), base_seed=gate_seed(study, mix, boot, horizon),
                    n_replicates=int(m))


def run_kgrid(cells: list[CellSpec], *, workers: int, study: st.Study, extra_k: bool = True) -> pd.DataFrame:
    """Fixed-K over the K-grid on `cells`; with `extra_k`, also K = 8 (fixed_K8) and K = T (always_search at
    cap T), which feed the gaps but not the range (the range is over `kse.DEFAULT_K_GRID`, as registered)."""
    frame = replay.kgrid(cells, workers=workers, study=study, split="glk_gate",
                         k_grid=tuple(sorted(set(kse.DEFAULT_K_GRID) | ({8} if extra_k else set()))))
    if extra_k:
        full = replay.kgrid(cells, workers=workers, study=study, split="glk_gate",
                            k_grid=tuple(sorted({int(c.horizon) for c in cells})))
        full = full[full["K"] == full["horizon"]]
        frame = pd.concat([frame, full], ignore_index=True).drop_duplicates(["env_id", "horizon", "K"])
    return frame


def range_and_gaps(kg: pd.DataFrame, horizon: int) -> dict:
    sub = kg[kg["horizon"] == horizon]
    on_grid = sub[sub["K"].isin(kse.DEFAULT_K_GRID)]
    lo = float(on_grid["regret"].min())
    out = {"regret_range": float(on_grid["regret"].max() - lo), "regret_min": lo,
           "regret_max": float(on_grid["regret"].max())}
    k8 = sub[sub["K"] == 8]["regret"]
    kt = sub[sub["K"] == horizon]["regret"]
    out["gap_fixed_K8"] = float(k8.iloc[0] - lo) if len(k8) else float("nan")
    out["gap_always_search"] = float(kt.iloc[0] - lo) if len(kt) else float("nan")
    return out


# ==== diagnostics =====================================================================================


def tau_set_sq(Y: np.ndarray, v: float, n_pairs: int, row_counts: np.ndarray | None = None) -> float:
    return float(het.tau_set_interval(Y, v, n_pairs, row_counts=row_counts)["tau_set"]) ** 2


def leave_one_out(Y: np.ndarray, v: float, n_pairs: int) -> dict:
    """Max relative change of tau_set^2 when one prompt is left out, and the max share of tau_set^2 one
    task carries (its removal's relative drop)."""
    full = tau_set_sq(Y, v, n_pairs)
    if not full > 0:
        return {"tau_set_sq": full, "loo_prompt_max": float("nan"), "loo_prompt_arg": -1,
                "loo_task_max": float("nan"), "loo_task_arg": -1}
    p = np.array([abs(tau_set_sq(np.delete(Y, i, axis=0), v, n_pairs) - full) / full for i in range(Y.shape[0])])
    t = np.array([(full - tau_set_sq(np.delete(Y, j, axis=1), v, n_pairs)) / full for j in range(Y.shape[1])])
    return {"tau_set_sq": full, "loo_prompt_max": float(p.max()), "loo_prompt_arg": int(p.argmax()),
            "loo_task_max": float(t.max()), "loo_task_arg": int(t.argmax())}


def length_adjusted_ratio(rates: np.ndarray, words: np.ndarray, tau_set: float) -> tuple[float, float]:
    """``(Spearman(words, rate), tau_set after removing the length-explained variance / tau_set)``."""
    rho = float(sps.spearmanr(words, rates).statistic) if np.std(words) > 0 and np.std(rates) > 0 else float("nan")
    if not tau_set > 0 or not np.std(words) > 0:  # no spread to adjust, or no length variation to regress on
        return rho, float("nan")
    slope, intercept = np.polyfit(np.asarray(words, float), np.asarray(rates, float), 1)
    explained = float(np.var(slope * np.asarray(words, float) + intercept))
    return rho, float(np.sqrt(max(tau_set ** 2 - explained, 0.0)) / tau_set)


def mix_noise(outcomes: pd.DataFrame, tasks: list[str], w: np.ndarray) -> tuple[float, int]:
    """Amendment 1 (review finding 6): the within-cell variance a mix's weighted prompt score carries,
    ``v_mix = sum_pairs w_t^2 s_p / sum_pairs w_t^2`` over the replicate pairs on tasks with weight > 0
    (``s_p = (x0 - x1)^2 / 2``). Equal weights on every task reproduce `het.noise_from_pairs`; S2 uses its own
    discriminating tasks' pairs (their noise is higher than the floor / ceiling tasks'); S1 weights hard tasks'."""
    ok = outcomes[(outcomes["status"] == emp.STATUS_OK) & (outcomes["pool"] == POOL)]
    wide = ok.assign(success=ok["success"].astype(float)).pivot_table(
        index=["arm_id", "task_id"], columns="replicate", values="success", aggfunc="first")
    if 0 not in wide.columns or 1 not in wide.columns:
        return float("nan"), 0
    wide = wide[[0, 1]].dropna().reset_index()
    wt = dict(zip(tasks, np.asarray(w, dtype=float), strict=True))
    ww = wide["task_id"].astype(str).map(wt).fillna(0.0).to_numpy() ** 2
    s = (wide[0].to_numpy(float) - wide[1].to_numpy(float)) ** 2 / 2.0
    keep = ww > 0
    if not keep.any():
        return float("nan"), 0
    return float(np.sum(ww[keep] * s[keep]) / np.sum(ww[keep])), int(keep.sum())


def noise_interval(outcomes: pd.DataFrame, *, n_boot: int, seed: int) -> tuple[float, float, float, int]:
    sq = replay._pair_sq_diffs(outcomes, POOL)
    if sq.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = [float(sq[rng.integers(0, sq.size, sq.size)].mean()) for _ in range(n_boot)]
    return float(sq.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), int(sq.size)


def ngram_overlaps(prompts: Mapping[str, str], task_texts: list[str], n: int = OVERLAP_NGRAM) -> dict[str, int]:
    tasks: set[tuple[str, ...]] = set()
    for t in task_texts:
        tasks |= mhp._ngram_windows(mhp._tokens(t), n)
    return {a: len(mhp._ngram_windows(mhp._tokens(p), n) & tasks) for a, p in prompts.items()}


def error_count(errors) -> int:
    if errors is None or (isinstance(errors, float) and np.isnan(errors)):
        return 0
    return sum(1 for e in list(errors) if e not in (None, "None", "") and not (isinstance(e, float) and np.isnan(e)))


def prompt_table(scored: pd.DataFrame, collected: pd.DataFrame, arms: list[str], tasks: list[str],
                 words: Mapping[str, int]) -> pd.DataFrame:
    def rep0(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["pool"] == POOL) & (frame["replicate"].astype(int) == 0) & (frame["status"] == emp.STATUS_OK)
                     & frame["task_id"].astype(str).isin(tasks)]
    s, c = rep0(scored), rep0(collected)
    rows = []
    for a in arms:
        ga, gc = s[s["arm_id"] == a], c[c["arm_id"] == a]
        rows.append({"arm_id": a, "n": len(ga), "rate": float(ga["success"].astype(float).mean()) if len(ga) else np.nan,
                     "rate_as_collected": float(gc["success"].astype(float).mean()) if len(gc) else np.nan,
                     "words": int(words.get(a, 0)),
                     "clock_share": float(ga["timed_out"].astype(bool).mean()) if len(ga) else np.nan,
                     "mean_steps": float(ga["steps"].astype(float).mean()) if len(ga) else np.nan,
                     "mean_errors": float(ga["errors"].map(error_count).mean()) if len(ga) and "errors" in ga else np.nan})
    return pd.DataFrame(rows)


# ==== the model-free T = 40 matrix replay =============================================================


def matrix_replay(cells: Mapping[tuple[str, str], list[int]], arms: list[str], tasks: list[str], *,
                  T: int = MATRIX_T, M: int = MATRIX_M, ks: tuple[int, ...] = MATRIX_KS, seed: int) -> pd.DataFrame:
    """Prompts drawn without replacement, a uniformly random task per pull, the outcome one of that
    (prompt, task) cell's recorded outcomes (drawn uniformly). ``fixed_K<k>`` recruits k prompts with one pull
    each, then spreads the remaining pulls round-robin over them; ``search_all_once`` pulls T distinct prompts
    once each (needs T <= the pool). Recommendation = highest Beta(1, 1) posterior mean (ties: first recruited).
    Regret = the pool's best empirical rate minus the recommended prompt's empirical rate. Refinement is
    round-robin with a Beta(1, 1) pick, not the harness's refine rule (disclosed beside C1)."""
    rng = np.random.default_rng(seed)
    # Amendment 1 (review finding 10): the prompt's rate is its replicate-0 rate (the first recorded outcome
    # of each observed cell); a pull draws a task among the prompt's *observed* cells, so a missing cell is
    # never scored as a failure.
    rate = {a: float(np.mean([cells[(a, t)][0] for t in tasks if cells.get((a, t))])) for a in arms}
    observed = {a: [t for t in tasks if cells.get((a, t))] for a in arms}
    best = max(rate.values())
    policies = [(f"fixed_K{k}", k) for k in ks] + ([("search_all_once", T)] if T <= len(arms) else [])
    regrets: dict[str, list[float]] = {name: [] for name, _ in policies}
    for _ in range(M):
        order = [arms[i] for i in rng.permutation(len(arms))]
        draws_task = rng.integers(0, len(tasks), T)
        draws_out = rng.random(T)
        for name, k in policies:
            held = order[:k]
            succ = dict.fromkeys(held, 0.0)
            n = dict.fromkeys(held, 0.0)
            for step in range(T):
                a = held[step % k]
                own = observed[a]
                obs = cells[(a, own[int(draws_task[step]) % len(own)])]
                y = obs[int(draws_out[step] * len(obs))]
                succ[a] += y
                n[a] += 1
            post = [((succ[a] + 1) / (n[a] + 2), -i) for i, a in enumerate(held)]
            pick = held[-max(post)[1]]
            regrets[name].append(best - rate[pick])
    rows = []
    boot = np.random.default_rng(seed + 1)
    for name, r in regrets.items():
        r = np.asarray(r)
        means = [float(r[boot.integers(0, r.size, r.size)].mean()) for _ in range(500)]
        rows.append({"policy": name, "T": T, "M": M, "regret": float(r.mean()),
                     "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5)),
                     "best_rate": best})
    return pd.DataFrame(rows)


def outcome_cells(outcomes: pd.DataFrame, tasks: list[str]) -> dict[tuple[str, str], list[int]]:
    """``{(arm, task): [replicate-0 outcome, replicate-1 outcome ...]}`` over ok rows (replicate 0 first)."""
    ok = outcomes[(outcomes["pool"] == POOL) & (outcomes["status"] == emp.STATUS_OK)
                  & outcomes["task_id"].astype(str).isin(tasks)].sort_values("replicate")
    out: dict[tuple[str, str], list[int]] = {}
    for r in ok.itertuples():
        out.setdefault((str(r.arm_id), str(r.task_id)), []).append(int(r.success))
    return out


# ==== orchestration ===================================================================================


def _words(data_dir: Path) -> tuple[dict[str, int], dict[str, str]]:
    doc = yaml.safe_load((Path(data_dir) / "pools" / f"{POOL}.yaml").read_text(encoding="utf-8"))
    texts = {str(a["arm_id"]): str(a.get("prompt_guidance") or "") for a in doc["arms"]}
    return {a: len(t.split()) for a, t in texts.items()}, texts


def _p3_boot_regret(out_dir: Path, study: st.Study, horizon: int, n_boot: int) -> np.ndarray:
    col = f"regret_{PRIMARY_RECOMMENDER}"
    vals = []
    for b in range(n_boot):
        path = (Path(out_dir) / "episodes" / study.boot_test
                / f"{study.test}_{POOL}_npmle_b{b:03d}_T{horizon}_cap{horizon}" / "p3_star.parquet")
        if not path.exists():
            raise FileNotFoundError(f"{path}: the boot episodes are incomplete; re-run `replay.py boot`")
        vals.append(float(pd.read_parquet(path, columns=[col])[col].mean()))
    return np.asarray(vals)


def stage_b(out_dir: Path, study: st.Study, classification_path: Path, manifest: Path, *, kstar: pd.DataFrame,
            gaps: pd.DataFrame, boot_kgrid: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for name, (policy, reference, rule) in (("C1", ("p3_star", "fixed_K8", "superior")),
                                            ("C2", ("always_search", "p3_star", "inferior"))):
        for hs in [(T,) for T in PRIMARY_HORIZONS] + [PRIMARY_HORIZONS]:
            r = rc.prompt_bootstrap_contrast(policy, reference, horizons=hs, mei=MEI, rule=rule, out_dir=out_dir,
                                             expected_n_boot=n_boot, study=study, cells=POOL,
                                             classification_path=classification_path,
                                             reservoir_manifest=manifest).iloc[0]
            rows.append({"contrast": name, "horizon": "all" if len(hs) > 1 else int(hs[0]), "policy": policy,
                         "reference": reference, "rule": rule, "delta": r["delta"], "lo": r["lo"], "hi": r["hi"],
                         "verdict": r["verdict"]})
    prim = kstar[(kstar["variant"] == "npmle") & (kstar["pool"] == POOL)]
    for T in PRIMARY_HORIZONS:
        k = prim[prim["horizon"] == T].iloc[0]
        p3 = gaps[(gaps["variant"] == "npmle") & (gaps["pool"] == POOL) & (gaps["horizon"] == T)
                  & (gaps["policy"] == "p3_star")].iloc[0]
        cap = captured_share(float(k["regret_max"]), float(k["regret_at_k_star"]), float(p3["regret"]))
        sub = boot_kgrid[(boot_kgrid["horizon"] == T) & (boot_kgrid["pool"] == POOL)]
        if {int(b) for b in sub["boot"]} != set(range(n_boot)):
            raise ValueError(f"{study.table('boot_kgrid')} at T={T} does not hold exactly replicates 0..{n_boot - 1}")
        p3b = _p3_boot_regret(out_dir, study, T, n_boot)
        g = sub.groupby("boot")["regret"]
        mx, mn = g.max(), g.min()
        caps = np.array([captured_share(float(mx.loc[b]), float(mn.loc[b]), p3b[b]) for b in range(n_boot)])
        cap_lo = float(np.nanpercentile(caps, C3_LO_PERCENTILE)) if np.isfinite(caps).any() else float("nan")
        rows.append({"contrast": "C3", "horizon": T, "policy": "p3_star", "reference": "k_grid",
                     "rule": "captured", "delta": cap, "lo": cap_lo, "hi": np.nan,
                     "gap_p3_star": float(p3["gap_to_k_star"]),
                     "verdict": "holds" if c3_holds(cap, cap_lo, float(p3["gap_to_k_star"])) else "fails"})
    for T in PRIMARY_HORIZONS:  # reported beside C1, not a verdict (Amendment 1, review finding 14)
        for pol in ("fixed_K4", "fixed_K8", "p3_star", "always_search"):
            g = gaps[(gaps["variant"] == "npmle") & (gaps["pool"] == POOL) & (gaps["horizon"] == T)
                     & (gaps["policy"] == pol)]
            if len(g):
                rows.append({"contrast": "gap_reported", "horizon": T, "policy": pol, "reference": "k_grid_min",
                             "rule": "reported", "delta": float(g["gap_to_k_star"].iloc[0]), "verdict": "reported"})
    frame = pd.DataFrame(rows)
    per = pd.DataFrame([{"horizon": T,
                         "c1": frame[(frame.contrast == "C1") & (frame.horizon == T)]["verdict"].iloc[0],
                         "c2": frame[(frame.contrast == "C2") & (frame.horizon == T)]["verdict"].iloc[0],
                         "c3": frame[(frame.contrast == "C3") & (frame.horizon == T)]["verdict"].iloc[0] == "holds",
                         "captured": frame[(frame.contrast == "C3") & (frame.horizon == T)]["delta"].iloc[0]}
                        for T in PRIMARY_HORIZONS])
    verdict, reason = stage_b_verdict(per)
    return pd.concat([frame, pd.DataFrame([{"contrast": "verdict", "horizon": "all", "verdict": verdict,
                                            "reason": reason}])], ignore_index=True)


def run_gate(*, study: st.Study, out_dir: Path, results: Path, tau_flat: float, calibration_csv: Path,
             n_boot: int = N_BOOT, point_m: int = POINT_M, boot_m: int = BOOT_M, workers: int = 12,
             matrix_m: int = MATRIX_M, registered_n_boot: int = replay.N_BOOT) -> dict[str, Path]:
    if study.pools != (POOL,):
        raise ValueError(f"glk_gate runs a one-pool GLK study, not {study.name} {study.pools}")
    cal_tau = calibrate.tau_flat_from_calibration(pd.read_csv(calibration_csv))
    if abs(cal_tau - tau_flat) > hv.TAU_FLAT_TOL:
        raise ValueError(f"--tau-flat {tau_flat} != {cal_tau} recomputed from {calibration_csv}")
    tables = Path(out_dir) / "tables"
    manifest = study.res_dir / replay.MANIFEST_FILE
    manifest_sha = rc.file_sha256(manifest)
    universe_name = replay.load_manifest(study.res_dir).get("task_universe")
    snapshot = replay.load_snapshot(study.res_dir)
    scored = st.restrict_to_universe(st.load_study_outcomes(study), study, universe_name)
    collected = st.restrict_to_universe(st.load_study_outcomes(dataclasses.replace(study, score_cap=None)),
                                        study, universe_name)
    key = ["pool", "arm_id", "task_id", "replicate"]

    def successes(frame: pd.DataFrame) -> pd.DataFrame:  # numeric, NaN-safe (review finding 3: dtypes differ)
        out = frame[key].astype(str).assign(success=pd.to_numeric(frame["success"], errors="coerce").fillna(-1).astype(int))
        return out.sort_values(key).reset_index(drop=True)

    if not successes(snapshot).equals(successes(scored)):
        raise ValueError("the live logs no longer match the frozen snapshot; re-run replay.py estimate")
    universe = hv.design_universe(study, task_universe=universe_name)
    Y, arms, tasks, n_imp, row_counts = hv._matrix(scored, POOL, universe)
    Yc, _, _, n_imp_c, rc_c = hv._matrix(collected, POOL, universe)
    v, n_pairs = het.noise_from_pairs(scored, POOL)
    v_c, n_pairs_c = het.noise_from_pairs(collected, POOL)
    unregistered = n_boot != replay.N_BOOT or point_m != POINT_M or boot_m != BOOT_M
    prov = {"study": study.name, "score_cap": study.score_cap, "task_universe": universe_name,
            "manifest_sha256": manifest_sha, "tau_flat": tau_flat, "calibration_sha256": rc.file_sha256(calibration_csv),
            "n_boot": n_boot, "unregistered": unregistered}

    kgrid = pd.read_csv(tables / study.table("kgrid"))
    kstar = describe.k_star_table(kgrid)
    gaps = pd.read_csv(tables / study.table("rule_gaps"))
    boot_kgrid = pd.read_csv(tables / study.table("boot_kgrid"))
    # review finding 13: every replay table must come from this manifest's reservoirs
    want = replay.load_manifest(study.res_dir)["reservoirs"][f"{POOL}_npmle"]
    got = set(kgrid.loc[kgrid["variant"] == "npmle", "reservoir_sha256"].astype(str))
    if got != {want}:
        raise ValueError(f"{study.table('kgrid')} npmle rows ran on reservoirs {sorted(got)}, not {want}; re-run kgrid")
    if set(boot_kgrid["manifest_sha256"].astype(str)) != {manifest_sha}:
        raise ValueError(f"{study.table('boot_kgrid')} was not computed on {manifest}; re-run replay.py boot_kgrid")
    rr_ci = hv.boot_regret_range_ci(boot_kgrid, expected_n_boot=registered_n_boot, horizons=(GATE_T,))

    def own_replay(name: str, scores: emp.PromptScores, v_mix: float) -> dict:
        """rr at every primary T, the T = 200 bootstrap interval and the T = 200 gaps, on the gate's own cells."""
        kg = run_kgrid([mix_cell(study, name, scores, v_mix, T, point_m) for T in PRIMARY_HORIZONS],
                       workers=workers, study=study)
        per_t = {T: range_and_gaps(kg, T) for T in PRIMARY_HORIZONS}
        rng = np.random.default_rng(gate_seed(study, name, None, 0))
        bcells = [mix_cell(study, name, resample_scores(scores, rng), v_mix, GATE_T, boot_m, boot=b)
                  for b in range(n_boot)]
        bk = run_kgrid(bcells, workers=workers, study=study, extra_k=False)
        ranges = bk.groupby("env_id")["regret"].agg(lambda r: float(r.max() - r.min())).to_numpy()
        return {"rr": {T: per_t[T]["regret_range"] for T in PRIMARY_HORIZONS},
                "lo": float(np.percentile(ranges, hv.RR_LO_PERCENTILE)),
                "hi": float(np.percentile(ranges, hv.RR_HI_PERCENTILE)),
                "gap_k8": per_t[GATE_T]["gap_fixed_K8"], "gap_as": per_t[GATE_T]["gap_always_search"]}

    base_scores = emp.prompt_scores(snapshot, POOL)
    primary_sh = het.split_half(Y, tasks, seed=hv.SPLIT_HALF_SEED)
    rows = []
    for mix in MIXES:
        w = mix_weights(tasks, Y, mix)
        keep = w > 0
        v_mix, n_mix = (v, n_pairs) if mix == "primary" else mix_noise(scored, tasks, w)
        if mix == "S1":
            # Amendment 1 (review finding 1): S1 re-weights the same prompts on the same tasks, so its split-half
            # is the primary mix's; the ANOVA spread estimates are unweighted and not defined for S1.
            ana = {"tau_main": np.nan, "tau_lo": np.nan, "tau_hi": np.nan, "tau_set": np.nan, "tau_set_lo": np.nan,
                   "tau_set_hi": np.nan, "tau_set_upper_one_sided": np.nan, "r_sb": primary_sh["r_sb"],
                   "split_half_p": primary_sh["p_value"]}
            scores = weighted_scores(Y, arms, w)
        else:
            sub_rc = None if mix == "S2" else row_counts
            ana = stage0.analyze_cell(Y[:, keep], [t for t, k in zip(tasks, keep, strict=True) if k],
                                      noise_var=v_mix, noise_df=n_mix, n_imputed=0 if mix == "S2" else n_imp,
                                      row_counts=sub_rc, seed=hv.SPLIT_HALF_SEED)
            scores = base_scores if mix == "primary" else weighted_scores(Y, arms, w)
        res = emp.npmle_reservoir(scores, v_mix, f"{POOL}_{mix}")
        row = {"mix": mix, "n_tasks": int(keep.sum()), "n_prompts": len(arms),
               **{k: ana.get(k, np.nan) for k in ("tau_main", "tau_lo", "tau_hi", "tau_set", "tau_set_lo", "tau_set_hi",
                                                  "tau_set_upper_one_sided", "r_sb")},
               "split_half_p": ana.get("split_half_p", ana.get("p_value", np.nan)),
               "raw_sd": float(np.std(scores.means)), "deconvolved_sd": describe.pool_summary(res)["sd"],
               "level": float(res.mean()), "noise_var": v_mix, "noise_pairs": n_mix}
        if mix == "primary":
            k = kstar[(kstar["variant"] == "npmle") & (kstar["pool"] == POOL)].set_index("horizon")
            rr = {T: float(k.loc[T, "regret_range"]) for T in PRIMARY_HORIZONS}
            lo, hi = rr_ci[(POOL, GATE_T)]
            g = gaps[(gaps["variant"] == "npmle") & (gaps["pool"] == POOL) & (gaps["horizon"] == GATE_T)].set_index("policy")
            gap_k8, gap_as = float(g.loc["fixed_K8", "gap_to_k_star"]), float(g.loc["always_search", "gap_to_k_star"])
        else:
            o = own_replay(mix, scores, v_mix)
            rr, lo, hi, gap_k8, gap_as = o["rr"], o["lo"], o["hi"], o["gap_k8"], o["gap_as"]
        drops: dict[int, float] = {}
        extra: dict = {}
        for kdrop in (2, 4):
            dname = f"{mix}_drop{kdrop}"
            dscores = drop_extremes(scores, kdrop)
            if mix == "primary":
                # Amendment 1 (review finding 2): the primary mix's tier is recomputed in full on each drop.
                o = own_replay(dname, dscores, v_mix)
                kept_rows = [arms.index(a) for a in dscores.arm_ids]
                sh = het.split_half(Y[kept_rows], tasks, seed=hv.SPLIT_HALF_SEED)
                dcls, dtier, _ = tier_of(o["rr"], o["lo"], o["hi"], gap_k8=o["gap_k8"], gap_always_search=o["gap_as"],
                                         drop2_range_t200=o["rr"][GATE_T], r_sb=float(sh["r_sb"]),
                                         split_half_p=float(sh["p_value"]))
                drops[kdrop] = o["rr"][GATE_T]
                extra.update({f"drop{kdrop}_class": dcls, f"drop{kdrop}_tier": dtier,
                              f"drop{kdrop}_rr_lo_T200": o["lo"], f"drop{kdrop}_gap_fixed_K8": o["gap_k8"],
                              f"drop{kdrop}_gap_always_search": o["gap_as"]})
            else:
                kg_d = run_kgrid([mix_cell(study, dname, dscores, v_mix, GATE_T, point_m)],
                                 workers=workers, study=study, extra_k=False)
                drops[kdrop] = range_and_gaps(kg_d, GATE_T)["regret_range"]
        cls, tier, reason = tier_of(rr, lo, hi, gap_k8=gap_k8, gap_always_search=gap_as, drop2_range_t200=drops[2],
                                    r_sb=float(row["r_sb"]), split_half_p=float(row["split_half_p"]))
        if mix == "S2" and tier in ("meaningful", "decision_relevant"):
            # Pre-reg 11: S2 may move a verdict from flat to moderate only (review finding 8: cap both).
            reason = f"S2 may move a verdict to moderate only (Pre-reg 11): capped from {tier}"
            cls = tier = "moderate"
        detectable = bool((np.isfinite(row["tau_set_lo"]) and row["tau_set_lo"] > 0)
                          or (np.isfinite(row["r_sb"]) and row["r_sb"] > 0 and row["split_half_p"] < DR_SPLIT_HALF_P))
        rows.append({**row, **{f"regret_range_T{T}": rr[T] for T in PRIMARY_HORIZONS}, "rr_lo_T200": lo,
                     "rr_hi_T200": hi, "gap_fixed_K8": gap_k8, "gap_always_search": gap_as,
                     "drop2_regret_range_T200": drops[2], "drop4_regret_range_T200": drops[4], **extra, "class": cls,
                     "tier": tier, "reason": reason, "detectable": detectable, **prov})
    classification = pd.DataFrame(rows)
    tiers = dict(zip(classification["mix"], classification["tier"], strict=True))
    passed, why = gate(tiers)

    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def write(name: str, frame: pd.DataFrame) -> None:
        path = results / f"{study.table_prefix}_{name}.csv"
        for k, val in prov.items():
            if k not in frame.columns:
                frame[k] = val
        frame.to_csv(path, index=False)
        written[name] = path

    write("gate_classification", classification)
    cls_path = results / f"{study.table_prefix}_classification.csv"
    pd.DataFrame([{"pool": POOL, "class": tiers_class(classification), "manifest_sha256": manifest_sha}]).to_csv(
        cls_path, index=False)
    written["classification"] = cls_path
    write("gate", pd.DataFrame([{"gate_pass": passed, "reason": why, **{f"tier_{m}": t for m, t in tiers.items()}}]))

    words, texts = _words(study.data_dir)
    ptab = prompt_table(scored, collected, arms, tasks, words)
    write("prompt_table", ptab)
    loo = leave_one_out(Y, v, n_pairs)
    tau_set = float(classification.set_index("mix").loc["primary", "tau_set"])
    rho_len, len_ratio = length_adjusted_ratio(ptab["rate"].to_numpy(float), ptab["words"].to_numpy(float), tau_set)
    v_hat, v_lo, v_hi, n_sq = noise_interval(scored, n_boot=1000, seed=study.seed_base + GATE_SEED_OFFSET - 1)
    overlaps = ngram_overlaps(texts, mhp.task_texts("gitlab"))
    tau_set_c = float(het.tau_set_interval(Yc, v_c, n_pairs_c, n_imputed=n_imp_c, row_counts=rc_c)["tau_set"]) \
        if n_pairs_c >= 1 else float("nan")

    def spearman(col: str) -> float:
        x, y = ptab[col].to_numpy(float), ptab["rate"].to_numpy(float)
        return float(sps.spearmanr(x, y).statistic) if np.nanstd(x) > 0 and np.nanstd(y) > 0 else float("nan")

    prim = classification.set_index("mix").loc["primary"]
    diag = [
        ("1_extremes", "drop2_regret_range_T200", prim["drop2_regret_range_T200"], DR_DROP2_RANGE,
         prim["drop2_regret_range_T200"] >= DR_DROP2_RANGE),
        ("1_extremes", "drop4_regret_range_T200", prim["drop4_regret_range_T200"], np.nan, None),
        ("1_extremes", "drop2_tier", prim["drop2_tier"], np.nan, None),
        ("1_extremes", "drop4_tier_top4_removed", prim["drop4_tier"], np.nan, None),
        ("2_loo_prompt", "max_rel_change_tau_set_sq", loo["loo_prompt_max"], LOO_PROMPT_MAX,
         loo["loo_prompt_max"] <= LOO_PROMPT_MAX if np.isfinite(loo["loo_prompt_max"]) else None),
        ("2_loo_prompt", "argmax_prompt", arms[loo["loo_prompt_arg"]] if loo["loo_prompt_arg"] >= 0 else "", np.nan, None),
        ("3_loo_task", "max_share_tau_set_sq", loo["loo_task_max"], LOO_TASK_MAX,
         loo["loo_task_max"] <= LOO_TASK_MAX if np.isfinite(loo["loo_task_max"]) else None),
        ("3_loo_task", "argmax_task", tasks[loo["loo_task_arg"]] if loo["loo_task_arg"] >= 0 else "", np.nan, None),
        ("3_loo_task", "split_half_r_sb", prim["r_sb"], 0.0, bool(np.isfinite(prim["r_sb"]) and prim["r_sb"] > 0
                                                                    and prim["split_half_p"] < DR_SPLIT_HALF_P)),
        ("3_loo_task", "split_half_p", prim["split_half_p"], DR_SPLIT_HALF_P, None),
        ("4_length", "spearman_words_rate", rho_len, np.nan, None),
        ("4_length", "length_adjusted_tau_set_ratio", len_ratio, LENGTH_ADJ_MIN,
         len_ratio >= LENGTH_ADJ_MIN if np.isfinite(len_ratio) else None),
        ("5_steps", "spearman_clock_share_rate", spearman("clock_share"), np.nan, None),
        ("5_steps", "spearman_mean_steps_rate", spearman("mean_steps"), np.nan, None),
        ("5_steps", "spearman_mean_errors_rate", spearman("mean_errors"), np.nan, None),
        ("5_steps", "tau_set_this_outcome", tau_set, np.nan, None),
        ("5_steps", "tau_set_as_collected", tau_set_c, np.nan, None),
        ("6_leak", f"prompts_with_{OVERLAP_NGRAM}gram_overlap", sum(1 for x in overlaps.values() if x), np.nan, None),
        ("6_leak", f"total_{OVERLAP_NGRAM}grams_shared", sum(overlaps.values()), np.nan, None),
        ("7_noise", "replicate_pairs", n_sq, np.nan, None),
        ("7_noise", "v_hat", v_hat, np.nan, None),
        ("7_noise", "v_hat_lo", v_lo, np.nan, None),
        ("7_noise", "v_hat_hi", v_hi, np.nan, None),
        ("7_noise", "raw_sd", prim["raw_sd"], np.nan, None),
        ("7_noise", "deconvolved_sd", prim["deconvolved_sd"], np.nan, None),
    ]
    write("diagnostics", pd.DataFrame(diag, columns=["diagnostic", "statistic", "value", "threshold", "pass"]))

    sb = stage_b(out_dir, study, cls_path, manifest, kstar=kstar, gaps=gaps, boot_kgrid=boot_kgrid,
                 n_boot=registered_n_boot)
    sb["read"] = passed
    write("stage_b", sb)
    write("matrix_replay_T40", matrix_replay(outcome_cells(scored, tasks), arms, tasks, M=matrix_m,
                                             seed=study.seed_base + MATRIX_SEED_OFFSET))

    verdict = sb[sb["contrast"] == "verdict"].iloc[0]
    log.info("tiers %s | gate %s (%s) | Stage B %s (%s; read=%s)", tiers, passed, why, verdict["verdict"],
             verdict["reason"], passed)
    (results / f"{study.table_prefix}_gate_provenance.json").write_text(json.dumps(
        {**prov, "tiers": tiers, "gate_pass": bool(passed), "stage_b": verdict["verdict"]}, indent=2, default=str))
    return written


def tiers_class(classification: pd.DataFrame) -> str:
    """The primary mix's Pre-reg 10 class (what `registered_contrast` selects cells by)."""
    return str(classification.set_index("mix").loc["primary", "class"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", choices=["glk30", "glk"], default="glk30")
    ap.add_argument("--tau-flat", type=float, required=True)
    ap.add_argument("--calibration", type=Path, default=hv.DEFAULT_CALIBRATION)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    study = replay.resolve_study(args.study)
    written = run_gate(study=study, out_dir=args.out_dir, results=args.results, tau_flat=args.tau_flat,
                       calibration_csv=args.calibration, n_boot=args.n_boot, workers=args.workers)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
