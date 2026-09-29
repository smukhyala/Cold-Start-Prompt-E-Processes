"""Pre-registration 10's verdicts (spec 2026-09-28-prompt-heterogeneity, sections 2 and 6.1-6.8).

    .venv/bin/python experiments/growing_bandits/empirical/het_verdicts.py --tau-flat <from Pre-reg 10>

Reads the heterogeneity study's frozen outcomes snapshot (`replay.load_snapshot`, sha-checked), its
K-grid (``tables/het_kgrid.csv``, `replay.py kgrid --study het`), its bootstrap K-grid
(``tables/het_boot_kgrid.csv``, `replay.py boot_kgrid --study het`) and its ``het``/``het_boot``
episodes, and writes under ``results/growing_bandits/heterogeneity/``:

* ``het_components.csv`` -- per cell, `stage0.analyze_cell`: variance components, tau_main and its MLS
  interval, tau_set and its MLS interval (execution noise from the cell's replicate pairs, or the
  study's declared borrow: GMB uses GMG's), split-half reliability, discriminating-task tau,
  upper-tail NPMLE mass with its prompt-bootstrap 95% CI (B = 200, NPMLE re-fit per resample).
* ``het_classification.csv`` -- section 6.5 per cell (`classify_cell`), with the regret range's bootstrap
  CI at every primary T (section 6.4; classification reads only T = 200's lower bound), stamped with the reservoir
  manifest's sha256 so the H3 contrasts can refuse a classification from another snapshot.
* ``het_policy_gaps.csv`` -- every ``het_*`` registration (`registered_contrast.REGISTRATIONS`), on its
  class's cells pooled (the registered row) and on each of those cells alone (section 6.6 reads the
  contrasts per meaningful cell).
* ``het_hypotheses.csv`` -- H1 (primary: the 20 bridge prompts, GitLab GLG vs Gmail-at-600 s GMB,
  prompts resampled jointly; secondary: all 50 G prompts, GLG vs GMG at 180 s), H2 per app (K vs G,
  prompts resampled independently, same task set), H3 (section 6.6), H4 (Spearman, reported).
* ``het_portability.csv`` -- the 50 G prompts' shrunken main effects in Gmail and GitLab (H4).
* ``het_timeout_sensitivity.csv`` / ``het_timeout_rates.csv`` -- section 6.8: the per-cell analysis
  with clock-ended episodes set to missing (imputed, counted, beyond the 5% ceiling of the primary
  analysis), and per-prompt timeout rates.

Every cell's matrix is built on the design universe (`design_universe`): the pool file's arms and the
app's task subset from ``data/heterogeneity/manifest.json`` -- a never-attempted arm or task is a missing
cell (counted toward the 5% ceiling), never a silently smaller matrix. Seeds: ``VERDICT_SEED`` plus a
fixed offset per bootstrap.

Heterogeneity quantities (section 2): tau_main = sqrt((MS_prompt - MS_resid)/J) for H1, H2, H4;
tau_set's one-sided upper 95% MLS bound for the section 6.5 flat rule. Contrast intervals for H1/H2/H4
come from `heterogeneity.prompt_bootstrap` (prompts resampled, each cell's task set held fixed).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable, Mapping
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

import describe  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402
import stage0  # noqa: E402
import study as st  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing import heterogeneity as het  # noqa: E402

log = logging.getLogger("empirical.het_verdicts")

OUT_DIR = ROOT / "results" / "growing_bandits" / "heterogeneity"
PRIMARY_HORIZONS: tuple[int, ...] = (50, 100, 200)
#: Section 6.5.
FLAT_RANGE = 0.005
MEANINGFUL_RANGE = 0.01
MEANINGFUL_MIN_HORIZONS = 2
RR_LO_HORIZON = 200
RR_LO_MIN = 0.005
RR_LO_PERCENTILE = 2.5
RR_HI_PERCENTILE = 97.5
#: Sections 6.3 / 6.6.
MEI = 0.002
H_REFUTE_BELOW = 0.01
N_BOOT = 2_000
#: The verdicts' seed base (fix round 1 ruling); each bootstrap uses base + a fixed offset.
VERDICT_SEED = 60_000_000_000
SPLIT_HALF_SEED = 0  # stage0's
#: Section 6.4: the upper-tail mass's prompt-bootstrap CI (NPMLE re-fit per resample), seeded
#: ``VERDICT_SEED + TAIL_SEED_OFFSET + <pool index in the study>``.
TAIL_N_BOOT = 200
TAIL_SEED_OFFSET = 100
UPPER_TAIL_DELTA = 0.10
#: Each pool's app, and the manifest key of that app's task set (``data/heterogeneity/manifest.json``).
POOL_APP: dict[str, str] = {"GMG": "gmail", "GMK": "gmail", "GMB": "gmail", "GLG": "gitlab", "GLK": "gitlab"}
TASK_SUBSET_KEY: dict[str, str] = {"gitlab": "gitlab_task_subset_60", "gmail": "gmail_task_subset_30"}
#: A relabelled pool's arms come from its source study's pool file (``study.extra_outcomes``' relabel).
EXTRA_POOL_FILES: dict[str, Path] = {"GMG": ROOT / "data" / "empirical_pool" / "pool_G.yaml"}
H3_SCALE, H3_SPREAD = "het_scale", "het_spread"
HET_REGISTRATIONS: tuple[str, ...] = tuple(k for k, r in rc.REGISTRATIONS.items() if r.get("study") == "het")

_GMG_ARM = re.compile(r"^GMG_G_(\d+)$")
_GLG_ARM = re.compile(r"^GLG_(\d+)$")


# ---- section 6.5 --------------------------------------------------------------------------


def classify_cell(regret_range: Mapping[int, float], rr_lo_T200: float, tau_set_upper: float,
                  tau_flat: float) -> str:
    """``flat`` iff the regret range is < 0.005 at T = 50, 100 and 200 and tau_set's one-sided upper 95%
    MLS bound is < `tau_flat`; ``meaningful`` iff the regret range is >= 0.01 at >= 2 of the 3 and the
    bootstrap lower bound of the regret range at T = 200 is > 0.005; ``moderate`` otherwise.
    A missing horizon raises `KeyError`; a non-finite input raises `ValueError`."""
    rr = [float(regret_range[T]) for T in PRIMARY_HORIZONS]
    for name, v in (("regret_range", rr), ("rr_lo_T200", [rr_lo_T200]), ("tau_set_upper", [tau_set_upper]),
                    ("tau_flat", [tau_flat])):
        if not np.all(np.isfinite(np.asarray(v, dtype=float))):
            raise ValueError(f"classify_cell: non-finite {name} {v}")
    if all(v < FLAT_RANGE for v in rr) and float(tau_set_upper) < float(tau_flat):
        return "flat"
    if sum(v >= MEANINGFUL_RANGE for v in rr) >= MEANINGFUL_MIN_HORIZONS and float(rr_lo_T200) > RR_LO_MIN:
        return "meaningful"
    return "moderate"


def boot_regret_ranges(boot_kgrid: pd.DataFrame, *, expected_n_boot: int, horizon: int) -> dict[str, np.ndarray]:
    """``{pool: per-replicate regret range (max - min over K), replicates 0 .. B-1}`` at `horizon`.
    Every pool must hold exactly replicates ``0 .. expected_n_boot - 1``, each on the same K-grid, one
    row per (replicate, K)."""
    sub = boot_kgrid[boot_kgrid["horizon"] == horizon]
    if "variant" in sub.columns:
        sub = sub[sub["variant"] == "npmle"]
    if sub.empty:
        raise ValueError(f"boot K-grid has no rows at T={horizon}")
    out: dict[str, np.ndarray] = {}
    for pool, g in sub.groupby("pool"):
        present, want = {int(b) for b in g["boot"]}, set(range(expected_n_boot))
        if present != want:
            raise ValueError(f"boot K-grid, pool {pool}, T={horizon}: missing "
                             f"{[f'b{b:03d}' for b in sorted(want - present)]}, "
                             f"unexpected {[f'b{b:03d}' for b in sorted(present - want)]}")
        if g.duplicated(["boot", "K"]).any():
            raise ValueError(f"boot K-grid, pool {pool}, T={horizon}: duplicate (replicate, K) rows")
        grids = {b: frozenset(int(k) for k in gb["K"]) for b, gb in g.groupby("boot")}
        if len(set(grids.values())) != 1:
            short = sorted(f"b{int(b):03d}" for b, ks in grids.items() if ks != max(grids.values(), key=len))
            raise ValueError(f"boot K-grid, pool {pool}, T={horizon}: replicates {short} do not share one K grid")
        ranges = g.groupby("boot")["regret"].agg(lambda r: float(r.max() - r.min())).sort_index()
        out[str(pool)] = ranges.to_numpy(dtype=float)
    return out


def boot_regret_range_ci(boot_kgrid: pd.DataFrame, *, expected_n_boot: int,
                         horizons: tuple[int, ...] = PRIMARY_HORIZONS) -> dict[tuple[str, int], tuple[float, float]]:
    """``{(pool, T): (2.5th, 97.5th percentile of the per-replicate regret range)}`` at every T (section 6.4)."""
    out: dict[tuple[str, int], tuple[float, float]] = {}
    for T in horizons:
        for pool, r in boot_regret_ranges(boot_kgrid, expected_n_boot=expected_n_boot, horizon=T).items():
            out[(pool, int(T))] = (float(np.percentile(r, RR_LO_PERCENTILE)), float(np.percentile(r, RR_HI_PERCENTILE)))
    return out


def boot_regret_range_lo(boot_kgrid: pd.DataFrame, *, expected_n_boot: int, horizon: int = RR_LO_HORIZON,
                         percentile: float = RR_LO_PERCENTILE) -> dict[str, float]:
    """``{pool: 2.5th percentile over bootstrap replicates of the regret range}`` at `horizon` (section 6.5
    classifies on T = 200's)."""
    return {pool: float(np.percentile(r, percentile))
            for pool, r in boot_regret_ranges(boot_kgrid, expected_n_boot=expected_n_boot, horizon=horizon).items()}


def classify_pools(components: pd.DataFrame, kgrid: pd.DataFrame, boot_kgrid: pd.DataFrame, *, tau_flat: float,
                   expected_n_boot: int, manifest_sha256: str) -> pd.DataFrame:
    """One section 6.5 row per pool of `components`: the point regret range at each primary T (the npmle
    K-grid, `describe.k_star_table`'s max - min over K) with its prompt-bootstrap 95% interval
    (``rr_lo_T<T>`` / ``rr_hi_T<T>``, section 6.4), tau_set's upper bound, the class. Classification
    reads only ``rr_lo_T200``."""
    table = describe.k_star_table(kgrid[kgrid["variant"] == "npmle"])
    rr_of = {(str(r.pool), int(r.horizon)): float(r.regret_range) for r in table.itertuples()}
    ci = boot_regret_range_ci(boot_kgrid, expected_n_boot=expected_n_boot)
    rows = []
    for r in components.itertuples():
        pool = str(r.pool)
        missing = [T for T in PRIMARY_HORIZONS if (pool, T) not in rr_of]
        missing_boot = [T for T in PRIMARY_HORIZONS if (pool, T) not in ci]
        if missing or missing_boot:
            raise ValueError(f"classify {pool}: K-grid lacks T={missing}; boot K-grid lacks T={missing_boot}")
        rr = {T: rr_of[(pool, T)] for T in PRIMARY_HORIZONS}
        up = float(r.tau_set_upper_one_sided)
        row = {"pool": pool}
        for T in PRIMARY_HORIZONS:
            row.update({f"regret_range_T{T}": rr[T], f"rr_lo_T{T}": ci[(pool, T)][0], f"rr_hi_T{T}": ci[(pool, T)][1]})
        row.update({"tau_set_upper_one_sided": up, "tau_flat": float(tau_flat),
                    "class": classify_cell(rr, ci[(pool, RR_LO_HORIZON)][0], up, tau_flat),
                    "n_boot": int(expected_n_boot), "manifest_sha256": manifest_sha256})
        rows.append(row)
    return pd.DataFrame(rows)


# ---- H1 / H2 / H4 ------------------------------------------------------------------------------


def h_verdict(delta_draws: np.ndarray, *, refute_below: float, estimate: float | None = None) -> dict:
    """Section 6.3's rule on bootstrap draws of a contrast: one-sided 95% lower bound = 5th percentile,
    upper bound = 95th; ``supported`` iff lo > 0, else ``refuted`` iff hi < `refute_below`, else
    ``inconclusive``. Both conditions can hold (a positive difference smaller than `refute_below`): the
    verdict is then ``supported`` (controller ruling) and ``effect_below_threshold`` is True -- the flag is
    True exactly when lo > 0 and hi < `refute_below`."""
    d = np.asarray(delta_draws, dtype=float)
    if d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("h_verdict needs finite bootstrap draws")
    lo, hi = float(np.percentile(d, 5)), float(np.percentile(d, 95))
    refute = hi < refute_below
    verdict = "supported" if lo > 0.0 else ("refuted" if refute else "inconclusive")
    return {"estimate": float(np.median(d)) if estimate is None else float(estimate), "lo": lo, "hi": hi,
            "verdict": verdict, "effect_below_threshold": bool(lo > 0.0 and refute), "n_boot": int(d.size)}


def _tau(Y: np.ndarray, row_counts: np.ndarray) -> float:
    """tau_main with `variance_components`' imputation correction (row_counts < J marks imputed cells)."""
    J = Y.shape[1]
    rcs = np.asarray(row_counts, dtype=float)
    return het.variance_components(Y, n_imputed=int(round(float(np.sum(J - rcs)))), row_counts=rcs)["tau"]


Universe = Mapping[str, tuple[list[str], list[str]]]


def pool_arm_ids(pool: str, *, study: st.Study, pool_file: Path | None = None) -> list[str]:
    """Every arm id of `pool`'s pool file: ``<data_dir>/pools/<pool>.yaml``, or -- for a relabelled pool
    (``study.extra_outcomes``) -- its source file (`EXTRA_POOL_FILES`) with ids relabelled exactly as
    `study.load_study_outcomes` relabels them (``<relabel>_<arm_id>``)."""
    relabels = {relabel: source for _, source, relabel in study.extra_outcomes}
    if pool_file is None:
        pool_file = EXTRA_POOL_FILES[pool] if pool in relabels else study.data_dir / "pools" / f"{pool}.yaml"
    with open(pool_file, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    ids = [str(a["arm_id"]) for a in doc["arms"]]
    if pool in relabels:
        ids = [f"{pool}_{a}" for a in ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{pool_file}: {len(ids)} arm ids, {len(set(ids))} distinct")
    return ids


def design_universe(study: st.Study = st.HETEROGENEITY, *, manifest: Mapping | None = None) -> dict[str, tuple[list[str], list[str]]]:
    """``{pool: (arm ids, task ids)}``: the design each cell's matrix is built on -- every arm of the pool
    file and the app's task subset from ``<data_dir>/manifest.json`` (`TASK_SUBSET_KEY`), independent of
    what the outcomes happen to contain."""
    if manifest is None:
        manifest = json.loads((study.data_dir / "manifest.json").read_text())
    out = {}
    for pool in study.pools:
        tasks = [str(t) for t in manifest[TASK_SUBSET_KEY[POOL_APP[pool]]]]
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError(f"manifest {TASK_SUBSET_KEY[POOL_APP[pool]]}: {len(tasks)} ids, {len(set(tasks))} distinct")
        out[pool] = (pool_arm_ids(pool, study=study), tasks)
    return out


def _matrix(outcomes: pd.DataFrame, pool: str, universe: Universe, **kw):
    """`het.success_matrix` on the design's full (arm x task) universe (`design_universe`: the pool file's
    arms, the app's task subset from the manifest), so an arm or task with no ``ok`` episode -- never
    attempted, or ``missing`` -- is a missing cell that counts toward the 5% ceiling (and a fully
    missing arm or task raises) instead of silently shrinking the matrix."""
    if pool not in universe:
        raise KeyError(f"no design universe for pool {pool}")
    arms, tasks = universe[pool]
    return het.success_matrix(outcomes, pool, expected_arms=list(arms), expected_tasks=list(tasks), **kw)


def _paired(Ya, rca, Yb, rcb, *, n_boot: int, seed: int, refute_below: float) -> dict:
    """tau(a) - tau(b) over row-aligned prompts, the same (n_boot, I) row indices in both cells."""
    if Ya.shape[0] != Yb.shape[0]:
        raise ValueError(f"paired contrast needs aligned prompts: {Ya.shape[0]} vs {Yb.shape[0]}")
    n = Ya.shape[0]
    R = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    da = het.prompt_bootstrap(Ya, _tau, rows=R, row_counts=rca)
    db = het.prompt_bootstrap(Yb, _tau, rows=R, row_counts=rcb)
    ta, tb = _tau(Ya, rca), _tau(Yb, rcb)
    out = h_verdict(da - db, refute_below=refute_below, estimate=ta - tb)
    return {**out, "tau_a": ta, "tau_b": tb, "draws": da - db, "n_prompts": n, "seed": int(seed)}


def h1(outcomes_gitlab_GLG: pd.DataFrame, outcomes_bridge: pd.DataFrame, bridge_arm_map: Mapping[str, str], *,
       n_boot: int, seed: int, universe: Universe, gitlab_pool: str = "GLG", bridge_pool: str = "GMB") -> dict:
    """H1 primary: tau_main(GitLab, the 20 bridge prompts) - tau_main(Gmail at 600 s, the same 20), paired by
    prompt through `bridge_arm_map` (``manifest.json["bridge_source"]``: GMB id -> GLG id); the 20 prompts
    are resampled jointly in both apps, each app's task set held fixed."""
    Ygl, arms_gl, _, _, rc_gl = _matrix(outcomes_gitlab_GLG, gitlab_pool, universe)
    Ygm, arms_gm, _, _, rc_gm = _matrix(outcomes_bridge, bridge_pool, universe)
    if set(bridge_arm_map) != set(arms_gm):
        raise ValueError(f"bridge map keys do not match the bridge arms: missing "
                         f"{sorted(set(arms_gm) - set(bridge_arm_map))}, extra {sorted(set(bridge_arm_map) - set(arms_gm))}")
    targets = [str(bridge_arm_map[b]) for b in arms_gm]
    absent = sorted(t for t in targets if t not in arms_gl)
    if absent or len(set(targets)) != len(targets):
        raise ValueError(f"bridge map targets not among the {gitlab_pool} arms: {absent} "
                         f"(or not distinct: {len(set(targets))} of {len(targets)})")
    order = [arms_gl.index(t) for t in targets]
    out = _paired(Ygl[order], rc_gl[order], Ygm, rc_gm, n_boot=n_boot, seed=seed, refute_below=H_REFUTE_BELOW)
    return {**out, "tau_gitlab": out["tau_a"], "tau_gmail": out["tau_b"],
            "pairs": list(zip(arms_gm, targets, strict=True))}


def pair_by_g_index(gmg_arms: Iterable[str], glg_arms: Iterable[str]) -> list[tuple[str, str]]:
    """``[(GMG_G_xx, GLG_xx)]`` by original G index (GLG_i is ``pool_G.yaml``'s i-th arm, G_i), sorted."""
    def index(arms, pattern):
        out = {}
        for a in arms:
            m = pattern.match(str(a))
            if m is None:
                raise ValueError(f"arm id {a!r} does not match {pattern.pattern}")
            out[int(m.group(1))] = str(a)
        return out
    gm, gl = index(gmg_arms, _GMG_ARM), index(glg_arms, _GLG_ARM)
    if set(gm) != set(gl):
        raise ValueError(f"G indices differ: Gmail only {sorted(set(gm) - set(gl))}, GitLab only {sorted(set(gl) - set(gm))}")
    return [(gm[i], gl[i]) for i in sorted(gm)]


def h1_secondary(outcomes_gitlab: pd.DataFrame, outcomes_gmail: pd.DataFrame, *, n_boot: int, seed: int,
                 universe: Universe, gitlab_pool: str = "GLG", gmail_pool: str = "GMG") -> dict:
    """H1 secondary: all 50 G prompts, tau_main(GitLab) - tau_main(Gmail at 180 s), paired by G index."""
    Ygl, arms_gl, _, _, rc_gl = _matrix(outcomes_gitlab, gitlab_pool, universe)
    Ygm, arms_gm, _, _, rc_gm = _matrix(outcomes_gmail, gmail_pool, universe)
    pairs = pair_by_g_index(arms_gm, arms_gl)
    ogl = [arms_gl.index(b) for _, b in pairs]
    ogm = [arms_gm.index(a) for a, _ in pairs]
    out = _paired(Ygl[ogl], rc_gl[ogl], Ygm[ogm], rc_gm[ogm], n_boot=n_boot, seed=seed, refute_below=H_REFUTE_BELOW)
    return {**out, "tau_gitlab": out["tau_a"], "tau_gmail": out["tau_b"], "pairs": pairs}


def h2(Y_K: np.ndarray, Y_G: np.ndarray, *, n_boot: int, seed: int, row_counts_K: np.ndarray | None = None,
       row_counts_G: np.ndarray | None = None, task_ids_K: list[str] | None = None,
       task_ids_G: list[str] | None = None) -> dict:
    """H2 in one app: tau_main(K) - tau_main(G) on the same task set (same column order), each pool's prompts
    resampled independently (K's indices drawn first, then G's, from one generator)."""
    YK, YG = np.asarray(Y_K, dtype=float), np.asarray(Y_G, dtype=float)
    if YK.shape[1] != YG.shape[1]:
        raise ValueError(f"H2 needs the same task set: {YK.shape[1]} vs {YG.shape[1]} tasks")
    if task_ids_K is not None and task_ids_G is not None and list(task_ids_K) != list(task_ids_G):
        raise ValueError("H2 needs the same task set in the same order: task ids differ")
    J = YK.shape[1]
    rcK = np.full(YK.shape[0], J) if row_counts_K is None else np.asarray(row_counts_K)
    rcG = np.full(YG.shape[0], J) if row_counts_G is None else np.asarray(row_counts_G)
    g = np.random.default_rng(seed)
    RK = g.integers(0, YK.shape[0], (n_boot, YK.shape[0]))
    RG = g.integers(0, YG.shape[0], (n_boot, YG.shape[0]))
    draws = (het.prompt_bootstrap(YK, _tau, rows=RK, row_counts=rcK)
             - het.prompt_bootstrap(YG, _tau, rows=RG, row_counts=rcG))
    tk, tg = _tau(YK, rcK), _tau(YG, rcG)
    out = h_verdict(draws, refute_below=H_REFUTE_BELOW, estimate=tk - tg)
    return {**out, "tau_K": tk, "tau_G": tg, "draws": draws, "seed": int(seed)}


def h4(effects_gmail: np.ndarray, effects_gitlab: np.ndarray, *, n_boot: int, seed: int) -> dict:
    """H4 (reported, not decided): Spearman rho between the G prompts' main effects in the two apps, with a
    95% percentile interval from resampling prompts (pairs) with replacement."""
    a, b = np.asarray(effects_gmail, dtype=float), np.asarray(effects_gitlab, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"H4 needs paired effect vectors, got {a.shape} and {b.shape}")
    X = np.column_stack([a, b])

    def rho(Xb: np.ndarray) -> float:
        return float(sps.spearmanr(Xb[:, 0], Xb[:, 1]).statistic)

    draws = het.prompt_bootstrap(X, rho, n_boot=n_boot, seed=seed)
    ok = draws[np.isfinite(draws)]
    if not np.isfinite(rho(X)) or ok.size == 0:
        raise ValueError("H4: Spearman rho is undefined (a constant effect vector)")
    return {"estimate": rho(X), "lo": float(np.percentile(ok, 2.5)), "hi": float(np.percentile(ok, 97.5)),
            "verdict": "reported", "n_boot": int(n_boot), "n_undefined": int(draws.size - ok.size),
            "n_prompts": int(a.size), "draws": draws, "seed": int(seed)}


# ---- H3 ------------------------------------------------------------------------------------------


def _contrast_name(registration: str) -> str:
    return str(registration).removesuffix("_flat")


def _shows_difference(row: Mapping, mei: float) -> bool:
    """A policy difference > MEI with a bootstrap interval excluding 0 (on delta/lo/hi when the row has
    them; otherwise, for the scale/spread contrasts, a ``supported`` or ``reversed`` verdict)."""
    vals = [row.get(k) for k in ("delta", "lo", "hi")]
    if all(v is not None and np.isfinite(float(v)) for v in vals):
        d, lo, hi = (float(v) for v in vals)
        return abs(d) > mei and (lo > 0.0 or hi < 0.0)
    return _contrast_name(row["registration"]) in (H3_SCALE, H3_SPREAD) and row["verdict"] in ("supported", "reversed")


def h3(classes: Mapping[str, str], contrasts: pd.DataFrame, mei: float = MEI) -> dict:
    """Section 6.6. `contrasts` rows: ``cells`` (one pool, or a class -- the pooled registered row),
    ``registration`` (``het_scale``, ``het_spread``, ... with or without ``_flat``), ``verdict`` and,
    when available, ``delta``/``lo``/``hi``.

    Only per-cell rows (``cells`` = a pool) decide -- the spec reads "a flat cell" / "a meaningful cell";
    pooled class rows are listed under ``supplementary`` and never enter the verdict (fix round 1).

    * **refuted** iff a flat cell shows a policy difference > MEI with an interval excluding 0 (any
      registration), or a meaningful cell shows contrast 1 or 2 ``reversed``. Refuted takes precedence
      over everything, including untestable (controller ruling);
    * **untestable** iff no cell is meaningful (and nothing refutes);
    * **supported** iff some meaningful cell has contrasts 1 and 2 both ``supported`` in its own rows.
      Condition (a), flat cells' regret range < 0.005 at every primary T, holds by the section 6.5
      definition of flat;
    * **inconclusive** otherwise.
    """
    meaningful = sorted(p for p, c in classes.items() if c == "meaningful")
    flat = sorted(p for p, c in classes.items() if c == "flat")
    records = contrasts.to_dict("records")
    refuting: list[str] = []
    supplementary: list[str] = []
    for r in records:
        sel = str(r["cells"])
        flat_hit = (sel == "flat" or sel in flat) and _shows_difference(r, mei)
        reversed_hit = ((sel == "meaningful" or sel in meaningful)
                        and _contrast_name(r["registration"]) in (H3_SCALE, H3_SPREAD) and r["verdict"] == "reversed")
        if not (flat_hit or reversed_hit):
            continue
        what = (f"flat {sel}: {r['registration']} differs beyond MEI ({r['verdict']})" if flat_hit
                else f"meaningful {sel}: {r['registration']} reversed")
        (supplementary if sel in rc.CELL_CLASSES else refuting).append(what)
    supporting: list[str] = []
    for p in meaningful:
        v = {_contrast_name(r["registration"]): r["verdict"] for r in records if str(r["cells"]) == p}
        if v.get(H3_SCALE) == "supported" and v.get(H3_SPREAD) == "supported":
            supporting.append(p)
    if refuting:
        verdict = "refuted"
    elif not meaningful:
        verdict = "untestable"
    elif supporting:
        verdict = "supported"
    else:
        verdict = "inconclusive"
    detail = (f"meaningful={meaningful}; flat={flat}; supporting={supporting}; refuting={refuting}; "
              f"supplementary (pooled rows, not decisive)={supplementary}")
    return {"verdict": verdict, "meaningful_cells": meaningful, "flat_cells": flat, "supporting_cells": supporting,
            "refuting": refuting, "supplementary": supplementary, "detail": detail}


def policy_gaps(out_dir: Path, classes: Mapping[str, str], classification_path: Path, *,
                registrations: Iterable[str] | None = None, expected_n_boot: int | None = None,
                reservoir_manifest: Path | None = None, study: st.Study = st.HETEROGENEITY) -> pd.DataFrame:
    """Every ``het_*`` registration on its class pooled (the registered row) and on each cell of that class."""
    rows = []
    for name in registrations or HET_REGISTRATIONS:
        reg = rc.REGISTRATIONS[name]
        for sel in [reg["cells"], *sorted(p for p, c in classes.items() if c == reg["cells"])]:
            out = rc.prompt_bootstrap_contrast(
                reg["policy"], reg["reference"], horizons=tuple(reg["horizons"]), mei=reg["mei"], rule=reg["rule"],
                out_dir=out_dir, expected_n_boot=expected_n_boot or reg["n_boot"],
                reservoir_manifest=reservoir_manifest, study=study, cells=sel,
                classification_path=classification_path).iloc[0].to_dict()
            rows.append({"registration": name, "contrast": _contrast_name(name), **out})
    frame = pd.DataFrame(rows)
    return frame[["registration", "contrast", "cells", *[c for c in frame.columns
                                                         if c not in ("registration", "contrast", "cells")]]]


# ---- per-cell analysis, noise, timeouts --------------------------------------------------------


def cell_noise(outcomes: pd.DataFrame, pool: str, *, study: st.Study) -> tuple[float, int, str]:
    """``(v, n_pairs, source pool)``: the pool's own replicate pairs, else its declared borrow, else (nan, 0, "")."""
    v, n = het.noise_from_pairs(outcomes, pool)
    if n >= 1 and np.isfinite(v):
        return v, n, pool
    donor = study.borrowed_noise.get(pool)
    if donor is not None:
        v, n = het.noise_from_pairs(outcomes, donor)
        if n >= 1 and np.isfinite(v):
            return v, n, donor
    return float("nan"), 0, ""


def _analyze(Y, tasks, n_imp, row_counts, noise, seed) -> dict:
    v, n, _ = noise
    return stage0.analyze_cell(Y, tasks, noise_var=v if n >= 1 else None, noise_df=n if n >= 1 else None,
                               n_imputed=n_imp, row_counts=row_counts, seed=seed)


def upper_tail_mass(Y: np.ndarray, row_counts: np.ndarray) -> float:
    """Section 6.4's upper-tail mass, exactly as `stage0.analyze_cell` computes it: an NPMLE on the prompt
    means with per-prompt variance MS_resid / J, and its mass at or above (weighted median + 0.10)."""
    Y = np.asarray(Y, dtype=float)
    J = Y.shape[1]
    rcs = np.asarray(row_counts, dtype=float)
    vc = het.variance_components(Y, n_imputed=int(round(float(np.sum(J - rcs)))), row_counts=rcs)
    grid, weights, _ = emp.npmle(Y.mean(axis=1), np.full(Y.shape[0], vc["ms_resid"] / J))
    median = stage0._weighted_median(grid, weights)
    return float(weights[grid >= median + UPPER_TAIL_DELTA].sum())


def upper_tail_ci(Y: np.ndarray, row_counts: np.ndarray, *, n_boot: int = TAIL_N_BOOT, seed: int) -> tuple[float, float]:
    """95% percentile interval of the upper-tail mass over a prompt bootstrap (tasks fixed, NPMLE re-fit
    on every resample)."""
    draws = het.prompt_bootstrap(Y, upper_tail_mass, n_boot=n_boot, seed=seed, row_counts=row_counts)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyze_pools(outcomes: pd.DataFrame, *, study: st.Study, universe: Universe, seed: int = SPLIT_HALF_SEED,
                  tail_n_boot: int = TAIL_N_BOOT, tail_seed: int = VERDICT_SEED) -> pd.DataFrame:
    """`stage0.analyze_cell` on every pool of `study` (its design universe; its noise or declared borrow), plus
    the upper-tail mass's prompt-bootstrap CI (``upper_tail_mass_lo``/``_hi``; seed ``tail_seed +
    TAIL_SEED_OFFSET + pool index``)."""
    rows = []
    for k, pool in enumerate(study.pools):
        Y, _, tasks, n_imp, row_counts = _matrix(outcomes, pool, universe)
        noise = cell_noise(outcomes, pool, study=study)
        out = _analyze(Y, tasks, n_imp, row_counts, noise, seed)
        tseed = int(tail_seed) + TAIL_SEED_OFFSET + k
        lo, hi = upper_tail_ci(Y, row_counts, n_boot=tail_n_boot, seed=tseed)
        rows.append({"pool": pool, **out, "upper_tail_mass_lo": lo, "upper_tail_mass_hi": hi,
                     "upper_tail_n_boot": int(tail_n_boot), "upper_tail_seed": tseed, "n_imputed": int(n_imp),
                     "missing_frac": n_imp / Y.size, "noise_df": int(noise[1]), "noise_from": noise[2]})
    return pd.DataFrame(rows)


def attach_ended_by(snapshot: pd.DataFrame, *, log_dirs: Iterable[Path],
                    extra_log_dirs: Mapping[str, tuple[Path, str]]) -> pd.DataFrame:
    """The snapshot (which does not carry it) plus each terminal episode's ``ended_by`` from the logs.
    `extra_log_dirs` ``{relabel: (log dir, source pool)}`` reads relabelled pools' own logs (GMG: Pre-reg 9's
    G), whose records predate ``ended_by``: ``timed_out`` maps to ``clock`` / ``not_clock``. A key whose
    log attempt differs from the snapshot's raises."""
    def ended(dir_: Path) -> pd.DataFrame | None:
        files = sorted(Path(dir_).glob("worker_*.jsonl")) if Path(dir_).is_dir() else []
        if not files:
            return None
        term = emp.terminal_outcomes(emp.load_attempts(files))
        eb = term["ended_by"] if "ended_by" in term.columns else pd.Series([None] * len(term), index=term.index)
        if "timed_out" in term.columns:
            fallback = term["timed_out"].map(lambda t: None if t is None or pd.isna(t) else ("clock" if t else "not_clock"))
            eb = eb.where(eb.notna(), fallback)
        return term.assign(ended_by=eb)[["pool", "arm_id", "task_id", "replicate", "attempt", "ended_by"]]

    parts = [f for f in (ended(d) for d in log_dirs) if f is not None]
    for relabel, (dir_, source) in extra_log_dirs.items():
        f = ended(dir_)
        if f is None:
            continue
        f = f[f["pool"] == source].copy()
        f["arm_id"] = [f"{relabel}_{a}" for a in f["arm_id"]]
        f["pool"] = relabel
        parts.append(f)
    keys = ["pool", "arm_id", "task_id", "replicate"]
    out = snapshot.drop(columns=["ended_by"], errors="ignore")
    if not parts:
        return out.assign(ended_by=None)
    logs = pd.concat(parts, ignore_index=True).rename(columns={"attempt": "_log_attempt"})
    logs["replicate"] = logs["replicate"].astype(int)
    merged = out.merge(logs, on=keys, how="left", validate="one_to_one")
    bad = merged["_log_attempt"].notna() & (merged["_log_attempt"].astype(float) != merged["attempt"].astype(float))
    if bad.any():
        raise ValueError(f"log attempts differ from the snapshot's for {int(bad.sum())} episodes, e.g. "
                         f"{merged.loc[bad, keys].head(3).to_dict('records')}")
    return merged.drop(columns=["_log_attempt"])


def timeout_sensitivity(outcomes: pd.DataFrame, *, study: st.Study, universe: Universe,
                        seed: int = SPLIT_HALF_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 6.8: per cell, the analysis with clock-ended episodes set to missing (every pool's, so a
    borrowed noise estimate drops them too), imputed additively and counted beside the primary tau; plus
    per-prompt timeout rates on the replicate-0 ``ok`` episodes.

    A prompt whose replicate-0 episodes are all clock-ended has no observation left: it is dropped from the
    sensitivity matrix and counted in ``n_prompts_dropped`` (fix round 1). The primary analysis's 5%
    ceiling does not apply to this declared sensitivity (clock-ended cells are missing by construction);
    `het.success_matrix`'s structural rule does -- if the matrix still has a task with no observed cell,
    the cell is NA with the reason in ``note``. A cell with any replicate-0 ``ok`` episode whose
    ``ended_by`` (or Pre-reg 9 ``timed_out``) is unknown is NA, never imputed."""
    has = "ended_by" in outcomes.columns
    dropped = outcomes[outcomes["ended_by"].ne("clock")] if has else outcomes
    rows, rates = [], []
    for pool in study.pools:
        sub0 = outcomes[(outcomes["pool"] == pool) & (outcomes["replicate"] == 0) & (outcomes["status"] == het.STATUS_OK)]
        n_unknown = int(sub0["ended_by"].isna().sum()) if has else len(sub0)
        if n_unknown:
            rows.append({"pool": pool, "available": False, "n_episodes": len(sub0),
                         "note": f"no ended_by / timed_out in the logs for {n_unknown} of {len(sub0)} replicate-0 "
                                 "ok episodes (logs or field absent); not imputed"})
            continue
        clock = sub0["ended_by"] == "clock"
        all_clock = []
        for arm, g in sub0.assign(clock=clock).groupby("arm_id"):
            rates.append({"pool": pool, "arm_id": str(arm), "n_episodes": len(g), "n_clock": int(g["clock"].sum()),
                          "timeout_rate": float(g["clock"].mean())})
            if bool(g["clock"].all()):
                all_clock.append(str(arm))
        Y, arms, tasks, n_imp, row_counts = _matrix(outcomes, pool, universe)
        base = _analyze(Y, tasks, n_imp, row_counts, cell_noise(outcomes, pool, study=study), seed)
        kept = [a for a in arms if a not in set(all_clock)]
        common = {"pool": pool, "n_episodes": len(sub0), "n_clock": int(clock.sum()),
                  "clock_rate": float(clock.mean()), "n_prompts_dropped": len(all_clock),
                  "dropped_prompts": ";".join(sorted(all_clock))}
        try:
            Ys, _, _, n_imp_s, rc_s = het.success_matrix(dropped, pool, expected_arms=kept, expected_tasks=tasks,
                                                         max_missing=None)
        except ValueError as exc:
            rows.append({**common, "available": False, "note": str(exc)})
            continue
        noise = cell_noise(dropped, pool, study=study)
        sens = _analyze(Ys, tasks, n_imp_s, rc_s, noise, seed)
        rows.append({**common, "available": True, "n_prompts": int(Ys.shape[0]), "n_imputed": int(n_imp_s),
                     "missing_frac": n_imp_s / Ys.size,
                     "tau_main": sens["tau_main"], "tau_lo": sens["tau_lo"], "tau_hi": sens["tau_hi"],
                     "tau_set": sens["tau_set"], "tau_set_upper_one_sided": sens["tau_set_upper_one_sided"],
                     "r_sb": sens["r_sb"], "tau_main_baseline": base["tau_main"], "tau_set_baseline": base["tau_set"],
                     "noise_var": noise[0], "noise_df": int(noise[1]), "noise_from": noise[2], "note": ""})
    return pd.DataFrame(rows), pd.DataFrame(rates, columns=["pool", "arm_id", "n_episodes", "n_clock", "timeout_rate"])


# ---- hypotheses -------------------------------------------------------------------------------


def _hrow(hypothesis, contrast, role, decides, res, interval) -> dict:
    return {"hypothesis": hypothesis, "contrast": contrast, "role": role, "decides": decides,
            "estimate": res.get("estimate"), "lo": res.get("lo"), "hi": res.get("hi"), "interval": interval,
            "verdict": res.get("verdict"), "effect_below_threshold": res.get("effect_below_threshold"),
            "n_prompts": res.get("n_prompts"), "n_boot": res.get("n_boot"), "seed": res.get("seed"),
            "detail": res.get("detail", "")}


def hypotheses(outcomes: pd.DataFrame, bridge_arm_map: Mapping[str, str], *, universe: Universe,
               n_boot: int = N_BOOT, seed: int = VERDICT_SEED) -> tuple[list[dict], pd.DataFrame]:
    """H1 (primary + secondary), H2 (Gmail, GitLab) and H4 rows, and the H4 portability table."""
    one = "one-sided 95% (5th / 95th percentile)"
    rows = []
    r = h1(outcomes, outcomes, bridge_arm_map, n_boot=n_boot, seed=seed + 1, universe=universe)
    r["detail"] = f"tau GLG(bridge 20)={r['tau_gitlab']:.4f}, tau GMB={r['tau_gmail']:.4f}"
    rows.append(_hrow("H1", "tau(GLG, bridge prompts) - tau(GMB)", "primary", True, r, one))
    r = h1_secondary(outcomes, outcomes, n_boot=n_boot, seed=seed + 2, universe=universe)
    r["detail"] = f"tau GLG={r['tau_gitlab']:.4f}, tau GMG={r['tau_gmail']:.4f}; secondary, not decided"
    rows.append(_hrow("H1", "tau(GLG) - tau(GMG, 180 s)", "secondary", False, r, one))
    for k, (pk, pg, env) in enumerate((("GMK", "GMG", "Gmail"), ("GLK", "GLG", "GitLab"))):
        YK, _, tK, _, rcK = _matrix(outcomes, pk, universe)
        YG, _, tG, _, rcG = _matrix(outcomes, pg, universe)
        r = h2(YK, YG, n_boot=n_boot, seed=seed + 3 + k, row_counts_K=rcK, row_counts_G=rcG,
               task_ids_K=tK, task_ids_G=tG)
        r["n_prompts"] = f"{YK.shape[0]}+{YG.shape[0]}"
        r["detail"] = f"tau {pk}={r['tau_K']:.4f}, tau {pg}={r['tau_G']:.4f}"
        rows.append(_hrow("H2", f"tau({pk}) - tau({pg})", env, True, r, one))
    Ygm, arms_gm, _, _, _ = _matrix(outcomes, "GMG", universe)
    Ygl, arms_gl, _, _, _ = _matrix(outcomes, "GLG", universe)
    pairs = pair_by_g_index(arms_gm, arms_gl)
    # Spearman is invariant to the BLUPs' common positive shrinkage factor, so rho on the unshrunk main
    # effects (row mean - grand mean) equals rho on the BLUPs whenever tau2 > 0, and stays defined when a
    # cell's tau2 truncates to 0 (every BLUP 0). The table reports both.
    raw_gm = dict(zip(arms_gm, Ygm.mean(axis=1) - Ygm.mean(), strict=True))
    raw_gl = dict(zip(arms_gl, Ygl.mean(axis=1) - Ygl.mean(), strict=True))
    eff_gm = dict(zip(arms_gm, het.prompt_effects(Ygm), strict=True))
    eff_gl = dict(zip(arms_gl, het.prompt_effects(Ygl), strict=True))
    raw_a = np.array([raw_gm[g] for g, _ in pairs])
    raw_b = np.array([raw_gl[gl] for _, gl in pairs])
    a = np.array([eff_gm[g] for g, _ in pairs])
    b = np.array([eff_gl[gl] for _, gl in pairs])
    r = h4(raw_a, raw_b, n_boot=n_boot, seed=seed + 5)
    r["detail"] = f"{r['n_undefined']} undefined bootstrap draws" if r["n_undefined"] else ""
    rows.append(_hrow("H4", "spearman(effect GMG, effect GLG)", "reported", False, r,
                      "two-sided 95% (2.5th / 97.5th percentile)"))
    port = pd.DataFrame({"g_index": [int(_GLG_ARM.match(gl).group(1)) for _, gl in pairs],
                         "arm_gmail": [g for g, _ in pairs], "arm_gitlab": [gl for _, gl in pairs],
                         "effect_gmail": a, "effect_gitlab": b, "raw_effect_gmail": raw_a, "raw_effect_gitlab": raw_b,
                         "rank_gmail": sps.rankdata(raw_a), "rank_gitlab": sps.rankdata(raw_b)})
    return rows, port


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau-flat", type=float, required=True,
                    help="section 6.5's tau_flat, as fixed in Pre-registration 10 from the calibration")
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR, help="the replay tree (tables/, episodes/)")
    ap.add_argument("--results-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="prompt bootstrap for H1, H2, H4")
    ap.add_argument("--seed", type=int, default=VERDICT_SEED)
    ap.add_argument("--prereg9-log-dir", type=Path, default=replay.LOG_DIR,
                    help="Pre-registration 9's logs, for GMG's timeout flags")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    study = replay.resolve_study(st.HETEROGENEITY.name)
    res_dir, results = study.res_dir, args.results_dir
    manifest = res_dir / replay.MANIFEST_FILE
    manifest_sha = rc.file_sha256(manifest)
    outcomes = replay.load_snapshot(res_dir)
    tables = args.out_dir / "tables"
    kgrid = pd.read_csv(tables / study.table("kgrid"))
    describe.check_kgrid_snapshot(kgrid, replay.verify_reservoirs(res_dir, study=study), study=study)
    boot_kgrid = pd.read_csv(tables / study.table("boot_kgrid"))
    if set(boot_kgrid["manifest_sha256"].astype(str)) != {manifest_sha}:
        raise ValueError(f"{study.table('boot_kgrid')} was not computed on {manifest}; re-run replay.py boot_kgrid")
    results.mkdir(parents=True, exist_ok=True)

    universe = design_universe(study)
    components = analyze_pools(outcomes, study=study, universe=universe, tail_seed=args.seed)
    components.to_csv(results / "het_components.csv", index=False)
    classification = classify_pools(components, kgrid, boot_kgrid, tau_flat=args.tau_flat,
                                    expected_n_boot=replay.N_BOOT, manifest_sha256=manifest_sha)
    cls_path = results / "het_classification.csv"
    classification.to_csv(cls_path, index=False)
    classes = dict(zip(classification["pool"], classification["class"], strict=True))
    log.info("classification: %s", classes)

    gaps = policy_gaps(args.out_dir, classes, cls_path, reservoir_manifest=manifest, study=study)
    gaps.to_csv(results / "het_policy_gaps.csv", index=False)

    bridge = json.loads((study.data_dir / "manifest.json").read_text())["bridge_source"]
    rows, port = hypotheses(outcomes, bridge, universe=universe, n_boot=args.n_boot, seed=args.seed)
    r3 = h3(classes, gaps)
    rows.append(_hrow("H3", "section 6.6", "thesis", True, r3, "prompt bootstrap, B = 200 (replay)"))
    pd.DataFrame(rows).to_csv(results / "het_hypotheses.csv", index=False)
    port.to_csv(results / "het_portability.csv", index=False)
    for r in rows:
        log.info("%s %-10s %-40s %s", r["hypothesis"], r["role"], r["contrast"], str(r["verdict"]).upper())

    extra = {relabel: (args.prereg9_log_dir, source) for _, source, relabel in study.extra_outcomes}
    timed = attach_ended_by(outcomes, log_dirs=study.log_dirs, extra_log_dirs=extra)
    sens, rates = timeout_sensitivity(timed, study=study, universe=universe)
    sens.to_csv(results / "het_timeout_sensitivity.csv", index=False)
    rates.to_csv(results / "het_timeout_rates.csv", index=False)
    log.info("wrote %s", results)


if __name__ == "__main__":
    main()
