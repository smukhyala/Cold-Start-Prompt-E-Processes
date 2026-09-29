"""Stage 0 -- free reanalyses (Pre-registration 10, section 6): per-cell heterogeneity on the existing
Gmail prompt pools and the project's old GitLab paired run. No data collection: both inputs already
exist on disk.

    .venv/bin/python experiments/growing_bandits/empirical/stage0.py

Gmail: reads the Pre-reg 9 outcomes snapshot (`replay.load_snapshot_unchecked(replay.RES_DIR)`) and
analyzes pools G and F (`replay.PREREG9.pools`), with execution noise per pool from its replicate pairs
(`heterogeneity.noise_from_pairs`). Writes `stage0_gmail.csv` (one row per pool).

GitLab: reads `results/gitlab_strong_arm/paired/paired_results.csv` (one run per (arm, task) cell, no
replicates -- `noise_var=None`, so interaction and execution noise are reported jointly as `ms_resid`).
Analyzes three arm subsets -- all 18 arms, all 18 without the `gitlab_oracle_operator` and `explorer`
anchors, and the 12 generic (non `gitlab_`-prefixed) arms -- since the anchors and the GitLab-specific
arms are not exchangeable draws from the same generic-prompt pool that `tau` characterizes. Writes
`stage0_gitlab_paired.csv` (one row per subset).

`analyze_cell` composes `heterogeneity`'s Task 2 functions: `tau_interval` (Graybill-Wang MLS 95% CI,
replacing the removed `two_way_bootstrap` -- see `heterogeneity.py`'s module docstring and
task-2-report.md fix round 1) for tau's interval, `split_half` for model-free reliability, and
`variance_components` restricted to `discriminating_tasks` for tau computed on the subset of tasks that
actually separate prompts (NaN if fewer than 2 such tasks -- `variance_components` needs at least 2
columns). The upper-tail mass is an NPMLE (`cold_start.growing.empirical.npmle`) on row (prompt) means
with per-row measurement variance `sigma_i^2 = MS_resid / J`: the share of posterior mass at or above
(weighted median + 0.10), i.e. prompts plausibly at least 0.10 better than the pool's typical prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import replay  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing import heterogeneity as het  # noqa: E402

OUT_DIR = ROOT / "results" / "growing_bandits" / "heterogeneity"
GITLAB_PAIRED_CSV = ROOT / "results" / "gitlab_strong_arm" / "paired" / "paired_results.csv"

#: GitLab arm subsets for `stage0_gitlab_paired.csv` (18 arms total; 6 are `gitlab_`-prefixed).
GITLAB_ANCHOR_EXCLUDE: tuple[str, ...] = ("gitlab_oracle_operator", "explorer")
GITLAB_SPECIFIC_ARMS: tuple[str, ...] = (
    "gitlab_auditor", "gitlab_domain_expert", "gitlab_oracle_operator",
    "gitlab_planner", "gitlab_rapid_executor", "gitlab_super_operator",
)
GITLAB_SUBSETS: dict[str, tuple[str, ...]] = {
    "all_18_arms": (),
    "without_oracle_and_explorer": GITLAB_ANCHOR_EXCLUDE,
    "generic_12_arms": GITLAB_SPECIFIC_ARMS,
}


def _weighted_median(grid: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(grid)
    g, w = grid[order], weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.5 * cum[-1]))
    idx = min(idx, g.size - 1)
    return float(g[idx])


def analyze_cell(
    Y: np.ndarray,
    task_ids: list[str],
    *,
    noise_var: float | None = None,
    n_imputed: int = 0,
    row_counts: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Variance components + tau's 95% MLS interval (`tau_interval`) + split-half reliability +
    discriminating-task tau + upper-tail NPMLE mass, for one (prompt x task) cell.

    Controller ruling (2026-09-28, task-3): tau's interval comes from `heterogeneity.tau_interval`
    (Graybill-Wang MLS), not a bootstrap -- `two_way_bootstrap` was removed for bias (task-2-report.md
    fix round 1); a single cell needs no `n_boot`.
    """
    Y = np.asarray(Y, dtype=float)
    I, J = Y.shape
    vc = het.variance_components(Y, noise_var=noise_var, n_imputed=n_imputed, row_counts=row_counts)
    iv = het.tau_interval(Y, n_imputed=n_imputed, row_counts=row_counts)
    sh = het.split_half(Y, task_ids, seed=seed)

    mask = het.discriminating_tasks(Y)
    n_discriminating = int(mask.sum())
    if n_discriminating >= 2:
        tau_discriminating = het.variance_components(Y[:, mask])["tau"]
    else:
        tau_discriminating = float("nan")

    row_means = Y.mean(axis=1)
    sigma2 = np.full(I, vc["ms_resid"] / J)
    grid, weights, _ = emp.npmle(row_means, sigma2)
    median = _weighted_median(grid, weights)
    upper_tail_mass = float(weights[grid >= median + 0.10].sum())

    return {
        "tau": iv["tau"],
        "tau_lo": iv["tau_lo"],
        "tau_hi": iv["tau_hi"],
        "tau_upper_one_sided": iv["tau_upper_one_sided"],
        "task_var": vc["task_var"],
        "interaction_var": vc["interaction_var"],
        "noise_var": vc["noise_var"],
        "r_sb": sh["r_sb"],
        "split_half_p": sh["p_value"],
        "n_discriminating_tasks": n_discriminating,
        "tau_discriminating": tau_discriminating,
        "upper_tail_mass": upper_tail_mass,
        "n_prompts": I,
        "n_tasks": J,
    }


def gitlab_paired_matrix(
    csv_path: str | Path, exclude: tuple[str, ...] = ()
) -> tuple[np.ndarray, list[str], list[str]]:
    """(prompt x task) success matrix from the old GitLab paired run (`paired_results.csv`: one row per
    (arm_id, task, timestep), a single run per cell -- no replicates), dropping `exclude` arms."""
    df = pd.read_csv(csv_path)
    excl = set(exclude)
    df = df[~df["arm_id"].isin(excl)]
    wide = df.pivot_table(index="arm_id", columns="task", values="success", aggfunc="first")
    Y = wide.to_numpy(dtype=float)
    arm_ids = [str(a) for a in wide.index]
    task_ids = [str(t) for t in wide.columns]
    return Y, arm_ids, task_ids


def _gmail_rows() -> list[dict]:
    outcomes = replay.load_snapshot_unchecked(replay.RES_DIR)
    rows = []
    for pool in replay.POOLS:
        Y, arm_ids, task_ids, n_imputed, row_counts = het.success_matrix(outcomes, pool)
        noise_var, n_pairs = het.noise_from_pairs(outcomes, pool)
        nv = None if not np.isfinite(noise_var) else noise_var
        out = analyze_cell(Y, task_ids, noise_var=nv, n_imputed=n_imputed, row_counts=row_counts, seed=0)
        out["pool"] = pool
        out["n_noise_pairs"] = n_pairs
        rows.append(out)
    return rows


def _gitlab_rows() -> list[dict]:
    rows = []
    for subset, exclude in GITLAB_SUBSETS.items():
        Y, arm_ids, task_ids = gitlab_paired_matrix(GITLAB_PAIRED_CSV, exclude=exclude)
        out = analyze_cell(Y, task_ids, noise_var=None, seed=0)
        out["subset"] = subset
        out["excluded_arms"] = ",".join(exclude)
        rows.append(out)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_gmail_rows()).to_csv(OUT_DIR / "stage0_gmail.csv", index=False)
    pd.DataFrame(_gitlab_rows()).to_csv(OUT_DIR / "stage0_gitlab_paired.csv", index=False)


if __name__ == "__main__":
    main()
