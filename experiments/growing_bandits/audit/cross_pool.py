#!/usr/bin/env python3
"""EXPLORATORY (audit step E3): put every real pool on the Pre-reg 9/11 footing and ask, descriptively,
whether measured spread / upper tail orders the value of selection across pools.

Per pool: NPMLE reservoir (src/cold_start/growing/empirical.py), fixed-n heterogeneity tests
(src/cold_start/growing/heterogeneity.py), task-held-out winner gain (retrospective/task_transfer.py),
and placement on the committed calibration grid. No API calls. Nothing here is confirmatory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing import heterogeneity as het  # noqa: E402

MATRICES = ROOT / "results" / "audit" / "matrices"
CALIBRATION = ROOT / "results" / "growing_bandits" / "heterogeneity" / "calibration.csv"
OUT = ROOT / "results" / "audit" / "e3"
TABLES = ROOT / "results" / "growing_bandits" / "deploy" / "tables"

COOKBOOK = "gitlab_oracle_operator"
HARMFUL = "explorer"
POOLS = ("G", "F", "GLK30", "GLK", "gitlab18", "gmail_paired12", "shopping_paired12")
# Pools without repeats borrow v from these pools (primary variant).
BORROW = {"gmail_paired12": ("G", "F"), "shopping_paired12": ("G", "F"), "gitlab18": ("GLK",)}
COMMITTED = {"G": 0.035, "F": 0.001, "GLK30": 0.041}
TAIL_DELTAS = (0.15, 0.25)
HORIZON = 200
N_SPLITS = 1000
SEED = 0


def _load_task_transfer():
    path = ROOT / "experiments" / "growing_bandits" / "retrospective" / "task_transfer.py"
    spec = importlib.util.spec_from_file_location("task_transfer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---- data ---------------------------------------------------------------------------


def load_outcomes() -> pd.DataFrame:
    """All matrices as one frame in the collector's schema (status ``ok``, so the repo code accepts it)."""
    frames = []
    for name in POOLS:
        frame = pd.read_csv(MATRICES / f"{name}.csv")
        frame["pool"] = name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out["success"].isna().any():
        raise ValueError("matrix CSVs must have no missing success values")
    out["status"] = emp.STATUS_OK
    out["success"] = out["success"].astype(float)
    return out


def variant_outcomes(outcomes: pd.DataFrame, source: str, variant: str, drop: tuple[str, ...]) -> pd.DataFrame:
    """Rows of pool `source` with `drop` arms removed, relabelled as pool `variant`."""
    sub = outcomes[outcomes["pool"] == source]
    keep = []
    for arm in sub["arm_id"]:
        keep.append(arm not in drop)
    sub = sub[np.array(keep, dtype=bool)].copy()
    sub["pool"] = variant
    return sub


def build_variants(outcomes: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """(label, source pool, outcomes relabelled to label). gitlab18 also appears without cookbook / explorer."""
    variants = []
    for name in POOLS:
        variants.append((name, name, outcomes[outcomes["pool"] == name].copy()))
    variants.append(("gitlab17_no_cookbook", "gitlab18",
                     variant_outcomes(outcomes, "gitlab18", "gitlab17_no_cookbook", (COOKBOOK,))))
    variants.append(("gitlab16_no_cookbook_no_explorer", "gitlab18",
                     variant_outcomes(outcomes, "gitlab18", "gitlab16_no_cookbook_no_explorer",
                                      (COOKBOOK, HARMFUL))))
    return variants


# ---- noise ----------------------------------------------------------------------------


def bernoulli_v(outcomes: pd.DataFrame, pool: str) -> float:
    """Mean over tasks of p_t (1 - p_t), p_t the task's success rate over arms (replicate 0)."""
    sub = outcomes[(outcomes["pool"] == pool) & (outcomes["replicate"] == 0)]
    p = sub.groupby("task_id")["success"].mean().to_numpy(dtype=float)
    return float(np.mean(p * (1.0 - p)))


def noise_variants(outcomes: pd.DataFrame, source: str, label: str, own: pd.DataFrame) -> list[dict]:
    """Every noise variant available for a pool; the primary one is flagged."""
    rows = []
    has_repeats = bool(((outcomes["pool"] == source) & (outcomes["replicate"] == 1)).any())
    if has_repeats:
        # Repeat cells belong to the source pool; for arm-dropped variants use only the kept arms' repeats.
        v, n = emp.within_cell_variance(own.assign(pool=source), [source])
        rows.append({"v_kind": "own_repeats", "v": v, "v_n_pairs": n, "primary": True})
    else:
        donors = BORROW[source]
        v, n = emp.within_cell_variance(outcomes, list(donors))
        rows.append({"v_kind": "borrowed_" + "+".join(donors), "v": v, "v_n_pairs": n, "primary": True})
    sub = own.assign(pool=label)
    rows.append({"v_kind": "bernoulli_task_difficulty", "v": bernoulli_v(sub, label),
                 "v_n_pairs": 0, "primary": False})
    return rows


# ---- reservoir -------------------------------------------------------------------------


def pool_scores(own: pd.DataFrame, label: str) -> emp.PromptScores:
    return emp.prompt_scores(own.assign(pool=label), label)


def reservoir_stats(scores: emp.PromptScores, v: float) -> dict:
    atoms, w, _ = emp.npmle(scores.means, v / scores.n)
    w = np.where(w < emp.MIN_WEIGHT, 0.0, w)
    w = w / w.sum()
    level = float(np.sum(atoms * w))
    sd = float(np.sqrt(np.sum(w * (atoms - level) ** 2)))
    out = {"level": level, "npmle_sd": sd, "raw_level": float(scores.means.mean()),
           "raw_sd": float(np.std(scores.means)), "n_arms": int(scores.means.size),
           "best_atom_minus_level": float(atoms[w > 0.0].max() - level)}
    for delta in TAIL_DELTAS:
        mass = 0.0
        for a, wi in zip(atoms, w):
            if a >= level + delta - 1e-12:
                mass += float(wi)
        out[f"upper_tail_mass_{int(round(delta * 100))}"] = mass
    return out


# ---- heterogeneity ---------------------------------------------------------------------


def hetero_stats(own: pd.DataFrame, label: str) -> dict:
    Y, _, task_ids, n_imp, row_counts = het.success_matrix(own.assign(pool=label), label)
    interval = het.tau_interval(Y, n_imputed=n_imp, row_counts=row_counts)
    half = het.split_half(Y, task_ids, seed=SEED, n_perm=10_000)
    return {"tau": interval["tau"], "tau_lo": interval["tau_lo"], "tau_hi": interval["tau_hi"],
            "split_half_r": half["r"], "split_half_p": half["p_value"]}


# ---- value of selection -------------------------------------------------------------------


def selection_stats(task_transfer, own: pd.DataFrame, label: str) -> dict:
    sub = own[own["replicate"] == 0]
    records = []
    for arm, task, s in zip(sub["arm_id"], sub["task_id"], sub["success"]):
        records.append((str(arm), str(task), int(s)))
    pool = task_transfer._matrix(label, records, MATRICES)
    rows = task_transfer.evaluate(pool, n_splits=N_SPLITS, seed=SEED)
    summary = task_transfer.summarize(rows)[0]
    arm_rates = pool.y.mean(axis=1)
    return {"n_tasks": len(pool.tasks),
            "winner_gain_pp": 100.0 * float(summary["mean_heldout_gain"]),
            "winner_gain_p05_pp": 100.0 * float(summary["split_p05_heldout_gain"]),
            "winner_gain_p95_pp": 100.0 * float(summary["split_p95_heldout_gain"]),
            "share_splits_positive": float(summary["share_splits_gain_positive"]),
            "mean_train_optimism_pp": 100.0 * float(summary["mean_train_optimism"]),
            "oracle_headroom_pp": 100.0 * float(arm_rates.max() - arm_rates.mean()),
            "winner_mode": summary["winner_mode"], "winner_mode_share": float(summary["winner_mode_share"])}


# ---- calibration -----------------------------------------------------------------------------


def load_calibration() -> pd.DataFrame:
    cal = pd.read_csv(CALIBRATION)
    return cal[cal["horizon"] == HORIZON].copy()


def beta_curve(cal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    beta = cal[cal["family"] == "beta"].sort_values("true_sd")
    x = np.concatenate([[0.0], beta["true_sd"].to_numpy(dtype=float)])  # sd 0 -> no range, by construction
    y = np.concatenate([[0.0], beta["regret_range"].to_numpy(dtype=float)])
    return x, y


def calibration_placement(cal: pd.DataFrame, sd: float, tail15: float) -> dict:
    x, y = beta_curve(cal)
    out = {"beta_regret_range_T200": float(np.interp(sd, x, y)),
           "beta_extrapolated": bool(sd > x.max())}
    mix = cal[cal["family"] == "mixture"]
    best_id, best_gap = "", np.inf
    for _, row in mix.iterrows():
        gap = abs(float(row["true_sd"]) - sd) + abs(float(row["upper_tail_mass"]) - tail15)
        if gap < best_gap:
            best_gap, best_id = gap, str(row["pool_id"])
    out["mixture_nearest_id"] = ""
    out["mixture_regret_range_T200"] = float("nan")
    if tail15 > 0.01:
        row = mix[mix["pool_id"] == best_id].iloc[0]
        out["mixture_nearest_id"] = best_id
        out["mixture_regret_range_T200"] = float(row["regret_range"])
    return out


# ---- association ------------------------------------------------------------------------------

PREDICTORS = ("npmle_sd", "upper_tail_mass_15", "upper_tail_mass_25", "split_half_r", "tau",
              "raw_sd", "best_atom_minus_level")


def spearman_rows(table: pd.DataFrame, subset_name: str) -> list[dict]:
    rows = []
    for col in PREDICTORS:
        x = table[col].to_numpy(dtype=float)
        y = table["winner_gain_pp"].to_numpy(dtype=float)
        rho, _ = stats.spearmanr(x, y)
        rows.append({"label": "EXPLORATORY", "subset": subset_name, "n": int(len(table)), "predictor": col,
                     "outcome": "winner_gain_pp", "spearman_rho": float(rho),
                     "note": "descriptive only; no p-value claimed"})
    return rows


def association(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    independent = list(POOLS)
    for gitlab_label in ("gitlab18", "gitlab17_no_cookbook", "gitlab16_no_cookbook_no_explorer"):
        keep = []
        for lab in primary["pool_variant"]:
            keep.append(lab in independent and lab != "gitlab18" or lab == gitlab_label)
        sub = primary[np.array(keep, dtype=bool)]
        rows.extend(spearman_rows(sub, f"7_pools_gitlab={gitlab_label}"))
    rows.extend(spearman_rows(primary, "all_9_rows_not_independent"))
    return pd.DataFrame(rows)


# ---- figure ----------------------------------------------------------------------------------------


def make_figure(primary: pd.DataFrame, cal: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg, grid = "#0e1013", "#c9ccd1", "#2a2e35"
    muted = {"G": "#7aa2c9", "F": "#6f8fa6", "GLK30": "#8fb58a", "GLK": "#a8c47f", "gitlab18": "#d9a35f",
             "gitlab17_no_cookbook": "#c27c6b", "gitlab16_no_cookbook_no_explorer": "#a06a8a",
             "gmail_paired12": "#9a8fc4", "shopping_paired12": "#8bb5b0"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color(grid)
            spine.set_linewidth(0.6)
        ax.tick_params(colors=fg, labelsize=8, width=0.5)
        ax.grid(color=grid, linewidth=0.4)
    ax = axes[0]
    for _, r in primary.iterrows():
        c = muted[r["pool_variant"]]
        ax.errorbar(r["npmle_sd"], r["winner_gain_pp"],
                    yerr=[[r["winner_gain_pp"] - r["winner_gain_p05_pp"]],
                          [r["winner_gain_p95_pp"] - r["winner_gain_pp"]]],
                    fmt="o", color=c, ms=5, lw=0.6, alpha=0.9)
        ax.annotate(r["pool_variant"], (r["npmle_sd"], r["winner_gain_pp"]), color=c, fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0, color=grid, lw=0.8)
    ax.set_xlabel("NPMLE sd of true prompt rate", color=fg, fontsize=9)
    ax.set_ylabel("held-out winner gain over pool mean (pp); bars = split 5-95%", color=fg, fontsize=9)
    ax.set_title("EXPLORATORY: value of selection vs measured spread", color=fg, fontsize=10, loc="left")
    ax = axes[1]
    x, y = beta_curve(cal)
    ax.plot(x, 100.0 * y, color=fg, lw=0.9, label="Beta family")
    mix = cal[cal["family"] == "mixture"]
    ax.scatter(mix["true_sd"], 100.0 * mix["regret_range"], s=14, color="#d9a35f", alpha=0.7,
               label="mixture family (tail atoms)", linewidths=0)
    for _, r in primary.iterrows():
        yy = 100.0 * float(np.interp(r["npmle_sd"], x, y))
        ax.scatter([r["npmle_sd"]], [yy], s=26, color=muted[r["pool_variant"]], zorder=3)
        ax.annotate(r["pool_variant"], (r["npmle_sd"], yy), color=muted[r["pool_variant"]], fontsize=7,
                    xytext=(5, -9), textcoords="offset points")
    ax.set_xlabel("NPMLE sd (calibration true_sd)", color=fg, fontsize=9)
    ax.set_ylabel(f"simulated regret range at T={HORIZON} (pp)", color=fg, fontsize=9)
    ax.set_title("EXPLORATORY: pools placed on the calibration curve", color=fg, fontsize=10, loc="left")
    leg = ax.legend(fontsize=7, facecolor=bg, edgecolor=grid)
    for text in leg.get_texts():
        text.set_color(fg)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=bg)
    plt.close(fig)


# ---- orchestration -----------------------------------------------------------------------------------


def crosscheck_committed(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, committed in COMMITTED.items():
        mine = float(table[(table["pool_variant"] == name) & (table["primary"])]["npmle_sd"].iloc[0])
        rows.append({"pool": name, "committed_npmle_sd": committed, "reproduced_npmle_sd": mine,
                     "abs_diff": abs(mine - committed), "ok_at_3dp": bool(abs(mine - committed) < 0.0015)})
    return pd.DataFrame(rows)


def run(out_dir: Path = OUT) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    task_transfer = _load_task_transfer()
    outcomes = load_outcomes()
    cal = load_calibration()
    rows = []
    for label, source, own in build_variants(outcomes):
        scores = pool_scores(own, label)
        hetero = hetero_stats(own, label)
        select = selection_stats(task_transfer, own, label)
        for noise in noise_variants(outcomes, source, label, own):
            res = reservoir_stats(scores, noise["v"])
            place = calibration_placement(cal, res["npmle_sd"], res["upper_tail_mass_15"])
            row = {"label": "EXPLORATORY", "pool_variant": label, "source_pool": source}
            row.update(noise)
            row.update(res)
            row.update(hetero)
            row.update(select)
            row.update(place)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "pool_table.csv", index=False)
    primary = table[table["primary"]].reset_index(drop=True)
    assoc = association(primary)
    assoc.to_csv(out_dir / "association.csv", index=False)
    check = crosscheck_committed(table)
    check.to_csv(out_dir / "crosscheck_committed.csv", index=False)
    make_figure(primary, cal, out_dir / "pools_on_curve.png")
    return {"table": table, "association": assoc, "crosscheck": check}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.out)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(result["crosscheck"].to_string(index=False))


if __name__ == "__main__":
    main()
