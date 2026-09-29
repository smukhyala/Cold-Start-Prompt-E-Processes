"""Pre-registration 10's paper figures (spec 2026-09-28-prompt-heterogeneity, section 7): every figure is
drawn from CSV tables (`het_verdicts.py`, `calibrate.py`, `stage0.py`, `replay.py`'s kgrid), never raw
episodes or reservoirs. No real data exists yet, so every test builds small synthetic tables matching the
producers' exact column names (read from their writers, not guessed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
import figures as fg  # noqa: E402

POOLS = ("GMG", "GMK", "GMB", "GLG", "GLK")


# ---- synthetic table builders -------------------------------------------------------------------


def _components_df(pools: tuple[str, ...] = POOLS) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(pools):
        tau = 0.02 + 0.01 * i
        rows.append({
            "pool": p, "tau_main": tau, "tau_lo": max(tau - 0.01, 0.0), "tau_hi": tau + 0.015,
            "tau_set": tau + 0.005, "tau_set_upper_one_sided": tau + 0.02,
            "task_var": 0.01 + 0.002 * i, "interaction_var": 0.005, "noise_var": 0.02,
            "r_sb": 0.3 + 0.05 * i, "split_half_p": 0.04,
            "upper_tail_mass": 0.1 + 0.02 * i, "upper_tail_mass_lo": 0.05 + 0.02 * i,
            "upper_tail_mass_hi": 0.2 + 0.02 * i, "n_prompts": 30, "n_tasks": 60,
        })
    return pd.DataFrame(rows)


def _classification_df(pools: tuple[str, ...] = POOLS, meaningful: tuple[str, ...] = ("GLG",)) -> pd.DataFrame:
    rows = []
    for p in pools:
        cls = "meaningful" if p in meaningful else "flat" if p == "GMG" else "moderate"
        row = {"pool": p, "tau_set_upper_one_sided": 0.03, "tau_flat": 0.03, "class": cls,
               "n_boot": 2000, "manifest_sha256": "deadbeef"}
        for T, base in ((50, 0.012), (100, 0.011), (200, 0.009)):
            rr = base if cls == "meaningful" else 0.002
            row[f"regret_range_T{T}"] = rr
            row[f"rr_lo_T{T}"] = max(rr - 0.003, 0.0)
            row[f"rr_hi_T{T}"] = rr + 0.003
        rows.append(row)
    return pd.DataFrame(rows)


def _policy_gaps_df(meaningful: tuple[str, ...] = ("GLG",)) -> pd.DataFrame:
    rows = [{
        "registration": "het_scale", "contrast": "het_scale", "cells": "meaningful", "row": "primary",
        "policy": "p3_star", "reference": "fixed_K8", "test": "het", "horizon": "all",
        "delta": -0.01, "lo": -0.02, "hi": -0.003, "paired_lo": -0.02, "paired_hi": -0.003,
        "n_informative": 3, "informative_cells": "het_GLG_npmle@200", "n_boot": 2000,
        "mei": 0.002, "rule": "min", "verdict": "supported", "as_registered": True,
    }]
    for p in meaningful:
        rows.append({**rows[0], "cells": p, "informative_cells": f"het_{p}_npmle@200"})
    return pd.DataFrame(rows)


def _portability_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 8
    a = rng.normal(0, 0.05, n)
    b = a * 0.7 + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "g_index": range(n), "arm_gmail": [f"GMG_G_{i}" for i in range(n)],
        "arm_gitlab": [f"GLG_{i}" for i in range(n)], "effect_gmail": a, "effect_gitlab": b,
        "raw_effect_gmail": a, "raw_effect_gitlab": b,
        "rank_gmail": pd.Series(a).rank(), "rank_gitlab": pd.Series(b).rank(),
    })


def _timeout_rates_df(pools: tuple[str, ...] = POOLS) -> pd.DataFrame:
    rows = []
    for p in pools:
        for a in range(4):
            rows.append({"pool": p, "arm_id": f"{p}_{a}", "n_episodes": 60,
                        "n_clock": a, "timeout_rate": a / 60.0})
    return pd.DataFrame(rows)


def _calibration_df() -> pd.DataFrame:
    rows = []
    spreads = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15)
    for T in (50, 100, 200):
        for sd in spreads:
            # A simple increasing curve, roughly matching the shape regret range takes vs spread.
            rr = min(0.15, sd * (0.6 if T == 200 else 0.4))
            rows.append({"pool_id": f"beta_sd{sd:g}", "family": "beta", "spread": sd,
                        "tail_frac": float("nan"), "tail_delta": float("nan"), "true_sd": sd,
                        "upper_tail_mass": 0.05, "horizon": T, "regret_range": rr, "k_star": 8,
                        "gap_fixed_K8": 0.001, "gap_p3_star": 0.0005, "gap_always_search": 0.002})
        for f in (0.02, 0.05):
            rows.append({"pool_id": f"tail_f{f:g}_d0.2", "family": "tail", "spread": float("nan"),
                        "tail_frac": f, "tail_delta": 0.2, "true_sd": 0.06 + f,
                        "upper_tail_mass": 0.2, "horizon": T, "regret_range": 0.02 + f, "k_star": 12,
                        "gap_fixed_K8": 0.002, "gap_p3_star": 0.001, "gap_always_search": 0.004})
    return pd.DataFrame(rows)


def _stage0_gmail_df() -> pd.DataFrame:
    rows = []
    for pool, tau in (("G", 0.03), ("F", 0.05)):
        rows.append({"pool": pool, "tau": tau, "tau_main": tau, "tau_lo": tau - 0.01, "tau_hi": tau + 0.01,
                    "tau_upper_one_sided": tau + 0.015, "tau_set": tau + 0.005, "tau_set_lo": tau - 0.005,
                    "tau_set_hi": tau + 0.02, "tau_set_upper_one_sided": tau + 0.03,
                    "task_var": 0.01, "interaction_var": 0.005, "noise_var": 0.02, "r_sb": 0.4,
                    "split_half_p": 0.03, "n_discriminating_tasks": 20, "tau_discriminating": tau,
                    "upper_tail_mass": 0.15, "n_prompts": 50, "n_tasks": 30, "n_noise_pairs": 40})
    return pd.DataFrame(rows)


def _stage0_gitlab_df() -> pd.DataFrame:
    rows = []
    for subset, tau in (("all_18_arms", 0.04), ("without_oracle_and_explorer", 0.025),
                        ("generic_12_arms", 0.02)):
        rows.append({"subset": subset, "excluded_arms": "", "tau": tau, "tau_main": tau,
                    "tau_lo": tau - 0.008, "tau_hi": tau + 0.012, "tau_upper_one_sided": tau + 0.02,
                    "tau_set": float("nan"), "tau_set_lo": float("nan"), "tau_set_hi": float("nan"),
                    "tau_set_upper_one_sided": float("nan"), "task_var": 0.01, "interaction_var": None,
                    "noise_var": None, "r_sb": 0.35, "split_half_p": 0.05, "n_discriminating_tasks": 10,
                    "tau_discriminating": tau, "upper_tail_mass": 0.1, "n_prompts": 18, "n_tasks": 60})
    return pd.DataFrame(rows)


def _kgrid_df(pools: tuple[str, ...] = POOLS) -> pd.DataFrame:
    rows = []
    ks = (2, 4, 8, 16, 32, 64)
    for p in pools:
        for T in (50, 100, 200):
            for K in ks:
                if K > T:
                    continue
                rows.append({"env_id": f"het_{p}_npmle", "pool": p, "variant": "npmle", "horizon": T,
                            "K": K, "regret": 0.05 / K, "reservoir_sha256": "abc123"})
    return pd.DataFrame(rows)


def _write_all_tables(tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    _components_df().to_csv(tables_dir / "het_components.csv", index=False)
    _classification_df().to_csv(tables_dir / "het_classification.csv", index=False)
    _policy_gaps_df().to_csv(tables_dir / "het_policy_gaps.csv", index=False)
    _portability_df().to_csv(tables_dir / "het_portability.csv", index=False)
    _timeout_rates_df().to_csv(tables_dir / "het_timeout_rates.csv", index=False)
    _calibration_df().to_csv(tables_dir / "calibration.csv", index=False)
    _stage0_gmail_df().to_csv(tables_dir / "stage0_gmail.csv", index=False)
    _stage0_gitlab_df().to_csv(tables_dir / "stage0_gitlab_paired.csv", index=False)
    _kgrid_df().to_csv(tables_dir / "het_kgrid.csv", index=False)


# ---- pure helper behaviour -----------------------------------------------------------------------


def test_pool_family_maps_every_study_pool():
    assert fg._pool_family("GMG") == "G"
    assert fg._pool_family("GLG") == "G"
    assert fg._pool_family("GMB") == "G"
    assert fg._pool_family("GMK") == "K"
    assert fg._pool_family("GLK") == "K"
    assert fg._pool_family("nonsense") == "?"


def test_meaningful_pools_reads_the_class_column():
    classes = _classification_df(meaningful=("GLG", "GMK"))
    assert fg._meaningful_pools(classes) == ["GLG", "GMK"]
    assert fg._meaningful_pools(None) == []
    assert fg._meaningful_pools(pd.DataFrame()) == []


def test_policy_gap_rows_for_meaningful_filters_to_meaningful_pools():
    classes = _classification_df(meaningful=("GLG",))
    gaps = _policy_gaps_df(meaningful=("GLG",))
    rows = fg._policy_gap_rows_for_meaningful(gaps, classes)
    assert list(rows["cells"]) == ["GLG"]


def test_policy_gap_rows_for_meaningful_is_empty_when_no_cell_is_meaningful():
    """The brief's own example: no meaningful cell -> an empty selection, not an exception."""
    classes = _classification_df(meaningful=())  # every cell flat/moderate
    gaps = _policy_gaps_df(meaningful=())
    rows = fg._policy_gap_rows_for_meaningful(gaps, classes)
    assert rows.empty


def test_policy_gap_rows_for_meaningful_handles_missing_table():
    assert fg._policy_gap_rows_for_meaningful(None, _classification_df()).empty
    assert fg._policy_gap_rows_for_meaningful(pd.DataFrame(), _classification_df()).empty


def test_calibration_edge_interpolates_the_crossing():
    cal = pd.DataFrame([
        {"family": "beta", "horizon": 200, "true_sd": 0.02, "regret_range": 0.003},
        {"family": "beta", "horizon": 200, "true_sd": 0.04, "regret_range": 0.007},
    ])
    edge = fg._calibration_edge(cal, target=0.005, horizon=200)
    assert edge == pytest.approx(0.03)


def test_calibration_edge_none_when_never_crossed_or_missing():
    assert fg._calibration_edge(None, target=0.005) is None
    assert fg._calibration_edge(pd.DataFrame(), target=0.005) is None
    cal = pd.DataFrame([{"family": "beta", "horizon": 200, "true_sd": 0.02, "regret_range": 0.001}])
    assert fg._calibration_edge(cal, target=0.5, horizon=200) is None


def test_anchor_columns_finds_gl_gm_anchor_prefixes():
    df = pd.DataFrame({"pool": ["GLG"], "GL_anchor_oracle": [0.9], "other": [1]})
    assert fg._anchor_columns(df) == ["GL_anchor_oracle"]
    assert fg._anchor_columns(pd.DataFrame({"pool": ["GLG"]})) == []
    assert fg._anchor_columns(None) == []


def test_fig7d_anchor_recovery_panel_draws_bars_when_columns_are_present():
    """Neither producer currently writes anchor columns (spec 7d's stand-in), but the panel must still
    draw real bars, not the annotated-empty fallback, if a future producer adds them."""
    import matplotlib.pyplot as plt

    components = pd.DataFrame({"pool": ["GLG", "GLK"], "GL_anchor_oracle": [0.9, 0.7],
                                "GM_anchor_explorer": [0.1, 0.2]})
    fig, ax = plt.subplots()
    try:
        fg._fig7d_anchor_recovery(ax, components, None)
        assert len(ax.patches) > 0
        assert not any("see gates output" in t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)


def test_fig7d_anchor_recovery_panel_is_annotated_empty_without_anchor_columns():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        fg._fig7d_anchor_recovery(ax, _components_df(), _timeout_rates_df())
        assert len(ax.patches) == 0
        assert any("see gates output" in t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)


def test_historical_points_combines_gmail_and_gitlab_stage0():
    pts = fg._historical_points(_stage0_gmail_df(), _stage0_gitlab_df())
    labels = {p["label"] for p in pts}
    assert "G (historical)" in labels
    assert "F (historical)" in labels
    assert "all_18_arms (historical)" in labels
    assert len(pts) == 5
    for p in pts:
        assert p["tau_lo"] <= p["tau_main"] <= p["tau_hi"]


def test_historical_points_empty_when_no_stage0_tables():
    assert fg._historical_points(None, None) == []


# ---- integration: make_figures --------------------------------------------------------------------


def test_make_figures_full_tables_produces_all_seven_figures(tmp_path, recwarn):
    tables_dir = tmp_path / "tables"
    _write_all_tables(tables_dir)
    out_dir = tmp_path / "figures"

    paths = fg.make_figures(
        tables_dir, out_dir,
        calibration_csv=tables_dir / "calibration.csv",
        stage0_dir=tables_dir,
        kgrid_csv=tables_dir / "het_kgrid.csv",
        tau_flat=0.03,
    )

    assert len(paths) == 14  # 7 figures x {png, pdf}
    stems = sorted({p.stem for p in paths})
    assert stems == [f"fig{n}_{slug}" for n, slug in enumerate((
        "variance_decomposition", "tau_bands", "value_of_search", "regret_vs_k",
        "policy_gaps", "portability", "supplement",
    ), start=1)]
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix in (".png", ".pdf")
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_make_figures_missing_required_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="het_components.csv"):
        fg.make_figures(tmp_path, tmp_path / "figures")


def test_make_figures_sparse_tables_produce_annotated_panels_not_exceptions(tmp_path, recwarn):
    """Only the required table exists; every optional table and the K-grid default location are
    absent (this is the real state before any data collection). No figure raises."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _components_df(pools=("GMG", "GLG")).to_csv(tables_dir / "het_components.csv", index=False)
    out_dir = tmp_path / "figures"

    paths = fg.make_figures(tables_dir, out_dir)

    assert len(paths) == 14
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_make_figures_no_meaningful_cell_yields_annotated_panel_not_exception(tmp_path):
    """The brief's Step 1 example: no meaningful cell -> an (effectively) empty het_policy_gaps.csv,
    and figure 5 must still be written rather than raising."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _components_df().to_csv(tables_dir / "het_components.csv", index=False)
    _classification_df(meaningful=()).to_csv(tables_dir / "het_classification.csv", index=False)
    # Only the pooled "meaningful" class row exists (no cell qualifies) -- what `policy_gaps()`
    # writes when no pool is classified meaningful.
    pd.DataFrame([{
        "registration": "het_scale", "contrast": "het_scale", "cells": "meaningful", "row": "primary",
        "policy": "p3_star", "reference": "fixed_K8", "test": "het", "horizon": "all",
        "delta": np.nan, "lo": np.nan, "hi": np.nan, "paired_lo": np.nan, "paired_hi": np.nan,
        "n_informative": 0, "informative_cells": "", "n_boot": 0, "mei": 0.002, "rule": "min",
        "verdict": "no_cells", "as_registered": True,
    }]).to_csv(tables_dir / "het_policy_gaps.csv", index=False)
    out_dir = tmp_path / "figures"

    paths = fg.make_figures(tables_dir, out_dir)

    fig5 = [p for p in paths if p.stem.startswith("fig5_")]
    assert len(fig5) == 2
    for p in fig5:
        assert p.exists() and p.stat().st_size > 0


def test_make_figures_empty_optional_csvs_are_annotated_not_raised(tmp_path):
    """Header-only (zero-row) optional tables must not crash any figure."""
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _components_df().to_csv(tables_dir / "het_components.csv", index=False)
    pd.DataFrame(columns=["pool", "class"]).to_csv(tables_dir / "het_classification.csv", index=False)
    pd.DataFrame(columns=["cells", "delta", "lo", "hi"]).to_csv(tables_dir / "het_policy_gaps.csv", index=False)
    pd.DataFrame(columns=["effect_gmail", "effect_gitlab"]).to_csv(tables_dir / "het_portability.csv", index=False)
    pd.DataFrame(columns=["pool", "arm_id", "timeout_rate"]).to_csv(tables_dir / "het_timeout_rates.csv", index=False)
    pd.DataFrame(columns=["family", "horizon", "true_sd", "regret_range"]).to_csv(tables_dir / "calibration.csv", index=False)

    paths = fg.make_figures(tables_dir, tmp_path / "figures")

    assert len(paths) == 14
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
