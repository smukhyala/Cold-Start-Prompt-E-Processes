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
            "tau_set": tau + 0.005, "tau_set_lo": tau - 0.002, "tau_set_hi": tau + 0.012,
            "tau_set_upper_one_sided": tau + 0.02,
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


def _flatness_df() -> pd.DataFrame:
    """A synthetic `emp_flatness.csv` (`describe.flatness`'s columns): npmle regret range per pool at
    every primary horizon, for Pre-reg 9's G and F (fix round 1: figure 3's source for a *measured*
    historical regret range)."""
    rows = []
    for pool, rr200 in (("G", 0.018), ("F", 0.041)):
        for T, frac in ((50, 0.5), (100, 0.75), (200, 1.0)):
            rows.append({"env_id": f"emp_{pool}_npmle", "pool": pool, "variant": "npmle", "horizon": T,
                        "regret_range": rr200 * frac, "informative": True, "reservoir_sha256": "beadfeed"})
    return pd.DataFrame(rows)


def _write_all_tables(tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    _components_df().to_csv(tables_dir / "het_components.csv", index=False)
    _classification_df().to_csv(tables_dir / "het_classification.csv", index=False)
    _policy_gaps_df().to_csv(tables_dir / "het_policy_gaps.csv", index=False)
    _portability_df().to_csv(tables_dir / "het_portability.csv", index=False)
    _timeout_rates_df().to_csv(tables_dir / "het_timeout_rates.csv", index=False)
    _anchor_recovery_df().to_csv(tables_dir / "het_anchor_recovery.csv", index=False)
    _calibration_df().to_csv(tables_dir / "calibration.csv", index=False)
    _stage0_gmail_df().to_csv(tables_dir / "stage0_gmail.csv", index=False)
    _stage0_gitlab_df().to_csv(tables_dir / "stage0_gitlab_paired.csv", index=False)
    _kgrid_df().to_csv(tables_dir / "het_kgrid.csv", index=False)
    _flatness_df().to_csv(tables_dir / "emp_flatness.csv", index=False)


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


def _anchor_recovery_df(recovered: bool = True) -> pd.DataFrame:
    """`het_verdicts.anchor_recovery`'s columns (read from the producer, plus its provenance stamp)."""
    return pd.DataFrame([{
        "oracle_arm": "GL_anchor_oracle", "oracle_rate": 0.9 if recovered else 0.6, "oracle_n": 60,
        "explorer_arm": "GL_anchor_explorer", "explorer_rate": 0.1, "explorer_n": 60,
        "gl_baseline_rate": 0.5, "gl_baseline_n": 60, "bulk_pools": "GLG+GLK", "bulk_n_prompts": 90,
        "bulk_p10": 0.3, "bulk_p90": 0.7, "n_tasks": 60, "oracle_above_p90": recovered,
        "explorer_below_p10": True, "recovered": recovered, "gm_anchor_baseline_rate": 0.63,
        "gm_anchor_baseline_n": 30, "prereg9_anchor_baseline_rate": 0.66, "prereg9_anchor_baseline_n": 30,
        "gm_anchor_drift": -0.03, "note": "", "task_universe": "subset_60",
    }])


def test_fig7d_anchor_recovery_panel_reads_the_anchor_recovery_table():
    import matplotlib.pyplot as plt

    for recovered in (True, False):
        fig, ax = plt.subplots()
        try:
            fg._fig7d_anchor_recovery(ax, _anchor_recovery_df(recovered))
            heights = [round(b.get_height(), 6) for b in ax.containers[0]]
            assert heights == [0.9 if recovered else 0.6, 0.1, 0.5, 0.63, 0.66]
            assert ax.get_title().endswith("recovered" if recovered else "NOT recovered")
            assert (ax.get_title() == "(d) anchor recovery: recovered") == recovered
        finally:
            plt.close(fig)


def test_fig7d_anchor_recovery_panel_is_annotated_empty_without_the_table():
    import matplotlib.pyplot as plt

    for table in (None, pd.DataFrame(), _components_df()):
        fig, ax = plt.subplots()
        try:
            fg._fig7d_anchor_recovery(ax, table)
            assert len(ax.patches) == 0
            assert any("het_anchor_recovery.csv" in t.get_text() for t in ax.texts)
        finally:
            plt.close(fig)


def test_tau_for_axis_prefers_tau_set_and_falls_back_to_tau_main():
    row = {"tau_set": 0.05, "tau_set_lo": 0.03, "tau_set_hi": 0.08, "tau_main": 0.04, "tau_lo": 0.0, "tau_hi": 0.09}
    assert fg._tau_for_axis(row) == {"tau": 0.05, "lo": 0.03, "hi": 0.08, "measure": "tau_set"}
    no_set = {**row, "tau_set": float("nan")}
    assert fg._tau_for_axis(no_set) == {"tau": 0.04, "lo": 0.0, "hi": 0.09, "measure": "tau_main"}
    assert fg._tau_for_axis({"tau_main": 0.04}) == {"tau": 0.04, "lo": 0.04, "hi": 0.04, "measure": "tau_main"}
    assert fg._tau_for_axis({"tau_set": float("nan"), "tau_main": float("nan")}) is None


def _capture(monkeypatch):
    """Keep the figure `_save` would close, so a test can read its axes."""
    kept = []

    def save(fig, out_dir, stem):
        kept.append(fig)
        return []

    monkeypatch.setattr(fg, "_save", save)
    return kept


def test_fig2_plots_tau_set_with_its_mls_interval_and_labels_a_tau_main_fallback(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    comp = _components_df()
    comp.loc[comp["pool"] == "GMB", ["tau_set", "tau_set_lo", "tau_set_hi"]] = float("nan")
    kept = _capture(monkeypatch)
    with plt.rc_context(fg.RC):
        fg._fig2_tau_bands(comp, _calibration_df(), 0.03, tmp_path)
    ax = kept[0].axes[0]
    try:
        ys = [line.get_ydata()[0] for line in ax.lines if len(line.get_ydata()) == 1 and line.get_marker() == "o"]
        want = [float(r["tau_set"]) if np.isfinite(r["tau_set"]) else float(r["tau_main"]) for _, r in comp.iterrows()]
        assert ys == pytest.approx(want)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "GMB (tau_main)" in labels and "GLK" in labels
        assert ax.get_ylabel().startswith("tau_set")
    finally:
        plt.close(kept[0])


def test_fig3_empirical_points_sit_on_tau_set(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    comp = _components_df()
    kept = _capture(monkeypatch)
    with plt.rc_context(fg.RC):
        fg._fig3_value_of_search(comp, _classification_df(), _calibration_df(), None, None, None, tmp_path)
    ax = kept[0].axes[0]
    try:
        texts = {t.get_text(): t.xy for t in ax.texts if hasattr(t, "xy")}
        for _, r in comp.iterrows():
            assert texts[str(r["pool"])][0] == pytest.approx(float(r["tau_set"]))
    finally:
        plt.close(kept[0])


def test_fig5_axis_is_policy_minus_reference(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    kept = _capture(monkeypatch)
    with plt.rc_context(fg.RC):
        fg._fig5_policy_gaps(_policy_gaps_df(), _classification_df(), tmp_path)
    ax = kept[0].axes[0]
    try:
        assert ax.get_ylabel().startswith("policy \u2212 reference")
        assert "ceiling" not in ax.get_ylabel() and "ceiling" not in ax.get_title()
    finally:
        plt.close(kept[0])


def test_historical_points_combines_gmail_and_gitlab_stage0():
    pts = fg._historical_points(_stage0_gmail_df(), _stage0_gitlab_df())
    names = {p["name"] for p in pts}
    assert "G" in names
    assert "F" in names
    assert "all_18_arms" in names
    assert len(pts) == 5
    source_of = {p["name"]: p["source"] for p in pts}
    assert source_of["G"] == "gmail"
    assert source_of["F"] == "gmail"
    assert source_of["all_18_arms"] == "gitlab"
    for p in pts:
        assert p["tau_lo"] <= p["tau"] <= p["tau_hi"]
    measure = {p["name"]: p["measure"] for p in pts}
    assert measure["G"] == "tau_set" and measure["all_18_arms"] == "tau_main"  # the old run has no tau_set


def test_historical_points_empty_when_no_stage0_tables():
    assert fg._historical_points(None, None) == []


# ---- fix round 1: measured vs interpolated historical points (figure 3) --------------------------


def _beta200(cal: pd.DataFrame) -> pd.DataFrame:
    return cal[(cal["family"] == "beta") & (cal["horizon"] == 200)].sort_values("true_sd")


def test_measured_regret_range_reads_the_npmle_t200_row_for_the_pool():
    flatness = _flatness_df()
    assert fg._measured_regret_range(flatness, "G") == pytest.approx(0.018)
    assert fg._measured_regret_range(flatness, "F") == pytest.approx(0.041)


def test_measured_regret_range_none_when_missing_or_ambiguous():
    assert fg._measured_regret_range(None, "G") is None
    assert fg._measured_regret_range(pd.DataFrame(), "G") is None
    flatness = _flatness_df()
    assert fg._measured_regret_range(flatness, "nonexistent_pool") is None
    duplicated = pd.concat([_flatness_df(), _flatness_df()], ignore_index=True)
    assert fg._measured_regret_range(duplicated, "G") is None  # two matching rows: ambiguous


def test_historical_figure_points_uses_measured_for_gmail_and_interpolates_gitlab():
    beta = _beta200(_calibration_df())
    flatness = _flatness_df()
    pts = fg._historical_figure_points(beta, _stage0_gmail_df(), _stage0_gitlab_df(), flatness)
    by_name = {p["name"]: p for p in pts}

    assert by_name["G"]["kind"] == "measured"
    assert by_name["G"]["y"] == pytest.approx(0.018)
    assert by_name["G"]["label"] == "G (historical)"
    assert by_name["F"]["kind"] == "measured"
    assert by_name["F"]["y"] == pytest.approx(0.041)
    assert by_name["F"]["label"] == "F (historical)"

    for name in ("all_18_arms", "without_oracle_and_explorer", "generic_12_arms"):
        p = by_name[name]
        assert p["kind"] == "interpolated"
        assert p["label"] == f"{name} (historical, interpolated, tau_main)"
        assert p["y"] == pytest.approx(fg._interp_regret_range(beta, p["tau"]))


def test_historical_figure_points_all_interpolated_without_flatness_table():
    beta = _beta200(_calibration_df())
    pts = fg._historical_figure_points(beta, _stage0_gmail_df(), _stage0_gitlab_df(), None)
    assert len(pts) == 5
    assert all(p["kind"] == "interpolated" for p in pts)
    assert all("(historical, interpolated" in p["label"] for p in pts)
    by_name = {p["name"]: p for p in pts}
    assert by_name["G"]["label"] == "G (historical, interpolated)"
    assert by_name["G"]["y"] == pytest.approx(fg._interp_regret_range(beta, by_name["G"]["tau"]))


def test_historical_figure_points_falls_back_when_pools_row_is_missing():
    """A flatness table that exists but has no row for a given pool (e.g. only G, not F) falls back to
    interpolation for that pool alone."""
    beta = _beta200(_calibration_df())
    flatness = _flatness_df()
    flatness_g_only = flatness[flatness["pool"] == "G"]
    pts = fg._historical_figure_points(beta, _stage0_gmail_df(), None, flatness_g_only)
    by_name = {p["name"]: p for p in pts}
    assert by_name["G"]["kind"] == "measured"
    assert by_name["F"]["kind"] == "interpolated"
    assert by_name["F"]["y"] == pytest.approx(fg._interp_regret_range(beta, by_name["F"]["tau"]))


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
        prereg9_flatness_csv=tables_dir / "emp_flatness.csv",
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
