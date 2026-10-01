"""Pre-registration 10's verdict machinery: section 6.5 classification, H1-H4, the H3 policy contrasts
(`registered_contrast`'s ``het_*`` registrations) and replay's ``boot_kgrid`` stage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("deploy", "empirical"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))
import het_verdicts as hv  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import study as st  # noqa: E402

from cold_start.growing import heterogeneity as het  # noqa: E402

H = st.HETEROGENEITY
COL = "regret_posterior_mean_shrunk"


# ---- section 6.5 classification ---------------------------------------------------------


@pytest.mark.parametrize("rr, lo, hi, expected", [
    ({50: 0.004, 100: 0.004, 200: 0.004}, 0.001, 0.009, "flat"),
    ({50: 0.004, 100: 0.004, 200: 0.004}, 0.001, 0.012, "moderate"),  # the CI does not exclude meaningful
    # spec section 6.5: >= 0.01 at 2 of 3 horizons (T=50, 100) and the T=200 bootstrap lower bound > 0.005;
    # nothing requires T=200 >= 0.01.
    ({50: 0.012, 100: 0.011, 200: 0.009}, 0.006, 0.02, "meaningful"),
    ({50: 0.012, 100: 0.004, 200: 0.009}, 0.006, 0.02, "moderate"),  # only 1 of 3 horizons >= 0.01
    ({50: 0.012, 100: 0.004, 200: 0.015}, 0.006, 0.02, "meaningful"),
    ({50: 0.012, 100: 0.011, 200: 0.015}, 0.004, 0.02, "moderate"),
])
def test_classify_cell(rr, lo, hi, expected):
    assert hv.classify_cell(rr, lo, hi) == expected


def test_classify_cell_flat_boundaries_are_the_amended_inequalities():
    # C1 ruling (spec section 6.5, amended before registration): flat = regret range < 0.005 at all three
    # primary T AND the bootstrap upper bound of the T=200 regret range < 0.01 -- both strict.
    below = {50: 0.0049, 100: 0.0049, 200: 0.0049}
    assert hv.classify_cell(below, 0.0, 0.0099) == "flat"
    assert hv.classify_cell(below, 0.0, 0.01) == "moderate"  # hi == 0.01 is not < 0.01
    assert hv.classify_cell({50: 0.0049, 100: 0.005, 200: 0.0049}, 0.0, 0.0) == "moderate"  # 0.005 not < 0.005
    assert hv.classify_cell({50: 0.005, 100: 0.0049, 200: 0.0049}, 0.0, 0.0) == "moderate"
    assert hv.classify_cell({50: 0.0049, 100: 0.0049, 200: 0.005}, 0.0, 0.0) == "moderate"
    # meaningful is unchanged: >= 0.01 at >= 2 of 3 (0.01 itself counts), T=200 lower bound strictly > 0.005
    assert hv.classify_cell({50: 0.01, 100: 0.0, 200: 0.01}, 0.0051, 0.1) == "meaningful"
    assert hv.classify_cell({50: 0.01, 100: 0.0, 200: 0.01}, 0.005, 0.1) == "moderate"
    assert hv.classify_cell({50: 0.01, 100: 0.0, 200: 0.0099}, 0.02, 0.1) == "moderate"


def test_classify_cell_takes_no_tau_argument():
    # the tau_set bound is reported (classify_pools' tau_bound_below_flat), never decisive
    import inspect
    assert list(inspect.signature(hv.classify_cell).parameters) == ["regret_range", "rr_lo_T200", "rr_hi_T200"]


def test_classify_cell_refuses_incomplete_inputs():
    with pytest.raises(KeyError):
        hv.classify_cell({50: 0.001, 100: 0.001}, 0.0, 0.01)
    with pytest.raises(ValueError):
        hv.classify_cell({50: 0.001, 100: np.nan, 200: 0.001}, 0.0, 0.01)
    with pytest.raises(ValueError):
        hv.classify_cell({50: 0.001, 100: 0.001, 200: 0.001}, 0.0, np.nan)
    with pytest.raises(ValueError):
        hv.classify_cell({50: 0.001, 100: 0.001, 200: 0.001}, np.nan, 0.001)


# ---- H1 / H2 / H4 decision rule ----------------------------------------------------------


def test_h_verdict_rules():
    assert hv.h_verdict(np.linspace(0.01, 0.05, 101), refute_below=0.01)["verdict"] == "supported"
    assert hv.h_verdict(np.linspace(-0.02, 0.005, 101), refute_below=0.01)["verdict"] == "refuted"
    assert hv.h_verdict(np.linspace(-0.02, 0.03, 101), refute_below=0.01)["verdict"] == "inconclusive"


def test_h_verdict_bounds_are_one_sided_95():
    draws = np.linspace(-1.0, 1.0, 2001)
    out = hv.h_verdict(draws, refute_below=0.01)
    assert out["lo"] == pytest.approx(np.percentile(draws, 5)) and out["hi"] == pytest.approx(np.percentile(draws, 95))
    assert not out["effect_below_threshold"]
    # both conditions met (a positive difference smaller than 0.01): supported, with the explicit flag
    both = hv.h_verdict(np.linspace(0.001, 0.005, 101), refute_below=0.01)
    assert both["verdict"] == "supported" and both["effect_below_threshold"] is True
    assert not hv.h_verdict(np.linspace(-0.02, 0.005, 101), refute_below=0.01)["effect_below_threshold"]
    assert not hv.h_verdict(np.linspace(0.01, 0.05, 101), refute_below=0.01)["effect_below_threshold"]
    with pytest.raises(ValueError):
        hv.h_verdict(np.array([0.1, np.nan]), refute_below=0.01)


# ---- H3 -----------------------------------------------------------------------------------


def _contrasts(rows):
    return pd.DataFrame(rows, columns=["cells", "registration", "verdict"])


def test_h3_supported_untestable_refuted():
    # the brief's cases, on per-cell rows (fix round 1: pooled class rows never decide)
    classes = {"GLK": "meaningful", "GLG": "flat", "GMG": "flat"}
    flat_ok = [(c, r, "inconclusive") for c in ("GLG", "GMG") for r in ("het_scale", "het_spread")]
    ok = _contrasts([("GLK", "het_scale", "supported"), ("GLK", "het_spread", "supported"), *flat_ok])
    assert hv.h3(classes, ok)["verdict"] == "supported"
    assert hv.h3({"GLG": "flat", "GMG": "moderate"}, ok)["verdict"] == "untestable"
    bad = _contrasts([("GLK", "het_scale", "reversed"), ("GLK", "het_spread", "supported"), *flat_ok])
    assert hv.h3(classes, bad)["verdict"] == "refuted"
    flat_diff = _contrasts([("GLK", "het_scale", "supported"), ("GLK", "het_spread", "supported"),
                            ("GLG", "het_scale", "supported"), *flat_ok[1:]])
    assert hv.h3(classes, flat_diff)["verdict"] == "refuted"


def test_h3_pooled_rows_are_supplementary_never_decisive():
    classes = {"GLK": "meaningful", "GLG": "flat", "GMG": "flat"}
    per_cell = [("GLK", "het_scale", "supported"), ("GLK", "het_spread", "supported"),
                ("GLG", "het_scale", "inconclusive"), ("GMG", "het_scale", "inconclusive")]
    # only the pooled rows would refute: a pooled flat difference and a pooled meaningful reversal
    rows = _contrasts([*per_cell, ("flat", "het_scale", "supported"), ("meaningful", "het_spread", "reversed")])
    out = hv.h3(classes, rows)
    assert out["verdict"] == "supported" and out["refuting"] == [] and len(out["supplementary"]) == 2
    # pooled rows alone never support either
    pooled = _contrasts([("meaningful", "het_scale", "supported"), ("meaningful", "het_spread", "supported")])
    assert hv.h3({"GLK": "meaningful"}, pooled)["verdict"] == "inconclusive"


def test_h3_needs_both_contrasts_in_one_meaningful_cell():
    classes = {"GLK": "meaningful", "GLG": "meaningful", "GMG": "flat"}
    split = _contrasts([("GLK", "het_scale", "supported"), ("GLK", "het_spread", "inconclusive"),
                        ("GLG", "het_scale", "inconclusive"), ("GLG", "het_spread", "supported")])
    out = hv.h3(classes, split)
    assert out["verdict"] == "inconclusive" and out["supporting_cells"] == []
    both = _contrasts([("GLK", "het_scale", "supported"), ("GLK", "het_spread", "supported"),
                       ("GLG", "het_scale", "inconclusive"), ("GLG", "het_spread", "supported")])
    out = hv.h3(classes, both)
    assert out["verdict"] == "supported" and out["supporting_cells"] == ["GLK"]
    # two meaningful cells: the pooled "meaningful" rows cannot say that one cell holds both
    pooled = _contrasts([("meaningful", "het_scale", "supported"), ("meaningful", "het_spread", "supported")])
    assert hv.h3(classes, pooled)["verdict"] == "inconclusive"


def test_h3_flat_difference_uses_the_numbers_when_present():
    classes = {"GLK": "meaningful", "GLG": "flat"}
    rows = pd.DataFrame([
        {"cells": "GLK", "registration": "het_scale", "verdict": "supported", "delta": -0.01, "lo": -0.02, "hi": -0.005},
        {"cells": "GLK", "registration": "het_spread", "verdict": "supported", "delta": 0.01, "lo": 0.005, "hi": 0.02},
        # |delta| > MEI and the interval excludes 0, though the superiority rule is only inconclusive
        {"cells": "GLG", "registration": "het_level_flat", "verdict": "inconclusive", "delta": 0.003, "lo": 0.001,
         "hi": 0.005},
    ])
    out = hv.h3(classes, rows)
    assert out["verdict"] == "refuted" and any("GLG" in r for r in out["refuting"])
    rows.loc[2, ["lo"]] = -0.001  # the interval now covers 0
    assert hv.h3(classes, rows)["verdict"] == "supported"
    rows.loc[2, ["delta", "lo", "hi"]] = [0.0015, 0.0005, 0.0025]  # excludes 0 but |delta| <= MEI
    assert hv.h3(classes, rows)["verdict"] == "supported"


def test_h3_refuted_beats_untestable():
    classes = {"GLG": "flat", "GMG": "moderate"}
    out = hv.h3(classes, _contrasts([("GLG", "het_scale", "supported")]))
    assert out["verdict"] == "refuted" and out["refuting"]
    assert hv.h3(classes, _contrasts([("GLG", "het_scale", "inconclusive")]))["verdict"] == "untestable"
    # a pooled row alone does not refute, so no meaningful cell is still untestable
    assert hv.h3(classes, _contrasts([("flat", "het_scale", "supported")]))["verdict"] == "untestable"


# ---- H2 ------------------------------------------------------------------------------------


def test_h2_shares_task_indices():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.3, 30)
    YG = (rng.random((50, 30)) < np.clip(0.6 + base, 0.02, 0.98)).astype(float)
    YK = (rng.random((40, 30)) < np.clip(0.6 + rng.normal(0, 0.12, 40)[:, None] + base, 0.02, 0.98)).astype(float)
    out = hv.h2(YK, YG, n_boot=300, seed=1)
    assert out["verdict"] == "supported"


def test_h2_point_estimate_and_independent_prompt_draws():
    rng = np.random.default_rng(3)
    YG = (rng.random((12, 9)) < 0.5).astype(float)
    YK = (rng.random((10, 9)) < 0.5).astype(float)
    out = hv.h2(YK, YG, n_boot=50, seed=7)
    assert out["estimate"] == pytest.approx(het.variance_components(YK)["tau"] - het.variance_components(YG)["tau"])
    g = np.random.default_rng(7)
    RK, RG = g.integers(0, 10, (50, 10)), g.integers(0, 12, (50, 12))
    want = np.array([het.variance_components(YK[RK[b]])["tau"] - het.variance_components(YG[RG[b]])["tau"]
                     for b in range(50)])
    np.testing.assert_allclose(out["draws"], want)
    with pytest.raises(ValueError, match="task"):
        hv.h2(YK, YG, n_boot=5, seed=0, task_ids_K=[f"t{j}" for j in range(9)],
              task_ids_G=[f"t{j}" for j in range(1, 10)])


# ---- H1 ------------------------------------------------------------------------------------


def _tasks(n_per=4):
    return [f"task_{s}{k}" for s in "emh" for k in range(n_per)]


def _rows(pool, arms, tasks, p, rng, *, pairs=0, clock=None):
    """replicate-0 rows for every (arm, task) with success prob p[i, j]; `pairs` random replicate-1 cells;
    ``clock[i, j]`` marks a replicate-0 episode ended by the wall clock (success 0)."""
    rows = []
    for i, a in enumerate(arms):
        for j, t in enumerate(tasks):
            timed = clock is not None and bool(clock[i, j])
            rows.append({"pool": pool, "arm_id": a, "task_id": t, "replicate": 0, "attempt": 1, "status": "ok",
                         "success": 0 if timed else int(rng.random() < p[i, j]),
                         "ended_by": "clock" if timed else "agent"})
    seen = set()
    for _ in range(pairs):
        i, j = int(rng.integers(len(arms))), int(rng.integers(len(tasks)))
        if (i, j) in seen:
            continue
        seen.add((i, j))
        rows.append({"pool": pool, "arm_id": arms[i], "task_id": tasks[j], "replicate": 1, "attempt": 1,
                     "status": "ok", "success": int(rng.random() < p[i, j]), "ended_by": "agent"})
    return rows


def _h1_frames(seed=0, gl_sd=0.2, gm_sd=0.0):
    rng = np.random.default_rng(seed)
    gl_tasks, gm_tasks = _tasks(8), _tasks(5)
    glg = [f"GLG_{i:02d}" for i in range(50)]
    gl_eff = rng.normal(0, gl_sd, 50)
    p_gl = np.clip(0.5 + gl_eff[:, None] + rng.normal(0, 0.1, len(gl_tasks))[None, :], 0.02, 0.98)
    bridge_idx = rng.choice(50, 20, replace=False)
    bridge_map = {f"GMB_{j:02d}": glg[int(i)] for j, i in enumerate(bridge_idx)}
    gmb = sorted(bridge_map)
    p_gm = np.clip(0.6 + rng.normal(0, gm_sd, 20)[:, None] + rng.normal(0, 0.1, len(gm_tasks))[None, :], 0.02, 0.98)
    gl = pd.DataFrame(_rows("GLG", glg, gl_tasks, p_gl, rng))
    gm = pd.DataFrame(_rows("GMB", gmb, gm_tasks, p_gm, rng))
    return gl, gm, bridge_map, {"GLG": (glg, gl_tasks), "GMB": (gmb, gm_tasks)}


def test_h1_pairs_the_bridge_prompts_and_resamples_them_jointly():
    gl, gm, bridge_map, uni = _h1_frames()
    out = hv.h1(gl, gm, bridge_map, n_boot=200, seed=11, universe=uni)
    Ygl, arms_gl, _, _, _ = het.success_matrix(gl, "GLG")
    Ygm, arms_gm, _, _, _ = het.success_matrix(gm, "GMB")
    order = [arms_gl.index(bridge_map[b]) for b in arms_gm]
    sub = Ygl[order]
    assert out["n_prompts"] == 20
    assert out["tau_gitlab"] == pytest.approx(het.variance_components(sub)["tau"])
    assert out["estimate"] == pytest.approx(out["tau_gitlab"] - het.variance_components(Ygm)["tau"])
    R = np.random.default_rng(11).integers(0, 20, (200, 20))
    want = np.array([het.variance_components(sub[R[b]])["tau"] - het.variance_components(Ygm[R[b]])["tau"]
                     for b in range(200)])
    np.testing.assert_allclose(out["draws"], want)
    assert out["verdict"] == "supported"


def test_h1_refuses_a_bridge_map_that_does_not_cover_the_bridge_arms():
    gl, gm, bridge_map, uni = _h1_frames()
    short = dict(list(bridge_map.items())[:19])
    with pytest.raises(ValueError, match="bridge"):
        hv.h1(gl, gm, short, n_boot=5, seed=0, universe=uni)
    wrong = {**bridge_map, "GMB_00": "GLG_99"}
    with pytest.raises(ValueError, match="GLG_99"):
        hv.h1(gl, gm, wrong, n_boot=5, seed=0, universe=uni)


def test_g_index_pairing():
    pairs = hv.pair_by_g_index(["GMG_G_01", "GMG_G_00", "GMG_G_02"], ["GLG_02", "GLG_00", "GLG_01"])
    assert pairs == [("GMG_G_00", "GLG_00"), ("GMG_G_01", "GLG_01"), ("GMG_G_02", "GLG_02")]
    with pytest.raises(ValueError):
        hv.pair_by_g_index(["GMG_G_00", "GMG_G_01"], ["GLG_00"])


def test_h1_secondary_pairs_all_g_prompts_by_index():
    rng = np.random.default_rng(2)
    tasks = _tasks(4)
    eff = rng.normal(0, 0.2, 50)
    p = np.clip(0.5 + eff[:, None] + np.zeros(len(tasks))[None, :], 0.02, 0.98)
    glg, gmg = [f"GLG_{i:02d}" for i in range(50)], [f"GMG_G_{i:02d}" for i in range(50)]
    gl = pd.DataFrame(_rows("GLG", glg, tasks, p, rng))
    gm = pd.DataFrame(_rows("GMG", gmg, tasks, np.full_like(p, 0.6), rng))
    out = hv.h1_secondary(gl, gm, n_boot=100, seed=3, universe={"GLG": (glg, tasks), "GMG": (gmg, tasks)})
    assert out["n_prompts"] == 50 and out["verdict"] == "supported"


# ---- H4 ------------------------------------------------------------------------------------


def test_h4_spearman_and_bootstrap_ci():
    x = np.linspace(-0.1, 0.1, 50)
    out = hv.h4(x, 3 * x + 0.01, n_boot=200, seed=0)
    assert out["estimate"] == pytest.approx(1.0) and out["lo"] == pytest.approx(1.0)
    assert out["verdict"] == "reported"
    rng = np.random.default_rng(1)
    out = hv.h4(rng.normal(size=50), rng.normal(size=50), n_boot=300, seed=1)
    assert out["lo"] < out["estimate"] < out["hi"]
    with pytest.raises(ValueError):
        hv.h4(np.zeros(50), np.zeros(49), n_boot=5, seed=0)


# ---- per-cell analysis, noise borrow, timeout sensitivity -----------------------------------


def _study_spec():
    gm_tasks, gl_tasks = _tasks(4), _tasks(6)
    return {"GMG": ([f"GMG_G_{i:02d}" for i in range(20)], gm_tasks, 0.0, 60),
            "GMK": ([f"GMK_{i:02d}" for i in range(16)], gm_tasks, 0.15, 60),
            "GMB": ([f"GMB_{i:02d}" for i in range(8)], gm_tasks, 0.0, 0),
            "GLG": ([f"GLG_{i:02d}" for i in range(20)], gl_tasks, 0.1, 60),
            "GLK": ([f"GLK_{i:02d}" for i in range(16)], gl_tasks, 0.2, 60)}


UNI = {pool: (arms, tasks) for pool, (arms, tasks, _, _) in _study_spec().items()}


def _study_outcomes(seed=0, *, clock_rate=0.0):
    rng = np.random.default_rng(seed)
    gm_tasks, gl_tasks = _tasks(4), _tasks(6)
    rows = []
    spec = {"GMG": ([f"GMG_G_{i:02d}" for i in range(20)], gm_tasks, 0.0, 60),
            "GMK": ([f"GMK_{i:02d}" for i in range(16)], gm_tasks, 0.15, 60),
            "GMB": ([f"GMB_{i:02d}" for i in range(8)], gm_tasks, 0.0, 0),
            "GLG": ([f"GLG_{i:02d}" for i in range(20)], gl_tasks, 0.1, 60),
            "GLK": ([f"GLK_{i:02d}" for i in range(16)], gl_tasks, 0.2, 60)}
    for pool, (arms, tasks, sd, pairs) in spec.items():
        p = np.clip(0.5 + rng.normal(0, sd, len(arms))[:, None] + rng.normal(0, 0.15, len(tasks))[None, :],
                    0.02, 0.98)
        clock = rng.random(p.shape) < clock_rate if clock_rate else None
        rows += _rows(pool, arms, tasks, p, rng, pairs=pairs, clock=clock)
    return pd.DataFrame(rows)


def test_cell_noise_borrows_the_declared_donor():
    out = _study_outcomes()
    v_gmg, n_gmg = het.noise_from_pairs(out, "GMG")
    assert hv.cell_noise(out, "GMB", study=H) == (v_gmg, n_gmg, "GMG")
    v, n, src = hv.cell_noise(out, "GLK", study=H)
    assert (v, n) == het.noise_from_pairs(out, "GLK") and src == "GLK"


def test_analyze_pools_runs_stage0_per_cell_with_its_noise():
    out = _study_outcomes()
    comp = hv.analyze_pools(out, study=H, universe=UNI, tail_n_boot=6)
    assert list(comp["pool"]) == list(H.pools)
    gmb = comp.set_index("pool").loc["GMB"]
    assert gmb["noise_from"] == "GMG" and np.isfinite(gmb["tau_set_upper_one_sided"])
    for key in ("tau_main", "tau_lo", "tau_hi", "tau_set", "r_sb", "tau_discriminating", "upper_tail_mass",
                "n_imputed"):
        assert key in comp.columns


def test_success_matrix_ceiling_is_a_parameter():
    rows = _rows("P", ["a", "b", "c"], _tasks(2), np.full((3, 6), 0.5), np.random.default_rng(0))
    frame = pd.DataFrame(rows).iloc[1:]  # 1 of 18 cells missing: 5.6%
    with pytest.raises(ValueError, match="5%"):
        het.success_matrix(frame, "P")
    Y, _, _, n_imp, _ = het.success_matrix(frame, "P", max_missing=None)
    assert n_imp == 1 and np.isfinite(Y).all()


def test_timeout_sensitivity_sets_clock_episodes_missing():
    out = _study_outcomes(1, clock_rate=0.08)
    sens, rates = hv.timeout_sensitivity(out, study=H, universe=UNI)
    row = sens.set_index("pool").loc["GLK"]
    sub = out[(out["pool"] == "GLK") & (out["replicate"] == 0)]
    n_clock = int((sub["ended_by"] == "clock").sum())
    assert row["n_clock"] == n_clock and row["n_imputed"] == n_clock  # counted, beyond the 5% main ceiling
    assert row["clock_rate"] == pytest.approx(n_clock / len(sub))
    assert np.isfinite(row["tau_main"]) and np.isfinite(row["tau_main_baseline"])
    r = rates[rates["pool"] == "GLK"].set_index("arm_id")
    arm = sub["arm_id"].iloc[0]
    assert r.loc[arm, "timeout_rate"] == pytest.approx((sub[sub["arm_id"] == arm]["ended_by"] == "clock").mean())


def test_timeout_sensitivity_reports_a_cell_without_ended_by():
    out = _study_outcomes(1, clock_rate=0.05)
    out.loc[out["pool"] == "GMG", "ended_by"] = None
    sens, _ = hv.timeout_sensitivity(out, study=H, universe=UNI)
    gmg = sens.set_index("pool").loc["GMG"]
    assert not gmg["available"] and "ended_by" in gmg["note"]


def test_attach_ended_by_from_logs(tmp_path):
    snap = pd.DataFrame([{"pool": "GMG", "arm_id": "GMG_G_00", "task_id": "task_e0", "replicate": 0, "attempt": 1,
                          "status": "ok", "success": 0},
                         {"pool": "GLG", "arm_id": "GLG_00", "task_id": "task_e0", "replicate": 0, "attempt": 2,
                          "status": "ok", "success": 1}])
    old = tmp_path / "old"
    old.mkdir()
    (old / "worker_0.jsonl").write_text(json.dumps(
        {"schema": "x", "pool": "G", "arm_id": "G_00", "task_id": "task_e0", "replicate": 0, "attempt": 1,
         "status": "ok", "success": 0, "cost_usd": 0.0, "timed_out": True}) + "\n")
    new = tmp_path / "new"
    new.mkdir()
    (new / "worker_0.jsonl").write_text(
        json.dumps({"schema": "x", "pool": "GLG", "arm_id": "GLG_00", "task_id": "task_e0", "replicate": 0,
                    "attempt": 1, "status": "infra_error", "success": None, "cost_usd": 0.0, "ended_by": None}) + "\n"
        + json.dumps({"schema": "x", "pool": "GLG", "arm_id": "GLG_00", "task_id": "task_e0", "replicate": 0,
                      "attempt": 2, "status": "ok", "success": 1, "cost_usd": 0.0, "ended_by": "agent"}) + "\n")
    got = hv.attach_ended_by(snap, log_dirs=(new,), extra_log_dirs={"GMG": (old, "G")})
    assert list(got["ended_by"]) == ["clock", "agent"]


# ---- classification from the K-grid tables ------------------------------------------------


def _kgrid_rows(pool, ranges):
    rows = []
    for T, rr in ranges.items():
        for K, frac in ((2, 1.0), (8, 0.0), (32, 0.5)):
            rows.append({"env_id": f"het_{pool}_npmle", "pool": pool, "variant": "npmle", "horizon": T, "K": K,
                         "regret": 0.1 + rr * frac})
    return rows


def _boot_kgrid_rows(pool, ranges_by_boot, T=200):
    rows = []
    for b, rr in enumerate(ranges_by_boot):
        for K, frac in ((2, 1.0), (8, 0.0), (32, 0.5)):
            rows.append({"env_id": f"het_{pool}_npmle_b{b:03d}", "pool": pool, "variant": "npmle", "boot": b,
                         "horizon": T, "K": K, "regret": 0.1 + rr * frac})
    return rows


def test_boot_regret_range_lower_bound():
    boots = list(np.linspace(0.0, 0.02, 40))
    frame = pd.DataFrame(_boot_kgrid_rows("GLK", boots))
    lo = hv.boot_regret_range_lo(frame, expected_n_boot=40)
    assert lo["GLK"] == pytest.approx(np.percentile(boots, 2.5))
    with pytest.raises(ValueError, match="b039"):
        hv.boot_regret_range_lo(frame[frame["boot"] != 39], expected_n_boot=40)
    with pytest.raises(ValueError, match="K"):
        hv.boot_regret_range_lo(frame.drop(index=[0]), expected_n_boot=40)


def test_classify_pools_from_tables():
    kgrid = pd.DataFrame(_kgrid_rows("GLK", {50: 0.02, 100: 0.015, 200: 0.012})
                         + _kgrid_rows("GLG", {50: 0.001, 100: 0.002, 200: 0.003}))
    boot = pd.DataFrame([r for T in (50, 100, 200) for r in
                         _boot_kgrid_rows("GLK", [0.01 + 0.001 * b for b in range(5)], T=T)
                         + _boot_kgrid_rows("GLG", [0.001] * 5, T=T)])
    comp = pd.DataFrame([{"pool": "GLK", "tau_set_upper_one_sided": 0.1},
                         {"pool": "GLG", "tau_set_upper_one_sided": 0.02}])
    cls = hv.classify_pools(comp, kgrid, boot, tau_flat=0.03, expected_n_boot=5, manifest_sha256="m")
    c = cls.set_index("pool")
    assert c.loc["GLK", "class"] == "meaningful" and c.loc["GLG", "class"] == "flat"
    ranges = [0.01 + 0.001 * b for b in range(5)]
    assert c.loc["GLK", "regret_range_T200"] == pytest.approx(0.012)
    for T in (50, 100, 200):
        assert c.loc["GLK", f"rr_lo_T{T}"] == pytest.approx(np.percentile(ranges, 2.5))
        assert c.loc["GLK", f"rr_hi_T{T}"] == pytest.approx(np.percentile(ranges, 97.5))
    assert set(cls["manifest_sha256"]) == {"m"} and set(cls["tau_flat"]) == {0.03}
    # the tau_set bound is reported beside the class, never decisive (C1 ruling)
    assert c.loc["GLG", "tau_set_upper_one_sided"] == 0.02 and bool(c.loc["GLG", "tau_bound_below_flat"])
    assert not bool(c.loc["GLK", "tau_bound_below_flat"])
    with pytest.raises(ValueError, match="T=50"):  # the CI is needed at every primary T
        hv.classify_pools(comp, kgrid, boot[boot["horizon"] != 50], tau_flat=0.03, expected_n_boot=5,
                          manifest_sha256="m")


def test_a_flat_cell_whose_tau_bound_exceeds_tau_flat_is_still_flat():
    # C1: the regret-range rule decides; tau_set's upper bound above tau_flat is only reported
    kgrid = pd.DataFrame(_kgrid_rows("GMK", {50: 0.002, 100: 0.003, 200: 0.004}))
    boot = pd.DataFrame([r for T in (50, 100, 200) for r in _boot_kgrid_rows("GMK", [0.002, 0.004, 0.006, 0.008, 0.009], T=T)])
    comp = pd.DataFrame([{"pool": "GMK", "tau_set_upper_one_sided": 0.2}])
    row = hv.classify_pools(comp, kgrid, boot, tau_flat=0.017, expected_n_boot=5, manifest_sha256="m").iloc[0]
    assert row["class"] == "flat" and not bool(row["tau_bound_below_flat"])
    assert row["rr_hi_T200"] < 0.01
    # the same cell with a T=200 upper bound above 0.01 is moderate
    boot_wide = pd.DataFrame([r for T in (50, 100, 200) for r in _boot_kgrid_rows("GMK", [0.002, 0.004, 0.011, 0.011, 0.011], T=T)])
    assert hv.classify_pools(comp, kgrid, boot_wide, tau_flat=0.017, expected_n_boot=5,
                             manifest_sha256="m").iloc[0]["class"] == "moderate"


def test_classification_reads_only_the_T200_bounds():
    kgrid = pd.DataFrame(_kgrid_rows("GLK", {50: 0.02, 100: 0.015, 200: 0.012}))
    comp = pd.DataFrame([{"pool": "GLK", "tau_set_upper_one_sided": 0.1}])
    for lo200, want in ((0.006, "meaningful"), (0.004, "moderate")):
        boot = pd.DataFrame(_boot_kgrid_rows("GLK", [0.0] * 5, T=50) + _boot_kgrid_rows("GLK", [0.0] * 5, T=100)
                            + _boot_kgrid_rows("GLK", [lo200] * 5, T=200))
        assert hv.classify_pools(comp, kgrid, boot, tau_flat=0.03, expected_n_boot=5,
                                 manifest_sha256="m").iloc[0]["class"] == want


# ---- registered_contrast: rules, registrations, the classification-selected cells ----------


def test_superior_and_inferior_rules():
    v = rc.verdict_by_rule
    assert {"superior", "inferior"} <= set(rc.RULES)
    assert v("superior", delta=-0.004, lo=-0.006, hi=-0.0021, mei=0.002) == "supported"
    assert v("superior", delta=-0.003, lo=-0.006, hi=-0.002, mei=0.002) == "inconclusive"  # hi < -MEI, strict
    assert v("superior", delta=0.004, lo=0.0021, hi=0.006, mei=0.002) == "reversed"
    assert v("inferior", delta=0.004, lo=0.0021, hi=0.006, mei=0.002) == "supported"
    assert v("inferior", delta=0.003, lo=0.002, hi=0.006, mei=0.002) == "inconclusive"
    assert v("inferior", delta=-0.004, lo=-0.006, hi=-0.0021, mei=0.002) == "reversed"


def test_het_registrations_are_written_down():
    expected = {"het_scale": ("p3_star", "fixed_K8", "superior"),
                "het_spread": ("always_search", "p3_star", "inferior"),
                "het_primary": ("p3_star", "fixed_K_star", "noninferiority"),
                "het_level": ("level_star", "p3_star", "not_better"),
                "het_phi": ("phi_k4", "p3_star", "not_better")}
    for name, (policy, reference, rule) in expected.items():
        for key, cells in ((name, "meaningful"), (f"{name}_flat", "flat")):
            reg = rc.REGISTRATIONS[key]
            assert (reg["policy"], reg["reference"], reg["rule"]) == (policy, reference, rule)
            assert reg["interval"] == "prompt_bootstrap" and reg["study"] == "het" and reg["n_boot"] == 200
            assert reg["horizons"] == (50, 100, 200) and reg["mei"] == 0.002 and reg["cells"] == cells
            assert reg["test"] == "het"
    # Pre-registration 9's registrations are untouched
    assert "cells" not in rc.REGISTRATIONS["emp_primary"] and "study" not in rc.REGISTRATIONS["emp_primary"]


def _cell(root, test, env, T, seed, deltas, n=10):
    for policy, shift in deltas.items():
        path = root / "episodes" / test / f"{env}_T{T}_cap{T}" / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"env_id": env, "horizon": T, "cap": T, "base_seed": seed, "episode": np.arange(n),
                      COL: np.full(n, 0.1) + shift}).to_parquet(path, index=False)


def _het_tree(root, point, boots, classes, policy="p3_star", reference="fixed_K8"):
    """point[pool] / boots[pool][b]: the policy's shift over the reference in that cell (every T)."""
    flat = []
    for pool in classes:
        env = f"het_{pool}_npmle"
        for T in (50, 100, 200):
            flat.append({"env_id": env, "pool": pool, "variant": "npmle", "horizon": T, "regret_range": 0.001,
                         "informative": False})  # the Pre-reg 9 guard would find nothing informative
            _cell(root, "het", env, T, 1, {policy: point[pool], reference: 0.0})
            for b, d in enumerate(boots[pool]):
                _cell(root, "het_boot", f"{env}_b{b:03d}", T, 100 + b, {policy: d, reference: 0.0})
    (root / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(root / "tables" / "het_flatness.csv", index=False)
    path = root / "het_classification.csv"
    pd.DataFrame([{"pool": p, "class": c, "manifest_sha256": "m"} for p, c in classes.items()]).to_csv(path, index=False)
    return path


def _het_run(root, cls_path, cells, rule="superior", n_boot=4, **kw):
    return rc.prompt_bootstrap_contrast("p3_star", "fixed_K8", horizons=(50, 100, 200), mei=0.002, rule=rule,
                                        out_dir=root, expected_n_boot=n_boot, study=H, cells=cells,
                                        classification_path=cls_path, **kw).iloc[0]


def test_het_contrast_selects_cells_from_the_classification(tmp_path):
    classes = {"GLK": "meaningful", "GLG": "meaningful", "GMG": "flat", "GMK": "moderate"}
    point = {"GLK": -0.01, "GLG": -0.006, "GMG": 0.0, "GMK": 0.05}
    boots = {"GLK": [-0.01, -0.009, -0.011, -0.012], "GLG": [-0.006, -0.005, -0.007, -0.004],
             "GMG": [0.0005, -0.0005, 0.0, 0.001], "GMK": [0.5, 0.5, 0.5, 0.5]}
    cls = _het_tree(tmp_path, point, boots, classes)
    row = _het_run(tmp_path, cls, "meaningful")
    assert row["n_informative"] == 6 and row["cells"] == "meaningful" and row["as_registered"]
    assert row["delta"] == pytest.approx(-0.008)
    stats_b = [np.mean([boots["GLK"][b], boots["GLG"][b]]) for b in range(4)]
    assert row["hi"] == pytest.approx(np.percentile(stats_b, 97.5)) and row["verdict"] == "supported"
    one = _het_run(tmp_path, cls, "GLG")
    assert one["n_informative"] == 3 and one["delta"] == pytest.approx(-0.006) and one["as_registered"]
    flat = _het_run(tmp_path, cls, "flat")
    assert flat["informative_cells"].startswith("het_GMG_npmle@") and flat["verdict"] == "inconclusive"
    moderate = _het_run(tmp_path, cls, "GMK")
    assert not moderate["as_registered"]  # no registration covers moderate cells
    with pytest.raises(ValueError, match="cells"):
        _het_run(tmp_path, cls, "nonsense")


def test_het_contrast_with_no_cell_of_the_class(tmp_path):
    classes = {"GLK": "moderate", "GLG": "flat"}
    cls = _het_tree(tmp_path, {"GLK": 0.0, "GLG": 0.0}, {"GLK": [0.0] * 4, "GLG": [0.0] * 4}, classes)
    row = _het_run(tmp_path, cls, "meaningful")
    assert row["verdict"] == "no_cells" and row["n_informative"] == 0


def test_het_contrast_refuses_a_classification_from_another_snapshot(tmp_path):
    classes = {"GLK": "meaningful", "GLG": "flat"}
    cls = _het_tree(tmp_path, {"GLK": -0.01, "GLG": 0.0}, {"GLK": [-0.01] * 4, "GLG": [0.0] * 4}, classes)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"reservoirs": {"GLK_npmle": "k", "GLG_npmle": "g"}}))
    flat = pd.read_csv(tmp_path / "tables" / "het_flatness.csv")
    flat["reservoir_sha256"] = flat["env_id"].map(lambda e: {"het_GLK_npmle": "k", "het_GLG_npmle": "g"}[e])
    flat.to_csv(tmp_path / "tables" / "het_flatness.csv", index=False)
    for test in ("het", "het_boot"):
        rc.write_reservoir_stamp(tmp_path, test, manifest)
    with pytest.raises(ValueError, match="classification"):
        _het_run(tmp_path, cls, "meaningful", reservoir_manifest=manifest)
    frame = pd.read_csv(cls)
    frame["manifest_sha256"] = rc.file_sha256(manifest)
    frame.to_csv(cls, index=False)
    assert _het_run(tmp_path, cls, "meaningful", reservoir_manifest=manifest)["verdict"] == "supported"


def test_policy_gaps_runs_pooled_and_per_cell_rows(tmp_path):
    classes = {"GLK": "meaningful", "GLG": "flat"}
    cls = _het_tree(tmp_path, {"GLK": -0.01, "GLG": 0.0}, {"GLK": [-0.01] * 4, "GLG": [0.0] * 4}, classes)
    gaps = hv.policy_gaps(tmp_path, classes, cls, registrations=["het_scale", "het_scale_flat"],
                          expected_n_boot=4)
    got = {(r.registration, r.cells): r.verdict for r in gaps.itertuples()}
    assert got == {("het_scale", "meaningful"): "supported", ("het_scale", "GLK"): "supported",
                   ("het_scale_flat", "flat"): "inconclusive", ("het_scale_flat", "GLG"): "inconclusive"}


# ---- replay boot_kgrid -----------------------------------------------------------------------


def _tiny_study(tmp_path):
    rng = np.random.default_rng(5)

    def pool_rows(pool, n_arms=6, n_tasks=12, n_reps=12):  # k -> (k % 6, k % 12): distinct for k < 12
        rows = []
        mus = rng.uniform(0.3, 0.8, n_arms)
        for i, mu in enumerate(mus):
            for t in range(n_tasks):
                rows.append({"schema": "empirical_pool/1", "cost_usd": 0.0, "pool": pool, "arm_id": f"{pool}_{i:02d}",
                             "task_id": f"t{t}", "replicate": 0, "attempt": 1, "status": "ok",
                             "success": int(rng.random() < mu)})
        for k in range(n_reps):
            i, t = k % n_arms, k % n_tasks
            rows.append({"schema": "empirical_pool/1", "cost_usd": 0.0, "pool": pool, "arm_id": f"{pool}_{i:02d}",
                         "task_id": f"t{t}", "replicate": 1, "attempt": 1, "status": "ok",
                         "success": int(rng.random() < mus[i])})
        return rows

    log = tmp_path / "logs" / "worker_0.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in pool_rows("A") + pool_rows("B")))
    s = st.Study(name="x", test="xt", boot_test="xt_boot", pools=("A", "B"), data_dir=tmp_path,
                 res_dir=tmp_path / "res", log_dirs=(tmp_path / "logs",), seed_base=900_000_000_000,
                 table_prefix="xt", noise_mode="per_pool", borrowed_noise={}, extra_outcomes=())
    replay.run_estimate(None, s.res_dir, tmp_path / "out", study=s)
    return s


def test_boot_kgrid_runs_every_bootstrap_reservoir(tmp_path):
    s = _tiny_study(tmp_path)
    frame = replay.run_boot_kgrid(s.res_dir, workers=1, study=s, n_boot=2, horizons=(20,), n_replicates=8,
                                  k_grid=(2, 4, 8))
    assert sorted({(int(b), p) for b, p in zip(frame["boot"], frame["pool"], strict=True)}) == [
        (0, "A"), (0, "B"), (1, "A"), (1, "B")]
    assert set(frame["horizon"]) == {20} and sorted(set(frame["K"])) == [2, 4, 8]
    assert set(frame["manifest_sha256"]) == {rc.file_sha256(s.res_dir / "manifest.json")}
    assert set(frame["split"]) == {"xt_boot"}
    boots = replay.bootstrap_reservoirs(replay.load_snapshot(s.res_dir),
                                        json.loads((s.res_dir / "noise.json").read_text()), n_boot=2, study=s)
    want = {replay.make_emp_cell(p, "npmle", 20, r, 8, boot=b, study=s).base_seed for b, p, r in boots}
    assert set(frame["base_seed"]) == want  # the boot stage's own seeds and env ids
    assert all(rc.boot_env_pattern(s).match(e) for e in frame["env_id"])


def test_boot_kgrid_refuses_a_saved_boot_reservoir_that_differs(tmp_path):
    s = _tiny_study(tmp_path)
    boot_dir = s.res_dir / "boot"
    boot_dir.mkdir(parents=True)
    (boot_dir / "A_npmle_b000.json").write_text('{"type": "empirical", "atoms": [0.5], "weights": [1.0]}')
    with pytest.raises(ValueError, match="A_npmle_b000"):
        replay.run_boot_kgrid(s.res_dir, workers=1, study=s, n_boot=1, horizons=(20,), n_replicates=8, k_grid=(2,))


def test_boot_kgrid_is_a_replay_stage(tmp_path, monkeypatch):
    seen = {}

    def fake(res_dir, *, workers, study, **kw):
        seen.update(res_dir=res_dir, study=study.name)
        return pd.DataFrame([{"env_id": "het_GLK_npmle_b000", "boot": 0, "K": 2, "regret": 0.1}])

    monkeypatch.setattr(replay, "run_boot_kgrid", fake)
    replay.main(["boot_kgrid", "--study", "het", "--out-dir", str(tmp_path), "--workers", "1"])
    assert seen["study"] == "het"
    assert (tmp_path / "tables" / "het_boot_kgrid.csv").exists()


def test_cli_requires_tau_flat(tmp_path):
    with pytest.raises(SystemExit):
        hv.main(["--out-dir", str(tmp_path), "--results-dir", str(tmp_path / "r")])


def test_hypotheses_rows_and_portability():
    out = _study_outcomes(4)
    bridge = {f"GMB_{j:02d}": f"GLG_{2 * j:02d}" for j in range(8)}
    rows, port = hv.hypotheses(out, bridge, universe=UNI, n_boot=50, seed=9)
    got = [(r["hypothesis"], r["role"], r["decides"]) for r in rows]
    assert got == [("H1", "primary", True), ("H1", "secondary", False), ("H2", "Gmail", True),
                   ("H2", "GitLab", True), ("H4", "reported", False)]
    assert rows[0]["n_prompts"] == 8 and rows[1]["n_prompts"] == 20
    assert all(r["verdict"] in ("supported", "refuted", "inconclusive", "reported") for r in rows)
    assert list(port["g_index"]) == list(range(20)) and list(port["arm_gmail"])[3] == "GMG_G_03"


def test_h4_is_the_blup_rank_correlation_and_refuses_constant_effects():
    rng = np.random.default_rng(6)
    Ya = (rng.random((50, 12)) < np.clip(0.5 + rng.normal(0, 0.2, 50)[:, None], 0.02, 0.98)).astype(float)
    Yb = (rng.random((50, 12)) < np.clip(0.5 + rng.normal(0, 0.2, 50)[:, None], 0.02, 0.98)).astype(float)
    from scipy import stats as sps
    blup = sps.spearmanr(het.prompt_effects(Ya), het.prompt_effects(Yb)).statistic
    raw = hv.h4(Ya.mean(axis=1) - Ya.mean(), Yb.mean(axis=1) - Yb.mean(), n_boot=20, seed=0)["estimate"]
    assert raw == pytest.approx(blup)
    with pytest.raises(ValueError, match="undefined"):
        hv.h4(np.zeros(50), np.arange(50.0), n_boot=20, seed=0)


# ---- fix round 1 ------------------------------------------------------------------------------


def test_verdict_seed_is_the_ruled_base():
    assert hv.VERDICT_SEED == 60_000_000_000


def _write_design(tmp_path, arms_by_pool, gm_tasks, gl_tasks):
    import yaml
    for pool, arms in arms_by_pool.items():
        path = tmp_path / "pools" / f"{pool}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"pool": pool, "arms": [{"arm_id": a} for a in arms]}))
    (tmp_path / "manifest.json").write_text(json.dumps({"gmail_task_subset_30": gm_tasks,
                                                        "gitlab_task_subset_60": gl_tasks}))


def test_design_universe_comes_from_pool_files_and_the_manifest(tmp_path):
    import dataclasses

    import yaml
    spec = _study_spec()
    arms = {p: a for p, (a, *_) in spec.items() if p != "GMG"}
    _write_design(tmp_path, arms, spec["GMK"][1], spec["GLK"][1])
    g = tmp_path / "pool_G.yaml"
    g.write_text(yaml.safe_dump({"pool": "G", "arms": [{"arm_id": f"G_{i:02d}"} for i in range(20)]}))
    study = dataclasses.replace(H, data_dir=tmp_path)
    # GMG's arms come from the Pre-reg 9 pool file (`EXTRA_POOL_FILES`), pointed at a tmp copy here
    import het_verdicts
    old = dict(het_verdicts.EXTRA_POOL_FILES)
    het_verdicts.EXTRA_POOL_FILES["GMG"] = g
    try:
        with pytest.raises(ValueError, match="task universe"):
            hv.design_universe(study)  # the GitLab universe is never assumed
        uni = hv.design_universe(study, task_universe="subset_60")
    finally:
        het_verdicts.EXTRA_POOL_FILES.clear()
        het_verdicts.EXTRA_POOL_FILES.update(old)
    assert uni["GMG"] == (spec["GMG"][0], spec["GMG"][1])  # relabelled GMG_G_xx, Gmail tasks
    assert uni["GLK"] == (spec["GLK"][0], spec["GLK"][1]) and uni["GMB"][1] == spec["GMB"][1]


def test_a_never_attempted_arm_or_task_is_missing_not_dropped():
    out = _study_outcomes()
    gone = out[out["arm_id"] != "GLK_03"]  # the snapshot has no row at all for one arm of the pool file
    with pytest.raises(ValueError, match="GLK_03"):
        hv.analyze_pools(gone, study=H, universe=UNI, tail_n_boot=2)
    # one never-attempted cell counts toward the 5% ceiling as an imputed cell
    one = out[~((out["arm_id"] == "GLK_03") & (out["task_id"] == "task_e0") & (out["replicate"] == 0))]
    Y, arms, tasks, n_imp, _ = hv._matrix(one, "GLK", UNI)
    assert Y.shape == (16, 18) and n_imp == 1 and arms == UNI["GLK"][0]
    # a task in the manifest subset that no episode reached: missing, never silently dropped
    uni = {**UNI, "GLK": (UNI["GLK"][0], [*UNI["GLK"][1], "task_h9"])}
    with pytest.raises(ValueError, match="task_h9"):
        hv._matrix(out, "GLK", uni)


def test_upper_tail_is_deconvolved_with_execution_noise_like_the_replay_reservoir():
    out = _study_outcomes()
    Y, _, tasks, n_imp, rcs = hv._matrix(out, "GLK", UNI)
    v, n_pairs = het.noise_from_pairs(out, "GLK")
    assert n_pairs > 0
    # the replay reservoir's deconvolution: per-prompt variance v / n (emp.npmle_reservoir)
    grid, w, _ = hv.emp.npmle(Y.mean(axis=1), v / rcs)
    want = float(w[grid >= hv.stage0._weighted_median(grid, w) + 0.10].sum())
    assert hv.upper_tail_mass(Y, rcs, v) == pytest.approx(want)
    assert np.isnan(hv.upper_tail_mass(Y, rcs, float("nan")))
    draws = het.prompt_bootstrap(Y, lambda Yb, r: hv.upper_tail_mass(Yb, r, v), n_boot=6, seed=5, row_counts=rcs)
    lo, hi = hv.upper_tail_ci(Y, rcs, v, n_boot=6, seed=5)
    assert (lo, hi) == (pytest.approx(np.percentile(draws, 2.5)), pytest.approx(np.percentile(draws, 97.5)))
    comp = hv.analyze_pools(out, study=H, universe=UNI, tail_n_boot=6).set_index("pool")
    assert comp.loc["GLK", "upper_tail_mass"] == pytest.approx(want)
    assert comp.loc["GLK", "upper_tail_noise_var"] == pytest.approx(v)
    # GMB has no pairs: it deconvolves with its declared borrow (GMG), as the replay reservoir does
    assert comp.loc["GMB", "upper_tail_noise_var"] == pytest.approx(het.noise_from_pairs(out, "GMG")[0])
    assert comp.loc["GLK", "upper_tail_seed"] == hv.VERDICT_SEED + hv.TAIL_SEED_OFFSET + H.pools.index("GLK")
    assert (comp["upper_tail_mass_lo"] <= comp["upper_tail_mass_hi"]).all()


def test_timeout_sensitivity_drops_a_prompt_that_always_timed_out():
    out = _study_outcomes(2, clock_rate=0.03)
    arm = "GLK_05"
    sel = (out["pool"] == "GLK") & (out["arm_id"] == arm) & (out["replicate"] == 0)
    out.loc[sel, "ended_by"] = "clock"
    out.loc[sel, "success"] = 0
    sens, _ = hv.timeout_sensitivity(out, study=H, universe=UNI)
    row = sens.set_index("pool").loc["GLK"]
    assert bool(row["available"]) and row["n_prompts_dropped"] == 1 and row["dropped_prompts"] == arm
    assert row["n_prompts"] == 15
    # a task every prompt timed out on still leaves the matrix structurally short: NA, with the reason
    out.loc[(out["pool"] == "GLG") & (out["task_id"] == "task_e0") & (out["replicate"] == 0), "ended_by"] = "clock"
    sens, _ = hv.timeout_sensitivity(out, study=H, universe=UNI)
    glg = sens.set_index("pool").loc["GLG"]
    assert not bool(glg["available"]) and "task_e0" in glg["note"]


def test_gmg_timeouts_are_na_with_a_note_when_prereg9_logs_are_absent(tmp_path):
    out = _study_outcomes(1)
    got = hv.attach_ended_by(out.drop(columns=["ended_by"]), log_dirs=(),
                             extra_log_dirs={"GMG": (tmp_path / "absent", "G")})
    assert got["ended_by"].isna().all()  # nothing fabricated
    got.loc[got["pool"] != "GMG", "ended_by"] = out.loc[out["pool"] != "GMG", "ended_by"]
    sens, _ = hv.timeout_sensitivity(got, study=H, universe=UNI)
    assert sens.set_index("pool").loc["GLK", "available"]
    gmg = sens.set_index("pool").loc["GMG"]
    assert not bool(gmg["available"]) and "not imputed" in gmg["note"]


def test_gmg_timed_out_is_joined_after_the_relabel(tmp_path):
    snap = pd.DataFrame([{"pool": "GMG", "arm_id": f"GMG_G_0{k}", "task_id": "task_e0", "replicate": r,
                          "attempt": 1, "status": "ok", "success": 0} for k in range(2) for r in (0, 1)])
    old = tmp_path / "old"
    old.mkdir()
    lines = [{"schema": "x", "pool": "G", "arm_id": f"G_0{k}", "task_id": "task_e0", "replicate": r, "attempt": 1,
              "status": "ok", "success": 0, "cost_usd": 0.0, "timed_out": bool(k == 1 and r == 0)}
             for k in range(2) for r in (0, 1)]
    lines.append({**lines[0], "pool": "F", "arm_id": "G_00"})  # another pool's record never leaks in
    (old / "worker_0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines[:4]))
    (old / "worker_1.jsonl").write_text(json.dumps({**lines[4], "arm_id": "F_00"}) + "\n")
    got = hv.attach_ended_by(snap, log_dirs=(), extra_log_dirs={"GMG": (old, "G")})
    assert list(got["ended_by"]) == ["not_clock", "not_clock", "clock", "not_clock"]


def test_boot_kgrid_covers_every_primary_horizon_with_the_boot_stage_cells(tmp_path):
    s = _tiny_study(tmp_path)
    frame = replay.run_boot_kgrid(s.res_dir, workers=1, study=s, n_boot=1, horizons=(10, 20), n_replicates=4,
                                  k_grid=(2, 8))
    assert sorted(set(frame["horizon"])) == [10, 20]
    for T in (10, 20):
        seeds = set(frame.loc[frame["horizon"] == T, "base_seed"])
        assert seeds == {replay.seed_for(p, T, 0, study=s) for p in s.pools}
    point = {replay.seed_for(p, T, None, study=s) for p in s.pools for T in replay.ALL_HORIZONS}
    assert not set(frame["base_seed"]) & point
    assert replay.BOOT_KGRID_HORIZONS == (50, 100, 200)


def _cluster_tree(root, test, policy, reference):
    rng = np.random.default_rng(0)
    for i in range(6):
        for T in (50, 100, 200):
            base = rng.normal(0.12, 0.02, 40)
            for pol, shift in ((policy, -0.004), (reference, 0.0)):
                path = root / "episodes" / test / f"env{i}_T{T}_cap{T}" / f"{pol}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"env_id": f"env{i}", "horizon": T, "cap": T, "base_seed": 100 + i, "episode": np.arange(40),
                              COL: base + shift}).to_parquet(path, index=False)


def test_cluster_t_path_never_marks_a_het_tuple_registered(tmp_path):
    _cluster_tree(tmp_path, "het", "p3_star", "fixed_K8")
    out = rc.registered_contrast("p3_star", "fixed_K8", test="het", horizons=(50, 100, 200), mei=0.002,
                                 out_dir=tmp_path, n_boot=200, rule="superior")
    assert not out["as_registered"].any()
    _cluster_tree(tmp_path, "robust", "phi_k4", "p3_star")  # Pre-registration 2's tuple is still registered
    out = rc.registered_contrast("phi_k4", "p3_star", test="robust", horizons=(50, 100, 200), mei=0.002,
                                 out_dir=tmp_path, n_boot=200)
    assert out["as_registered"].all()
    _cluster_tree(tmp_path, "emp", "p3_star", "fixed_K_star")  # a Pre-reg 9 tuple belongs to the bootstrap path
    out = rc.registered_contrast("p3_star", "fixed_K_star", test="emp", horizons=(50, 100, 200), mei=0.002,
                                 out_dir=tmp_path, n_boot=200, rule="noninferiority")
    assert not out["as_registered"].any()


# ---- final fix wave: anchor recovery (I2), tau_flat provenance (I3), task universe (I1) -----------


def _anchor_outcomes(oracle, explorer, bulk_rates, tasks, gm_rate=0.6):
    rows = []
    arms = {**{f"GLG_{i:02d}": r for i, r in enumerate(bulk_rates)},
            "GL_anchor_oracle": oracle, "GL_anchor_explorer": explorer, "GL_anchor_baseline": 0.5}
    for arm, rate in arms.items():
        k = int(round(rate * len(tasks)))
        for j, t in enumerate(tasks):
            rows.append({"pool": "GLG" if arm.startswith("GLG") else "anchor", "arm_id": arm, "task_id": t,
                         "replicate": 0, "status": "ok", "success": int(j < k)})
    for j, t in enumerate(_tasks(2)):
        rows.append({"pool": "anchor", "arm_id": "GM_anchor_baseline", "task_id": t, "replicate": 0, "status": "ok",
                     "success": int(j < round(gm_rate * 6))})
    return pd.DataFrame(rows)


def test_anchor_recovery_rule_and_percentiles(tmp_path):
    tasks = [f"task_e{i}" for i in range(10)]
    bulk = [0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8]
    uni = {"GLG": ([f"GLG_{i:02d}" for i in range(8)], tasks), "GLK": ([], tasks), "GMK": ([], _tasks(2))}
    p10, p90 = np.percentile(bulk, 10), np.percentile(bulk, 90)
    row = hv.anchor_recovery(_anchor_outcomes(0.9, 0.1, bulk, tasks), universe=uni, prereg9_snapshot=None).iloc[0]
    assert (row["bulk_p10"], row["bulk_p90"]) == (pytest.approx(p10), pytest.approx(p90))
    assert row["recovered"] and row["oracle_rate"] == 0.9 and row["explorer_rate"] == 0.1
    assert row["bulk_n_prompts"] == 8 and row["n_tasks"] == 10
    # the oracle not above the 90th percentile (0.7 < p90 = 0.73) -> not recovered
    assert not hv.anchor_recovery(_anchor_outcomes(0.7, 0.1, bulk, tasks), universe=uni,
                                  prereg9_snapshot=None).iloc[0]["recovered"]
    # the explorer not below the 10th percentile
    assert not hv.anchor_recovery(_anchor_outcomes(0.9, 0.3, bulk, tasks), universe=uni,
                                  prereg9_snapshot=None).iloc[0]["recovered"]
    # an anchor with no episode is never recovered
    out = _anchor_outcomes(0.9, 0.1, bulk, tasks)
    none = hv.anchor_recovery(out[out["arm_id"] != "GL_anchor_oracle"], universe=uni, prereg9_snapshot=None).iloc[0]
    assert not none["recovered"] and none["oracle_n"] == 0
    # rates are read on the analysis universe only (block A under the fallback)
    uni_a = {"GLG": (uni["GLG"][0], tasks[:5]), "GLK": ([], tasks[:5]), "GMK": ([], _tasks(2))}
    assert hv.anchor_recovery(out, universe=uni_a, prereg9_snapshot=None).iloc[0]["n_tasks"] == 5


def test_anchor_recovery_reports_the_gmail_anchor_beside_prereg9s(tmp_path):
    import hashlib
    tasks = [f"task_e{i}" for i in range(10)]
    uni = {"GLG": ([f"GLG_{i:02d}" for i in range(4)], tasks), "GLK": ([], tasks), "GMK": ([], _tasks(2))}
    snap = tmp_path / "outcomes_snapshot.jsonl"
    rows = [{"arm_id": "anchor_baseline", "pool": "anchor", "replicate": 0, "status": "ok", "task_id": t,
             "success": int(j < 3), "attempt": 1} for j, t in enumerate(_tasks(2))]
    snap.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "manifest.json").write_text(json.dumps({"outcomes_snapshot": {
        "file": snap.name, "sha256": hashlib.sha256(snap.read_bytes()).hexdigest()}}))
    row = hv.anchor_recovery(_anchor_outcomes(0.9, 0.1, [0.4, 0.5, 0.5, 0.6], tasks, gm_rate=4 / 6),
                             universe=uni, prereg9_snapshot=snap).iloc[0]
    assert row["prereg9_anchor_baseline_rate"] == pytest.approx(0.5) and row["prereg9_anchor_baseline_n"] == 6
    assert row["gm_anchor_baseline_rate"] == pytest.approx(4 / 6)
    assert row["gm_anchor_drift"] == pytest.approx(4 / 6 - 0.5)
    snap.write_text(snap.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        hv.anchor_recovery(_anchor_outcomes(0.9, 0.1, [0.5] * 4, tasks), universe=uni, prereg9_snapshot=snap)


def _calibration(tmp_path):
    rows = []
    for sd, rr in ((0.01, 0.001), (0.02, 0.003), (0.035, 0.007), (0.05, 0.012)):
        rows.append({"pool_id": f"beta_sd{sd}", "family": "beta", "true_sd": sd, "horizon": 200, "regret_range": rr})
    path = tmp_path / "calibration.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_tau_flat_is_recomputed_from_the_calibration_and_a_mismatch_refuses(tmp_path):
    path = _calibration(tmp_path)
    tau = 0.02 + (0.005 - 0.003) / (0.007 - 0.003) * 0.015
    got = hv.registered_tau_flat(tau, path)
    assert got["tau_flat"] == pytest.approx(tau) and got["calibration_sha256"] == rc.file_sha256(path)
    with pytest.raises(ValueError, match="tau-flat"):
        hv.registered_tau_flat(tau + 1e-9, path)
    with pytest.raises(FileNotFoundError):
        hv.registered_tau_flat(tau, tmp_path / "absent.csv")


def test_provenance_stamps_unregistered_seed_or_n_boot():
    base = dict(task_universe="block_a", tau_flat=0.02, calibration_sha256="c")
    assert not hv.provenance(**base, seed=hv.VERDICT_SEED, n_boot=hv.N_BOOT)["unregistered"]
    assert hv.provenance(**base, seed=hv.VERDICT_SEED + 1, n_boot=hv.N_BOOT)["unregistered"]
    assert hv.provenance(**base, seed=hv.VERDICT_SEED, n_boot=100)["unregistered"]
    frame = hv.stamp(pd.DataFrame({"pool": ["GLK"]}), hv.provenance(**base, seed=1, n_boot=2))
    assert list(frame.columns) == ["pool", *hv.PROVENANCE_COLUMNS]
    assert frame.iloc[0]["task_universe"] == "block_a" and frame.iloc[0]["calibration_sha256"] == "c"
    with pytest.raises(ValueError, match="tau_flat"):
        hv.stamp(pd.DataFrame({"tau_flat": [0.03]}), hv.provenance(**base, seed=1, n_boot=2))


def test_design_universe_detects_the_block_a_fallback(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    import het_fake_tree
    s, pool_g = het_fake_tree.build(tmp_path, drop_block_b=("GL_anchor_explorer",))
    old = dict(hv.EXTRA_POOL_FILES)
    hv.EXTRA_POOL_FILES["GMG"] = pool_g
    try:
        out = st.load_study_outcomes(s)
        uni = hv.design_universe(s, outcomes=out)
        forced = hv.design_universe(s, task_universe="subset_60")
    finally:
        hv.EXTRA_POOL_FILES.clear()
        hv.EXTRA_POOL_FILES.update(old)
    assert uni["GLG"][1] == het_fake_tree.GL_BLOCK_A and uni["GLK"][1] == het_fake_tree.GL_BLOCK_A
    assert uni["GMK"][1] == het_fake_tree.GM_TASKS  # Gmail cells never fall back
    assert forced["GLG"][1] == het_fake_tree.GL_BLOCK_A + het_fake_tree.GL_BLOCK_B
