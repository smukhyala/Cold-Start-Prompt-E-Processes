"""Pre-registration 11's gate script (`glk_gate`): the tier / gate / Stage B rules on hand-built numbers, the
bank weighting, the leave-one-out diagnostics, the T = 40 matrix replay, and one end-to-end run on a tiny fake
GitLab tree (estimate -> K-grid -> boot K-grid -> point / boot episodes -> describe -> gate), sizes shrunk only
through the functions' own parameters."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("experiments/growing_bandits/empirical", "experiments/growing_bandits/deploy", "tests", "src"):
    sys.path.insert(0, str(ROOT / sub))

import calibrate  # noqa: E402
import describe  # noqa: E402
import glk_gate as gg  # noqa: E402
import het_fake_tree  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402
import study as st  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402

DEPLOY = ROOT / "results" / "growing_bandits" / "deploy"
RR_DR = {50: 0.03, 100: 0.03, 200: 0.03}
DR_KW = dict(gap_k8=0.01, gap_always_search=0.02, drop2_range_t200=0.02, r_sb=0.5, split_half_p=0.01)


# ---- tier / gate -------------------------------------------------------------------------------------


def test_tier_branches():
    assert gg.tier_of({50: 0.001, 100: 0.002, 200: 0.003}, 0.0, 0.006, **DR_KW)[:2] == ("flat", "flat")
    assert gg.tier_of({50: 0.004, 100: 0.008, 200: 0.009}, 0.002, 0.02, **DR_KW)[:2] == ("moderate", "moderate")
    assert gg.tier_of({50: 0.012, 100: 0.013, 200: 0.015}, 0.008, 0.03, **DR_KW)[:2] == ("meaningful", "meaningful")
    assert gg.tier_of(RR_DR, 0.015, 0.05, **DR_KW)[:2] == ("meaningful", "decision_relevant")


@pytest.mark.parametrize("override,needle", [
    ({"drop2_range_t200": 0.01}, "drop2"),
    ({"r_sb": float("nan")}, "split-half"),
    ({"split_half_p": 0.2}, "split-half"),
    ({"gap_k8": 0.001}, "gap_fixed_K8"),
    ({"gap_always_search": 0.005}, "gap_always_search"),
])
def test_decision_relevance_refusals(override, needle):
    cls, tier, reason = gg.tier_of(RR_DR, 0.015, 0.05, **{**DR_KW, **override})
    assert (cls, tier) == ("meaningful", "meaningful") and needle in reason


def test_decision_relevance_needs_range_and_lower_bound():
    assert gg.tier_of({50: 0.03, 100: 0.03, 200: 0.019}, 0.015, 0.05, **DR_KW)[1] == "meaningful"
    assert gg.tier_of(RR_DR, 0.009, 0.05, **DR_KW)[1] == "meaningful"


def test_gate_rule():
    assert gg.gate({"primary": "decision_relevant", "S1": "flat"})[0]
    assert gg.gate({"primary": "moderate", "S1": "decision_relevant"})[0]
    assert not gg.gate({"primary": "meaningful", "S1": "decision_relevant"})[0]
    assert not gg.gate({"primary": "flat", "S1": "decision_relevant"})[0]
    assert not gg.gate({"primary": "moderate", "S1": "meaningful"})[0]


# ---- Stage B -----------------------------------------------------------------------------------------


def _per(c1, c2, c3, cap):
    return pd.DataFrame({"horizon": [50, 100, 200], "c1": c1, "c2": c2, "c3": c3, "captured": cap})


def test_stage_b_verdicts():
    s, i, r = "supported", "inconclusive", "reversed"
    assert gg.stage_b_verdict(_per([s, s, i], [s, s, s], [True, True, True], [0.9, 0.9, 0.9]))[0] == "supported"
    v, why = gg.stage_b_verdict(_per([i, i, i], [s, s, s], [True, True, False], [0.9, 0.8, 0.6]))
    assert v == "partially_supported" and "C1 fails" in why
    v, why = gg.stage_b_verdict(_per([s, s, s], [s, s, s], [False, False, True], [0.6, 0.6, 0.9]))
    assert v == "partially_supported" and "C3 fails" in why
    assert gg.stage_b_verdict(_per([s, r, s], [s, s, s], [True] * 3, [0.9] * 3))[0] == "contradicted"
    assert gg.stage_b_verdict(_per([s, s, s], [s, s, s], [False] * 3, [0.4, 0.3, 0.9]))[0] == "contradicted"
    assert gg.stage_b_verdict(_per([i] * 3, [i] * 3, [False] * 3, [0.6] * 3))[0] == "inconclusive"


def test_c3_gap_escape_never_counts_toward_contradiction():
    s = "supported"
    # T = 50 / 100: tiny ranges, captured < 0.5 but C3 holds through the gap branch -> not contradicted
    v, _ = gg.stage_b_verdict(_per([s, s, s], [s, s, s], [True, True, True], [0.375, 0.4, 0.9]))
    assert v == "supported"
    assert gg.stage_b_verdict(_per([s, s, s], [s, s, s], [False, False, True], [0.375, 0.4, 0.9]))[0] == "contradicted"


def test_mix_noise_weights_the_replicate_pairs():
    rows = []
    for t, (a, b) in {"task_e1": (1, 1), "task_e2": (0, 0), "task_h1": (1, 0), "task_h2": (0, 1)}.items():
        for r, y in ((0, a), (1, b)):
            rows.append({"pool": "GLK", "status": "ok", "arm_id": "p0", "task_id": t, "replicate": r, "success": y})
    frame = pd.DataFrame(rows)
    tasks = ["task_e1", "task_e2", "task_h1", "task_h2"]
    v_eq, n_eq = gg.mix_noise(frame, tasks, np.full(4, 0.25))
    assert (v_eq, n_eq) == (pytest.approx(0.25), 4)  # the plain pair mean: (0 + 0 + .5 + .5) / 4
    v_hard, n_hard = gg.mix_noise(frame, tasks, np.array([0.0, 0.0, 0.5, 0.5]))
    assert (v_hard, n_hard) == (pytest.approx(0.5), 2)  # only the noisy tasks' pairs


def test_matrix_replay_never_scores_an_unobserved_cell():
    arms = [f"a{i}" for i in range(10)]
    tasks = ["task_e1", "task_e2"]
    cells = {(a, "task_e1"): [1] for a in arms}  # task_e2 never observed: must not be drawn as a failure
    out = gg.matrix_replay(cells, arms, tasks, T=8, M=50, ks=(4,), seed=0).set_index("policy")
    assert out.loc["fixed_K4", "regret"] == pytest.approx(0.0) and out.loc["fixed_K4", "best_rate"] == 1.0


def test_captured_share_and_c3():
    assert gg.captured_share(0.10, 0.02, 0.04) == pytest.approx(0.75)
    assert np.isnan(gg.captured_share(0.05, 0.05, 0.05))
    assert gg.c3_holds(0.75, 0.55, 0.02) and not gg.c3_holds(0.75, 0.45, 0.02)
    assert gg.c3_holds(float("nan"), float("nan"), 0.004)  # the small-gap branch


# ---- mixes, scores, diagnostics ----------------------------------------------------------------------


def test_bank_weighting_and_effective_n():
    tasks = ["task_e1", "task_e2", "task_m1", "task_m2", "task_h1", "task_h2"]
    Y = np.array([[1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=float)
    w = gg.mix_weights(tasks, Y, "S1")
    assert w.sum() == pytest.approx(1.0) and w[4] == pytest.approx(100 / 140 / 2)
    s = gg.weighted_scores(Y, ["a", "b"], w)
    assert s.means[0] == pytest.approx(40 / 140) and s.n[0] == pytest.approx(1 / np.sum(w ** 2))
    assert s.n[0] < len(tasks)  # unequal weights lose effective sample size
    eq = gg.mix_weights(tasks, Y, "primary")
    assert gg.weighted_scores(Y, ["a", "b"], eq).n[0] == pytest.approx(len(tasks))


def test_drop_extremes():
    s = emp.PromptScores(pool="GLK", arm_ids=("a", "b", "c", "d", "e"), successes=np.array([1., 5, 3, 2, 4]),
                         n=np.full(5, 10.0))
    assert gg.drop_extremes(s, 2).arm_ids == ("c", "d", "e")  # top (b) and bottom (a) removed
    assert gg.drop_extremes(s, 4).arm_ids == ("a",)  # the top four removed
    with pytest.raises(ValueError):
        gg.drop_extremes(s, 3)


def _pool_matrix(n_prompts=20, n_tasks=24, outlier=False, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(0.55 + rng.normal(0, 0.06, n_prompts)[:, None] + rng.normal(0, 0.2, n_tasks)[None, :], 0.02, 0.98)
    if outlier:
        p[0] = 0.99
        p[1] = 0.01
    return (rng.random((n_prompts, n_tasks)) < p).astype(float)


def test_leave_one_out_flags_a_planted_outlier():
    clean = gg.leave_one_out(_pool_matrix(), 0.08, 60)
    planted = gg.leave_one_out(_pool_matrix(outlier=True), 0.08, 60)
    assert planted["loo_prompt_max"] > gg.LOO_PROMPT_MAX and planted["loo_prompt_arg"] in (0, 1)
    assert planted["loo_prompt_max"] > clean["loo_prompt_max"]


def test_length_adjustment():
    words = np.arange(20, dtype=float)
    rho, ratio = gg.length_adjusted_ratio(0.4 + 0.01 * words, words, tau_set=float(np.std(0.01 * words)))
    assert rho == pytest.approx(1.0) and ratio == pytest.approx(0.0, abs=1e-6)
    rng = np.random.default_rng(1)
    rates = 0.5 + rng.normal(0, 0.05, 20)
    _, ratio = gg.length_adjusted_ratio(rates, rng.permutation(words), tau_set=0.05)
    assert ratio > 0.8


def test_matrix_replay_finds_the_best_prompt():
    arms = [f"a{i}" for i in range(40)]
    tasks = [f"task_e{j}" for j in range(10)]
    cells = {(a, t): [int(i == 7)] for i, a in enumerate(arms) for t in tasks}  # only a7 ever succeeds
    out = gg.matrix_replay(cells, arms, tasks, T=40, M=300, seed=3).set_index("policy")
    assert set(out.index) == {"fixed_K4", "fixed_K8", "search_all_once"}
    assert out.loc["search_all_once", "regret"] < out.loc["fixed_K4", "regret"]
    assert out.loc["search_all_once", "regret"] == pytest.approx(0.0)


def test_error_count():
    assert gg.error_count(["None", "None", "boom"]) == 1 and gg.error_count(None) == 0


# ---- end to end on a fake tree -----------------------------------------------------------------------

M, N_BOOT = 8, 3


def _calibration(path: Path) -> float:
    rows = [{"pool_id": f"beta_sd{sd:g}", "family": "beta", "spread": sd, "tail_frac": float("nan"),
             "tail_delta": float("nan"), "true_sd": sd, "upper_tail_mass": 0.0, "horizon": T,
             "regret_range": rr * (T / 200), "k_star": 4, "gap_fixed_K8": 0.0, "gap_p3_star": 0.0,
             "gap_always_search": 0.0}
            for sd, rr in ((0.01, 0.001), (0.035, 0.004), (0.05, 0.008), (0.10, 0.02)) for T in (50, 100, 200)]
    pd.DataFrame(rows, columns=list(calibrate.COLUMNS)).to_csv(path, index=False)
    return calibrate.tau_flat_from_calibration(pd.read_csv(path))


def test_gate_end_to_end(tmp_path):
    het, _ = het_fake_tree.build(tmp_path, n_k=10, sd_k=0.2, drop_block_b=("GLG_01",))
    log_file = tmp_path / "logs" / "gitlab" / "worker_0.jsonl"
    rng = np.random.default_rng(0)
    rows = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    for r in rows:
        r["steps"] = int(rng.integers(3, 45))
        r["errors"] = ["None", "boom"] if rng.random() < 0.2 else ["None"]
    log_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    study = dataclasses.replace(st.GLK30, data_dir=het.data_dir, res_dir=het.data_dir / "reservoirs_glk30",
                                log_dirs=(tmp_path / "logs" / "gitlab",))
    pool_yaml = het.data_dir / "pools" / "GLK.yaml"
    doc = pool_yaml.read_text()
    assert "arm_id" in doc

    out, tables = tmp_path / "out", tmp_path / "out" / "tables"
    replay.run_estimate(None, study.res_dir, out, study=study)
    shas = replay.verify_reservoirs(study.res_dir, study=study)
    cells = [replay.make_emp_cell("GLK", v, T, replay.load_reservoir(study.res_dir / f"GLK_{v}.json"), M, study=study)
             for v in replay.VARIANTS for T in replay.PRIMARY_HORIZONS]
    kgrid = replay.stamp_kgrid(replay.kgrid(cells, workers=1, k_grid=(2, 4, 8), study=study), shas)
    tables.mkdir(parents=True)
    kgrid.to_csv(tables / study.table("kgrid"), index=False)
    describe.flatness(kgrid).to_csv(tables / study.table("flatness"), index=False)
    replay.run_boot_kgrid(study.res_dir, workers=1, study=study, n_boot=N_BOOT, n_replicates=M,
                          k_grid=(2, 4, 8)).to_csv(tables / study.table("boot_kgrid"), index=False)
    common = ["--workers", "1", "--out-dir", str(out), "--baseline-params", str(DEPLOY / "baseline_params.json"),
              "--thresholds", str(DEPLOY / "thresholds.json"), "--skip-summary"]
    npmle_cells = [c for c in cells if c.env_id.endswith("_npmle")]
    rd.main(["--test", study.test, *common], cells=npmle_cells)
    rc.write_reservoir_stamp(out, study.test, study.res_dir / replay.MANIFEST_FILE)
    noise = json.loads((study.res_dir / "noise.json").read_text())
    boot_cells = [replay.make_emp_cell(pool, "npmle", T, res, M, boot=b, study=study)
                  for b, pool, res in replay.bootstrap_reservoirs(replay.load_snapshot(study.res_dir), noise,
                                                                  n_boot=N_BOOT, study=study)
                  for T in replay.PRIMARY_HORIZONS]
    rd.main(["--test", study.boot_test, *common], cells=boot_cells)
    rc.write_reservoir_stamp(out, study.boot_test, study.res_dir / replay.MANIFEST_FILE)
    describe.rule_gaps(out, describe.k_star_table(kgrid), study=study).to_csv(tables / study.table("rule_gaps"),
                                                                             index=False)

    cal = tmp_path / "calibration.csv"
    tau_flat = _calibration(cal)
    results = tmp_path / "results"
    with pytest.raises(ValueError, match="tau-flat"):
        gg.run_gate(study=study, out_dir=out, results=results, tau_flat=tau_flat + 0.01, calibration_csv=cal)
    written = gg.run_gate(study=study, out_dir=out, results=results, tau_flat=tau_flat, calibration_csv=cal,
                          n_boot=N_BOOT, point_m=M, boot_m=M, workers=1, matrix_m=50, registered_n_boot=N_BOOT)
    assert set(written) == {"gate_classification", "classification", "gate", "prompt_table", "diagnostics",
                            "stage_b", "matrix_replay_T40"}
    cls = pd.read_csv(written["gate_classification"]).set_index("mix")
    assert list(cls.index) == ["primary", "S1", "S2"]
    assert set(cls["tier"]) <= {"flat", "moderate", "meaningful", "decision_relevant"}
    assert cls.loc["S2", "tier"] in {"flat", "moderate"}
    assert {"drop2_tier", "drop4_tier"} <= set(cls.columns)
    assert cls.loc["S1", "r_sb"] == pytest.approx(cls.loc["primary", "r_sb"], nan_ok=True)
    assert (cls["task_universe"] == "block_a").all() and (cls["score_cap"] == 30).all() and cls["unregistered"].all()
    gate = pd.read_csv(written["gate"]).iloc[0]
    assert isinstance(bool(gate["gate_pass"]), bool)
    sb = pd.read_csv(written["stage_b"])
    assert set(sb["contrast"]) == {"C1", "C2", "C3", "gap_reported", "verdict"}
    assert set(sb.loc[sb.contrast == "gap_reported", "policy"]) == {"fixed_K4", "fixed_K8", "p3_star", "always_search"}
    assert sb[sb.contrast == "verdict"]["verdict"].iloc[0] in {"supported", "partially_supported", "contradicted",
                                                               "inconclusive"}
    assert sb["read"].nunique() == 1 and bool(sb["read"].iloc[0]) == bool(gate["gate_pass"])
    ptab = pd.read_csv(written["prompt_table"])
    assert len(ptab) == 10 and (ptab["rate"] <= ptab["rate_as_collected"] + 1e-12).all()
    assert (ptab["mean_errors"] > 0).any()
    diag = pd.read_csv(written["diagnostics"])
    assert {"1_extremes", "2_loo_prompt", "3_loo_task", "4_length", "5_steps", "6_leak", "7_noise"} <= set(diag["diagnostic"])
