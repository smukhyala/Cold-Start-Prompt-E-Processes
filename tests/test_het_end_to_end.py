"""Synthetic end-to-end dry run of Pre-registration 10's analysis (final review minor ruling): a tiny fake
``logs/heterogeneity`` tree (few arms and tasks, with replicate pairs; one GitLab arm missing block B, so
the block-A fallback is exercised) -> ``replay estimate`` -> K-grid at small M on a tiny K-grid -> boot
K-grid -> point / bootstrap policy episodes -> ``het_verdicts`` -> ``make_figures``.

Only sizes are shrunk, through the functions' own parameters (M, the K-grid, B, the tail bootstrap); no
production constant is modified. Everything lives under tmp_path: the deployment's tuned constants and
models are read, never written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("experiments/growing_bandits/empirical", "experiments/growing_bandits/deploy", "tests"):
    sys.path.insert(0, str(ROOT / sub))

import calibrate  # noqa: E402
import describe  # noqa: E402
import het_fake_tree  # noqa: E402
import het_verdicts as hv  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

import figures as fg  # noqa: E402

M = 8  # episodes per replay cell (production: 1000 point, 250 boot)
K_GRID = (1, 2, 4)  # production: kse.DEFAULT_K_GRID
N_BOOT = 2  # prompt-bootstrap reservoirs (production: 200)
DEPLOY = ROOT / "results" / "growing_bandits" / "deploy"


def _calibration(path: Path) -> float:
    rows = [{"pool_id": f"beta_sd{sd:g}", "family": "beta", "spread": sd, "tail_frac": float("nan"),
             "tail_delta": float("nan"), "true_sd": sd, "upper_tail_mass": 0.0, "horizon": T,
             "regret_range": rr * (T / 200), "k_star": 4, "gap_fixed_K8": 0.0, "gap_p3_star": 0.0,
             "gap_always_search": 0.0}
            for sd, rr in ((0.01, 0.001), (0.035, 0.004), (0.05, 0.008), (0.10, 0.02)) for T in (50, 100, 200)]
    pd.DataFrame(rows, columns=list(calibrate.COLUMNS)).to_csv(path, index=False)
    return calibrate.tau_flat_from_calibration(pd.read_csv(path))


def test_synthetic_dry_run_from_logs_to_figures(tmp_path, monkeypatch):
    study, pool_g = het_fake_tree.build(tmp_path, drop_block_b=("GLK_02",))
    monkeypatch.setitem(hv.EXTRA_POOL_FILES, "GMG", pool_g)
    out = tmp_path / "out"
    tables = out / "tables"

    # estimate: STATUS checked, prompt shas checked, the fallback decided and frozen
    replay.run_estimate(None, study.res_dir, out, study=study)
    assert replay.load_manifest(study.res_dir)["task_universe"] == "block_a"

    # K-grid (point reservoirs) + flatness, then the bootstrap K-grid
    shas = replay.verify_reservoirs(study.res_dir, study=study)
    cells = [replay.make_emp_cell(p, "npmle", T, replay.load_reservoir(study.res_dir / f"{p}_npmle.json"), M,
                                  study=study) for p in study.pools for T in replay.PRIMARY_HORIZONS]
    kgrid = replay.stamp_kgrid(replay.kgrid(cells, workers=1, k_grid=K_GRID, study=study), shas)
    tables.mkdir(parents=True)
    kgrid.to_csv(tables / study.table("kgrid"), index=False)
    describe.flatness(kgrid).to_csv(tables / study.table("flatness"), index=False)
    replay.run_boot_kgrid(study.res_dir, workers=1, study=study, n_boot=N_BOOT, n_replicates=M,
                          k_grid=K_GRID).to_csv(tables / study.table("boot_kgrid"), index=False)

    # the point / bootstrap policy episodes the H3 contrasts read (tuned constants read-only)
    common = ["--workers", "1", "--out-dir", str(out), "--baseline-params", str(DEPLOY / "baseline_params.json"),
              "--thresholds", str(DEPLOY / "thresholds.json"), "--skip-summary"]
    rd.main(["--test", study.test, *common], cells=cells)
    rc.write_reservoir_stamp(out, study.test, study.res_dir / replay.MANIFEST_FILE)
    noise = json.loads((study.res_dir / "noise.json").read_text())
    boot_cells = [replay.make_emp_cell(pool, "npmle", T, res, M, boot=b, study=study)
                  for b, pool, res in replay.bootstrap_reservoirs(replay.load_snapshot(study.res_dir), noise,
                                                                  n_boot=N_BOOT, study=study)
                  for T in replay.PRIMARY_HORIZONS]
    rd.main(["--test", study.boot_test, *common], cells=boot_cells)
    rc.write_reservoir_stamp(out, study.boot_test, study.res_dir / replay.MANIFEST_FILE)

    # verdicts
    cal = tmp_path / "calibration.csv"
    tau_flat = _calibration(cal)
    results = tmp_path / "results"
    with pytest.raises(ValueError, match="tau-flat"):
        hv.run_verdicts(study=study, out_dir=out, results=results, tau_flat=tau_flat + 0.001, calibration_csv=cal)
    written = hv.run_verdicts(study=study, out_dir=out, results=results, tau_flat=tau_flat, calibration_csv=cal,
                              n_boot=40, prereg9_log_dir=tmp_path / "no_prereg9_logs", expected_boot_n=N_BOOT,
                              tail_n_boot=4, registration_n_boot=N_BOOT)
    expected = {"het_components.csv", "het_classification.csv", "het_policy_gaps.csv", "het_anchor_recovery.csv",
                "het_hypotheses.csv", "het_portability.csv", "het_timeout_sensitivity.csv", "het_timeout_rates.csv"}
    assert set(written) == expected
    for name in expected:
        frame = pd.read_csv(results / name)
        assert set(hv.PROVENANCE_COLUMNS) <= set(frame.columns), name
        assert set(frame["task_universe"]) == {"block_a"}, name
        assert set(frame["unregistered"]) == {True}, name  # n_boot 40 is not the registered 2000
        assert set(frame["calibration_sha256"]) == {rc.file_sha256(cal)}, name
    comp = pd.read_csv(results / "het_components.csv").set_index("pool")
    assert list(comp.index) == list(study.pools)
    assert (comp.loc[["GLG", "GLK"], "n_tasks"] == len(het_fake_tree.GL_BLOCK_A)).all()  # the fallback applied
    assert comp["upper_tail_noise_var"].notna().all()
    cls = pd.read_csv(results / "het_classification.csv")
    assert set(cls["class"]) <= {"flat", "moderate", "meaningful"} and "tau_bound_below_flat" in cls.columns
    anchors = pd.read_csv(results / "het_anchor_recovery.csv").iloc[0]
    assert anchors["n_tasks"] == len(het_fake_tree.GL_BLOCK_A) and anchors["prereg9_anchor_baseline_n"] == 6
    hyp = pd.read_csv(results / "het_hypotheses.csv")
    assert set(hyp["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    prov = json.loads((results / "het_provenance.json").read_text())
    assert prov["tau_flat"] == pytest.approx(tau_flat) and prov["task_universe"] == "block_a"

    # figures from the tables alone
    paths = fg.make_figures(results, tmp_path / "figures", calibration_csv=cal, stage0_dir=tmp_path / "none",
                            kgrid_csv=tables / study.table("kgrid"), tau_flat=tau_flat,
                            prereg9_flatness_csv=tmp_path / "none" / "emp_flatness.csv")
    assert len(paths) == 14 and all(p.exists() and p.stat().st_size > 0 for p in paths)
