"""Pre-registration 11's study variants: `study.GLK30` (the GLK cell re-scored at the 30-step budget) and
`study.GLK` (the same cell as collected), on a tiny fake ``logs/heterogeneity/gitlab`` tree whose records
carry ``steps``. Checks the re-scoring rule, the block-A fallback on a one-pool study, and that estimate ->
K-grid -> point episodes -> rule gaps run end to end under each variant's own test ids and table prefix.

Only sizes are shrunk (M, the K-grid); no production constant is modified; the deployment's tuned constants
are read, never written.
"""

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

import describe  # noqa: E402
import het_fake_tree  # noqa: E402
import policy_table as pt  # noqa: E402
import registered_contrast as rc  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402
import study as st  # noqa: E402

M = 8
K_GRID = (1, 2, 4)
DEPLOY = ROOT / "results" / "growing_bandits" / "deploy"


def _add_steps(log_file: Path, seed: int = 0) -> None:
    """Give every fake record a ``steps`` count: successes spread over 3..60 steps so the 30-step budget bites."""
    rng = np.random.default_rng(seed)
    rows = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
    for r in rows:
        r["steps"] = int(rng.integers(3, 61))
    log_file.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _glk_tree(tmp_path: Path, base: st.Study) -> st.Study:
    het, _ = het_fake_tree.build(tmp_path, drop_block_b=("GLG_01",))
    _add_steps(tmp_path / "logs" / "gitlab" / "worker_0.jsonl")
    return dataclasses.replace(base, data_dir=het.data_dir, res_dir=het.data_dir / f"reservoirs_{base.name}",
                               log_dirs=(tmp_path / "logs" / "gitlab",))


def test_rescore_rule():
    frame = pd.DataFrame([
        {"status": "ok", "success": 1, "steps": 12, "timed_out": False},   # stays a success
        {"status": "ok", "success": 1, "steps": 30, "timed_out": False},   # boundary: <= 30 counts
        {"status": "ok", "success": 1, "steps": 31, "timed_out": False},   # succeeded after the budget
        {"status": "ok", "success": 1, "steps": 20, "timed_out": True},    # clock-ended
        {"status": "ok", "success": 0, "steps": 5, "timed_out": False},    # failures stay failures
        {"status": "missing", "success": None, "steps": None, "timed_out": None},  # untouched
    ])
    out = st.rescore(frame, 30)
    assert out["success"].tolist()[:5] == [1, 1, 0, 0, 0]
    assert out.loc[5, "success"] is None or pd.isna(out.loc[5, "success"])
    with pytest.raises(ValueError, match="steps"):
        st.rescore(frame.drop(columns="steps"), 30)


def test_registered_variants():
    assert st.GLK30.pools == ("GLK",) and st.GLK30.score_cap == 30 and st.GLK.score_cap is None
    assert {st.GLK30.test, st.GLK30.boot_test, st.GLK.test, st.GLK.boot_test} <= set(rd.TESTS)
    assert "fixed_K4" in pt.TEST_POLICIES["glk30"] and "fixed_K4" in pt.TEST_POLICIES["glk_boot"]
    seeds = {st.PREREG9.seed_base, st.HETEROGENEITY.seed_base, st.GLK30.seed_base, st.GLK.seed_base}
    assert len(seeds) == 4


@pytest.mark.parametrize("base", [st.GLK30, st.GLK], ids=["glk30", "glk"])
def test_glk_variant_end_to_end(tmp_path, base):
    study = _glk_tree(tmp_path, base)
    raw = st.load_study_outcomes(dataclasses.replace(study, score_cap=None))
    scored = st.load_study_outcomes(study)
    ok_raw, ok_scored = raw[raw.status == "ok"], scored[scored.status == "ok"]
    late = (ok_raw["success"].astype(int) == 1) & (ok_raw["steps"] > 30) & ~ok_raw["timed_out"].astype(bool)
    assert late.any()  # the fixture exercises the budget
    if base.score_cap is None:
        assert (ok_scored["success"].astype(int) == ok_raw["success"].astype(int)).all()
    else:
        assert (ok_scored.loc[late[late].index, "success"].astype(int) == 0).all()
        assert ok_scored["success"].astype(int).sum() < ok_raw["success"].astype(int).sum()

    out = tmp_path / "out"
    tables = out / "tables"
    replay.run_estimate(None, study.res_dir, out, study=study)
    assert replay.load_manifest(study.res_dir)["task_universe"] == "block_a"
    snap = replay.load_snapshot(study.res_dir)
    assert set(snap["pool"]) >= {"GLK"} and set(snap.loc[snap.pool == "GLK", "task_id"]) <= set(het_fake_tree.GL_BLOCK_A)
    assert sorted(p.name for p in study.res_dir.glob("GLK_*.json")) == [
        "GLK_npmle.json", "GLK_parametric.json", "GLK_raw.json"]

    shas = replay.verify_reservoirs(study.res_dir, study=study)
    cells = [replay.make_emp_cell("GLK", "npmle", T, replay.load_reservoir(study.res_dir / "GLK_npmle.json"), M,
                                  study=study) for T in replay.PRIMARY_HORIZONS]
    assert all(c.env_id == f"{study.test}_GLK_npmle" for c in cells)
    kgrid = replay.stamp_kgrid(replay.kgrid(cells, workers=1, k_grid=K_GRID, study=study), shas)
    tables.mkdir(parents=True)
    kgrid.to_csv(tables / study.table("kgrid"), index=False)
    assert (tables / f"{base.table_prefix}_kgrid.csv").exists()

    common = ["--workers", "1", "--out-dir", str(out), "--baseline-params", str(DEPLOY / "baseline_params.json"),
              "--thresholds", str(DEPLOY / "thresholds.json"), "--skip-summary"]
    rd.main(["--test", study.test, *common], cells=cells)
    rc.write_reservoir_stamp(out, study.test, study.res_dir / replay.MANIFEST_FILE)
    episodes = out / "episodes" / study.test
    assert episodes.is_dir() and not (out / "episodes" / "het").exists()

    gaps = describe.rule_gaps(out, describe.k_star_table(kgrid), study=study)
    assert set(pt.TEST_POLICIES[study.test]) == set(gaps["policy"])
    assert gaps["gap_to_k_star"].notna().all()
