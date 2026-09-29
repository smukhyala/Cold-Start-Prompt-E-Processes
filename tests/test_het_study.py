"""The Study refactor: Pre-reg 9 unchanged by default; a second study with its own pools, ids and seeds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("empirical", "deploy"):
    sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / sub))

import replay  # noqa: E402
import run_deployment as rd  # noqa: E402
import study as st  # noqa: E402

from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402

RES = EmpiricalReservoir([0.3, 0.6, 0.9], [0.5, 0.3, 0.2])


def test_prereg9_study_is_the_default_and_unchanged():
    s = st.PREREG9
    assert (s.test, s.boot_test, s.pools, s.seed_base, s.table_prefix) == ("emp", "emp_boot", ("G", "F"), 400_000_000, "emp")
    assert replay.env_id("G", "npmle") == "emp_G_npmle"
    assert replay.seed_for("F", 200) == 400_000_000 + 1_000 * (1 * replay.POOL_STRIDE + 200)


def test_het_study_ids_and_seeds_are_disjoint_from_prereg9():
    h = st.HETEROGENEITY
    assert replay.env_id("GLK", "npmle", study=h) == "het_GLK_npmle"
    het_seeds = {replay.seed_for(p, T, b, study=h) for p in h.pools for T in (50, 100, 200, 500, 1000)
                 for b in (None, 0, 199)}
    emp_seeds = {replay.seed_for(p, T, b) for p in ("G", "F") for T in (50, 100, 200, 500, 1000)
                 for b in (None, 0, 199)}
    assert not het_seeds & emp_seeds
    import cells
    cells.assert_seed_disjointness(sorted(het_seeds))


def _pairs(pool, v, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        x0 = int(rng.random() < 0.5)
        x1 = x0 if rng.random() > 2 * v else 1 - x0
        for rep, x in ((0, x0), (1, x1)):
            rows.append({"pool": pool, "arm_id": f"{pool}_{k % 10}", "task_id": f"t{k}", "replicate": rep,
                         "status": "ok", "success": x})
    return rows


def test_cell_without_replicates_borrows_declared_noise():
    rows = _pairs("GMG", 0.05) + [{"pool": "GMB", "arm_id": "GMB_0", "task_id": "t0", "replicate": 0,
                                   "status": "ok", "success": 1}]
    for pool in ("GMK", "GLG", "GLK"):
        rows += _pairs(pool, 0.08, seed=1)
    noise = replay.noise_model(pd.DataFrame(rows), study=st.HETEROGENEITY)
    assert noise["v"]["GMB"] == noise["v"]["GMG"]
    assert noise["borrowed"] == {"GMB": "GMG"}


def test_undeclared_pool_without_pairs_raises():
    rows = _pairs("GMG", 0.05) + [{"pool": "GLK", "arm_id": "a", "task_id": "t", "replicate": 0,
                                   "status": "ok", "success": 1}]
    rows += _pairs("GMK", 0.05) + _pairs("GLG", 0.05)
    with pytest.raises(ValueError, match="GLK"):
        replay.noise_model(pd.DataFrame(rows), study=st.HETEROGENEITY)


def test_runner_and_policy_table_know_the_new_tests():
    import policy_table as pt
    assert {"het", "het_boot", "cal"} <= set(rd.TESTS)
    assert pt.POLICIES["fixed_K8"]["params"] == {"K": 8}
    assert "fixed_K8" in pt.TEST_POLICIES["het"] and "fixed_K8" in pt.TEST_POLICIES["cal"]


def test_load_study_outcomes_relabels_extras(tmp_path):
    snap = tmp_path / "snap.jsonl"
    pd.DataFrame([{"pool": "G", "arm_id": "G_00", "task_id": "t1", "replicate": 0, "status": "ok", "success": 1},
                  {"pool": "F", "arm_id": "F_00", "task_id": "t1", "replicate": 0, "status": "ok", "success": 0}]
                 ).to_json(snap, orient="records", lines=True)
    s = st.Study(name="x", test="x", boot_test="x_boot", pools=("GMG",), data_dir=tmp_path, res_dir=tmp_path,
                 log_dirs=(tmp_path / "none",), seed_base=900_000, table_prefix="x", noise_mode="per_pool",
                 borrowed_noise={}, extra_outcomes=((snap, "G", "GMG"),))
    with pytest.raises(FileNotFoundError, match="manifest.json"):  # an extra snapshot is sha-checked
        st.load_study_outcomes(s)
    _freeze(snap)
    out = st.load_study_outcomes(s)
    assert list(out["pool"]) == ["GMG"] and list(out["arm_id"]) == ["GMG_G_00"]
    snap.write_text(snap.read_text() + "\n")  # any byte change after the freeze
    with pytest.raises(ValueError, match="sha256"):
        st.load_study_outcomes(s)


def _freeze(snap: Path) -> None:
    """The source study's reservoir manifest beside `snap`, as `replay.write_manifest` writes it."""
    import hashlib
    import json
    (snap.parent / "manifest.json").write_text(json.dumps({"outcomes_snapshot": {
        "file": snap.name, "sha256": hashlib.sha256(snap.read_bytes()).hexdigest()}}))


def test_the_real_prereg9_extra_snapshot_matches_its_manifest():
    for snapshot, _, _ in st.HETEROGENEITY.extra_outcomes:
        st.check_extra_snapshot(snapshot)


# ---- beyond the brief: the study reaches every id, path and pattern -------------------------


def test_het_study_config_is_as_planned():
    h = st.HETEROGENEITY
    assert (h.name, h.test, h.boot_test, h.table_prefix, h.noise_mode) == ("het", "het", "het_boot", "het", "per_pool")
    assert h.pools == ("GMG", "GMK", "GMB", "GLG", "GLK")
    assert h.seed_base == 10_000_000_000
    assert h.data_dir == ROOT / "data" / "heterogeneity" and h.res_dir == h.data_dir / "reservoirs"
    assert h.log_dirs == tuple(ROOT / "logs" / "heterogeneity" / p for p in ("gitlab", "gmail", "bridge"))
    assert h.extra_outcomes == ((ROOT / "data" / "empirical_pool" / "reservoirs" / "outcomes_snapshot.jsonl",
                                 "G", "GMG"),)
    assert dict(h.borrowed_noise) == {"GMB": "GMG"}
    assert st.STUDIES == {"prereg9": st.PREREG9, "het": st.HETEROGENEITY}


def test_prereg9_study_paths_are_the_module_constants():
    import registered_contrast as rc
    p = st.PREREG9
    assert (p.data_dir, p.res_dir, p.log_dirs) == (replay.DATA_DIR, replay.RES_DIR, (replay.LOG_DIR,))
    assert p.res_dir / replay.MANIFEST_FILE == rc.EMP_RESERVOIR_MANIFEST
    assert p.extra_outcomes == () and p.noise_mode == "pairwise_decision"


def test_het_seed_uses_its_own_boot_stride():
    h = st.HETEROGENEITY
    stride = replay.POOL_STRIDE * len(h.pools)
    assert replay.seed_for("GLK", 200, 3, study=h) == h.seed_base + 1_000 * (4 * stride + 4 * replay.POOL_STRIDE + 200)
    assert replay.seed_for("F", 200, 3) == 400_000_000 + 1_000 * (4 * replay.BOOT_STRIDE + replay.POOL_STRIDE + 200)
    cell = replay.make_emp_cell("GMB", "npmle", 100, RES, 7, boot=0, study=h)
    assert cell.env_id == "het_GMB_npmle_b000" and cell.base_seed == replay.seed_for("GMB", 100, 0, study=h)


def test_per_pool_noise_keeps_each_pools_own_v():
    rows = _pairs("GMG", 0.02) + _pairs("GMK", 0.10, seed=1) + _pairs("GLG", 0.05, seed=2) + _pairs("GLK", 0.05, seed=3)
    rows += _pairs("GMB", 0.2, seed=4)  # GMB has pairs here: its own v, no borrow
    frame = pd.DataFrame(rows)
    noise = replay.noise_model(frame, study=st.HETEROGENEITY)
    from cold_start.growing import empirical as emp
    for p in st.HETEROGENEITY.pools:
        assert noise["v"][p] == emp.within_cell_variance(frame, [p])[0]
    assert noise["borrowed"] == {} and noise["per_pool"] is True


def test_pairwise_decision_rejects_a_study_that_is_not_two_pools():
    import dataclasses
    three = dataclasses.replace(st.PREREG9, pools=("G", "F", "X"))
    with pytest.raises(ValueError, match="two pools"):
        replay.noise_model(pd.DataFrame(_pairs("G", 0.05)), study=three)


def test_purge_is_scoped_to_the_study(tmp_path):
    comp = tmp_path / "comparators"
    comp.mkdir()
    names = ["emp_G_npmle_seed1_M10.npy", "emp_G_npmle_b000_seed2_M10.npy",
             "het_GLK_npmle_seed3_M10.npy", "het_GLK_npmle_b001_seed4_M10.json", "phi_other_seed5.npy"]
    for n in names:
        (comp / n).write_text("x")
    assert replay.purge_stale_emp_comparators(tmp_path, boot_only=True, study=st.HETEROGENEITY) == 1
    assert replay.purge_stale_emp_comparators(tmp_path, study=st.HETEROGENEITY) == 1
    assert sorted(p.name for p in comp.iterdir()) == ["emp_G_npmle_b000_seed2_M10.npy", "emp_G_npmle_seed1_M10.npy",
                                                      "phi_other_seed5.npy"]
    assert replay.purge_stale_emp_comparators(tmp_path) == 2


def test_prompt_bootstrap_contrast_reads_the_studys_tests(tmp_path):
    import registered_contrast as rc
    col = "regret_posterior_mean_shrunk"
    h = st.HETEROGENEITY
    flat = []
    for pool in ("GLG", "GLK"):
        for T in (50, 100, 200):
            env = f"het_{pool}_npmle"
            flat.append({"env_id": env, "pool": pool, "variant": "npmle", "horizon": T, "regret_range": 0.05,
                         "informative": True})
            for test, e, seed, shift in [("het", env, 1, 0.0)] + [("het_boot", f"{env}_b{b:03d}", 100 + b, 0.0005 * b)
                                                                   for b in range(4)]:
                for policy, d in (("p3_star", shift), ("fixed_K_star", 0.0)):
                    path = tmp_path / "episodes" / test / f"{e}_T{T}_cap{T}" / f"{policy}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"env_id": e, "horizon": T, "cap": T, "base_seed": seed, "episode": np.arange(10),
                                  col: np.full(10, 0.1) + d}).to_parquet(path, index=False)
    (tmp_path / "tables").mkdir()
    pd.DataFrame(flat).to_csv(tmp_path / "tables" / "het_flatness.csv", index=False)
    row = rc.prompt_bootstrap_contrast("p3_star", "fixed_K_star", horizons=(50, 100, 200), mei=0.002,
                                       rule="noninferiority", out_dir=tmp_path, expected_n_boot=4, study=h).iloc[0]
    assert row["test"] == "het" and row["n_informative"] == 6 and row["n_boot"] == 4
    assert row["hi"] == pytest.approx(np.percentile([0.0005 * b for b in range(4)], 97.5))
    assert not row["as_registered"]  # het registrations name a section 6.5 class, never the flatness guard
    with pytest.raises(FileNotFoundError):  # Pre-reg 9 (the default) finds no emp tree here
        rc.prompt_bootstrap_contrast("p3_star", "fixed_K_star", horizons=(50, 100, 200), mei=0.002,
                                     rule="noninferiority", out_dir=tmp_path, expected_n_boot=4)


def test_check_reservoir_snapshot_uses_the_studys_prefix_and_stamps(tmp_path):
    import json

    import registered_contrast as rc
    h = st.HETEROGENEITY
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"reservoirs": {"GLK_npmle": "aa"}}))
    flat = pd.DataFrame([{"env_id": "het_GLK_npmle", "variant": "npmle", "horizon": 200, "reservoir_sha256": "aa"}])
    with pytest.raises(ValueError, match="het_reservoir_stamp"):
        rc.check_reservoir_snapshot(tmp_path, flat, manifest, (200,), study=h)
    rc.write_reservoir_stamp(tmp_path, "het", manifest)
    rc.write_reservoir_stamp(tmp_path, "het_boot", manifest)
    rc.check_reservoir_snapshot(tmp_path, flat, manifest, (200,), study=h)
    with pytest.raises(ValueError, match="het_GLK_npmle"):
        rc.check_reservoir_snapshot(tmp_path, flat.assign(reservoir_sha256="bb"), manifest, (200,), study=h)


def test_runner_het_tests_are_replay_only():
    import policy_table as pt
    assert rd.DEFAULT_REPLICATES["het"] == 1000 and rd.DEFAULT_REPLICATES["het_boot"] == 250
    assert rd.DEFAULT_REPLICATES["cal"] == 1000
    assert rd.seeds_may_repeat("het") and rd.seeds_may_repeat("cal") and not rd.seeds_may_repeat("het_boot")
    for test in ("het", "het_boot", "cal"):
        with pytest.raises(RuntimeError, match=test):
            rd._cell_grid(test, None)
    assert pt.TEST_POLICIES["het"] == pt.TEST_POLICIES["het_boot"] == (
        "always_search", "p3_star", "fixed_K_star", "level_star", "phi_k4", "fixed_K8")
    assert pt.TEST_POLICIES["cal"] == ("always_search", "p3_star", "fixed_K8")
    assert pt.POLICIES["fixed_K8"] == {"kind": "fixed_K", "params": {"K": 8}, "group": "baseline", "requires": []}


def test_clis_take_a_study_flag(tmp_path, monkeypatch):
    import describe
    import registered_contrast as rc
    for mod, argv in ((replay, ["kgrid"]), (describe, []), (rc, ["--registration", "emp_primary"])):
        with pytest.raises(SystemExit):
            mod.main(argv + ["--study", "nope", "--out-dir", str(tmp_path)])


def test_estimate_and_boot_run_end_to_end_on_a_multi_dir_study(tmp_path):
    import json

    rng = np.random.default_rng(5)

    def pool_rows(pool, n_arms=8, n_tasks=20, n_reps=40):
        rows = []
        mus = rng.uniform(0.3, 0.8, n_arms)
        for i, mu in enumerate(mus):
            for t in range(n_tasks):
                rows.append({"pool": pool, "arm_id": f"{pool}_{i:02d}", "task_id": f"t{t}", "replicate": 0,
                             "attempt": 1, "status": "ok", "success": int(rng.random() < mu)})
        for _ in range(n_reps):
            i, t = int(rng.integers(n_arms)), int(rng.integers(n_tasks))
            rows.append({"pool": pool, "arm_id": f"{pool}_{i:02d}", "task_id": f"t{t}", "replicate": 1,
                         "attempt": 1, "status": "ok", "success": int(rng.random() < mus[i])})
        return pd.DataFrame(rows).drop_duplicates(["arm_id", "task_id", "replicate"])

    def write(path, frame, schema=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for r in frame.to_dict("records"):
                fh.write(json.dumps({"schema": "empirical_pool/1", "cost_usd": 0.01, **r} if schema else r) + "\n")

    write(tmp_path / "logs" / "one" / "worker_0.jsonl", pool_rows("B"))
    no_pairs = pool_rows("C", n_reps=0)
    write(tmp_path / "logs" / "two" / "worker_0.jsonl", no_pairs[no_pairs["replicate"] == 0])
    write(tmp_path / "snap.jsonl", pd.concat([pool_rows("G"), pool_rows("F")]), schema=False)
    _freeze(tmp_path / "snap.jsonl")
    s = st.Study(name="x", test="xt", boot_test="xt_boot", pools=("A", "B", "C"), data_dir=tmp_path,
                 res_dir=tmp_path / "res", log_dirs=(tmp_path / "logs" / "one", tmp_path / "logs" / "two",
                                                     tmp_path / "logs" / "absent"),
                 seed_base=900_000_000_000, table_prefix="xt", noise_mode="per_pool", borrowed_noise={"C": "A"},
                 extra_outcomes=((tmp_path / "snap.jsonl", "G", "A"),))
    replay.run_estimate(None, s.res_dir, tmp_path / "out", study=s)
    noise = json.loads((s.res_dir / "noise.json").read_text())
    assert noise["borrowed"] == {"C": "A"} and noise["v"]["C"] == noise["v"]["A"]
    snap = replay.load_snapshot(s.res_dir)
    assert set(snap["pool"]) == {"A", "B", "C"} and all(a.startswith("A_G_") for a in snap[snap["pool"] == "A"]["arm_id"])
    shas = replay.verify_reservoirs(s.res_dir, study=s)
    assert set(shas) == {f"xt_{p}_{v}" for p in s.pools for v in replay.VARIANTS}
    boots = replay.bootstrap_reservoirs(snap, noise, n_boot=2, study=s)
    assert [(b, p) for b, p, _ in boots] == [(b, p) for b in range(2) for p in s.pools]
    cells = [replay.make_emp_cell(p, "npmle", 50, r, 5, boot=b, study=s) for b, p, r in boots]
    assert len({c.base_seed for c in cells}) == len(cells)
    import registered_contrast as rc
    assert all(rc.boot_env_pattern(s).match(c.env_id) for c in cells)


# ---- final fix wave: strict logs (I5, minors) and the block-A fallback (I1) ------------------------


def _tree(tmp_path, **kw):
    sys.path.insert(0, str(ROOT / "tests"))
    import het_fake_tree
    return het_fake_tree.build(tmp_path, **kw)


def test_strict_study_loads_a_clean_tree(tmp_path):
    s, _ = _tree(tmp_path)
    out = st.load_study_outcomes(s)
    assert {"GLG", "GLK", "GMK", "GMB", "GMG", "anchor"} == set(out["pool"])


def test_strict_study_refuses_a_missing_log_dir(tmp_path):
    import shutil
    s, _ = _tree(tmp_path)
    shutil.rmtree(s.log_dirs[2])
    with pytest.raises(FileNotFoundError, match="bridge"):
        st.load_study_outcomes(s)
    import dataclasses
    lax = dataclasses.replace(s, strict_logs=False)  # the Pre-reg 9 behaviour: skipped
    assert "GMB" not in set(st.load_study_outcomes(lax)["pool"])


def test_strict_study_refuses_a_record_whose_prompt_sha_is_not_the_pool_files(tmp_path):
    import json
    s, _ = _tree(tmp_path)
    path = s.log_dirs[1] / "worker_0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[3]["prompt_sha256"] = "f" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="prompt_sha256"):
        st.load_study_outcomes(s)


def test_estimate_refuses_unfinished_logs_unless_overridden(tmp_path):
    s, _ = _tree(tmp_path)
    (s.log_dirs[0] / "STATUS").write_text("running\n")
    with pytest.raises(RuntimeError, match="STATUS"):
        replay.run_estimate(None, s.res_dir, tmp_path / "out", study=s)
    for status in st.ESTIMABLE_STATUSES:
        (s.log_dirs[0] / "STATUS").write_text(status + "\n")
        st.check_log_status(s)
    (s.log_dirs[0] / "STATUS").write_text("provider_down\n")
    replay.run_estimate(None, s.res_dir, tmp_path / "out", study=s, allow_unfinished=True)
    args = replay.build_parser().parse_args(["estimate", "--study", "het", "--allow-unfinished-logs"])
    assert args.allow_unfinished_logs and not replay.build_parser().parse_args(["estimate"]).allow_unfinished_logs


def test_task_universe_is_block_a_when_any_gitlab_pool_misses_block_b(tmp_path):
    s, _ = _tree(tmp_path)
    full = st.load_study_outcomes(s)
    assert st.task_universe(full, s) == "subset_60"
    for arm in ("GLG_03", "GLK_00", "GL_anchor_oracle"):  # any GitLab pool, anchors included
        sub = tmp_path / arm
        s2, _ = _tree(sub, drop_block_b=(arm,))
        out = st.load_study_outcomes(s2)
        assert st.task_universe(out, s2) == "block_a"
        kept = st.restrict_to_universe(out, s2, "block_a")
        gl = kept[kept["arm_id"].isin(st.fallback_arms(s2))]
        assert set(gl["task_id"]) == set(st.load_data_manifest(s2)["gitlab_block_a"])
        assert (gl["replicate"] == 1).any()  # block-A replicate pairs survive
        gm = kept[~kept["arm_id"].isin(st.fallback_arms(s2))]
        assert len(gm) == len(out[~out["arm_id"].isin(st.fallback_arms(s2))])  # Gmail untouched
    # a missing (terminal) record still counts as attempted
    import pandas as pd
    miss = full.copy()
    sel = (miss["arm_id"] == "GLG_00") & (miss["task_id"] == "task_e2") & (miss["replicate"] == 0)
    miss.loc[sel, "status"] = "missing"
    assert st.task_universe(miss, s) == "subset_60"
    assert st.task_universe(pd.concat([full[~sel]]), s) == "block_a"


def test_estimate_applies_the_fallback_and_records_the_universe(tmp_path):
    import json
    s, _ = _tree(tmp_path, drop_block_b=("GLK_01",))
    replay.run_estimate(None, s.res_dir, tmp_path / "out", study=s)
    manifest = json.loads((s.res_dir / "manifest.json").read_text())
    assert manifest["task_universe"] == "block_a"
    snap = replay.load_snapshot(s.res_dir)
    gl = snap[snap["arm_id"].isin(st.fallback_arms(s))]
    assert set(gl["task_id"]) == set(st.load_data_manifest(s)["gitlab_block_a"])
    s2, _ = _tree(tmp_path / "full")
    replay.run_estimate(None, s2.res_dir, tmp_path / "out2", study=s2)
    assert json.loads((s2.res_dir / "manifest.json").read_text())["task_universe"] == "subset_60"


def test_prereg9_reservoir_manifest_gains_no_task_universe_key(tmp_path):
    import json
    res = tmp_path / "res"
    res.mkdir()
    for name in ("outcomes_snapshot.jsonl", "noise.json", "parametric_fits.csv", "G_npmle.json"):
        (res / name).write_text("x")
    replay.write_manifest(res, ["G_npmle"])
    assert set(json.loads((res / "manifest.json").read_text())) == {"outcomes_snapshot", "noise", "parametric_fits",
                                                                   "reservoirs"}
