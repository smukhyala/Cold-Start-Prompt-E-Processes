"""Pre-registration 13 (winner check on fresh GitLab tasks): the frozen data dir, the analyzer's gate and its maths.

No model call, no server: the frozen files are read, the collector's own validation functions are run, and the
analyzer is exercised on fabricated (clearly marked) worker logs in tmp dirs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import analyze_winner_check as ana  # noqa: E402
import collect  # noqa: E402
import make_pools as mp  # noqa: E402
import prepare_winner_check as prep  # noqa: E402

from cold_start.types import RunResult, Task  # noqa: E402

DATA = prep.OUT


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DATA / "manifest.json").read_text())


@pytest.fixture(scope="module")
def queue():
    return mp.read_queue(DATA / prep.QUEUE_REL)


@pytest.fixture(scope="module")
def ever_run():
    return prep.ever_run_by_source()


# ---- the frozen queue ------------------------------------------------------------------------


def test_queue_has_n_tasks_times_six_episodes_each_once(manifest, queue):
    n_tasks = manifest["n_tasks"]
    assert len(queue) == n_tasks * 6 == manifest["n_episodes"]
    keys = set()
    for item in queue:
        keys.add((item.task_id, item.arm_id, item.replicate))
    assert len(keys) == len(queue)
    assert len({q.task_id for q in queue}) == n_tasks
    for task in manifest["task_ids"]:
        cells = set()
        for item in queue:
            if item.task_id == task:
                cells.add((item.arm_id, item.replicate))
        expect = set()
        for role in ("W", "D", "O"):
            for rep in (0, 1):
                expect.add((manifest["arms"][role]["arm_id"], rep))
        assert cells == expect


def test_queue_indices_contiguous_and_fields_match_the_heterogeneity_schema(queue):
    for i in range(len(queue)):
        assert queue[i].index == i
        assert queue[i].pilot is False
    het_first = json.loads((prep.HET / prep.QUEUE_REL).read_text().splitlines()[0])
    ours_first = json.loads((DATA / prep.QUEUE_REL).read_text().splitlines()[0])
    assert list(ours_first.keys()) == list(het_first.keys())


def test_queue_order_is_seeded_and_groups_each_task(manifest, queue):
    again = prep.build_queue(sorted(manifest["task_ids"]), manifest["selected_arm"])
    assert again == queue
    seen = []
    for item in queue:
        if not seen or seen[-1] != item.task_id:
            seen.append(item.task_id)
    assert seen == manifest["task_ids"] and len(set(seen)) == len(seen)


def test_pools_match_the_arms(manifest, queue):
    w = manifest["arms"]["W"]
    assert w["pool"] == "GLK" and w["arm_id"].startswith("GLK_")
    assert manifest["arms"]["D"]["arm_id"] == "GL_anchor_baseline"
    assert manifest["arms"]["O"]["arm_id"] == "GL_anchor_oracle"
    prompts = collect.load_prompts(DATA, "gitlab")
    collect.check_queue_arms(queue, prompts)


# ---- never-run ---------------------------------------------------------------------------------


def test_all_tasks_are_never_run_under_the_manifests_own_eligibility(manifest, ever_run):
    sets, _ = ever_run
    sources = None if manifest["eligibility"] == "strict" else prep.HET_SOURCES
    ever = prep.ever_run_union(sets, sources)
    assert set(manifest["task_ids"]).isdisjoint(ever)
    assert set(manifest["task_ids"]) <= set(prep.bank_ids())


def test_manifest_records_the_strict_count_and_it_is_what_the_logs_say(manifest, ever_run):
    sets, _ = ever_run
    bank = prep.bank_ids()
    strict = prep.eligible_tasks("strict", bank, sets)
    assert manifest["bank_size"] == len(bank) == 140
    assert manifest["never_run_count_strict"] == len(strict)
    assert manifest["ever_run_count_union"] == len(prep.ever_run_union(sets))
    # the June strong-arm runs cycled through the whole bank: no task is strictly fresh
    assert len(strict) == 0
    if manifest["eligibility"] == "strict":
        assert manifest["n_tasks"] == len(strict)


def test_strict_eligibility_refuses_when_too_few_tasks_are_fresh(ever_run, monkeypatch):
    monkeypatch.setattr(prep, "ever_run_by_source", lambda: ever_run)
    with pytest.raises(ValueError, match="strict eligibility leaves"):
        prep.build_plan("strict")


# ---- manifest / collector checks ----------------------------------------------------------------


def test_manifest_hashes_match_files(manifest):
    assert set(manifest["files"]) == {"gitlab/queue.jsonl", *prep.POOL_FILES}
    for rel, digest in manifest["files"].items():
        assert prep.sha(DATA / rel) == digest
    het = json.loads((prep.HET / "manifest.json").read_text())
    for rel in prep.POOL_FILES:  # pool files are unchanged byte copies of the frozen heterogeneity pools
        assert manifest["files"][rel] == het["files"][rel]
    collect.verify_queue(DATA, "gitlab")


def test_collector_startup_checks_pass_without_a_model(tmp_path):
    lines = prep.validate(DATA)
    assert any("verify_queue" in line for line in lines)
    assert any("check_queue_arms" in line for line in lines)


def test_verify_queue_refuses_a_tampered_queue(tmp_path):
    copy = tmp_path / "data"
    (copy / "gitlab").mkdir(parents=True)
    (copy / "pools").mkdir()
    for rel in ["manifest.json", prep.QUEUE_REL, *prep.POOL_FILES]:
        (copy / rel).write_bytes((DATA / rel).read_bytes())
    lines = (copy / prep.QUEUE_REL).read_text().splitlines()
    lines.reverse()
    (copy / prep.QUEUE_REL).write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="refusing to start"):
        collect.verify_queue(copy, "gitlab")


def test_prepare_is_idempotent_and_refuses_to_overwrite_a_different_manifest(tmp_path, ever_run, monkeypatch):
    monkeypatch.setattr(prep, "ever_run_by_source", lambda: ever_run)
    out = tmp_path / "wc"
    first = prep.prepare("het_unrun", out)
    before = (out / "manifest.json").read_bytes()
    prep.prepare("het_unrun", out)
    assert (out / "manifest.json").read_bytes() == before
    assert first["manifest"]["files"] == json.loads(before)["files"]
    tampered = json.loads(before)
    tampered["seed"] = 1
    (out / "manifest.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="manifest differs"):
        prep.prepare("het_unrun", out)


# ---- the step cap reaches the adapter -------------------------------------------------------------


class _CapRecorder:
    def __init__(self):
        self.max_steps = []

    def register_prompt(self, arm_id, text):
        pass

    def task_by_id(self, task_id):
        return Task(task_id=task_id, payload={}, metadata={})

    def run_arm(self, arm, task, runner, max_steps, artifact_subdir=None):
        self.max_steps.append(max_steps)
        return RunResult(success=True, reward=1.0, steps=3, wallclock_s=1.0,
                         trace={"is_done": True, "errors": []}, tokens={"cost_usd": 0.01})


def test_30_step_cap_is_what_the_collector_hands_the_adapter(tmp_path):
    assert collect.MAX_STEPS == 30
    arm = mp.freeform_arm("F_00", "Be precise. " * 10)
    prompts = {"F_00": mp.PoolArm(pool="F", arm=arm, template="x", text=arm.prompt_guidance, sha256="s")}
    queue = [mp.QueueItem(index=0, pool="F", arm_id="F_00", task_id="t0", replicate=0, pilot=False)]
    adapter = _CapRecorder()
    cfg = collect.WorkerConfig(worker=0, n_workers=1, log_dir=tmp_path, budget_usd=10.0, profile="gitlab")
    assert collect.run_worker(cfg, queue, prompts, adapter) == "done"
    assert adapter.max_steps == [30]
    text = (ROOT / "src" / "cold_start" / "tasks" / "webarena.py").read_text()
    assert "agent.run(max_steps=self.max_steps)" in text  # the 2026-10-10 fix: the cap reaches browser-use's run()


# ---- the analyzer ----------------------------------------------------------------------------------


def _synthetic(tmp_path, seed=0):
    ana.make_synthetic_log(DATA, tmp_path / "logs", seed=seed)
    return tmp_path / "logs"


def test_analyzer_refuses_a_log_missing_an_episode(tmp_path):
    logs = _synthetic(tmp_path)
    lines = (logs / "worker_0.jsonl").read_text().splitlines()
    (logs / "worker_0.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ana.AnalysisRefused, match="1 of 660 queued episodes have no terminal record"):
        ana.analyze(DATA, logs, tmp_path / "out", allow_synthetic=True)
    assert not (tmp_path / "out" / "primary.json").exists()


def test_analyzer_refuses_a_missing_status_episode(tmp_path):
    logs = _synthetic(tmp_path)
    lines = (logs / "worker_0.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    last["status"] = "missing"
    last["success"] = None
    lines[-1] = json.dumps(last)
    (logs / "worker_0.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ana.AnalysisRefused, match="not status ok"):
        ana.analyze(DATA, logs, tmp_path / "out", allow_synthetic=True)


def test_analyzer_refuses_infra_error_only_episode_and_missing_dir(tmp_path):
    logs = _synthetic(tmp_path)
    lines = (logs / "worker_0.jsonl").read_text().splitlines()
    last = json.loads(lines[-1])
    last["status"] = "infra_error"
    lines[-1] = json.dumps(last)
    (logs / "worker_0.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ana.AnalysisRefused, match="no terminal record"):
        ana.analyze(DATA, logs, tmp_path / "out", allow_synthetic=True)
    with pytest.raises(ana.AnalysisRefused, match="does not exist"):
        ana.analyze(DATA, tmp_path / "nowhere", tmp_path / "out")


def test_analyzer_never_analyses_synthetic_records_as_real(tmp_path):
    logs = _synthetic(tmp_path)
    with pytest.raises(ana.AnalysisRefused, match="fabricated data is never analysed"):
        ana.analyze(DATA, logs, tmp_path / "out", allow_synthetic=False)


def test_analyzer_refuses_a_wrong_prompt_sha(tmp_path):
    logs = _synthetic(tmp_path)
    lines = (logs / "worker_0.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["prompt_sha256"] = "0" * 64
    lines[0] = json.dumps(first)
    (logs / "worker_0.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ana.AnalysisRefused, match="prompt_sha256"):
        ana.analyze(DATA, logs, tmp_path / "out", allow_synthetic=True)


def test_synthetic_dry_run_writes_the_three_outputs(tmp_path):
    primary = ana.synthetic_dry_run(DATA, tmp_path / "res")
    for name in ("primary.json", "task_table.csv", "secondary.json"):
        assert (tmp_path / "res" / name).exists()
    on_disk = json.loads((tmp_path / "res" / "primary.json").read_text())
    assert on_disk["synthetic"] is True and "SYNTHETIC" in on_disk["WARNING"]
    assert on_disk["n_tasks"] == primary["n_tasks"] == 110
    sec = json.loads((tmp_path / "res" / "secondary.json").read_text())
    assert set(sec["anytime_valid_reference"]["paths"]) == {
        "E5_futility_delta_0.10", "E5_futility_delta_0.15",
        "E1b_efficacy_O_vs_D_replicate0", "E1b_efficacy_O_vs_D_replicate1"}
    assert sec["task_cluster_bootstrap"]["n_boot"] == 10000
    assert set(sec["operations_by_arm"]) == {"W", "D", "O"}
    rows = (tmp_path / "res" / "task_table.csv").read_text().splitlines()
    assert len(rows) == 111


def test_real_results_dir_is_untouched_by_the_synthetic_cli(tmp_path):
    assert ana.main(["--synthetic", "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "primary.json").exists()
    with pytest.raises(SystemExit):
        ana.main(["--synthetic", "--out", str(ana.RESULTS)])


def test_upper_bound_matches_scipy_on_a_toy_example():
    d = np.array([0.5, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, -0.5, 0.5, 0.0])
    n = len(d)
    got = ana.t_summary(d)
    lo, hi = stats.t.interval(0.90, n - 1, loc=d.mean(), scale=stats.sem(d))  # two-sided 90%
    assert got["upper_bound_one_sided"] == pytest.approx(hi)
    assert got["lower_bound_one_sided"] == pytest.approx(lo)
    assert got["interval_two_sided_90"] == pytest.approx([lo, hi])
    # UB < margin  <=>  one-sided t test of H0: mean >= margin rejects at 5%
    for margin in (0.05, 0.2, 0.3, 0.5):
        p = stats.ttest_1samp(d, margin, alternative="less").pvalue
        assert (got["upper_bound_one_sided"] < margin) == (p < 0.05)


def test_bootstrap_is_seeded_and_brackets_the_t_bound_loosely():
    rng = np.random.default_rng(1)
    d = rng.choice([-0.5, 0.0, 0.5], size=80)
    a = ana.bootstrap_means(d, d + 0.3, n_boot=2000)
    b = ana.bootstrap_means(d, d + 0.3, n_boot=2000)
    assert a == b
    t = ana.t_summary(d)
    assert abs(a["W_minus_D"]["upper_bound_one_sided_95"] - t["upper_bound_one_sided"]) < 0.03
