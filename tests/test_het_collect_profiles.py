"""Collector profiles (prereg9, gitlab, gmail, bridge), `ended_by`, the watchdog's profile
forwarding, and the per-cell / anchor-order gates. Fakes and tmp_path only: no process, no
WebArena, no LLM call, nothing written under data/, logs/ or results/."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in (ROOT / "experiments" / "growing_bandits" / "empirical", ROOT / "scripts"):
    sys.path.insert(0, str(sub))

import collect  # noqa: E402
import gates  # noqa: E402
import make_pools as mp  # noqa: E402
import watchdog_empirical_pool as wd  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.types import RunResult, Task  # noqa: E402

#: The Pre-reg 9 agent, verbatim: every profile must send exactly this except web_app/timeout_s.
PREREG9_AGENT = {
    "web_app": "apps/gmail",
    "task_suite": "real-tasks",
    "use_vision": False,
    "headless": True,
    "timeout_s": 180,
    "llm_provider": "openai",
    "llm_model": "gpt-5.4-mini",
    "llm_reasoning_effort": "low",
}


@pytest.fixture(autouse=True)
def _guard_real_signals(monkeypatch):
    """No test here may reach a real kill syscall (the watchdog's whole job is to send them)."""
    def guard(pid, sig):
        raise AssertionError(f"real kill({pid!r}, {sig!r}) reached from a test")

    monkeypatch.setattr(wd.os, "killpg", guard)
    monkeypatch.setattr(wd.os, "kill", guard)


# ---- the profile table ---------------------------------------------------------------------


def test_profile_table_round_trip():
    p = collect.PROFILES
    assert set(p) == {"prereg9", "gitlab", "gmail", "bridge"}
    het = ROOT / "data" / "heterogeneity"
    expected = {
        "gitlab": ("apps/gitlab-plan-and-track", 600, 140, "gitlab/queue.jsonl",
                   ("pools/GLG.yaml", "pools/GLK.yaml", "pools/anchors_gitlab.yaml")),
        "gmail": ("apps/gmail", 180, 60, "gmail/queue.jsonl", ("pools/GMK.yaml", "pools/anchors_gmail.yaml")),
        "bridge": ("apps/gmail", 600, 60, "bridge/queue.jsonl", ("pools/GMB.yaml",)),
    }
    for name, (web_app, timeout, bank, queue, pools) in expected.items():
        spec = p[name]
        assert spec["agent"] == {**PREREG9_AGENT, "web_app": web_app, "timeout_s": timeout}, name
        assert spec["n_bank_tasks"] == bank
        assert spec["data_dir"] == het
        assert spec["log_dir"] == ROOT / "logs" / "heterogeneity" / name
        assert spec["queue"] == queue
        assert spec["pools"] == pools
        assert spec["frozen"] == (queue, *pools)  # the manifest entries checked before any server starts
        assert spec["budget_usd"] is None  # a paid stage must name its own hard stop
    # distinct ports: two profiles on one machine can never share a WebArena server port
    ports = {name: set(range(s["base_port"], s["base_port"] + 8)) for name, s in p.items()}
    names = sorted(ports)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not ports[a] & ports[b], (a, b)
    # the dynamic accessor agrees with the import-time table
    for name in p:
        assert collect.profile_spec(name) == p[name]


def test_default_profile_reproduces_the_prereg9_collector_exactly():
    spec = collect.PROFILES["prereg9"]
    assert collect.DEFAULT_PROFILE == "prereg9"
    assert spec["data_dir"] == ROOT / "data" / "empirical_pool" == mp.DATA_DIR
    assert spec["log_dir"] == ROOT / "logs" / "empirical_pool" == collect.LOG_DIR
    assert spec["agent"] == PREREG9_AGENT == collect.AGENT
    assert spec["n_bank_tasks"] == 60 == collect.N_BANK_TASKS
    assert spec["queue"] == "queue.jsonl"
    assert spec["pools"] == ("pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml")
    assert spec["frozen"] == ("queue.jsonl",)  # Pre-reg 9 verified only the queue; unchanged
    assert spec["base_port"] == 8001 == collect.BASE_PORT
    assert spec["budget_usd"] == 260.0 == collect.BUDGET_USD
    args = collect.parse_args([])
    assert (args.profile, args.data, args.log_dir, args.budget) == ("prereg9", mp.DATA_DIR, collect.LOG_DIR, 260.0)
    assert collect.WorkerConfig(worker=0, n_workers=1, log_dir=Path("x"), budget_usd=1.0).profile == "prereg9"
    # the watchdog and the gate CLI default to the same place
    assert wd.profile_log_dir("prereg9") == ROOT / "logs" / "empirical_pool"
    assert wd.collector_args(8, True, 40.0) == ["--workers", "8", "--budget", "40.0", "--pilot"]
    assert gates.cli_paths("prereg9", None, None) == (ROOT / "logs" / "empirical_pool",
                                                      ROOT / "data" / "empirical_pool" / "queue.jsonl")


def test_profile_spec_reads_the_prereg9_constants_live(monkeypatch):
    monkeypatch.setattr(collect, "N_BANK_TASKS", 3)
    assert collect.profile_spec("prereg9")["n_bank_tasks"] == 3


def test_parse_args_selects_the_profile_and_data_log_dir_still_override(tmp_path):
    args = collect.parse_args(["--profile", "gitlab", "--budget", "50"])
    assert args.data == ROOT / "data" / "heterogeneity"
    assert args.log_dir == ROOT / "logs" / "heterogeneity" / "gitlab"
    assert args.budget == 50.0
    args = collect.parse_args(["--profile", "bridge", "--budget", "5", "--data", str(tmp_path / "d"),
                               "--log-dir", str(tmp_path / "l")])
    assert (args.data, args.log_dir) == (tmp_path / "d", tmp_path / "l")
    with pytest.raises(SystemExit):
        collect.parse_args(["--profile", "shopping"])


@pytest.mark.parametrize("name", ["gitlab", "gmail", "bridge"])
def test_a_heterogeneity_profile_refuses_to_start_without_an_explicit_budget(tmp_path, monkeypatch, name):
    logs = tmp_path / "logs"
    monkeypatch.setattr(collect, "_worker_main", lambda *a, **kw: pytest.fail("no worker may start"))
    with pytest.raises(SystemExit):
        collect.main(["--profile", name, "--data", str(tmp_path), "--log-dir", str(logs)])
    assert not (logs / "STATUS").exists() and not (logs / "collect.lock").exists()


def test_archive_out_of_queue_is_prereg9_only(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "archive_out_of_queue", lambda *a, **kw: pytest.fail("must not archive"))
    with pytest.raises(SystemExit):
        collect.main(["--profile", "gitlab", "--budget", "5", "--archive-out-of-queue", "--data", str(tmp_path),
                      "--log-dir", str(tmp_path / "logs")])


# ---- a small data/heterogeneity fixture ----------------------------------------------------

GL_TASKS = ["task_e1", "task_m1", "task_h1"]
GL_ARMS = [("GLG", "GLG_00"), ("GLK", "GLK_00"), ("anchor", "GL_anchor_baseline"),
           ("anchor", "GL_anchor_explorer"), ("anchor", "GL_anchor_oracle")]


def _het_data(tmp_path):
    """What make_het_pools writes for the gitlab profile: pools/, gitlab/queue.jsonl, manifest.json."""
    data = tmp_path / "heterogeneity"
    vec = mp.PromptVector(**mp.BASELINE_VECTOR)
    grid, free, gl_tmpl = mp.GRID_TEMPLATE, mp.FREEFORM_TEMPLATE, ROOT / "configs" / "template_gitlab.jinja"
    mp.write_pool(data / "pools" / "GLG.yaml", "GLG", [mp.grid_arm("GLG_00", vec)], grid, mp.AXES_PATH, meta={})
    mp.write_pool(data / "pools" / "GLK.yaml", "GLK", [mp.freeform_arm("GLK_00", "Open the board first. " * 10)],
                  free, mp.AXES_PATH, meta={})
    mp.write_pool(data / "pools" / "anchors_gitlab.yaml", "anchor",
                  [mp.grid_arm(a, vec) for p, a in GL_ARMS if p == "anchor"], gl_tmpl, mp.AXES_PATH, meta={})
    queue, k = [], 0
    for pool, arm in GL_ARMS:
        for task in GL_TASKS:
            queue.append(mp.QueueItem(index=k, pool=pool, arm_id=arm, task_id=task, replicate=0, pilot=k < 6))
            k += 1
    (data / "gitlab").mkdir(parents=True)
    mp.write_queue(data / "gitlab" / "queue.jsonl", queue)
    files = ["pools/GLG.yaml", "pools/GLK.yaml", "pools/anchors_gitlab.yaml", "gitlab/queue.jsonl"]
    (data / "manifest.json").write_text(json.dumps({"files": {f: mp.file_sha256(data / f) for f in files}}))
    return data


def _thread_pool(max_workers, mp_context=None):
    import concurrent.futures as cf

    return cf.ThreadPoolExecutor(max_workers=max_workers)


def test_load_prompts_reads_the_profiles_pools(tmp_path):
    data = _het_data(tmp_path)
    prompts = collect.load_prompts(data, "gitlab")
    assert sorted(prompts) == sorted(a for _, a in GL_ARMS)
    assert {prompts[a].pool for _, a in GL_ARMS} == {"GLG", "GLK", "anchor"}


def test_verify_queue_checks_the_profiles_manifest_entries(tmp_path):
    data = _het_data(tmp_path)
    collect.verify_queue(data, "gitlab")
    with open(data / "gitlab" / "queue.jsonl", "a") as fh:
        fh.write("\n")
    with pytest.raises(RuntimeError, match="gitlab/queue.jsonl sha256"):
        collect.verify_queue(data, "gitlab")


def test_verify_queue_also_freezes_the_profiles_pool_files(tmp_path):
    data = _het_data(tmp_path)
    with open(data / "pools" / "GLK.yaml", "a") as fh:
        fh.write("# edited\n")
    with pytest.raises(RuntimeError, match="pools/GLK.yaml sha256"):
        collect.verify_queue(data, "gitlab")


def test_verify_queue_refuses_a_profile_missing_from_the_manifest(tmp_path):
    data = _het_data(tmp_path)
    with pytest.raises(RuntimeError, match="bridge/queue.jsonl"):
        collect.verify_queue(data, "bridge")


def test_check_queue_arms_refuses_an_arm_or_pool_the_prompts_do_not_have(tmp_path):
    data = _het_data(tmp_path)
    prompts = collect.load_prompts(data, "gitlab")
    queue = mp.read_queue(data / "gitlab" / "queue.jsonl")
    collect.check_queue_arms(queue, prompts)
    stray = mp.QueueItem(index=99, pool="GLG", arm_id="GLG_77", task_id="task_e1", replicate=0, pilot=False)
    with pytest.raises(RuntimeError, match="GLG_77"):
        collect.check_queue_arms(queue + [stray], prompts)
    relabelled = mp.QueueItem(index=99, pool="GLG", arm_id="GLK_00", task_id="task_e1", replicate=0, pilot=False)
    with pytest.raises(RuntimeError, match="GLK_00"):
        collect.check_queue_arms(queue + [relabelled], prompts)


def test_main_runs_the_gitlab_profile_against_its_own_queue(tmp_path, monkeypatch):
    data, logs = _het_data(tmp_path), tmp_path / "logs"
    seen = {}

    def fake_worker(worker, n_workers, data_dir, log_dir, budget, pilot, assigned, profile="MISSING"):
        seen[worker] = (profile, data_dir, budget, pilot, assigned)
        return "done"

    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", _thread_pool)
    monkeypatch.setattr(collect, "_worker_main", fake_worker)
    code = collect.main(["--profile", "gitlab", "--workers", "2", "--budget", "7.5", "--pilot",
                         "--data", str(data), "--log-dir", str(logs)])
    assert code == collect.EXIT_CODES["done"]
    assert (logs / "STATUS").read_text().strip() == "done"
    assert seen[0] == ("gitlab", str(data), 7.5, True, (0, 2, 4))
    assert seen[1] == ("gitlab", str(data), 7.5, True, (1, 3, 5))


class _ProfileAdapter:
    instances: list = []

    def __init__(self, **kw):
        self.kw = kw
        self.tasks = [f"task_x{i}" for i in range(140 - len(GL_TASKS))] + GL_TASKS
        self.prompts, self.calls = {}, []
        _ProfileAdapter.instances.append(self)

    def reset(self, seed):
        pass

    def close(self):
        pass

    def task_ids(self):
        return list(self.tasks)

    def task_by_id(self, task_id):
        return Task(task_id=task_id, payload={}, metadata={})

    def register_prompt(self, arm_id, text):
        self.prompts[arm_id] = text

    def run_arm(self, arm, task, runner, max_steps, artifact_subdir=None):
        self.calls.append((arm.arm_id, task.task_id, max_steps))
        return RunResult(success=True, reward=1.0, steps=max_steps, wallclock_s=1.0,
                         trace={"is_done": False, "errors": []}, tokens={"cost_usd": 0.01})


def test_worker_main_builds_the_adapter_and_records_from_the_profile(tmp_path, monkeypatch):
    from cold_start.tasks import webarena as wa

    data, logs = _het_data(tmp_path), tmp_path / "logs"
    _ProfileAdapter.instances.clear()
    monkeypatch.setattr(wa, "WebArenaInfinityAdapter", _ProfileAdapter)
    out = collect._worker_main(1, 2, str(data), str(logs), 100.0, True, (1, 3), profile="gitlab")
    assert out == "done"
    adapter = _ProfileAdapter.instances[-1]
    spec = collect.PROFILES["gitlab"]
    assert adapter.kw["port"] == spec["base_port"] + 1
    for key, value in spec["agent"].items():
        assert adapter.kw[key] == value
    assert adapter.kw["web_app"] == "apps/gitlab-plan-and-track" and adapter.kw["timeout_s"] == 600
    assert set(adapter.prompts) == {a for _, a in GL_ARMS}
    assert [c[2] for c in adapter.calls] == [collect.MAX_STEPS] * 2
    recs = emp.load_attempts(sorted(logs.glob("worker_*.jsonl")))
    assert set(recs["profile"]) == {"gitlab"}
    assert set(recs["timeout_s"]) == {600}
    assert set(recs["ended_by"]) == {"steps"}
    assert set(recs["port"]) == {spec["base_port"] + 1}


def test_worker_main_refuses_a_bank_of_the_wrong_size(tmp_path, monkeypatch):
    from cold_start.tasks import webarena as wa

    class _SmallBank(_ProfileAdapter):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.tasks = GL_TASKS

    data, logs = _het_data(tmp_path), tmp_path / "logs"
    monkeypatch.setattr(wa, "WebArenaInfinityAdapter", _SmallBank)
    with pytest.raises(RuntimeError, match="expected 140"):
        collect._worker_main(0, 1, str(data), str(logs), 100.0, False, (0,), profile="gitlab")


# ---- ended_by ------------------------------------------------------------------------------


def _rr(steps, trace):
    return RunResult(success=False, reward=0.0, steps=steps, wallclock_s=1.0, trace=trace, tokens={})


def test_ended_by_classifies_clock_steps_and_agent():
    assert collect.ended_by(_rr(30, {"timed_out": True}), 30) == "clock"
    assert collect.ended_by(_rr(12, {"timed_out": True, "is_done": False}), 30) == "clock"
    assert collect.ended_by(_rr(30, {"is_done": False}), 30) == "steps"
    assert collect.ended_by(_rr(31, {}), 30) == "steps"
    assert collect.ended_by(_rr(30, {"is_done": True}), 30) == "agent"
    assert collect.ended_by(_rr(7, {"is_done": False}), 30) == "agent"
    assert collect.ended_by(None, 30) is None  # the harness raised: nothing ended an episode


def test_records_carry_profile_timeout_and_ended_by(tmp_path):
    class _Scripted:
        def __init__(self):
            self.outs = [_rr(30, {"timed_out": True}), _rr(30, {"is_done": False}), _rr(4, {"is_done": True})]

        def task_by_id(self, task_id):
            return Task(task_id=task_id, payload={}, metadata={})

        def register_prompt(self, arm_id, text):
            pass

        def run_arm(self, arm, task, runner, max_steps, artifact_subdir=None):
            return self.outs.pop(0)

    arm = mp.freeform_arm("F_00", "Be precise. " * 10)
    prompts = {"F_00": mp.PoolArm(pool="F", arm=arm, template="x", text=arm.prompt_guidance, sha256="s")}
    queue = [mp.QueueItem(index=i, pool="F", arm_id="F_00", task_id=f"t{i}", replicate=0, pilot=False)
             for i in range(3)]
    cfg = collect.WorkerConfig(worker=0, n_workers=1, log_dir=tmp_path, budget_usd=100.0, profile="bridge",
                               timeout_s=600, base_port=9000)
    collect.run_worker(cfg, queue, prompts, _Scripted())
    recs = emp.load_attempts([tmp_path / "worker_0.jsonl"])
    assert recs["ended_by"].tolist() == ["clock", "steps", "agent"]
    assert set(recs["profile"]) == {"bridge"} and set(recs["timeout_s"]) == {600}
    assert set(recs["port"]) == {9000}


# ---- watchdog ------------------------------------------------------------------------------


def test_watchdog_profile_log_dirs_match_the_collectors():
    assert set(wd.PROFILE_NAMES) == set(collect.PROFILES)
    for name, spec in collect.PROFILES.items():
        assert wd.profile_log_dir(name) == spec["log_dir"], name
    assert wd.profile_log_dir("gitlab") == ROOT / "logs" / "heterogeneity" / "gitlab"


def test_watchdog_collector_args_forward_the_profile_and_its_log_dir():
    log_dir = wd.profile_log_dir("gitlab")
    extra = wd.collector_args(8, True, 40.0, profile="gitlab", log_dir=log_dir)
    assert extra[extra.index("--profile") + 1] == "gitlab"
    assert Path(extra[extra.index("--log-dir") + 1]) == ROOT / "logs" / "heterogeneity" / "gitlab"
    assert extra[:5] == ["--workers", "8", "--budget", "40.0", "--pilot"]
    # the collector parses exactly what the watchdog forwards, and lands in the same log dir
    args = collect.parse_args(extra)
    assert args.profile == "gitlab" and args.log_dir == log_dir and args.budget == 40.0 and args.pilot


def _proc(alive):
    p = mock.Mock(spec=subprocess.Popen)
    p.poll.return_value = None if alive else 1
    p.pid = 4242
    return p


def test_watchdog_main_runs_a_profile_in_its_own_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "HET_LOG_ROOT", tmp_path / "het")
    monkeypatch.setattr(wd, "LOG_DIR", tmp_path / "prereg9")  # must stay untouched
    monkeypatch.setattr("time.sleep", lambda s: None)
    launches = []

    def launch(extra, log_dir_):
        launches.append((list(extra), log_dir_, (log_dir_ / "STATUS").read_text().strip()))
        return _proc(True)

    monkeypatch.setattr(wd, "_launch", launch)
    monkeypatch.setattr(wd, "_status", lambda log_dir_: "done")
    monkeypatch.setattr(wd, "line_count", lambda log_dir_: 0)
    assert wd.main(["--profile", "gitlab", "--pilot", "--budget", "40", "--workers", "8"]) == 0
    log_dir = tmp_path / "het" / "gitlab"
    extra, launched_in, status = launches[0]
    assert launched_in == log_dir and status == "launching"
    assert extra == ["--workers", "8", "--budget", "40.0", "--pilot", "--profile", "gitlab", "--log-dir", str(log_dir)]
    assert (log_dir / wd.RELAUNCH_LOG).exists()
    assert not (tmp_path / "prereg9").exists()


def test_watchdog_refuses_a_heterogeneity_profile_without_a_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "HET_LOG_ROOT", tmp_path / "het")
    monkeypatch.setattr(wd, "_launch", lambda *a: pytest.fail("must not launch"))
    with pytest.raises(SystemExit):
        wd.main(["--profile", "gmail"])
    assert not (tmp_path / "het").exists()


def test_watchdog_print_log_dir_prints_and_touches_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(wd, "HET_LOG_ROOT", tmp_path / "het")
    monkeypatch.setattr(wd, "LOG_DIR", tmp_path / "prereg9")
    monkeypatch.setattr(wd, "_launch", lambda *a: pytest.fail("must not launch"))
    assert wd.main(["--print-log-dir", "--profile", "bridge", "--budget", "9"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "het" / "bridge")
    assert wd.main(["--print-log-dir"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "prereg9")
    assert not (tmp_path / "het").exists() and not (tmp_path / "prereg9").exists()


def test_launcher_asks_the_watchdog_for_its_log_dir_and_passes_every_argument_unchanged():
    text = (ROOT / "scripts" / "run_empirical_pool.sh").read_text()
    assert "--print-log-dir \"$@\"" in text
    assert "scripts/watchdog_empirical_pool.py \"$@\"" in text
    body = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("logs/empirical_pool" in line for line in body)  # the log dir comes from the profile


# ---- gates ---------------------------------------------------------------------------------


def _pilot_rows(oracle_successes, explorer_successes, n=30, workers=8, cost=0.1):
    rows = []
    arms = [("GLG", "GLG_00"), ("GLK", "GLK_00"), ("anchor", "GL_anchor_baseline"),
            ("anchor", "GL_anchor_oracle"), ("anchor", "GL_anchor_explorer")]
    for pool, arm in arms:
        for j in range(n):
            if arm == "GL_anchor_oracle":
                success = int(j < oracle_successes)
            elif arm == "GL_anchor_explorer":
                success = int(j < explorer_successes)
            else:
                success = j % 2
            rows.append({"pool": pool, "arm_id": arm, "task_id": f"t{j}", "replicate": 0, "attempt": 1,
                         "status": "ok", "success": success, "cost_usd": cost, "worker": j % workers})
    return pd.DataFrame(rows)


def _queue_of(frame, pilot=True, extra=()):
    keys = list(dict.fromkeys(zip(frame["pool"], frame["arm_id"], frame["task_id"], frame["replicate"], strict=True)))
    keys += list(extra)
    return [mp.QueueItem(index=i, pool=p, arm_id=a, task_id=t, replicate=int(r), pilot=pilot)
            for i, (p, a, t, r) in enumerate(keys)]


def _gitlab_pilot(frame, **kw):
    base = dict(n_workers=8, anchor_arm="GL_anchor_baseline", anchor_rate=None, relaunches=0,
                max_cost=0.25, anchor_order=("GL_anchor_oracle", "GL_anchor_explorer"))
    base.update(kw)
    return gates.pilot_gate(frame, _queue_of(frame), **base)


def test_pilot_gate_anchor_order_passes_when_oracle_beats_explorer():
    g = _gitlab_pilot(_pilot_rows(oracle_successes=20, explorer_successes=5))
    assert g.passed, g.report()
    assert "anchor_order" in g.checks and "anchor_drift" not in g.checks


@pytest.mark.parametrize("oracle, explorer", [(10, 10), (5, 20)])
def test_pilot_gate_anchor_order_fails_when_explorer_ties_or_beats_oracle(oracle, explorer):
    g = _gitlab_pilot(_pilot_rows(oracle_successes=oracle, explorer_successes=explorer))
    assert not g.passed and not g.checks["anchor_order"]["passed"]
    assert all(c["passed"] for k, c in g.checks.items() if k != "anchor_order"), g.report()


def test_pilot_gate_anchor_order_fails_without_ok_anchor_episodes():
    frame = _pilot_rows(20, 5)
    frame = frame[frame["arm_id"] != "GL_anchor_explorer"]
    g = _gitlab_pilot(frame)
    assert not g.checks["anchor_order"]["passed"]


def test_gitlab_pilot_reports_the_baseline_anchor_without_a_drift_test():
    g = _gitlab_pilot(_pilot_rows(20, 5))
    check = g.checks["anchor_baseline_rate"]
    assert check["passed"] and "15/30" in check["detail"] and "not gated" in check["detail"]


def test_gitlab_pilot_cost_limit_is_025():
    assert not _gitlab_pilot(_pilot_rows(20, 5, cost=0.26)).checks["cost_per_episode"]["passed"]
    assert _gitlab_pilot(_pilot_rows(20, 5, cost=0.24)).checks["cost_per_episode"]["passed"]
    assert gates.GATE_PROFILES["gitlab"]["pilot"]["max_cost"] == 0.25


def test_pilot_gate_keeps_its_prereg9_checks_when_the_new_arguments_are_omitted():
    frame = _pilot_rows(20, 5)
    frame.loc[frame["arm_id"] == "GL_anchor_baseline", "success"] = [int(j < 20) for j in range(30)]
    g = gates.pilot_gate(frame, _queue_of(frame), n_workers=8, anchor_arm="GL_anchor_baseline", anchor_rate=0.66,
                         relaunches=0)
    assert set(g.checks) == {"cost_per_episode", "records_in_queue", "missing_rate", "no_watchdog_relaunch",
                             "every_worker_produced", "anchor_drift"}


def _cells(missing_a, n_a=100, missing_b=0, n_b=100):
    rows = []
    for pool, n, n_missing in (("GLG", n_a, missing_a), ("GLK", n_b, missing_b)):
        for i in range(n):
            status = "missing" if i < n_missing else "ok"
            rows.append({"pool": pool, "arm_id": f"{pool}_{i % 5:02d}", "task_id": f"t{i}", "replicate": 0,
                         "attempt": 3 if status == "missing" else 1, "status": status,
                         "success": None if status == "missing" else 1, "cost_usd": 0.1, "worker": 0})
    return pd.DataFrame(rows)


def test_collection_gate_fails_one_cell_at_6_percent_even_when_the_total_is_3():
    frame = _cells(missing_a=6)
    q = _queue_of(frame, pilot=False)
    g = gates.collection_gate(frame, q, per_pool=True)
    assert g.checks["missing_rate"]["passed"]  # 6 / 200 = 3% overall
    assert not g.checks["missing_rate[GLG]"]["passed"]
    assert g.checks["missing_rate[GLK]"]["passed"]
    assert not g.passed
    assert gates.collection_gate(frame, q).passed  # Pre-reg 9's pooled rule, unchanged when not asked


def test_collection_gate_per_pool_counts_never_attempted_items():
    frame = _cells(missing_a=0)
    extra = [("GLK", "GLK_09", f"never{i}", 0) for i in range(6)]
    g = gates.collection_gate(frame, _queue_of(frame, pilot=False, extra=extra), per_pool=True)
    assert not g.checks["missing_rate[GLK]"]["passed"]
    assert "6 with no terminal record" in g.checks["missing_rate[GLK]"]["detail"]
    assert g.checks["missing_rate[GLG]"]["passed"]


def test_gmail_collection_gate_tests_the_anchor_for_drift():
    rows = []
    for j in range(30):
        rows.append({"pool": "anchor", "arm_id": "GM_anchor_baseline", "task_id": f"t{j}", "replicate": 0,
                     "attempt": 1, "status": "ok", "success": int(j < 20), "cost_usd": 0.1, "worker": 0})
    frame = pd.DataFrame(rows)
    cfg = gates.GATE_PROFILES["gmail"]["collection"]
    g = gates.collection_gate(frame, _queue_of(frame, pilot=False), **cfg)
    assert g.checks["anchor_drift"]["passed"], g.report()
    frame["success"] = [int(j < 8) for j in range(30)]
    assert not gates.collection_gate(frame, _queue_of(frame, pilot=False), **cfg).checks["anchor_drift"]["passed"]


def test_gate_cli_paths_follow_the_profile(tmp_path):
    assert gates.cli_paths("gitlab", None, None) == (ROOT / "logs" / "heterogeneity" / "gitlab",
                                                     ROOT / "data" / "heterogeneity" / "gitlab" / "queue.jsonl")
    assert gates.cli_paths("bridge", tmp_path / "l", tmp_path / "q") == (tmp_path / "l", tmp_path / "q")


def _write_logs(log_dir, frame):
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "worker_0.jsonl", "w") as fh:
        for rec in frame.to_dict("records"):
            fh.write(json.dumps(rec) + "\n")


def test_gate_cli_runs_the_gitlab_pilot_gate(tmp_path, capsys):
    frame = _pilot_rows(20, 5)
    logs, queue_path = tmp_path / "logs", tmp_path / "queue.jsonl"
    _write_logs(logs, frame)
    (logs / "relaunches.log").touch()
    mp.write_queue(queue_path, _queue_of(frame))
    code = gates.main(["pilot", "--profile", "gitlab", "--log-dir", str(logs), "--queue", str(queue_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "anchor_order" in out and "anchor_drift" not in out and "limit $0.25" in out


def test_gate_cli_has_no_pilot_for_gmail_or_bridge(tmp_path, capsys):
    for name in ("gmail", "bridge"):
        code = gates.main(["pilot", "--profile", name, "--log-dir", str(tmp_path), "--queue", str(tmp_path / "q")])
        assert code == 2
        assert "no pilot" in capsys.readouterr().out


def test_gate_cli_collection_is_per_cell_for_a_heterogeneity_profile(tmp_path, capsys):
    frame = _cells(missing_a=6)
    logs, queue_path = tmp_path / "logs", tmp_path / "queue.jsonl"
    _write_logs(logs, frame)
    mp.write_queue(queue_path, _queue_of(frame, pilot=False))
    assert gates.main(["collection", "--profile", "gitlab", "--log-dir", str(logs), "--queue", str(queue_path)]) == 1
    assert "missing_rate[GLG]" in capsys.readouterr().out
