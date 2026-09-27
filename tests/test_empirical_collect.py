"""The collector: resumable, partitioned, budget-capped, and honest about infra failures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import collect  # noqa: E402
import make_pools as mp  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.types import RunResult, Task  # noqa: E402

TASKS = ["t0", "t1", "t2"]


class FakeAdapter:
    """Scripted outcomes: ``script[(arm_id, task_id)]`` is a list consumed one call at a time."""

    def __init__(self, script=None, default_cost=0.01, tasks=TASKS):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default_cost = default_cost
        self.tasks = tasks
        self.calls, self.prompts, self.resets = [], {}, 0

    def task_ids(self):
        return list(self.tasks)

    def task_by_id(self, task_id):
        return Task(task_id=task_id, payload={}, metadata={})

    def register_prompt(self, arm_id, text):
        self.prompts[arm_id] = text

    def run_arm(self, arm, task, runner, max_steps):
        self.calls.append((arm.arm_id, task.task_id))
        todo = self.script.get((arm.arm_id, task.task_id))
        out = todo.pop(0) if todo else True
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, RunResult):
            return out
        return RunResult(success=bool(out), reward=float(bool(out)), steps=3, wallclock_s=1.0,
                         trace={"is_done": True, "errors": []}, tokens={"cost_usd": self.default_cost})

    def close(self):
        pass

    def reset(self, seed):
        self.resets += 1


def _prompts():
    arms = [mp.freeform_arm("F_00", "Be precise. " * 10), mp.freeform_arm("F_01", "Be quick. " * 10)]
    return {a.arm_id: mp.PoolArm(pool="F", arm=a, template="x", text=a.prompt_guidance, sha256="s") for a in arms}


def _queue(pilot_first=0):
    items, k = [], 0
    for arm in ("F_00", "F_01"):
        for task in TASKS:
            items.append(mp.QueueItem(index=k, pool="F", arm_id=arm, task_id=task, replicate=0, pilot=k < pilot_first))
            k += 1
    return items


def _cfg(tmp_path, **kw):
    base = dict(worker=0, n_workers=1, log_dir=tmp_path, budget_usd=100.0)
    base.update(kw)
    return collect.WorkerConfig(**base)


def _records(tmp_path):
    return emp.load_attempts(sorted(tmp_path.glob("worker_*.jsonl")))


def test_runs_every_item_once_and_writes_the_schema(tmp_path):
    adapter = FakeAdapter()
    assert collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter) == "done"
    recs = _records(tmp_path)
    assert len(recs) == 6 and set(recs["status"]) == {"ok"}
    assert set(recs["schema"]) == {emp.SCHEMA}
    assert adapter.prompts.keys() == {"F_00", "F_01"}


def test_workers_partition_the_queue(tmp_path):
    q = _queue()
    a, b = FakeAdapter(), FakeAdapter()
    collect.run_worker(_cfg(tmp_path, worker=0, n_workers=2), q, _prompts(), a)
    collect.run_worker(_cfg(tmp_path, worker=1, n_workers=2), q, _prompts(), b)
    assert len(a.calls) == 3 and len(b.calls) == 3
    assert not set(a.calls) & set(b.calls)


def test_resume_skips_finished_items(tmp_path):
    q = _queue()
    collect.run_worker(_cfg(tmp_path), q[:4], _prompts(), FakeAdapter())
    again = FakeAdapter()
    collect.run_worker(_cfg(tmp_path), q, _prompts(), again)
    assert len(again.calls) == 2
    assert len(emp.terminal_outcomes(_records(tmp_path))) == 6


def test_infra_errors_retry_then_go_missing_never_zero(tmp_path):
    boom = [RuntimeError("browser crashed")] * 3
    adapter = FakeAdapter(script={("F_00", "t0"): boom})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)
    recs = _records(tmp_path)
    cell = recs[(recs["arm_id"] == "F_00") & (recs["task_id"] == "t0")]
    assert cell["status"].tolist() == ["infra_error", "infra_error", "missing"]
    assert cell["success"].isna().all()
    assert adapter.resets == 2


def test_an_infra_error_that_recovers_is_scored(tmp_path):
    adapter = FakeAdapter(script={("F_00", "t0"): [RuntimeError("503"), False]})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)
    out = emp.terminal_outcomes(_records(tmp_path))
    row = out[(out["arm_id"] == "F_00") & (out["task_id"] == "t0")].iloc[0]
    assert row["status"] == "ok" and row["success"] == 0 and row["attempt"] == 2


def test_classify(monkeypatch):
    """Ruling (fix round 1): infra_error iff the episode failed AND ended in a terminal streak
    of provider-error steps (one trailing forced-done step with no error tolerated); a single
    transient error the agent recovered from is scored normally, not resampled."""
    monkeypatch.setattr(collect, "_default_failure_streak", lambda: 2)

    timeout = RunResult(success=False, reward=0.0, steps=30, wallclock_s=180.0, trace={"timed_out": True})
    assert collect.classify(timeout, None) == emp.STATUS_OK

    recovered_then_genuine_failure = RunResult(
        success=False, reward=0.0, steps=6, wallclock_s=30.0,
        trace={"is_done": True, "error_tail": [None, "Error code: 429 - rate limited", None, None]},
    )
    assert collect.classify(recovered_then_genuine_failure, None) == emp.STATUS_OK

    streak_to_the_end = RunResult(
        success=False, reward=0.0, steps=6, wallclock_s=30.0,
        trace={"is_done": False, "error_tail": [None, "Connection error.", "Connection error."]},
    )
    assert collect.classify(streak_to_the_end, None) == emp.STATUS_INFRA

    streak_then_forced_done = RunResult(
        success=False, reward=0.0, steps=7, wallclock_s=30.0,
        trace={"is_done": True, "error_tail": ["Request timed out.", "Connection error.", None]},
    )
    assert collect.classify(streak_then_forced_done, None) == emp.STATUS_INFRA

    success_with_early_429 = RunResult(
        success=True, reward=1.0, steps=5, wallclock_s=10.0,
        trace={"is_done": True, "error_tail": ["Error code: 429 - rate limited", None, None, None]},
    )
    assert collect.classify(success_with_early_429, None) == emp.STATUS_OK

    assert collect.classify(None, RuntimeError("x")) == emp.STATUS_INFRA


def test_default_failure_streak_reads_browser_use_or_falls_back():
    """Reads ``browser_use.agent.views.AgentSettings.max_failures`` (5 today) when importable;
    unimportable/moved falls back to a conservative 3."""
    assert collect._default_failure_streak() == 5


def test_budget_stop(tmp_path):
    adapter = FakeAdapter(default_cost=0.1)
    status = collect.run_worker(_cfg(tmp_path, budget_usd=0.25), _queue(), _prompts(), adapter)
    assert status == "budget"
    assert len(adapter.calls) == 3
    assert collect.spent_usd(tmp_path) == pytest.approx(0.3)


def test_pilot_only(tmp_path):
    adapter = FakeAdapter()
    collect.run_worker(_cfg(tmp_path, pilot_only=True), _queue(pilot_first=2), _prompts(), adapter)
    assert len(adapter.calls) == 2


def test_bank_mismatch_refuses_to_start():
    with pytest.raises(RuntimeError, match="task bank"):
        collect.check_bank(FakeAdapter(tasks=["t0", "t1"]), _queue())


def test_adapter_prompt_hooks(tmp_path):
    from cold_start.tasks.webarena import WebArenaInfinityAdapter

    # Absolute paths so this test doesn't depend on pytest's CWD being the repo root.
    adapter = WebArenaInfinityAdapter(
        artifacts_dir=str(tmp_path),
        axes_path=str(ROOT / "configs" / "axes.yaml"),
        template_path=str(ROOT / "configs" / "template.jinja"),
    )
    arm = mp.grid_arm("G_00", mp.PromptVector(**mp.BASELINE_VECTOR))
    rendered = adapter.prompt_for(arm)
    assert "agent" in rendered
    adapter.register_prompt("G_00", "PINNED")
    assert adapter.prompt_for(arm) == "PINNED"
    adapter._tasks = [{"id": "task_e1", "instruction": "x", "verify": "v"}]
    adapter._web_app_abs = "/tmp"
    assert adapter.task_ids() == ["task_e1"]
    assert adapter.task_by_id("task_e1").task_id == "task_e1"
    with pytest.raises(KeyError):
        adapter.task_by_id("nope")


def test_run_arm_sends_the_pinned_prompt_to_the_agent(tmp_path, monkeypatch):
    """`run_arm` must route the prompt through `prompt_for` (pinned text wins), not re-render."""
    from cold_start.tasks import webarena as wa

    adapter = wa.WebArenaInfinityAdapter(
        artifacts_dir=str(tmp_path),
        axes_path=str(ROOT / "configs" / "axes.yaml"),
        template_path=str(ROOT / "configs" / "template.jinja"),
    )
    arm = mp.grid_arm("G_00", mp.PromptVector(**mp.BASELINE_VECTOR))
    adapter.register_prompt("G_00", "PINNED-TEXT")

    class _FakeAgent:
        def __init__(self):
            self.extension = None
            self.max_steps = None

        def set_prompt_extension(self, extension):
            self.extension = extension

    adapter._agent = _FakeAgent()

    class _FakeTasksMod:
        @staticmethod
        async def run_task(**kwargs):
            return {"passed": True, "steps": 1, "elapsed": 0.1, "verifier_message": "", "final_result": "",
                    "errors": [], "is_done": True}

    monkeypatch.setattr(wa, "_import_webarena", lambda: (None, None, _FakeTasksMod))

    task = Task(task_id="t0", payload={}, metadata={"raw": {"id": "t0"}, "web_app_dir": "/tmp"})
    result = adapter.run_arm(arm, task, None, 30)

    assert adapter._agent.extension == "PINNED-TEXT"
    assert result.success is True
    assert result.trace["error_tail"] == []


# ---- fix round 1: recovery robustness, recycling, the lockfile, and torn writes -----------------


class _CloseRaisesAdapter(FakeAdapter):
    def close(self):
        raise RuntimeError("close boom")


class _NeverResetsAdapter(FakeAdapter):
    def reset(self, seed):
        self.resets += 1
        raise RuntimeError("server down")


class _AlwaysBoomAdapter(FakeAdapter):
    """Every episode is a harness exception, and recovery can never bring the server back."""

    def run_arm(self, arm, task, runner, max_steps):
        self.calls.append((arm.arm_id, task.task_id))
        raise RuntimeError("browser crashed")

    def reset(self, seed):
        self.resets += 1
        raise RuntimeError("server down")


def test_recover_swallows_a_close_error_and_still_resets(monkeypatch):
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    adapter = _CloseRaisesAdapter()
    collect._recover(adapter)  # must not raise
    assert adapter.resets == 1


def test_recover_retries_reset_with_backoff_then_raises(monkeypatch):
    sleeps = []
    monkeypatch.setattr(collect.time, "sleep", lambda s: sleeps.append(s))
    adapter = _NeverResetsAdapter()
    with pytest.raises(RuntimeError, match="server down"):
        collect._recover(adapter)
    assert adapter.resets == collect.RECOVERY_ATTEMPTS
    assert sleeps == list(collect.RECOVERY_BACKOFF_S[:-1])


def test_worker_gives_up_after_repeated_failed_recoveries(tmp_path, monkeypatch):
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    adapter = _AlwaysBoomAdapter()
    with pytest.raises(RuntimeError, match="giving up"):
        collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)


def test_recycles_every_n_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "RECYCLE_EVERY", 2)
    adapter = FakeAdapter()
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)  # 6 episodes, no infra errors
    assert adapter.resets == 3  # recycles after episodes 2, 4, 6


def test_resume_continues_attempt_count_from_a_prior_infra_error(tmp_path):
    """A worker that logged one infra_error attempt and then crashed (before this test starts)
    must resume from attempt 2, not attempt 1 -- and still go missing at attempt 3."""
    cfg = _cfg(tmp_path)
    path = tmp_path / "worker_0.jsonl"
    prompt = _prompts()["F_00"]
    item = mp.QueueItem(index=0, pool="F", arm_id="F_00", task_id="t0", replicate=0, pilot=False)
    collect._append(path, collect._record(item, 1, emp.STATUS_INFRA, None, RuntimeError("boom"), cfg, prompt))

    adapter = FakeAdapter(script={("F_00", "t0"): [RuntimeError("still down")] * 2})
    collect.run_worker(cfg, _queue(), _prompts(), adapter)

    recs = _records(tmp_path)
    cell = recs[(recs["arm_id"] == "F_00") & (recs["task_id"] == "t0")]
    assert cell["attempt"].tolist() == [1, 2, 3]
    assert cell["status"].tolist() == ["infra_error", "infra_error", "missing"]


def test_a_truncated_trailing_line_is_dropped_not_glued(tmp_path):
    """A hard kill mid-append leaves a byte fragment with no trailing newline; a resume must
    drop it (never glue the next append onto it, never choke reading it)."""
    path = tmp_path / "worker_0.jsonl"
    good = json.dumps({"schema": emp.SCHEMA, "pool": "F", "arm_id": "F_00", "task_id": "t0", "replicate": 0,
                       "attempt": 1, "status": "ok", "success": 1, "cost_usd": 0.01})
    path.write_text(good + "\n" + '{"schema": "empirical_pool/1", "pool": "F", "arm_id": "F_0')

    assert collect.spent_usd(tmp_path) == pytest.approx(0.01)

    collect._append(path, {"schema": emp.SCHEMA, "pool": "F", "arm_id": "F_00", "task_id": "t1", "replicate": 0,
                           "attempt": 1, "status": "ok", "success": 1, "cost_usd": 0.02})
    recs = emp.load_attempts([path])
    assert len(recs) == 2
    assert set(recs["task_id"]) == {"t0", "t1"}
    assert recs["cost_usd"].astype(float).sum() == pytest.approx(0.03)


def test_a_complete_malformed_line_still_raises(tmp_path):
    """Only a torn (unterminated) final line is tolerated; a complete-but-garbage line is a
    real data bug and must still surface, exactly as ``load_attempts`` already guarantees."""
    path = tmp_path / "worker_0.jsonl"
    path.write_text("not json at all\n")
    with pytest.raises(json.JSONDecodeError):
        collect.spent_usd(tmp_path)


def test_lock_refuses_when_pid_is_alive(tmp_path):
    lock = tmp_path / "collect.lock"
    lock.write_text(str(os.getpid()))  # our own test process is certainly alive
    with pytest.raises(RuntimeError, match="collector"):
        collect._acquire_lock(lock)


def test_lock_replaces_a_stale_pid(tmp_path, monkeypatch):
    lock = tmp_path / "collect.lock"
    lock.write_text("999999")

    def _dead(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(collect.os, "kill", _dead)
    collect._acquire_lock(lock)  # must not raise; replaces the stale lock
    assert lock.read_text().strip() == str(os.getpid())
    collect._release_lock(lock)
    assert not lock.exists()
