"""The collector: resumable, partitioned, budget-capped, and honest about infra failures."""

from __future__ import annotations

import json
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


def test_classify():
    ok = RunResult(success=False, reward=0.0, steps=30, wallclock_s=180.0, trace={"timed_out": True})
    assert collect.classify(ok, None) == emp.STATUS_OK
    api = RunResult(success=False, reward=0.0, steps=2, wallclock_s=5.0,
                    trace={"is_done": False, "errors": ["RateLimitError: Error code: 429"]})
    assert collect.classify(api, None) == emp.STATUS_INFRA
    done_with_noise = RunResult(success=True, reward=1.0, steps=5, wallclock_s=5.0,
                                trace={"is_done": True, "errors": ["Error code: 429"]})
    assert collect.classify(done_with_noise, None) == emp.STATUS_OK
    assert collect.classify(None, RuntimeError("x")) == emp.STATUS_INFRA


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

    adapter = WebArenaInfinityAdapter(artifacts_dir=str(tmp_path))
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
