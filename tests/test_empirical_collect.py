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
        self.calls, self.prompts, self.resets, self.subdirs = [], {}, 0, []

    def task_ids(self):
        return list(self.tasks)

    def task_by_id(self, task_id):
        return Task(task_id=task_id, payload={}, metadata={})

    def register_prompt(self, arm_id, text):
        self.prompts[arm_id] = text

    def run_arm(self, arm, task, runner, max_steps, artifact_subdir=None):
        self.calls.append((arm.arm_id, task.task_id))
        self.subdirs.append(artifact_subdir)
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

    def run_arm(self, arm, task, runner, max_steps, artifact_subdir=None):
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


def test_lock_refuses_when_the_process_group_is_alive(tmp_path, monkeypatch):
    """Fix round 1, item 3: the lock names a pid whose process GROUP is still alive (checked
    with `os.killpg`, not a single-pid `os.kill`) -- a real collector runs with
    `start_new_session=True`, so its own pid is its pgid."""
    lock = tmp_path / "collect.lock"
    lock.write_text("4242")
    monkeypatch.setattr(collect.os, "killpg", lambda pgid, sig: None)  # group alive: no exception
    with pytest.raises(RuntimeError, match="collector"):
        collect._acquire_lock(lock)


def test_lock_refuses_when_a_dead_leaders_group_still_has_a_live_member(tmp_path, monkeypatch):
    """The recorded pid (the old leader) may itself be gone, but a worker it spawned can
    still be alive in the same process group -- `PermissionError` from `os.killpg` means the
    group exists (just owned by someone else), which must still count as alive."""
    lock = tmp_path / "collect.lock"
    lock.write_text("4242")
    monkeypatch.setattr(collect.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(RuntimeError, match="collector"):
        collect._acquire_lock(lock)


def test_lock_replaces_a_stale_pid(tmp_path, monkeypatch):
    lock = tmp_path / "collect.lock"
    lock.write_text("999999")

    def _dead(pgid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(collect.os, "killpg", _dead)
    collect._acquire_lock(lock)  # must not raise; replaces the stale lock
    assert lock.read_text().strip() == str(os.getpid())
    collect._release_lock(lock)
    assert not lock.exists()


# ---- final-review fix wave -------------------------------------------------------------------


def _line(task_id, cost=0.01, status="ok", arm="F_00", replicate=0, attempt=1):
    return json.dumps({"schema": emp.SCHEMA, "pool": "F", "arm_id": arm, "task_id": task_id,
                       "replicate": replicate, "attempt": attempt, "status": status,
                       "success": 1 if status == "ok" else None, "cost_usd": cost})


# C2: readers never modify another worker's file; only the owner repairs its own tail.


def test_readers_skip_a_torn_tail_in_memory_and_never_modify_the_file(tmp_path):
    other = tmp_path / "worker_3.jsonl"
    raw = (_line("t0", 0.01) + "\n" + '{"schema": "empirical_pool/1", "pool": "F", "arm_id": "F_0').encode()
    other.write_bytes(raw)
    assert collect.spent_usd(tmp_path) == pytest.approx(0.01)
    done, _ = collect._progress(tmp_path)
    assert done == {("F_00", "t0", 0)}
    assert other.read_bytes() == raw  # a live worker may be mid-write: never truncated by a reader


def test_readers_count_a_complete_but_unterminated_record(tmp_path):
    other = tmp_path / "worker_3.jsonl"
    raw = (_line("t0", 0.01) + "\n" + _line("t1", 0.02)).encode()  # second record's newline not yet written
    other.write_bytes(raw)
    assert collect.spent_usd(tmp_path) == pytest.approx(0.03)  # its spend is real
    done, _ = collect._progress(tmp_path)
    assert ("F_00", "t1", 0) in done
    assert other.read_bytes() == raw


def test_a_worker_run_leaves_another_workers_torn_file_untouched(tmp_path):
    other = tmp_path / "worker_1.jsonl"
    raw = (_line("t0", 0.01, arm="F_01") + "\n" + '{"schema": "empir').encode()
    other.write_bytes(raw)
    collect.run_worker(_cfg(tmp_path, worker=0, n_workers=2), _queue(), _prompts(), FakeAdapter())
    assert other.read_bytes() == raw


def test_owner_terminates_a_complete_unterminated_record_rather_than_dropping_it(tmp_path):
    path = tmp_path / "worker_0.jsonl"
    path.write_text(_line("t0", 0.01) + "\n" + _line("t1", 0.02))
    collect._append(path, json.loads(_line("t2", 0.04)))
    recs = emp.load_attempts([path])  # strict loader: every line must be well formed
    assert list(recs["task_id"]) == ["t0", "t1", "t2"]
    assert recs["cost_usd"].astype(float).sum() == pytest.approx(0.07)


def test_owner_repairs_its_own_tail_before_reading_progress(tmp_path):
    """A kill that left this worker's last terminal record unterminated must not re-run that item."""
    path = tmp_path / "worker_0.jsonl"
    path.write_text(_line("t0"))
    adapter = FakeAdapter()
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)
    assert ("F_00", "t0") not in adapter.calls
    assert len(emp.terminal_outcomes(_records(tmp_path))) == 6


# I1: provider outage backs off, then stops the worker as provider_down.


def _provider_streak_result():
    return RunResult(success=False, reward=0.0, steps=6, wallclock_s=5.0,
                     trace={"is_done": True, "error_tail": ["Error code: 429 - insufficient_quota"] * 5 + [None]},
                     tokens={"cost_usd": 0.0})


def test_provider_errors_back_off_before_each_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_default_failure_streak", lambda: 5)
    sleeps = []
    adapter = FakeAdapter(script={("F_00", "t0"): [_provider_streak_result()] * 2 + [True]})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter, sleep=sleeps.append)
    assert sleeps == [60.0, 300.0]
    recs = _records(tmp_path)
    cell = recs[(recs["arm_id"] == "F_00") & (recs["task_id"] == "t0")]
    assert cell["infra_reason"].fillna("none").tolist() == ["provider", "provider", "none"]


def test_harness_exceptions_do_not_back_off(tmp_path):
    sleeps = []
    adapter = FakeAdapter(script={("F_00", "t0"): [RuntimeError("browser crashed"), True]})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter, sleep=sleeps.append)
    assert sleeps == []


def test_three_items_missing_on_provider_errors_stop_the_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_default_failure_streak", lambda: 5)
    down = [_provider_streak_result()] * 3
    script = {("F_00", t): list(down) for t in TASKS}
    adapter = FakeAdapter(script=script)
    status = collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter, sleep=lambda s: None)
    assert status == "provider_down"
    out = emp.terminal_outcomes(_records(tmp_path))
    assert len(out) == 3 and set(out["status"]) == {"missing"}
    assert not any(arm == "F_01" for arm, _ in adapter.calls)  # the rest of the queue is untouched


def test_an_ok_item_resets_the_provider_down_streak(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_default_failure_streak", lambda: 5)
    down = [_provider_streak_result()] * 3
    script = {("F_00", "t0"): list(down), ("F_00", "t1"): list(down), ("F_01", "t0"): list(down),
              ("F_01", "t1"): list(down)}
    status = collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), FakeAdapter(script=script),
                                sleep=lambda s: None)
    assert status == "done"  # F_00/t2 is ok between the two pairs of missing items


def test_exception_missing_items_do_not_count_as_provider_down(tmp_path):
    boom = [RuntimeError("browser crashed")] * 3
    script = {("F_00", t): list(boom) for t in TASKS}
    status = collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), FakeAdapter(script=script),
                                sleep=lambda s: None)
    assert status == "done"


@pytest.mark.parametrize("outcomes, expected", [
    (["done", "done"], "done"),
    (["done", "budget"], "budget"),
    (["budget", "failed"], "failed"),
    (["failed", "provider_down", "budget"], "provider_down"),
])
def test_final_status_precedence(outcomes, expected):
    assert collect.final_status(outcomes) == expected


def test_classify_covers_every_provider_5xx(monkeypatch):
    monkeypatch.setattr(collect, "_default_failure_streak", lambda: 2)
    for code in (500, 502, 503, 504, 520, 522, 524, 529):
        r = RunResult(success=False, reward=0.0, steps=3, wallclock_s=1.0,
                      trace={"error_tail": [f"Error code: {code} - upstream"] * 2})
        assert collect.classify(r, None) == emp.STATUS_INFRA, code
    r = RunResult(success=False, reward=0.0, steps=3, wallclock_s=1.0,
                  trace={"error_tail": ["Error code: 404 - element not found"] * 2})
    assert collect.classify(r, None) == emp.STATUS_OK


def test_classify_refuses_an_error_tail_too_short_for_the_streak(monkeypatch):
    monkeypatch.setattr(collect, "_default_failure_streak", lambda: collect.ERROR_TAIL_LEN)
    r = RunResult(success=False, reward=0.0, steps=3, wallclock_s=1.0, trace={"error_tail": []})
    with pytest.raises(RuntimeError, match="ERROR_TAIL_LEN"):
        collect.classify(r, None)


def test_error_tail_holds_the_live_streak_plus_one():
    assert collect.ERROR_TAIL_LEN > collect._default_failure_streak()


# I2: audit fields and per-attempt artifacts.


def test_records_carry_the_audit_fields(tmp_path):
    long = "v" * 1000
    result = RunResult(success=False, reward=0.0, steps=4, wallclock_s=2.0,
                       trace={"is_done": True, "verifier_message": long, "final_result": long,
                              "error_tail": [None, "a step error", None], "task_dir": "/x"},
                       tokens={"cost_usd": 0.02})
    adapter = FakeAdapter(script={("F_00", "t0"): [result]})
    collect.run_worker(_cfg(tmp_path), _queue(), _prompts(), adapter)
    recs = _records(tmp_path)
    row = recs[(recs["arm_id"] == "F_00") & (recs["task_id"] == "t0")].iloc[0]
    assert row["verifier_message"] == "v" * collect.TEXT_LEN
    assert row["final_result"] == "v" * collect.TEXT_LEN
    assert bool(row["is_done"]) is True
    assert row["error_tail"] == [None, "a step error", None]


def test_every_attempt_gets_its_own_artifact_dir(tmp_path):
    adapter = FakeAdapter(script={("F_00", "t0"): [RuntimeError("x"), True]})
    q = _queue() + [mp.QueueItem(index=6, pool="F", arm_id="F_00", task_id="t0", replicate=1, pilot=False)]
    collect.run_worker(_cfg(tmp_path), q, _prompts(), adapter)
    assert adapter.subdirs[:2] == ["r0_a1", "r0_a2"]
    assert adapter.subdirs[-1] == "r1_a1"


def test_adapter_run_arm_honours_the_artifact_subdir(tmp_path, monkeypatch):
    from cold_start.tasks import webarena as wa

    adapter = wa.WebArenaInfinityAdapter(artifacts_dir=str(tmp_path), axes_path=str(ROOT / "configs" / "axes.yaml"),
                                         template_path=str(ROOT / "configs" / "template.jinja"))
    adapter._agent = type("A", (), {"set_prompt_extension": lambda self, e: None, "max_steps": None})()

    class _FakeTasksMod:
        @staticmethod
        async def run_task(**kwargs):
            return {"passed": False, "steps": 1, "elapsed": 0.1, "errors": [], "is_done": True,
                    "task_dir_seen": str(kwargs["task_dir"])}

    monkeypatch.setattr(wa, "_import_webarena", lambda: (None, None, _FakeTasksMod))
    arm = mp.grid_arm("G_00", mp.PromptVector(**mp.BASELINE_VECTOR))
    task = Task(task_id="t0", payload={}, metadata={"raw": {"id": "t0"}, "web_app_dir": "/tmp"})
    a = adapter.run_arm(arm, task, None, 30, artifact_subdir="r0_a1")
    b = adapter.run_arm(arm, task, None, 30, artifact_subdir="r0_a2")
    c = adapter.run_arm(arm, task, None, 30)
    assert a.trace["task_dir"] != b.trace["task_dir"]
    assert a.trace["task_dir"].endswith("G_00/t0/r0_a1")
    assert c.trace["task_dir"].endswith("G_00/t0")


# I3: the remaining items are dealt once; the first reset is retried.


def test_partition_remaining_deals_each_open_item_to_exactly_one_worker(tmp_path):
    q = _queue(pilot_first=4)
    path = tmp_path / "worker_5.jsonl"
    path.write_text(_line("t0") + "\n" + _line("t1", status="infra_error") + "\n")
    shares = collect.partition_remaining(q, tmp_path, 3, pilot=False)
    flat = [i for share in shares for i in share]
    assert sorted(flat) == [1, 2, 3, 4, 5]  # item 0 (F_00/t0) is terminal; item 1 only had an infra_error
    assert len(flat) == len(set(flat))
    assert [len(s) for s in shares] == [2, 2, 1]
    pilot_shares = collect.partition_remaining(q, tmp_path, 3, pilot=True)
    assert sorted(i for s in pilot_shares for i in s) == [1, 2, 3]


def test_run_worker_runs_exactly_its_assigned_items(tmp_path):
    adapter = FakeAdapter()
    collect.run_worker(_cfg(tmp_path, worker=1, n_workers=2, assigned=(0, 5)), _queue(), _prompts(), adapter)
    assert adapter.calls == [("F_00", "t0"), ("F_01", "t2")]


def _data_dir(tmp_path):
    """A frozen data dir like make_pools writes: pools, queue and manifest."""
    data = tmp_path / "data"
    data.mkdir()
    tmpl_free, tmpl_grid, axes = (ROOT / "configs" / "template_freeform.jinja", ROOT / "configs" / "template.jinja",
                                  ROOT / "configs" / "axes.yaml")
    mp.write_pool(data / "pool_G.yaml", "G", [mp.grid_arm("G_00", mp.PromptVector(**mp.BASELINE_VECTOR))],
                  tmpl_grid, axes, meta={})
    mp.write_pool(data / "pool_F.yaml", "F", [mp.freeform_arm("F_00", "Be precise. " * 10),
                                              mp.freeform_arm("F_01", "Be quick. " * 10)], tmpl_free, axes, meta={})
    mp.write_pool(data / "pool_anchor.yaml", "anchor",
                  [mp.grid_arm(mp.ANCHOR_ARM_ID, mp.PromptVector(**mp.BASELINE_VECTOR))], tmpl_grid, axes, meta={})
    mp.write_queue(data / "queue.jsonl", _queue())
    files = ["pool_G.yaml", "pool_F.yaml", "pool_anchor.yaml", "queue.jsonl"]
    (data / "manifest.json").write_text(json.dumps({"files": {f: mp.file_sha256(data / f) for f in files}}))
    return data


def test_verify_queue_refuses_a_moved_queue(tmp_path):
    data = _data_dir(tmp_path)
    collect.verify_queue(data)  # untouched: fine
    with open(data / "queue.jsonl", "a") as fh:
        fh.write("\n")
    with pytest.raises(RuntimeError, match="queue.jsonl sha256"):
        collect.verify_queue(data)


def _thread_pool(max_workers, mp_context=None):
    import concurrent.futures as cf

    return cf.ThreadPoolExecutor(max_workers=max_workers)


def test_main_partitions_the_open_items_and_writes_status_and_exit_markers(tmp_path, monkeypatch):
    data, logs = _data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    (logs / "worker_7.jsonl").write_text(_line("t0") + "\n")  # item 0 is already terminal
    (logs / "worker_2.exit").write_text("done\n")  # stale marker from an earlier, wider launch
    seen = {}

    def fake_worker(worker, n_workers, data_dir, log_dir, budget, pilot, assigned):
        seen[worker] = assigned
        return {0: "done", 1: "budget"}[worker]

    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", _thread_pool)
    monkeypatch.setattr(collect, "_worker_main", fake_worker)
    code = collect.main(["--workers", "2", "--data", str(data), "--log-dir", str(logs), "--budget", "5"])
    assert code == collect.EXIT_CODES["budget"]
    assert (logs / "STATUS").read_text().strip() == "budget"
    assert seen == {0: (1, 3, 5), 1: (2, 4)}
    assert (logs / "worker_0.exit").read_text().strip() == "done"
    assert (logs / "worker_1.exit").read_text().strip() == "budget"
    assert not (logs / "worker_2.exit").exists()  # cleared at start: it would exempt a live worker
    assert not (logs / "collect.lock").exists()


def test_main_reports_provider_down_over_a_failed_worker(tmp_path, monkeypatch):
    data, logs = _data_dir(tmp_path), tmp_path / "logs"

    def fake_worker(worker, *args):
        if worker == 0:
            raise RuntimeError("crash")
        return "provider_down"

    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", _thread_pool)
    monkeypatch.setattr(collect, "_worker_main", fake_worker)
    code = collect.main(["--workers", "2", "--data", str(data), "--log-dir", str(logs)])
    assert code == collect.EXIT_CODES["provider_down"]
    assert (logs / "STATUS").read_text().strip() == "provider_down"
    assert (logs / "worker_0.exit").read_text().strip() == "failed"


def test_main_refuses_a_queue_that_does_not_match_the_manifest(tmp_path, monkeypatch):
    data, logs = _data_dir(tmp_path), tmp_path / "logs"
    (data / "queue.jsonl").write_text((data / "queue.jsonl").read_text() + "\n")
    monkeypatch.setattr(collect, "_worker_main", lambda *a: pytest.fail("no worker may start"))
    with pytest.raises(RuntimeError, match="sha256"):
        collect.main(["--workers", "2", "--data", str(data), "--log-dir", str(logs)])
    assert (logs / "STATUS").read_text().strip() == "failed"


class _FlakyStartAdapter(FakeAdapter):
    instances: list = []

    def __init__(self, **kw):
        super().__init__()
        self.closes = 0
        _FlakyStartAdapter.instances.append(self)

    def reset(self, seed):
        self.resets += 1
        if self.resets == 1:
            raise RuntimeError("server did not come up within 15s")

    def close(self):
        self.closes += 1


def test_worker_main_retries_the_first_reset(tmp_path, monkeypatch):
    from cold_start.tasks import webarena as wa

    data, logs = _data_dir(tmp_path), tmp_path / "logs"
    _FlakyStartAdapter.instances.clear()
    monkeypatch.setattr(wa, "WebArenaInfinityAdapter", _FlakyStartAdapter)
    monkeypatch.setattr(collect, "N_BANK_TASKS", len(TASKS))
    monkeypatch.setattr(collect.time, "sleep", lambda s: None)
    out = collect._worker_main(0, 1, str(data), str(logs), 100.0, False, (0, 1))
    adapter = _FlakyStartAdapter.instances[-1]
    assert out == "done" and adapter.resets == 2
    assert len(adapter.calls) == 2


# C1(a): the adapter drains the server's pipes, so a chatty server never blocks on a full pipe.


def _fake_webarena(monkeypatch, tmp_path, start_server):
    import types

    from cold_start.tasks import webarena as wa

    class _Agent:
        def __init__(self, *a, **kw):
            pass

        async def setup(self, url):
            return None

        async def teardown(self):
            return None

    def stop_server(proc):
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)

    server = types.SimpleNamespace(start_server=start_server, wait_for_server=lambda port, timeout: True,
                                   stop_server=stop_server)
    tasks = types.SimpleNamespace(load_tasks=lambda app, suite: [{"id": "t0", "instruction": "x", "verify": "v"}])
    monkeypatch.setattr(wa, "_import_webarena", lambda: (None, server, tasks))
    monkeypatch.setattr(wa, "_webarena_root", lambda: tmp_path)
    monkeypatch.setattr(wa, "_get_armed_agent_cls", lambda: _Agent)
    monkeypatch.setattr(wa, "_build_llm", lambda *a, **kw: None)
    return wa.WebArenaInfinityAdapter(artifacts_dir=str(tmp_path / "art"), port=8765,
                                      axes_path=str(ROOT / "configs" / "axes.yaml"),
                                      template_path=str(ROOT / "configs" / "template.jinja"))


def test_reset_drains_a_chatty_servers_pipes(tmp_path, monkeypatch):
    import subprocess

    # ~1 MB on each pipe: far past the ~64 KB a pipe buffers, so an undrained child blocks forever.
    script = ("import sys\n"
              "for i in range(20000):\n"
              "    sys.stdout.write('GET /api/emails %d\\n' % i); sys.stderr.write('127.0.0.1 - - log line %d\\n' % i)\n"
              "sys.stdout.flush(); sys.stderr.flush()\n")
    procs = []

    def start_server(app, port):
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append(proc)
        return proc

    adapter = _fake_webarena(monkeypatch, tmp_path, start_server)
    adapter.reset(seed=0)
    try:
        assert procs[0].wait(timeout=30) == 0  # finished writing: nothing blocked on a full pipe
    finally:
        adapter.close()
    log_text = (tmp_path / "art" / "server_8765.log").read_text()
    assert "GET /api/emails 19999" in log_text and "log line 19999" in log_text
    assert adapter._drain_threads == [] and adapter._server_log is None


def test_reset_tolerates_a_server_without_pipes(tmp_path, monkeypatch):
    class _NoPipes:
        stdout = None
        stderr = None

        def poll(self):
            return 0

    adapter = _fake_webarena(monkeypatch, tmp_path, lambda app, port: _NoPipes())
    adapter.reset(seed=0)
    assert adapter._drain_threads == []
    adapter.close()


# ---- amendment 1: archiving pilot records that fell outside the new 30-task queue -------------


def _archive_queue():
    return [mp.QueueItem(index=0, pool="F", arm_id="F_00", task_id="t0", replicate=0, pilot=False),
            mp.QueueItem(index=1, pool="F", arm_id="F_00", task_id="t1", replicate=0, pilot=False)]


def test_archive_out_of_queue_keeps_in_queue_and_archives_the_rest_verbatim(tmp_path):
    path = tmp_path / "worker_0.jsonl"
    in_line = _line("t0")
    out_line1 = _line("t2")  # F_00/t2: not in the new queue
    out_line2 = _line("t3", arm="F_01")  # F_01 is not in the new queue at all
    path.write_text(in_line + "\n" + out_line1 + "\n" + out_line2 + "\n")

    counts = collect.archive_out_of_queue(tmp_path, _archive_queue())

    assert counts == {"kept": 1, "archived": 2, "files": 1}
    assert path.read_text() == in_line + "\n"
    archive_path = tmp_path / "archive" / "worker_0.pre_amendment_1.jsonl"
    assert archive_path.read_text() == out_line1 + "\n" + out_line2 + "\n"


def test_archive_out_of_queue_drops_a_torn_trailing_fragment_from_both(tmp_path):
    path = tmp_path / "worker_0.jsonl"
    good = _line("t0")
    torn = '{"schema": "empirical_pool/1", "pool": "F", "arm_id": "F_0'
    path.write_text(good + "\n" + torn)

    counts = collect.archive_out_of_queue(tmp_path, _archive_queue())

    assert counts == {"kept": 1, "archived": 0, "files": 1}
    assert path.read_text() == good + "\n"
    assert not (tmp_path / "archive" / "worker_0.pre_amendment_1.jsonl").exists()


def test_archive_out_of_queue_refuses_when_the_lock_names_a_live_process_group(tmp_path, monkeypatch):
    """Fix round 1, item 3: liveness is a process-GROUP probe (`os.killpg`), not a single-pid
    check -- a worker in the collector's group can outlive its leader and still be appending
    to the very file this function is about to rewrite."""
    (tmp_path / "collect.lock").write_text("4242")
    monkeypatch.setattr(collect.os, "killpg", lambda pgid, sig: None)  # group alive
    with pytest.raises(RuntimeError, match="collector"):
        collect.archive_out_of_queue(tmp_path, _archive_queue())


def test_archive_out_of_queue_refuses_when_a_dead_leaders_group_still_has_a_live_member(tmp_path, monkeypatch):
    (tmp_path / "collect.lock").write_text("4242")
    monkeypatch.setattr(collect.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(RuntimeError, match="collector"):
        collect.archive_out_of_queue(tmp_path, _archive_queue())


def test_archive_out_of_queue_tolerates_a_stale_lock(tmp_path, monkeypatch):
    (tmp_path / "collect.lock").write_text("999999")

    def _dead(pgid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(collect.os, "killpg", _dead)
    (tmp_path / "worker_0.jsonl").write_text(_line("t0") + "\n")
    counts = collect.archive_out_of_queue(tmp_path, _archive_queue())
    assert counts["files"] == 1


def test_archive_out_of_queue_holds_the_lock_for_its_duration_then_releases_it(tmp_path, monkeypatch):
    """Fix round 1, item 3: the archive acquires `collect.lock` itself (so a fresh collector
    can't start mid-rewrite) and always releases it when done, even with no prior lock file."""
    seen_locked = {}
    real_acquire = collect._acquire_lock

    def spy_acquire(lock_path):
        real_acquire(lock_path)
        seen_locked["during"] = lock_path.exists() and lock_path.read_text().strip() == str(os.getpid())

    monkeypatch.setattr(collect, "_acquire_lock", spy_acquire)
    (tmp_path / "worker_0.jsonl").write_text(_line("t0") + "\n")

    collect.archive_out_of_queue(tmp_path, _archive_queue())

    assert seen_locked["during"] is True
    assert not (tmp_path / "collect.lock").exists()  # released afterward


def test_archive_out_of_queue_counts_across_multiple_worker_files(tmp_path):
    (tmp_path / "worker_0.jsonl").write_text(_line("t0") + "\n" + _line("t5") + "\n")
    (tmp_path / "worker_1.jsonl").write_text(_line("t1") + "\n")
    counts = collect.archive_out_of_queue(tmp_path, _archive_queue())
    assert counts == {"kept": 2, "archived": 1, "files": 2}


def test_archive_out_of_queue_is_idempotent(tmp_path):
    path = tmp_path / "worker_0.jsonl"
    path.write_text(_line("t0") + "\n" + _line("t2") + "\n")

    first = collect.archive_out_of_queue(tmp_path, _archive_queue())
    assert first == {"kept": 1, "archived": 1, "files": 1}
    second = collect.archive_out_of_queue(tmp_path, _archive_queue())
    assert second == {"kept": 1, "archived": 0, "files": 1}

    archive_path = tmp_path / "archive" / "worker_0.pre_amendment_1.jsonl"
    assert archive_path.read_text() == _line("t2") + "\n"  # not duplicated by the second run


def _requeued_data_dir(tmp_path):
    """A `_data_dir` that already looks post-`make_pools.py --requeue` (has `n_tasks`), the
    only state `--archive-out-of-queue` may run against."""
    data = _data_dir(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["n_tasks"] = 30
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data


def test_main_archive_out_of_queue_prints_counts_writes_paused_and_launches_nothing(tmp_path, monkeypatch, capsys):
    data, logs = _requeued_data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    (logs / "worker_0.jsonl").write_text(_line("t0") + "\n" + _line("t9") + "\n")  # t9 is not in _data_dir's queue

    monkeypatch.setattr(collect, "_worker_main", lambda *a, **kw: pytest.fail("must not launch a worker"))
    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", lambda *a, **kw: pytest.fail("must not launch a pool"))

    code = collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])

    assert code == 0
    assert (logs / "STATUS").read_text().strip() == "paused"
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"kept": 1, "archived": 1, "files": 1}


# ---- fix round 1, item 5: --archive-out-of-queue enforces requeue -> archive order -------------


def test_main_archive_out_of_queue_refuses_without_a_requeued_manifest(tmp_path, monkeypatch):
    """`_data_dir`'s manifest (like a pre-amendment `manifest.json`) has no `n_tasks`: it
    hasn't been through `make_pools.py --requeue` yet, so archiving must refuse."""
    data, logs = _data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(collect, "archive_out_of_queue", lambda *a, **kw: pytest.fail("must not archive"))

    with pytest.raises(RuntimeError, match="n_tasks"):
        collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])
    assert not (logs / "STATUS").exists() or (logs / "STATUS").read_text().strip() != "paused"


def test_main_archive_out_of_queue_refuses_if_the_queue_does_not_match_the_manifest(tmp_path, monkeypatch):
    data, logs = _requeued_data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    with open(data / "queue.jsonl", "a") as fh:
        fh.write("\n")  # moves queue.jsonl's sha256 away from the manifest's frozen one
    monkeypatch.setattr(collect, "archive_out_of_queue", lambda *a, **kw: pytest.fail("must not archive"))

    with pytest.raises(RuntimeError, match="queue.jsonl sha256"):
        collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])


# ---- fix round 1, item 2: spent_usd must count archived cost too -------------------------------


def test_spent_usd_sums_archived_records_too(tmp_path):
    (tmp_path / "worker_0.jsonl").write_text(_line("t0", cost=0.02) + "\n" + _line("t1", cost=0.05) + "\n")
    collect.archive_out_of_queue(tmp_path, [mp.QueueItem(index=0, pool="F", arm_id="F_00", task_id="t0",
                                                         replicate=0, pilot=False)])
    # t1 (F_00) is now archived, no longer in worker_0.jsonl -- its $0.05 must still count toward
    # the real budget cap, not vanish just because the current queue doesn't include it any more.
    assert collect.spent_usd(tmp_path) == pytest.approx(0.07)


def test_spent_usd_tolerates_a_torn_trailing_line_in_an_archive_file(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    good = _line("t0", cost=0.03)
    torn = '{"schema": "empirical_pool/1", "pool": "F", "arm_id": "F_0'
    (archive_dir / "worker_0.pre_amendment_1.jsonl").write_text(good + "\n" + torn)
    assert collect.spent_usd(tmp_path) == pytest.approx(0.03)


def test_spent_usd_with_no_archive_dir_is_unaffected(tmp_path):
    (tmp_path / "worker_0.jsonl").write_text(_line("t0", cost=0.04) + "\n")
    assert collect.spent_usd(tmp_path) == pytest.approx(0.04)


# ---- fix round 2: `_pgid_is_alive`'s invariant (a lock's pid is also its pgid) must hold ------
# even when collect.py is run directly, not just when the watchdog launches it with
# start_new_session=True. Ruling: main() makes itself its own process-group leader, in every
# mode, before any lock acquisition.


def test_become_process_group_leader_setpgids_when_not_a_leader(monkeypatch):
    monkeypatch.setattr(collect.os, "getpid", lambda: 111)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 222)  # not our own pid: not a leader
    calls = []
    monkeypatch.setattr(collect.os, "setpgid", lambda pid, pgrp: calls.append((pid, pgrp)))

    collect._become_process_group_leader()

    assert calls == [(0, 0)]


def test_become_process_group_leader_is_a_noop_when_already_a_leader(monkeypatch):
    monkeypatch.setattr(collect.os, "getpid", lambda: 111)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 111)  # already our own pid: a leader
    monkeypatch.setattr(collect.os, "setpgid", lambda pid, pgrp: pytest.fail("must not be called"))

    collect._become_process_group_leader()  # must not raise, must not call setpgid


@pytest.mark.parametrize("exc", [PermissionError(), OSError()])
def test_become_process_group_leader_tolerates_a_setpgid_failure(monkeypatch, exc):
    """A session leader cannot change its own process group (`setpgid` raises EPERM); its pid
    already equals its pgid by definition, so this is harmless -- log and move on."""
    monkeypatch.setattr(collect.os, "getpid", lambda: 111)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 222)

    def boom(pid, pgrp):
        raise exc

    monkeypatch.setattr(collect.os, "setpgid", boom)

    collect._become_process_group_leader()  # must not raise


def test_main_calls_setpgid_when_not_a_process_group_leader(tmp_path, monkeypatch):
    data, logs = _requeued_data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(collect.os, "getpid", lambda: 111)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 222)
    calls = []
    monkeypatch.setattr(collect.os, "setpgid", lambda pid, pgrp: calls.append((pid, pgrp)))
    monkeypatch.setattr(collect, "archive_out_of_queue", lambda *a, **kw: {"kept": 0, "archived": 0, "files": 0})

    code = collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])

    assert code == 0
    assert calls == [(0, 0)]


def test_main_skips_setpgid_when_already_a_process_group_leader(tmp_path, monkeypatch):
    data, logs = _requeued_data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(collect.os, "getpid", lambda: 111)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 111)
    monkeypatch.setattr(collect.os, "setpgid", lambda pid, pgrp: pytest.fail("must not be called"))
    monkeypatch.setattr(collect, "archive_out_of_queue", lambda *a, **kw: {"kept": 0, "archived": 0, "files": 0})

    code = collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])

    assert code == 0


@pytest.mark.parametrize("exc", [PermissionError(), OSError()])
def test_main_tolerates_a_setpgid_failure(tmp_path, monkeypatch, exc):
    data, logs = _requeued_data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(collect.os, "getpid", lambda: 111)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 222)

    def boom(pid, pgrp):
        raise exc

    monkeypatch.setattr(collect.os, "setpgid", boom)
    monkeypatch.setattr(collect, "archive_out_of_queue", lambda *a, **kw: {"kept": 0, "archived": 0, "files": 0})

    code = collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])

    assert code == 0


def test_main_becomes_leader_before_any_lock_acquisition_on_the_archive_path(tmp_path, monkeypatch):
    data, logs = _requeued_data_dir(tmp_path), tmp_path / "logs"
    logs.mkdir()
    (logs / "worker_0.jsonl").write_text(_line("t0") + "\n")
    monkeypatch.setattr(collect.os, "getpid", lambda: 4242)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 1)

    events: list[str] = []
    monkeypatch.setattr(collect.os, "setpgid", lambda pid, pgrp: events.append("setpgid"))
    real_acquire = collect._acquire_lock

    def spy_acquire(lock_path):
        events.append("acquire_lock")
        return real_acquire(lock_path)

    monkeypatch.setattr(collect, "_acquire_lock", spy_acquire)

    code = collect.main(["--archive-out-of-queue", "--data", str(data), "--log-dir", str(logs)])

    assert code == 0
    assert events == ["setpgid", "acquire_lock"]


def test_main_becomes_leader_before_any_lock_acquisition_on_the_normal_path(tmp_path, monkeypatch):
    data, logs = _data_dir(tmp_path), tmp_path / "logs"
    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", _thread_pool)
    monkeypatch.setattr(collect, "_worker_main", lambda *a: "done")
    monkeypatch.setattr(collect.os, "getpid", lambda: 4242)
    monkeypatch.setattr(collect.os, "getpgid", lambda pid: 1)

    events: list[str] = []
    monkeypatch.setattr(collect.os, "setpgid", lambda pid, pgrp: events.append("setpgid"))
    real_acquire = collect._acquire_lock

    def spy_acquire(lock_path):
        events.append("acquire_lock")
        return real_acquire(lock_path)

    monkeypatch.setattr(collect, "_acquire_lock", spy_acquire)

    code = collect.main(["--workers", "2", "--data", str(data), "--log-dir", str(logs)])

    assert code == collect.EXIT_CODES["done"]
    assert events == ["setpgid", "acquire_lock"]


# ---- --pools: work only the chosen pools' queue items (composes with --through-index) -------------

_GL_TASKS = ["task_e1", "task_m1", "task_h1"]
_GL_ARMS = [("GLG", "GLG_00"), ("GLK", "GLK_00"), ("anchor", "GL_anchor_baseline")]


def _het_data(tmp_path):
    """What make_het_pools writes for the gitlab profile: pools/, gitlab/queue.jsonl, manifest.json
    (GLG and GLK items interleave in the queue: index 0..2 GLG, 3..5 GLK, 6..8 anchor)."""
    data = tmp_path / "heterogeneity"
    vec = mp.PromptVector(**mp.BASELINE_VECTOR)
    mp.write_pool(data / "pools" / "GLG.yaml", "GLG", [mp.grid_arm("GLG_00", vec)], mp.GRID_TEMPLATE, mp.AXES_PATH,
                  meta={})
    mp.write_pool(data / "pools" / "GLK.yaml", "GLK", [mp.freeform_arm("GLK_00", "Open the board first. " * 10)],
                  mp.FREEFORM_TEMPLATE, mp.AXES_PATH, meta={})
    mp.write_pool(data / "pools" / "anchors_gitlab.yaml", "anchor", [mp.grid_arm("GL_anchor_baseline", vec)],
                  ROOT / "configs" / "template_gitlab.jinja", mp.AXES_PATH, meta={})
    queue, k = [], 0
    for pool, arm in _GL_ARMS:
        for task in _GL_TASKS:
            queue.append(mp.QueueItem(index=k, pool=pool, arm_id=arm, task_id=task, replicate=0, pilot=False))
            k += 1
    (data / "gitlab").mkdir(parents=True)
    mp.write_queue(data / "gitlab" / "queue.jsonl", queue)
    files = ["pools/GLG.yaml", "pools/GLK.yaml", "pools/anchors_gitlab.yaml", "gitlab/queue.jsonl"]
    (data / "manifest.json").write_text(json.dumps({"files": {f: mp.file_sha256(data / f) for f in files}}))
    return data


def test_partition_remaining_keeps_only_the_chosen_pools_and_composes_with_the_through_index(tmp_path):
    pools = ("GLG", "GLK")
    q = [mp.QueueItem(index=i, pool=pools[i % 2], arm_id=f"{pools[i % 2]}_00", task_id=f"task_e{i}", replicate=0,
                      pilot=False) for i in range(10)]
    assert collect.partition_remaining(q, tmp_path, 2, pilot=False, pools=("GLK",)) == [(1, 5, 9), (3, 7)]
    assert collect.partition_remaining(q, tmp_path, 2, pilot=False, pools=("GLK",), through_index=6) == [(1, 5), (3,)]
    assert collect.partition_remaining(q, tmp_path, 2, pilot=False, pools=("GLG", "GLK")) == \
        collect.partition_remaining(q, tmp_path, 2, pilot=False) == [(0, 2, 4, 6, 8), (1, 3, 5, 7, 9)]
    (tmp_path / "worker_0.jsonl").write_text(_line("task_e1", arm="GLK_00") + "\n")  # item 1 is terminal
    assert collect.partition_remaining(q, tmp_path, 2, pilot=False, pools=("GLK",)) == [(3, 7), (5, 9)]


def test_parse_args_pools_rejects_an_unknown_pool_and_prereg9(tmp_path):
    data = _het_data(tmp_path)
    base = ["--profile", "gitlab", "--budget", "1", "--data", str(data)]
    with pytest.raises(SystemExit):
        collect.parse_args(base + ["--pools", "GLX"])  # not a pool in gitlab/queue.jsonl
    with pytest.raises(SystemExit):
        collect.parse_args(base + ["--pools", "GLK,GLX"])
    with pytest.raises(SystemExit):
        collect.parse_args(base + ["--pools", ","])
    with pytest.raises(SystemExit):
        collect.parse_args(["--pools", "F"])  # prereg9 never stages
    assert collect.parse_args(base + ["--pools", "GLK"]).pools == ("GLK",)
    assert collect.parse_args(base + ["--pools", "GLK, GLG"]).pools == ("GLG", "GLK")  # sorted, whitespace-tolerant
    assert collect.parse_args(base).pools is None
    assert collect.parse_args([]).pools is None  # prereg9 unchanged


def test_main_with_pools_runs_only_those_items_and_writes_status_through(tmp_path, monkeypatch):
    data, logs = _het_data(tmp_path), tmp_path / "logs"
    monkeypatch.setattr(collect, "HET_LOCK_PATH", tmp_path / "het.lock")
    seen = {}

    def fake_worker(worker, n_workers, data_dir, log_dir, budget, pilot, assigned, profile="MISSING"):
        seen[worker] = (profile, assigned)
        return "done"

    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", _thread_pool)
    monkeypatch.setattr(collect, "_worker_main", fake_worker)
    base = ["--profile", "gitlab", "--workers", "2", "--budget", "5", "--data", str(data), "--log-dir", str(logs)]
    code = collect.main(base + ["--pools", "GLK"])
    assert code == collect.EXIT_CODES["through"] == 0
    assert (logs / "STATUS").read_text().strip() == "through"  # the queue is not exhausted: never `done`
    assert seen == {0: ("gitlab", (3, 5)), 1: ("gitlab", (4,))}  # GLK is queue items 3..5
    seen.clear()
    logs2 = tmp_path / "logs2"
    base2 = [*base[:-1], str(logs2)]
    assert collect.main(base2 + ["--pools", "GLK", "--through-index", "4"]) == 0
    assert (logs2 / "STATUS").read_text().strip() == "through"
    assert seen == {0: ("gitlab", (3,)), 1: ("gitlab", (4,))}
    assert not (tmp_path / "het.lock").exists()


def test_a_relaunch_with_another_stage_refuses_before_touching_status(tmp_path, monkeypatch):
    data, logs = _het_data(tmp_path), tmp_path / "logs"
    monkeypatch.setattr(collect, "HET_LOCK_PATH", tmp_path / "het.lock")
    seen = {}

    def fake_worker(worker, n_workers, data_dir, log_dir, budget, pilot, assigned, profile="MISSING"):
        seen[worker] = assigned
        return "done"

    monkeypatch.setattr(collect.cf, "ProcessPoolExecutor", _thread_pool)
    monkeypatch.setattr(collect, "_worker_main", fake_worker)
    base = ["--profile", "gitlab", "--workers", "2", "--budget", "5", "--data", str(data), "--log-dir", str(logs)]
    assert collect.main(base + ["--pools", "GLK", "--through-index", "5"]) == 0
    assert json.loads((logs / collect.STAGE_FILE).read_text()) == {"pools": ["GLK"], "through_index": 5, "pilot": False}
    assert collect.main(base + ["--pools", "GLK", "--through-index", "5"]) == 0  # the same stage resumes
    for other in (["--through-index", "5"], [], ["--pools", "GLG"], ["--pools", "GLK"]):
        seen.clear()
        (logs / "STATUS").write_text("sentinel\n")
        assert collect.main(base + other) == collect.EXIT_STAGE_MISMATCH
        assert seen == {} and (logs / "STATUS").read_text() == "sentinel\n" and not (logs / "collect.lock").exists()
