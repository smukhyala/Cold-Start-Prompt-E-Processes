"""WebArena-Infinity adapter.

Wires coldStartPrompts arms into the sibling webarena-infinity benchmark
(https://github.com/web-arena-x/webarena-infinity). The arm's rendered system
prompt is injected as `extend_system_message` on browser-use's Agent so the
default prompt (which teaches the JSON tool-call format) stays intact.

The `AgentRunner` argument to `run_arm` is intentionally unused: webarena-infinity
drives its own browser-use agent with its own LLM client. Token accounting
therefore differs from sanity-style envs; the e-process only consumes the
binary reward.

Setup
-----
1. Clone webarena-infinity as a sibling directory (or set WEBARENA_INFINITY_ROOT).
2. Install browser-use + Pillow in this env (see `pip install .[webarena]`).
3. Export ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from cold_start.prompts.axes import load_axes
from cold_start.prompts.template import render_prompt
from cold_start.registry import register
from cold_start.tasks.base import AgentRunner, EnvironmentAdapter
from cold_start.types import Arm, RunResult, Task

_DEFAULT_WEBARENA_ROOT = (
    Path(__file__).resolve().parents[4] / "webarena-infinity"
)

_PERSISTENT_LOOP: asyncio.AbstractEventLoop | None = None
_ARMED_AGENT_CLS: type | None = None


def _webarena_root() -> Path:
    root = Path(os.environ.get("WEBARENA_INFINITY_ROOT", _DEFAULT_WEBARENA_ROOT))
    if not (root / "evaluation").exists():
        raise RuntimeError(
            f"webarena-infinity not found at {root}. Clone it next to this repo "
            "or set WEBARENA_INFINITY_ROOT to the repo root."
        )
    return root


def _import_webarena() -> tuple[Any, Any, Any]:
    """Import webarena-infinity's evaluation modules.

    Its imports are not prefixed (e.g. `from agents import AgentResult`), so we
    insert its `evaluation/` dir on sys.path before importing.
    """
    eval_dir = str(_webarena_root() / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import agents  # type: ignore
    import server  # type: ignore
    import tasks as wa_tasks  # type: ignore
    return agents, server, wa_tasks


def _loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop for the adapter's lifetime.

    Creating fresh loops between tasks can leave browser-use's async state bound
    to a dead loop; keeping one loop avoids that whole class of RuntimeErrors.
    """
    global _PERSISTENT_LOOP
    if _PERSISTENT_LOOP is None or _PERSISTENT_LOOP.is_closed():
        _PERSISTENT_LOOP = asyncio.new_event_loop()
    return _PERSISTENT_LOOP


def _get_armed_agent_cls() -> type:
    """Build a BrowserUseAgent subclass that injects extend_system_message.

    Lazily defined so `import cold_start.tasks.webarena` doesn't pull in
    browser-use for users who only run the sanity env.
    """
    global _ARMED_AGENT_CLS
    if _ARMED_AGENT_CLS is not None:
        return _ARMED_AGENT_CLS

    agents_mod, _, _ = _import_webarena()
    BaseAgent = agents_mod.BrowserUseAgent

    class _ArmedBrowserUseAgent(BaseAgent):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._prompt_extension: str | None = None

        def set_prompt_extension(self, extension: str) -> None:
            self._prompt_extension = extension

        async def run(self, task: str, server_url: str, task_dir: Path):  # type: ignore[override]
            # Re-implements BrowserUseAgent.run() with extend_system_message injected.
            # Kept in lockstep with webarena-infinity/evaluation/agents.py:189.
            import shutil

            from browser_use import Agent

            instruction = (
                f"You are interacting with a web application at {server_url}. "
                f"Your task: {task}"
            )

            agent_kwargs: dict[str, Any] = dict(
                task=instruction,
                llm=self.llm,
                browser_session=self._session,
                use_vision=self.use_vision,
                save_conversation_path=str(task_dir / "conversations"),
                max_steps=self.max_steps,
            )
            if self._prompt_extension:
                agent_kwargs["extend_system_message"] = self._prompt_extension

            agent = Agent(**agent_kwargs)

            timed_out = False
            t0 = time.time()
            try:
                history = await asyncio.wait_for(agent.run(), timeout=self.timeout)
            except asyncio.TimeoutError:
                timed_out = True
                history = agent.history
            elapsed = time.time() - t0

            history.save_to_file(task_dir / "history.json")
            screenshots_dst = task_dir / "screenshots"
            for step_idx, path_str in enumerate(history.screenshot_paths()):
                if path_str and Path(path_str).exists():
                    screenshots_dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path_str, screenshots_dst / f"step_{step_idx}.png")

            if timed_out:
                raise asyncio.TimeoutError()

            return agents_mod.AgentResult(
                elapsed=round(elapsed, 1),
                steps=len(history.history),
                is_done=history.is_done(),
                final_result=history.final_result(),
                errors=history.errors(),
            )

    _ARMED_AGENT_CLS = _ArmedBrowserUseAgent
    return _ARMED_AGENT_CLS


def _build_llm(model_id: str) -> Any:
    from browser_use.llm.anthropic.chat import ChatAnthropic
    return ChatAnthropic(model=model_id)


@register("webarena", kind="task_source")
class WebArenaInfinityAdapter(EnvironmentAdapter):
    """Run coldStartPrompts arms against webarena-infinity apps."""

    def __init__(
        self,
        web_app: str = "apps/gmail",
        task_suite: str = "real-tasks",
        port: int = 8001,
        sample_mode: str = "cycle",
        use_vision: bool = False,
        headless: bool = True,
        timeout_s: int = 300,
        # Paper § 3.7: "we will use Claude Opus 4.7 Max for all testing".
        # Override per-experiment via task_source.params.llm_model in YAML.
        llm_model: str = "claude-opus-4-7",
        artifacts_dir: str = "logs/webarena",
        axes_path: str = "configs/axes.yaml",
        template_path: str = "configs/template.jinja",
    ) -> None:
        if sample_mode != "cycle":
            raise ValueError(
                f"unsupported sample_mode={sample_mode!r}; only 'cycle' is implemented"
            )
        self._web_app_rel = web_app
        self._task_suite = task_suite
        self._port = port
        self._sample_mode = sample_mode
        self._use_vision = use_vision
        self._headless = headless
        self._timeout_s = timeout_s
        self._llm_model = llm_model
        self._artifacts_dir = Path(artifacts_dir)
        self._axes = load_axes(axes_path)
        self._template_path = template_path

        self._prompt_cache: dict[str, str] = {}
        self._tasks: list[dict] | None = None
        self._server_proc = None
        self._agent: Any = None
        self._server_url = f"http://localhost:{port}"
        self._web_app_abs: str | None = None

    def reset(self, seed: int) -> None:
        _, server_mod, tasks_mod = _import_webarena()
        self._web_app_abs = str(_webarena_root() / self._web_app_rel)
        self._tasks = tasks_mod.load_tasks(self._web_app_abs, self._task_suite)

        self._server_proc = server_mod.start_server(self._web_app_abs, self._port)
        if not server_mod.wait_for_server(self._port, timeout=15):
            server_mod.stop_server(self._server_proc)
            raise RuntimeError(
                f"webarena server for {self._web_app_rel} on :{self._port} "
                "did not come up within 15s"
            )

        ArmedCls = _get_armed_agent_cls()
        self._agent = ArmedCls(
            _build_llm(self._llm_model),
            use_vision=self._use_vision,
            max_steps=50,
            timeout=self._timeout_s,
            headless=self._headless,
        )
        _loop().run_until_complete(self._agent.setup(self._server_url))

    def sample_task(self, t: int) -> Task:
        if self._tasks is None:
            raise RuntimeError("reset() must be called before sample_task()")
        idx = (t - 1) % len(self._tasks)
        raw = self._tasks[idx]
        return Task(
            task_id=raw["id"],
            payload={"instruction": raw["instruction"]},
            metadata={
                "difficulty": raw.get("difficulty", ""),
                "verify_path": raw["verify"],
                "web_app_dir": self._web_app_abs,
                "server_url": self._server_url,
                "bank_index": idx,
                "raw": raw,
            },
        )

    def run_arm(
        self,
        arm: Arm,
        task: Task,
        runner: AgentRunner,
        max_steps: int,
    ) -> RunResult:
        del runner  # browser agent drives its own LLM; text-only runner is unused
        _, _, tasks_mod = _import_webarena()

        extension = self._prompt_cache.get(arm.arm_id)
        if extension is None:
            extension = render_prompt(arm.vector, self._axes, self._template_path)
            self._prompt_cache[arm.arm_id] = extension

        assert self._agent is not None, "reset() must be called before run_arm()"
        self._agent.set_prompt_extension(extension)
        self._agent.max_steps = max_steps

        task_dir = self._artifacts_dir / arm.arm_id / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        raw_task = task.metadata["raw"]
        try:
            result_dict = _loop().run_until_complete(
                tasks_mod.run_task(
                    task=raw_task,
                    agent_runner=self._agent,
                    server_url=self._server_url,
                    web_app_dir=task.metadata["web_app_dir"],
                    task_dir=task_dir,
                )
            )
        except asyncio.TimeoutError:
            return RunResult(
                success=False,
                reward=0.0,
                steps=max_steps,
                wallclock_s=float(self._timeout_s),
                trace={
                    "env": "webarena",
                    "timed_out": True,
                    "task_dir": str(task_dir),
                },
                tokens={},
            )

        # webarena's verifier emits a binary `passed`; some forks add a graded
        # `partial_score` ∈ [0,1] (e.g. fraction of subgoals satisfied). Surface
        # it when present so ContinuousReward picks it up; otherwise fall back
        # to None and binary rewards continue to behave as before.
        raw_partial = result_dict.get("partial_score")
        partial_score = float(raw_partial) if raw_partial is not None else None

        return RunResult(
            success=bool(result_dict["passed"]),
            reward=float(bool(result_dict["passed"])),
            steps=int(result_dict.get("steps", 0)),
            wallclock_s=float(result_dict.get("elapsed", 0.0)),
            trace={
                "env": "webarena",
                "verifier_message": result_dict.get("verifier_message", ""),
                "final_result": result_dict.get("final_result") or "",
                "errors": (result_dict.get("errors") or [])[:5],
                "is_done": bool(result_dict.get("is_done", False)),
                "task_dir": str(task_dir),
            },
            tokens={},
            partial_score=partial_score,
        )

    def close(self) -> None:
        try:
            if self._agent is not None:
                _loop().run_until_complete(self._agent.teardown())
        finally:
            self._agent = None
        if self._server_proc is not None:
            _, server_mod, _ = _import_webarena()
            server_mod.stop_server(self._server_proc)
            self._server_proc = None
