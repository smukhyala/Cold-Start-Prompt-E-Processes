"""Main trial driver.

Flow (per trial):
    for t in 1..T:
        task = env.sample_task(t)
        arm  = arms[policy.next_arm(t, state)]
        res  = env.run_arm(arm, task, runner, max_steps)
        r    = reward_fn(task, res)
        per_arm_eprocess[arm.id].update(r)
        policy.update(arm.id, r, info)
        logger.log_step(...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cold_start.config import RunConfig
from cold_start.inference.confidence_sequence import ConfidenceSequence
from cold_start.inference.gate import reject_null
from cold_start.inference.global_null import GlobalNullEProcess
from cold_start.logging_setup import JSONLWriter, run_id, setup_console_logger, utc_now_iso
from cold_start.models.base import ModelClient
from cold_start.policies.base import PolicyState, SamplingPolicy
from cold_start.policies.warmstart import WarmStart
from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms
from cold_start.prompts.template import render_prompt
from cold_start.registry import get_registered
from cold_start.rewards.base import RewardFn
from cold_start.rng import make_rng, spawn
from cold_start.tasks.base import AgentRunner, EnvironmentAdapter
from cold_start.types import Arm, PerArmState


def run_trial(
    cfg: RunConfig,
    trial_idx: int = 0,
    log_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    start_at: int = 1,
) -> Path:
    logger = setup_console_logger(cfg.logging.level)
    rng = make_rng(cfg.trial.seed + trial_idx)

    axes = load_axes(cfg.prompts.axes)
    arms = load_arms(cfg.prompts.arms, axes)

    env = _build_env(cfg)
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    reward_fn = _build_reward(cfg)
    policy = _build_policy(cfg, rng, len(arms))
    model_client = _build_model(cfg)
    runner: AgentRunner = _make_runner(model_client, cfg.trial.max_agent_steps)

    # Per-arm e-processes and confidence sequences.
    EProcCls = get_registered("eprocess", cfg.inference.per_arm.type)
    per_arm_e = {a.arm_id: EProcCls(**cfg.inference.per_arm.params) for a in arms}
    per_arm_cs = {
        a.arm_id: ConfidenceSequence(
            alpha=float(cfg.inference.confidence_sequence.get("alpha", 0.05)),
            grid_size=int(cfg.inference.confidence_sequence.get("grid_size", 64)),
        )
        for a in arms
    }
    global_e = GlobalNullEProcess(**cfg.inference.global_.params)
    alpha = float(cfg.inference.confidence_sequence.get("alpha", 0.05))

    # Render and cache system prompts per arm (deterministic per vector).
    system_prompts = {a.arm_id: render_prompt(a.vector, axes, cfg.prompts.template) for a in arms}

    state = PolicyState(arm_ids=[a.arm_id for a in arms])
    per_arm_summary = {a.arm_id: PerArmState() for a in arms}

    out_dir = Path(log_dir) if log_dir else Path(cfg.trial.output_dir)
    if resume_from is not None:
        # Inherit the original run_id so the UI and analysis group the
        # continuation with its origin by shared base_id.
        origin = _read_first_record(Path(resume_from))
        base_rid = str(origin.get("run_id", "")) if origin else ""
        if not base_rid:
            base_rid = run_id(cfg.trial.name, trial_idx)
        # Strip any prior _resume_from_N suffix so chained resumes still merge.
        base_rid = base_rid.split("_resume_from_")[0]
        rid = f"{base_rid}_resume_from_{start_at}"
    else:
        rid = run_id(cfg.trial.name, trial_idx)
    log_path = out_dir / f"{rid}.jsonl"
    cfg_hash = cfg.hash()

    arms_by_id: dict[str, Arm] = {a.arm_id: a for a in arms}

    if resume_from is not None:
        resume_path = Path(resume_from)
        replayed = _replay_state(
            resume_path,
            start_at=start_at,
            arms_by_id=arms_by_id,
            per_arm_e=per_arm_e,
            per_arm_cs=per_arm_cs,
            per_arm_summary=per_arm_summary,
            state=state,
            policy=policy,
        )
        logger.info(
            f"resumed from {resume_path.name}: replayed {replayed} records, "
            f"starting at t={start_at}"
        )

    logger.info(
        f"trial {rid}: T={cfg.trial.num_tasks}, arms={len(arms)}, "
        f"policy={cfg.policy.type}, start_at={start_at}"
    )

    with JSONLWriter(log_path) as writer:
        for t in range(start_at, cfg.trial.num_tasks + 1):
            task = env.sample_task(t)
            arm_id = policy.next_arm(t, state)
            arm = arms_by_id[arm_id]

            result = env.run_arm(
                arm=arm,
                task=task,
                runner=runner,
                max_steps=cfg.trial.max_agent_steps,
            )
            reward = reward_fn(task, result)

            log_e = per_arm_e[arm_id].update(reward)
            cs_lo, cs_hi = per_arm_cs[arm_id].update(reward)
            # One-sided upward-betting log-wealth, used by best-arm UCB policies.
            log_e_up = getattr(per_arm_e[arm_id], "log_e_upper", log_e)

            per_arm_summary[arm_id].pulls += 1
            per_arm_summary[arm_id].successes += int(reward >= 0.5)
            per_arm_summary[arm_id].log_e = log_e
            per_arm_summary[arm_id].cs_lo = cs_lo
            per_arm_summary[arm_id].cs_hi = cs_hi

            state.record(arm_id, reward, log_e, log_e_upper=log_e_up)
            policy.update(arm_id, reward, info={"steps": result.steps})

            global_log_e = global_e.log_e(per_arm_e.values())

            record = {
                "schema_version": "1.0",
                "run_id": rid,
                "t": t,
                "timestamp_utc": utc_now_iso(),
                "task_id": task.task_id,
                "task_meta": task.metadata,
                "arm_id": arm_id,
                "prompt_vector": arm.vector.as_dict(),
                "prompt_id": arm.vector.id(),
                "success": int(result.success),
                "reward": reward,
                "steps": result.steps,
                "wallclock_s": result.wallclock_s,
                "tokens": result.tokens,
                "policy": {"type": cfg.policy.type, "scores": policy.scores},
                "per_arm_state": {aid: per_arm_summary[aid].to_record() for aid in state.arm_ids},
                "global_e": {
                    "log_e": global_log_e,
                    "rejected": reject_null(global_log_e, alpha),
                },
                "inference": {
                    "alpha": alpha,
                    "m0": cfg.inference.per_arm.params.get("m0", 0.5),
                },
                "seed": cfg.trial.seed + trial_idx,
                "config_hash": cfg_hash,
            }
            writer.write(record)

            if t % max(cfg.trial.num_tasks // 10, 1) == 0:
                logger.info(
                    f"t={t}/{cfg.trial.num_tasks} arm={arm_id} r={reward:.0f} "
                    f"log_e(arm)={log_e:.3f} log_e(global)={global_log_e:.3f}"
                )

    env.close()
    logger.info(f"trial {rid} complete; log={log_path}")
    return log_path


def _build_env(cfg: RunConfig) -> EnvironmentAdapter:
    cls = get_registered("task_source", cfg.task_source.type)
    return cls(**cfg.task_source.params)


def _build_reward(cfg: RunConfig) -> RewardFn:
    cls = get_registered("reward", cfg.reward.type)
    return cls(**cfg.reward.params)


def _build_policy(cfg: RunConfig, rng: np.random.Generator, n_arms: int) -> SamplingPolicy:
    cls = get_registered("policy", cfg.policy.type)
    sub = spawn(rng, "policy")
    inner = cls(rng=sub, **cfg.policy.params)
    if cfg.policy.warmstart:
        min_pulls = int(cfg.policy.warmstart.get("min_pulls_per_arm", 0))
        if min_pulls > 0:
            return WarmStart(inner=inner, min_pulls_per_arm=min_pulls, rng=spawn(rng, "warmstart"))
    return inner


def _build_model(cfg: RunConfig) -> ModelClient:
    cls = get_registered("model", cfg.model.type)
    params: dict[str, Any] = {
        "id": cfg.model.id,
        "max_tokens": cfg.model.max_tokens,
        "temperature": cfg.model.temperature,
        "prompt_cache": cfg.model.prompt_cache,
        "params": cfg.model.params,
    }
    # Mock / non-anthropic clients typically don't take these fields.
    if cfg.model.type != "anthropic":
        params = cfg.model.params or {}
    return cls(**params)


def _read_first_record(p: Path) -> dict[str, Any] | None:
    with p.open() as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def _replay_state(
    resume_path: Path,
    start_at: int,
    arms_by_id: dict[str, Arm],
    per_arm_e: dict[str, Any],
    per_arm_cs: dict[str, Any],
    per_arm_summary: dict[str, PerArmState],
    state: PolicyState,
    policy: SamplingPolicy,
) -> int:
    """Replay prior (arm_id, reward) tuples into per-arm e-processes, confidence
    sequences, and policy state. Only records with t < start_at are consumed.

    The policy's RNG is NOT advanced to match the original run — resume drops
    the stream of arm choices that *would have been* made. For uniform or
    memoryless policies this is harmless; for policies with path-dependent
    sampling (Thompson, SPRUCE with state) the resumed trajectory diverges.
    """
    count = 0
    for line in resume_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        t = int(rec["t"])
        if t >= start_at:
            continue
        arm_id = rec["arm_id"]
        if arm_id not in arms_by_id:
            raise ValueError(
                f"resume record t={t} references unknown arm {arm_id!r}; "
                f"arms.yaml may have changed since the original run"
            )
        reward = float(rec["reward"])
        log_e = per_arm_e[arm_id].update(reward)
        cs_lo, cs_hi = per_arm_cs[arm_id].update(reward)
        log_e_up = getattr(per_arm_e[arm_id], "log_e_upper", log_e)

        per_arm_summary[arm_id].pulls += 1
        per_arm_summary[arm_id].successes += int(reward >= 0.5)
        per_arm_summary[arm_id].log_e = log_e
        per_arm_summary[arm_id].cs_lo = cs_lo
        per_arm_summary[arm_id].cs_hi = cs_hi

        state.record(arm_id, reward, log_e, log_e_upper=log_e_up)
        policy.update(arm_id, reward, info={"steps": int(rec.get("steps", 0))})
        count += 1
    return count


def _make_runner(client: ModelClient, max_steps: int) -> AgentRunner:
    from cold_start.runner.agent_loop import run_agent_once

    def runner(system_prompt: str, task: Any, max_steps: int = max_steps):
        return run_agent_once(
            client=client,
            system_prompt=system_prompt,
            task=task,
            tools=None,
            tool_handler=None,
            max_steps=max_steps,
        )

    return runner  # type: ignore[return-value]
