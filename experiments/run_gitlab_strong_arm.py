#!/usr/bin/env python
# ruff: noqa: E402,I001
"""GitLab positive-control experiments for adaptive prompt allocation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cold_start.cli import _bootstrap  # noqa: F401
from cold_start.config import RunConfig
from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms
from cold_start.runner.orchestrator import run_trial
from cold_start.types import Arm, PromptVector


GENERIC_ARM_IDS = [
    "baseline",
    "planner",
    "cautious",
    "explorer",
    "balanced",
    "overthinker",
    "rapid",
    "verifier",
    "exploratory",
    "algorithmic",
    "junior_reactive",
    "domain_expert",
]

POLICIES: dict[str, dict[str, Any]] = {
    "uniform": {"type": "uniform", "params": {"mode": "round_robin"}},
    "epsilon_greedy": {"type": "epsilon_greedy", "params": {"epsilon": 0.1}},
    "ucb": {"type": "ucb", "params": {"exploration_c": 1.0, "tie_break": "first"}},
    "thompson": {"type": "thompson", "params": {"alpha": 1.0, "beta": 1.0}},
    "spruce": {"type": "spruce", "params": {"exploration_c": 1.0, "tie_break": "first"}},
}


@dataclass(frozen=True)
class RunSpec:
    mode: str
    policy: str
    replicate: int
    budget: int
    n_arms: int
    log_path: Path


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "paired":
        run_paired(args, out)
    elif args.mode == "allocation":
        run_allocation(args, out, args.arms_config)
    elif args.mode == "arm-scaling":
        run_arm_scaling(args, out)
    elif args.mode == "all":
        run_paired(args, out / "paired")
        run_allocation(args, out / "allocation", args.arms_config)
        run_arm_scaling(args, out / "arm_scaling")

    write_readme(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["paired", "allocation", "arm-scaling", "all"], required=True)
    p.add_argument("--task-family", default="gitlab", choices=["gitlab"])
    p.add_argument("--num-tasks", type=int, default=40)
    p.add_argument("--budgets", type=int, nargs="+", default=[60, 120, 240, 480])
    p.add_argument("--num-replicates", type=int, default=5)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--arms-config", default="configs/arms_gitlab_strong.yaml")
    p.add_argument("--axes", default="configs/axes.yaml")
    p.add_argument("--template", default="configs/template_gitlab.jinja")
    p.add_argument("--max-agent-steps", type=int, default=30)
    p.add_argument("--timeout-s", type=int, default=180)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--llm-model", default="gpt-5.4-mini")
    p.add_argument("--llm-reasoning-effort", default="low")
    p.add_argument("--use-vision", action="store_true")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--scaling-arm-counts", type=int, nargs="+", default=[12, 25, 50, 100])
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip WebArena execution and rebuild CSV summaries/plots from existing logs.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerun a job when a complete JSONL already exists for its name.",
    )
    return p.parse_args()


def run_paired(args: argparse.Namespace, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arms = load_arms(args.arms_config, load_axes(args.axes))
    log_specs: list[RunSpec] = []
    for arm in arms:
        name = f"gitlab_strong_paired_{arm.arm_id}"
        log_path = (
            find_existing_log(out / "logs", name, args.num_tasks)
            if args.skip_existing or args.summarize_only
            else None
        )
        if log_path is None and not args.summarize_only:
            one_arm_path = write_arm_catalog(out / "generated_configs" / f"arms_{arm.arm_id}.yaml", [arm])
            cfg = build_config(args, name, one_arm_path, "uniform", args.num_tasks, out / "logs")
            log_path = run_trial(cfg, trial_idx=0, log_dir=out / "logs")
        if log_path is not None:
            log_specs.append(RunSpec("paired", "paired", 0, args.num_tasks, len(arms), log_path))

    summarize_paired(out, log_specs)


def run_allocation(args: argparse.Namespace, out: Path, arms_config: str | Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paired_summary = read_paired_summary(out.parent / "paired")
    true_best = choose_true_best(paired_summary)
    log_specs: list[RunSpec] = []
    arms = load_arms(arms_config, load_axes(args.axes))
    for budget in args.budgets:
        for rep in range(args.num_replicates):
            for policy in POLICIES:
                name = f"gitlab_strong_allocation_{policy}_budget{budget}_rep{rep}"
                log_path = (
                    find_existing_log(out / "logs", name, budget)
                    if args.skip_existing or args.summarize_only
                    else None
                )
                if log_path is None and not args.summarize_only:
                    cfg = build_config(
                        args,
                        name,
                        arms_config,
                        policy,
                        budget,
                        out / "logs",
                        seed=args.seed + rep,
                    )
                    log_path = run_trial(cfg, trial_idx=0, log_dir=out / "logs")
                if log_path is not None:
                    log_specs.append(RunSpec("allocation", policy, rep, budget, len(arms), log_path))

    summarize_allocation(out, log_specs, paired_summary, true_best)


def run_arm_scaling(args: argparse.Namespace, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    all_specs: list[RunSpec] = []
    for n_arms in args.scaling_arm_counts:
        arms_path = out / "generated_configs" / f"arms_gitlab_{n_arms}.yaml"
        arms = generate_scaled_arms(args.arms_config, args.axes, n_arms)
        write_arm_catalog(arms_path, arms)

        paired_dir = out / f"paired_{n_arms}"
        if not args.summarize_only:
            paired_args = argparse.Namespace(**vars(args))
            paired_args.output_dir = paired_dir
            paired_args.num_tasks = args.num_tasks
            paired_args.arms_config = str(arms_path)
            run_paired(paired_args, paired_dir)
        paired_summary = read_paired_summary(paired_dir)
        true_best = choose_true_best(paired_summary)

        scale_dir = out / f"n_arms_{n_arms}"
        scale_dir.mkdir(parents=True, exist_ok=True)
        specs: list[RunSpec] = []
        for rep in range(args.num_replicates):
            for policy in POLICIES:
                budget = max(args.budgets)
                name = f"gitlab_strong_scaling_n{n_arms}_{policy}_budget{budget}_rep{rep}"
                log_path = (
                    find_existing_log(scale_dir / "logs", name, budget)
                    if args.skip_existing or args.summarize_only
                    else None
                )
                if log_path is None and not args.summarize_only:
                    cfg = build_config(
                        args,
                        name,
                        arms_path,
                        policy,
                        budget,
                        scale_dir / "logs",
                        seed=args.seed + rep,
                    )
                    log_path = run_trial(cfg, trial_idx=0, log_dir=scale_dir / "logs")
                if log_path is not None:
                    specs.append(RunSpec("arm_scaling", policy, rep, budget, n_arms, log_path))
        summarize_allocation(scale_dir, specs, paired_summary, true_best, n_arms=n_arms)
        all_specs.extend(specs)

    summarize_scaling(out)


def build_config(
    args: argparse.Namespace,
    name: str,
    arms_config: str | Path,
    policy: str,
    num_tasks: int,
    output_dir: Path,
    seed: int | None = None,
) -> RunConfig:
    web_app = {"gitlab": "apps/gitlab-plan-and-track"}[args.task_family]
    payload = {
        "trial": {
            "name": name,
            "seed": args.seed if seed is None else seed,
            "num_tasks": num_tasks,
            "num_trials": 1,
            "paired_streams": True,
            "max_agent_steps": args.max_agent_steps,
            "output_dir": str(output_dir),
        },
        "prompts": {
            "axes": args.axes,
            "template": args.template,
            "arms": str(arms_config),
        },
        "model": {
            "type": "mock",
            "id": "mock",
            "max_tokens": 64,
            "temperature": 0.0,
            "prompt_cache": False,
        },
        "task_source": {
            "type": "webarena",
            "params": {
                "web_app": web_app,
                "task_suite": "real-tasks",
                "port": args.port,
                "sample_mode": "cycle",
                "use_vision": bool(args.use_vision),
                "headless": not bool(args.headed),
                "timeout_s": args.timeout_s,
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
                "llm_reasoning_effort": args.llm_reasoning_effort,
                "artifacts_dir": str(output_dir / "webarena" / name),
                "axes_path": args.axes,
                "template_path": args.template,
            },
        },
        "reward": {"type": "binary"},
        "inference": {
            "per_arm": {"type": "upward_capital", "params": {"m0": 0.5}},
            "global": {"type": "global_null", "params": {"combine": "linear_mixture"}},
            "confidence_sequence": {"alpha": 0.05, "grid_size": 64},
        },
        "policy": POLICIES[policy],
        "logging": {"level": "INFO", "jsonl": True},
    }
    cfg_path = output_dir / "configs" / f"{name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return RunConfig.model_validate(payload)


def write_arm_catalog(path: Path, arms: list[Arm]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"arms": []}
    for arm in arms:
        entry = {
            "arm_id": arm.arm_id,
            "name": arm.name,
            "vector": arm.vector.as_dict(),
        }
        if arm.prompt_guidance:
            entry["prompt_guidance"] = arm.prompt_guidance
        payload["arms"].append(entry)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def find_existing_log(log_dir: Path, trial_name: str, expected_rows: int) -> Path | None:
    candidates = sorted(log_dir.glob(f"{trial_name}_trial0_*.jsonl"))
    for path in reversed(candidates):
        rows = sum(1 for line in path.read_text().splitlines() if line.strip())
        if rows >= expected_rows:
            return path
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_paired(out: Path, specs: list[RunSpec]) -> pd.DataFrame:
    result_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        records = read_jsonl(spec.log_path)
        if not records:
            continue
        arm_id = str(records[0]["arm_id"])
        rewards = np.array([float(r["reward"]) for r in records], dtype=float)
        runtimes = np.array([float(r.get("wallclock_s", 0.0)) for r in records], dtype=float)
        costs = np.array([token_cost(r.get("tokens", {})) for r in records], dtype=float)
        steps = np.array([float(r.get("steps", 0.0)) for r in records], dtype=float)
        for r in records:
            result_rows.append(
                {
                    "arm_id": arm_id,
                    "task": r.get("task_id"),
                    "timestep": r.get("t"),
                    "reward": r.get("reward"),
                    "success": r.get("success"),
                    "runtime": r.get("wallclock_s", 0.0),
                    "cost": token_cost(r.get("tokens", {})),
                    "steps": r.get("steps", 0),
                    "log_path": str(spec.log_path),
                }
            )
        successes = int(rewards.sum())
        n = int(len(rewards))
        rate = float(rewards.mean()) if n else 0.0
        ci_lo, ci_hi = bootstrap_ci(rewards)
        summary_rows.append(
            {
                "arm_id": arm_id,
                "tasks": n,
                "successes": successes,
                "success_rate": rate,
                "standard_error": math.sqrt(rate * (1.0 - rate) / n) if n else 0.0,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "total_runtime": float(runtimes.sum()),
                "total_token_cost": float(costs.sum()),
                "cost_per_success": float(costs.sum() / successes) if successes else math.inf,
                "average_runtime": float(runtimes.mean()) if n else 0.0,
                "average_interaction_steps": float(steps.mean()) if n else 0.0,
                "log_path": str(spec.log_path),
            }
        )
    result_columns = [
        "arm_id",
        "task",
        "timestep",
        "reward",
        "success",
        "runtime",
        "cost",
        "steps",
        "log_path",
    ]
    summary_columns = [
        "arm_id",
        "tasks",
        "successes",
        "success_rate",
        "standard_error",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "total_runtime",
        "total_token_cost",
        "cost_per_success",
        "average_runtime",
        "average_interaction_steps",
        "log_path",
    ]
    paired_results = pd.DataFrame(result_rows, columns=result_columns)
    paired_summary = pd.DataFrame(summary_rows, columns=summary_columns)
    if not paired_summary.empty:
        paired_summary = paired_summary.sort_values(
            ["success_rate", "successes", "arm_id"], ascending=[False, False, True]
        )
    paired_results.to_csv(out / "paired_results.csv", index=False)
    paired_summary.to_csv(out / "paired_summary.csv", index=False)
    if not paired_summary.empty:
        (out / "true_best_arm.txt").write_text(str(paired_summary.iloc[0]["arm_id"]) + "\n")
    write_paired_config_json(out, paired_summary, specs)
    return paired_summary


def summarize_allocation(
    out: Path,
    specs: list[RunSpec],
    paired_summary: pd.DataFrame,
    true_best: str,
    n_arms: int | None = None,
) -> None:
    rates = dict(zip(paired_summary["arm_id"], paired_summary["success_rate"], strict=False))
    raw_rows: list[dict[str, Any]] = []
    for spec in specs:
        records = read_jsonl(spec.log_path)
        counts: dict[str, int] = {}
        cumulative_successes = 0
        cumulative_regret = 0.0
        cumulative_runtime = 0.0
        cumulative_cost = 0.0
        for rec in records:
            t = int(rec["t"])
            arm_id = str(rec["arm_id"])
            reward = float(rec["reward"])
            counts[arm_id] = counts.get(arm_id, 0) + 1
            cumulative_successes += int(reward >= 0.5)
            inst_regret = float(rates.get(true_best, 0.0) - rates.get(arm_id, 0.0))
            cumulative_regret += inst_regret
            runtime = float(rec.get("wallclock_s", 0.0))
            cost = token_cost(rec.get("tokens", {}))
            cumulative_runtime += runtime
            cumulative_cost += cost
            entropy, entropy_norm = allocation_entropy(counts, spec.n_arms)
            current_most = current_most_pulled(counts)
            best_share = counts.get(true_best, 0) / t
            raw_rows.append(
                {
                    "policy": spec.policy,
                    "replicate": spec.replicate,
                    "budget": spec.budget,
                    "n_arms": n_arms or spec.n_arms,
                    "task": rec.get("task_id"),
                    "timestep": t,
                    "selected_arm": arm_id,
                    "reward": reward,
                    "cumulative_successes": cumulative_successes,
                    "cumulative_success_rate": cumulative_successes / t,
                    "instantaneous_regret": inst_regret,
                    "cumulative_regret": cumulative_regret,
                    "current_most_pulled_arm": current_most,
                    "true_best_arm": true_best,
                    "selected_true_best": arm_id == true_best,
                    "entropy": entropy,
                    "normalized_entropy": entropy_norm,
                    "best_arm_pull_share": best_share,
                    "runtime": runtime,
                    "cost": cost,
                    "cumulative_runtime": cumulative_runtime,
                    "cumulative_cost": cumulative_cost,
                    "global_log_e": float(rec.get("global_e", {}).get("log_e", 0.0)),
                    "log_path": str(spec.log_path),
                }
            )
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out / "raw_runs.csv", index=False)
    if raw.empty:
        return

    raw[["policy", "replicate", "budget", "timestep", "entropy", "normalized_entropy"]].to_csv(
        out / "entropy_timeseries.csv", index=False
    )
    raw[["policy", "replicate", "budget", "timestep", "best_arm_pull_share"]].to_csv(
        out / "best_arm_share_timeseries.csv", index=False
    )
    raw[["policy", "replicate", "budget", "timestep", "cumulative_regret"]].to_csv(
        out / "regret_timeseries.csv", index=False
    )

    final_rows: list[dict[str, Any]] = []
    group_cols = ["policy", "replicate", "budget"]
    for key, g in raw.groupby(group_cols, sort=False):
        policy, rep, budget = key
        final = g.sort_values("timestep").iloc[-1]
        identification_time = first_stable_identification_time(g, true_best)
        final_rows.append(
            {
                "policy": policy,
                "replicate": rep,
                "budget": budget,
                "n_arms": int(final["n_arms"]),
                "true_best_arm": true_best,
                "successes": int(final["cumulative_successes"]),
                "success_rate": float(final["cumulative_success_rate"]),
                "cumulative_regret": float(final["cumulative_regret"]),
                "final_entropy": float(final["entropy"]),
                "final_normalized_entropy": float(final["normalized_entropy"]),
                "best_arm_pull_share": float(final["best_arm_pull_share"]),
                "identified_true_best": identification_time is not None,
                "identification_time": identification_time,
                "total_runtime": float(final["cumulative_runtime"]),
                "total_token_cost": float(final["cumulative_cost"]),
                "cost_per_success": (
                    float(final["cumulative_cost"]) / int(final["cumulative_successes"])
                    if int(final["cumulative_successes"])
                    else math.inf
                ),
            }
        )
    finals = pd.DataFrame(final_rows)
    summary = (
        finals.groupby(["policy", "budget", "n_arms"], as_index=False)
        .agg(
            success_rate_mean=("success_rate", "mean"),
            success_rate_se=("success_rate", sem),
            regret_mean=("cumulative_regret", "mean"),
            regret_se=("cumulative_regret", sem),
            identification_probability=("identified_true_best", "mean"),
            final_entropy_mean=("final_entropy", "mean"),
            final_normalized_entropy_mean=("final_normalized_entropy", "mean"),
            best_arm_pull_share_mean=("best_arm_pull_share", "mean"),
            total_runtime_mean=("total_runtime", "mean"),
            total_token_cost_mean=("total_token_cost", "mean"),
            cost_per_success_mean=("cost_per_success", finite_mean),
        )
        .sort_values(["budget", "policy"])
    )
    finals.to_csv(out / "final_runs.csv", index=False)
    summary.to_csv(out / "summary_by_policy_budget.csv", index=False)
    write_config_json(out, paired_summary, true_best, specs)
    make_allocation_plots(out, raw, summary)


def summarize_scaling(out: Path) -> None:
    pieces = []
    for p in sorted(out.glob("n_arms_*/summary_by_policy_budget.csv")):
        pieces.append(pd.read_csv(p))
    if not pieces:
        return
    summary = pd.concat(pieces, ignore_index=True)
    summary.to_csv(out / "summary_by_policy_budget.csv", index=False)

    for filename in [
        "raw_runs.csv",
        "final_runs.csv",
        "entropy_timeseries.csv",
        "best_arm_share_timeseries.csv",
        "regret_timeseries.csv",
    ]:
        parts = [pd.read_csv(p) for p in sorted(out.glob(f"n_arms_*/{filename}"))]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / filename, index=False)
    payload = {
        "mode": "arm_scaling",
        "n_arms": sorted(int(v) for v in summary["n_arms"].dropna().unique()),
        "policies": sorted(str(v) for v in summary["policy"].dropna().unique()),
        "budgets": sorted(int(v) for v in summary["budget"].dropna().unique()),
    }
    (out / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_scaling_plots(out, summary)


def read_paired_summary(paired_dir: Path) -> pd.DataFrame:
    path = paired_dir / "paired_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"paired summary not found at {path}. Run --mode paired first, or use --mode all."
        )
    return pd.read_csv(path)


def choose_true_best(paired_summary: pd.DataFrame) -> str:
    if paired_summary.empty:
        raise ValueError("paired_summary.csv is empty; cannot define true_best_arm")
    ordered = paired_summary.sort_values(
        ["success_rate", "successes", "arm_id"], ascending=[False, False, True]
    )
    return str(ordered.iloc[0]["arm_id"])


def bootstrap_ci(rewards: np.ndarray, n_boot: int = 4000, seed: int = 20260629) -> tuple[float, float]:
    if len(rewards) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rewards), size=(n_boot, len(rewards)))
    means = rewards[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def token_cost(tokens: Any) -> float:
    if not isinstance(tokens, dict):
        return 0.0
    return float(tokens.get("cost_usd", 0.0) or 0.0)


def allocation_entropy(counts: dict[str, int], n_arms: int) -> tuple[float, float]:
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    probs = [v / total for v in counts.values() if v > 0]
    h = -sum(p * math.log(p) for p in probs)
    h_norm = h / math.log(n_arms) if n_arms > 1 else 0.0
    return float(h), float(h_norm)


def current_most_pulled(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    best = max(counts.values())
    ties = sorted(a for a, n in counts.items() if n == best)
    return ties[0] if len(ties) == 1 else None


def first_stable_identification_time(g: pd.DataFrame, true_best: str) -> int | None:
    ordered = g.sort_values("timestep")
    values = list(ordered["current_most_pulled_arm"])
    times = list(ordered["timestep"])
    for idx, arm in enumerate(values):
        if arm == true_best and all(v == true_best for v in values[idx:]):
            return int(times[idx])
    return None


def sem(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def finite_mean(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else math.inf


def generate_scaled_arms(base_config: str | Path, axes_path: str | Path, n_arms: int) -> list[Arm]:
    axes = load_axes(axes_path)
    base = load_arms(base_config, axes)
    if n_arms <= len(base):
        return base[:n_arms]
    used = {a.vector.as_tuple() for a in base}
    arms = list(base)
    idx = 0
    for planning in range(4):
        for verification in range(4):
            for agency in range(3):
                for expertise in range(4):
                    for fmt in range(4):
                        for goal in range(3):
                            vec = PromptVector(planning, verification, agency, expertise, fmt, goal)
                            if vec.as_tuple() in used:
                                continue
                            idx += 1
                            used.add(vec.as_tuple())
                            arms.append(
                                Arm(
                                    arm_id=f"synthetic_{idx:03d}",
                                    name=f"Synthetic {idx:03d}",
                                    vector=vec,
                                    prompt_guidance=(
                                        "Synthetic prompt-search variant generated by perturbing "
                                        "the behavioral dimensions of the seed catalog."
                                    ),
                                )
                            )
                            if len(arms) >= n_arms:
                                return arms
    raise ValueError(f"could not generate {n_arms} unique arms")


def write_config_json(
    out: Path,
    paired_summary: pd.DataFrame,
    true_best: str,
    specs: list[RunSpec],
) -> None:
    payload = {
        "true_best_arm": true_best,
        "paired_estimates": paired_summary.to_dict(orient="records"),
        "runs": [
            {
                "mode": s.mode,
                "policy": s.policy,
                "replicate": s.replicate,
                "budget": s.budget,
                "n_arms": s.n_arms,
                "log_path": str(s.log_path),
            }
            for s in specs
        ],
    }
    (out / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_paired_config_json(out: Path, paired_summary: pd.DataFrame, specs: list[RunSpec]) -> None:
    payload = {
        "true_best_arm": None if paired_summary.empty else str(paired_summary.iloc[0]["arm_id"]),
        "paired_estimates": paired_summary.to_dict(orient="records"),
        "runs": [{"log_path": str(s.log_path), "budget": s.budget} for s in specs],
    }
    (out / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def lineplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    outbase: Path,
    group: str = "policy",
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for key, g in df.groupby(group):
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", linewidth=2, label=str(key))
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def make_allocation_plots(out: Path, raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    lineplot(summary, "budget", "success_rate_mean", "Budget vs Success Rate", "Success rate", plot_dir / "budget_success_rate")
    lineplot(summary, "budget", "regret_mean", "Budget vs Regret", "Cumulative regret", plot_dir / "budget_regret")
    lineplot(
        summary,
        "budget",
        "identification_probability",
        "Budget vs Identification Probability",
        "Probability",
        plot_dir / "budget_identification_probability",
    )
    lineplot(summary, "budget", "final_normalized_entropy_mean", "Final Entropy vs Budget", "Final normalized entropy", plot_dir / "final_entropy_budget")
    lineplot(summary, "budget", "cost_per_success_mean", "Cost Per Success vs Budget", "Cost per success", plot_dir / "cost_per_success_budget")

    averaged = (
        raw.groupby(["policy", "timestep"], as_index=False)
        .agg(
            normalized_entropy=("normalized_entropy", "mean"),
            best_arm_pull_share=("best_arm_pull_share", "mean"),
        )
    )
    lineplot(averaged, "timestep", "normalized_entropy", "Allocation Entropy vs Time", "Normalized entropy", plot_dir / "entropy_time")
    lineplot(averaged, "timestep", "best_arm_pull_share", "Best Arm Pull Share vs Time", "Best-arm pull share", plot_dir / "best_arm_share_time")


def make_scaling_plots(out: Path, summary: pd.DataFrame) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    by_n = (
        summary.groupby(["policy", "n_arms"], as_index=False)
        .agg(success_rate_mean=("success_rate_mean", "mean"), regret_mean=("regret_mean", "mean"))
    )
    lineplot(by_n, "n_arms", "success_rate_mean", "Large Prompt Pool Scaling: Success", "Success rate", plot_dir / "scaling_success_rate")
    lineplot(by_n, "n_arms", "regret_mean", "Large Prompt Pool Scaling: Regret", "Cumulative regret", plot_dir / "scaling_regret")


def write_readme(out: Path) -> None:
    text = """# GitLab Strong-Arm Positive-Control Experiments

GitLab is used as a positive control because software-engineering browser tasks
have clear stateful workflows: inspect the repository or issue, change a small
piece of state, then verify the final state. That structure makes it plausible
that prompt strategy can create a real, measurable gap between arms.

The suite adds several GitLab-specialized strategies, but it does not assume any
of them wins. The paired benchmark runs every arm on the same task sequence and
defines `true_best_arm` as the arm with the highest empirical paired success
rate. If a generic arm wins, the adaptive experiments use that generic arm as
the reference.

Adaptive allocation should begin with broad exploration and then reduce
allocation entropy as evidence accumulates. Entropy is useful because it shows
whether a policy is still spreading budget uniformly or concentrating future
trials on a smaller set of promising prompt strategies.

Figures:

- `budget_success_rate`: whether adaptive policies improve with more budget.
- `budget_regret`: cumulative loss relative to the paired empirical best arm.
- `budget_identification_probability`: how often the true best arm is stably
  most-pulled by the end of a run.
- `entropy_time`: exploration-to-exploitation dynamics. Uniform should remain
  near maximum entropy; adaptive policies should decline if they exploit.
- `best_arm_share_time`: whether adaptive policies concentrate trials on the
  strongest empirical arm.
- `final_entropy_budget`: how concentration changes as budget increases.
- `cost_per_success_budget`: cost efficiency of each allocation rule.
- `scaling_success_rate` and `scaling_regret`: behavior as the prompt pool grows.

These experiments complement the Gmail and Shopping sweeps by providing a
positive-control setting where specialized software-engineering behavior should
be easier for adaptive policies to discover and exploit.
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
