#!/usr/bin/env python3
# ruff: noqa: E402,I001
"""Offline replay of GitLab allocation policies on paired arm-task outcomes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cold_start.inference.hedged_capital import HedgedCapitalEProcess
from cold_start.policies.base import PolicyState
from cold_start.policies.epsilon_greedy import EpsilonGreedyPolicy
from cold_start.policies.spruce import SprucePolicy
from cold_start.policies.thompson import ThompsonPolicy
from cold_start.policies.ucb import UCBPolicy
from cold_start.policies.uniform import UniformPolicy
from cold_start.policies.warmstart import WarmStart
from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms


TRUE_BEST = "gitlab_oracle_operator"


class ReplayPolicy(Protocol):
    def next_arm(self, t: int, state: PolicyState) -> str: ...

    def update(self, arm_id: str, reward: float, info: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class PolicySpec:
    name: str
    factory: Any


@dataclass(frozen=True)
class ReplayRow:
    reward: float
    cost: float
    runtime: float


class GreedyAfterWarmstart:
    def __init__(self, arm_ids: list[str], min_pulls: int, rng: np.random.Generator):
        self.arm_ids = arm_ids
        self.min_pulls = int(min_pulls)
        self.rng = rng
        self._last_scores: dict[str, float] = {}

    def next_arm(self, t: int, state: PolicyState) -> str:
        under = [aid for aid in self.arm_ids if state.pulls[aid] < self.min_pulls]
        if under:
            return under[(t - 1) % len(under)]
        means = {
            aid: state.successes[aid] / state.pulls[aid] if state.pulls[aid] else 0.0
            for aid in self.arm_ids
        }
        self._last_scores = means
        best = max(means.values())
        ties = [aid for aid, score in means.items() if score == best]
        return str(self.rng.choice(ties))

    def update(self, arm_id: str, reward: float, info: dict[str, Any]) -> None:
        return None

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._last_scores)


class CommitOnEvidence:
    """Warmstart, then commit once one arm has enough empirical/e-process lead."""

    def __init__(
        self,
        arm_ids: list[str],
        min_pulls: int,
        rng: np.random.Generator,
        *,
        min_success_rate: float = 0.75,
        min_margin: float = 0.15,
        min_log_e_upper: float = 0.0,
        fallback_c: float = 0.25,
    ):
        self.arm_ids = arm_ids
        self.min_pulls = int(min_pulls)
        self.rng = rng
        self.min_success_rate = float(min_success_rate)
        self.min_margin = float(min_margin)
        self.min_log_e_upper = float(min_log_e_upper)
        self.fallback = SprucePolicy(rng, exploration_c=fallback_c, tie_break="random")
        self.committed_arm: str | None = None

    def next_arm(self, t: int, state: PolicyState) -> str:
        if self.committed_arm is not None:
            return self.committed_arm
        under = [aid for aid in self.arm_ids if state.pulls[aid] < self.min_pulls]
        if under:
            return under[(t - 1) % len(under)]
        means = {
            aid: state.successes[aid] / state.pulls[aid]
            for aid in self.arm_ids
            if state.pulls[aid] >= self.min_pulls
        }
        leader = unique_argmax(means)
        if leader is not None:
            runner_up = max(v for aid, v in means.items() if aid != leader)
            margin = means[leader] - runner_up
            if (
                means[leader] >= self.min_success_rate
                and margin >= self.min_margin
                and state.log_e_upper[leader] >= self.min_log_e_upper
            ):
                self.committed_arm = leader
                return leader
        return self.fallback.next_arm(t, state)

    def update(self, arm_id: str, reward: float, info: dict[str, Any]) -> None:
        return None

    @property
    def scores(self) -> dict[str, float]:
        return dict(self.fallback.scores)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired_summary = pd.read_csv(args.paired_summary)
    arm_order = [
        arm.arm_id
        for arm in load_arms(args.arms_config, load_axes(args.axes))
        if arm.arm_id in set(paired_summary["arm_id"])
    ]
    table, task_order = load_reward_table(paired_summary)
    pools = build_pools(arm_order, paired_summary)
    specs = build_policy_specs()

    all_rows: list[pd.DataFrame] = []
    final_rows: list[dict[str, Any]] = []
    for pool_name, pool_arms in pools.items():
        for spec in specs:
            reps = 1 if spec.name == "uniform_rr" else args.replicates
            for rep in range(reps):
                ts = replay_one(
                    table=table,
                    task_order=task_order,
                    arm_ids=pool_arms,
                    policy_spec=spec,
                    rep=rep,
                    budget=args.budget,
                    seed=args.seed + 1009 * rep,
                    pool_name=pool_name,
                )
                all_rows.append(ts)
                final_rows.append(summarize_replay(ts, paired_summary))

    timeseries = pd.concat(all_rows, ignore_index=True)
    finals = pd.DataFrame(final_rows)
    policy_summary = summarize_policy(finals)

    timeseries.to_csv(out / "replay_timeseries.csv", index=False)
    finals.to_csv(out / "replay_final_runs.csv", index=False)
    policy_summary.to_csv(out / "replay_summary_by_policy.csv", index=False)
    make_plots(timeseries, policy_summary, out / "plots")
    write_report(out, paired_summary, pools, policy_summary)
    print(policy_summary.sort_values(["arm_pool", "success_rate_mean"], ascending=[True, False]).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=ROOT / "results/gitlab_strong_arm/paired/paired_summary.csv",
    )
    parser.add_argument("--arms-config", default="configs/arms_gitlab_strong.yaml")
    parser.add_argument("--axes", default="configs/axes.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gitlab_strong_arm/replay_oracle_policy")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def load_reward_table(paired_summary: pd.DataFrame) -> tuple[dict[tuple[str, str], ReplayRow], list[str]]:
    table: dict[tuple[str, str], ReplayRow] = {}
    task_order: list[str] = []
    for row in paired_summary.itertuples(index=False):
        arm_id = str(row.arm_id)
        path = ROOT / str(row.log_path)
        records = read_jsonl(path)
        if not task_order:
            task_order = [str(r["task_id"]) for r in records]
        for rec in records:
            task_id = str(rec["task_id"])
            table[(arm_id, task_id)] = ReplayRow(
                reward=float(rec.get("reward", 0.0)),
                cost=token_cost(rec.get("tokens", {})),
                runtime=float(rec.get("wallclock_s", 0.0) or 0.0),
            )
    missing = [
        (arm, task)
        for arm in paired_summary["arm_id"]
        for task in task_order
        if (str(arm), task) not in table
    ]
    if missing:
        raise RuntimeError(f"paired logs are missing {len(missing)} arm-task outcomes")
    return table, task_order


def build_pools(arm_order: list[str], paired_summary: pd.DataFrame) -> dict[str, list[str]]:
    ranked = list(paired_summary.sort_values("success_rate", ascending=False)["arm_id"])
    top6 = [TRUE_BEST] + [arm for arm in ranked if arm != TRUE_BEST][:5]
    top4 = [TRUE_BEST] + [arm for arm in ranked if arm != TRUE_BEST][:3]
    return {
        "full18": arm_order,
        "top6": [arm for arm in arm_order if arm in set(top6)],
        "top4": [arm for arm in arm_order if arm in set(top4)],
    }


def build_policy_specs() -> list[PolicySpec]:
    return [
        PolicySpec("uniform_rr", lambda arm_ids, rng: UniformPolicy(rng, mode="round_robin")),
        PolicySpec(
            "spruce_c1_warm3",
            lambda arm_ids, rng: WarmStart(
                SprucePolicy(rng, exploration_c=1.0, tie_break="random"), 3, rng
            ),
        ),
        PolicySpec(
            "spruce_c0.25_warm3",
            lambda arm_ids, rng: WarmStart(
                SprucePolicy(rng, exploration_c=0.25, tie_break="random"), 3, rng
            ),
        ),
        PolicySpec(
            "spruce_c0.10_warm2",
            lambda arm_ids, rng: WarmStart(
                SprucePolicy(rng, exploration_c=0.10, tie_break="random"), 2, rng
            ),
        ),
        PolicySpec(
            "ucb_c0.25_warm2",
            lambda arm_ids, rng: WarmStart(UCBPolicy(rng, exploration_c=0.25), 2, rng),
        ),
        PolicySpec(
            "ucb_c0.05_warm2",
            lambda arm_ids, rng: WarmStart(UCBPolicy(rng, exploration_c=0.05), 2, rng),
        ),
        PolicySpec(
            "epsilon0.05",
            lambda arm_ids, rng: EpsilonGreedyPolicy(rng, epsilon=0.05),
        ),
        PolicySpec("thompson", lambda arm_ids, rng: ThompsonPolicy(rng, alpha=1.0, beta=1.0)),
        PolicySpec(
            "greedy_after_warm2",
            lambda arm_ids, rng: GreedyAfterWarmstart(arm_ids, min_pulls=2, rng=rng),
        ),
        PolicySpec(
            "commit_margin_warm2",
            lambda arm_ids, rng: CommitOnEvidence(
                arm_ids,
                min_pulls=2,
                rng=rng,
                min_success_rate=0.75,
                min_margin=0.15,
                min_log_e_upper=0.0,
                fallback_c=0.10,
            ),
        ),
    ]


def replay_one(
    *,
    table: dict[tuple[str, str], ReplayRow],
    task_order: list[str],
    arm_ids: list[str],
    policy_spec: PolicySpec,
    rep: int,
    budget: int,
    seed: int,
    pool_name: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    policy: ReplayPolicy = policy_spec.factory(arm_ids, rng)
    state = PolicyState(arm_ids=arm_ids)
    eprocs = {arm: HedgedCapitalEProcess(m0=0.5) for arm in arm_ids}
    successes = 0
    cost = 0.0
    runtime = 0.0
    counts = {arm: 0 for arm in arm_ids}
    rows: list[dict[str, Any]] = []
    for t in range(1, budget + 1):
        task_id = task_order[(t - 1) % len(task_order)]
        arm_id = policy.next_arm(t, state)
        outcome = table[(arm_id, task_id)]
        log_e = eprocs[arm_id].update(outcome.reward)
        log_e_upper = eprocs[arm_id].log_e_upper
        state.record(arm_id, outcome.reward, log_e, log_e_upper=log_e_upper)
        policy.update(arm_id, outcome.reward, info={})
        counts[arm_id] += 1
        successes += int(outcome.reward >= 0.5)
        cost += outcome.cost
        runtime += outcome.runtime
        rows.append(
            {
                "arm_pool": pool_name,
                "policy": policy_spec.name,
                "replicate": rep,
                "timestep": t,
                "task_id": task_id,
                "selected_arm": arm_id,
                "reward": outcome.reward,
                "cumulative_successes": successes,
                "cumulative_success_rate": successes / t,
                "oracle_pulls": counts.get(TRUE_BEST, 0),
                "oracle_pull_share": counts.get(TRUE_BEST, 0) / t,
                "current_most_pulled_arm": unique_argmax({a: float(n) for a, n in counts.items()}),
                "cumulative_cost": cost,
                "cumulative_runtime": runtime,
            }
        )
    return pd.DataFrame(rows)


def summarize_replay(ts: pd.DataFrame, paired_summary: pd.DataFrame) -> dict[str, Any]:
    final = ts.iloc[-1]
    counts = ts["selected_arm"].value_counts().to_dict()
    most_pulled = unique_argmax({str(k): float(v) for k, v in counts.items()})
    paired_rates = dict(zip(paired_summary["arm_id"], paired_summary["success_rate"], strict=False))
    oracle_rate = float(paired_rates[TRUE_BEST])
    regret = sum(oracle_rate - float(paired_rates.get(arm, 0.0)) for arm in ts["selected_arm"])
    stable_most_oracle = first_stable_most_pulled_oracle(ts)
    return {
        "arm_pool": final["arm_pool"],
        "policy": final["policy"],
        "replicate": int(final["replicate"]),
        "budget": int(final["timestep"]),
        "successes": int(final["cumulative_successes"]),
        "success_rate": float(final["cumulative_success_rate"]),
        "oracle_pulls": int(final["oracle_pulls"]),
        "oracle_pull_share": float(final["oracle_pull_share"]),
        "most_pulled_arm": most_pulled,
        "oracle_most_pulled": most_pulled == TRUE_BEST,
        "stable_oracle_most_pulled_t": stable_most_oracle,
        "cumulative_regret": float(regret),
        "total_cost": float(final["cumulative_cost"]),
        "total_runtime": float(final["cumulative_runtime"]),
    }


def summarize_policy(finals: pd.DataFrame) -> pd.DataFrame:
    return (
        finals.groupby(["arm_pool", "policy"], as_index=False)
        .agg(
            success_rate_mean=("success_rate", "mean"),
            success_rate_se=("success_rate", sem),
            oracle_pull_share_mean=("oracle_pull_share", "mean"),
            oracle_pull_share_se=("oracle_pull_share", sem),
            oracle_most_pulled_rate=("oracle_most_pulled", "mean"),
            stable_oracle_most_pulled_t_mean=("stable_oracle_most_pulled_t", "mean"),
            regret_mean=("cumulative_regret", "mean"),
            total_cost_mean=("total_cost", "mean"),
        )
        .sort_values(["arm_pool", "success_rate_mean"], ascending=[True, False])
    )


def make_plots(timeseries: pd.DataFrame, policy_summary: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    averaged = (
        timeseries.groupby(["arm_pool", "policy", "timestep"], as_index=False)
        .agg(
            cumulative_success_rate=("cumulative_success_rate", "mean"),
            oracle_pull_share=("oracle_pull_share", "mean"),
        )
        .sort_values(["arm_pool", "policy", "timestep"])
    )
    for pool, g in averaged.groupby("arm_pool"):
        lineplot(g, "cumulative_success_rate", f"{pool}: replay success rate", plot_dir / f"{pool}_success_rate")
        lineplot(g, "oracle_pull_share", f"{pool}: oracle pull share", plot_dir / f"{pool}_oracle_pull_share")
    barplot(policy_summary, "success_rate_mean", "Replay mean success rate", plot_dir / "summary_success_rate")
    barplot(policy_summary, "oracle_pull_share_mean", "Replay oracle pull share", plot_dir / "summary_oracle_pull_share")


def lineplot(df: pd.DataFrame, y: str, title: str, outbase: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for policy, g in df.groupby("policy"):
        ax.plot(g["timestep"], g[y], linewidth=1.8, label=str(policy))
    ax.set_title(title)
    ax.set_xlabel("Task index")
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def barplot(df: pd.DataFrame, y: str, title: str, outbase: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labels = [f"{r.arm_pool}\n{r.policy}" for r in df.itertuples(index=False)]
    ax.bar(range(len(df)), df[y])
    ax.set_xticks(range(len(df)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(y.replace("_", " "))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    out: Path,
    paired_summary: pd.DataFrame,
    pools: dict[str, list[str]],
    policy_summary: pd.DataFrame,
) -> None:
    lines = [
        "# GitLab Oracle Offline Replay",
        "",
        "This replay uses the paired 40-task outcomes as a deterministic reward table.",
        "Each simulated run cycles through the 40 tasks four times for a 160-task budget.",
        "",
        "Arm pools:",
        "",
    ]
    for name, arms in pools.items():
        lines.append(f"- `{name}`: {', '.join(arms)}")
    lines.extend(["", "Paired ranking:", ""])
    lines.append(paired_summary[["arm_id", "success_rate", "successes"]].to_string(index=False))
    lines.extend(["", "Replay summary:", ""])
    cols = [
        "arm_pool",
        "policy",
        "success_rate_mean",
        "oracle_pull_share_mean",
        "oracle_most_pulled_rate",
        "stable_oracle_most_pulled_t_mean",
        "regret_mean",
    ]
    lines.append(policy_summary[cols].to_string(index=False))
    lines.extend(
        [
            "",
            "Main diagnostic: a good positive-control policy should have both high",
            "`success_rate_mean` and high `oracle_pull_share_mean`; otherwise it may",
            "be winning by incidental task/arm interactions rather than identifying oracle.",
            "",
        ]
    )
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def first_stable_most_pulled_oracle(ts: pd.DataFrame) -> int | None:
    values = list(ts["current_most_pulled_arm"] == TRUE_BEST)
    times = list(ts["timestep"])
    for idx, value in enumerate(values):
        if value and all(values[idx:]):
            return int(times[idx])
    return None


def unique_argmax(values: dict[str, float]) -> str | None:
    if not values:
        return None
    best = max(values.values())
    winners = sorted(k for k, v in values.items() if v == best)
    return winners[0] if len(winners) == 1 else None


def token_cost(tokens: Any) -> float:
    if not isinstance(tokens, dict):
        return 0.0
    return float(tokens.get("cost_usd", 0.0) or 0.0)


def sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.std(ddof=1) / math.sqrt(len(vals)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
