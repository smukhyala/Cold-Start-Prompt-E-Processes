"""CLI entrypoint: `cold-start-compare`.

Runs a single base experiment under multiple allocation policies, holding the
seed, task stream, e-process, and global null fixed. Writes a comparison
report contrasting:

- τ_α (first global rejection time) per policy
- cumulative regret proxy Σ_t (μ̂_{a*} − μ̂_{A_t})
- final per-arm pull counts
- overlaid global log-e trajectories

Motivated by Mukhyala & Waudby-Smith § 3.8: "comparing adaptive allocation
with uniform evaluation, determining whether evidence-guided sampling reaches
conclusions faster".

Usage:
    cold-start-compare \\
        --base configs/pilot_toy_upward.yaml \\
        --policies uniform,epsilon_greedy,thompson,ucb,spruce \\
        --out reports/pilot_toy_upward_comparison
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from cold_start.cli import _bootstrap  # noqa: F401  (registry population)
from cold_start.config import PolicySpec, load_config
from cold_start.runner.orchestrator import run_trial

_DEFAULT_POLICY_PARAMS: dict[str, dict[str, object]] = {
    "uniform": {},
    "epsilon_greedy": {"epsilon": 0.1},
    "thompson": {},
    "ucb": {"exploration_c": 1.0},
    "spruce": {"exploration_c": 1.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="cold-start-compare")
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Base experiment config; one run is performed per policy with this base.",
    )
    parser.add_argument(
        "--policies",
        required=True,
        help="Comma-separated policy names (registry kind='policy'). "
        "Defaults are filled in from a built-in table when params are not overridden.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for comparison artifacts. "
        "Defaults to reports/<base_name>_comparison.",
    )
    parser.add_argument(
        "--keep-warmstart",
        action="store_true",
        help="Keep the base config's warmstart wrapper. By default it is stripped "
        "so different policies are compared head-to-head from t=1.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.base)
    policy_names = [s.strip() for s in args.policies.split(",") if s.strip()]
    if not policy_names:
        parser.error("--policies must list at least one policy name")

    out_dir = args.out or Path("reports") / f"{base_cfg.trial.name}_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One sub-run per policy, all sharing the same seed / task stream / e-process.
    # Each policy is wrapped in try/except so one crash doesn't lose the others'
    # data — important for long unattended runs where Anthropic 5xx, browser
    # hangs, or port collisions are plausible mid-experiment.
    runs: dict[str, Path] = {}
    failures: dict[str, str] = {}
    for pname in policy_names:
        cfg = copy.deepcopy(base_cfg)
        cfg.policy = PolicySpec(
            type=pname,
            params=dict(_DEFAULT_POLICY_PARAMS.get(pname, {})),
            warmstart=cfg.policy.warmstart if args.keep_warmstart else None,
        )
        cfg.trial.name = f"{base_cfg.trial.name}__cmp_{pname}"
        try:
            log_path = run_trial(cfg, trial_idx=0, log_dir=out_dir / pname)
        except KeyboardInterrupt:
            # Don't swallow Ctrl-C: the user wants to stop the whole comparison,
            # not skip this policy. Any partial log for this policy is still on
            # disk and can be picked up via --resume-from on a follow-up run.
            print(f"  ! {pname}: interrupted; remaining policies skipped")
            failures[pname] = "interrupted (KeyboardInterrupt)"
            break
        except Exception as exc:  # noqa: BLE001 — comparison-level resilience
            failures[pname] = f"{type(exc).__name__}: {exc}"
            # Best-effort: any JSONL the orchestrator did write before the crash
            # is still on disk under out_dir / pname / *.jsonl.
            partials = sorted((out_dir / pname).glob("*.jsonl"))
            partial_str = partials[-1] if partials else "(no records written)"
            print(f"  ! {pname}: FAILED — {exc!r}. Partial: {partial_str}")
            continue
        runs[pname] = log_path
        print(f"  · {pname}: {log_path}")

    if not runs:
        print("no policies completed; nothing to compare")
        if failures:
            for pname, why in failures.items():
                print(f"  {pname}: {why}")
        raise SystemExit(1)

    # Build per-policy summaries: τ_α, cum-regret, final pulls, log_e series.
    summaries = {name: _summarize_run(path) for name, path in runs.items()}

    # Write comparison.md
    md = _render_markdown(base_cfg.trial.name, summaries, failures=failures)
    (out_dir / "comparison.md").write_text(md)

    # Overlaid global log-e plot.
    _plot_global_log_e_overlay(summaries, out_dir / "global_log_e_overlay.png")

    # Side-by-side pull-count plot.
    _plot_pull_histograms(summaries, out_dir / "pulls_by_policy.png")

    print(f"comparison report written to {out_dir}")


def _summarize_run(log_path: Path) -> dict:
    """Read a JSONL run and extract the comparison-relevant series.

    Returns a dict with:
        ts:               list[int]   – time index per record
        global_log_e:     list[float] – global log E_t at each t
        arm_pulls_series: dict[arm_id, list[int]] – cumulative pulls per arm per t
        arm_success_series: dict[arm_id, list[int]]
        rewards:          list[float] – per-step reward
        chosen:           list[str]   – per-step arm id
        final_pulls:      dict[arm_id, int]
        final_means:      dict[arm_id, float]
        alpha:            float
        tau_alpha:        int | None  – first t at which global log_e ≥ log(1/α)
    """
    ts: list[int] = []
    g_log_e: list[float] = []
    rewards: list[float] = []
    chosen: list[str] = []
    arm_pulls: dict[str, list[int]] = {}
    arm_succ: dict[str, list[int]] = {}
    alpha: float | None = None

    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = int(rec["t"])
            ts.append(t)
            g_log_e.append(float(rec["global_e"]["log_e"]))
            rewards.append(float(rec["reward"]))
            chosen.append(rec["arm_id"])
            if alpha is None:
                alpha = float(rec["inference"]["alpha"])
            for aid, st in rec["per_arm_state"].items():
                arm_pulls.setdefault(aid, []).append(int(st["pulls"]))
                arm_succ.setdefault(aid, []).append(int(st["successes"]))

    if alpha is None:
        raise ValueError(f"empty run log {log_path}")

    log_thresh = math.log(1.0 / alpha)
    tau_alpha: int | None = None
    for t, v in zip(ts, g_log_e):
        if v >= log_thresh:
            tau_alpha = t
            break

    final_pulls = {aid: series[-1] for aid, series in arm_pulls.items()}
    final_succ = {aid: series[-1] for aid, series in arm_succ.items()}
    final_means = {
        aid: (final_succ[aid] / final_pulls[aid]) if final_pulls[aid] else float("nan")
        for aid in final_pulls
    }

    return {
        "ts": ts,
        "global_log_e": g_log_e,
        "arm_pulls_series": arm_pulls,
        "rewards": rewards,
        "chosen": chosen,
        "final_pulls": final_pulls,
        "final_means": final_means,
        "alpha": alpha,
        "tau_alpha": tau_alpha,
    }


def _cumulative_regret_proxy(summary: dict) -> float:
    """Σ_t (μ̂_{a*} − μ̂_{A_t}) at end of run, using final per-arm means.

    Not a true regret (which would need the true μ_a*), but a self-contained
    proxy comparable across policies on the same task stream.
    """
    best_mu = max(
        (mu for mu in summary["final_means"].values() if not math.isnan(mu)),
        default=0.0,
    )
    return sum(
        best_mu - summary["final_means"].get(aid, 0.0)
        for aid in summary["chosen"]
        if not math.isnan(summary["final_means"].get(aid, float("nan")))
    )


def _render_markdown(
    base_name: str,
    summaries: dict[str, dict],
    failures: dict[str, str] | None = None,
) -> str:
    any_summary = next(iter(summaries.values()))
    alpha = any_summary["alpha"]
    T = max(s["ts"][-1] for s in summaries.values())
    log_thresh = math.log(1.0 / alpha)
    failures = failures or {}

    lines: list[str] = [
        f"# Allocation comparison — `{base_name}`",
        "",
        f"- T = {T}",
        f"- α = {alpha}; log(1/α) = {log_thresh:.3f}",
        f"- policies completed: {', '.join(summaries.keys())}",
    ]
    if failures:
        lines.append(f"- policies failed: {', '.join(failures.keys())}")
    lines.append("")

    if failures:
        lines += [
            "## ⚠️ Failed policies",
            "",
            "These policies crashed mid-run and are excluded from the comparison",
            "below. Their partial JSONL logs (if any) remain under "
            f"`<out_dir>/<policy>/`; rerun individually with `cold-start-run "
            "--resume-from <partial>` to recover.",
            "",
            "| policy | failure |",
            "|---|---|",
        ]
        for pname, why in failures.items():
            lines.append(f"| {pname} | `{why}` |")
        lines.append("")

    lines += [
        "## τ_α — first global rejection time",
        "",
        "| policy | τ_α | rejected by T |",
        "|---|---|---|",
    ]
    for name, s in summaries.items():
        tau = s["tau_alpha"]
        lines.append(
            f"| {name} | {tau if tau is not None else '—'} | "
            f"{'yes' if tau is not None else 'no'} |"
        )
    lines.append("")

    lines += [
        "## Cumulative regret proxy",
        "",
        "Σ_t (μ̂_{a*} − μ̂_{A_t}) using final per-arm sample means. Lower is better.",
        "",
        "| policy | cum. regret proxy |",
        "|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(f"| {name} | {_cumulative_regret_proxy(s):.2f} |")
    lines.append("")

    arm_ids = sorted(any_summary["final_pulls"].keys())
    lines += [
        "## Final pulls per arm",
        "",
        "| policy | " + " | ".join(arm_ids) + " |",
        "|---|" + "---|" * len(arm_ids),
    ]
    for name, s in summaries.items():
        cells = [str(s["final_pulls"].get(aid, 0)) for aid in arm_ids]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## Artifacts",
        "",
        "- `global_log_e_overlay.png` — global log-e trajectories overlaid",
        "- `pulls_by_policy.png` — per-arm pull-count bars per policy",
        "- `<policy>/<run>.jsonl` — full per-step logs per policy",
        "",
    ]
    return "\n".join(lines) + "\n"


def _plot_global_log_e_overlay(summaries: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    any_s = next(iter(summaries.values()))
    log_thresh = math.log(1.0 / any_s["alpha"])
    for name, s in summaries.items():
        ax.plot(s["ts"], s["global_log_e"], label=name)
    ax.axhline(log_thresh, color="red", linestyle="--", label=f"log(1/α)={log_thresh:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("global log E_t")
    ax.set_title("Global log-e trajectories by allocation policy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _plot_pull_histograms(summaries: dict[str, dict], out: Path) -> None:
    any_s = next(iter(summaries.values()))
    arm_ids = sorted(any_s["final_pulls"].keys())
    n_policies = len(summaries)
    fig, axes = plt.subplots(
        nrows=n_policies, ncols=1,
        figsize=(0.5 * len(arm_ids) + 3, 2.5 * n_policies),
        sharex=True,
    )
    if n_policies == 1:
        axes = [axes]
    for ax, (name, s) in zip(axes, summaries.items()):
        pulls = [s["final_pulls"].get(aid, 0) for aid in arm_ids]
        ax.bar(arm_ids, pulls)
        ax.set_ylabel("pulls")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
