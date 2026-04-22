"""Streamlit dashboard for coldStartPrompts runs.

Launch with either:
    .venv/bin/cold-start-viz
    .venv/bin/python -m cold_start.viz
    .venv/bin/streamlit run src/cold_start/viz/app.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import streamlit as st

_LOG_ALPHA_05 = math.log(20.0)  # reject global/per-arm null at α=0.05 one-sided

from cold_start.viz.charts import (
    cs_band_chart,
    cumulative_pass_rate_chart,
    global_log_e_chart,
    per_arm_log_e_chart,
)
from cold_start.viz.loaders import RunMeta, discover_runs, load_run, merge_runs

st.set_page_config(page_title="Cold Start Results", layout="wide")


@st.cache_data(show_spinner=False)
def _cached_discover(logs_dir: str) -> list[RunMeta]:
    return discover_runs(logs_dir)


@st.cache_data(show_spinner=True)
def _cached_load(base_id: str, paths: tuple[str, ...], logs_dir: str):
    runs = discover_runs(logs_dir)
    meta = next(r for r in runs if r.base_id == base_id)
    return load_run(meta)


def _interpret(df, total_pulls: int, global_le: float, rejected: bool, n_records: int) -> str:
    """Auto-generate a short interpretation from the final per-arm state."""
    lines: list[str] = []

    # Global verdict.
    if rejected:
        lines.append(
            f"- **Global null rejected.** log_e = {global_le:.2f} > {_LOG_ALPHA_05:.3f}. "
            f"There is >95% confidence that *at least one* arm's true pass rate is "
            f"strictly greater than m₀=0.5. This does NOT say which arm."
        )
    else:
        gap = _LOG_ALPHA_05 - global_le
        lines.append(
            f"- **Global null not yet rejected.** log_e = {global_le:.2f}, "
            f"need > {_LOG_ALPHA_05:.3f} (distance {gap:.2f} on log scale)."
        )

    if len(df) == 0:
        return "\n".join(lines)

    top = df.iloc[0]
    worst = df.iloc[-1]

    top_sig = "is" if top.log_e > _LOG_ALPHA_05 else "is NOT yet"
    top_cs = "crosses 0.5" if top.cs_lo > 0.5 else "does not yet cross 0.5"
    lines.append(
        f"- **Top arm: `{top.arm}`** — {int(top.passes)}/{int(top.pulls)} passes "
        f"({top.pass_rate:.0%}), log_e = {top.log_e:.2f}. "
        f"Individually {top_sig} significant at α=0.05, and its confidence lower "
        f"bound ({top.cs_lo:.2f}) {top_cs}."
    )

    if worst.arm != top.arm:
        lines.append(
            f"- **Laggard: `{worst.arm}`** — {int(worst.passes)}/{int(worst.pulls)} "
            f"({worst.pass_rate:.0%}), log_e = {worst.log_e:.2f}. "
            f"Consider filtering the per-task table by arm=`{worst.arm}` with "
            f"'Only failures' on to inspect where it's losing."
        )

    # Cohort in the middle.
    middle_arms = df.iloc[1:-1]
    if len(middle_arms) > 0:
        middle_rates = middle_arms["pass_rate"].round(3).tolist()
        if len(set(middle_rates)) == 1:
            lines.append(
                f"- **Middle cohort tied** at {middle_rates[0]:.0%} "
                f"({', '.join(f'`{a}`' for a in middle_arms['arm'])}). "
                f"Statistically indistinguishable from each other at this n."
            )

    # Caveats.
    min_pulls = int(df["pulls"].min())
    if min_pulls < 20:
        lines.append(
            f"- **Caveat — thin data.** Minimum pulls per arm = {min_pulls}. "
            f"Separating a 100% arm from a 75% arm at α=0.05 typically needs "
            f"20–30 pulls per arm. Treat these rankings as provisional."
        )

    if n_records < total_pulls:
        lines.append(
            f"- **View note.** This JSONL has {n_records} records but the "
            f"e-process has seen {total_pulls} pulls (the extras were replayed "
            f"from a prior JSONL). Use sidebar → *Merge multiple runs* to load "
            f"both files for a complete per-task table."
        )

    return "\n".join(lines)


def _format_run(meta: RunMeta) -> str:
    parts = [f"{meta.n_records} tasks"]
    if meta.task_source_type != "?":
        parts.append(meta.task_source_type)
    if len(meta.jsonl_paths) > 1:
        parts.append(f"{len(meta.jsonl_paths)} files")
    return f"{meta.latest_ts[:16]} · {meta.base_id} ({', '.join(parts)})"


def main() -> None:
    st.sidebar.title("Cold Start")
    logs_dir = st.sidebar.text_input("Logs directory", value="logs")
    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        _cached_discover.clear()
        _cached_load.clear()

    runs = _cached_discover(logs_dir)
    if not runs:
        st.sidebar.warning(f"No JSONLs found in `{logs_dir}/`.")
        st.stop()

    merge_mode = st.sidebar.toggle(
        "Merge multiple runs",
        value=False,
        help="Combine several JSONLs (e.g. a pre-fix original + its resume) into one view.",
    )

    if merge_mode:
        chosen = st.sidebar.multiselect(
            "Runs to merge (order = time)",
            options=[r.base_id for r in runs],
            format_func=lambda bid: _format_run(next(r for r in runs if r.base_id == bid)),
        )
        if not chosen:
            st.sidebar.info("Pick ≥2 runs to merge.")
            st.stop()
        metas = [next(r for r in runs if r.base_id == bid) for bid in chosen]
        run = merge_runs(metas)
    else:
        selected_base = st.sidebar.selectbox(
            "Run",
            options=[r.base_id for r in runs],
            format_func=lambda bid: _format_run(next(r for r in runs if r.base_id == bid)),
        )
        meta = next(r for r in runs if r.base_id == selected_base)
        run = _cached_load(meta.base_id, tuple(str(p) for p in meta.jsonl_paths), logs_dir)

    st.title(run.run_id)

    if run.n_records == 0:
        st.warning("Run has no records.")
        st.stop()

    last = run.records[-1]
    total_pulls = sum(int(s["pulls"]) for s in last["per_arm_state"].values())
    global_le = float(last["global_e"]["log_e"])
    rejected = bool(last["global_e"]["rejected"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Records in this view",
        run.n_records,
        help="Rows loaded from the selected JSONL(s). If this is a resume file "
        "without its origin merged in, this will be less than the total pulls "
        "the e-process has seen (see Pulls below). Use sidebar → "
        "Merge multiple runs to combine origin + resume.",
    )
    c2.metric(
        "Pulls in e-process",
        total_pulls,
        help="Total arm pulls the per-arm and global e-processes have seen "
        "(sum of per-arm 'pulls'). Reflects replayed history on a resumed run.",
    )
    c3.metric(
        "Global log_e",
        f"{global_le:.2f}",
        help=f"Combined evidence across all arms against H₀ = every arm's true "
        f"pass rate ≤ m₀=0.5. Natural-log scale. Reject H₀ when log_e > "
        f"log(1/α) = log(20) ≈ {_LOG_ALPHA_05:.3f} (α=0.05).",
    )
    c4.metric(
        "Global null rejected (α=0.05)",
        "yes" if rejected else "no",
        help="If yes, there is >95% confidence that at least one arm has a "
        "true pass rate > 0.5. This does NOT identify which arm.",
    )
    c5.metric(
        "Arms",
        len(last["per_arm_state"]),
        help="Distinct system-prompt variants in this trial. Defined in "
        "configs/arms_initial.yaml.",
    )
    st.caption(
        f"config_hash=`{run.config_hash}` · task source: {meta.task_source_type} · "
        f"sources: {', '.join(p.name for p in meta.jsonl_paths)}"
    )

    tab_summary, tab_charts, tab_tasks, tab_detail = st.tabs(
        ["Summary", "Time series", "Per-task table", "Task detail"]
    )

    with tab_summary:
        st.subheader("Per-arm summary (final state)")
        df = run.per_arm_summary().round(3)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Column glossary — what each variable means"):
            st.markdown(
                "- **arm** — a rendered system-prompt variant. One row per arm. "
                "Defined in `configs/arms_initial.yaml`; rendered via `configs/template.jinja` over the axes.\n"
                "- **pulls** — total tasks this arm has been assigned (cumulative, "
                "including replayed records on a resumed run).\n"
                "- **passes** — how many of those pulls the benchmark's verifier "
                "returned `success=1`.\n"
                "- **pass_rate** — `passes / pulls`. Point estimate of the arm's "
                "true success rate.\n"
                "- **mean_wallclock_s** — average seconds per task. Proxy for "
                "compute cost and how deliberative the arm's prompt makes the agent.\n"
                "- **mean_steps** — average number of agent browser actions per "
                "task (including the final `done`). Lower = more efficient.\n"
                "- **log_e** — *per-arm* e-process value. Log-scale evidence that "
                "this specific arm's true pass rate ≠ m₀=0.5 (two-sided). "
                f"Reject that arm's null at α=0.05 when `log_e > log(20) ≈ {_LOG_ALPHA_05:.3f}`. "
                "Under H₀, log_e > threshold with probability ≤ α at any sample size "
                "(anytime-valid).\n"
                "- **cs_lo / cs_hi** — 95% *confidence sequence* on the arm's true "
                "pass rate. Anytime-valid: the interval covers the true rate at "
                "every sample size simultaneously, so you can peek without inflating "
                "error. If `cs_lo > 0.5`, that arm is individually proven > 50% at α=0.05."
            )

        st.markdown("### Interpretation")
        st.markdown(_interpret(df, total_pulls, global_le, rejected, run.n_records))

    with tab_charts:
        with st.expander("How to read these charts"):
            st.markdown(
                "- **Per-arm log e-process** — each line = one arm's evidence "
                "against its own null m₀=0.5, over time. An arm that's truly "
                "better than 50% drifts upward; truly worse drifts negative. "
                "The dotted line at log(20) ≈ 2.996 is the α=0.05 rejection "
                "threshold.\n"
                "- **Global log e-process** — the combined null across all arms. "
                "Crosses the dotted line when there's ≥95% confidence that at "
                "least one arm beats 50%.\n"
                "- **Cumulative pass rate per arm** — running mean pass rate "
                "plotted at each task index t where that arm was pulled. "
                "Converges slowly for small samples.\n"
                "- **Confidence-sequence bands** — shaded region is the 95% "
                "two-sided band on true pass rate, *valid at every t* "
                "simultaneously (you can peek without inflating error). Bands "
                "shrink as more pulls accumulate."
            )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(per_arm_log_e_chart(run), use_container_width=True)
            st.plotly_chart(cumulative_pass_rate_chart(run), use_container_width=True)
        with col_b:
            st.plotly_chart(global_log_e_chart(run), use_container_width=True)
        st.plotly_chart(cs_band_chart(run), use_container_width=True)

    with tab_tasks:
        st.caption(
            "One row per task executed. `t` is 1-indexed task number in the run; "
            "`arm_id` is the prompt variant pulled; `difficulty` comes from the "
            "benchmark. `success=1` means the benchmark's verifier said pass. "
            "Filter by arm or 'Only failures' to inspect where an arm is losing."
        )
        df = run.df[
            [
                "t",
                "arm_id",
                "task_id",
                "difficulty",
                "success",
                "reward",
                "steps",
                "wallclock_s",
            ]
        ]
        c1, c2, c3 = st.columns(3)
        arm_opts = ["(all)"] + sorted(df["arm_id"].unique().tolist())
        arm_filt = c1.selectbox("Arm", arm_opts)
        diff_opts = ["(all)"] + sorted(df["difficulty"].unique().tolist())
        diff_filt = c2.selectbox("Difficulty", diff_opts)
        only_fails = c3.checkbox("Only failures", value=False)
        filt = df
        if arm_filt != "(all)":
            filt = filt[filt["arm_id"] == arm_filt]
        if diff_filt != "(all)":
            filt = filt[filt["difficulty"] == diff_filt]
        if only_fails:
            filt = filt[filt["success"] == 0]
        st.dataframe(filt, use_container_width=True, hide_index=True, height=500)

    with tab_detail:
        st.caption(
            "Inspect one task's full trace. For webarena runs this includes the "
            "verifier's result, the LLM conversation transcript (the arm's system "
            "prompt is near the top), and optionally the step-by-step screenshots "
            "(loaded on demand — they're ~100KB each)."
        )
        available_ts = sorted({int(r["t"]) for r in run.records})
        min_t, max_t = available_ts[0], available_ts[-1]
        t_sel = st.number_input(
            "t",
            min_value=int(min_t),
            max_value=int(max_t),
            value=int(min_t),
            step=1,
            help=(
                f"Task index in the run. Available t in this view: {min_t}..{max_t}"
                + (" (non-contiguous)" if len(available_ts) != max_t - min_t + 1 else "")
                + ". Merge with the origin JSONL to see earlier t."
            ),
        )
        rec = next((r for r in run.records if int(r["t"]) == int(t_sel)), None)
        if rec is None:
            st.warning(
                f"No record at t={t_sel} in this view. "
                f"Gaps are normal for a resume-only JSONL — try a nearby t or "
                f"merge runs in the sidebar."
            )
            rec = None

        if rec is None:
            st.stop()

        meta_info = rec.get("task_meta", {})
        instruction = meta_info.get("raw", {}).get("instruction") or meta_info.get("instruction", "?")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("arm", rec["arm_id"])
        c2.metric("task", rec["task_id"])
        c3.metric("reward", f"{rec['reward']:.2f}")
        c4.metric("wallclock", f"{rec['wallclock_s']:.1f}s")
        st.markdown(f"**Instruction:** {instruction}")
        st.caption(f"difficulty={meta_info.get('difficulty','?')} · steps={rec['steps']}")

        artifacts = run.task_artifacts(rec["arm_id"], rec["task_id"])
        if not artifacts["exists"]:
            st.info(
                f"No artifacts at `{artifacts['dir']}` — either this run didn't write "
                "per-task artifacts (e.g. sanity/toy env), or a later pull of the "
                "same (arm, task) overwrote them."
            )
            st.json(rec, expanded=False)
        else:
            st.caption(f"artifact dir: `{artifacts['dir']}`")
            if rj_path := artifacts.get("result.json"):
                with open(rj_path) as f:
                    st.json(json.load(f), expanded=False)
            if convs := artifacts.get("conversations"):
                with st.expander(f"Conversation transcripts ({len(convs)} file(s))"):
                    for path in convs:
                        st.text_area(path.name, path.read_text()[:30000], height=280, key=str(path))
            if hist := artifacts.get("history.json"):
                with st.expander("history.json"):
                    with open(hist) as f:
                        st.json(json.load(f), expanded=False)
            if screenshots := artifacts.get("screenshots"):
                show_ss = st.toggle(f"Load {len(screenshots)} screenshot(s)", key=f"ss_{t_sel}")
                if show_ss:
                    for path in screenshots:
                        st.image(str(path), caption=path.name)


if __name__ == "__main__":
    main()
