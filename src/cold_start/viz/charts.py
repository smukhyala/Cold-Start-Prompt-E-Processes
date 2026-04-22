"""Plotly chart builders for the run detail view."""

from __future__ import annotations

import math

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cold_start.viz.loaders import Run

_LOG_ALPHA_05 = math.log(20.0)  # ≈ 2.996; reject null at α=0.05 one-sided


def per_arm_log_e_chart(run: Run) -> go.Figure:
    traj = run.state_trajectory()
    fig = px.line(traj, x="t", y="log_e", color="arm", markers=True)
    fig.add_hline(y=_LOG_ALPHA_05, line_dash="dot", annotation_text="α=0.05", annotation_position="top right")
    fig.add_hline(y=0, line_dash="dash", opacity=0.3)
    fig.update_layout(title="Per-arm log e-process (two-sided, m₀=0.5)", yaxis_title="log E_t")
    return fig


def global_log_e_chart(run: Run) -> go.Figure:
    fig = px.line(run.df, x="t", y="global_log_e", markers=True)
    fig.add_hline(y=_LOG_ALPHA_05, line_dash="dot", annotation_text="α=0.05", annotation_position="top right")
    fig.add_hline(y=0, line_dash="dash", opacity=0.3)
    fig.update_layout(title="Global log e-process (combined null)", yaxis_title="log E_t")
    return fig


def cumulative_pass_rate_chart(run: Run) -> go.Figure:
    df = run.df.copy().sort_values("t")
    df["cum_success"] = df.groupby("arm_id")["success"].cumsum()
    df["cum_pulls"] = df.groupby("arm_id").cumcount() + 1
    df["cum_pass_rate"] = df["cum_success"] / df["cum_pulls"]
    fig = px.line(df, x="t", y="cum_pass_rate", color="arm_id", markers=True)
    fig.add_hline(y=0.5, line_dash="dash", opacity=0.3)
    fig.update_layout(
        title="Cumulative pass rate (per arm, over that arm's own pulls)",
        yaxis_title="pass rate",
        yaxis_range=[0, 1.05],
    )
    return fig


def cs_band_chart(run: Run) -> go.Figure:
    traj = run.state_trajectory()
    arms = sorted(traj["arm"].unique())
    cols = min(3, len(arms))
    rows = max(1, (len(arms) + cols - 1) // cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=arms,
        shared_yaxes=True,
        shared_xaxes=True,
    )
    for i, arm in enumerate(arms):
        r = i // cols + 1
        c = i % cols + 1
        sub = traj[traj["arm"] == arm].sort_values("t")
        fig.add_trace(
            go.Scatter(
                x=sub["t"],
                y=sub["cs_hi"],
                line=dict(color="rgba(80,120,200,0.9)"),
                name="cs_hi",
                showlegend=(i == 0),
                legendgroup="cs_hi",
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["t"],
                y=sub["cs_lo"],
                line=dict(color="rgba(80,120,200,0.9)"),
                fill="tonexty",
                fillcolor="rgba(80,120,200,0.15)",
                name="cs_lo",
                showlegend=(i == 0),
                legendgroup="cs_lo",
            ),
            row=r,
            col=c,
        )
        fig.add_hline(y=0.5, line_dash="dash", opacity=0.3, row=r, col=c)
    fig.update_layout(
        title="Confidence-sequence bands per arm (1−α coverage of pass rate)",
        height=max(260, 220 * rows),
        showlegend=False,
    )
    fig.update_yaxes(range=[0, 1.05])
    return fig
