#!/usr/bin/env python3
"""Plot the exploratory dated SWE-bench score-leader audit."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "results/paper/swe_leaderboard/official.json").read_text())
rows = data["comparison_rows"]
if not rows:
    raise RuntimeError("No comparison rows")

def date(value):
    return datetime.strptime(value, "%Y%m%d")

first = (date(data["initial_leader"][:8]), rows[0]["leader_resolved"] / 5)
claims = [r for r in rows if r["leader_claim"]]
steps = [first] + [(date(r["directory_date"]), r["entrant_resolved"] / 5) for r in claims]

fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
ax.step([x for x, _ in steps] + [date(rows[-1]["directory_date"])],
        [y for _, y in steps] + [steps[-1][1]], where="post", color="#334155", linewidth=1.4)

styles = (
    ("No 5% crossing", lambda r: r["nominal_cross_task"] is None, "#94a3b8", "o", 18),
    ("Nominal crossing only", lambda r: r["nominal_cross_task"] is not None and r["spent_cross_task"] is None,
     "#2563eb", "o", 34),
    ("Entrant-spending crossing", lambda r: r["spent_cross_task"] is not None, "#dc2626", "D", 42),
)
for label, select, color, marker, size in styles:
    selected = [r for r in claims if select(r)]
    ax.scatter([date(r["directory_date"]) for r in selected],
               [r["entrant_resolved"] / 5 for r in selected],
               s=size, c=color, marker=marker, label=f"{label} (n={len(selected)})",
               zorder=3, linewidths=.5, edgecolors="white")

ax.set_ylabel("Running best resolved (%)")
ax.set_xlabel("Date prefix of submission directory")
ax.set_ylim(0, 85)
ax.set_xlim(date(data["initial_leader"][:8]), date(rows[-1]["directory_date"]))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="y", color="#e2e8f0", linewidth=.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", fontsize=8, frameon=False)
out = ROOT / "paper/framework_figures/swe_leaderboard_audit.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
