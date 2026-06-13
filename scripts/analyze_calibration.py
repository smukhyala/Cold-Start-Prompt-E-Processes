#!/usr/bin/env python
"""Read a calibration JSONL log and report μ̂_baseline + suggested μ_0.

Usage:
    python scripts/analyze_calibration.py logs/<calibration_log>.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: analyze_calibration.py <jsonl>")
    p = Path(sys.argv[1])
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    n = len(recs)
    if n == 0:
        sys.exit("empty calibration log")

    rewards = [float(r["reward"]) for r in recs]
    successes = sum(1 for r in rewards if r >= 0.5)
    mu = successes / n
    # Wilson 95% CI for a Bernoulli mean — preferred over the normal approx
    # at small n because it stays inside [0,1].
    z = 1.96
    denom = 1 + z**2 / n
    center = (mu + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(mu * (1 - mu) / n + z**2 / (4 * n**2))) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)

    total_cost = sum(r.get("tokens", {}).get("cost_usd", 0.0) for r in recs)
    total_wall = sum(r["wallclock_s"] for r in recs)
    failures = [r for r in recs if r["success"] == 0]

    print(f"# Calibration analysis — {p.name}")
    print()
    print(f"- n = {n}")
    print(f"- successes = {successes}")
    print(f"- failures = {len(failures)}")
    print(f"- μ̂_baseline = {mu:.4f}")
    print(f"- Wilson 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"- total wallclock = {total_wall:.0f}s ({total_wall/n:.1f}s/task)")
    print(f"- total cost = ${total_cost:.3f} (${total_cost/n:.4f}/task)")
    print()

    # Recommended μ_0 — three options ordered by interpretive strength.
    print("## μ_0 recommendations")
    print()
    print(f"| choice | μ_0 | interpretation |")
    print(f"|---|---|---|")
    print(f"| Strict (CI lower bound) | {lo:.3f} | H_0: arm is no better than the *worst* baseline could be |")
    print(f"| Plain (point estimate) | {mu:.3f} | H_0: arm is no better than baseline's empirical mean |")
    print(f"| Lenient (CI lower − margin) | {max(0.0, lo - 0.05):.3f} | H_0: arm is no better than baseline minus 5pp margin |")
    print()
    print(f"Recommended: **{lo:.3f}** (Wilson lower bound).")
    print()

    if failures:
        print("## Failures (tasks the baseline arm couldn't complete)")
        for r in failures:
            tid = r["task_id"]
            diff = r["task_meta"].get("difficulty", "?")
            steps = r["steps"]
            print(f"- {tid} (diff={diff}, steps={steps})")


if __name__ == "__main__":
    main()
