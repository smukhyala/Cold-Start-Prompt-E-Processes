"""Gates G1-G3 of the empirical-pool study (spec section 8), as code, so a gate is a verdict.

    .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot
    .venv/bin/python experiments/growing_bandits/empirical/gates.py collection
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cold_start.growing import empirical as emp  # noqa: E402

MAX_COST_PER_EPISODE = 0.05
MAX_MISSING = 0.05
#: The historical anchor: the hand-written `baseline` arm on Gmail, gpt-5.4-mini, all logs.
ANCHOR_RATE = 0.66
N_WORKERS = 8


@dataclass
class GateResult:
    name: str
    checks: dict[str, dict] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks.values())

    def add(self, key: str, passed: bool, detail: str) -> None:
        self.checks[key] = {"passed": bool(passed), "detail": detail}

    def report(self) -> str:
        lines = [f"{self.name}: {'PASS' if self.passed else 'FAIL'}"]
        lines += [f"  [{'ok' if c['passed'] else 'XX'}] {k}: {c['detail']}" for k, c in self.checks.items()]
        return "\n".join(lines)


def _missing_rate(terminal: pd.DataFrame) -> float:
    return float((terminal["status"] == emp.STATUS_MISSING).mean()) if len(terminal) else 0.0


def pilot_gate(attempts: pd.DataFrame, *, n_workers: int, anchor_arm: str, anchor_rate: float,
               max_cost: float = MAX_COST_PER_EPISODE, max_missing: float = MAX_MISSING) -> GateResult:
    g = GateResult("G2 pilot")
    terminal = emp.terminal_outcomes(attempts)
    cost = float(attempts["cost_usd"].fillna(0.0).astype(float).sum()) / max(len(terminal), 1)
    g.add("cost_per_episode", cost <= max_cost, f"${cost:.4f} per terminal episode (limit ${max_cost})")
    miss = _missing_rate(terminal)
    g.add("missing_rate", miss <= max_missing, f"{miss:.2%} missing (limit {max_missing:.0%})")
    ok = terminal[terminal["status"] == emp.STATUS_OK]
    workers = set(int(w) for w in ok["worker"].dropna())
    g.add("every_worker_produced", workers >= set(range(n_workers)),
          f"workers with an ok episode: {sorted(workers)} of {n_workers}")
    anchor = ok[(ok["arm_id"] == anchor_arm) & (ok["replicate"] == 0)]
    n = len(anchor)
    k = int(anchor["success"].astype(int).sum())
    lo = stats.binom.ppf(0.025, n, anchor_rate) / max(n, 1)
    hi = stats.binom.ppf(0.975, n, anchor_rate) / max(n, 1)
    rate = k / max(n, 1)
    g.add("anchor_drift", n > 0 and lo <= rate <= hi,
          f"anchor {k}/{n} = {rate:.3f}; historical {anchor_rate} gives [{lo:.3f}, {hi:.3f}]")
    return g


def collection_gate(attempts: pd.DataFrame, *, max_missing: float = MAX_MISSING) -> GateResult:
    g = GateResult("G3 collection")
    miss = _missing_rate(emp.terminal_outcomes(attempts))
    g.add("missing_rate", miss <= max_missing, f"{miss:.2%} missing (limit {max_missing:.0%})")
    return g


def rehearsal_gate(results: dict) -> GateResult:
    g = GateResult("G1 rehearsal")
    for name in ("flat", "wide"):
        r = results[name]
        err = abs(r["sd_npmle"] - r["sd_true"])
        g.add(f"{name}_npmle_sd", err <= 0.01, f"NPMLE sd {r['sd_npmle']:.4f} vs true {r['sd_true']:.4f}")
    flat = results["flat"]
    g.add("flat_raw_inflated", flat["sd_raw"] >= 1.25 * flat["sd_true"],
          f"raw sd {flat['sd_raw']:.4f} vs true {flat['sd_true']:.4f} (needs >= 1.25x)")
    grid = list(results["k_grid"])
    step = abs(grid.index(results["k_star_est"]) - grid.index(results["k_star_true"]))
    g.add("k_star_within_one_step", step <= 1,
          f"K*(T=200) estimated {results['k_star_est']} vs true {results['k_star_true']} ({step} grid steps)")
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gate", choices=["pilot", "collection"])
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "empirical_pool")
    args = ap.parse_args(argv)
    attempts = emp.load_attempts(sorted(args.log_dir.glob("worker_*.jsonl")))
    if args.gate == "pilot":
        g = pilot_gate(attempts, n_workers=N_WORKERS, anchor_arm="anchor_baseline", anchor_rate=ANCHOR_RATE)
    else:
        g = collection_gate(attempts)
    print(g.report())
    return 0 if g.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
