"""Gates G1-G3 of the empirical-pool study (spec section 8), as code, so a gate is a verdict.

    .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot
    .venv/bin/python experiments/growing_bandits/empirical/gates.py collection

Both gates judge coverage against the frozen ``queue.jsonl``: every queue item in scope (the
pilot items for G2, all items for G3) needs exactly one terminal record, and an item with no
terminal record at all counts as missing -- a never-attempted item is not invisible. G2 also
requires zero watchdog relaunches during the pilot (read from the watchdog's
``relaunches.log``, which must exist).
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

import make_pools  # noqa: E402

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


RELAUNCH_LOG = "relaunches.log"
Key = tuple[str, str, int]


def _keys(frame: pd.DataFrame) -> list[Key]:
    return [(str(a), str(t), int(r)) for a, t, r in zip(frame["arm_id"], frame["task_id"], frame["replicate"],
                                                        strict=True)]


def _in(frame: pd.DataFrame, keys: set[Key]) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[[k in keys for k in _keys(frame)]]


def _coverage(g: GateResult, terminal: pd.DataFrame, scope: set[Key], queue_keys: set[Key],
              max_missing: float) -> None:
    """Missing rate over the queue items in `scope`, counting an item with no terminal record as missing."""
    term_keys = set(_keys(terminal)) if len(terminal) else set()
    outside = term_keys - queue_keys
    g.add("records_in_queue", not outside,
          f"{len(outside)} terminal records are not queue items" + (f", e.g. {sorted(outside)[:3]}" if outside else ""))
    mine = _in(terminal, scope)
    n_ok = int((mine["status"] == emp.STATUS_OK).sum()) if len(mine) else 0
    n_missing = int((mine["status"] == emp.STATUS_MISSING).sum()) if len(mine) else 0
    n_absent = len(scope - term_keys)
    rate = (n_missing + n_absent) / max(len(scope), 1)
    g.add("missing_rate", len(scope) > 0 and rate <= max_missing,
          f"{rate:.2%} missing (limit {max_missing:.0%}): {len(scope)} queue items = {n_ok} ok + "
          f"{n_missing} missing + {n_absent} with no terminal record")


def count_relaunches(path: Path, mode: str | None = None) -> int | None:
    """``RELAUNCH`` lines (scripts/watchdog_empirical_pool.py) in `path`, optionally of one mode;
    ``None`` if the log does not exist (no watchdog supervised the run)."""
    path = Path(path)
    if not path.exists():
        return None
    n = 0
    for line in path.read_text().splitlines():
        if line.startswith("RELAUNCH ") and (mode is None or f" mode={mode} " in f"{line} "):
            n += 1
    return n


def pilot_gate(attempts: pd.DataFrame, queue: list[make_pools.QueueItem], *, n_workers: int, anchor_arm: str,
               anchor_rate: float, relaunches: int | None, max_cost: float = MAX_COST_PER_EPISODE,
               max_missing: float = MAX_MISSING) -> GateResult:
    g = GateResult("G2 pilot")
    queue_keys = {(q.arm_id, q.task_id, q.replicate) for q in queue}
    scope = {(q.arm_id, q.task_id, q.replicate) for q in queue if q.pilot}
    all_terminal = emp.terminal_outcomes(attempts)
    attempts = _in(attempts, scope)
    terminal = _in(all_terminal, scope)
    cost = float(attempts["cost_usd"].fillna(0.0).astype(float).sum()) / max(len(terminal), 1)
    g.add("cost_per_episode", cost <= max_cost, f"${cost:.4f} per terminal episode (limit ${max_cost})")
    _coverage(g, all_terminal, scope, queue_keys, max_missing)
    g.add("no_watchdog_relaunch", relaunches == 0,
          "no relaunches.log: stability unverifiable (was the pilot run under the watchdog?)" if relaunches is None
          else f"{relaunches} watchdog relaunches during the pilot (must be 0)")
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


def collection_gate(attempts: pd.DataFrame, queue: list[make_pools.QueueItem], *,
                    max_missing: float = MAX_MISSING) -> GateResult:
    g = GateResult("G3 collection")
    keys = {(q.arm_id, q.task_id, q.replicate) for q in queue}
    _coverage(g, emp.terminal_outcomes(attempts), keys, keys, max_missing)
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
    ap.add_argument("--queue", type=Path, default=make_pools.DATA_DIR / "queue.jsonl")
    ap.add_argument("--relaunch-log", type=Path, default=None, help="default: <log-dir>/relaunches.log")
    args = ap.parse_args(argv)
    attempts = emp.load_attempts(sorted(args.log_dir.glob("worker_*.jsonl")))
    queue = make_pools.read_queue(args.queue)
    if args.gate == "pilot":
        relaunches = count_relaunches(args.relaunch_log or args.log_dir / RELAUNCH_LOG, mode="pilot")
        g = pilot_gate(attempts, queue, n_workers=N_WORKERS, anchor_arm="anchor_baseline", anchor_rate=ANCHOR_RATE,
                       relaunches=relaunches)
    else:
        g = collection_gate(attempts, queue)
    print(g.report())
    return 0 if g.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
