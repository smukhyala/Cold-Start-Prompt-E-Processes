"""Gates G1-G3 of the empirical-pool study (spec section 8), as code, so a gate is a verdict.

    .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot
    .venv/bin/python experiments/growing_bandits/empirical/gates.py collection
    .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot --profile gitlab
    .venv/bin/python experiments/growing_bandits/empirical/gates.py collection --profile {gitlab,gmail,bridge}

Both gates judge coverage against the frozen ``queue.jsonl``: every queue item in scope (the
pilot items for G2, all items for G3) needs exactly one terminal record, and an item with no
terminal record at all counts as missing -- a never-attempted item is not invisible. G2 also
requires zero watchdog relaunches during the pilot (read from the watchdog's
``relaunches.log``, which must exist).

``--profile`` (default ``prereg9``, `GATE_PROFILES`) picks the log dir and queue (collect.py's
`PROFILES`) and the thresholds. Heterogeneity study (spec 4.5): the pilot is GitLab only --
cost <= $0.25/episode and anchor order ``GL_anchor_oracle`` > ``GL_anchor_explorer`` on the
pilot items, plus the Pre-reg 9 coverage/relaunch/worker checks. ``GL_anchor_baseline`` has no
historical rate under this agent, so it gets no drift test: its pilot rate is reported, not
gated. G3 checks missing <= 5% per pool (cell) as well as overall; the gmail profile's G3 also
tests ``GM_anchor_baseline`` for drift against Pre-reg 9's 0.66 (the gmail profile has no pilot).
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

import collect  # noqa: E402
import make_pools  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402

#: Fix round 1, item 6 (ruling): the original $0.05 encoded a mistaken budget estimate: the
#: pilot's real cost ran ~$0.104-0.11/episode. $0.15 gives real episodes headroom while still
#: catching a genuine cost blowup.
MAX_COST_PER_EPISODE = 0.15
MAX_MISSING = 0.05
#: The historical anchor: the hand-written `baseline` arm on Gmail, gpt-5.4-mini, all logs.
ANCHOR_RATE = 0.66
N_WORKERS = 8
#: Heterogeneity spec 4.5: GitLab's pilot cost limit.
MAX_COST_PER_EPISODE_GITLAB = 0.25

#: Per-profile gate arguments. ``pilot`` is None where the queue has no pilot items.
GATE_PROFILES: dict[str, dict] = {
    "prereg9": {
        "pilot": {"anchor_arm": make_pools.ANCHOR_ARM_ID, "anchor_rate": ANCHOR_RATE},
        "collection": {},
    },
    "gitlab": {
        "pilot": {"anchor_arm": "GL_anchor_baseline", "anchor_rate": None, "max_cost": MAX_COST_PER_EPISODE_GITLAB,
                  "anchor_order": ("GL_anchor_oracle", "GL_anchor_explorer")},
        "collection": {"per_pool": True},
    },
    "gmail": {
        "pilot": None,
        "collection": {"per_pool": True, "anchor_arm": "GM_anchor_baseline", "anchor_rate": ANCHOR_RATE},
    },
    "bridge": {
        "pilot": None,
        "collection": {"per_pool": True},
    },
}


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


def _missing_check(g: GateResult, key: str, terminal: pd.DataFrame, term_keys: set[Key], scope: set[Key],
                   max_missing: float) -> None:
    mine = _in(terminal, scope)
    n_ok = int((mine["status"] == emp.STATUS_OK).sum()) if len(mine) else 0
    n_missing = int((mine["status"] == emp.STATUS_MISSING).sum()) if len(mine) else 0
    n_absent = len(scope - term_keys)
    rate = (n_missing + n_absent) / max(len(scope), 1)
    g.add(key, len(scope) > 0 and rate <= max_missing,
          f"{rate:.2%} missing (limit {max_missing:.0%}): {len(scope)} queue items = {n_ok} ok + "
          f"{n_missing} missing + {n_absent} with no terminal record")


def _coverage(g: GateResult, terminal: pd.DataFrame, scope: set[Key], queue_keys: set[Key],
              max_missing: float, pools: dict[str, set[Key]] | None = None) -> None:
    """Missing rate over the queue items in `scope`, counting an item with no terminal record as
    missing; with `pools`, also one ``missing_rate[<pool>]`` check per pool's items."""
    term_keys = set(_keys(terminal)) if len(terminal) else set()
    outside = term_keys - queue_keys
    g.add("records_in_queue", not outside,
          f"{len(outside)} terminal records are not queue items" + (f", e.g. {sorted(outside)[:3]}" if outside else ""))
    _missing_check(g, "missing_rate", terminal, term_keys, scope, max_missing)
    for pool, keys in sorted((pools or {}).items()):
        _missing_check(g, f"missing_rate[{pool}]", terminal, term_keys, keys & scope, max_missing)


def _rate(ok: pd.DataFrame, arm: str) -> tuple[int, int]:
    """(successes, ok episodes) of `arm`'s replicate-0 records in `ok`."""
    rows = ok[(ok["arm_id"] == arm) & (ok["replicate"] == 0)]
    return int(rows["success"].astype(int).sum()), len(rows)


def _anchor_drift(g: GateResult, ok: pd.DataFrame, anchor_arm: str, anchor_rate: float) -> None:
    k, n = _rate(ok, anchor_arm)
    lo = stats.binom.ppf(0.025, n, anchor_rate) / max(n, 1)
    hi = stats.binom.ppf(0.975, n, anchor_rate) / max(n, 1)
    rate = k / max(n, 1)
    g.add("anchor_drift", n > 0 and lo <= rate <= hi,
          f"anchor {k}/{n} = {rate:.3f}; historical {anchor_rate} gives [{lo:.3f}, {hi:.3f}]")


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


def pilot_gate(attempts: pd.DataFrame, queue: list[make_pools.QueueItem], *, n_workers: int,
               anchor_arm: str | None, anchor_rate: float | None, relaunches: int | None,
               max_cost: float = MAX_COST_PER_EPISODE, max_missing: float = MAX_MISSING,
               anchor_order: tuple[str, str] | None = None) -> GateResult:
    """G2. `anchor_rate` None: `anchor_arm` has no historical rate, so its pilot rate is
    reported (``anchor_baseline_rate``, never failing) instead of drift-tested. `anchor_order`
    ``(a, b)``: adds ``anchor_order``, passing iff a's success rate > b's on the pilot items."""
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
    if anchor_arm is not None and anchor_rate is not None:
        _anchor_drift(g, ok, anchor_arm, anchor_rate)
    elif anchor_arm is not None:
        k, n = _rate(ok, anchor_arm)
        g.add("anchor_baseline_rate", True,
              f"{anchor_arm} {k}/{n} = {k / max(n, 1):.3f} on the pilot items (no historical rate under this "
              "agent: reported, not gated)")
    if anchor_order is not None:
        first, second = anchor_order
        k1, n1 = _rate(ok, first)
        k2, n2 = _rate(ok, second)
        r1, r2 = k1 / max(n1, 1), k2 / max(n2, 1)
        g.add("anchor_order", n1 > 0 and n2 > 0 and r1 > r2,
              f"{first} {k1}/{n1} = {r1:.3f} vs {second} {k2}/{n2} = {r2:.3f} (first must be higher)")
    return g


def collection_gate(attempts: pd.DataFrame, queue: list[make_pools.QueueItem], *,
                    max_missing: float = MAX_MISSING, per_pool: bool = False, anchor_arm: str | None = None,
                    anchor_rate: float | None = None) -> GateResult:
    """G3. `per_pool`: missing <= `max_missing` in every pool (cell) too, not only overall.
    `anchor_arm` + `anchor_rate`: drift-test that anchor over all its replicate-0 items."""
    g = GateResult("G3 collection")
    keys = {(q.arm_id, q.task_id, q.replicate) for q in queue}
    pools: dict[str, set[Key]] | None = None
    if per_pool:
        pools = {}
        for q in queue:
            pools.setdefault(q.pool, set()).add((q.arm_id, q.task_id, q.replicate))
    terminal = emp.terminal_outcomes(attempts)
    _coverage(g, terminal, keys, keys, max_missing, pools)
    if anchor_arm is not None and anchor_rate is not None:
        ok = _in(terminal, keys)
        _anchor_drift(g, ok[ok["status"] == emp.STATUS_OK], anchor_arm, anchor_rate)
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


def cli_paths(profile: str, log_dir: Path | None, queue: Path | None) -> tuple[Path, Path]:
    """(log dir, queue path) for `profile`, unless given explicitly."""
    spec = collect.PROFILES[profile]
    return (log_dir if log_dir is not None else spec["log_dir"],
            queue if queue is not None else spec["data_dir"] / spec["queue"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["pilot", "collection"])
    ap.add_argument("--profile", choices=sorted(GATE_PROFILES), default=collect.DEFAULT_PROFILE)
    ap.add_argument("--log-dir", type=Path, default=None, help="default: the profile's log dir")
    ap.add_argument("--queue", type=Path, default=None, help="default: the profile's queue.jsonl")
    ap.add_argument("--relaunch-log", type=Path, default=None, help="default: <log-dir>/relaunches.log")
    args = ap.parse_args(argv)
    cfg = GATE_PROFILES[args.profile]
    if args.gate == "pilot" and cfg["pilot"] is None:
        print(f"profile {args.profile} has no pilot items (the pilot is GitLab only); run the collection gate")
        return 2
    log_dir, queue_path = cli_paths(args.profile, args.log_dir, args.queue)
    attempts = emp.load_attempts(sorted(log_dir.glob("worker_*.jsonl")))
    queue = make_pools.read_queue(queue_path)
    if args.gate == "pilot":
        relaunches = count_relaunches(args.relaunch_log or log_dir / RELAUNCH_LOG, mode="pilot")
        g = pilot_gate(attempts, queue, n_workers=N_WORKERS, relaunches=relaunches, **cfg["pilot"])
    else:
        g = collection_gate(attempts, queue, **cfg["collection"])
    print(g.report())
    return 0 if g.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
