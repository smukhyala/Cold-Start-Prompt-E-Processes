"""Gate G1: the whole estimation-and-replay pipeline on synthetic data with a known answer.

    .venv/bin/python experiments/growing_bandits/empirical/rehearsal.py --workers 12

Pool G's truth is flat (level 0.6, sd 0.03, the hand-written arms' measured spread); pool
F's is the corpus's `beta_good_common` (sd 0.16). Outcomes are generated per (prompt, task)
from ``p_ij = expit(a_i + d_j)``, where ``d_j`` are the 60 logged Gmail task difficulties
and ``a_i`` is solved so prompt i's mean over tasks is exactly its true rate. Passing means
the NPMLE recovers both spreads, the raw spread is visibly inflated on the flat pool, and
the pipeline locates K*(T = 200) on the wide pool to within one K-grid step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import describe  # noqa: E402
import gates  # noqa: E402
import k_star_envelope as kse  # noqa: E402
import replay  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402
from cold_start.growing.reservoirs import BetaReservoir, Reservoir  # noqa: E402

log = logging.getLogger("empirical.rehearsal")

#: Per-task success rates of the 12 hand-written arms on Gmail (1,631 logged episodes, 2026-09-26).
TASK_RATES: tuple[float, ...] = (
    *[0.0] * 6, *[0.04] * 3, 0.07, *[0.11] * 2, *[0.15] * 2, *[0.19] * 2, 0.26, 0.30, *[0.33] * 3, 0.37,
    *[0.41] * 3, *[0.52] * 5, *[0.63] * 2, 0.67, 0.87, *[0.93] * 4, *[0.96] * 8, 0.97, *[1.0] * 13,
)
expit = special.expit
OUT = ROOT / "results" / "growing_bandits" / "empirical" / "rehearsal.json"
REHEARSAL_T = 200


def stratified_half(rates: tuple[float, ...]) -> tuple[float, ...]:
    """Every other element of `rates` (already sorted ascending), starting at index 0.

    Amendment 1: 30 of the 60 logged Gmail task rates, spanning the same range as the full 60
    (`rates` is never itself modified -- this only picks a subset of it).
    """
    return tuple(rates[0::2])


def task_offsets(rates) -> np.ndarray:
    d = special.logit(np.clip(np.asarray(rates, dtype=float), 0.02, 0.98))
    return d - d.mean()


def solve_level(mu: float, d: np.ndarray) -> float:
    return float(optimize.brentq(lambda a: float(np.mean(expit(a + d))) - mu, -30.0, 30.0, xtol=1e-12))


def synthesize_outcomes(truth: dict[str, Reservoir], *, n_prompts: int = 50, n_replicates: int = 300,
                        seed: int = 0, task_rates: tuple[float, ...] = TASK_RATES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = task_offsets(task_rates)
    rows, probs, mu_true_of = [], {}, {}
    for pool in sorted(truth):
        mus = truth[pool].sample(rng, n_prompts)
        for i, mu in enumerate(mus):
            mu_true = float(np.clip(mu, 0.005, 0.995))
            p = expit(solve_level(mu_true, d) + d)
            arm = f"{pool}_{i:02d}"
            mu_true_of[(pool, arm)] = mu_true
            for j, pj in enumerate(p):
                probs[(pool, arm, f"t{j:02d}")] = pj
                rows.append({"pool": pool, "arm_id": arm, "task_id": f"t{j:02d}", "replicate": 0, "attempt": 1,
                             "status": emp.STATUS_OK, "success": int(rng.random() < pj), "cost_usd": 0.0,
                             "mu_true": mu_true})
    keys = sorted(probs)
    for k in rng.choice(len(keys), size=n_replicates, replace=False):
        pool, arm, task = keys[int(k)]
        rows.append({"pool": pool, "arm_id": arm, "task_id": task, "replicate": 1, "attempt": 1,
                     "status": emp.STATUS_OK, "success": int(rng.random() < probs[keys[int(k)]]), "cost_usd": 0.0,
                     "mu_true": mu_true_of[(pool, arm)]})
    return pd.DataFrame(rows)


def _sd(res: Reservoir) -> float:
    u = (np.arange(20_001) + 0.5) / 20_001
    return float(np.std(res.sample_from_uniforms(u)))


def realized_sd(outcomes: pd.DataFrame, pool: str) -> float:
    """SD (ddof=0) of `pool`'s realized per-prompt true rates (one `mu_true` per arm).

    G1's NPMLE-recovery check must compare against what these ``n_prompts`` draws
    actually landed on, not the reservoir's population SD: on the wide pool the sample
    SD of 50 iid draws misses the population SD by more than 0.01 about half the time,
    which would make the +-0.01 gate a coin flip independent of estimator quality.
    """
    per_arm = outcomes.loc[outcomes["pool"] == pool].groupby("arm_id")["mu_true"].first()
    return float(np.std(per_arm.to_numpy(dtype=float), ddof=0))


def run_rehearsal(seed: int = 20260926, workers: int = 12, m: int = 1000, n_tasks: int = 30) -> dict:
    if n_tasks == 30:
        task_rates = stratified_half(TASK_RATES)
    elif n_tasks == 60:
        task_rates = TASK_RATES
    else:
        raise ValueError(f"n_tasks must be 30 or 60 (stratified halves of the logged 60), got {n_tasks}")
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    outcomes = synthesize_outcomes(truth, seed=seed, task_rates=task_rates)
    reservoirs, noise, _ = replay.estimate(outcomes)
    results: dict = {"noise": noise, "k_grid": [K for K in kse.DEFAULT_K_GRID if K <= REHEARSAL_T]}
    for name, pool in (("flat", "G"), ("wide", "F")):
        results[name] = {"sd_true": realized_sd(outcomes, pool), "sd_population": _sd(truth[pool]),
                         "sd_npmle": reservoirs[(pool, "npmle")].sd(),
                         "sd_raw": reservoirs[(pool, "raw")].sd(), "mean_true": float(truth[pool].mean()),
                         "mean_npmle": reservoirs[(pool, "npmle")].mean()}
    true_wide = EmpiricalReservoir(emp.GRID, emp.grid_masses(truth["F"]), label="F_truth")
    cells = [replay.make_emp_cell("F", "truth", REHEARSAL_T, true_wide, m),
             replay.make_emp_cell("F", "npmle", REHEARSAL_T, reservoirs[("F", "npmle")], m)]
    kstar = describe.k_star_table(replay.kgrid(cells, workers=workers))
    results["k_star_true"] = int(kstar[kstar["variant"] == "truth"]["k_star"].iloc[0])
    results["k_star_est"] = int(kstar[kstar["variant"] == "npmle"]["k_star"].iloc[0])
    gate = gates.rehearsal_gate(results)
    results["gate"] = {"passed": gate.passed, "checks": gate.checks}
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--n-tasks", type=int, default=30,
                    help="amendment 1: 30 (a stratified half of the logged 60) or 60 (the full bank)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = run_rehearsal(workers=args.workers, m=args.replicates, n_tasks=args.n_tasks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=float))
    gate = gates.rehearsal_gate(results)
    print(gate.report())
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
