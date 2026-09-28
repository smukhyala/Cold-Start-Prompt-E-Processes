"""Gate G1 as amended (Pre-registration 9, Amendment 1): the estimator's bias and precision over seeds.

    .venv/bin/python experiments/growing_bandits/empirical/g1_multiseed.py

The original G1 check read one rehearsal seed against a +/-0.01 tolerance on the NPMLE spread. At 30 tasks
that check fails on about 4 seeds in 10 purely from sampling, so Amendment 1 judges the estimator over
12 seeds instead: for both synthetic pools, |mean error| <= 0.005 (unbiased) and RMS error <= 0.02
(precision), error = NPMLE sd - sd of the realized true prompt rates. The K*(T=200) check is unchanged and
comes from `rehearsal.py --n-tasks 30`. Writes results/growing_bandits/empirical/g1_multiseed.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rehearsal as rh  # noqa: E402
import replay  # noqa: E402

from cold_start.growing.reservoirs import BetaReservoir  # noqa: E402

N_SEEDS = 12
MAX_ABS_MEAN_ERR = 0.005
MAX_RMS_ERR = 0.02
OUT = ROOT / "results" / "growing_bandits" / "empirical" / "g1_multiseed.json"


def run(task_rates, n_seeds: int = N_SEEDS) -> dict:
    truth = {"G": BetaReservoir(159.4, 106.3, validate=False), "F": BetaReservoir.from_preset("good_common")}
    errs: dict[str, list[float]] = {"G": [], "F": []}
    for seed in range(n_seeds):
        out = rh.synthesize_outcomes(truth, seed=seed, task_rates=task_rates)
        res, _, _ = replay.estimate(out)
        for pool in ("G", "F"):
            errs[pool].append(res[(pool, "npmle")].sd() - rh.realized_sd(out, pool))
    summary = {}
    for pool, name in (("G", "flat"), ("F", "wide")):
        e = np.asarray(errs[pool])
        summary[name] = {"errors": [float(x) for x in e], "mean_err": float(e.mean()),
                         "rms_err": float(np.sqrt(np.mean(e**2))),
                         "frac_outside_0.01": float(np.mean(np.abs(e) > 0.01))}
    summary["passed"] = all(abs(summary[n]["mean_err"]) <= MAX_ABS_MEAN_ERR and summary[n]["rms_err"] <= MAX_RMS_ERR
                            for n in ("flat", "wide"))
    return summary


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    results = {"n_seeds": N_SEEDS, "criteria": {"max_abs_mean_err": MAX_ABS_MEAN_ERR, "max_rms_err": MAX_RMS_ERR},
               "tasks_30": run(rh.stratified_half(rh.TASK_RATES)), "tasks_60": run(rh.TASK_RATES)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    for label in ("tasks_30", "tasks_60"):
        r = results[label]
        print(f"{label}: " + "; ".join(f"{n} mean {r[n]['mean_err']:+.4f} rms {r[n]['rms_err']:.4f} "
                                       f"outside 0.01 {r[n]['frac_outside_0.01']:.0%}" for n in ("flat", "wide"))
              + f" -> {'PASS' if r['passed'] else 'FAIL'}")
    return 0 if results["tasks_30"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
