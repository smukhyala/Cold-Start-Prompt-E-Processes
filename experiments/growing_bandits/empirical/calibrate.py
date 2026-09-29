"""Value-of-search calibration (spec 2026-09-28-prompt-heterogeneity, Task 5).

Runs the same replay machinery as `replay.py`/`describe.py` on *synthetic* prompt pools of
known spread, rather than the real collected prompts, so the pre-registration can pick
``tau_flat`` (the Beta-pool spread below which adaptive search cannot beat a fixed schedule)
from a number that is known to be flat, instead of assuming one.

Two pool families, both discretized to the empirical grid so a cell runs the same
``EmpiricalReservoir`` machinery the real pools do:

* ``beta_sd<sd>`` -- ``Beta`` reservoirs at a fixed level (`LEVEL`) over a grid of spreads
  (`SPREADS`), the "how much heterogeneity is there" axis.
* ``tail_f<frac>_d<delta>`` -- a "flat bulk + rare great arm" mixture: almost all mass at
  `LEVEL` with `BULK_SD` spread, plus a `tail_frac` chance of an arm drawn `tail_delta`
  above `LEVEL` (also `BULK_SD` spread) -- the family the level rule (a single number) can't
  see coming.

    .venv/bin/python experiments/growing_bandits/empirical/calibrate.py --workers 12

writes ``results/growing_bandits/heterogeneity/calibration.csv`` and runs the deployment's
`always_search` / `p3_star` / `fixed_K8` policies into its own
``results/growing_bandits/heterogeneity/cal_run/`` (never the real deployment tree).

Seeds: `CAL_SEED_BASE` (5e10) lies above every seed range already in use -- Pre-registration
9's (4.0e8 .. 4.41e9), the heterogeneity study's (1e10 .. ~2.0e10) and the corpus's/smoke's
(<= 21M) -- per `study.HETEROGENEITY`'s module docstring, which already reserves this range.
Cells are Study-free (no pool pairing, no bootstrap): one seed per (pool, horizon),
``CAL_SEED_BASE + 1_000 * (pool_index * 10_000 + horizon)``, mirroring `replay.seed_for`'s
stride scheme without needing a `study.Study` object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import describe  # noqa: E402
import replay  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402
from cold_start.growing.reservoirs import BetaReservoir, MixtureReservoir, Reservoir  # noqa: E402

LEVEL = 0.6
SPREADS: tuple[float, ...] = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15)
TAIL_FRACS: tuple[float, ...] = (0.01, 0.02, 0.05)
TAIL_DELTAS: tuple[float, ...] = (0.10, 0.20, 0.30)
BULK_SD = 0.02
#: Above every seed range already reserved: Pre-registration 9's (4.0e8 .. 4.41e9), the
#: heterogeneity study's (1e10 .. ~2.0e10) and the corpus/smoke ranges (<= 21M); see the
#: module docstring and `study`'s.
CAL_SEED_BASE = 50_000_000_000
POOL_STRIDE = 10_000
SEED_STRIDE = 1_000

HORIZONS: tuple[int, ...] = replay.PRIMARY_HORIZONS  # (50, 100, 200)
K_GRID: tuple[int, ...] = tuple(int(k) for k in replay.kse.DEFAULT_K_GRID)
#: A pool's probability of an arm strictly better than the flat bulk's mean + 3 sd
#: (i.e. clearly not just bulk noise): the same threshold for every pool, beta or mixture,
#: so it is comparable across the whole table.
UPPER_TAIL_THRESHOLD = LEVEL + 3.0 * BULK_SD

CAL_RUN_SUBDIR = "cal_run"
CALIBRATION_CSV = "calibration.csv"
DEPLOY_PARAMS_DIR = ROOT / "results" / "growing_bandits" / "deploy"

COLUMNS: tuple[str, ...] = (
    "pool_id", "family", "spread", "tail_frac", "tail_delta", "true_sd", "upper_tail_mass",
    "horizon", "regret_range", "k_star", "gap_fixed_K8", "gap_p3_star", "gap_always_search",
)


def _beta(mean: float, sd: float) -> BetaReservoir:
    s = mean * (1 - mean) / sd**2 - 1
    return BetaReservoir(mean * s, (1 - mean) * s, validate=False)


def calibration_pools() -> dict[str, Reservoir]:
    """The named grid of synthetic pools: Beta spreads, then tail-mixture (frac, delta)."""
    pools: dict[str, Reservoir] = {f"beta_sd{sd:g}": _beta(LEVEL, sd) for sd in SPREADS}
    for f in TAIL_FRACS:
        for d in TAIL_DELTAS:
            pools[f"tail_f{f:g}_d{d:g}"] = MixtureReservoir(
                [_beta(LEVEL, BULK_SD), _beta(LEVEL + d, BULK_SD)], [1 - f, f], validate=False)
    return pools


def _pool_params(pool_id: str, pool: Reservoir) -> dict:
    """Which of ``spread``/``tail_frac``/``tail_delta`` a pool id carries, NaN otherwise."""
    if pool.family == "beta":
        sd = float(pool_id.removeprefix("beta_sd"))
        return {"spread": sd, "tail_frac": float("nan"), "tail_delta": float("nan")}
    rest = pool_id.removeprefix("tail_f")
    frac_s, delta_s = rest.split("_d")
    return {"spread": float("nan"), "tail_frac": float(frac_s), "tail_delta": float(delta_s)}


def _cal_seed(pool_index: int, horizon: int) -> int:
    return CAL_SEED_BASE + SEED_STRIDE * (pool_index * POOL_STRIDE + int(horizon))


def tau_flat_from_calibration(frame: pd.DataFrame, horizon: int = 200, target: float = 0.005) -> float:
    """The Beta-pool ``true_sd`` at which ``regret_range`` linearly crosses `target`, at `horizon`.

    Restricted to ``family == "beta"`` (the level-fixed spread axis `tau_flat` is defined
    over) and sorted by `true_sd`; interpolates between the first adjacent pair that
    straddles `target` (regret range is increasing in spread, but this does not assume
    monotonicity -- it takes the first crossing in spread order, which is what "the spread
    below which pools are flat" means operationally). Raises if `horizon` is absent from the
    frame or the target is never crossed within the grid.
    """
    beta = frame[frame["family"] == "beta"]
    sub = beta[beta["horizon"] == int(horizon)].sort_values("true_sd")
    if sub.empty:
        raise ValueError(f"no beta-pool rows at horizon={horizon}")
    sds = sub["true_sd"].to_numpy(dtype=float)
    ranges = sub["regret_range"].to_numpy(dtype=float)
    for i in range(len(sds) - 1):
        lo, hi = ranges[i], ranges[i + 1]
        if (lo <= target <= hi) or (hi <= target <= lo):
            if hi == lo:
                return float(sds[i])
            frac = (target - lo) / (hi - lo)
            return float(sds[i] + frac * (sds[i + 1] - sds[i]))
    raise ValueError(
        f"regret_range never crosses target={target} over true_sd in "
        f"[{sds.min():g}, {sds.max():g}] at horizon={horizon}"
    )


def calibrate(workers: int, m: int = 1000, out_dir: Path | str = ROOT / "results" / "growing_bandits" / "heterogeneity"
              ) -> pd.DataFrame:
    """Run the calibration grid and write ``<out_dir>/calibration.csv``.

    ``m`` is the per-cell replicate count (the K-grid sweep and the always_search/p3_star/
    fixed_K8 deployment share it). The deployment runs into ``<out_dir>/cal_run/`` with
    ``--baseline-params``/``--thresholds`` pointed at the real deployment's tuned constants
    (read-only; never written to), so it needs no copy step and never touches
    ``results/growing_bandits/deploy/``.
    """
    out_dir = Path(out_dir)
    pools = calibration_pools()
    cells: list[CellSpec] = []
    meta_rows: list[dict] = []
    for pool_index, (pool_id, pool) in enumerate(pools.items()):
        weights = emp.grid_masses(pool)
        res = EmpiricalReservoir(emp.GRID, weights, label=pool_id, validate=False)
        u = (np.arange(20_001) + 0.5) / 20_001
        true_sd = float(np.std(res.sample_from_uniforms(u)))
        upper_tail_mass = float(res.tail_prob(UPPER_TAIL_THRESHOLD))
        meta_rows.append({"pool_id": pool_id, "family": pool.family, "true_sd": true_sd,
                          "upper_tail_mass": upper_tail_mass, **_pool_params(pool_id, pool)})
        for T in HORIZONS:
            cells.append(CellSpec(
                env_id=f"cal_{pool_id}", env_spec=res.to_spec(), horizon=int(T), cap=int(T),
                base_seed=_cal_seed(pool_index, int(T)), n_replicates=int(m),
            ))
    meta = pd.DataFrame(meta_rows)

    kgrid_frame = replay.kgrid(cells, workers=workers, k_grid=K_GRID)
    kstar = describe.k_star_table(kgrid_frame)
    kstar = kstar.assign(pool_id=kstar["env_id"].str.removeprefix("cal_"))

    k8 = kgrid_frame[kgrid_frame["K"] == 8][["env_id", "horizon", "regret"]].rename(
        columns={"regret": "regret_fixed_k8"})

    run_dir = out_dir / CAL_RUN_SUBDIR
    rd.main([
        "--test", "cal", "--workers", str(workers), "--out-dir", str(run_dir),
        "--baseline-params", str(DEPLOY_PARAMS_DIR / "baseline_params.json"),
        "--thresholds", str(DEPLOY_PARAMS_DIR / "thresholds.json"),
    ], cells=cells)
    summary = pd.read_csv(run_dir / "summary_cal.csv")
    prim = summary[summary["recommender"] == PRIMARY_RECOMMENDER]
    wide = prim.pivot_table(index=["env_id", "horizon"], values="regret", columns="policy").reset_index()
    wide.columns = [str(c) for c in wide.columns]

    frame = kstar[["env_id", "pool_id", "horizon", "regret_range", "k_star", "regret_at_k_star"]].merge(
        k8, on=["env_id", "horizon"], how="left").merge(wide, on=["env_id", "horizon"], how="left")
    frame = frame.merge(meta, on="pool_id", how="left")
    frame["gap_fixed_K8"] = frame["regret_fixed_k8"] - frame["regret_at_k_star"]
    frame["gap_p3_star"] = frame["p3_star"] - frame["regret_at_k_star"]
    frame["gap_always_search"] = frame["always_search"] - frame["regret_at_k_star"]
    frame = frame[list(COLUMNS)].sort_values(["pool_id", "horizon"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / CALIBRATION_CSV, index=False)
    return frame


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--m", type=int, default=1000, help="replicates per cell")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "growing_bandits" / "heterogeneity")
    args = ap.parse_args(argv)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    frame = calibrate(workers=args.workers, m=args.m, out_dir=args.out_dir)
    logging.getLogger("empirical.calibrate").info("wrote %d rows to %s", len(frame), args.out_dir / CALIBRATION_CSV)


if __name__ == "__main__":
    main()
