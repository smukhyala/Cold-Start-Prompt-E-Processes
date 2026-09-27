"""Replay the growing-bandit policies on the real prompt pools (spec sections 4-5).

    .venv/bin/python experiments/growing_bandits/empirical/replay.py estimate
    .venv/bin/python experiments/growing_bandits/empirical/replay.py point --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py boot  --workers 12

``estimate`` reads the collector's logs and writes the six reservoirs (G/F x npmle/raw/
parametric), the noise model and the parametric fit table under
``data/empirical_pool/reservoirs/``. ``point`` deploys `TEST_POLICIES["emp"]` on every
(pool, variant, T) at M = 1000; ``kgrid`` runs fixed-K over the K-grid on the same cells;
``boot`` re-estimates the NPMLE on B = 200 prompt resamples and deploys the four contrast
policies at the primary horizons, M = 250.

Seeds: every (pool, T) has one seed shared by its three variants (CRN across variants);
every bootstrap replicate has its own seeds and its own env id, so the per-cell
comparator cache (`run_deployment.prepare_cell_constants`) can never mix replicates.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import k_star_envelope as kse  # noqa: E402
import run_deployment as rd  # noqa: E402

import cold_start.growing.empirical_reservoir  # noqa: E402,F401  (registers "empirical" in workers)
from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

log = logging.getLogger("empirical.replay")

EMP_SEED_BASE = 400_000_000
EMP_CELL_STRIDE = 1_000
POOLS: tuple[str, ...] = ("G", "F")
VARIANTS: tuple[str, ...] = ("npmle", "raw", "parametric")
PRIMARY_HORIZONS: tuple[int, ...] = (50, 100, 200)
ALL_HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)
EMP_REPLICATES = 1000
BOOT_REPLICATES = 250
N_BOOT = 200
BOOT_SEED = 20260926
CONTRAST_POLICIES: tuple[str, ...] = ("p3_star", "fixed_K_star", "level_star", "phi_k4")
NOISE_BOOT = 1000

DATA_DIR = ROOT / "data" / "empirical_pool"
LOG_DIR = ROOT / "logs" / "empirical_pool"
RES_DIR = DATA_DIR / "reservoirs"


def env_id(pool: str, variant: str, boot: int | None = None) -> str:
    return f"emp_{pool}_{variant}" + ("" if boot is None else f"_b{boot:03d}")


#: Component strides for `seed_for`'s index, chosen so a pool's block comfortably
#: exceeds any horizon (including a unit test's ad-hoc small T outside `ALL_HORIZONS`,
#: e.g. `test_kgrid_runs_on_empirical_cells`'s T=20) and a bootstrap's block comfortably
#: exceeds a pool's: `POOL_STRIDE` > max horizon, `BOOT_STRIDE` = `POOL_STRIDE * len(POOLS)`.
POOL_STRIDE = 10_000
BOOT_STRIDE = POOL_STRIDE * len(POOLS)


def seed_for(pool: str, horizon: int, boot: int | None = None) -> int:
    b = 0 if boot is None else boot + 1
    index = b * BOOT_STRIDE + POOLS.index(pool) * POOL_STRIDE + int(horizon)
    return EMP_SEED_BASE + EMP_CELL_STRIDE * index


def make_emp_cell(pool: str, variant: str, horizon: int, reservoir: EmpiricalReservoir, n_replicates: int,
                  boot: int | None = None) -> CellSpec:
    return CellSpec(env_id=env_id(pool, variant, boot), env_spec=reservoir.to_spec(), horizon=int(horizon),
                    cap=int(horizon), base_seed=seed_for(pool, horizon, boot), n_replicates=int(n_replicates))


# ---- estimation ---------------------------------------------------------------------


def _pair_sq_diffs(outcomes: pd.DataFrame, pool: str) -> np.ndarray:
    ok = outcomes[(outcomes["status"] == emp.STATUS_OK) & (outcomes["pool"] == pool)]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success",
                          aggfunc="first")[[0, 1]].dropna()
    return (wide[0].to_numpy(float) - wide[1].to_numpy(float)) ** 2 / 2.0


def noise_model(outcomes: pd.DataFrame, *, per_pool: bool | None = None, seed: int = 0) -> dict:
    """v per pool; pooled unless the pools differ by more than the bootstrap SE of their difference.

    With `per_pool` given (the prompt bootstrap reuses the full-sample decision) the SE is not
    needed and is not computed.
    """
    pooled, n_pairs = emp.within_cell_variance(outcomes, POOLS)
    by_pool = {p: emp.within_cell_variance(outcomes, [p])[0] for p in POOLS}
    se_diff = float("nan")
    if per_pool is None:
        sq = {p: _pair_sq_diffs(outcomes, p) for p in POOLS}
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(NOISE_BOOT):
            v_b = [float(sq[p][rng.integers(0, sq[p].size, sq[p].size)].mean()) for p in POOLS]
            diffs.append(v_b[0] - v_b[1])
        se_diff = float(np.std(diffs, ddof=1))
        per_pool = abs(by_pool["G"] - by_pool["F"]) > se_diff
    v = dict(by_pool) if per_pool else {p: pooled for p in POOLS}
    return {"v": v, "pooled": pooled, "by_pool": by_pool, "per_pool": bool(per_pool), "se_diff": se_diff,
            "n_pairs": n_pairs}


def estimate(outcomes: pd.DataFrame) -> tuple[dict[tuple[str, str], EmpiricalReservoir], dict, pd.DataFrame]:
    noise = noise_model(outcomes)
    reservoirs: dict[tuple[str, str], EmpiricalReservoir] = {}
    fit_tables = []
    for pool in POOLS:
        scores = emp.prompt_scores(outcomes, pool)
        v = noise["v"][pool]
        reservoirs[(pool, "npmle")] = emp.npmle_reservoir(scores, v, f"{pool}_npmle")
        reservoirs[(pool, "raw")] = emp.raw_reservoir(scores, f"{pool}_raw")
        res, fits = emp.fit_parametric(scores, v, f"{pool}_parametric")
        reservoirs[(pool, "parametric")] = res
        fit_tables.append(fits.assign(pool=pool))
    return reservoirs, noise, pd.concat(fit_tables, ignore_index=True)


def _resample_pool(outcomes: pd.DataFrame, pool: str, rng: np.random.Generator) -> pd.DataFrame:
    sub = outcomes[outcomes["pool"] == pool]
    arms = sorted(sub["arm_id"].unique())
    picks = rng.integers(0, len(arms), len(arms))
    parts = [sub[sub["arm_id"] == arms[int(i)]].assign(arm_id=f"{arms[int(i)]}#{k}") for k, i in enumerate(picks)]
    return pd.concat(parts, ignore_index=True)


def bootstrap_reservoirs(outcomes: pd.DataFrame, noise: dict, *, n_boot: int = N_BOOT,
                         seed: int = BOOT_SEED) -> list[tuple[int, str, EmpiricalReservoir]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, str, EmpiricalReservoir]] = []
    for b in range(n_boot):
        resampled = pd.concat([_resample_pool(outcomes, p, rng) for p in POOLS], ignore_index=True)
        nb = noise_model(resampled, per_pool=noise["per_pool"], seed=b)
        for pool in POOLS:
            scores = emp.prompt_scores(resampled, pool)
            out.append((b, pool, emp.npmle_reservoir(scores, nb["v"][pool], f"{pool}_npmle_b{b:03d}")))
    return out


def save_reservoir(res: EmpiricalReservoir, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.to_spec()))


def load_reservoir(path: Path) -> EmpiricalReservoir:
    return build_reservoir(json.loads(path.read_text()))


# ---- runs -----------------------------------------------------------------------------


def _kgrid_item(item: kse.Item) -> dict:
    return kse.run_item(item)


def kgrid(cells: list[CellSpec], *, workers: int, k_grid: tuple[int, ...] = kse.DEFAULT_K_GRID) -> pd.DataFrame:
    items = [kse.Item(split="emp", spec=c, K=int(K)) for c in cells for K in k_grid if int(K) <= c.horizon]
    if workers <= 1:
        rows = [_kgrid_item(i) for i in items]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_kgrid_item, items, chunksize=1)
    frame = pd.DataFrame(rows)
    parts = frame["env_id"].str.split("_", expand=True)
    return frame.assign(pool=parts[1], variant=parts[2])


def point_cells() -> list[CellSpec]:
    return [make_emp_cell(p, v, T, load_reservoir(RES_DIR / f"{p}_{v}.json"), EMP_REPLICATES)
            for p in POOLS for v in VARIANTS for T in ALL_HORIZONS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["estimate", "point", "kgrid", "boot"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.stage == "estimate":
        outcomes = emp.terminal_outcomes(emp.load_attempts(sorted(LOG_DIR.glob("worker_*.jsonl"))))
        reservoirs, noise, fits = estimate(outcomes)
        for (pool, variant), res in reservoirs.items():
            save_reservoir(res, RES_DIR / f"{pool}_{variant}.json")
        (RES_DIR / "noise.json").write_text(json.dumps(noise, indent=2))
        fits.to_csv(RES_DIR / "parametric_fits.csv", index=False)
        for (pool, variant), res in reservoirs.items():
            log.info("%s %-10s mean %.4f sd %.4f atoms %d %s", pool, variant, res.mean(), res.sd(),
                     res.atoms.size, res.validation_error or "")
    elif args.stage == "point":
        rd.main(["--test", "emp", "--workers", str(args.workers), "--out-dir", str(args.out_dir)],
                cells=point_cells())
    elif args.stage == "kgrid":
        frame = kgrid(point_cells(), workers=args.workers)
        out = args.out_dir / "tables" / "emp_kgrid.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(frame))
    else:
        outcomes = emp.terminal_outcomes(emp.load_attempts(sorted(LOG_DIR.glob("worker_*.jsonl"))))
        noise = json.loads((RES_DIR / "noise.json").read_text())
        cells = []
        for b, pool, res in bootstrap_reservoirs(outcomes, noise):
            save_reservoir(res, RES_DIR / "boot" / f"{pool}_npmle_b{b:03d}.json")
            cells.extend(make_emp_cell(pool, "npmle", T, res, BOOT_REPLICATES, boot=b) for T in PRIMARY_HORIZONS)
        rd.main(["--test", "emp_boot", "--workers", str(args.workers), "--out-dir", str(args.out_dir),
                 "--skip-summary"], cells=cells)


if __name__ == "__main__":
    main()
