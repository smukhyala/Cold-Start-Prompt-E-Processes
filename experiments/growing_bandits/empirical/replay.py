"""Replay the growing-bandit policies on the real prompt pools (spec sections 4-5).

    .venv/bin/python experiments/growing_bandits/empirical/replay.py estimate
    .venv/bin/python experiments/growing_bandits/empirical/replay.py point --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py boot  --workers 12
    .venv/bin/python experiments/growing_bandits/empirical/replay.py boot_kgrid --study het --workers 12

``estimate`` reads the collector's logs and writes the six reservoirs (G/F x npmle/raw/
parametric), the noise model and the parametric fit table under
``data/empirical_pool/reservoirs/``. ``point`` deploys `TEST_POLICIES["emp"]` on every
(pool, variant, T) at M = 1000; ``kgrid`` runs fixed-K over the K-grid on the same cells;
``boot`` re-estimates the NPMLE on B = 200 prompt resamples and deploys the four contrast
policies at the primary horizons, M = 250; ``boot_kgrid`` (Pre-registration 10, section 6.5) runs
fixed-K over the K-grid at T = 200, M = 250 on every bootstrap reservoir of every cell, for the
bootstrap lower bound of the regret range (``tables/<prefix>_boot_kgrid.csv``).

One data snapshot ties the verdict together: ``estimate`` freezes the terminal outcomes it
read into ``reservoirs/outcomes_snapshot.jsonl`` and writes ``reservoirs/manifest.json`` with
the sha256 of that snapshot, ``noise.json``, the fit table and each reservoir. ``boot``
resamples the snapshot (never the live logs); ``kgrid`` rows carry their reservoir's sha256;
``point``/``boot`` refuse reservoirs that no longer match the manifest and stamp the
manifest they ran on (``tables/{emp,emp_boot}_reservoir_stamp.json``), which
`registered_contrast.prompt_bootstrap_contrast` checks before it rules.

Seeds: every (pool, T) has one seed shared by its three variants (CRN across variants);
every bootstrap replicate has its own seeds and its own env id, so the per-cell
comparator cache (`run_deployment.prepare_cell_constants`) can never mix replicates.

Studies: every stage runs one `study.Study` (``--study prereg9|het``, default ``prereg9``),
which names the pools, the deployment tests (``emp``/``emp_boot`` or ``het``/``het_boot``),
the data / reservoir / log directories, the seed base, the table prefix and the noise
model. The paths above are Pre-registration 9's; `HETEROGENEITY` writes under
``data/heterogeneity/reservoirs/`` and ``tables/het_*``. This module's constants (`POOLS`,
`RES_DIR`, `LOG_DIR`, `EMP_SEED_BASE`, ...) stay Pre-registration 9's values, and the CLI
reads Pre-registration 9's paths from them at call time (`resolve_study`).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import multiprocessing as mp
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent / "deploy", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import k_star_envelope as kse  # noqa: E402
import policy_table as pt  # noqa: E402
import registered_contrast as rc  # noqa: E402
import run_deployment as rd  # noqa: E402
import study as st  # noqa: E402

import cold_start.growing.empirical_reservoir  # noqa: E402,F401  (registers "empirical" in workers)
from cold_start.growing import empirical as emp  # noqa: E402
from cold_start.growing.deploy.harness import CellSpec  # noqa: E402
from cold_start.growing.empirical_reservoir import EmpiricalReservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

log = logging.getLogger("empirical.replay")

#: Pre-registration 9's values (`study.PREREG9`); every function reads its study's instead.
EMP_SEED_BASE = st.PREREG9.seed_base
EMP_CELL_STRIDE = 1_000
POOLS: tuple[str, ...] = st.PREREG9.pools
VARIANTS: tuple[str, ...] = ("npmle", "raw", "parametric")
PRIMARY_HORIZONS: tuple[int, ...] = (50, 100, 200)
ALL_HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)
EMP_REPLICATES = 1000
BOOT_REPLICATES = 250
N_BOOT = 200
BOOT_SEED = 20260926
CONTRAST_POLICIES: tuple[str, ...] = ("p3_star", "fixed_K_star", "level_star", "phi_k4")
NOISE_BOOT = 1000
#: `boot_kgrid`: the horizon whose regret-range lower bound section 6.5 classifies on (Pre-reg 10).
BOOT_KGRID_HORIZON = 200
#: `phi_k4`'s model variant (`policy_table.POLICIES["phi_k4"]["params"]["artifact"]`), i.e.
#: the key `thresholds.json["by_cap"]["<cap>"]` must carry for `missing_cap_constants`.
PHI_K4_VARIANT: str = pt.POLICIES["phi_k4"]["params"]["artifact"]

DATA_DIR = st.PREREG9.data_dir
LOG_DIR = st.PREREG9.log_dirs[0]
RES_DIR = st.PREREG9.res_dir
SNAPSHOT_FILE = "outcomes_snapshot.jsonl"
MANIFEST_FILE = "manifest.json"
SNAPSHOT_COLUMNS: tuple[str, ...] = ("pool", "arm_id", "task_id", "replicate", "attempt", "status", "success")


def resolve_study(name: str) -> st.Study:
    """The study the CLI runs. Pre-registration 9's directories are read from this module's
    `LOG_DIR`/`RES_DIR` at call time, so they stay its one override point."""
    study = st.get_study(name)
    if study.name == st.PREREG9.name:
        study = dataclasses.replace(study, log_dirs=(Path(LOG_DIR),), res_dir=Path(RES_DIR))
    return study


def env_id(pool: str, variant: str, boot: int | None = None, *, study: st.Study = st.PREREG9) -> str:
    return f"{study.test}_{pool}_{variant}" + ("" if boot is None else f"_b{boot:03d}")


#: Component strides for `seed_for`'s index, chosen so a pool's block comfortably
#: exceeds any horizon (including a unit test's ad-hoc small T outside `ALL_HORIZONS`,
#: e.g. `test_kgrid_runs_on_empirical_cells`'s T=20) and a bootstrap's block comfortably
#: exceeds a pool's: `POOL_STRIDE` > max horizon, `BOOT_STRIDE` = `POOL_STRIDE * len(POOLS)`
#: (per study: `boot_stride`).
POOL_STRIDE = 10_000
BOOT_STRIDE = POOL_STRIDE * len(POOLS)


def boot_stride(study: st.Study = st.PREREG9) -> int:
    return POOL_STRIDE * len(study.pools)


def seed_for(pool: str, horizon: int, boot: int | None = None, *, study: st.Study = st.PREREG9) -> int:
    b = 0 if boot is None else boot + 1
    index = b * boot_stride(study) + study.pools.index(pool) * POOL_STRIDE + int(horizon)
    return study.seed_base + EMP_CELL_STRIDE * index


def make_emp_cell(pool: str, variant: str, horizon: int, reservoir: EmpiricalReservoir, n_replicates: int,
                  boot: int | None = None, *, study: st.Study = st.PREREG9) -> CellSpec:
    return CellSpec(env_id=env_id(pool, variant, boot, study=study), env_spec=reservoir.to_spec(),
                    horizon=int(horizon), cap=int(horizon), base_seed=seed_for(pool, horizon, boot, study=study),
                    n_replicates=int(n_replicates))


# ---- estimation ---------------------------------------------------------------------


def _pair_sq_diffs(outcomes: pd.DataFrame, pool: str) -> np.ndarray:
    ok = outcomes[(outcomes["status"] == emp.STATUS_OK) & (outcomes["pool"] == pool)]
    ok = ok.assign(success=ok["success"].astype(float))
    wide = ok.pivot_table(index=["arm_id", "task_id"], columns="replicate", values="success",
                          aggfunc="first")[[0, 1]].dropna()
    return (wide[0].to_numpy(float) - wide[1].to_numpy(float)) ** 2 / 2.0


def noise_model(outcomes: pd.DataFrame, *, per_pool: bool | None = None, seed: int = 0,
                study: st.Study = st.PREREG9) -> dict:
    """Within-cell noise v per pool, by the study's `noise_mode`.

    ``"pairwise_decision"`` (Pre-registration 9, exactly two pools): pooled unless the pools
    differ by more than the bootstrap SE of their difference. With `per_pool` given (the
    prompt bootstrap reuses the full-sample decision) the SE is not needed and is not computed.

    ``"per_pool"``: every pool's own v from its own replicate pairs (`per_pool` is ignored); a
    pool with no pairs uses ``v[study.borrowed_noise[pool]]`` and is listed under
    ``"borrowed"``, and one with no pairs and no declared borrow raises `ValueError`.
    """
    if study.noise_mode == "per_pool":
        return _per_pool_noise(outcomes, study)
    if len(study.pools) != 2:
        raise ValueError(f"{study.name}: noise_mode 'pairwise_decision' needs exactly two pools, got {study.pools}")
    pools = study.pools
    pooled, n_pairs = emp.within_cell_variance(outcomes, pools)
    by_pool = {p: emp.within_cell_variance(outcomes, [p])[0] for p in pools}
    se_diff = float("nan")
    if per_pool is None:
        sq = {p: _pair_sq_diffs(outcomes, p) for p in pools}
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(NOISE_BOOT):
            v_b = [float(sq[p][rng.integers(0, sq[p].size, sq[p].size)].mean()) for p in pools]
            diffs.append(v_b[0] - v_b[1])
        se_diff = float(np.std(diffs, ddof=1))
        per_pool = abs(by_pool[pools[0]] - by_pool[pools[1]]) > se_diff
    v = dict(by_pool) if per_pool else {p: pooled for p in pools}
    return {"v": v, "pooled": pooled, "by_pool": by_pool, "per_pool": bool(per_pool), "se_diff": se_diff,
            "n_pairs": n_pairs}


def _per_pool_noise(outcomes: pd.DataFrame, study: st.Study) -> dict:
    by_pool: dict[str, float] = {}
    n_pairs: dict[str, int] = {}
    no_pairs: list[str] = []
    for p in study.pools:
        try:
            v_p, n_p = emp.within_cell_variance(outcomes, [p])
        except ValueError:  # no replicate pairs in this pool
            no_pairs.append(p)
            continue
        by_pool[p] = v_p
        n_pairs[p] = n_p
    borrowed: dict[str, str] = {}
    for p in no_pairs:
        donor = study.borrowed_noise.get(p)
        if donor is None:
            raise ValueError(f"{study.name}: pool {p} has no replicate pairs and declares no borrowed noise")
        if donor not in by_pool:
            raise ValueError(f"{study.name}: pool {p} borrows its noise from {donor}, which has no replicate pairs")
        borrowed[p] = donor
    v = {p: by_pool[p] if p in by_pool else by_pool[borrowed[p]] for p in study.pools}
    return {"v": v, "by_pool": by_pool, "n_pairs": n_pairs, "borrowed": borrowed, "per_pool": True,
            "noise_mode": "per_pool"}


def estimate(outcomes: pd.DataFrame, *, study: st.Study = st.PREREG9
             ) -> tuple[dict[tuple[str, str], EmpiricalReservoir], dict, pd.DataFrame]:
    noise = noise_model(outcomes, study=study)
    reservoirs: dict[tuple[str, str], EmpiricalReservoir] = {}
    fit_tables = []
    for pool in study.pools:
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
                         seed: int = BOOT_SEED, study: st.Study = st.PREREG9
                         ) -> list[tuple[int, str, EmpiricalReservoir]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[int, str, EmpiricalReservoir]] = []
    for b in range(n_boot):
        resampled = pd.concat([_resample_pool(outcomes, p, rng) for p in study.pools], ignore_index=True)
        nb = noise_model(resampled, per_pool=noise["per_pool"], seed=b, study=study)
        for pool in study.pools:
            scores = emp.prompt_scores(resampled, pool)
            out.append((b, pool, emp.npmle_reservoir(scores, nb["v"][pool], f"{pool}_npmle_b{b:03d}")))
    return out


# ---- the frozen data snapshot -----------------------------------------------------------


def write_snapshot(outcomes: pd.DataFrame, res_dir: Path) -> Path:
    """The terminal outcomes `estimate` used, one JSON line each, sorted -- deterministic bytes."""
    frame = outcomes[list(SNAPSHOT_COLUMNS)].sort_values(["pool", "arm_id", "task_id", "replicate"])
    path = Path(res_dir) / SNAPSHOT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in frame.to_dict("records"):
            success = r["success"]
            fh.write(json.dumps({
                "pool": str(r["pool"]), "arm_id": str(r["arm_id"]), "task_id": str(r["task_id"]),
                "replicate": int(r["replicate"]), "attempt": int(r["attempt"]), "status": str(r["status"]),
                "success": None if success is None or pd.isna(success) else int(success),
            }, sort_keys=True) + "\n")
    return path


def write_manifest(res_dir: Path, reservoir_names: Iterable[str]) -> Path:
    res_dir = Path(res_dir)
    manifest = {
        "outcomes_snapshot": {"file": SNAPSHOT_FILE, "sha256": rc.file_sha256(res_dir / SNAPSHOT_FILE)},
        "noise": rc.file_sha256(res_dir / "noise.json"),
        "parametric_fits": rc.file_sha256(res_dir / "parametric_fits.csv"),
        "reservoirs": {name: rc.file_sha256(res_dir / f"{name}.json") for name in sorted(reservoir_names)},
    }
    path = res_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def load_manifest(res_dir: Path) -> dict:
    path = Path(res_dir) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"no {path}: run `replay.py estimate` first")
    return json.loads(path.read_text())


def load_snapshot(res_dir: Path) -> pd.DataFrame:
    """The frozen outcomes, after checking their sha256 against the manifest."""
    manifest = load_manifest(res_dir)
    path = Path(res_dir) / manifest["outcomes_snapshot"]["file"]
    got = rc.file_sha256(path)
    if got != manifest["outcomes_snapshot"]["sha256"]:
        raise ValueError(f"{path} sha256 {got} != manifest's {manifest['outcomes_snapshot']['sha256']}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))


def verify_reservoirs(res_dir: Path, *, study: st.Study = st.PREREG9) -> dict[str, str]:
    """``{env_id: sha256}`` of the point reservoirs (pools x variants); raises if any file moved since `estimate`."""
    shas = load_manifest(res_dir)["reservoirs"]
    out: dict[str, str] = {}
    for pool in study.pools:
        for variant in VARIANTS:
            name = f"{pool}_{variant}"
            got = rc.file_sha256(Path(res_dir) / f"{name}.json")
            if shas.get(name) != got:
                raise ValueError(f"reservoir {name}.json sha256 {got} != manifest's {shas.get(name)}; "
                                 "re-run `replay.py estimate`")
            out[env_id(pool, variant, study=study)] = got
    return out


def stamp_kgrid(frame: pd.DataFrame, shas: dict[str, str]) -> pd.DataFrame:
    """Each K-grid row carries the sha256 of the reservoir its cell ran on."""
    missing = sorted(set(frame["env_id"]) - set(shas))
    if missing:
        raise ValueError(f"no reservoir sha for {missing}")
    return frame.assign(reservoir_sha256=frame["env_id"].map(shas))


def run_estimate(log_dir: Path | None, res_dir: Path, out_dir: Path, *, study: st.Study = st.PREREG9) -> dict:
    """Read the logs once, freeze them, estimate, and write reservoirs + manifest.

    The outcomes are `study.load_study_outcomes`: every log dir of the study (or `log_dir`
    alone when given) plus its relabelled extra snapshots.
    """
    source = study if log_dir is None else dataclasses.replace(study, log_dirs=(Path(log_dir),))
    outcomes = st.load_study_outcomes(source)
    write_snapshot(outcomes, res_dir)
    reservoirs, noise, fits = estimate(load_snapshot_unchecked(res_dir), study=study)
    for (pool, variant), res in reservoirs.items():
        save_reservoir(res, Path(res_dir) / f"{pool}_{variant}.json")
    (Path(res_dir) / "noise.json").write_text(json.dumps(noise, indent=2))
    fits.to_csv(Path(res_dir) / "parametric_fits.csv", index=False)
    write_manifest(res_dir, [f"{p}_{v}" for p, v in reservoirs])
    for (pool, variant), res in reservoirs.items():
        log.info("%s %-10s mean %.4f sd %.4f atoms %d %s", pool, variant, res.mean(), res.sd(),
                 res.atoms.size, res.validation_error or "")
    n_purged = purge_stale_emp_comparators(out_dir, study=study)
    log.info("purged %d stale %s comparator file(s) under %s (new reservoirs invalidate any "
             "prefix cached under the previous ones)", n_purged, study.test, Path(out_dir) / "comparators")
    return {"reservoirs": reservoirs, "noise": noise, "fits": fits}


def load_snapshot_unchecked(res_dir: Path) -> pd.DataFrame:
    """The snapshot just written (before a manifest exists): estimate reads what it froze, not the logs."""
    path = Path(res_dir) / SNAPSHOT_FILE
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))


def save_reservoir(res: EmpiricalReservoir, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res.to_spec()))


def load_reservoir(path: Path) -> EmpiricalReservoir:
    return build_reservoir(json.loads(path.read_text()))


# ---- runs -----------------------------------------------------------------------------


def _kgrid_item(item: kse.Item) -> dict:
    return kse.run_item(item)


def kgrid(cells: list[CellSpec], *, workers: int, k_grid: tuple[int, ...] = kse.DEFAULT_K_GRID,
          study: st.Study = st.PREREG9, split: str | None = None) -> pd.DataFrame:
    """Fixed-K over `k_grid` on every cell; ``split`` (recorded on each row) defaults to the study's test."""
    split = study.test if split is None else split
    items = [kse.Item(split=split, spec=c, K=int(K)) for c in cells for K in k_grid if int(K) <= c.horizon]
    # Built once, here, in the parent: `CSTable.load_or_build` writes through a fixed
    # `.tmp` name (`run_deployment.prebuild_shared_tables` does the same for the study's
    # own cells), so two pool workers racing to build the same (T, alpha) table for the
    # first time could tear each other's write.
    for horizon, alpha in sorted({(int(c.horizon), float(c.alpha)) for c in cells}):
        CSTable.load_or_build(horizon, alpha)
    if workers <= 1:
        rows = [_kgrid_item(i) for i in items]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_kgrid_item, items, chunksize=1)
    frame = pd.DataFrame(rows)
    parts = frame["env_id"].str.split("_", expand=True)
    return frame.assign(pool=parts[1], variant=parts[2])


def run_boot_kgrid(res_dir: Path, *, workers: int, study: st.Study = st.PREREG9, n_boot: int = N_BOOT,
                   horizon: int = BOOT_KGRID_HORIZON, n_replicates: int = BOOT_REPLICATES,
                   k_grid: tuple[int, ...] = kse.DEFAULT_K_GRID) -> pd.DataFrame:
    """The K-grid at `horizon` on every bootstrap reservoir of every pool (Pre-registration 10, section 6.5).

    The reservoirs are the ``boot`` stage's, regenerated deterministically from the frozen snapshot
    and ``noise.json`` (both checked against the manifest) with the same `bootstrap_reservoirs` call and
    seed; a reservoir the ``boot`` stage saved under ``<res_dir>/boot/`` must equal its regenerated
    spec byte for byte, or this raises (the episodes and this K-grid would describe different
    replicates). Cells are the boot stage's own (`make_emp_cell` with ``boot=b``: same env id and
    seed). Each row carries its replicate (``boot``), its reservoir's sha256 and the manifest's.
    """
    res_dir = Path(res_dir)
    verify_reservoirs(res_dir, study=study)
    manifest_sha = rc.file_sha256(res_dir / MANIFEST_FILE)
    outcomes = load_snapshot(res_dir)
    if rc.file_sha256(res_dir / "noise.json") != load_manifest(res_dir)["noise"]:
        raise ValueError("noise.json no longer matches the reservoir manifest; re-run `replay.py estimate`")
    noise = json.loads((res_dir / "noise.json").read_text())
    cells: list[CellSpec] = []
    shas: dict[str, str] = {}
    for b, pool, res in bootstrap_reservoirs(outcomes, noise, n_boot=n_boot, study=study):
        name = f"{pool}_npmle_b{b:03d}"
        text = json.dumps(res.to_spec())
        saved = res_dir / "boot" / f"{name}.json"
        if saved.exists() and saved.read_text() != text:
            raise ValueError(f"{saved} differs from the reservoir regenerated from the snapshot ({name}); "
                             "re-run `replay.py boot` so its episodes and this K-grid share one set of replicates")
        cell = make_emp_cell(pool, "npmle", horizon, res, n_replicates, boot=b, study=study)
        shas[cell.env_id] = hashlib.sha256(text.encode()).hexdigest()
        cells.append(cell)
    frame = kgrid(cells, workers=workers, k_grid=k_grid, study=study, split=study.boot_test)
    boots = frame["env_id"].str.extract(r"_b(\d{3})$")[0]
    if boots.isna().any():
        raise ValueError(f"boot_kgrid rows without a replicate suffix: {sorted(frame.loc[boots.isna(), 'env_id'])}")
    return frame.assign(boot=boots.astype(int), reservoir_sha256=frame["env_id"].map(shas),
                        manifest_sha256=manifest_sha)


def point_cells(res_dir: Path | None = None, *, study: st.Study = st.PREREG9) -> list[CellSpec]:
    res_dir = study.res_dir if res_dir is None else Path(res_dir)
    return [make_emp_cell(p, v, T, load_reservoir(res_dir / f"{p}_{v}.json"), EMP_REPLICATES, study=study)
            for p in study.pools for v in VARIANTS for T in ALL_HORIZONS]


# ---- cap-T constants preflight ----------------------------------------------------------


def missing_cap_constants(
    baseline_params: dict | None, thresholds: dict | None, horizons: Iterable[int]
) -> list[str]:
    """Every cap-T requirement `point`/`boot` would otherwise silently substitute a
    placeholder (or a different cap's constant) for, over `horizons`.

    Every `emp`/`emp_boot` cell is uncapped (`make_emp_cell` sets ``cap = horizon``), so
    cap and T are the same number here. `baseline_params.json["by_cap"]` today (2026-09-26)
    has blocks only for cap 200 and 1000 -- 50, 100, 500 are added by a later retune -- and
    without this check `pt.baseline_params_for_cap` falls back to the cap-64 block for the
    missing ones, stamping only ``params_tuned: False`` deep in the manifest rather than
    stopping the run. This mirrors exactly what `pt.resolve_params`/`pt.tau_for_cap` look
    up (`pt._lookup_by_number`, the same tolerant numeric-key match), so a cap this
    reports clean is a cap those functions cannot silently fall back on. Returns ``[]``
    when `point`/`boot` may run; otherwise one named entry per missing piece.
    """
    problems: list[str] = []
    by_cap = (baseline_params or {}).get(pt.BY_CAP_KEY) or {}
    thresh_by_cap = (thresholds or {}).get(pt.BY_CAP_KEY) or {}
    for T in horizons:
        cap = int(T)
        block = pt._lookup_by_number(by_cap, cap)
        if not block:
            problems.append(f"baseline_params.json: by_cap has no block for cap {cap}")
        else:
            p3 = pt._lookup_by_number(block.get("p3_star") or {}, cap)
            if not p3 or "alpha" not in p3 or "c" not in p3:
                problems.append(f"baseline_params.json: by_cap[{cap}].p3_star has no entry for T={cap}")
            fixed_k = pt._lookup_by_number(block.get("fixed_K_star") or {}, cap)
            if not fixed_k or "K" not in fixed_k:
                problems.append(f"baseline_params.json: by_cap[{cap}].fixed_K_star has no entry for T={cap}")
            level = block.get("level_star") or {}
            if not all(k in level for k in ("alpha", "c", "b")):
                problems.append(f"baseline_params.json: by_cap[{cap}].level_star is missing alpha/c/b")
        tau_block = thresh_by_cap.get(str(cap)) or {}
        if not tau_block.get(PHI_K4_VARIANT):
            problems.append(f"thresholds.json: by_cap has no {PHI_K4_VARIANT} threshold for cap {cap}")
    return problems


def _require_cap_constants(out_dir: Path, horizons: Iterable[int], stage: str, test: str = "emp") -> None:
    baseline_params = pt.load_baseline_params(Path(out_dir) / "baseline_params.json")
    thresholds = pt.load_thresholds(Path(out_dir) / "thresholds.json")
    problems = missing_cap_constants(baseline_params, thresholds, horizons)
    if problems:
        raise SystemExit(
            f"replay.py {stage}: cap-T constants are incomplete; every uncapped {test} cell's "
            "p3_star/fixed_K_star/level_star and phi_k4 threshold must be present for its "
            "own cap, or the deployment silently substitutes the cap-64 tuning:\n  "
            + "\n  ".join(problems)
        )


# ---- comparator cache -------------------------------------------------------------------


def purge_stale_emp_comparators(out_dir: str | Path, *, boot_only: bool = False,
                                study: st.Study = st.PREREG9) -> int:
    """Delete cached comparator prefix/metadata files for the study's point and bootstrap cells
    (``emp``/``emp_boot`` for Pre-registration 9; every glob below uses ``<study.test>_``).

    `run_deployment.prepare_cell_constants` caches the reservoir prefix by
    ``<cell>_seed<seed>_M<M>`` alone (shared with the shipped deployment tests, so that
    key is deliberately not touched here): a cell's name and seed are the same before and
    after a reservoir is re-fit from more data, so a prefix cached under the old
    reservoir survives untouched and `harness.run_cell`'s CRN check then fails against the
    freshly estimated one. This affects both call sites that re-fit a reservoir in place:
    `estimate` (the six point reservoirs) and `boot` (the `N_BOOT` bootstrap ones), so
    each `env_id` starts with ``"emp_"`` -- point or bootstrapped alike -- and one glob
    under ``comparators/`` finds every stale file for either.

    ``boot_only=True`` (used by the `boot` stage) restricts the glob to a bootstrap
    cell's stem -- ``emp_<pool>_<variant>_b<NNN>_...`` -- so a re-run of `boot` cannot
    delete the point cells' caches (`estimate` already owns invalidating those, and they
    are far more expensive to rebuild: M is 4x `boot`'s and every horizon runs, not just
    the three primary ones). ``boot_only=False`` (the default, used by `estimate`, which
    can invalidate *either* kind since it is what changes both) matches every emp
    comparator file, point and bootstrapped alike.

    Called unconditionally rather than only when a reservoir's spec actually changed: a
    dict/float comparison could miss a change that matters, or flag one that does not,
    and a wasted recompute of a cached prefix is far cheaper than a silently stale one.
    """
    comparators_dir = Path(out_dir) / "comparators"
    if not comparators_dir.exists():
        return 0
    pattern = f"{study.test}_*_b[0-9][0-9][0-9]_*" if boot_only else f"{study.test}_*"
    n = 0
    for path in comparators_dir.glob(pattern):
        if path.suffix in (".npy", ".json"):
            path.unlink()
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["estimate", "point", "kgrid", "boot", "boot_kgrid"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--study", choices=sorted(st.STUDIES), default=st.PREREG9.name)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    study = resolve_study(args.study)
    res_dir = study.res_dir

    if args.stage == "estimate":
        run_estimate(None, res_dir, args.out_dir, study=study)
    elif args.stage == "point":
        _require_cap_constants(args.out_dir, ALL_HORIZONS, "point", study.test)
        verify_reservoirs(res_dir, study=study)
        rd.main(["--test", study.test, "--workers", str(args.workers), "--out-dir", str(args.out_dir)],
                cells=point_cells(res_dir, study=study))
        rc.write_reservoir_stamp(args.out_dir, study.test, res_dir / MANIFEST_FILE)
    elif args.stage == "kgrid":
        shas = verify_reservoirs(res_dir, study=study)
        frame = stamp_kgrid(kgrid(point_cells(res_dir, study=study), workers=args.workers, study=study), shas)
        out = args.out_dir / "tables" / study.table("kgrid")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(frame))
    elif args.stage == "boot_kgrid":
        frame = run_boot_kgrid(res_dir, workers=args.workers, study=study)
        out = args.out_dir / "tables" / study.table("boot_kgrid")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        log.info("wrote %s (%d rows)", out, len(frame))
    else:
        _require_cap_constants(args.out_dir, PRIMARY_HORIZONS, "boot", study.test)
        n_purged = purge_stale_emp_comparators(args.out_dir, boot_only=True, study=study)
        log.info("purged %d stale %s comparator file(s) under %s (a re-run re-fits every "
                 "bootstrap reservoir from the current logs)", n_purged, study.boot_test,
                 Path(args.out_dir) / "comparators")
        outcomes = load_snapshot(res_dir)  # the frozen snapshot `estimate` used, never the live logs
        if rc.file_sha256(res_dir / "noise.json") != load_manifest(res_dir)["noise"]:
            raise ValueError("noise.json no longer matches the reservoir manifest; re-run `replay.py estimate`")
        noise = json.loads((res_dir / "noise.json").read_text())
        cells = []
        for b, pool, res in bootstrap_reservoirs(outcomes, noise, study=study):
            save_reservoir(res, res_dir / "boot" / f"{pool}_npmle_b{b:03d}.json")
            cells.extend(make_emp_cell(pool, "npmle", T, res, BOOT_REPLICATES, boot=b, study=study)
                         for T in PRIMARY_HORIZONS)
        rd.main(["--test", study.boot_test, "--workers", str(args.workers), "--out-dir", str(args.out_dir),
                 "--skip-summary"], cells=cells)
        rc.write_reservoir_stamp(args.out_dir, study.boot_test, res_dir / MANIFEST_FILE)


if __name__ == "__main__":
    main()
