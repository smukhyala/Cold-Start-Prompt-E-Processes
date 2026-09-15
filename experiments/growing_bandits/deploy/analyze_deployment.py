#!/usr/bin/env python
"""M7: every table the deployment report cites, from the runner's per-episode frames.

Inputs are `run_deployment.py`'s outputs for one test (``episodes/<test>/<cell>/
<policy>.parquet``, the dynamics ``.npz``, the logged on-policy ``Snapshot`` pickles and
``manifest_<test>.jsonl``) plus the offline artifacts (`offline_metrics.csv`,
`reservoir_diagnostics_offline.csv`, `threshold_selection.csv`). Outputs are CSV
tables under ``results/growing_bandits/deploy/tables/``; every number in the report
must trace to one of them.

Three rules shape the statistics (DEPLOYMENT_PLAN.md "Metrics"; register #4, #8, #9):

* **Paired, per episode.** Within a cell every policy saw the same episodes (CRN), so
  a comparison is the mean of per-episode differences ``regret_policy - regret_ref``
  (negative = the policy is better) with a bootstrap over episodes -- never a
  difference of two independently bootstrapped means.
* **Stratified first, pooled second.** Every table is reported per family x horizon;
  the pooled block gives each cell equal weight (a cell-stratified bootstrap) and adds
  a cluster bootstrap over environments, the honest interval for "a fresh environment".
* **Pre-registered rows are separate.** `primary_contrasts_<test>_<rec>.csv` holds
  exactly H1a / H1b / H2 per stratum; everything else is in `secondary_contrasts_*`
  and labelled exploratory.

Bootstrap mechanics. The pre-registered rows go through `stats.paired_bootstrap` /
`stats.stratified_pooled` / `stats.cluster_bootstrap_over_envs` verbatim. The bulk
tables (30+ policies x 5 recommenders x 9 quantities x 40 cells) use `CellBoot`: one
multinomial resample-weight matrix per cell, shared by every column of that cell, so
a cell's 10k resamples cost one matmul instead of ~1,500 index gathers. It is the
same percentile bootstrap over episodes; the resample draws are merely reused across
columns, which is also what pairing suggests.

The parity gate. Before any main table is written, every logged on-policy snapshot
of every learned policy is scored by the scalar corpus extractor and by the
vectorized deployment extractor (`ood.snapshot_pass`); if any column fails the M1
tolerance, `onpolicy_parity*.csv` is written and the main tables are NOT (register
#5). The same pass feeds the on-policy shift (`ood.py`) and reservoir-tail
(`reservoir_analysis.py`) diagnostics.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ood  # noqa: E402
import policy_table as pt  # noqa: E402
import reservoir_analysis as ra  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy import stats  # noqa: E402
from cold_start.growing.deploy.harness import DYNAMICS_KEYS  # noqa: E402
from cold_start.growing.deploy.recommenders import (  # noqa: E402
    PRIMARY_RECOMMENDER,
    RECOMMENDER_NAMES,
)
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

log = logging.getLogger("deploy.analyze")

DEFAULT_OUT_DIR = rd.DEFAULT_OUT_DIR
REFERENCES: tuple[str, ...] = ("cp0", "p3_star")
PRIMARY_POLICY = "phi_k16"
#: The three pre-registered contrasts (DEPLOYMENT_PLAN.md "Pre-registered primary comparisons").
PRIMARY_HYPOTHESES: tuple[tuple[str, str, str, str], ...] = (
    ("H1a", "phi_k16", "cp0", "P9_16 vs cp0, the label continuation policy (necessary condition)"),
    ("H1b", "phi_k16", "p3_star", "P9_16 vs the validation-selected power schedule P3*"),
    ("H2", "phi_k16", "phi_k16_quality", "P9_16 (clock+quality+evidence) vs P7 (clock+quality): e-process value"),
)
#: Contrast references for the bulk tables: the two pre-registered ones and the primary policy.
CONTRAST_REFS: tuple[str, ...] = ("cp0", "p3_star", PRIMARY_POLICY)
KEY_POLICIES: tuple[str, ...] = tuple(pt.TEST_POLICIES["robust"])
#: Tables defined on Test A keep the plan's file name there; other tests get a suffix.
A_NAMED_TABLES: frozenset[str] = frozenset(
    {"offline_vs_deployed", "surrogate_validity", "tau_curves", "recommender_sensitivity",
     "recommender_kendall", "onpolicy_parity", "onpolicy_parity_detail", "ood_flags"}
)
PER_TEST_TABLE_NAMES: dict[str, str] = {
    "D": "transfer_D", "C": "heldout_C", "B": "regime_B", "cap": "cap_sweep",
}
DESCRIPTIVE_MEANS: tuple[str, ...] = (
    "regret_disc", "k_final", "search_frac", "cap_hit", "n_eliminated_final", "herfindahl",
    "n_singletons_final", "mu_star", "mu_star_cap", "best_discovered",
)


def _seed(*parts) -> int:
    return int(zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")))


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    log.info("wrote %s (%d rows)", path.name, len(df))
    return path


def table_name(base: str, test: str, rec: str | None = None) -> str:
    if base in A_NAMED_TABLES:
        name = base if test == "A" else f"{base}_{test}"
    else:
        name = f"{base}_{test}"
    if rec is not None:
        name += "_primary" if rec == PRIMARY_RECOMMENDER else f"_{rec}"
    return name + ".csv"


# ---- episode frames ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One cell's per-episode frames, one per policy, aligned by episode."""

    name: str
    env_id: str
    family: str
    horizon: int
    cap: int
    base_seed: int
    n: int
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)

    @property
    def policies(self) -> list[str]:
        return list(self.frames)


def recommenders_in(frame: pd.DataFrame) -> list[str]:
    return [r for r in RECOMMENDER_NAMES if f"regret_{r}" in frame.columns]


def load_cells(out_dir: Path, test: str) -> list[Cell]:
    """Every cell of `test` from ``episodes/<test>``; frames sorted by episode and checked
    to share the seed and episode set (the CRN pairing the paired statistics assume)."""
    root = out_dir / "episodes" / test
    if not root.is_dir():
        raise FileNotFoundError(f"no episodes for test {test!r} under {root}")
    cells: list[Cell] = []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames: dict[str, pd.DataFrame] = {}
        groups: dict[str, str] = {}
        meta: dict | None = None
        for pq in sorted(cell_dir.glob("*.parquet")):
            df = pd.read_parquet(pq)
            df = df.drop(columns=[c for c in df.columns if c.startswith("regret_sup_")])
            df = df.sort_values("episode").reset_index(drop=True)
            policy = pq.stem
            head = df.iloc[0]
            this = {
                "env_id": str(head["env_id"]), "family": str(head["family"]),
                "horizon": int(head["horizon"]), "cap": int(head["cap"]),
                "base_seed": int(head["base_seed"]), "n": int(len(df)),
                "episodes": df["episode"].to_numpy(),
            }
            if meta is None:
                meta = this
            else:
                same = all(meta[k] == this[k] for k in ("env_id", "horizon", "cap", "base_seed", "n"))
                if not same or not np.array_equal(meta["episodes"], this["episodes"]):
                    raise RuntimeError(
                        f"{cell_dir.name}/{policy}: episodes are not aligned with the cell's "
                        "other policies (different seed, count or episode ids)"
                    )
            frames[policy] = df
            groups[policy] = str(head["group"])
        if meta is None:
            continue
        cells.append(
            Cell(
                name=cell_dir.name, env_id=meta["env_id"], family=meta["family"],
                horizon=meta["horizon"], cap=meta["cap"], base_seed=meta["base_seed"],
                n=meta["n"], frames=frames, groups=groups,
            )
        )
    if not cells:
        raise FileNotFoundError(f"no parquet frames under {root}")
    log.info("test %s: %d cells, %d policies in the first", test, len(cells), len(cells[0].frames))
    return cells


# ---- bootstrap engine ---------------------------------------------------------------------


class CellBoot:
    """Shared-index multinomial bootstrap of a cell's episode means.

    ``W`` is ``(n_boot, n)`` with ``W[b] ~ Multinomial(n, 1/n) / n``: the resample
    weights of draw ``b``. ``means(X)`` for an ``(n, m)`` column stack is ``W @ X``,
    the ``(n_boot, m)`` resampled means of every column at once.
    """

    def __init__(self, n: int, n_boot: int, seed: int) -> None:
        if n < 1 or n_boot < 1:
            raise ValueError("n and n_boot must be >= 1")
        rng = np.random.default_rng(seed)
        self.n = int(n)
        self.n_boot = int(n_boot)
        self.W = rng.multinomial(self.n, np.full(self.n, 1.0 / self.n), size=self.n_boot)
        self.W = self.W.astype(np.float64) / self.n

    def means(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != self.n:
            raise ValueError(f"expected {self.n} rows; got {X.shape[0]}")
        return self.W @ X


def percentile_ci(boot: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    tail = 100.0 * (1.0 - ci) / 2.0
    lo, hi = np.percentile(np.asarray(boot, dtype=np.float64), [tail, 100.0 - tail])
    return float(lo), float(hi)


def win_rate(diff: np.ndarray) -> float:
    """Share of episodes the policy wins (``diff < 0``), a tie counting half."""
    d = np.asarray(diff, dtype=np.float64)
    return float(np.mean(d < 0.0) + 0.5 * np.mean(d == 0.0))


@dataclass
class CellStats:
    """Per-policy point statistics and resampled-mean vectors for one cell and recommender."""

    cell: Cell
    rec: str
    quantities: list[str]
    mean: dict[tuple[str, str], float]
    se: dict[tuple[str, str], float]
    win: dict[tuple[str, str], float]
    boot: dict[tuple[str, str], np.ndarray]  # (n_boot,) float32
    descriptive: dict[str, dict[str, float]]  # policy -> column -> mean


def cell_quantities(cell: Cell, rec: str) -> tuple[list[str], dict[tuple[str, str], np.ndarray]]:
    """Per-episode arrays of every bootstrapped quantity, per policy."""
    out: dict[tuple[str, str], np.ndarray] = {}
    quantities = ["regret", "regret_disc", "regret_sel"]
    refs = [r for r in CONTRAST_REFS if r in cell.frames]
    quantities += [f"d_regret_vs_{r}" for r in refs]
    if "cp0" in cell.frames:
        quantities += ["d_disc_vs_cp0", "d_sel_vs_cp0"]
    ref_regret = {r: cell.frames[r][f"regret_{rec}"].to_numpy(dtype=np.float64) for r in refs}
    if "cp0" in cell.frames:
        cp0_disc = cell.frames["cp0"]["regret_disc"].to_numpy(dtype=np.float64)
        cp0_sel = cell.frames["cp0"][f"regret_sel_{rec}"].to_numpy(dtype=np.float64)
    for policy, df in cell.frames.items():
        regret = df[f"regret_{rec}"].to_numpy(dtype=np.float64)
        disc = df["regret_disc"].to_numpy(dtype=np.float64)
        sel = df[f"regret_sel_{rec}"].to_numpy(dtype=np.float64)
        out[(policy, "regret")] = regret
        out[(policy, "regret_disc")] = disc
        out[(policy, "regret_sel")] = sel
        for r in refs:
            out[(policy, f"d_regret_vs_{r}")] = regret - ref_regret[r]
        if "cp0" in cell.frames:
            out[(policy, "d_disc_vs_cp0")] = disc - cp0_disc
            out[(policy, "d_sel_vs_cp0")] = sel - cp0_sel
    return quantities, out


def descriptive_means(df: pd.DataFrame, rec: str) -> dict[str, float]:
    out = {c: float(df[c].to_numpy(dtype=np.float64).mean()) for c in DESCRIPTIVE_MEANS}
    out["q"] = float(df[f"q_{rec}"].to_numpy(dtype=np.float64).mean())
    out["n_rec"] = float(df[f"n_rec_{rec}"].to_numpy(dtype=np.float64).mean())
    demoted = df["n_demoted"].to_numpy(dtype=np.float64)
    out["n_demoted"] = float(demoted.mean()) if bool(np.all(demoted >= 0)) else float("nan")
    hit = df["cap_hit"].to_numpy(dtype=bool)
    t_hit = df["t_cap_hit"].to_numpy(dtype=np.float64)
    out["t_cap_hit_given_hit"] = float(t_hit[hit].mean()) if hit.any() else float("nan")
    return out


def compute_cell_stats(cell: Cell, rec: str, n_boot: int) -> CellStats:
    quantities, arrays = cell_quantities(cell, rec)
    keys = list(arrays)
    X = np.stack([arrays[k] for k in keys], axis=1)
    boot = CellBoot(cell.n, n_boot, _seed("cellboot", cell.name, rec)).means(X)
    mean: dict[tuple[str, str], float] = {}
    se: dict[tuple[str, str], float] = {}
    win: dict[tuple[str, str], float] = {}
    boots: dict[tuple[str, str], np.ndarray] = {}
    for j, k in enumerate(keys):
        x = arrays[k]
        mean[k] = float(x.mean())
        se[k] = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
        win[k] = win_rate(x) if k[1].startswith("d_") else float("nan")
        boots[k] = boot[:, j].astype(np.float32)
    desc = {p: descriptive_means(df, rec) for p, df in cell.frames.items()}
    return CellStats(cell, rec, quantities, mean, se, win, boots, desc)


# ---- strata -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stratum:
    level: str  # cell | family_horizon | family | horizon | pooled
    family: str  # "all" for pooled-over-families
    horizon: str  # "all" for pooled-over-horizons
    cell: str  # cell name at level "cell", else ""
    cells: tuple[str, ...]


def strata_of(cells: list[Cell]) -> list[Stratum]:
    families = sorted({c.family for c in cells})
    horizons = sorted({c.horizon for c in cells})
    out: list[Stratum] = []
    for c in cells:
        out.append(Stratum("cell", c.family, str(c.horizon), c.name, (c.name,)))
    for f in families:
        for T in horizons:
            names = tuple(c.name for c in cells if c.family == f and c.horizon == T)
            if names:
                out.append(Stratum("family_horizon", f, str(T), "", names))
    if len(horizons) > 1:
        for f in families:
            names = tuple(c.name for c in cells if c.family == f)
            out.append(Stratum("family", f, "all", "", names))
    if len(families) > 1:
        for T in horizons:
            names = tuple(c.name for c in cells if c.horizon == T)
            out.append(Stratum("horizon", "all", str(T), "", names))
    out.append(Stratum("pooled", "all", "all", "", tuple(c.name for c in cells)))
    return out


def aggregate(
    per_cell: dict[str, CellStats], stratum: Stratum, policy: str, quantity: str
) -> dict:
    """Equal-weight mean over the stratum's cells with the cell-stratified bootstrap CI.

    The stratum's bootstrap distribution is the average of its cells' resampled-mean
    vectors (each cell resampled within itself), exactly `stats.stratified_pooled`'s
    construction; ``se`` is the equal-weight combination of the per-cell paired SEs;
    ``win`` the mean of per-cell win rates. Cells lacking the policy or the
    reference are skipped and counted in ``n_cells``.
    """
    key = (policy, quantity)
    have = [per_cell[c] for c in stratum.cells if c in per_cell and key in per_cell[c].mean]
    if not have:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "win": np.nan,
                "n_cells": 0, "n_episodes": 0, "n_envs": 0, "cluster_lo": np.nan, "cluster_hi": np.nan}
    means = np.array([cs.mean[key] for cs in have])
    ses = np.array([cs.se[key] for cs in have])
    boot = np.mean([cs.boot[key].astype(np.float64) for cs in have], axis=0)
    lo, hi = percentile_ci(boot)
    out = {
        "mean": float(means.mean()),
        "lo": lo,
        "hi": hi,
        "se": float(np.sqrt(np.nansum(ses**2)) / len(have)),
        "win": float(np.mean([cs.win[key] for cs in have])) if quantity.startswith("d_") else np.nan,
        "n_cells": len(have),
        "n_episodes": int(sum(cs.cell.n for cs in have)),
        "n_envs": len({cs.cell.env_id for cs in have}),
        "cluster_lo": np.nan,
        "cluster_hi": np.nan,
    }
    envs = [cs.cell.env_id for cs in have]
    if len(set(envs)) >= 2 and stratum.level != "cell":
        cm = pd.DataFrame({"env_id": envs, "value": means})
        cb = stats.cluster_bootstrap_over_envs(
            cm, "env_id", "value", n_boot=min(have[0].boot[key].shape[0], 10_000),
            seed=_seed("cluster", stratum.level, stratum.family, stratum.horizon, policy, quantity),
        )
        out["cluster_lo"], out["cluster_hi"] = cb["lo"], cb["hi"]
    return out


def descriptive_for(per_cell: dict[str, CellStats], stratum: Stratum, policy: str) -> dict:
    have = [per_cell[c] for c in stratum.cells if c in per_cell and policy in per_cell[c].descriptive]
    if not have:
        return {}
    keys = list(have[0].descriptive[policy])
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([cs.descriptive[policy][k] for cs in have], dtype=np.float64)
        out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return out


def _stratum_meta(test: str, rec: str, s: Stratum, cells_by_name: dict[str, Cell]) -> dict:
    meta = {"test": test, "recommender": rec, "level": s.level, "family": s.family,
            "horizon": s.horizon, "cell": s.cell}
    if s.level == "cell":
        c = cells_by_name[s.cell]
        meta.update({"env_id": c.env_id, "cap": c.cap, "base_seed": c.base_seed})
    else:
        meta.update({"env_id": "", "cap": "", "base_seed": ""})
    return meta


def policy_group(policy: str, cells: list[Cell]) -> str:
    for c in cells:
        if policy in c.groups:
            return c.groups[policy]
    return pt.POLICIES.get(policy, {}).get("group", "?")


def main_table(
    test: str, rec: str, cells: list[Cell], per_cell: dict[str, CellStats], strata: list[Stratum]
) -> pd.DataFrame:
    """Per (stratum, policy): descriptive means, mean regret with CI, paired deltas vs
    the two references with paired-bootstrap CI, win rate and cluster CI."""
    cells_by_name = {c.name: c for c in cells}
    policies = sorted({p for c in cells for p in c.policies})
    rows: list[dict] = []
    for s in strata:
        for policy in policies:
            desc = descriptive_for(per_cell, s, policy)
            if not desc:
                continue
            row = _stratum_meta(test, rec, s, cells_by_name)
            row.update({"policy": policy, "group": policy_group(policy, cells)})
            reg = aggregate(per_cell, s, policy, "regret")
            row.update({
                "n_cells": reg["n_cells"], "n_envs": reg["n_envs"], "n_episodes": reg["n_episodes"],
                "regret": reg["mean"], "regret_lo": reg["lo"], "regret_hi": reg["hi"], "regret_se": reg["se"],
                "q": desc["q"],
            })
            sel = aggregate(per_cell, s, policy, "regret_sel")
            row.update({"regret_disc": desc["regret_disc"], "regret_sel": sel["mean"]})
            for k in ("k_final", "search_frac", "n_rec", "n_singletons_final", "herfindahl",
                      "n_eliminated_final", "n_demoted", "t_cap_hit_given_hit", "mu_star",
                      "mu_star_cap", "best_discovered"):
                row[k] = desc[k]
            row["cap_hit_frac"] = desc["cap_hit"]
            for ref in REFERENCES:
                d = aggregate(per_cell, s, policy, f"d_regret_vs_{ref}")
                p = f"d_regret_vs_{ref}"
                row.update({
                    p: d["mean"], f"{p}_lo": d["lo"], f"{p}_hi": d["hi"], f"{p}_se": d["se"],
                    f"{p}_win": d["win"], f"{p}_cluster_lo": d["cluster_lo"],
                    f"{p}_cluster_hi": d["cluster_hi"], f"{p}_n_cells": d["n_cells"],
                })
            rows.append(row)
    return pd.DataFrame(rows)


def decomposition_table(
    test: str, rec: str, cells: list[Cell], per_cell: dict[str, CellStats], strata: list[Stratum]
) -> pd.DataFrame:
    """`R_T = R_disc + R_sel` per policy per stratum (cells included), each with a CI, and
    the paired decomposition of the delta vs cp0."""
    cells_by_name = {c.name: c for c in cells}
    policies = sorted({p for c in cells for p in c.policies})
    rows: list[dict] = []
    for s in strata:
        for policy in policies:
            reg = aggregate(per_cell, s, policy, "regret")
            if reg["n_cells"] == 0:
                continue
            row = _stratum_meta(test, rec, s, cells_by_name)
            row.update({"policy": policy, "group": policy_group(policy, cells),
                        "n_cells": reg["n_cells"], "n_episodes": reg["n_episodes"],
                        "regret": reg["mean"], "regret_lo": reg["lo"], "regret_hi": reg["hi"]})
            for q, name in (("regret_disc", "regret_disc"), ("regret_sel", "regret_sel"),
                            ("d_disc_vs_cp0", "d_disc_vs_cp0"), ("d_sel_vs_cp0", "d_sel_vs_cp0"),
                            ("d_regret_vs_cp0", "d_regret_vs_cp0")):
                a = aggregate(per_cell, s, policy, q)
                row.update({name: a["mean"], f"{name}_lo": a["lo"], f"{name}_hi": a["hi"]})
            rows.append(row)
    return pd.DataFrame(rows)


# ---- contrasts ------------------------------------------------------------------------------


def _paired_diffs(cells: list[Cell], names: tuple[str, ...], policy: str, ref: str, rec: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in cells:
        if c.name in names and policy in c.frames and ref in c.frames:
            out[c.name] = (
                c.frames[policy][f"regret_{rec}"].to_numpy(dtype=np.float64)
                - c.frames[ref][f"regret_{rec}"].to_numpy(dtype=np.float64)
            )
    return out


def contrast_via_stats(
    cells: list[Cell], s: Stratum, policy: str, ref: str, rec: str, n_boot: int
) -> dict:
    """One contrast through the landed `stats` functions (used for the pre-registered rows)."""
    diffs = _paired_diffs(cells, s.cells, policy, ref, rec)
    if not diffs:
        return {"status": f"missing:{policy if not any(policy in c.frames for c in cells) else ref}",
                "delta": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "win": np.nan,
                "n_cells": 0, "n_episodes": 0, "n_envs": 0, "cluster_lo": np.nan, "cluster_hi": np.nan}
    seed = _seed("primary", s.level, s.family, s.horizon, s.cell, policy, ref, rec)
    if len(diffs) == 1:
        d = next(iter(diffs.values()))
        b = stats.paired_bootstrap(d, n_boot=n_boot, seed=seed)
        mean, lo, hi, se = b["mean"], b["lo"], b["hi"], b["se_paired"]
        win = b["frac_negative"] + 0.5 * b["frac_zero"]
        cell_means = {next(iter(diffs)): b["mean"]}
    else:
        b = stats.stratified_pooled(diffs, n_boot=n_boot, seed=seed)
        mean, lo, hi, se = b["mean"], b["lo"], b["hi"], b["se"]
        win = float(np.mean([win_rate(d) for d in diffs.values()]))
        cell_means = b["cell_means"]
    env_of = {c.name: c.env_id for c in cells}
    cm = pd.DataFrame({"env_id": [env_of[n] for n in cell_means], "value": list(cell_means.values())})
    cluster_lo = cluster_hi = np.nan
    if cm["env_id"].nunique() >= 2:
        cb = stats.cluster_bootstrap_over_envs(cm, "env_id", "value", n_boot=n_boot, seed=seed + 1)
        cluster_lo, cluster_hi = cb["lo"], cb["hi"]
    return {"status": "ok", "delta": mean, "lo": lo, "hi": hi, "se": se, "win": win,
            "n_cells": len(diffs), "n_episodes": int(sum(d.size for d in diffs.values())),
            "n_envs": int(cm["env_id"].nunique()), "cluster_lo": cluster_lo, "cluster_hi": cluster_hi}


def primary_contrasts(
    test: str, rec: str, cells: list[Cell], strata: list[Stratum], n_boot: int
) -> pd.DataFrame:
    """Exactly the three pre-registered rows per stratum, whether or not both policies ran."""
    cells_by_name = {c.name: c for c in cells}
    rows: list[dict] = []
    for s in strata:
        if s.level == "cell":
            continue
        for hyp, policy, ref, label in PRIMARY_HYPOTHESES:
            r = contrast_via_stats(cells, s, policy, ref, rec, n_boot)
            row = _stratum_meta(test, rec, s, cells_by_name)
            row.update({"hypothesis": hyp, "pre_registered": True, "label": label,
                        "policy": policy, "reference": ref, **r, "n_boot": n_boot})
            rows.append(row)
    return pd.DataFrame(rows)


def secondary_contrasts(
    test: str, rec: str, cells: list[Cell], per_cell: dict[str, CellStats], strata: list[Stratum]
) -> pd.DataFrame:
    """Every other (policy, reference) pair with reference in `CONTRAST_REFS`, exploratory."""
    cells_by_name = {c.name: c for c in cells}
    primary = {(p, r) for _, p, r, _ in PRIMARY_HYPOTHESES}
    policies = sorted({p for c in cells for p in c.policies})
    rows: list[dict] = []
    for s in strata:
        if s.level == "cell":
            continue
        for ref in CONTRAST_REFS:
            for policy in policies:
                if policy == ref or (policy, ref) in primary:
                    continue
                a = aggregate(per_cell, s, policy, f"d_regret_vs_{ref}")
                if a["n_cells"] == 0:
                    continue
                row = _stratum_meta(test, rec, s, cells_by_name)
                row.update({"hypothesis": "", "pre_registered": False, "label": "exploratory",
                            "policy": policy, "reference": ref, "status": "ok",
                            "delta": a["mean"], "lo": a["lo"], "hi": a["hi"], "se": a["se"],
                            "win": a["win"], "n_cells": a["n_cells"], "n_episodes": a["n_episodes"],
                            "n_envs": a["n_envs"], "cluster_lo": a["cluster_lo"],
                            "cluster_hi": a["cluster_hi"]})
                rows.append(row)
    return pd.DataFrame(rows)


# ---- offline vs deployed ----------------------------------------------------------------------


def canonical_policy_of_variant() -> dict[str, str]:
    """variant -> the policy that deploys it with default mechanics (tau_val, committed, no guard)."""
    out: dict[str, str] = {}
    for name, entry in pt.POLICIES.items():
        if entry["kind"] != "model":
            continue
        p = entry["params"]
        if p.get("tau") is None and not p.get("per_step") and not p.get("affordability_guard"):
            out.setdefault(p["artifact"], name)
    return out


def offline_vs_deployed(
    test: str, main: pd.DataFrame, offline_metrics: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per learned variant (its canonical policy): offline OOF metrics next to the
    deployed pooled regret; then `surrogate_validity` = Spearman(offline, deployed) (H3)."""
    variant_of = canonical_policy_of_variant()
    rows: list[dict] = []
    recs = sorted(main["recommender"].unique())
    for variant, policy in variant_of.items():
        off = {}
        if offline_metrics is not None:
            sub = offline_metrics[(offline_metrics["variant"] == variant) & (offline_metrics["grouping"] == "meta_env")]
            if len(sub):
                r = sub.iloc[0]
                off = {"oof_auc_env": float(r["auc"]), "oof_bal_acc_tau_off": float(r["bal_acc_tau_off"]),
                       "tau_off": float(r["tau_off"]), "oof_bal_acc_05": float(r["bal_acc_05"]),
                       "k": int(r["k"]), "feature_set": str(r["feature_set"]), "n_features": int(r["n_features"])}
        for rec in recs:
            sub = main[(main["recommender"] == rec) & (main["policy"] == policy)]
            pooled = sub[sub["level"] == "pooled"]
            if pooled.empty:
                continue
            p = pooled.iloc[0]
            row = {"test": test, "recommender": rec, "variant": variant, "policy": policy,
                   "oof_auc_env": np.nan, "oof_bal_acc_tau_off": np.nan, "tau_off": np.nan,
                   "oof_bal_acc_05": np.nan, "k": np.nan, "feature_set": "", "n_features": np.nan}
            row.update(off)
            row.update({
                "regret_pooled": float(p["regret"]), "regret_lo": float(p["regret_lo"]),
                "regret_hi": float(p["regret_hi"]), "d_regret_vs_cp0": float(p["d_regret_vs_cp0"]),
                "d_regret_vs_cp0_lo": float(p["d_regret_vs_cp0_lo"]),
                "d_regret_vs_cp0_hi": float(p["d_regret_vs_cp0_hi"]),
                "search_frac": float(p["search_frac"]), "n_cells": int(p["n_cells"]),
            })
            for fam in sorted(main["family"].unique()):
                if fam == "all":
                    continue
                f = sub[(sub["level"] == "family") & (sub["family"] == fam)]
                row[f"regret_family_{fam}"] = float(f["regret"].iloc[0]) if len(f) else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    validity_rows: list[dict] = []
    if not table.empty:
        for rec in recs:
            sub = table[table["recommender"] == rec]
            targets = [("regret_pooled", "pooled")] + [
                (c, c.replace("regret_family_", "family:")) for c in sub.columns if c.startswith("regret_family_")
            ]
            for offline_col, offline_name in (("oof_auc_env", "oof_auc"), ("oof_bal_acc_tau_off", "oof_bal_acc")):
                for target_col, target_name in targets:
                    ok = sub[offline_col].notna() & sub[target_col].notna()
                    n = int(ok.sum())
                    row = {"test": test, "recommender": rec, "hypothesis": "H3", "offline": offline_name,
                           "deployed": target_name, "n_variants": n,
                           "rho": np.nan, "lo": np.nan, "hi": np.nan}
                    if n >= 3:
                        sp = stats.spearman_with_ci(
                            sub.loc[ok, offline_col].to_numpy(), sub.loc[ok, target_col].to_numpy(),
                            n_boot=2000, seed=_seed("h3", rec, offline_name, target_name),
                        )
                        row.update({"rho": sp["rho"], "lo": sp["lo"], "hi": sp["hi"]})
                    validity_rows.append(row)
    return table, pd.DataFrame(validity_rows)


# ---- tau curves ---------------------------------------------------------------------------------


def tau_curves(
    test: str, main: pd.DataFrame, threshold_selection: pd.DataFrame | None, manifest: dict[tuple[str, str, str], dict]
) -> pd.DataFrame:
    """Validation `tau -> pooled regret` per variant, plus the post-hoc test-set points for
    phi_k16 (deployed tau and the fixed-0.5 twin), labelled ``posthoc=True``."""
    rows: list[dict] = []
    if threshold_selection is not None and len(threshold_selection):
        ts = threshold_selection
        if "split" in ts.columns:
            ts = ts[ts["split"] == "val"]
        for (variant, tau), grp in ts.groupby(["variant", "tau"], sort=True):
            rows.append({
                "test": test, "source": "validation", "posthoc": False, "variant": str(variant),
                "policy": "", "tau": float(tau), "n_cells": int(len(grp)),
                "regret": float(grp["mean_regret"].mean()),
                "regret_lo": np.nan, "regret_hi": np.nan,
                "search_frac": float(grp["mean_search_frac"].mean()) if "mean_search_frac" in grp else np.nan,
            })
    primary = main[(main["recommender"] == PRIMARY_RECOMMENDER) & (main["level"] == "pooled")]
    for policy in ("phi_k16", "phi_k16_tau05"):
        sub = primary[primary["policy"] == policy]
        if sub.empty:
            continue
        taus = {rec["params"].get("tau") for (t, c, p), rec in manifest.items() if p == policy and rec.get("params")}
        taus.discard(None)
        p = sub.iloc[0]
        rows.append({
            "test": test, "source": f"test:{test}", "posthoc": True,
            "variant": pt.variant_of(policy) or "", "policy": policy,
            "tau": float(next(iter(taus))) if len(taus) == 1 else np.nan,
            "n_cells": int(p["n_cells"]), "regret": float(p["regret"]),
            "regret_lo": float(p["regret_lo"]), "regret_hi": float(p["regret_hi"]),
            "search_frac": float(p["search_frac"]),
        })
    return pd.DataFrame(rows)


# ---- recommender sensitivity ------------------------------------------------------------------


def recommender_sensitivity(test: str, cells_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank of every policy under every recommender per cell (1 = lowest regret), and the
    Kendall tau between recommenders' rankings per cell and pooled."""
    from scipy.stats import kendalltau

    ranks = cells_table[["test", "recommender", "cell", "family", "horizon", "policy", "group", "regret"]].copy()
    ranks["rank"] = ranks.groupby(["recommender", "cell"])["regret"].rank(method="average")
    recs = sorted(ranks["recommender"].unique())
    rows: list[dict] = []
    for cell, grp in ranks.groupby("cell", sort=True):
        wide = grp.pivot(index="policy", columns="recommender", values="regret")
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                if a not in wide or b not in wide:
                    continue
                ok = wide[a].notna() & wide[b].notna()
                tau = np.nan
                if ok.sum() >= 3 and wide.loc[ok, a].nunique() > 1 and wide.loc[ok, b].nunique() > 1:
                    tau = float(kendalltau(wide.loc[ok, a], wide.loc[ok, b]).statistic)
                rows.append({"test": test, "level": "cell", "cell": cell, "rec_a": a, "rec_b": b,
                             "n_policies": int(ok.sum()), "kendall_tau": tau})
    kendall = pd.DataFrame(rows)
    if not kendall.empty:
        pooled = kendall.groupby(["rec_a", "rec_b"], sort=True)["kendall_tau"].mean().reset_index()
        pooled.insert(0, "test", test)
        pooled.insert(1, "level", "pooled")
        pooled.insert(2, "cell", "all")
        pooled["n_policies"] = kendall.groupby(["rec_a", "rec_b"])["n_policies"].max().to_numpy()
        kendall = pd.concat([kendall, pooled[kendall.columns]], ignore_index=True)
    return ranks, kendall


# ---- dynamics and cap tables ---------------------------------------------------------------------


def dynamics_table(out_dir: Path, test: str, cells: list[Cell]) -> pd.DataFrame:
    """Long frame (cell, policy, t_frac, metric, value) from every dynamics ``.npz``."""
    rows: list[pd.DataFrame] = []
    root = out_dir / "dynamics" / test
    for c in cells:
        for policy in c.policies:
            path = root / c.name / f"{policy}.npz"
            if not path.exists():
                continue
            with np.load(path) as d:
                t = d["t"].astype(np.float64)
                for key in DYNAMICS_KEYS:
                    if key == "t" or key not in d:
                        continue
                    rows.append(pd.DataFrame({
                        "test": test, "cell": c.name, "family": c.family, "horizon": c.horizon,
                        "policy": policy, "group": c.groups[policy], "t": t, "t_frac": t / c.horizon,
                        "metric": key, "value": d[key].astype(np.float64),
                    }))
    if not rows:
        return pd.DataFrame(columns=["test", "cell", "family", "horizon", "policy", "group", "t", "t_frac", "metric", "value"])
    return pd.concat(rows, ignore_index=True)


def cap_demotion_table(test: str, cells: list[Cell], manifest: dict[tuple[str, str, str], dict]) -> pd.DataFrame:
    """Per (cell, policy): demotions, cap hits, the `t_cap_hit` distribution and, for learned
    policies, the share of fresh model decisions that were SEARCH (manifest counters)."""
    rows: list[dict] = []
    for c in cells:
        for policy, df in c.frames.items():
            hit = df["cap_hit"].to_numpy(dtype=bool)
            t_hit = df["t_cap_hit"].to_numpy(dtype=np.float64)[hit]
            demoted = df["n_demoted"].to_numpy(dtype=np.float64)
            row = {
                "test": test, "cell": c.name, "env_id": c.env_id, "family": c.family,
                "horizon": c.horizon, "cap": c.cap, "policy": policy, "group": c.groups[policy],
                "n": c.n, "cap_hit_frac": float(hit.mean()),
                "n_demoted_mean": float(demoted.mean()) if bool(np.all(demoted >= 0)) else np.nan,
                "n_demoted_max": float(demoted.max()) if bool(np.all(demoted >= 0)) else np.nan,
                "frac_episodes_demoted": float(np.mean(demoted > 0)) if bool(np.all(demoted >= 0)) else np.nan,
                "k_final_mean": float(df["k_final"].mean()),
                "search_frac_mean": float(df["search_frac"].mean()),
            }
            qs = (0.0, 0.25, 0.5, 0.75, 1.0)
            for q in qs:
                row[f"t_cap_hit_q{int(q * 100):02d}"] = float(np.quantile(t_hit, q)) if t_hit.size else np.nan
            row["t_cap_hit_frac_of_T_median"] = (
                float(np.median(t_hit) / c.horizon) if t_hit.size else np.nan
            )
            counters = (manifest.get((test, c.name, policy)) or {}).get("counters") or {}
            n_dec = counters.get("n_decisions")
            row.update({
                "n_decisions": n_dec if n_dec is not None else np.nan,
                "n_search_decided": counters.get("n_search_decided", np.nan),
                "frac_decisions_search": (
                    counters["n_search_decided"] / n_dec if n_dec else np.nan
                ),
                "n_committed_steps": counters.get("n_committed_steps", np.nan),
                "n_guard_vetoes": counters.get("n_guard_vetoes", np.nan),
                "n_cap_demoted_counter": counters.get("n_cap_demoted", np.nan),
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ---- snapshot diagnostics (parity gate, OOD, reservoir) ------------------------------------------


_CORPUS: pd.DataFrame | None = None
_STANDARDIZERS: dict[tuple[int, tuple[str, ...]], tuple[ood.Standardizer, float]] = {}
_OOF: pd.DataFrame | None = None


def _corpus() -> pd.DataFrame:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = ood.load_corpus_reference()
    return _CORPUS


def _standardizer(horizon: int, features: tuple[str, ...], corpus_rows: pd.DataFrame):
    key = (int(horizon), tuple(features))
    if key not in _STANDARDIZERS:
        st = ood.Standardizer.fit(corpus_rows.loc[:, list(features)].to_numpy(), features)
        thr = ood.self_knn_threshold(st.transform(corpus_rows.loc[:, list(features)].to_numpy()))
        _STANDARDIZERS[key] = (st, thr)
    return _STANDARDIZERS[key]


@dataclass(frozen=True)
class DiagItem:
    item: ood.SnapshotItem
    offline_path: str | None
    oof_path: str | None
    features_cache: str | None
    skip_ood: bool


def _worker_init(log_level: int) -> None:
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s")
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover
        pass


def _with_meta(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """`meta` columns first, then the frame's own."""
    if df.empty:
        return df
    out = df.assign(**meta)
    return out.loc[:, list(meta) + [c for c in df.columns if c not in meta]]


def run_diag_item(d: DiagItem) -> dict:
    """Parity + OOD + reservoir rows for one (cell, policy); the feature frame goes to disk."""
    global _OOF
    item = d.item
    t0 = time.perf_counter()
    res = ood.snapshot_pass(item)
    feats = res.features
    meta = {"test": item.test, "cell": item.cell, "policy": item.policy, "env_id": item.env_id,
            "family": item.family, "horizon": int(item.horizon), "cap": int(item.cap)}

    reservoir = build_reservoir(item.env_spec)
    feats = ra.with_oracle_columns(feats, reservoir)
    offline = pd.read_csv(d.offline_path) if d.offline_path and Path(d.offline_path).exists() else None
    reservoir_row = ra.reservoir_row(feats, meta, offline)
    reservoir_bins = ra.decision_by_oracle_bins(feats, meta)

    ood_summary: dict = dict(meta)
    ood_features = pd.DataFrame()
    score_hist = pd.DataFrame()
    if not d.skip_ood:
        corpus = _corpus()
        corpus_rows, scope = ood.corpus_rows_for(corpus, item.horizon)
        st, thr = _standardizer(item.horizon, res.policy_features, corpus_rows)
        p_corpus = None
        kind = "none"
        if item.kind == "model" and item.artifact_path:
            variant = pt.variant_of(item.policy)
            oof = None
            if d.oof_path and Path(d.oof_path).exists():
                if _OOF is None:
                    _OOF = pd.read_parquet(d.oof_path)
                oof = _OOF
            if oof is not None and variant in oof.columns and len(oof) == len(corpus):
                p_corpus = oof[variant].to_numpy(dtype=np.float64)[corpus_rows.index.to_numpy()]
                kind = "oof"
            else:
                from cold_start.growing.deploy.artifacts import load_model

                p_corpus = ood.model_scores(load_model(item.artifact_path), corpus_rows)
                kind = "insample_refit"
        r = ood.ood_for_policy(
            feats, corpus_rows, scope, res.policy_features, standardizer=st, knn_threshold=thr,
            p_corpus=p_corpus, seed=_seed("ood", item.cell, item.policy),
        )
        ood_summary.update(r.summary)
        ood_summary["corpus_score_kind"] = kind
        ood_features = _with_meta(r.features, meta)
        score_hist = _with_meta(r.score_hist, meta)
        score_hist["corpus_score_kind"] = kind

    if d.features_cache:
        path = Path(d.features_cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        feats.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    return {
        "cell": item.cell, "policy": item.policy, "n_states": int(len(feats)),
        "seconds": time.perf_counter() - t0,
        "parity": res.parity, "ood_summary": ood_summary, "ood_features": ood_features,
        "score_hist": score_hist, "reservoir_row": reservoir_row, "reservoir_bins": reservoir_bins,
    }


def build_diag_items(
    out_dir: Path, test: str, cells: list[Cell], manifest: dict[tuple[str, str, str], dict],
    *, sample: int | None, skip_ood: bool, oof_path: Path | None, offline_path: Path,
) -> list[DiagItem]:
    """One item per logged (cell, policy) of the test whose pickle exists on disk."""
    items: list[DiagItem] = []
    env_specs = {c.name: c for c in cells}
    for (t, cell, policy), rec in sorted(manifest.items()):
        if t != test or cell not in env_specs:
            continue
        path = rec.get("snapshots")
        if not path or not Path(path).exists():
            continue
        entry = pt.POLICIES.get(policy)
        if entry is None or entry["kind"] not in ("model", "reservoir_rule"):
            continue
        c = env_specs[cell]
        params = rec.get("params") or {}
        env_spec = _env_spec_of(c.env_id)
        item = ood.SnapshotItem(
            test=test, cell=cell, policy=policy, env_id=c.env_id, family=c.family,
            horizon=c.horizon, cap=c.cap, env_spec=env_spec, snapshots_path=str(path),
            kind=entry["kind"], artifact_path=params.get("artifact") if entry["kind"] == "model" else None,
            tau=params.get("tau"), sample=sample, seed=_seed("sample", cell, policy),
        )
        items.append(DiagItem(
            item=item, offline_path=str(offline_path) if offline_path.exists() else None,
            oof_path=str(oof_path) if oof_path is not None else None,
            features_cache=str(out_dir / "onpolicy" / test / cell / f"{policy}.parquet"),
            skip_ood=skip_ood,
        ))
    return items


def ensure_oof_scores(out_dir: Path, variants: list[str], n_jobs: int) -> Path:
    """`oof_scores.parquet` (one column per variant, one row per corpus row), computing
    only the variants it does not hold yet."""
    from corpus import load_corpus

    path = out_dir / "oof_scores.parquet"
    have = pd.read_parquet(path) if path.exists() else None
    missing = [v for v in variants if have is None or v not in have.columns]
    if not missing:
        return path
    log.info("computing OOF scores for %d variant(s) (trainer protocol): %s", len(missing), missing)
    corpus_full = load_corpus()
    fresh = ood.corpus_oof_scores(corpus_full, missing, n_jobs=n_jobs)
    if have is not None and len(have) == len(fresh):
        for v in missing:
            have[v] = fresh[v].to_numpy()
        fresh = have
    tmp = path.with_suffix(".parquet.tmp")
    fresh.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def _env_spec_of(env_id: str) -> dict:
    import cells as cells_mod

    return cells_mod.ALL_ENVS[env_id]


def run_diagnostics(items: list[DiagItem], workers: int) -> list[dict]:
    if not items:
        return []
    results: list[dict] = []
    t0 = time.time()
    if workers <= 1:
        for i, d in enumerate(items, 1):
            results.append(run_diag_item(d))
            log.info("[%d/%d] %s/%s %.1fs", i, len(items), d.item.cell, d.item.policy, results[-1]["seconds"])
        return results
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(workers), initializer=_worker_init, initargs=(logging.getLogger().level or logging.INFO,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_diag_item, items, chunksize=1), 1):
            results.append(r)
            log.info("[%d/%d] %s/%s %.1fs (elapsed %.0fs)", i, len(items), r["cell"], r["policy"], r["seconds"], time.time() - t0)
    return results


def prebuild_tables(cells: list[Cell]) -> None:
    """CS and pairwise tables for every horizon, in the parent, before any worker exists."""
    from cold_start.growing.deploy.pairwise_table import MAX_TABULATED_N, get_pairwise_table
    from cold_start.growing.tables import CSTable

    for T in sorted({c.horizon for c in cells}):
        CSTable.load_or_build(T)
        if T <= MAX_TABULATED_N:
            get_pairwise_table(T)


# ---- CLI --------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", required=True, choices=rd.TESTS)
    p.add_argument("--recommender", default="all", help="all | one of " + ", ".join(RECOMMENDER_NAMES))
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tables-dir", type=Path, default=None, help="default: <out-dir>/tables")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--parity-sample", type=int, default=0,
                   help="snapshots per (cell, policy) for the diagnostics pass (0 = all)")
    p.add_argument("--skip-snapshots", action="store_true",
                   help="reuse an existing, passing onpolicy_parity table instead of re-scoring")
    p.add_argument("--skip-ood", action="store_true", help="parity + reservoir only (no corpus load)")
    p.add_argument("--no-oof", action="store_true",
                   help="score corpus rows with the deployed refit model instead of out-of-fold "
                        "(the OOF pass follows the trainer's protocol, ~2 min, cached in oof_scores.parquet)")
    p.add_argument("--gate-from", default=None, metavar="TEST",
                   help="for a test run without --log-states: accept another test's passing "
                        "onpolicy_parity table (the same policies and feature layer) as the gate")
    p.add_argument("--allow-unverified", action="store_true",
                   help="write main tables even when no on-policy snapshots exist to verify parity")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    test = args.test
    out_dir = Path(args.out_dir)
    tables_dir = Path(args.tables_dir) if args.tables_dir else out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    n_boot = int(args.n_boot)
    t_run = time.time()
    written: list[Path] = []

    cells = load_cells(out_dir, test)
    manifest = rd.latest_records(rd.read_manifest(rd.manifest_path(out_dir, test)))
    recs_present = recommenders_in(cells[0].frames[cells[0].policies[0]])
    if args.recommender == "all":
        recs = recs_present
    else:
        if args.recommender not in recs_present:
            raise SystemExit(f"recommender {args.recommender!r} not in the frames: {recs_present}")
        recs = [args.recommender]
    if PRIMARY_RECOMMENDER in recs:
        recs = [PRIMARY_RECOMMENDER] + [r for r in recs if r != PRIMARY_RECOMMENDER]

    # ---- 1. on-policy parity gate + diagnostics ------------------------------------------
    parity_path = tables_dir / table_name("onpolicy_parity", test)
    gate_ok = False
    n_states = 0
    if args.skip_snapshots and parity_path.exists():
        prev = pd.read_csv(parity_path)
        n_states = int(prev["n_rows"].max()) if "n_rows" in prev and len(prev) else 0
        gate_ok = bool(prev["passed"].all()) and n_states >= ood.PARITY_MIN_STATES if "passed" in prev else False
        log.info("reusing %s: passed=%s, n_states=%d", parity_path.name, gate_ok, n_states)
    else:
        items = build_diag_items(
            out_dir, test, cells, manifest, sample=args.parity_sample or None,
            skip_ood=args.skip_ood, oof_path=None,
            offline_path=out_dir / "reservoir_diagnostics_offline.csv",
        )
        log.info("diagnostics: %d logged (cell, policy) items", len(items))
        if items:
            prebuild_tables(cells)
            if not args.no_oof and not args.skip_ood:
                variants = sorted({pt.variant_of(d.item.policy) for d in items} - {None})
                oof_path = ensure_oof_scores(out_dir, variants, n_jobs=int(args.workers))
                items = [replace(d, oof_path=str(oof_path)) for d in items]
        results = run_diagnostics(items, int(args.workers))
        if results:
            detail = pd.concat([r["parity"] for r in results], ignore_index=True)
            summary = ood.parity_summary(detail)
            written.append(write_csv(summary, parity_path))
            written.append(write_csv(detail, tables_dir / table_name("onpolicy_parity_detail", test)))
            n_states = int(sum(r["n_states"] for r in results))
            failures = ood.parity_failures(detail)
            gate_ok = not failures and n_states >= ood.PARITY_MIN_STATES
            if failures:
                worst = summary.set_index("column").loc[failures, ["max_abs_diff", "n_fail"]]
                log.error("PARITY GATE FAILED on %d column(s):\n%s", len(failures), worst.to_string())
            elif n_states < ood.PARITY_MIN_STATES:
                log.error("parity gate undersized: %d on-policy states < %d", n_states, ood.PARITY_MIN_STATES)
            else:
                log.info("parity gate passed on %d on-policy states (max diff %.2e)",
                         n_states, float(summary["max_abs_diff"].max()))
            if not args.skip_ood:
                ood_summary = pd.DataFrame([r["ood_summary"] for r in results])
                written.append(write_csv(ood_summary, tables_dir / f"ood_{test}.csv"))
                written.append(write_csv(ood.flag_cells(ood_summary), tables_dir / table_name("ood_flags", test)))
                feats = [r["ood_features"] for r in results if len(r["ood_features"])]
                if feats:
                    written.append(write_csv(pd.concat(feats, ignore_index=True), tables_dir / f"ood_features_{test}.csv"))
                hists = [r["score_hist"] for r in results if len(r["score_hist"])]
                if hists:
                    written.append(write_csv(pd.concat(hists, ignore_index=True), tables_dir / f"ood_score_hist_{test}.csv"))
                n_flag = int((ood_summary["ood_frac"] > ood.OOD_FLAG_THRESHOLD).sum()) if "ood_frac" in ood_summary else 0
                if n_flag:
                    log.warning("OOD: %d (cell, policy) items exceed %.0f%% OOD fraction -- see ood_flags",
                                n_flag, 100 * ood.OOD_FLAG_THRESHOLD)
            written.append(write_csv(pd.DataFrame([r["reservoir_row"] for r in results]), tables_dir / f"reservoir_{test}.csv"))
            bins = [r["reservoir_bins"] for r in results if len(r["reservoir_bins"])]
            if bins:
                written.append(write_csv(pd.concat(bins, ignore_index=True), tables_dir / f"reservoir_bins_{test}.csv"))
        else:
            log.warning("no logged snapshots for test %s: the parity gate cannot be evaluated", test)

    if not gate_ok and n_states == 0 and args.gate_from:
        other = tables_dir / table_name("onpolicy_parity", args.gate_from)
        if other.exists():
            prev = pd.read_csv(other)
            n_other = int(prev["n_rows"].max()) if len(prev) else 0
            gate_ok = bool(prev["passed"].all()) and n_other >= ood.PARITY_MIN_STATES
            log.warning("gate taken from test %s (%s): passed=%s on %d states", args.gate_from, other.name, gate_ok, n_other)
        else:
            log.error("--gate-from %s: %s does not exist", args.gate_from, other)
    if not gate_ok:
        if args.allow_unverified and n_states == 0:
            log.warning("--allow-unverified: writing main tables WITHOUT the on-policy parity gate")
        else:
            log.error("refusing to write the main tables for test %s (parity gate not passed)", test)
            return {"test": test, "gate_ok": False, "n_states": n_states, "written": [str(p) for p in written]}

    # ---- 2. main tables per recommender ------------------------------------------------------
    strata = strata_of(cells)
    main_by_rec: dict[str, pd.DataFrame] = {}
    cells_by_rec: dict[str, pd.DataFrame] = {}
    for rec in recs:
        t0 = time.perf_counter()
        per_cell = {c.name: compute_cell_stats(c, rec, n_boot) for c in cells}
        full = main_table(test, rec, cells, per_cell, strata)
        main = full[full["level"] != "cell"].reset_index(drop=True)
        cells_tab = full[full["level"] == "cell"].reset_index(drop=True)
        written.append(write_csv(main, tables_dir / table_name("main", test, rec)))
        written.append(write_csv(cells_tab, tables_dir / table_name("cells", test, rec)))
        written.append(write_csv(primary_contrasts(test, rec, cells, strata, n_boot),
                                 tables_dir / table_name("primary_contrasts", test, rec)))
        written.append(write_csv(secondary_contrasts(test, rec, cells, per_cell, strata),
                                 tables_dir / table_name("secondary_contrasts", test, rec)))
        written.append(write_csv(decomposition_table(test, rec, cells, per_cell, strata),
                                 tables_dir / table_name("decomposition", test, rec)))
        main_by_rec[rec] = main
        cells_by_rec[rec] = cells_tab
        log.info("recommender %s: tables in %.1fs", rec, time.perf_counter() - t0)
        del per_cell

    main_all = pd.concat(main_by_rec.values(), ignore_index=True)
    cells_all = pd.concat(cells_by_rec.values(), ignore_index=True)

    # ---- 3. cross-cutting tables -------------------------------------------------------------------
    offline_path = out_dir / "offline_metrics.csv"
    offline_metrics = pd.read_csv(offline_path) if offline_path.exists() else None
    ovd, validity = offline_vs_deployed(test, main_all, offline_metrics)
    written.append(write_csv(ovd, tables_dir / table_name("offline_vs_deployed", test)))
    written.append(write_csv(validity, tables_dir / table_name("surrogate_validity", test)))

    ts_path = out_dir / "threshold_selection.csv"
    ts = pd.read_csv(ts_path) if ts_path.exists() else None
    if ts is None:
        log.warning("%s absent: tau_curves holds only the post-hoc test points", ts_path.name)
    written.append(write_csv(tau_curves(test, main_all, ts, manifest), tables_dir / table_name("tau_curves", test)))

    ranks, kendall = recommender_sensitivity(test, cells_all)
    written.append(write_csv(ranks, tables_dir / table_name("recommender_sensitivity", test)))
    written.append(write_csv(kendall, tables_dir / table_name("recommender_kendall", test)))

    written.append(write_csv(dynamics_table(out_dir, test, cells), tables_dir / f"dynamics_{test}.csv"))
    written.append(write_csv(cap_demotion_table(test, cells, manifest), tables_dir / f"cap_demotion_{test}.csv"))

    if test in PER_TEST_TABLE_NAMES:
        # The per-cell table under the plan's name (transfer_D, heldout_C, regime_B,
        # cap_sweep): the primary recommender's per-cell rows plus the paired delta vs
        # phi_k16, which is the comparison those tests are about.
        rec0 = PRIMARY_RECOMMENDER if PRIMARY_RECOMMENDER in recs else recs[0]
        extra = cells_by_rec[rec0].merge(
            _per_cell_delta_vs(cells, PRIMARY_POLICY, rec0, n_boot), on=["cell", "policy"], how="left"
        )
        written.append(write_csv(extra, tables_dir / f"{PER_TEST_TABLE_NAMES[test]}.csv"))

    log.info("test %s: %d tables in %.1fs", test, len(written), time.time() - t_run)
    return {"test": test, "gate_ok": gate_ok, "n_states": n_states, "written": [str(p) for p in written]}


def _per_cell_delta_vs(cells: list[Cell], ref: str, rec: str, n_boot: int) -> pd.DataFrame:
    """Paired delta vs `ref` per (cell, policy) through `stats.paired_bootstrap`."""
    rows: list[dict] = []
    for c in cells:
        if ref not in c.frames:
            continue
        ref_regret = c.frames[ref][f"regret_{rec}"].to_numpy(dtype=np.float64)
        for policy, df in c.frames.items():
            d = df[f"regret_{rec}"].to_numpy(dtype=np.float64) - ref_regret
            b = stats.paired_bootstrap(d, n_boot=n_boot, seed=_seed("percell", c.name, policy, ref, rec))
            rows.append({"cell": c.name, "policy": policy, f"d_regret_vs_{ref}": b["mean"],
                         f"d_regret_vs_{ref}_lo": b["lo"], f"d_regret_vs_{ref}_hi": b["hi"],
                         f"d_regret_vs_{ref}_win": b["frac_negative"] + 0.5 * b["frac_zero"]})
    if not rows:
        return pd.DataFrame(columns=["cell", "policy"])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
