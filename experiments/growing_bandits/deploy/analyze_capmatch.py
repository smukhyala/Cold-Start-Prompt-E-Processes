"""The K-matched control (NEXT-STEPS 2.4): is Φ a learned rule, or only a K_final chooser?

Every deployed learned policy ends an episode holding some number of arms, K_final. The
tables compare it against schedules that hold a different number. The control that the
question "does the *timing* of SEARCH matter?" needs is the dumbest policy that holds the
same number: `always_search` capped at the learned policy's own realised K_final, on the
same episodes. `run_deployment.py --test capmatch --match <policy>` deploys that control
(plus `uniform`) and the policy itself in a cell per (env, T) whose cap is
``round(k_final)`` from `cells_A_primary.csv` / `cells_C_primary.csv` and whose seed is the
deployed cell's, so the three are CRN-paired. Cells whose K_final is already the 64-arm
cap are the deployed cells themselves and are read from Test A / C directly.

Statistic: Δ = regret(policy) − regret(`always_search` @ K) per episode, then the same
paired / stratified / environment-mean-t machinery as the main tables
(`stats.paired_bootstrap`, `stats.stratified_pooled`, `analyze_deployment.cluster_bounds`).
Negative favours the learned policy.

How to read it: the per-cell cap is data-dependent (it is the policy's own endpoint), so
this is a **diagnostic control, never a pre-registered hypothesis**. `always_search` at
cap K reaches K arms by round K and refines from then on, far earlier than a learned
policy that reaches K at the horizon, so **a tie is stronger evidence than a loss**: if Φ
ties its own K-matched control, "the learned SEARCH-vs-REFINE policy" is a device for
choosing a final arm count, and the honest description is "we learned a growth schedule".
If Φ beats it at short horizons there is a state-dependent rule with a valid control for
the first time. Two limitations travel with the number: the cap is the *mean* K_final
(episodes above it are clipped, so the in-cell policy is not exactly the deployed one --
`k_final` next to `k_final_deployed` measures that), and τ stays at the value selected at
cap 64 (Ruling 20's asymmetry, unchanged here: re-selecting τ would make it a different
policy from the one whose K was matched).

Usage::

    .venv/bin/python experiments/growing_bandits/deploy/analyze_capmatch.py \\
        --match phi_k4,phi_k1,phi_k16,phi_k16_perstep,phi_k16_notrunc,p3_star

writes ``tables/capmatch_contrasts.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analyze_deployment as ad  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy import stats  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402

log = logging.getLogger("deploy.capmatch")

DEFAULT_MATCH: tuple[str, ...] = (
    "phi_k4", "phi_k1", "phi_k16", "phi_k16_perstep", "phi_k16_notrunc", "p3_star",
)
PRIMARY_REFERENCE = "always_search"

COLUMNS: tuple[str, ...] = (
    "policy", "reference", "level", "test", "family", "horizon", "cell", "env_id", "cap",
    "source_test", "delta", "lo", "hi", "se", "win", "n_cells", "n_episodes", "n_envs",
    *ad.CLUSTER_KEYS, "regret", "regret_ref", "k_final", "k_final_deployed", "k_final_ref",
    "search_frac", "search_frac_ref",
)


def _episodes(out_dir: Path, test: str, cell: str, policy: str) -> pd.DataFrame:
    path = out_dir / "episodes" / test / cell / f"{policy}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{policy} has no episodes in {test}/{cell}: {path}")
    return pd.read_parquet(path)


def _paired(a: pd.DataFrame, b: pd.DataFrame, cell: str) -> np.ndarray:
    """Per-episode regret difference a − b, aligned on the episode index (CRN)."""
    if not np.array_equal(a["episode"].to_numpy(), b["episode"].to_numpy()):
        raise ValueError(f"{cell}: episode indices differ between the two policies -- not paired")
    if int(a["base_seed"].iloc[0]) != int(b["base_seed"].iloc[0]):
        raise ValueError(f"{cell}: base_seed differs between the two policies -- not paired")
    col = f"regret_{PRIMARY_RECOMMENDER}"
    return a[col].to_numpy(dtype=np.float64) - b[col].to_numpy(dtype=np.float64)


def capmatch_contrasts(
    match: str,
    *,
    out_dir: Path,
    horizons: tuple[int, ...] = rd.CAPMATCH_HORIZONS,
    n_boot: int = 10_000,
    references: tuple[str, ...] = rd.CAPMATCH_COMPARATORS,
) -> pd.DataFrame:
    """Every level of the K-matched contrast for one learned policy, against each reference."""
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    grid = rd.matched_grid(match, tables_dir=tables, horizons=horizons, include_at_cap=True)
    deployed_k = {}
    for test in rd.CAPMATCH_SOURCE_TESTS:
        path = tables / f"cells_{test}_primary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[frame["policy"] == match]
            for row in frame.itertuples(index=False):
                deployed_k[(str(row.env_id), int(row.horizon))] = float(row.k_final)

    rows: list[dict] = []
    for ref in references:
        per_cell: list[dict] = []
        diffs: dict[str, np.ndarray] = {}
        for source_test, env_id, horizon, k in grid:
            at_cap = k >= rd.DEFAULT_CAP
            test_dir = source_test if at_cap else "capmatch"
            cell = f"{env_id}_T{horizon}_cap{k}"
            a = _episodes(out_dir, test_dir, cell, match)
            b = _episodes(out_dir, test_dir, cell, ref)
            d = _paired(a, b, cell)
            seed = ad._seed("capmatch", match, ref, cell)
            pb = stats.paired_bootstrap(d, n_boot=n_boot, seed=seed)
            col = f"regret_{PRIMARY_RECOMMENDER}"
            row = {
                "policy": match, "reference": ref, "level": "cell", "test": source_test,
                "family": str(a["family"].iloc[0]), "horizon": int(horizon), "cell": cell,
                "env_id": env_id, "cap": int(k), "source_test": test_dir,
                "delta": pb["mean"], "lo": pb["lo"], "hi": pb["hi"], "se": pb["se_paired"],
                "win": ad.win_rate(d), "n_cells": 1, "n_episodes": int(d.size), "n_envs": 1,
                **ad.EMPTY_CLUSTER,
                "regret": float(a[col].mean()), "regret_ref": float(b[col].mean()),
                "k_final": float(a["k_final"].mean()),
                "k_final_deployed": deployed_k.get((env_id, int(horizon)), np.nan),
                "k_final_ref": float(b["k_final"].mean()),
                "search_frac": float(a["search_frac"].mean()),
                "search_frac_ref": float(b["search_frac"].mean()),
            }
            per_cell.append(row)
            diffs[cell] = d
        cells = pd.DataFrame(per_cell)
        rows.extend(per_cell)

        # Pooled and per-horizon strata within each source test (A: 8 envs, C: 3).
        for test, sub in cells.groupby("test"):
            strata = [("pooled", None, sub)]
            for T, hsub in sub.groupby("horizon"):
                strata.append(("horizon", int(T), hsub))
            for level, T, ssub in strata:
                d_by_cell = {c: diffs[c] for c in ssub["cell"]}
                sp = stats.stratified_pooled(d_by_cell, n_boot=n_boot,
                                             seed=ad._seed("capmatch", match, ref, test, level, T))
                cm = pd.DataFrame({"env_id": ssub["env_id"].to_numpy(),
                                   "value": [sp["cell_means"][c] for c in ssub["cell"]]})
                bounds = ad.cluster_bounds(cm, n_boot=n_boot,
                                           seed=ad._seed("capmatch", match, ref, test, level, T) + 1)
                rows.append({
                    "policy": match, "reference": ref, "level": level, "test": test,
                    "family": "all", "horizon": T if T is not None else "all", "cell": "",
                    "env_id": "all", "cap": "matched", "source_test": "",
                    "delta": sp["mean"], "lo": sp["lo"], "hi": sp["hi"], "se": sp["se"],
                    "win": float(ssub["win"].mean()), "n_cells": int(len(ssub)),
                    "n_episodes": int(ssub["n_episodes"].sum()),
                    "n_envs": bounds["n_envs"], **{k: bounds[k] for k in ad.CLUSTER_KEYS},
                    "regret": float(ssub["regret"].mean()), "regret_ref": float(ssub["regret_ref"].mean()),
                    "k_final": float(ssub["k_final"].mean()),
                    "k_final_deployed": float(ssub["k_final_deployed"].mean()),
                    "k_final_ref": float(ssub["k_final_ref"].mean()),
                    "search_frac": float(ssub["search_frac"].mean()),
                    "search_frac_ref": float(ssub["search_frac_ref"].mean()),
                })
    return pd.DataFrame(rows)[list(COLUMNS)]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default=",".join(DEFAULT_MATCH),
                    help="comma-separated learned policies that were run under --test capmatch")
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <out-dir>/tables/capmatch_contrasts.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    frames = []
    for match in [m.strip() for m in args.match.split(",") if m.strip()]:
        frame = capmatch_contrasts(match, out_dir=args.out_dir, n_boot=args.n_boot)
        frames.append(frame)
        for _, r in frame[(frame["level"] == "pooled") & (frame["reference"] == PRIMARY_REFERENCE)].iterrows():
            log.info("%-18s test %s: Δ vs always_search@K = %+.6f paired [%+.6f, %+.6f] cluster [%+.6f, %+.6f] "
                     "p=%.3g n_envs=%d | K deployed %.1f, in-cell %.1f, control %.1f",
                     match, r["test"], r["delta"], r["lo"], r["hi"], r["cluster_lo"], r["cluster_hi"],
                     r["cluster_p"], r["n_envs"], r["k_final_deployed"], r["k_final"], r["k_final_ref"])
    out = pd.concat(frames, ignore_index=True)
    path = args.out or (args.out_dir / "tables" / "capmatch_contrasts.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(out))
    return out


if __name__ == "__main__":
    main()
