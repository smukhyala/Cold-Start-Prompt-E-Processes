"""The registered contrast H1b' (DEPLOYMENT_PLAN.md, "Pre-registration 2"), and nothing else.

One primary row: Δ = regret(policy) − regret(reference) per episode, pooled over the
registered horizons of one test with the cell-stratified paired bootstrap, and the
environment-mean t interval over the panel's environments with its two-sided `cluster_p`.
Three secondary rows (one per registered horizon) Holm-corrected among themselves. The
remaining horizons are reported as `structural`, uncorrected, because the registration
says the two policies cannot differ there.

The verdict is computed from the rule written down in the registration, so the script
cannot be talked into a different reading afterwards:

* ``supported``   iff ``cluster_p < 0.05`` and ``delta < 0`` and ``|delta| >= mei``;
* ``refuted``     iff ``delta >= 0`` or the t interval lies entirely above ``-mei``
                  (the effect is significantly smaller than the minimum of interest);
* ``inconclusive`` otherwise.

Usage::

    .venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py

writes ``tables/h1b_prime.csv``. The defaults ARE the registration; passing anything
else produces a different contrast and the output says which.
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

log = logging.getLogger("deploy.registered")

#: The registration, verbatim (DEPLOYMENT_PLAN.md, Pre-registration 2).
REGISTERED = {
    "policy": "phi_k4",
    "reference": "p3_star",
    "test": "robust",
    "horizons": (50, 100, 200),
    "mei": 0.002,
}
ALL_HORIZONS: tuple[int, ...] = (50, 100, 200, 500, 1000)

COLUMNS: tuple[str, ...] = (
    "row", "policy", "reference", "test", "horizon", "delta", "lo", "hi", "se", "win",
    "n_cells", "n_episodes", "n_envs", *ad.CLUSTER_KEYS, "p_holm", "mei", "verdict",
    "as_registered",
)


def _holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [np.nan] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def _diffs(out_dir: Path, test: str, policy: str, reference: str, horizons: tuple[int, ...]):
    """Per-episode paired differences by cell, with each cell's environment."""
    root = out_dir / "episodes" / test
    if not root.exists():
        raise FileNotFoundError(f"no episodes for test {test!r} under {out_dir}")
    col = f"regret_{PRIMARY_RECOMMENDER}"
    diffs: dict[str, np.ndarray] = {}
    env_of: dict[str, str] = {}
    horizon_of: dict[str, int] = {}
    for cell_dir in sorted(root.iterdir()):
        ref_path = cell_dir / f"{reference}.parquet"
        pol_path = cell_dir / f"{policy}.parquet"
        if not ref_path.exists():
            continue
        b = pd.read_parquet(ref_path)
        T = int(b["horizon"].iloc[0])
        if T not in horizons:
            continue
        if not pol_path.exists():
            raise FileNotFoundError(f"{policy} has no episodes in {test}/{cell_dir.name}")
        a = pd.read_parquet(pol_path)
        if not np.array_equal(a["episode"].to_numpy(), b["episode"].to_numpy()):
            raise ValueError(f"{cell_dir.name}: episode indices differ -- not paired")
        if int(a["base_seed"].iloc[0]) != int(b["base_seed"].iloc[0]):
            raise ValueError(f"{cell_dir.name}: base_seed differs -- not paired")
        diffs[cell_dir.name] = a[col].to_numpy(dtype=np.float64) - b[col].to_numpy(dtype=np.float64)
        env_of[cell_dir.name] = str(a["env_id"].iloc[0])
        horizon_of[cell_dir.name] = T
    if not diffs:
        raise FileNotFoundError(f"no cells of {test} at horizons {horizons} hold both {policy} and {reference}")
    return diffs, env_of, horizon_of


def _stratum(diffs: dict[str, np.ndarray], env_of: dict[str, str], *, n_boot: int, seed: int) -> dict:
    sp = stats.stratified_pooled(diffs, n_boot=n_boot, seed=seed)
    cm = pd.DataFrame({"env_id": [env_of[c] for c in diffs], "value": [sp["cell_means"][c] for c in diffs]})
    bounds = ad.cluster_bounds(cm, n_boot=n_boot, seed=seed + 1)
    return {
        "delta": sp["mean"], "lo": sp["lo"], "hi": sp["hi"], "se": sp["se"],
        "win": float(np.mean([ad.win_rate(d) for d in diffs.values()])),
        "n_cells": len(diffs), "n_episodes": int(sum(d.size for d in diffs.values())),
        "n_envs": bounds["n_envs"], **{k: bounds[k] for k in ad.CLUSTER_KEYS},
    }


def verdict(delta: float, cluster_lo: float, cluster_hi: float, cluster_p: float, mei: float) -> str:
    if not np.isfinite(cluster_p):
        return "inconclusive"
    if delta >= 0.0 or cluster_lo > -mei:
        return "refuted"
    if cluster_p < 0.05 and abs(delta) >= mei:
        return "supported"
    return "inconclusive"


def registered_contrast(
    policy: str, reference: str, *, test: str, horizons: tuple[int, ...], mei: float,
    out_dir: Path, n_boot: int = 10_000,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    horizons = tuple(int(h) for h in horizons)
    as_registered = (policy, reference, test, horizons, float(mei)) == (
        REGISTERED["policy"], REGISTERED["reference"], REGISTERED["test"],
        tuple(REGISTERED["horizons"]), float(REGISTERED["mei"]),
    )
    if not as_registered:
        log.warning("NOT the registered contrast: %s vs %s on %s at %s, mei %s",
                    policy, reference, test, horizons, mei)
    diffs, env_of, horizon_of = _diffs(out_dir, test, policy, reference, ALL_HORIZONS)
    base = {"policy": policy, "reference": reference, "test": test, "mei": float(mei),
            "as_registered": as_registered}
    rows: list[dict] = []

    primary_cells = {c: d for c, d in diffs.items() if horizon_of[c] in horizons}
    if not primary_cells:
        raise FileNotFoundError(f"no cells of {test} at the registered horizons {horizons}")
    s = _stratum(primary_cells, env_of, n_boot=n_boot, seed=ad._seed("registered", policy, reference, test))
    rows.append({**base, "row": "primary", "horizon": "all", **s, "p_holm": np.nan,
                 "verdict": verdict(s["delta"], s["cluster_lo"], s["cluster_hi"], s["cluster_p"], mei)})

    secondary = []
    for T in horizons:
        cells_T = {c: d for c, d in diffs.items() if horizon_of[c] == T}
        if not cells_T:
            continue
        s = _stratum(cells_T, env_of, n_boot=n_boot, seed=ad._seed("registered", policy, reference, test, T))
        secondary.append({**base, "row": "secondary", "horizon": T, **s, "verdict": ""})
    ps = [r["cluster_p"] if np.isfinite(r["cluster_p"]) else 1.0 for r in secondary]
    for r, adj in zip(secondary, _holm(ps), strict=True):
        r["p_holm"] = adj
    rows.extend(secondary)

    for T in ALL_HORIZONS:
        if T in horizons:
            continue
        cells_T = {c: d for c, d in diffs.items() if horizon_of[c] == T}
        if not cells_T:
            continue
        s = _stratum(cells_T, env_of, n_boot=n_boot, seed=ad._seed("registered", policy, reference, test, T))
        rows.append({**base, "row": "structural", "horizon": T, **s, "p_holm": np.nan, "verdict": ""})
    return pd.DataFrame(rows)[list(COLUMNS)]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=REGISTERED["policy"])
    ap.add_argument("--reference", default=REGISTERED["reference"])
    ap.add_argument("--test", default=REGISTERED["test"])
    ap.add_argument("--horizons", default=",".join(str(h) for h in REGISTERED["horizons"]))
    ap.add_argument("--mei", type=float, default=REGISTERED["mei"])
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=None, help="default: <out-dir>/tables/h1b_prime.csv")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    out = registered_contrast(args.policy, args.reference, test=args.test, horizons=horizons,
                              mei=args.mei, out_dir=args.out_dir, n_boot=args.n_boot)
    path = args.out or (args.out_dir / "tables" / "h1b_prime.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    p = out[out["row"] == "primary"].iloc[0]
    log.info("H1b' %s vs %s on %s at T in %s: delta=%+.6f paired [%+.6f, %+.6f] cluster t [%+.6f, %+.6f] "
             "p=%.3g sign p=%s n_envs=%d n_cells=%d  ->  %s",
             args.policy, args.reference, args.test, horizons, p["delta"], p["lo"], p["hi"],
             p["cluster_lo"], p["cluster_hi"], p["cluster_p"], p["cluster_p_sign"], p["n_envs"], p["n_cells"],
             p["verdict"].upper())
    for _, r in out[out["row"] != "primary"].iterrows():
        log.info("  %-10s T=%-5s delta=%+.6f cluster [%+.6f, %+.6f] p=%.3g holm=%s",
                 r["row"], r["horizon"], r["delta"], r["cluster_lo"], r["cluster_hi"], r["cluster_p"],
                 "-" if not np.isfinite(r["p_holm"]) else f"{r['p_holm']:.3g}")
    log.info("wrote %s", path)
    return out


if __name__ == "__main__":
    main()
