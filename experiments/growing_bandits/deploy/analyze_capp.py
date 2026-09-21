"""The CRN-paired cap sweep (``--test capp``, roadmap 3.3): its tables, and the gate that makes them paired.

Three tables, all pooled over the eight main environments with the environment-mean t
interval (n = 8, above `CLUSTER_MIN_ENVS`), so no cap row is "directional evidence, not
a test" any more:

* ``capp_policies.csv`` -- per (T, cap, policy): regret, K_final, search fraction, and
  whether every constant the policy deployed was selected at that cap (from the manifest's
  ``params_tuned`` / ``params_cap`` / ``tau_cap`` stamps).
* ``capp_contrasts.csv`` -- per (T, cap) and registered pair: the within-cap paired
  contrast (`stats.stratified_pooled` over the eight cells, `cluster_bounds` over
  environments), exactly as the main tables compute it.
* ``capp_crosscap.csv`` -- per (T, cap, policy): regret at `cap` minus regret at cap 64,
  **paired by episode** -- the measurement the original sweep could not make, because
  its caps drew different seeds (Ruling 26).

The gate: `assert_paired` refuses the tree unless ``mu_star`` is bit-identical across
caps within every (env, T, policy). That is what "paired" means here, and the analogue
of the CRN guard in `harness.py`.
"""

from __future__ import annotations

import argparse
import json
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

log = logging.getLogger("deploy.capp")

TEST = "capp"
REFERENCE_CAP = rd.DEFAULT_CAP
REGRET = f"regret_{PRIMARY_RECOMMENDER}"

#: The within-cap contrasts the document reads, in order.
PAIRS: tuple[tuple[str, str], ...] = (
    ("phi_k4", "level_star"),
    ("level_star", "p3_star"),
    ("level_star", "fixed_K_star"),
    ("phi_k4", "p3_star"),
    ("phi_k16", "p3_star"),
    ("phi_k16", "always_search"),
    ("phi_k4", "cp0"),
    ("p3_star", "always_search"),
)


def load_episodes(out_dir: Path) -> pd.DataFrame:
    root = Path(out_dir) / "episodes" / TEST
    if not root.exists():
        raise FileNotFoundError(f"no episodes for test {TEST!r} under {out_dir}")
    frames = []
    for cell_dir in sorted(root.iterdir()):
        for pq in sorted(cell_dir.glob("*.parquet")):
            f = pd.read_parquet(pq, columns=["cell", "policy", "env_id", "horizon", "cap", "base_seed",
                                             "episode", "mu_star", REGRET, "k_final", "search_frac"])
            frames.append(f)
    if not frames:
        raise FileNotFoundError(f"no parquet under {root}")
    return pd.concat(frames, ignore_index=True)


def assert_paired(frame: pd.DataFrame) -> None:
    """mu_star bit-identical across caps within (env, T, policy), or refuse."""
    for (env, T, policy), sub in frame.groupby(["env_id", "horizon", "policy"]):
        ref = None
        for cap, g in sub.groupby("cap"):
            arr = g.sort_values("episode")["mu_star"].to_numpy()
            if ref is None:
                ref = (cap, arr)
                continue
            if arr.shape != ref[1].shape or not np.array_equal(arr, ref[1]):
                raise ValueError(
                    f"{env} T={T} {policy}: mu_star differs between cap {ref[0]} and cap {cap} -- "
                    "the caps are not CRN-paired; refusing to analyse"
                )


def tuned_flags(out_dir: Path) -> dict[tuple[str, str], dict]:
    """``(cell, policy) -> params`` of the latest completion line, from the manifest."""
    path = Path(out_dir) / f"manifest_{TEST}.jsonl"
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind", "completion") != "completion":
            continue
        out[(rec["cell"], rec["policy"])] = rec.get("params") or {}
    return out


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


def analyze(out_dir: Path, *, n_boot: int = 10_000) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    frame = load_episodes(out_dir)
    assert_paired(frame)
    flags = tuned_flags(out_dir)

    policies_rows: list[dict] = []
    for (T, cap, policy), sub in frame.groupby(["horizon", "cap", "policy"]):
        per_cell = sub.groupby("cell").agg(regret=(REGRET, "mean"), k_final=("k_final", "mean"),
                                           search_frac=("search_frac", "mean"), env_id=("env_id", "first"))
        params = [flags.get((c, policy), {}) for c in per_cell.index]
        tuned = all(p.get(rd.pt.PARAMS_TUNED, True) for p in params) if params else True
        policies_rows.append({
            "test": TEST, "horizon": int(T), "cap": int(cap), "policy": policy,
            "regret": float(per_cell["regret"].mean()),
            "regret_se_envs": float(per_cell["regret"].std(ddof=1) / np.sqrt(len(per_cell))) if len(per_cell) > 1 else np.nan,
            "k_final": float(per_cell["k_final"].mean()), "search_frac": float(per_cell["search_frac"].mean()),
            "n_envs": int(per_cell["env_id"].nunique()), "n_episodes": int(len(sub)),
            "params_tuned": bool(tuned),
            "params_cap": sorted({p.get(rd.pt.PARAMS_CAP) for p in params if p.get(rd.pt.PARAMS_CAP) is not None}),
            "tau_cap": sorted({p.get(rd.pt.TAU_CAP) for p in params if p.get(rd.pt.TAU_CAP) is not None}),
        })
    policies = pd.DataFrame(policies_rows)

    contrast_rows: list[dict] = []
    for (T, cap), sub in frame.groupby(["horizon", "cap"]):
        by_pol = {p: g.sort_values(["cell", "episode"]) for p, g in sub.groupby("policy")}
        for policy, reference in PAIRS:
            if policy not in by_pol or reference not in by_pol:
                continue
            a, b = by_pol[policy], by_pol[reference]
            diffs: dict[str, np.ndarray] = {}
            env_of: dict[str, str] = {}
            for cell, ga in a.groupby("cell"):
                gb = b[b["cell"] == cell]
                if len(gb) != len(ga) or not np.array_equal(ga["episode"].to_numpy(), gb["episode"].to_numpy()):
                    raise ValueError(f"{cell}: {policy} and {reference} are not episode-aligned")
                diffs[cell] = ga[REGRET].to_numpy() - gb[REGRET].to_numpy()
                env_of[cell] = str(ga["env_id"].iloc[0])
            s = _stratum(diffs, env_of, n_boot=n_boot, seed=ad._seed(TEST, T, cap, policy, reference))
            contrast_rows.append({"test": TEST, "horizon": int(T), "cap": int(cap), "policy": policy,
                                  "reference": reference, **s})
    contrasts = pd.DataFrame(contrast_rows)

    cross_rows: list[dict] = []
    for (T, policy), sub in frame.groupby(["horizon", "policy"]):
        ref = sub[sub["cap"] == REFERENCE_CAP].sort_values(["env_id", "episode"])
        if ref.empty:
            continue
        for cap, g in sub.groupby("cap"):
            if int(cap) == REFERENCE_CAP:
                continue
            g = g.sort_values(["env_id", "episode"])
            diffs, env_of = {}, {}
            for env, ge in g.groupby("env_id"):
                gr = ref[ref["env_id"] == env]
                if len(gr) != len(ge) or not np.array_equal(ge["episode"].to_numpy(), gr["episode"].to_numpy()):
                    raise ValueError(f"{env} T={T} {policy}: cap {cap} and cap {REFERENCE_CAP} are not episode-aligned")
                diffs[str(env)] = ge[REGRET].to_numpy() - gr[REGRET].to_numpy()
                env_of[str(env)] = str(env)
            s = _stratum(diffs, env_of, n_boot=n_boot, seed=ad._seed(TEST, "crosscap", T, cap, policy))
            cross_rows.append({"test": TEST, "horizon": int(T), "cap": int(cap), "reference_cap": REFERENCE_CAP,
                               "policy": policy, **s})
    crosscap = pd.DataFrame(cross_rows)
    return {"policies": policies, "contrasts": contrasts, "crosscap": crosscap}


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = analyze(args.out_dir, n_boot=args.n_boot)
    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        frame.to_csv(tables / f"capp_{name}.csv", index=False)
        log.info("wrote capp_%s.csv (%d rows)", name, len(frame))
    pol = out["policies"]
    for T, sub in pol.groupby("horizon"):
        log.info("T=%d regret by cap (rows) x policy:\n%s", T,
                 sub.pivot(index="cap", columns="policy", values="regret").round(4).to_string())
        log.info("T=%d K_final by cap x policy:\n%s", T,
                 sub.pivot(index="cap", columns="policy", values="k_final").round(1).to_string())
        untuned = sub[~sub["params_tuned"]]
        if len(untuned):
            log.warning("T=%d: %d (cap, policy) rows deployed a constant not selected at that cap: %s", T,
                        len(untuned), sorted({(int(c), p) for c, p in zip(untuned["cap"], untuned["policy"], strict=True)}))
    con = out["contrasts"]
    for (T, policy, reference), sub in con.groupby(["horizon", "policy", "reference"]):
        log.info("T=%d %s - %s by cap: %s", T, policy, reference,
                 "  ".join(f"{int(r.cap)}:{r.delta:+.4f}(p={r.cluster_p:.2g})" for r in sub.sort_values("cap").itertuples()))
    return out


if __name__ == "__main__":
    main()
