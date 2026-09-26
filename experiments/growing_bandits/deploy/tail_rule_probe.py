"""Is the reservoir's tail the generalizing signal? An oracle probe (diagnostic, not a claim).

The infinite-armed theory says the optimal number of arms grows like ``T^(beta/(beta+1))``,
``beta`` the upper-tail exponent of the reservoir (``P(mu >= mu* - eps) ~ c eps^beta``).
The study's reservoirs are parameterized by that exponent, so a rule can be handed the
*true* beta and asked what it buys: if a schedule with the true tail barely beats the
level rule on the held-out mixtures, the generalization gap is not about the signal.

Per environment (all 33) x T in {200, 1000}, uncapped, test split, M = 1000:

* ``ceiling``      -- fixed K at the environment's own K*(T) (tune-split envelope): the
                      per-env achievable reference, never a deployable rule;
* ``pooled_K``     -- fixed K at the pooled K*(T);
* ``level_star``   -- the level rule with its cap-T constants (`by_cap`);
* ``p3_star``      -- the power schedule with its cap-T constants;
* ``oracle_beta``  -- fixed K = c * T^(beta/(beta+1)), true beta, one c for all envs
                      chosen on the 30 corpus environments (the mixtures never vote).

Writes ``tables/tail_rule_probe.csv`` and prints the corpus / mixture means and the gap
to the ceiling for each rule.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT / "src", HERE.parent, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cells  # noqa: E402
import k_star_envelope as ks  # noqa: E402
import policy_table as pt  # noqa: E402
import run_deployment as rd  # noqa: E402

from cold_start.growing.deploy.harness import CellSpec, run_cell  # noqa: E402
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER  # noqa: E402
from cold_start.growing.deploy.rules import make_policy  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402

log = logging.getLogger("deploy.tail_probe")

HORIZONS: tuple[int, ...] = (200, 1000)
C_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
#: The upper-tail exponent of each mixture preset is its top component's Beta ``b``.
MIXTURE_BETA = {"many_mediocre_rare_excellent": 3.0, "bulk_half_tiny_cluster_high": 10.0, "broad_low_narrow_high": 8.0}


def true_beta(spec: dict) -> float:
    if spec["type"] == "beta":
        return float(spec["params"]["b"])
    if spec["type"] == "tail":
        p = spec["params"]
        return float(p["beta"] if "beta" in p else p["b"])
    return MIXTURE_BETA[spec["params"]["preset"]]


@dataclass(frozen=True)
class Item:
    spec: CellSpec
    rule: str
    kind: str
    params: tuple[tuple[str, float], ...]


def run_item(item: Item) -> dict:
    spec = item.spec
    table = ks._table(spec.horizon, spec.alpha)
    policy = make_policy(item.kind, params=dict(item.params), horizon=spec.horizon,
                         n_replicates=spec.n_replicates, table=table)
    res = run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
    regret = res.regret(PRIMARY_RECOMMENDER)
    return {"env_id": spec.env_id, "family": cells.ALL_ENVS[spec.env_id]["type"], "horizon": int(spec.horizon),
            "rule": item.rule, **dict(item.params), "regret": float(regret.mean()),
            "regret_se": float(regret.std(ddof=1) / np.sqrt(len(regret))), "k_final": float(res.k_final.mean())}


def build_items(envelope: pd.DataFrame, baseline_params: dict, n_replicates: int) -> list[Item]:
    env_k = envelope[(envelope["level"] == "env") & envelope["is_argmin"]].set_index(["env_id", "horizon"])["K"]
    pooled_k = envelope[(envelope["level"] == "pooled") & envelope["is_argmin"]].set_index("horizon")["K"]
    items: list[Item] = []
    for env_id, spec_env in cells.ALL_ENVS.items():
        beta = true_beta(spec_env)
        for T in HORIZONS:
            spec = cells.make_matched_cell(rd.TEST_SPLIT, env_id, T, T, n_replicates, seed_cap=64)
            bp = pt.baseline_params_for_cap(baseline_params, T)[0]
            level = bp["level_star"]
            p3 = bp["p3_star"][str(T)]
            items.append(Item(spec, "ceiling", "fixed_K16", (("K", float(env_k[(env_id, T)])),)))
            items.append(Item(spec, "pooled_K", "fixed_K16", (("K", float(pooled_k[T])),)))
            items.append(Item(spec, "level_star", "level_K",
                              (("alpha", float(level["alpha"])), ("c", float(level["c"])), ("b", float(level["b"])))))
            items.append(Item(spec, "p3_star", "power_sqrt", (("alpha", float(p3["alpha"])), ("c", float(p3["c"])))))
            for c in C_GRID:
                K = max(3.0, min(float(T), round(c * T ** (beta / (beta + 1)))))
                items.append(Item(spec, f"oracle_beta_c{c}", "fixed_K16", (("K", K),)))
    return items


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=rd.DEFAULT_OUT_DIR)
    ap.add_argument("--envelope", type=Path, default=None,
                    help="k_star_envelope CSV covering every environment (default: tables/k_star_envelope_all33.csv)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    envelope = pd.read_csv(args.envelope or args.out_dir / "tables" / "k_star_envelope_all33.csv")
    baseline_params = pt.load_baseline_params(args.out_dir / "baseline_params.json")
    items = build_items(envelope, baseline_params, args.replicates)
    log.info("%d items", len(items))
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=ks._worker_init, initargs=(logging.INFO,)) as pool:
        rows = list(pool.imap_unordered(run_item, items, chunksize=2))
    frame = pd.DataFrame(rows)
    frame["beta"] = [true_beta(cells.ALL_ENVS[e]) for e in frame["env_id"]]
    frame["is_mixture"] = frame["family"] == "mixture"
    out = args.out_dir / "tables" / "tail_rule_probe.csv"
    frame.to_csv(out, index=False)

    ceiling = frame[frame["rule"] == "ceiling"].set_index(["env_id", "horizon"])["regret"]
    frame["gap_to_ceiling"] = [r - ceiling[(e, T)] for e, T, r in zip(frame["env_id"], frame["horizon"], frame["regret"], strict=True)]
    corpus = frame[~frame["is_mixture"]]
    # Select oracle_beta's c on the corpus only.
    by_c = corpus[corpus["rule"].str.startswith("oracle_beta")].groupby("rule")["regret"].mean()
    best = by_c.idxmin()
    log.info("oracle_beta c selected on the 30 corpus environments: %s (corpus mean regret %.4f)", best, by_c[best])
    keep = ["ceiling", "pooled_K", "level_star", "p3_star", best]
    summary = (frame[frame["rule"].isin(keep)]
               .groupby(["horizon", "is_mixture", "rule"])[["regret", "gap_to_ceiling", "k_final"]].mean()
               .reset_index())
    for T, sub in summary.groupby("horizon"):
        log.info("T=%d  mean regret / gap to the per-env ceiling / mean K:\n%s", T,
                 sub.pivot(index="rule", columns="is_mixture", values=["regret", "gap_to_ceiling", "k_final"])
                 .round(4).to_string())
    log.info("wrote %s", out)
    return frame


if __name__ == "__main__":
    main()
