#!/usr/bin/env python
"""M6/section 17: run the learned decision rule as an actual algorithm.

Fitting Phi to oracle labels shows the labels are predictable. It does not show the
resulting policy is any good -- those are separate claims, and only this file tests the
second one. Phi is deployed from scratch against the schedule baselines it is supposed to
beat, and scored on final simple regret.

The learned policy is implemented as a genuinely vectorized `SearchPolicy`, computing its
own features from the (M, Kmax) state each round. That is deliberate: a policy that could
only be evaluated by calling the scalar feature extractor per replicate would not be a
policy anyone could deploy, and the cost would hide in the benchmark rather than in the
algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.growing.allocation import LUCB, leader_and_challenger, plausible_mask  # noqa: E402
from cold_start.growing.evidence import PairwiseEvidence  # noqa: E402
from cold_start.growing.recommend import (  # noqa: E402
    oracle_prior_from_reservoir,
    recommend_with_oracle_prior,
)
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.search_policies import (  # noqa: E402
    BernoulliSearch,
    BracketExpansion,
    DecisionContext,
    EvidenceGatedSchedule,
    OSEInspired,
    PowerSchedule,
    SearchPolicy,
    UniformRandom,
)
from cold_start.growing.simulator import Simulator, seed_initial_arms  # noqa: E402
from cold_start.growing.state import GrowingState  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

# The six theoretical variables, in the order the fitted coefficients expect.
PHI_VARS = ("z", "u", "k", "tau", "r", "h")


def _p_new_beats_incumbent(
    post: np.ndarray, incumbent: np.ndarray, n_arms: np.ndarray
) -> np.ndarray:
    """Vectorized twin of `features._fit_beta_moments` + the Beta survival function.

    Method-of-moments Beta fit to the arms discovered so far, then P(draw > incumbent).
    The prior strength is capped by the arm count for the same reason it is in
    `recommend.fit_empirical_bayes_prior`: a prior estimated from a handful of
    survivorship-biased arms must not carry more weight than the evidence behind it.
    """
    from scipy.stats import beta as _beta

    with np.errstate(invalid="ignore"):
        m = np.clip(np.nanmean(post, axis=1), 1e-6, 1 - 1e-6)
        v = np.nanvar(post, axis=1, ddof=1)
    v = np.where(np.isfinite(v), v, 0.0)
    usable = (n_arms >= 2) & (v > 1e-12) & (v < m * (1 - m))
    nu = np.where(usable, m * (1 - m) / np.maximum(v, 1e-12) - 1.0, 2.0)
    nu = np.clip(nu, 2.0, np.maximum(n_arms.astype(np.float64), 2.0))
    a, b = m * nu, (1.0 - m) * nu
    out = _beta.sf(np.clip(incumbent, 0.0, 1.0), a, b)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class LearnedPolicy(SearchPolicy):
    """SEARCH iff Phi(S_t) > 0, with Phi a linear or degree-2 form in the six variables."""

    name = "learned_phi"
    needs_evidence = False

    def __init__(
        self,
        coef: dict[str, float],
        intercept: float = 0.0,
        pairwise: PairwiseEvidence | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(rng)
        self.coef = dict(coef)
        self.intercept = float(intercept)
        self.pairwise = pairwise or PairwiseEvidence()

    def features(self, state: GrowingState, ctx: DecisionContext) -> dict[str, np.ndarray]:
        """Vectorized versions of the six variables, one value per replicate."""
        lcb, ucb = state.view(state.lcb), state.view(state.ucb)
        active = state.active_mask()
        width = np.where(active, ucb - lcb, 0.0)
        plaus = plausible_mask(state)
        n_plaus = np.maximum(plaus.sum(axis=1), 1)

        leader, challenger = leader_and_challenger(state)
        lin_l = state.row_off + leader
        lin_c = state.row_off + challenger
        z = self.pairwise.log_e(
            state.n[lin_l], state.S[lin_l], state.n[lin_c], state.S[lin_c]
        )

        post = np.where(active, state.view(state.empirical_mean()), np.nan)
        incumbent = np.nanmax(post, axis=1)
        # Estimated chance a fresh arm beats the incumbent. This MUST be the same
        # quantity `features.extract_features` emits as `est_p_new_beats_incumbent`,
        # or the policy being benchmarked is not the policy that was fitted. An
        # earlier version counted arms strictly above their own maximum, which is
        # empty by construction -- so `r` was identically zero at deployment while
        # training had seen a real survival probability, and its coefficient was
        # being applied to a constant.
        r = _p_new_beats_incumbent(post, incumbent, active.sum(axis=1))

        return {
            "z": np.asarray(z, dtype=np.float64),
            "u": (width * plaus).sum(axis=1) / n_plaus,
            "k": np.log(np.maximum(state.Kt, 1)).astype(np.float64),
            "tau": np.full(state.M, np.log(max(ctx.t, 1)), dtype=np.float64),
            "r": r.astype(np.float64),
            "h": np.full(state.M, ctx.remaining / max(ctx.horizon, 1), dtype=np.float64),
        }

    def should_search(self, state: GrowingState, ctx: DecisionContext) -> np.ndarray:
        f = self.features(state, ctx)
        phi = np.full(state.M, self.intercept, dtype=np.float64)
        for name, c in self.coef.items():
            if name in f:
                phi += c * f[name]
                continue
            if ":" in name:  # interaction term a:b
                a, b = name.split(":", 1)
                if a in f and b in f:
                    phi += c * f[a] * f[b]
                continue
            if name.endswith("^2") and name[:-2] in f:  # quadratic term
                phi += c * f[name[:-2]] ** 2
        decision = phi > 0.0
        decision |= state.Kt == 0
        return decision


def _stable_seed(name: str) -> int:
    """Deterministic per-policy seed offset.

    `hash(str)` is salted per interpreter process, so seeding with it makes the whole
    benchmark unreproducible AND gives each policy a different reward stream on every
    run -- policies were being compared on different draws.
    """
    import zlib

    return int(zlib.crc32(name.encode())) % 10_000


def baselines(rng: np.random.Generator) -> dict:
    return {
        "uniform_random": lambda: UniformRandom(rng=rng),
        "fixed_K4": lambda: PowerSchedule(alpha=0.0, c=4.0, rng=rng),
        "fixed_K16": lambda: PowerSchedule(alpha=0.0, c=16.0, rng=rng),
        "sqrt_t": lambda: PowerSchedule(alpha=0.5, c=1.0, rng=rng),
        "t^1/3": lambda: PowerSchedule(alpha=1 / 3, c=1.5, rng=rng),
        "t^2/3": lambda: PowerSchedule(alpha=2 / 3, c=0.8, rng=rng),
        "pure_search": lambda: BernoulliSearch(p=1.0, rng=rng),
        "bracket": lambda: BracketExpansion(base_width=4, rng=rng),
        "ose": lambda: OSEInspired(target0=0.9, decay=0.5, rng=rng),
        "cp0_evidence_gated": lambda: EvidenceGatedSchedule(alpha=0.5, c=1.0, rng=rng),
    }


def run_policy(policy, reservoir, table, horizon: int, n_seeds: int, cap: int, seed: int) -> dict:
    """Run one policy from scratch and score it on final simple regret."""
    state = GrowingState(n_seeds, 8, horizon, base_seed=seed)
    sim = Simulator(
        table=table,
        reservoir=reservoir,
        allocation=LUCB(),
        search_policy=policy,
        horizon=horizon,
        max_live_arms=cap,
    )
    seed_initial_arms(state, reservoir, 2, table)
    sim.run_to_horizon(state, start_t=2)

    prior = oracle_prior_from_reservoir(reservoir)
    mu_rec = recommend_with_oracle_prior(state, prior).mu
    mu_star = float(reservoir.essential_sup())
    regret = mu_star - mu_rec
    return {
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "se_regret": float(regret.std(ddof=1) / np.sqrt(len(regret))),
        "p10": float(np.percentile(regret, 10)),
        "p90": float(np.percentile(regret, 90)),
        "mean_mu_recommended": float(mu_rec.mean()),
        "mean_K_final": float(state.Kt.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phi", type=str, default="", help="JSON with {coef:{...}, intercept:float}")
    ap.add_argument("--horizons", type=int, nargs="+", default=[100, 200, 500])
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--max-live-arms", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=str, default="results/growing_bandits")
    ap.add_argument(
        "--environments", type=str, nargs="+",
        default=["beta_good_common", "beta_rare_excellent", "tail_b2", "mix_broad_low_narrow_high"],
    )
    args = ap.parse_args()

    specs = {
        "beta_good_common": {"type": "beta", "params": {"a": 5.0, "b": 2.0}},
        "beta_rare_excellent": {"type": "beta", "params": {"a": 1.0, "b": 9.0}},
        "tail_b2": {"type": "tail", "params": {"beta": 2.0, "mu_star": 1.0, "c": 1.0}},
        "tail_b8": {"type": "tail", "params": {"beta": 8.0, "mu_star": 1.0, "c": 1.0}},
        "mix_broad_low_narrow_high": {
            "type": "mixture", "params": {"preset": "broad_low_narrow_high"},
        },
    }

    rng = np.random.default_rng(args.seed)
    policies = baselines(rng)
    if args.phi:
        spec = json.loads(Path(args.phi).read_text())
        policies["LEARNED_phi"] = lambda: LearnedPolicy(
            spec.get("coef", {}), spec.get("intercept", 0.0), rng=rng
        )
        print(f"loaded Phi from {args.phi}: {len(spec.get('coef', {}))} coefficients")
    else:
        print("no --phi supplied: benchmarking baselines only")

    out_dir = ROOT / args.out / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for horizon in args.horizons:
        table = CSTable.load_or_build(horizon, alpha=args.alpha)
        for env_id in args.environments:
            reservoir = build_reservoir(specs[env_id])
            print(f"\n=== T={horizon}  {env_id}  (mu* = {reservoir.essential_sup():.3f}, "
                  f"{args.seeds} seeds) ===")
            print(f"  {'policy':<22}{'mean regret':>13}{'SE':>9}{'median':>9}"
                  f"{'p90':>9}{'K_final':>9}")
            scored = []
            for name, make in policies.items():
                res = run_policy(
                    make(), reservoir, table, horizon, args.seeds, args.max_live_arms,
                    seed=args.seed + _stable_seed(name),
                )
                res.update({"policy": name, "horizon": horizon, "environment": env_id})
                rows.append(res)
                scored.append((res["mean_regret"], name, res))
            for _, name, res in sorted(scored):
                print(f"  {name:<22}{res['mean_regret']:>13.5f}{res['se_regret']:>9.5f}"
                      f"{res['median_regret']:>9.5f}{res['p90']:>9.5f}{res['mean_K_final']:>9.1f}")

    csv_path = out_dir / "policy_benchmark.csv"
    header = ["environment", "horizon", "policy", "mean_regret", "se_regret", "median_regret",
              "p10", "p90", "mean_mu_recommended", "mean_K_final"]
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in header) + "\n")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
