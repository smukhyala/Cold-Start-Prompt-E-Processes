#!/usr/bin/env python
"""M3 pilot: ~100 labelled states end to end, plus the section-21 sanity cases.

This is the stop gate. Its job is not to produce results but to make the oracle
labels *inspectable* before any real compute is spent: are the sanity cases recovered,
how precise are the labels, how much does the answer depend on the recommendation
rule, and does the Monte Carlo agree with an exact dynamic program on toy problems.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_states import GenSpec, harvest  # noqa: E402

from cold_start.growing.allocation import LUCB, UCBChallenger, WidthRacing  # noqa: E402
from cold_start.growing.evidence import PairwiseEvidence  # noqa: E402
from cold_start.growing.features import extract_features  # noqa: E402
from cold_start.growing.labeling import Snapshot, label_state, label_state_multi  # noqa: E402
from cold_start.growing.recommend import oracle_prior_from_reservoir  # noqa: E402
from cold_start.growing.reservoirs import build_reservoir  # noqa: E402
from cold_start.growing.search_policies import (  # noqa: E402
    BernoulliSearch,
    EvidenceGatedSchedule,
    PowerSchedule,
    UniformRandom,
)
from cold_start.growing.simulator import Simulator  # noqa: E402
from cold_start.growing.tables import CSTable  # noqa: E402

ENVIRONMENTS = [
    ("beta_good_common", {"type": "beta", "params": {"a": 5.0, "b": 2.0}}),
    ("beta_rare_excellent", {"type": "beta", "params": {"a": 1.0, "b": 9.0}}),
    ("tail_beta2", {"type": "tail", "params": {"beta": 2.0, "mu_star": 1.0, "c": 1.0}}),
    ("tail_beta8", {"type": "tail", "params": {"beta": 8.0, "mu_star": 1.0, "c": 1.0}}),
]


def continuation(table, reservoir, horizon: int, cap: int):
    def factory(offset: int) -> Simulator:
        return Simulator(
            table=table,
            reservoir=reservoir,
            allocation=LUCB(),
            search_policy=EvidenceGatedSchedule(alpha=0.5, c=1.0, min_pulls_per_arm=2),
            horizon=horizon,
            max_live_arms=cap,
        )

    return factory


def run_sanity_cases(table, horizon: int, cap: int) -> None:
    """Section 21, as CONTRASTS between paired states rather than sign tests.

    Two lessons from the pilot shape this. First, a single forced action only matters
    when the remaining budget is small, so each case is posed near the end of the run.
    Second -- and more importantly -- the interesting claim is comparative. "Searching
    should be attractive when the reservoir is rich" is a statement about rich versus
    poor, and its control is a state where A_t is legitimately *zero*: with a poor
    reservoir there is nothing to gain by searching and often nothing to gain by
    refining either. Asserting a sign on a quantity that is genuinely zero tests
    nothing, so each case pairs a treatment against a control and asserts the gap.
    """
    rich = build_reservoir({"type": "tail", "params": {"beta": 1.0, "mu_star": 1.0, "c": 1.0}})
    poor = build_reservoir({"type": "beta", "params": {"a": 1.0, "b": 9.0}})

    print("\n" + "=" * 78)
    print("SECTION 21 SANITY CASES  (paired contrasts, short remaining budget)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    print(f"  rich reservoir mean ~ {rich.sample(rng, 200_000).mean():.3f}   "
          f"poor reservoir mean ~ {poor.sample(rng, 200_000).mean():.3f}")

    def at(rem, n, s, mu, seed):
        return Snapshot(np.array(n), np.array(s), np.array(mu),
                        horizon - rem, horizon, len(n), seed)

    def label(snap, res, k=1):
        return label_state(snap, continuation(table, res, horizon, cap), table,
                           target_se=2e-4, max_replicates=8192,
                           oracle_prior=oracle_prior_from_reservoir(res), commit_steps=k)

    weak = ([60, 40, 40], [21, 6, 5], [0.35, 0.15, 0.13])
    pair = ([5, 5], [4, 2], [0.85, 0.55])
    strong = ([90, 90], [72, 71], [0.80, 0.79])

    contrasts = [
        ("A  reservoir quality drives search value",
         "rich tail", label(at(6, *weak, 101), rich),
         "poor tail", label(at(6, *weak, 101), poor), 0.05),
        ("C  incumbent quality drives search value",
         "incumbent 0.30", label(at(1, [90, 90], [27, 26], [0.30, 0.29], 103), rich),
         "incumbent 0.80", label(at(1, *strong, 104), rich), 0.05),
        ("B  holding an unbeatable arm makes refining preferred (k=16)",
         "rich tail", label(at(25, *pair, 102), rich, k=16),
         "poor tail", label(at(25, *pair, 102), poor, k=16), None),
    ]

    print(f"\n{'contrast':<52}{'treat':>10}{'control':>10}{'gap':>10}  ok")
    n_ok = 0
    for name, t_lab, t_r, c_lab, c_r, margin in contrasts:
        gap = t_r.advantage - c_r.advantage
        if margin is None:
            # A directional claim, not a contrast: the incumbent at 0.85 beats
            # anything either reservoir produces, so committing to SEARCH must lose
            # in BOTH arms. Asserting a gap here would test prior shrinkage, not the
            # claim the case is about.
            ok = t_r.advantage < 0 and c_r.advantage < 0
        else:
            ok = gap > margin
        n_ok += ok
        print(f"{name:<52}{t_r.advantage:>+10.4f}{c_r.advantage:>+10.4f}{gap:>+10.4f}"
              f"  {'yes' if ok else 'NO'}")
        print(f"{'    ' + t_lab + ' vs ' + c_lab:<52}"
              f"{'SE ' + format(t_r.se, '.4f'):>10}{'SE ' + format(c_r.se, '.4f'):>10}"
              f"{('both<0' if margin is None else 'need >' + format(margin, '.3f')):>10}")
    print(f"\n  {n_ok}/{len(contrasts)} contrasts recovered")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--trajectories", type=int, default=6)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--max-live-arms", type=int, default=24)
    ap.add_argument("--target-se", type=float, default=0.0003)
    ap.add_argument("--max-replicates", type=int, default=4096)
    ap.add_argument("--commit-steps", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    t_start = time.time()
    table = CSTable.load_or_build(args.horizon, alpha=0.05)
    pairwise = PairwiseEvidence()

    policies = [
        PowerSchedule(alpha=0.5, c=1.0),
        BernoulliSearch(p=0.35, rng=np.random.default_rng(args.seed)),
        UniformRandom(rng=np.random.default_rng(args.seed + 1)),
    ]
    allocations = [LUCB(), UCBChallenger(), WidthRacing()]

    print("=" * 78)
    print("GROWING BANDITS -- M3 PILOT")
    print("=" * 78)
    print(f"horizon={args.horizon}  target_se={args.target_se}  max_M={args.max_replicates}")

    snaps: list[Snapshot] = []
    for i, (env_id, spec) in enumerate(ENVIRONMENTS):
        reservoir = build_reservoir(spec)
        for j, policy in enumerate(policies):
            alloc = allocations[(i + j) % len(allocations)]
            gspec = GenSpec(
                horizon=args.horizon,
                n_trajectories=args.trajectories,
                snapshot_times=tuple(range(args.snapshots)),
                max_live_arms=args.max_live_arms,
                seed=args.seed,
            )
            snaps.extend(
                harvest(reservoir, table, policy, alloc, gspec, env_id, seed_offset=100 * i + j)
            )
    print(f"\nharvested {len(snaps)} states from {len(ENVIRONMENTS)} environments "
          f"x {len(policies)} policies")

    rows = []
    n_undefined = 0
    t0 = time.time()
    for s in snaps:
        if s.k >= args.max_live_arms:
            n_undefined += 1
            continue
        reservoir = build_reservoir(dict(ENVIRONMENTS)[s.meta["env_id"]])
        feat = extract_features(
            n=s.n, successes=s.successes, mu_true=s.mu, t=s.t, horizon=s.horizon,
            table=table, pairwise=pairwise, reservoir=reservoir,
            history=s.meta.get("history"),
        )
        labs = label_state_multi(
            s, continuation(table, reservoir, args.horizon, args.max_live_arms), table,
            commit_steps=tuple(args.commit_steps),
            target_se=args.target_se, max_replicates=args.max_replicates,
            oracle_prior=oracle_prior_from_reservoir(reservoir),
            max_live_arms=args.max_live_arms,
        )
        lab = labs[args.commit_steps[0]]  # k=1 is canonical
        for k, lk in labs.items():
            feat[f"label_A_k{k}"] = lk.advantage
            feat[f"label_se_k{k}"] = lk.se
            feat[f"label_M_k{k}"] = float(lk.n_replicates)
        feat.update({
            "label_A": lab.advantage, "label_se": lab.se,
            "label_M": float(lab.n_replicates), "label_batches": float(lab.n_batches),
            "label_mean_search": lab.mean_search, "label_mean_refine": lab.mean_refine,
            "label_se_unpaired": lab.se_unpaired, "label_frac_identical": lab.frac_identical,
        })
        feat["meta_env"] = s.meta["env_id"]
        feat["meta_policy"] = s.meta["policy"]
        feat["meta_allocation"] = s.meta["allocation"]
        rows.append(feat)
    label_secs = time.time() - t0

    A = np.array([r["label_A"] for r in rows])
    se = np.array([r["label_se"] for r in rows])
    M = np.array([r["label_M"] for r in rows])

    print(f"\n  {n_undefined} of {len(snaps)} states ({100*n_undefined/max(len(snaps),1):.1f}%) "
          f"were AT THE ARM CAP and have no SEARCH counterfactual -- excluded, not zeroed")
    print(f"labelled {len(rows)} states in {label_secs:.1f}s "
          f"({1000*label_secs/max(len(rows),1):.0f} ms/state)")
    print("\n" + "=" * 78)
    print("ORACLE LABEL DISTRIBUTION")
    print("=" * 78)
    print(f"  A_t   mean {A.mean():+.4f}   sd {A.std():.4f}   "
          f"min {A.min():+.4f}   max {A.max():+.4f}")
    for q in (5, 25, 50, 75, 95):
        print(f"        p{q:<3d} {np.percentile(A, q):+.4f}")
    print(f"  SEARCH preferred in {100*(A > 0).mean():.1f}% of states")
    print(f"  |A_t| > 2*SE (clearly decided) in {100*(np.abs(A) > 2*se).mean():.1f}% of states")
    se_un = np.array([r["label_se_unpaired"] for r in rows])
    fid = np.array([r["label_frac_identical"] for r in rows])
    print(f"  SE     mean {se.mean():.5f}   max {se.max():.5f}   "
          f"target {args.target_se}")
    ok = se > 1e-9
    gain = np.median(se_un[ok] / se[ok]) if ok.any() else float("nan")
    print(f"  CRN    paired SE is {gain:.0f}x smaller than unpaired (median over the "
          f"{ok.sum()} states with non-zero paired SE)")
    print(f"         both branches recommend the same arm in {100*fid.mean():.1f}% of replicates")
    print(f"         {100*(~ok).mean():.1f}% of states had EVERY replicate agree (A_t exactly 0)")
    print(f"  M      mean {M.mean():.0f}      max {M.max():.0f}   "
          f"hit cap in {100*(M >= args.max_replicates).mean():.1f}% of states")

    print("\n  by environment:")
    for env, _ in ENVIRONMENTS:
        idx = [i for i, r in enumerate(rows) if r["meta_env"] == env]
        if idx:
            a = A[idx]
            print(f"    {env:<24} n={len(idx):3d}  mean A_t {a.mean():+.4f}  "
                  f"SEARCH {100*(a > 0).mean():5.1f}%")

    print("\n  by remaining-budget quartile (the region the boundary should move in):")
    rem = np.array([r["f_remaining_frac"] for r in rows])
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        m = (rem >= lo) & (rem < hi)
        if m.any():
            print(f"    remaining {lo:.2f}-{hi:.2f}  n={m.sum():3d}  "
                  f"mean A_t {A[m].mean():+.4f}  SEARCH {100*(A[m] > 0).mean():5.1f}%")

    print("\n  |A_t| scale vs remaining budget (is the signal concentrated near the horizon?):")
    for lo_, hi_ in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        m = (rem >= lo_) & (rem < hi_)
        if m.any():
            print(f"    remaining {lo_:.2f}-{hi_:.2f}  n={m.sum():3d}  "
                  f"mean |A_t| {np.abs(A[m]).mean():.5f}  "
                  f"decided {100*(np.abs(A[m]) > 2*se[m]).mean():5.1f}%")

    print("\n  commitment horizon k -- how much can a SEARCH/REFINE decision be worth?")
    print(f"    {'k':>4}{'mean |A|':>12}{'decided':>10}{'SEARCH pref':>13}")
    for k in args.commit_steps:
        ak = np.array([r[f"label_A_k{k}"] for r in rows])
        sk = np.array([r[f"label_se_k{k}"] for r in rows])
        print(f"    {k:>4}{np.abs(ak).mean():>12.5f}"
              f"{100*(np.abs(ak) > 2*sk).mean():>9.1f}%{100*(ak > 0).mean():>12.1f}%")

    run_sanity_cases(table, args.horizon, args.max_live_arms)

    print("\n" + "=" * 78)
    print(f"pilot complete in {time.time()-t_start:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
