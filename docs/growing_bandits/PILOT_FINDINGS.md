# CORRECTION (2026-09-11)

Sections 1 and 2 below were measured with two defects present and are **superseded** by
this section. Both defects were mine, and both flattered the pessimistic reading.

1. **Forced SEARCH was silently converted to REFINE at the live-arm cap.** At the cap a
   new arm has nowhere to go, and the constraint was ANDed over the forced action. Both
   oracle branches then did the same thing, so the labeller recorded `A_t = 0.000000`
   with `SE = 0.000000` -- a fabricated label that looks maximally precise and stops the
   adaptive rule on its first batch. At the pilot's cap, **48.7% of harvested states**
   were affected; with an aggressive search policy it reached 100%.
2. **The snapshot schedule left the middle of the budget range nearly empty** (2.5% of
   states in the 0.25-0.50 band), so the reported "signal only near the horizon" was
   partly a statement about where states had been sampled from.

Corrected measurements, at the production arm cap of 64, both fixed:

| quantity | as first reported | corrected |
|---|---|---|
| states at the cap | (labelled 0) | 16.1%, **excluded** as undefined |
| `A_t` exactly zero | 74.7% | 49.7% (at cap 24); far lower at cap 64 |
| SEARCH preferred | 14.9% | **44.1%** |
| decided at k=1 | 7.0% | **25.2%** |
| decided at k=4 | 20.5% | **60.6%** |
| decided at k=16 | 59.3% | **90.4%** |

Decided fraction by remaining budget, which was the basis for the "signal lives only near
the horizon" claim: **35.3% / 26.7% / 25.3% / 36.5% / 11.5%** across bands
0-0.10 / 0.10-0.25 / 0.25-0.50 / 0.50-0.75 / 0.75-1.0. Signal is present throughout, and
is weakest at *large* remaining budget rather than absent everywhere but the end.

**What survives.** The commitment-horizon result is unchanged in direction and stronger in
magnitude: a single action carries less information than a sustained one, 25% -> 61% -> 90%
decided for k = 1 -> 4 -> 16. Recording all three horizons remains the right call. The
schedule-versus-level result (section 3) is unaffected -- it was measured through the
policy benchmark, which never used the forced-action path.

**What this means for the study.** The corpus will be considerably more informative than
the first pilot suggested: a near-balanced label (44% SEARCH), a quarter of states decided
at the canonical horizon, and usable signal across the whole budget range.

---

# Growing bandits — findings from the M3 pilot and the policy benchmark

Everything here is measured on the synthetic simulator in `src/cold_start/growing/`,
with the oracle-prior recommender and the budget-aware `cp0` continuation policy.

## 1. The single-action advantage is ~0 in about three quarters of states

`A_t` as the brief defines it — force one action, then follow a fixed continuation
policy — is indistinguishable from zero wherever budget is plentiful. In the pilot,
**74.7% of states had every one of 256+ paired replicates recommend the identical arm**,
so `A_t` was exactly 0. Signal appears only in roughly the last 15% of the budget:

| remaining budget | 150 | 100 | 60 | 30 | 10 | 5 |
|---|---|---|---|---|---|---|
| `A_t` (k=1) | +0.00013 | +0.00030 | −0.00008 | +0.00251 | −0.00282 | **+0.182** |
| decided? | no | no | no | yes | yes | yes |

A good continuation policy simply absorbs one forced action. This is a property of the
formulation, not a defect — but it means a corpus labelled at k=1 is mostly ties.

## 2. Holding the action for k steps recovers the signal

| k | mean \|A\| | decided | SEARCH preferred |
|---|---|---|---|
| 1 | 0.00035 | 7.6% | 14.6% |
| 4 | 0.00131 | 20.5% | 30.9% |
| 16 | 0.01189 | **59.3%** | 52.5% |

`k=1` remains the canonical label; `k∈{1,4,16}` are all recorded.

## 3. The growth *schedule* barely matters; the arm *level* does

36 cells (4 reservoirs × 3 horizons × 3 caps), 1500 seeds each, fresh objects per cell:

| policy | cells won |
|---|---|
| fill_to_cap | **27** |
| sqrt(t) | 3 |
| t^(2/3) | 3 |
| t^(1/3) | 3 |

The nine cells not won by `fill_to_cap` are ties in the fourth decimal, all at cap=8
where every policy reaches the same arm count. Regret is essentially a function of the
K actually reached, not the path taken to it — "recruit quickly to a good K, then
refine" dominates gradual growth everywhere tested.

Regret is monotone and well behaved: it falls with the horizon at fixed cap
(0.2134 → 0.1816 → 0.1640 → 0.1591 for tail β=2, cap=32, T=100→1000) and falls with the
cap at fixed horizon. Budget is exactly conserved in every run.

## 4. The recommendation rule does NOT drive the benchmark

A plausible worry was that the oracle-prior recommender makes identification free and so
biases everything toward SEARCH. Measured: for a fixed policy, regret under the oracle
prior, LCB, and Beta(1,1) posterior mean agree to three decimals (0.1528 / 0.1528 /
0.1527). The recommender matters a great deal for the *label* of an individual state, but
not for which *policy* wins.

## Implication

Taken together: the per-step SEARCH/REFINE decision carries little marginal value when
budget is plentiful, and the path to a given arm count does not matter — only the count.
`Phi`'s job may therefore reduce to choosing an appropriate K for the environment and
horizon, which is a substantially lower-dimensional problem than the brief assumes. That
is a result about the research question, and it was the point of generating the data
before proposing a policy.

## 5. Elimination works, which is the direct answer to the sqrt(K/t) concern

The brief rules out an estimation model of the form `sqrt(K/t)` because it assumes every
discovered arm needs comparable precision, which good pure-exploration rules make false.
Measured here at T=1000 with a cap of 32 arms and a Beta(5,2) reservoir:

| | eliminated arms | surviving arms |
|---|---|---|
| count | 4.5 of 32 | 27.5 |
| budget consumed | **31 pulls (3.1%)** | 969 pulls (96.9%) |
| mean pulls each | **7.0** | 35.1 |

Weak arms cost O(1) pulls in total rather than O(n) each — a 5x concentration ratio, and
direct quantitative support for the design choice.

One honest qualification: elimination is weak at short horizons (0.05-0.2 arms eliminated
at T=100-200, rising to 4.5 at T=1000). That is a consequence of using a genuine
anytime-valid confidence sequence rather than a heuristic bound — it is conservative by
construction, so eliminating an arm requires real evidence. The frontier features
(`f_n_eliminated`, `f_frac_plausible`, `f_is_separated`) are therefore near-constant at
T=100 and informative only at the longer horizons.

## Bugs found and fixed during the pilot

1. **Unpaired standard error** in the labeller, discarding the CRN coupling — paired
   variance is 0.3–3% of unpaired at long horizons.
2. **Continuation policy not budget-aware** — a `K_t < c·sqrt(t)` schedule tried to catch
   up near the horizon, recruiting arms it could never evaluate. 100% of recommendations
   were arms with ≤2 pulls; now 0.5%.
3. **Recommendation confound** — under Beta(1,1) a 1-pull arm outranked a 120/200
   incumbent. An empirical-Bayes fix was itself survivorship-biased (Beta(424,92) from two
   arms). Settled on the posterior mean under a prior matched to the true reservoir.
4. **LUCB p-dependence** — tie-breaking on confidence-sequence width locked onto arms near
   0.5, spending 18.4 pulls on an inferior arm against 10.2 on the leader and costing 8
   points of recommendation accuracy. Fixed by balancing pull counts.

## Bugs found by the adversarial audit (before any production compute)

5. **The shard plan confounded environment with horizon.** A first version indexed
   environment, policy and allocation independently by shard number; because 36
   environments and 8 policies share a divisor, a quarter of the combinations could never
   occur. The fix — walk the full cross product — produced *perfectly even marginal counts*,
   which is exactly what made the real defect invisible: `combos` is enumerated
   environment-major, so 24 consecutive entries share an environment, and with fewer shards
   per horizon (312) than combinations (864) each horizon took a **contiguous** window and
   therefore only ~13 of 30 environments.

   Measured on the production defaults: **69 of 180 environment × horizon cells populated**,
   Beta reservoirs present at T=50 and T=1000 and *nowhere else*. Because Family B is
   enumerated with the tail exponent outermost, `beta` became nearly monotone in the
   horizon — T=50 saw only β ∈ {0.5, 1}, T=500 saw {2, 4, 8}. The aliased effect is
   **larger than the effect it aliases**: P(A_t > 0) swings 0.22 across β but only 0.155
   across the horizon, and in the same direction. A fitted budget coefficient would have
   silently absorbed the tail-exponent trend, and the hold-out-by-horizon test — one of the
   study's stated generalization checks — would have been meaningless.

   Fixed by walking the product with a stride coprime to its length. Now **150/150**
   environment × horizon cells, all 30 environments at every horizon.

   *Side finding, same defect class:* 6 of 30 Family B parameter combinations raise
   `DegenerateReservoirError`, and 288 of 1560 shards targeted them. Every one would have
   died at run time, and because they are unevenly spread across horizons their deaths
   would have skewed the surviving environment mixture on top of everything else. The grid
   now filters them at construction.

6. **A dead feature column.** `est_frac_arms_above_incumbent` computed
   `(post > post.max()).mean()` — identically 0.0 for every state, since nothing exceeds
   its own maximum. Per-row finiteness checks cannot catch this, and a constant column
   survives a fit *silently*: `StandardScaler` maps it to zero and ridge assigns it a
   harmless coefficient, so nothing ever complains. Replaced with `est_top_gap`, which
   measures how isolated the leader is among discovered arms.

   The structural fix matters more than the column: there is now a test that harvests 120
   diverse states — varying arm counts, pull counts, true means *and* decision histories —
   and asserts that no deployable column is constant. That is the only kind of check that
   catches a dead feature.

**On the value of the audit.** Four bugs were found by hand-probing and two by the audit.
Finding #5 alone would have cost the entire overnight run: every individual label would have
been computed correctly, and the corpus would still have been unusable for the study's
headline question.
