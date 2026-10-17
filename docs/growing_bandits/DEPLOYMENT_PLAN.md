# Growing Bandits — End-to-End Deployment of the Learned SEARCH-vs-REFINE Policy

> Frozen copy of the approved plan (2026-09-14). Source of truth for the deployment study; every deviation
> during execution is recorded in the ledger rulings reproduced in DEPLOYMENT_RESULTS.md.
> Work on a new branch `deploy-policy` off current HEAD; commit at each milestone.

## Context

The offline study (`docs/growing_bandits/RESULTS.md`) established that oracle SEARCH-vs-REFINE labels are
modestly predictable (k=16 balanced acc 0.647 for the 71-feature model vs 0.596 for the `c·t^{1/3}` schedule),
that e-process evidence adds nothing over "how good is my current pool", and that k=1 and k=16 labels are
different control problems. RESULTS.md itself says the deployed claim is **NOT established**: "It has not been
run with a fitted Φ yet."

This experiment answers: *if we deploy a learned SEARCH-vs-REFINE policy from scratch, does it achieve lower
simple regret than principled growth schedules and other baselines?* — and, if offline skill fails to transfer,
*why* (state-distribution shift, label/continuation mismatch, commitment coarseness, cap effects, reservoir
tail vs. incumbent).

### Facts already established from direct reading (not from agents)

1. `experiments/growing_bandits/evaluate_policy.py` exists but is **not a fair harness yet**:
   - `run_policy(..., seed=args.seed + _stable_seed(name))` → every policy gets a *different* seed. No common
     random numbers; the docstring's "same draws" claim is wrong.
   - All baselines share one `np.random.default_rng(args.seed)` object (`baselines(rng)`), so any policy's
     internal randomness depends on which policies ran before it.
   - Recommender is `recommend_with_oracle_prior(state, oracle_prior_from_reservoir(reservoir))` — hidden
     reservoir parameters inside the deployed algorithm.
   - `mu_star = reservoir.essential_sup()` = 1.0 for every family → "regret" is just `1 − Q_T`.
   - `LearnedPolicy` only supports a linear/quadratic form over six variables `(z,u,k,tau,r,h)`; the 71-feature
     models in `fit_models.py` cannot be deployed with it.
   - Only mean/median/SE/p10/p90 per cell; no per-episode output, no paired CIs, no dynamics, no decomposition.
2. Old benchmark (`results/growing_bandits/tables/policy_benchmark.csv`, T=200, cap=64, 2000 seeds):
   `pure_search` (fill to cap 64, then refine) is best or tied-best in both environments; `fixed_K4` is worst.
   PILOT_FINDINGS §3: `fill_to_cap` won 27/36 cells. **The 64-arm cap is a hidden policy parameter** and
   "always SEARCH" is not pathological under it. The experiment must (a) report cap-hit fraction per policy,
   (b) run a cap-sensitivity arm (cap ∈ {32, 64, uncapped=T}) so P0 becomes genuinely pathological there.
3. Corpus: 5 horizons {50,100,200,500,1000}, 8 behavioural policies, 64-arm cap, k∈{1,4,16}, LUCB refine
   allocation, `cp0` budget-aware continuation, oracle-prior recommender; Family C (mixtures) **held out** of
   the corpus by default → natural Test C.
4. Compute: 14 cores, 38 GB RAM, Python 3.13, numpy/scipy/sklearn/pandas/pyarrow (no lightgbm/xgboost).
   Simulator is vectorized over M replicate episodes `(M, Kmax)`.

## Audit synthesis (Agents A–E)

### A — Simulator / environment (complete)
- Reservoir is **infinite**; every SEARCH maps a counter-indexed uniform through an inverse CDF
  (`reservoirs.py:114-130`). Families: A `beta(a,b)` (6 presets), B `tail(beta, mu_star, c)` (24 valid of 45;
  production uses 24 with c∈{1,2}), C `mixture` (3 presets; **absent from corpus**). Means ∈ [0,1].
  Production grid = 30 envs (6 Beta + 24 tail). `build_reservoir(spec)` from `{"type","params"}`.
- Rewards: Bernoulli(μ) only; integer-threshold compare (`rng.py:88-94`, `state.py:226-260`).
- `GrowingState` hidden fields: `mu`, `thresh`. Observable: `n, S, lcb, ucb, log_e, sum_sq_dev, lam, uid, Kt, n_draws`.
  Column contract (`schema.py:21-28,44-65`): `f_*`/`est_*` deployable, `oracle_*` analysis-only (12 cols in corpus).
- Horizon semantics: T = total pulls incl. 2 warm-start pulls; loop `t=2..T-1`; `remaining = T - t`.
  Corpus: T∈{50,100,200,500,1000}, rows 19,968/19,968/18,127/15,763/13,322 (attrition from cap-exclusion at long T).
  CS tables cached for many T (`data/cs_tables/`); new T built on demand (`tables.py:126-169`).
- Cap: `max_live_arms=64` counts all discovered incl. eliminated; no recycling. Always-SEARCH → K=64 at t=64, then
  silently REFINE (LUCB). At T=50 always-SEARCH never refines (all n=1; recommendation is a jitter-broken tie).
- **RNG / CRN (`rng.py`)**: stateless splitmix64 keys. Domain 0 rewards keyed `(seed, rep, uid, pull-counter)`;
  domain 1 reservoir draw keyed `(seed, rep, draw-index)`; domain 2 tie jitter. ⇒ the k-th searched arm's mean is a
  function of `(base_seed, replicate, k)` only, independent of *when* it is searched. Agent A verified bit-identical
  arm means/reward prefixes across two different policies with the same `base_seed`. **Sharing `base_seed` across
  policies gives exact CRN pairing with no cross-policy influence.** Policy-internal randomness uses a separate
  `np.random.Generator` per policy (must be freshly seeded per policy/cell).
- Elimination = CS plausibility `ucb_i >= max_j lcb_j` (per-arm truncated-Beta(1,1) mixture CS, α=0.05), recomputed
  each step, not sticky; inactive <T≈500. Eliminated arms hold cap slots and remain recommendable.
- **Compute blocker**: `PairwiseEvidence.log_e` (`evidence.py:189-204,615-636`) materializes `(M, 512, S_max+1)`;
  measured 1.8 GiB / 1.9 s per step at M=200, n=400 → ~17 GiB, ~19 s/step at M=2000, T=500. `PairwiseGridTable`
  tabulation refuses >512 MB. Any deployed policy using `z` (pairwise log-e) needs a tabulated/chunked path.
- Feature-parity hazards already present: `_p_new_beats_incumbent` fallback (`evaluate_policy.py:70-73`, ν=2) ≠
  `features._fit_beta_moments` fallback (`features.py:88-92`, Beta(1,1)); `z` uses jittered `leader_and_challenger`
  in deployment vs unjittered argmax in `features.py:150-159`.
- No `phi.json` exists and nothing produces one; `fit_models.py` pipelines are `StandardScaler→[Poly→Scaler]→Ridge/Logit`.
- No test exercises `evaluate_policy.py`.
- Comparator: `essential_sup()` = 1.0 for all Beta/mixture/μ*=1 tails, never attained (Beta(1,9) q99.9 = 0.536).
  Available alternatives: `reservoir.quantile(q)`, `oracle_best_true_mu`. **Chosen (see design): per-episode
  `μ*_ep = max of the first T reservoir draws in the episode's CRN stream`** — attainable, CRN-paired, makes
  discovery regret ≥ 0 by construction since `D_T ⊆ first K_T draws ⊆ first T draws`.

### B — Oracle-label pipeline (complete)
- Corpus: 87,148 rows × 116 cols; 1,560 shards = 30 envs × 8 policies × 3 allocations × 5 horizons, coprime-stride
  walk; each shard = 8 trajectories × 8 snapshots; shard seed `20260910 + 1_000_003·s + 31·T`. Single 445-min run.
- Generating policies (8): `aggressive` Bern(0.5), `conservative` Bern(0.05), `sqrt`, `cbrt`, `t23`, `random`,
  `epsilon`, `bracket`. Allocation rotated `{lucb, ucb, racing}`. Stored as `meta_policy`, `meta_allocation`.
- **Continuation** (both branches, all shards): `EvidenceGatedSchedule(cp0: α=0.5,c=1.0,min_pulls=2)` + `LUCB()` +
  oracle-prior recommender. cp0 is inert in ~64% of rows (K ≥ √t already) and 19% are unaffordable → continuation
  ≈ "LUCB-refine to T". cp0 is T-aware. Labels are therefore *deviations from a refine-heavy continuation*.
- Labels: force action for `min(k, T−t)` rounds (`strict` only on step 0; later at-cap forced SEARCH silently
  demoted to REFINE — 3.66% of k16 rows affected), then continuation; `A = mean(μ_rec|S) − mean(μ_rec|R)`, **+ = SEARCH**.
  Batches (256,256,512,1024,1024…) up to 4096, stop when paired SE < 3e-4. k16: 77% hit the 4096 cap, mean SE 1.07e-3.
- Ties: exact `A==0` (28.3% at k1, 11.6% k4, 5.1% k16), almost all decided by the first 256-batch (SE=0).
  Dropped from train and test in `fit_models.py:168`. 16% of non-tie rows have |A| < SE and still enter with their sign.
  `--min-precision` exists but is dead. Sign agreement among decided pairs: k1–k16 70.4% (RESULTS' 52.2% counts ties).
- Features: 58 `f_*` + 13 `est_*` = **71 deployable**; 12 `oracle_*`; 25 `label_*`; 8 `meta_*`. `f_T` = horizon (known).
  **`f_new_arms_last_{10,25,50}`, `f_best_mean_gain_last_*`, `f_search_frac_last_{10,25}`, `f_time_since_last_search`**
  are the behavioural policy's decision history → fingerprint in corpus, but on-policy would be Φ's own history.
- Trajectory id = `(meta_shard, meta_state_index % 8)` (12,477 trajectories). `GroupKFold` by `meta_env` (30 groups)
  is coarser than trajectory → **no row/trajectory leakage** in existing splits. 5,865 rows share identical feature
  vectors across envs with conflicting labels (early states) → irreducible noise floor.
- **Cap-exclusion selection effect**: 12,692 states (12.7%) dropped at K=64; `aggressive`/`random` retain ~12% of
  T=1000 states, all at `remaining_frac≈0.93`. Policy identity ⇔ early-t at long T. Partly manufactures the
  "policy fingerprint" and horizon-trend findings.
- Weighting (`fit_models.py:87-108`): `1/max(se, q25)^2`, capped at 20× median, mean-normalised. ESS not reported.
- No DP cross-check exists; DP assumes optimal continuation anyway (MC labels are relative to cp0).
- Corpus is reproducible from seeds (pure function of shard fields). Family C absent → `by_family` = A-vs-B only.

### C — Learned models (complete)
- Every reported number is `SIGN_logistic` = `make_pipeline(StandardScaler(), LogisticRegression(C=1.0))`
  (`fit_models.py:242-248`), fixed 0.5 threshold, `precision_weights` (floor q25, cap 20×median — cap never active).
  `SIGN_boosted` = `HistGradientBoostingClassifier(max_depth=6, max_iter=300, lr=0.06)`; only in the ladder.
- **Precision weighting hurts**: held-out AUC 0.737 (weighted) vs 0.750 (unweighted) for E@k16; ESS 43.6k/87k; top
  weight quartile has mean |A| = 0.0012. The doc's "pooled AUC 0.750" was an *unweighted* fit. **Drop weighting for
  the deployed models; keep it as a sensitivity variant.**
- Fixed 0.5 threshold is badly miscalibrated under class imbalance + weights (B predicts SEARCH 93% → bal 0.528
  despite AUC 0.70). Balanced-accuracy orderings in RESULTS.md are threshold artefacts; AUC orderings differ.
- Ablation selectors are substring hacks (`fit_models.py:50-64`): `D_time_plus_eprocess` = clock + **only `f_log_e_pair`**;
  `C_time_plus_counts` includes 6 history features + 2 CS-width means; `B_evidence_only` (16 cols incl. UCB/LCB, AUC 0.70)
  unreported. **Feature groups must be defined explicitly by column list for this experiment.**
- Single-feature AUCs in RESULTS.md are sign-flipped (`f_challenger_ucb` raw 0.294 → high UCB ⇒ REFINE).
- 116 columns = 57 `f_` + 14 `est_` (71 deployable) + 12 `oracle_` + 25 `label_` + 8 `meta_`.
- Feature classes (from Agent C's table): (a) observable: clock, leader/challenger stats, gaps, widths, log-e, pool
  summaries; (b) derivable-from-own-history: `est_*`, 9 history cols; (c) oracle: `oracle_*`, `label_*`;
  (d) policy-contaminated: `f_K`-derived (unavoidable, essential), `f_n_singletons`, `f_mean_n`, **9 history cols**
  (`f_new_arms_last_*`, `f_best_mean_gain_last_*`, `f_search_frac_last_*`, `f_time_since_last_search` — ranks 3–7
  among top ridge coefficients); (e) metadata: `f_T`, `f_remaining_*`, `f_t_over_T`, `f_K_over_T` (T known → allowed).
  No deployable column reads true means (`tests/test_growing_features.py:156-165`).
- Metadata-only classifier (policy+horizon+allocation one-hots) scores 0.588 bal / 0.624 AUC > clock-only 0.579;
  GBM predicts `meta_policy` from `(t,K,T)` with 76% accuracy. Contamination is larger than RESULTS.md states.
- **No model is saved anywhere; no `phi.json`; no export path; `LearnedPolicy` supports only 6 POLY_VARS**
  (`z,u,k,tau,r,h`), whose k16 bal acc is 0.606 (vs 0.647 for 71 features, 0.596 schedule).
- Ties dropped in fit (`fit_models.py:168`); `--min-precision` dead; k16 run must pass `--se-col label_se_k16`.
- No tests cover `fit_models.py` / `evaluate_policy.py`; `results/` gitignored.

### D — Baselines (complete)
- Policies (`src/cold_start/growing/search_policies.py`): `BernoulliSearch(p)`, `PowerSchedule(alpha,c)` = SEARCH iff
  `K_t < c·max(t,1)^α`, `EpsilonSchedule`, `BracketExpansion(base_width)`, `UniformRandom`, `EvidenceThreshold`,
  `OSEInspired` (T-aware), `EvidenceGatedSchedule` = `cp0` (T-aware; `(K_t<c t^α) & (n_plausible>1) & affordable`).
  `Simulator.decide` (`simulator.py:105-141`) forces SEARCH at `K_t==0` and **forces REFINE at `K_t>=max_live_arms`**.
- `c` is **hard-coded** (`sqrt: (0.5,1.0)`, `cbrt: (1/3,1.5)`, `t23: (2/3,0.8)`), never tuned for regret; offline
  `fit_models.py:293-307` tunes c on a 160-point grid by balanced acc in env-grouped folds (k16 best: 1.31/2.76/0.75).
  `α=1/4` exists nowhere. `fixed_K4/16` = `PowerSchedule(alpha=0, c=K)`. No "always-REFINE-after-init" baseline.
- Warm start everywhere: `seed_initial_arms(state, reservoir, 2, table)` then `start_t=2` (`simulator.py:191-206`).
- REFINE rule shared: `LUCB()` in benchmark and labelling; corpus *generation* rotated `{lucb, ucb, racing}`.
- Recommenders in `recommend.py:68`: `posterior_mean` (Beta(1,1)), `posterior_mean_shrunk` (EB, capped strength),
  `lcb`, `empirical`; production uses `recommend_with_oracle_prior` (Beta moment-matched to the *true* reservoir).
- Seeds: corpus `20,262,460 … 343,348,312`; benchmark `4,504 … 14,111`; tests ≤ 4,242 + 20,260,910.
  **Free ranges: `[20_000, 20_000_000)` or `> 400_000_000`.**
- `cp0` (label continuation policy) is also a benchmark baseline → circularity; must be stated.
- `dp.py` is dead code outside tests; no `TwoPointReservoir` class exists → DP ceiling would need new env class; only
  T ≤ 12 feasible. Low value given 2-arm warm start consumes 2 of ≤12 pulls. **Skip as ceiling; keep as label sanity.**
- Unreproducible doc claims: `fill_to_cap` 36-cell grid, recommender-agnostic 0.1528 triple, "0.5889→0.5854→0.5218".
- Minor: seed-stride collision between shard s labelling batch 1 and shard s+1 harvest (both stride 1_000_003).

### E — Adversarial methodology review (complete) → failure-mode register
| # | Failure mode | Mitigation in this plan |
|---|---|---|
| 1 | Labels are off-policy: `A_t` is one policy-improvement step from `cp0`; pooled `K/t` signal has the **opposite sign** to within-policy (T=200: within-sqrt P(SEARCH) 0.43→0.77 with K/t; pooled 0.62→0.58) | `cp0` is the **primary reference** (H1a); offline diagnostic: per-policy fits + coefficient-sign table + P(SEARCH \| K/t tercile) within vs pooled, written before deployment; M9 on-policy relabelling (reduced) |
| 2 | 64-arm cap converts aggressive policies into "fill to 64 then LUCB"; corpus excludes K≥64 states (12.7%, policy-correlated) | cap-hit fraction + `t_cap_hit` per policy; cap sweep {32, 64, T} at T∈{200,1000}; P0 reported at every cap; training sensitivity dropping cap-demoted rows (`K+k>64`) and horizon-truncated rows (`T−t<k`) |
| 3 | No CRN in `evaluate_policy.py` | shared `base_seed` per cell; CRN identity test; paired bootstrap |
| 4 | Oracle-prior recommender + unattainable μ* | primary deployable recommender; **all five rules scored from the same final state** (free); per-episode best-of-T comparator; never pool raw regret across envs — pool paired differences, cell-normalized |
| 5 | Vectorized ≠ scalar features (`r` fallback confirmed: 0.333 vs 0.568); scaler folding | parity test on harvested **and on-policy** Snapshots (≥1000); deploy the sklearn pipeline object itself (no coefficient folding); prediction-parity test |
| 6 | k=4/16 commitment semantics undefined on-policy; horizon/cap truncation; k16 fit weighted with k1 SEs | explicit semantics (commit `min(k,T−t)`); P6 raw, P6' per-step, **P6g with cp0's affordability guard**; SE column matched to k |
| 7 | Asymmetric tuning; intercept encodes corpus class prior | one protocol for c and τ (same tuning/validation seeds, both global over in-dist envs, family hidden); τ=0.5 reported as a row; Tests B/C: c tuned on in-dist seeds only |
| 8 | Simpson's across envs/horizons (Family B = 80% of corpus; per-env P(SEARCH) 0.35–0.81) | stratified tables first; pooled = cluster bootstrap over environments of per-cell paired differences |
| 9 | Pseudoreplication; horizons sharing seeds; multiplicity | distinct `base_seed` per (env,T,cap) cell; episode = unit; pre-registered primary contrasts; rest labelled exploratory |
| 10 | Seed collisions / no disjointness guard | reserved ranges + startup assertion against corpus/benchmark seed sets |
| 11 | Shift beyond K: allocation (2/3 of corpus is ucb/racing, deployment LUCB), history fingerprints, tie exclusion | history excluded from primary variants (P10 = with); `lucb-only` training variant; shift analysis via scalar `extract_features` on logged Snapshots |
| 12 | Horizon transfer extrapolates raw `f_t, f_T, f_remaining_budget` | Test D also run with a **scale-free CLOCK** variant (`f_t_over_T, f_remaining_frac, f_K_over_sqrt_t, f_log_t, f_log_K, f_K_over_T`) |
| 13 | Degenerate baselines at cap=T (singletons; Beta(1,1) prefers 1-of-1) | run P0/P1 at cap=T; report `n_rec`, singleton count |
| 14 | DP validation claimed but absent | out of scope; state plainly in the report |
- RESULTS.md errata to record: ablation D = clock + one column; `f_challenger_ucb` is CS-derived; k1–k16 agreement 70.4% among decided; 6-var k=1 Φ is at chance (0.5065); PILOT §4 recommender claim measured on one policy; "~100k" → 87,148 with 12.7% cap-dropped.

## Experimental design (final)

### Pre-registered primary comparisons (stated before any deployment run)
- **H1a (necessary condition):** on Test A, the learned k=16 clock+quality+evidence policy (P9₁₆, τ_val) has lower
  mean paired regret than `cp0` — the continuation policy its labels are defined against. If this fails, nothing
  else about "learned beats X" is interpretable (labels are one improvement step from cp0).
- **H1b (practical claim):** P9₁₆ vs the best *validation-tuned* power schedule P3*. One paired contrast.
- **H2 (e-process value):** P9 (clock+quality+evidence) vs P7 (clock+quality), same k, same τ protocol.
- **H3 (surrogate validity):** Spearman rank correlation between offline OOF AUC and deployed Test-A regret across
  all learned variants; a null/negative correlation is reported as a finding, not tuned away.
- Everything else (P4/P5, P10, P11, P12, Tests B/C/D, cap sweep, recommender sensitivity) is secondary/exploratory
  and labelled as such; no multiplicity correction is claimed for exploratory rows.

### Shared harness (identical for every policy)
- Simulator `Simulator(table, reservoir, allocation=LUCB(), search_policy, horizon=T, max_live_arms=cap)`; warm start
  `seed_initial_arms(state, reservoir, 2, table)`, `run_to_horizon(start_t=2)` — matches the corpus harness.
- **CRN pairing**: every policy in a cell `(env, T, split, cap)` uses the *same* `base_seed`; M replicates are the
  episodes; policy-internal RNG = `np.random.default_rng([base_seed, crc32(policy_name)])` (fresh per policy per cell).
  Test that arm-mean sequences and reward prefixes are bit-identical across policies with the same `base_seed`.
- **Recommender (deployable, primary)**: `posterior_mean_shrunk` (EB prior fitted from own discovered arms, strength
  capped at K; `recommend.py:114-176`) — the repo's designed deployable counterpart of the label harness's oracle prior.
  **All five rules** (`posterior_mean_shrunk`, `lcb`, `posterior_mean`, `empirical`, `oracle_prior`) are scored from
  the same final state in every run (the recommendation does not affect the trajectory, so this is free); every
  table is emitted per recommender; conclusions stated per recommender. `R_disc` is recommender-independent.
- **Metrics per episode**: `Q_T = μ_rec` (true mean of recommended arm; primary quality);
  `μ*_ep = max_{k<T} μ_k` over the episode's CRN reservoir stream (`reservoir_uniforms(base_seed, rep, k)` →
  `sample_from_uniforms`) — the "full-search oracle" comparator; `R_T = μ*_ep − Q_T` (primary regret);
  `R_sup = 1 − Q_T` (repo's existing definition, for continuity); decomposition
  `R_disc = μ*_ep − max_{a∈D_T} μ_a ≥ 0`, `R_sel = max_{a∈D_T} μ_a − Q_T`, `R_T = R_disc + R_sel`.
  Also: `K_final`, `search_frac`, `cap_hit` (bool), `t_cap_hit`, `n_rec` (pulls on recommended arm),
  `n_eliminated_final`, Herfindahl of pulls over active arms. Also the cap-feasible comparator
  `max_{k<min(T,cap)} μ_k` as a side column.
- **Unit of independence = episode (replicate)**. Per cell: paired differences vs reference, mean ± 95% paired
  bootstrap CI (10k resamples), paired win-rate. Pooled: equal weight per cell, cell-stratified bootstrap. Always
  report per family × horizon; pooled numbers are secondary (Simpson's guard).

### Seeds (disjoint from corpus `[20.26M, 343.4M]` and old benchmark `[4.5k, 14.1k]`)
- Tuning (schedule c): `base_seed = 1_000_000 + cell_id·1_000`; Validation (τ, model selection):
  `5_000_000 + cell_id·1_000`; **Test**: `10_000_000 + cell_id·1_000`. `cell_id` enumerates (env, T, cap).

### Environments / horizons / episodes
- **Main panel (Test A, in-distribution)**: 8 corpus envs spanning both families (ids verified in
  `label_states.py:63-79`, none among the 6 degenerate tail combos): `beta_good_common` (5,2),
  `beta_rare_excellent` (1,9), `beta_mostly_mediocre` (8,8), `beta_skewed` (0.5,3); `tail_b0.5_mu1.0_c1.0`,
  `tail_b2.0_mu1.0_c1.0`, `tail_b8.0_mu1.0_c1.0`, `tail_b1.0_mu0.9_c2.0`.
  Horizons `{50,100,200,500,1000}`; **M = 2000 test episodes per cell**; cap 64.
- **Test C (held-out family)**: 3 mixture presets (`MIXTURE_PRESETS`) × 5 horizons, M=2000; plus leave-one-family-out
  models (train Beta-only → deploy on tail cells; train tail-only → deploy on Beta cells).
- **Test B (held-out generating regime)**: models trained excluding `{aggressive, random, epsilon}` (the cap-selected,
  fingerprint-heavy policies) and excluding history features; deployed on Test A cells. Compares to full model.
- **Test D (horizon transfer)**: models trained on T∈{50,100,200,500} deployed at T=1000; trained on {50,100,500,1000}
  deployed at T=200; extrapolation to T=2000 on 3 envs, M=1000 (CS table built on demand).
- **Robustness sweep**: all 30 corpus envs × 5 horizons, M=500, policies {P0, P1, P3*, cp0, P7₁₆, P9₁₆} only.
- **Cap sensitivity**: T∈{200,1000}, cap∈{32, 64, T}, 4 envs, M=1000, policies {P0, P1, P3*, cp0, P7₁₆, P9₁₆}.
  (Learned models were trained on cap-64 states; K>64 is out of support at cap=T — reported, not hidden.)
- **Seed-disjointness assertion** at startup of every run script: the set of `base_seed`s used ∩ (corpus seed set ∪
  old-benchmark seed set) = ∅, computed from `label_states.build_shards` and the formulas in Agent D §11.

### Policies
| id | name | rule | tuning |
|---|---|---|---|
| P0 | always_search | `BernoulliSearch(p=1)` (fills cap then LUCB) | — |
| P1 | refine_after_init_K0 | `PowerSchedule(alpha=0, c=K0)`, K0∈{2,4,8} | K0 chosen on tuning seeds per T |
| P2 | fixed growth | `UniformRandom` (coin), `fixed_K16` | — |
| P3 | power schedule | `K_t < c·t^α`, α∈{1/4,1/3,1/2,2/3} | c ∈ 40-pt log-grid per (α,T) on **tuning seeds**, pooled over in-dist envs (family hidden); P3* = best (α,c) per T on **validation** seeds. Also report per-(env,T) oracle-tuned schedule as a labelled upper bound |
| **cp0** | label continuation | `EvidenceGatedSchedule(0.5,1.0,2)` — **primary reference (H1a)**; circularity stated | — |
| P4/P5/P6 | Φ_k, k=1/4/16, feature set = CLOCK+QUALITY+EVIDENCE (no history) | commit `min(k, T−t)` rounds then re-evaluate | τ on validation |
| P6' | Φ₁₆ **without** commitment (per-step use of the k=16 model) | isolates model vs commitment mechanism | τ on validation |
| P6g | Φ₁₆ commitment **with cp0's affordability guard** (`remaining ≥ 2·(K+1)` else REFINE) | same guard cp0 has; tests whether the learned rule fails by over-recruiting late | τ on validation |
| P7 | quality-only Φ₁₆ = CLOCK+QUALITY | | τ on validation |
| P8 | clock-only Φ₁₆ = CLOCK | learned analogue of a schedule | τ on validation |
| P9 | CLOCK+QUALITY+EVIDENCE Φ₁₆ (= P6) ; P9a = without `f_log_e_pair` (CS-bounds only), P9b = with it | tests e-process value | τ on validation |
| P10 | full-71 Φ₁₆ (adds HISTORY) — the exact model behind RESULTS.md's 0.647 | tests fingerprint shift on-policy | τ on validation |
| P11 | reservoir-aware simple rule: logistic on `{log Î_t, log p̂_t, remaining_frac, f_leader_width, log K}` + a hand rule `Î_t·(T−t) > τ·f_leader_width` | §13 | τ on validation |
| P12 | boosted Φ₁₆ (HGB) on P9 features | model-class sensitivity | τ on validation |

**Feature groups (explicit column lists; replaces substring selectors):**
- CLOCK (11): `f_t, f_T, f_remaining_budget, f_remaining_frac, f_t_over_T, f_K, f_K_over_t, f_K_over_T, f_log_t, f_log_K, f_K_over_sqrt_t`
- QUALITY (point estimates only, 22): `f_leader_n, f_leader_mean, f_leader_n_frac, f_best_mean, f_second_best_mean,
  f_mean_of_means, f_sd_of_means, f_max_n, f_mean_n, f_n_singletons, f_empirical_gap, est_top_gap,
  est_top_gap_normalized, est_frac_arms_within_5pct_of_best, est_beta_a, est_beta_b, est_beta_mean,
  est_p_new_beats_incumbent{,_plus_0.01,_plus_0.05,_plus_0.1}, est_hill_tail_index, est_quantile_{0.5,0.9,0.99}`
  (exact count fixed at implementation; anything derived from CS bounds or log-e is *not* QUALITY)
- EVIDENCE (CS/e-process derived, ~24): `f_leader_{lcb,ucb,width}, f_csleader_{n,mean,lcb,ucb,width,n_frac},
  f_challenger_{n,mean,lcb,ucb,width,n_frac}, f_lcb_lead_minus_ucb_chal, f_ucb_chal_minus_lcb_lead, f_is_separated,
  f_n_plausible, f_frac_plausible, f_n_eliminated, f_frac_eliminated, f_max_width_plausible, f_mean_width_plausible,
  f_mean_width_all, f_log_e_pair`
- HISTORY (9, fingerprint): `f_new_arms_last_{10,25,50}, f_best_mean_gain_last_{10,25,50}, f_search_frac_last_{10,25},
  f_time_since_last_search`
- CLOCK_SF (scale-free, for Test D): `f_t_over_T, f_remaining_frac, f_K_over_sqrt_t, f_log_t, f_log_K, f_K_over_T`
- Feature-hygiene table (per column: class a/b/c/d/e/f, group, used-by-policies) written to
  `results/growing_bandits/deploy/feature_hygiene.csv`. Hard rule: no `oracle_*`, `label_*`, `meta_*` in any policy.

### Training protocol (`fit_models.py` reproduced, corrected)
- Rows: exact ties dropped (`A_k == 0`); **sensitivities**: (i) also drop `|A_k| < 1·SE_k` ("ambiguous");
  (ii) drop horizon-truncated (`T−t < k`) and cap-demoted (`K+k > 64`) rows for k∈{4,16}; (iii) `lucb`-allocation
  rows only. Label = `A_k > 0`. Use `label_se_k{k}` matching k.
- **Offline diagnostics written before any deployment** (`offline_diagnostics.csv`): per-generating-policy fits with
  coefficient signs for `f_K, f_K_over_t, f_n_singletons`; P(SEARCH | K/t tercile) within-policy vs pooled per
  horizon (the Simpson's reversal); metadata-only classifier score; ESS of the weighted variant.
- Estimator: `StandardScaler → LogisticRegression(C=1.0, max_iter=2000)`, **unweighted** (primary);
  precision-weighted variant (existing `precision_weights`) as sensitivity with ESS reported.
  `HistGradientBoostingClassifier(max_depth=6, max_iter=300, lr=0.06, random_state=0)` for P12.
- Offline metrics: OOF pooled AUC and balanced acc under `GroupKFold(5)` by `meta_env`; also by `meta_horizon`,
  `meta_policy`, `meta_family`; within-regime AUC (cells of horizon × allocation × policy × remaining band).
- Final deployed model for Test A: refit on all corpus rows (no test-episode overlap by construction). Save each
  model as `results/growing_bandits/deploy/models/<variant>.joblib` with `{pipeline, feature_list, k, tau, meta}`.
- Threshold τ: (i) offline τ_off = argmax OOF balanced acc; (ii) **τ_val = argmin mean validation regret over
  τ∈{0.3,0.4,0.5,0.6,0.7}** (primary, deployment objective); report both, and the full τ→regret curve on validation.
  Test-set τ-sensitivity is reported but explicitly labelled post-hoc.

### Diagnostics
- **On-policy state shift (§10)**: log feature vectors at 24 normalized times for 128 replicates per Test-A cell for
  each learned policy; vs corpus rows of the same horizon: per-feature quantile coverage (frac outside corpus
  [0.5%,99.5%]), standardized mean shift, KS statistic, kNN(k=5) distance in standardized space relative to the
  corpus self-kNN 95th percentile ("OOD fraction"), and predicted-probability histograms. Flag prominently if
  OOD fraction > 10% in any cell.
- **Dynamics (§11)**: per cell, per step, cross-episode mean of `K_t`, best discovered true μ, best posterior mean,
  quality of the arm that would be recommended now, `n_eliminated`, search rate, pull Herfindahl; stored on a
  50-point `t/T` grid; plotted per family × horizon (dark, minimal style).
- **Decomposition (§12)**: `R_disc`, `R_sel` per episode; bars per policy per cell; paired CIs.
- **Reservoir-quality (§13)**: oracle `p_t = 1−F(μ_inc)`, `I_t = ∫_{μ_inc}^1 (1−F(x))dx` (via `reservoir.tail_prob`,
  numeric quadrature) — diagnostic only; observable `p̂_t`, `Î_t` from the EB Beta fit. Report (a) AUC of oracle
  `p_t`/`I_t` for the corpus label (ceiling for reservoir-aware rules) vs `p̂_t`/`Î_t`; (b) on Test A, correlation
  between the learned policy's SEARCH decisions and oracle `I_t`; (c) P11 deployed regret vs P7/P9.
- **Cap analysis**: fraction of episodes hitting the cap and `t_cap_hit` per policy; cap-sensitivity table.

## Implementation milestones (framework-first; new code in `src/cold_start/growing/deploy/`)

**M0 — Freeze the spec.** Copy this plan to `docs/growing_bandits/DEPLOYMENT_PLAN.md`; write the failure-mode
register (`docs/growing_bandits/DEPLOYMENT_FAILURE_MODES.md`) from Agent E; commit. Regenerate nothing yet.

**M1 — Feature parity layer** (`deploy/features_vec.py`, `deploy/feature_groups.py`, `deploy/history_vec.py`)
- `VecSearchHistory` (M×t bool decisions, M×t best-mean trace) mirroring `features.SearchHistory` semantics exactly.
  Verified semantics (`generate_states.py:102-109,148`): snapshot at `t` is taken *before* step `t`; history holds
  `res.searched` and `sim.best_posterior_mean(state)` appended *after* each step `2..t−1` (length `t−2`).
- `extract_features_vec(state, ctx, table, pairwise_table, history) -> dict[str, (M,) array]` for all 71 columns.
  `f_log_e_pair` via `PairwiseEvidence().tabulate(max_n=T, max_bytes=8 GiB)` cached as memmap under
  `data/pairwise_tables/pair_T{T}_G512.npz`; fall back to chunked exact `log_e` if tabulation is impossible.
- Fix the two known fallbacks to match `features.py` exactly (`_fit_beta_moments` Beta(1,1) fallback; unjittered
  argmax for leader/challenger identity in feature computation only — allocation keeps jitter).
- **Test** `tests/test_deploy_feature_parity.py`: harvest ≥200 diverse states via `generate_states.harvest`, compute
  scalar `extract_features` vs vectorized; assert `allclose(rtol=1e-5, atol=1e-6)` for all 71 columns (float32
  table tolerance for `f_log_e_pair`: `atol=1e-3`). Also parity of history features across a full trajectory.
  A second, slower parity check runs inside the M6 smoke: ≥1000 **on-policy** states from the deployed `ModelPolicy`
  dumped as `Snapshot`s → scalar `extract_features` vs the vectorized values the policy actually used (Agent E #5).

**M2 — Harness** (`deploy/harness.py`, `deploy/comparators.py`, `deploy/recommenders.py`, `deploy/stats.py`)
- `run_cell(spec: CellSpec, policy: SearchPolicy, ...) -> EpisodeTable` with per-episode metrics + optional dynamics
  grid + optional state logging. Common `base_seed`. Recommender pluggable by name.
- `mu_star_episode(reservoir, base_seed, M, T)` via `reservoir_uniforms`.
- Stats: paired bootstrap CI, stratified pooling, Spearman with bootstrap CI.
- **Tests**: CRN identity across policies; budget conservation; `R_disc ≥ 0`; `R_T == R_disc + R_sel`;
  always-search hits cap at t=64; recommender names resolve; stats on synthetic data.

**M3 — Deployable model policy** (`deploy/model_policy.py`)
- `ModelPolicy(SearchPolicy)`: `{pipeline, features, tau, k}`; per-replicate commitment counter and committed action;
  at decision time if counter==0 → predict_proba → action, counter = `min(k, T−t)`; else follow commitment,
  decrement. `needs_evidence=False`; K==0 forced search; cap demotion handled by the simulator (record demotions).
- `RuleFactory` registry: `always_search`, `refine_after_init`, `power`, `cp0`, `uniform`, `fixed_K`,
  `reservoir_rule`, `model:<path>`.
- **Tests**: commitment truncation at horizon; k=1 equals per-step; τ monotonicity (higher τ → fewer searches);
  loaded model reproduces offline predictions on corpus rows (`predict_proba` parity vs training pipeline).

**M4 — Training** (`experiments/growing_bandits/deploy/train_policies.py`)
- Loads corpus; builds explicit feature groups; trains every variant (k∈{1,4,16} × {CLOCK, CLOCK+QUALITY,
  CLOCK+QUALITY+EVIDENCE−log_e, +log_e, +HISTORY} × {logit, HGB(P9 only)} × {unweighted, weighted(P9 only)} ×
  {ties-dropped, ambiguous-dropped(P9 only)}; generalization variants for Tests B/C/D). Writes `offline_metrics.csv`
  (OOF AUC/bal-acc by grouping, within-regime AUC, class balance, ESS), `feature_hygiene.csv`, models as joblib.
  Also `reservoir_diagnostics_offline.csv` (oracle p_t/I_t AUC vs estimates).
- **Test**: training is deterministic (re-run → identical coefficients); no prohibited column can enter (schema guard).

**M5 — Tuning & threshold selection** (`tune_baselines.py`, `select_thresholds.py`)
- Schedules: grid over c per (α, T) on tuning seeds, M=500, 8 envs pooled → `schedule_tuning.csv`, P3* per T chosen on
  validation seeds. K0 for P1 likewise.
- τ per learned model on validation seeds (M=500, 8 envs × 5 T) → `threshold_selection.csv` with the full curve.

**M6 — Deployment runs** (`run_deployment.py`; resumable manifest; `multiprocessing` over cells, 12 workers;
per-episode results to `results/growing_bandits/deploy/episodes/<test>/<cell>/<policy>.parquet` + CSV summaries)
- Order: Test A main panel → Test C → Test B → Test D → robustness sweep → cap sensitivity → recommender sensitivity.
- Smoke first: 2 envs × T=100 × M=100 × all policies, checked end-to-end before launching long jobs.

**M7 — Analysis & figures** (`analyze_deployment.py`, `make_deploy_figures.py`)
- Tables: main (per family×T and pooled): mean R_T, Q_T, R_disc, R_sel, K_final, search_frac, cap_hit, with paired
  CIs vs P3*; offline-vs-deployed table (§9) + Spearman; τ curves; OOD table; cap table; recommender sensitivity.
- Figures (dark, minimal): regret vs T per family; dynamics panels; decomposition bars; offline-AUC vs deployed-regret
  scatter; OOD coverage heatmap; τ sensitivity; reservoir p_t/I_t vs decision.

**M8 — Adversarial re-review + report.** Dispatch a fresh reviewer agent over the produced tables/code with the
failure-mode register as checklist; fix; then write `docs/growing_bandits/DEPLOYMENT_RESULTS.md` (what was found,
per-recommender, per-test, mechanistic explanation from dynamics/decomposition, what is and is not established),
plus a short errata section for RESULTS.md (Agent E's list). Update RUNBOOK with exact commands. Commit code +
docs + CSV tables (episodes parquet gitignored).

**M9 — On-policy relabelling (reduced scale; the direct test of "labels are off-policy").** Harvest ~10k states
from the deployed P9₁₆ on Test-A envs (fresh seeds), label k=16 only with the **learned policy as the continuation**
(reuse `label_state` with a `sim_factory` whose `search_policy` is the `ModelPolicy`; `max_replicates=1024`),
retrain Φ' on the union (corpus + on-policy labels) and on-policy-only, deploy both on Test A. Report whether
one policy-iteration step changes the deployed result. Run after M7 (background), ~1–2 h; skipped only if the
main runs overrun, and then stated as not done.

### Compute estimate (14 cores, vectorized over M)
- A cell = one `(env, T, policy)` run of M episodes; cost ≈ T steps × (LUCB + pull + features). Baselines ≈ 5–20 ms/step,
  learned ≈ 30–80 ms/step with the tabulated log-e path (chunk M if the (M,512) gather gets large).
- Test A: 8 envs × 5 T × ~22 policies × M=2000 ≈ 880 cells; worst cells (T=1000, learned) ≈ 1–2 min → ≈ 6–10 core-hours
  → **< 1 h wall on 12 workers**. Tests B/C/D + robustness (30×5×6, M=500) + cap + tuning/validation add ≈ 2–3× →
  total ≈ 3–5 h wall of simulation, run in the background with a resumable manifest. Pairwise tables: T=1000 ≈ 4 GB
  memmap, built once (minutes). M9 relabelling ≈ 1–2 h.

### Subagent orchestration (execution phase; ultracode on)
- Wave 1 (parallel, 4 agents, worktree-isolated by file ownership): M1 features+parity; M2 harness+stats;
  M3 model policy (stubs against M1 interface); M4 trainer. Shared `feature_groups.py` written first by the lead.
- Wave 2: integration + smoke; M5 tuning runs (background); analysis scaffolding against smoke output.
- Wave 3: M6 long runs in background (`run_in_background`), monitored; M7 analysis code finished meanwhile.
- Wave 4: M8 adversarial reviewer (independent agent) + report writer; lead verifies every number against CSVs.

## Verification
- `make test` green including new `tests/test_deploy_*.py`; `make lint` clean.
- Parity: vectorized features == scalar features on ≥200 harvested states; `ModelPolicy.predict_proba` on a corpus
  row == training pipeline output.
- CRN: two policies with the same `base_seed` see identical first-64 arm means and reward prefixes (test).
- Smoke deployment (2 envs × T=100 × M=100) produces the full table schema; budget conserved; `R_disc ≥ 0`.
- Reproduce RESULTS.md's headline offline numbers (0.647 bal-acc for weighted E@k16) from the new trainer before
  changing the protocol, to prove the reproduction is faithful.
- Every number in `DEPLOYMENT_RESULTS.md` traceable to a CSV in `results/growing_bandits/deploy/`.
- Final check that no deployed policy imports/reads `state.mu`, `reservoir`, `oracle_*`, or `meta_*` (grep + test).

---

## Pre-registration 2 — the replacement contrast H1b′ (registered 2026-09-20, before any run)

This section is a second, separate pre-registration. It is written **after** the first study's results
were read and **before** a single episode of the run it registers exists; the commit that adds it precedes
the commit that adds the run's manifest lines, and that ordering is the whole of its claim to honesty.
It is hypothesis switching after seeing the data, and it is labelled as such.

### H1b as written is answered: null, and under-powered

The pre-registered H1b — `phi_k16` (P9₁₆, τ_val) vs the validation-tuned schedule P3\*, pooled over five
horizons at cap 64 — is **null** in all six tests (`DEPLOYMENT_RESULTS.md` §3.2, §7.5), and the study will
not spend more compute on it. Three measured reasons it could not have been anything else:

1. **Power.** The best available effect size is d = 0.328 (mean −0.000891, between-environment sd 0.002719,
   n = 30, `cells_robust_primary.csv`), which needs ~73 environments for 80% power at α = 0.05; Test A has 8.
2. **Structural zeros.** 12 of the 40 Test-A cells have `d_regret_vs_p3_star` exactly 0.0 for `phi_k16`
   (17/40 for `phi_k4`, 20/40 for `phi_k1`): at T ≥ 200 the 64-arm cap makes Φ and P3\* the same policy.
3. **The contrast at T ≥ 200 is not the claimed one.** It is "Φ against fill-the-cap", not "Φ against a
   schedule" (`DEPLOYMENT_RESULTS.md` §4).

### What motivated the replacement, and why that evidence is excluded

On Test A, the two short-commitment variants beat P3\* in eight of eight environments (`phi_k4` −0.003445,
cl[−0.006266, −0.000624], p = 0.023; `phi_k1` −0.003429, cl[−0.006280, −0.000578], p = 0.025), where
`phi_k16` does not (5 of 8). Those numbers were **selected** from 25 exploratory variants after the fact and
do not survive Holm (min adjusted 0.585). They are the reason this contrast exists; **they are never quoted
as evidence for it**, and Test A is not part of H1b′'s evidence.

### H1b′ (one contrast)

- **Policy:** `phi_k4` — the k = 4 commitment model, τ from `thresholds.json` exactly as deployed in Test A.
  Chosen over `phi_k1` because the Test-A effects are indistinguishable (Δ 1.6e−5) and `phi_k4` costs 3.6×
  less to run; that choice was made on Test A, which is why Test A is excluded above.
- **Reference:** `p3_star`, the validation-tuned power schedule, constants as deployed (`baseline_params.json`).
- **Panel:** the 30-environment robustness panel (`--test robust`: all corpus environments, test-split seeds,
  cap 64, M = 500). Its 150 cells already hold `p3_star`; the run adds `phi_k4` to them and nothing else.
- **Horizons for the primary statistic:** T ∈ {50, 100, 200} — the horizons where the two policies can differ
  (reason 2 above). The T ∈ {500, 1000} cells are run too so the standard tables are complete, and are
  reported as secondary.
- **Primary statistic:** pooled over the 90 cells (30 environments × 3 horizons), Δ = regret(`phi_k4`) −
  regret(`p3_star`) per episode, cell-stratified paired bootstrap for the paired CI, and the
  **environment-mean t interval on n = 30** with its two-sided `cluster_p`. Negative favours `phi_k4`.
- **Minimum effect of interest:** |Δ| ≥ **0.002** pooled — about 2% of P3\*'s regret on this panel, and the
  smallest difference the study would act on given that the cap mis-sizing alone is worth 0.02
  (`k_star_envelope.csv`). A significant Δ smaller than this is reported as "detectable, not material".
- **Decision rule, stated in advance:** H1b′ is *supported* iff pooled `cluster_p` < 0.05, Δ < 0 and
  |Δ| ≥ 0.002. It is *refuted* iff Δ ≥ 0 or the t interval excludes −0.002 from below (the effect is
  significantly smaller than the MEI). Anything else is *inconclusive* and is reported as that word.
- **Secondary rows:** the three per-horizon strata (n = 30 each), Holm-corrected among themselves; the two
  long-horizon strata, uncorrected and labelled structural.
- **Family:** one primary row. No other contrast in this registration.

### What either outcome means

- **Supported:** a learned short-commitment SEARCH rule beats a validation-tuned schedule *on the training
  corpus's environments*. That is in-distribution robustness at n = 30 (non-claim 11 of the results
  document still applies) — never generalization, which Test C already answered against.
- **Refuted or inconclusive:** the learned-policy line closes. The write-up is the negative result the data
  already support: the arm budget and the commitment horizon set deployed regret, not the decision
  classifier.

### Run, in order

```
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test robust --resume --workers 12 \
    --policies phi_k4
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test robust --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py     # writes tables/h1b_prime.csv
```

---

## Pre-registration 3 — the null model H1b′ must beat (registered 2026-09-20, before selection or run)

H1b′ was supported (`DEPLOYMENT_RESULTS.md` §12.3), and §12.2 says why in a way that undercuts it: `phi_k4`
merely matches front-loaded search at its own arm count. The question that remains is whether the learned
model carries *any* information a schedule cannot. This registers the schedule it must beat and the rule for
reading the result, before the schedule's constants are selected.

### The null model, and why this form (a declared researcher degree of freedom)

**`fixed_K_star`: recruit to K(T) arms as fast as possible, then refine only; K(T) chosen per horizon on the
validation split.** The functional form was chosen *after* reading §12.1–12.2 — they say the whole effect is
"which K, and reach it early" — so the roadmap's 3-parameter feature rule (NEXT-STEPS §3.4) is replaced by
the simplest rule that embodies that finding. This is a researcher degree of freedom and is declared as such.
It has no features, costs what `p3_star` costs, and is the existing `fixed_K` kind with a per-horizon K
(`policy_table.py`, `fixed_K_star`; constants in `baseline_params.json["fixed_K_star"]`).

- **Selection:** for each T ∈ {50, 100, 200, 500, 1000}, the 8 main environments on the **validation** split,
  cap 64, M = 2000, K over the grid {4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64} ∩ [3, T]
  (≤ 17 candidates per horizon); K(T) = argmin of the environment-equal-weight pooled regret
  (`select_fixed_k.py`, `tables/fixed_k_selection.csv`). The same protocol P3\* was selected under, with a
  smaller candidate set. Selection is on deployed validation regret, so it carries the same winner's curse as
  P3\*; at the measured selection SE that is ≤ 1.7e−3, below the minimum effect of interest.
- **Deployment:** `fixed_K_star` on the robustness panel (150 cells), Test A and Test C, test-split seeds,
  CRN-paired with every policy already there.

### The registered contrasts

- **Primary — H1b″:** `phi_k4` − `fixed_K_star` on the robustness panel at T ∈ {50, 100, 200}, pooled over 90
  cells, environment-mean t on n = 30, MEI 0.002, the same rule as H1b′:
  *supported* iff `cluster_p` < 0.05, Δ < 0 and |Δ| ≥ 0.002 — the learned model carries information a
  per-horizon fixed K does not; *refuted* iff Δ ≥ 0 or the t interval lies above −0.002 — it does not;
  otherwise *inconclusive*.
- **Secondary:** `fixed_K_star` − `p3_star` on the same cells (does the null model itself beat the tuned power
  schedule, i.e. is H1b′'s effect available without a classifier?), and the per-horizon rows of the primary,
  Holm-corrected among the three.

### What either outcome means

- **Refuted:** "a per-horizon fixed K selected on validation regret matches a 62-feature logistic policy
  trained on 87,148 oracle-labelled states." The corpus-labelling line retires; the paper is about the arm
  budget.
- **Supported:** the study's first defensible claim that the learned model carries information a simple rule
  cannot — on the training corpus's environments, at T ≤ 200, against this null.

### Run, in order

```
.venv/bin/python experiments/growing_bandits/deploy/select_fixed_k.py --workers 12        # validation split
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test robust --resume --workers 12 --policies fixed_K_star
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test A      --resume --workers 12 --policies fixed_K_star
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test C      --resume --workers 12 --policies fixed_K_star
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration h1b_null   # tables/h1b_null.csv
```

---

## Pre-registration 4 — an environment-adaptive schedule as the null model (registered 2026-09-20, before code, selection or run)

§12.4 of the results says the learned model's value is sizing K to the environment. The question that
closes the programme is whether a schedule that *observes* the environment through something already on
the policy's hot path can do the same. If it can, the 62-feature classifier is a lookup table with extra
steps; if it cannot, the learned model is the result.

### The null model (a declared researcher degree of freedom, chosen after reading §12.4)

**`adaptive_K_star`: SEARCH while K_t < K\_target(t), with K\_target = c · T^α · (1 + b · (1 − q_t))**, where
q_t is the fraction of currently held arms whose posterior mean lies within 0.05 of the best held arm's
(`est_frac_arms_within_5pct_of_best`, a QUALITY feature the study already computes; `(S+1)/(n+2)` means, no
confidence sequence, no e-process). In a thin-tailed reservoir q_t → 1 and the target collapses to the plain
schedule c · T^α; in a heavy-tailed one q_t → 0 and it is up to (1 + b) times larger. Three scalars (α, c, b);
b = 0 is a per-horizon power fixed K, so the null nests Pre-registration 3's. It is a new `search_policy`
kind (`adaptive_K`), feature-free apart from that one statistic, evaluated vectorized each step.

- **Selection:** the 8 main environments × T ∈ {50, 100, 200} on the **validation** split, cap 64,
  M = 2000; one (α, c, b) for all horizons, the argmin of pooled regret over the 24 cells, grid
  α ∈ {0.5, 0.75} × c ∈ {1, 2, 3, 4, 6} × b ∈ {0, 1, 2, 4, 8} — 50 candidates (`select_rule.py --rule adaptive_K_star`,
  `tables/adaptive_k_selection.csv`, `baseline_params.json["adaptive_K_star"]`).
- **Deployment:** robustness panel, Test A, Test C; test-split seeds; CRN-paired with everything there.

### The registered contrasts (robustness panel, T ∈ {50, 100, 200}, env-mean t on n = 30, MEI 0.002)

- **Primary — H1b‴:** `phi_k4` − `adaptive_K_star`. *Supported* iff `cluster_p` < 0.05, Δ < 0 and
  |Δ| ≥ 0.002: the learned model carries information a three-parameter environment-adaptive schedule does
  not. *Refuted* iff Δ ≥ 0 or the t interval lies above −0.002: it does not, and the classifier reduces to
  "size K to the observed tail". Otherwise *inconclusive*.
- **Secondary:** `adaptive_K_star` − `fixed_K_star` (does observing the tail buy what §12.4 attributed to it?),
  and the per-horizon rows of the primary, Holm-corrected among the three.

### What either outcome means

- **Refuted:** the programme's result is a three-parameter rule — recruit to c · T^α, more when the held arms
  are spread out — and the oracle-labelling pipeline was an expensive route to it.
- **Supported:** the learned model reads something about the environment that the held arms' 5%-band does
  not carry, on the corpus environments at T ≤ 200. One registered claim, in-distribution.

### Run, in order

```
.venv/bin/python experiments/growing_bandits/deploy/select_rule.py --rule adaptive_K_star --workers 12
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test robust --resume --workers 12 --policies adaptive_K_star
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test A      --resume --workers 12 --policies adaptive_K_star
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test C      --resume --workers 12 --policies adaptive_K_star
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration h1b_adaptive
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration h1b_adaptive_secondary
```

---

## Pre-registration 5 — the rule the model appears to be (registered 2026-09-20, before code, selection or run)

`model_reads.py` (results §12.6) asked `phi_k4` directly what it reads on the H1b′ cells. The answer is the
**best held arm's posterior mean**: the top feature at every horizon, tracking the per-environment K at
|ρ| ≥ 0.93, and a rule on it plus log K reproduces 96 / 93 / 85% of the model's decisions at T = 50 / 100 /
200. This registers that rule as the null model and states, before selection, what beating or matching it
means.

### The null model (a declared researcher degree of freedom, chosen from `model_reads.py`)

**`bestmean_star`: SEARCH while `best_mean_t < θ` **and** `K_t < c · T^α`** — recruit until the best arm you
hold is good enough, never beyond a per-horizon ceiling, then refine. `best_mean_t` is the maximum posterior
mean `(S+1)/(n+2)` over held arms, computed by the rule itself (no evidence pass). Three scalars (θ, α, c);
θ ≥ 1 is the fixed-K schedule of Pre-registration 3, so the null nests it.

- **Selection:** the 8 main environments × T ∈ {50, 100, 200} on the **validation** split, cap 64, M = 2000;
  one (θ, α, c) for all horizons; grid θ ∈ {0.55, 0.60, 0.625, 0.65, 0.675, 0.70} ×
  (α, c) ∈ {(0.5, 3), (0.5, 4), (0.5, 6), (0.5, 8), (0.75, 1), (0.75, 1.5), (0.75, 2), (0.75, 3)} —
  48 candidates (`select_rule.py --rule bestmean_star`, `tables/bestmean_selection.csv`,
  `baseline_params.json["bestmean_star"]`).
- **Deployment:** robustness panel, Test A, Test C; test-split seeds.

### The registered contrasts (robustness panel, T ∈ {50, 100, 200}, env-mean t on n = 30, MEI 0.002)

- **Primary — H1b⁗:** `phi_k4` − `bestmean_star`. *Supported* iff `cluster_p` < 0.05, Δ < 0 and |Δ| ≥ 0.002:
  the model carries information beyond its own top feature. *Refuted* iff Δ ≥ 0 or the t interval lies
  above −0.002: **the model is this rule**, and the programme's result is three numbers.
- **Secondary:** `bestmean_star` − `fixed_K_star` (does the best-mean gate deliver the environment sizing of
  §12.4?), and the per-horizon rows of the primary, Holm-corrected among the three.

### Run, in order

```
.venv/bin/python experiments/growing_bandits/deploy/select_rule.py --rule bestmean_star --workers 12
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test robust --resume --workers 12 --policies bestmean_star
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test A      --resume --workers 12 --policies bestmean_star
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test C      --resume --workers 12 --policies bestmean_star
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration h1b_bestmean
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration h1b_bestmean_secondary
```

---

## Pre-registration 6 — the level-scaled schedule (registered 2026-09-20, before code, selection or run)

Pre-registration 5's gate was inert: on the validation split every θ ≤ 0.65 closed it on a single lucky
first pull (a 1-pull success has posterior mean 0.667) after ~2.5 arms, and selection escaped to a θ the
best mean never reaches before the ceiling. What `model_reads.py` actually shows is that the k = 4 model
stops while every arm has 1–3 pulls — where "the best posterior mean" is no more than *whether first pulls
succeeded*, i.e. an estimate of the **reservoir's mean level**. `f_mean_of_means` tracks the per-environment
K at ρ = −0.96 / −0.98 / −0.93 exactly as `est_quantile_0.99` does. High level (thin-tailed reservoir, the
best is near the typical arm) → few arms; low level (heavy-tailed, excellent arms are rare and far above
typical) → many. This registers that as the null model.

### The null model (a declared researcher degree of freedom, chosen from `model_reads.py` and §12.6)

**`level_star`: SEARCH while K_t < c · T^α · exp(b · (0.5 − level_t))**, where level_t is the mean posterior
mean `(S+1)/(n+2)` over held arms. Three scalars (α, c, b); b = 0 is the fixed-K schedule. At b = 4 a
level of 0.60 scales the target by 0.67 and a level of 0.37 by 1.68 — the 2.5× contrast between
`phi_k4`'s 14 arms in `beta_good_common` and 36 in `tail_b8.0_mu1.0_c1.0` at T = 50.

- **Selection:** as Pre-registrations 4–5 (8 main environments × T ∈ {50, 100, 200}, validation split, cap 64,
  M = 2000), grid (α, c) ∈ {(0.5, 3), (0.5, 4), (0.5, 6), (0.5, 8), (0.75, 1), (0.75, 1.5), (0.75, 2), (0.75, 3)} ×
  b ∈ {0, 2, 4, 6, 8} — 40 candidates (`select_rule.py --rule level_star`).
- **Deployment:** robustness panel, Test A, Test C.

### The registered contrasts (robustness panel, T ∈ {50, 100, 200}, env-mean t on n = 30, MEI 0.002)

- **Primary — H1b⁵:** `phi_k4` − `level_star`; the same rule as before. *Refuted* means the model is a
  level-scaled schedule and the programme's result is three numbers. *Supported* means it reads more than
  the level.
- **Secondary:** `level_star` − `fixed_K_star` (does scaling by the level deliver §12.4's environment sizing?).

Every null registered here is a rule the *learned policy* must beat, so adding registrations makes the
claim "the model carries information a rule cannot" harder to sustain, not easier; the sequence 3 → 4 → 5
→ 6 is reported in full.

---

## Pre-registration 7 — the CRN-paired, per-cap-tuned cap sweep (registered 2026-09-21, before tuning or deployment)

§4 of the results compares caps whose cells drew different seeds (Ruling 26) with constants tuned at cap 64
only (Ruling 20), and §12.1 says the cap was mis-sized four-fold at T = 1000. This registers the sweep that
replaces §4, and what will be read from it.

### Design

- **Cells:** the 8 main environments × T ∈ {200, 1000} × caps {32, 48, 64, 96, 128, 160, 200} at T = 200 and
  {32, 48, 64, 96, 128, 192, 256, 384, 512, 1000} at T = 1000 (`run_deployment.CAPP_HORIZON_CAPS`), M = 2000,
  test-split seeds. **Every cap of an (env, T) runs on that cell's cap-64 seed**
  (`cells.make_matched_cell`), so `mu_star` is bit-identical across caps and `analyze_capp.py` refuses the
  tree if it is not. Test id `capp`; never pooled with `--test cap`.
- **Constants, selected at each cap before deployment** (roadmap 3.3, second half), on the tune / validation
  splits at that cap and the sweep's horizons, M = 500 for every schedule so the three are selected alike:
  `p3_star`'s (α, c) per horizon (`tune_baselines.py --cap X`), `fixed_K_star`'s K per horizon
  (`select_fixed_k.py --cap X`), `level_star`'s (α, c, b) (`select_rule.py --rule level_star --cap X`), and
  τ for `phi_k4` and `phi_k16` (`select_thresholds.py --cap X`). Each lands under `by_cap["X"]`; the
  cap-64 blocks every shipped table used are untouched. A row of `capp_policies.csv` whose deployed
  constant was *not* selected at its cap is stamped `params_tuned = False` and is not read.
- **Policies:** `always_search`, `cp0`, `refine_after_init`, `p3_star`, `fixed_K_star`, `level_star`,
  `phi_k4`, `phi_k16`.

### What will be read (env-mean t on n = 8 throughout; nothing here is multiplicity-corrected, and nothing
here is a claim about a policy beating another — those were Pre-registrations 2–6)

1. **The cross-cap curve, paired for the first time** (`capp_crosscap.csv`): for each policy and T, regret
   at cap X minus regret at cap 64 on the same episodes. The question §12.1 raised from the tune split —
   does a larger cap help at T = 1000, and by how much — answered on the test split with each policy's
   constants selected at that cap.
2. **The schedules under a like-for-like cap** (`capp_contrasts.csv`): `level_star` − `p3_star` and
   `level_star` − `fixed_K_star` at every cap. §12.6 established the level rule's edge at cap 64, T ≤ 200;
   this reports whether it persists at T = 1000 and at caps where K\* is reachable, with τ and every
   constant re-selected.
3. **`phi_k4` − `level_star` at every cap.** Pre-registration 6 found them indistinguishable at cap 64,
   T ≤ 200. A non-null here at a larger cap or T = 1000 would mean the model reads something the level
   rule does not *when it has headroom*; a null everywhere closes that question too.
4. **§4's withdrawn claims, re-measured:** `phi_k16` − `always_search` and `phi_k16` − `p3_star` at
   cap 128 and cap = T, which the document declined to sign on unpaired seeds.

### Run, in order

```
for X in 32 48 96 128: tune / select at X for horizons 200,1000; for X in 160 200: horizon 200;
for X in 192 256 384 512 1000: horizon 1000   (tune_baselines, select_fixed_k, select_rule level_star,
                                              select_thresholds for the two learned variants)
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test capp --workers 12
.venv/bin/python experiments/growing_bandits/deploy/analyze_capp.py
```

---

## Pre-registration 8 — the held-out family with the cap lifted (registered 2026-09-21, before any run)

Test C (§7.2) is the study's only held-out environment family, and it ran under the 64-arm cap that §12.7
shows makes every policy the same policy at T ≥ 200. The probe of §12.8 (unregistered, M = 1000) suggested
that with the cap lifted a schedule tuned on the corpus is within 0.001 of the per-environment ceiling on
those mixtures. This registers that statement.

### Design

- **Cells:** the 3 mixture environments × T ∈ {200, 1000} × cap = T (uncapped), M = 2000, test-split seeds,
  each cell on its Test-C cap-64 seed (`cells.make_matched_cell`) so it is CRN-paired with §7.2's cells.
  Test id `capc`. Policies: the eight of `--test capp`, every constant the cap-T one selected on the
  **corpus** environments in Pre-registration 7 — the mixtures never voted on any constant.
- **Inference at n_envs = 3.** No environment-level interval exists below `CLUSTER_MIN_ENVS`. Every rule
  below is stated on the cell-stratified **paired** CI over the six cells, with the three-environment range
  reported beside it. This is weaker than every other registration and is labelled so wherever quoted.
  MEI = 0.002 throughout.

### The registered contrasts

1. **Primary — the schedule generalizes (a non-inferiority claim):** Δ = `p3_star` (corpus-tuned, cap-T
   constants) − `fixed_K_star` (its corpus-selected K(T) at cap = T; "the best single K"). *Supported* iff the
   paired CI's upper bound is below +MEI — the schedule is not worse off-family than the best single K by
   more than the minimum effect of interest; *refuted* iff the paired CI's lower bound is at or above +MEI;
   otherwise *inconclusive*.
2. **Secondary — the signals do not help off-family (not-better claims):** Δ = `level_star` − `p3_star` and
   Δ = `phi_k4` − `p3_star`. *Supported* iff the paired CI's lower bound is above −MEI — the level rule /
   the learned policy is not better off-family than the schedule by more than the MEI; *refuted* iff the
   paired CI's upper bound is at or below −MEI; otherwise *inconclusive*. Reported, not corrected.
3. **Descriptive:** each rule's gap to the per-environment ceiling (`fixed_K` at the mixture's own K\*(T)
   from `k_star_envelope_all33.csv`, a tune-split argmin, never a deployable policy).

### Run

```
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test capc --workers 12
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration capc_primary
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration capc_level
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration capc_phi
```
