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

---

## Pre-registration 9 — real prompt pools on WebArena Gmail (registered 2026-09-27, before any paid episode)

Every result in §1.0 and §12 was measured on synthetic reservoirs. This registers the first test on real
prompts. Design: `docs/superpowers/specs/2026-09-26-empirical-reservoir-design.md`; implementation plan
`docs/superpowers/plans/2026-09-26-empirical-reservoir.md`.

### Design

- **Pools (frozen; sha256 in `data/empirical_pool/manifest.json`):** G = 50 prompts sampled from the
  2,304-point axis grid (seed 20260926); F = 50 instructions written by `claude-opus-4-7` from one fixed
  prompt that never sees the task bank; anchor = the hand-written `baseline`.
  `pool_G.yaml` `20da6911d1b79749aa134f8255257bc4393a24c43930d82588d5d44eba62a4f6`,
  `pool_F.yaml` `b47c2327d0ed6565878ec1c545108587cee3b2fb88ad3fcc250dca87301b15a0`,
  `pool_anchor.yaml` `9681872cd65cdf43374d18e5dd35be3a606653a9235c1330c691cf177e3238e6`,
  `f_generation_raw.json` `6ea12a1e7d7ab6132a788c2c5855c408d5b3e6c0e1fe1c51f8496eb5b0deaa4c`,
  `queue.jsonl` `367cb223a125c1c9c6dbaf74b7d388d3b7e06be5792cffd18b76cd949261aefc`.
- **Episodes:** every prompt × the 60 Gmail `real-tasks`, once; 300 random (prompt, task) cells a second
  time; `gpt-5.4-mini`, low effort, 30 steps, 180 s; one shuffled queue, pilot (5 + 5 + anchor = 660
  episodes) first. An agent timeout is a task failure (0). An infrastructure error — a harness exception, or
  an episode that did not succeed and ended in a terminal streak of LLM-provider errors — is retried twice,
  then `missing`, never 0; a transient provider error the agent recovered from is scored normally.
- **Reservoirs:** primary = the NPMLE of the true-rate distribution under x̄ᵢ ~ N(μᵢ, v̂/nᵢ), v̂ from the
  replicate pairs (pooled unless the pools differ by more than the bootstrap SE of their difference);
  sensitivity = raw rates and the best-AIC parametric fit (Beta, TailReservoir, two-component Beta mixture).
  `replay.py estimate` freezes one outcomes snapshot and a reservoir manifest; every later stage verifies it.
- **Replay:** cap = T, M = 1,000, CRN seeds from 400,000,000; policies `always_search`, `p3_star`,
  `fixed_K_star`, `level_star`, `phi_k4` with cap-T constants selected **on the corpus only** (caps 50,
  100 and 500 selected for this registration, before any episode, by Pre-registration 7's procedure); fixed K
  over `DEFAULT_K_GRID` for the U-curves.
- **Interval of record:** the prompt bootstrap — B = 200 resamples of each pool's prompts with their
  outcomes, v̂ and the NPMLE re-estimated, the four contrast policies redeployed at M = 250 on fresh seeds;
  95% percentile interval; exactly replicates 0–199 must be present. The episode-paired CI is reported beside.
- **Flatness guard:** a primary cell (npmle, T ∈ {50, 100, 200}) is informative iff its fixed-K regret
  range exceeds 5 × MEI = 0.01. Fewer than 2 informative cells → every verdict is **uninformative: K
  barely matters on real Gmail prompts**. Otherwise contrasts pool the informative cells, equally weighted.

### The registered contrasts (MEI = 0.002)

1. **Primary (non-inferiority):** Δ = `p3_star` − `fixed_K_star`. Supported iff the bootstrap upper bound
   < +MEI; refuted iff the lower bound ≥ +MEI; else inconclusive. (`emp_primary`)
2. **Secondary (not-better):** Δ = `level_star` − `p3_star` (`emp_level`) and `phi_k4` − `p3_star`
   (`emp_phi`). Supported iff the lower bound > −MEI; refuted iff the upper bound ≤ −MEI; else
   inconclusive. Uncorrected.
3. **Secondary:** the lower-level pool has the strictly larger K\*(T) at each primary T (ties reported as
   ties); reported with the ratio against exp(4 · (ℓ_high − ℓ_low)).
4. **Descriptive:** pool level / sd / q99 − mean against the 33 corpus environments; each pool's K\*(T)
   located in the corpus K\* envelope (`k_star_envelope_all33.csv`); U-curves; each rule's gap to the
   pool's K\*; cap-64 cost at T ∈ {500, 1000} (extrapolation beyond 50 prompts).
5. **Sensitivity:** 1–2 on raw and parametric reservoirs, no verdicts.

### Gates

- **G1 rehearsal — passed** (`results/growing_bandits/empirical/rehearsal.json`): flat pool NPMLE sd 0.0199
  vs realized 0.0271 (raw 0.0431); wide pool 0.1555 vs 0.1579; K\*(T = 200) estimated 24 vs true 32 (one
  grid step). Noted: on a flat pool the NPMLE under-states the spread by about a quarter.
- **G2 pilot:** cost ≤ $0.05 per episode; missing ≤ 5% counting never-attempted pilot items; all 8 workers
  productive; anchor inside the Binomial(n, 0.66) 95% band; **zero watchdog relaunches**. Budget cap $40.
- **G3 collection:** missing ≤ 5% counting never-attempted queue items. Hard budget stop $260; a worker
  that sees 3 consecutive provider-caused `missing` items stops the run with STATUS `provider_down`.

### Run

```
scripts/run_empirical_pool.sh --pilot --budget 40 && .venv/bin/python experiments/growing_bandits/empirical/gates.py pilot
scripts/run_empirical_pool.sh --budget 260        && .venv/bin/python experiments/growing_bandits/empirical/gates.py collection
.venv/bin/python experiments/growing_bandits/empirical/replay.py estimate
.venv/bin/python experiments/growing_bandits/empirical/replay.py point --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --workers 12
.venv/bin/python experiments/growing_bandits/empirical/describe.py
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot --workers 12
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration emp_primary
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration emp_level
.venv/bin/python experiments/growing_bandits/deploy/registered_contrast.py --registration emp_phi
```

### Pre-registration 9 — Amendment 1 (2026-09-27, after a paused pilot, before any analysis)

**Why.** The pilot was paused at 197 of 660 episodes ($20.21) because real cost is ~$0.10–0.11 per
episode, not the $0.036 the budget assumed from the hand-written arms' historical runs; the diverse
prompts produce longer episodes (≥ 20 steps: 30% of episodes, 66% of spend). The full design would have
cost ~$660. Sanjay chose to halve the task bank. **The change is driven by cost alone.**

**Disclosure.** Before this amendment the controller saw, while diagnosing cost, the pilot's aggregate
success rates by pool (G 0.726 over 95 episodes, F 0.684 over 76, anchor 0.60 over 20) and per-step-bucket
success rates. No per-prompt score, reservoir, K-grid or contrast was computed. The task subset below is
chosen by the pool seed, stratified by difficulty, without reference to any outcome.

**What changes.**
- **Task bank:** every prompt runs a fixed 30 of the 60 Gmail tasks — 10 easy, 10 medium, 10 hard,
  sampled by the pool seed (20260926), stratum by stratum (`make_pools.select_tasks`): task_e2, task_e10, task_e11, task_e12, task_e13, task_e14, task_e15, task_e16, task_e19, task_e20, task_m3, task_m4, task_m7, task_m13, task_m14, task_m15, task_m16, task_m17, task_m18, task_m20, task_h6, task_h8, task_h10, task_h12, task_h13, task_h15, task_h16, task_h17, task_h19, task_h20.
- **Queue:** 2 × 50 × 30 + 30 anchor = 3,030 main episodes + 300 replicate cells = 3,330; pilot 330.
  `queue.jsonl` sha256 `4b68eafde264b0ed52cddbc266f5935872731757a680ce355b2282213dec0f33`; `manifest.json` records `n_tasks`, `task_subset`, `amendment`.
  The pool files and `f_generation_raw.json` are unchanged (hashes above verified).
- **Pilot records:** the 108 pilot records on subset tasks count; the 89 on other tasks are moved verbatim
  to `logs/empirical_pool/archive/` (not used in estimation; their cost still counts toward the budget).
- **Noise:** σ̂ᵢ² = v̂ / nᵢ with nᵢ ≤ 30 (≈ ±0.056 per prompt instead of ±0.039).
- **G1, amended.** The single-seed ±0.01 spread check fails at 30 tasks (flat pool NPMLE sd 0.0510 vs
  realized 0.0318, `rehearsal.json`), and it failed on 17% of seeds even at 60 tasks. It is replaced —
  after that failure, and said so — by a multi-seed check of the estimator (`g1_multiseed.py`, 12 seeds):
  |mean error| ≤ 0.005 and RMS error ≤ 0.02 on both pools. Result at 30 tasks: flat mean −0.0003, RMS
  0.0151; wide mean −0.0001, RMS 0.0081 → **pass** (60 tasks: flat RMS 0.0102). The K\*(T = 200) check
  still passes at 30 tasks (estimated 24 vs true 32, one grid step). Consequence stated in advance: a
  flat pool's spread is resolved to about ±0.015; flat and wide pools remain clearly distinguishable.
- **G2:** cost limit $0.15 per episode (the $0.05 limit encoded the mistaken budget); pilot budget cap
  $50 cumulative (includes the $20.21 already spent). Other G2 checks unchanged, anchor band over n = 30.
- **Budget:** estimated total ≈ $345; the hard stop stays $260 and is raised by Sanjay when reached.
- Everything else in Pre-registration 9 — pools, estimator, cells, policies, flatness guard, contrasts,
  MEI, B = 200 — is unchanged.

## Pre-registration 10 — where prompt heterogeneity comes from (registered 2026-09-29, before any paid episode)

Pre-registration 9 found both real Gmail pools flat, so every contrast was uninformative (§13). This
registers the test of the claim that **prompt heterogeneity is a prerequisite for adaptive prompt search**:
cells that differ in environment (Gmail, GitLab plan-and-track) and prompt content (stylistic grid prompts G,
manual-derived knowledge prompts K), anchors that check the instrument, and a calibration curve that places
every pool. Design: `docs/superpowers/specs/2026-09-28-prompt-heterogeneity-design.md` (as amended before
registration, last amendment `955c89c`); plan `docs/superpowers/plans/2026-09-28-prompt-heterogeneity.md`.
Code frozen at `ce6c376`.

### Hypotheses

- **H1 (environment):** τ_main(GitLab, G) − τ_main(Gmail, G) > 0.
- **H2 (content):** τ_main(env, K) − τ_main(env, G) > 0, per env.
- **H3 (thesis):** in flat cells all policies tie; in meaningful cells the simulation's ordering reappears.
- **H4 (portability, reported):** G prompts' main effects in Gmail predict those in GitLab (Spearman ρ).

### Design

- **Agent (identical to Pre-reg 9):** `gpt-5.4-mini`, low effort, 30 steps, no vision, browser-use 0.13.1,
  webarena-infinity `b3015d79` (working tree dirty in `uv.lock` only).
- **Cells and episodes (8,070 new):**

  | cell | pool | prompts × tasks | wall clock | episodes |
  |---|---|---|---|---|
  | Gmail × G (`GMG`) | Pre-reg 9's G, reused, relabelled `GMG_G_xx` | 50 × 30 | 180 s | 0 new |
  | Gmail × K (`GMK`) | 40 knowledge prompts | 40 × 30 | 180 s | 1,200 + 120 replicates |
  | GitLab × G (`GLG`) | the same 50 G prompts, byte-identical | 50 × 60 | 600 s | 3,000 + 300 replicates |
  | GitLab × K (`GLK`) | 40 knowledge prompts | 40 × 60 | 600 s | 2,400 + 240 replicates |
  | GitLab anchors | `baseline`, `explorer`, `gitlab_oracle_operator` | 3 × 60 | 600 s | 180 |
  | Gmail anchor | `baseline` (drift) | 1 × 30 | 180 s | 30 |
  | Timeout bridge (`GMB`) | 20 of the G prompts in Gmail | 20 × 30 | 600 s | 600 |

- **Tasks:** GitLab 60 of 140 (20 easy / 20 medium / 20 hard, seed 20260928) in two stratified blocks of 30;
  block A = task_e1, e2, e3, e5, e7, e9, e11, e13, e14, e19, task_m2, m4, m5, m7, m8, m11, m12, m16, m18, m19,
  task_h8, h12, h18, h19, h26, h72, h77, h84, h89, h100 (block B in the manifest). Gmail: Pre-reg 9's 30-task
  subset (Amendment 1).
- **K pools:** written by `claude-opus-4-7` from frozen manual-only bundles (GitLab 23,343 words over labels,
  boards, milestones, issues, epics, iterations; Gmail 24,999 words over settings, organize, compose), never
  from task text; 40 kept per app, **0 leak-guard rejections** (8-token overlap with any task instruction,
  or a task entity). Disclosed after generation, before any episode: `GMK_26` shares a 6-token phrase with
  Gmail task "Change button labels to show text instead of icons" — it quotes the manual's Button labels
  procedure. It passes the registered guard and is kept (knowledge content is the treatment); §14 reports
  the Gmail K analyses with and without that task as a sensitivity.
- **Frozen files** (`data/heterogeneity/manifest.json`, sha256 `96c51fef470bfa5190786e2c18b01d0308feffcf53fdca151fb3d18df0b5e946`):
  `pools/GLG.yaml` `f463fc5c…ec00f`, `pools/GLK.yaml` `aa0bd7d8…e75e145`, `pools/GMK.yaml` `5ec8d439…c7a1`,
  `pools/GMB.yaml` `184600cb…6d79`, `pools/anchors_gitlab.yaml` `3eed1138…864df`,
  `pools/anchors_gmail.yaml` `9c8a214d…6465d`, `bundles/gitlab.md` `98dc4b9f…8e70e`,
  `bundles/gmail.md` `dacadc05…1ce3`, `k_generation_gitlab.json` `7297a92a…2a35`,
  `k_generation_gmail.json` `2f437480…a99`, `gitlab/queue.jsonl` `2b87a9fe…aedbb9`,
  `gmail/queue.jsonl` `c36e27c3…f9a6`, `bridge/queue.jsonl` `e354a094…eaf` (full hashes in the manifest;
  every collector start and `replay.py estimate` verifies them). GMG is read from Pre-reg 9's
  sha-checked outcomes snapshot.
- **Queues and staging** (mutually exclusive profiles, shared `logs/heterogeneity/het.lock`): (1) GitLab
  pilot = 5 GLG + 5 GLK + 3 anchors × block A (queue items 0–389); (2) GitLab through block A and its
  replicates (items ≤ 3059, STATUS `through`); (3) Gmail (GMK + anchor, then 120 GMK replicates); (4) the rest
  of GitLab (block B, then its replicates, items ≤ 6119); (5) the bridge. GitLab replicates: 150 GLG + 120 GLK
  from block-A cells after block A, the same from block-B cells after block B.
- **Block-A fallback (mechanical):** if any GitLab arm lacks a terminal replicate-0 record on a block-B task,
  every GitLab cell is analysed on block A; every output records `task_universe`.
- **If the bridge never runs:** the operator creates `logs/heterogeneity/bridge/` with STATUS `budget` and no
  records so `replay.py estimate` can run; H1's primary is then reported **untestable** and the secondary
  (all 50 G prompts, Gmail at 180 s) is reported without a verdict.
- **Scoring:** as Pre-reg 9 — agent timeout = 0; infrastructure error retried twice then `missing`; every
  record carries `ended_by` (clock / steps / agent) and the prompt's sha256 (asserted against the pool).

### Statistics

- **Variance decomposition (§6.1):** balanced crossed MoM per cell; τ_main = √((MS_prompt − MS_resid·c)/J)
  with the Graybill–Wang MLS interval; τ_set = √((MS_prompt − c·σ̂²_e)/J), σ̂²_e from the cell's replicate
  pairs (GMB borrows GMG's), with its MLS interval; ≤ 5% missing cells filled additively (df corrected),
  more is an error.
- **Split-half (§6.2):** difficulty-stratified halves, Spearman–Brown r, 10,000 label permutations.
- **H1/H2 intervals:** prompt-only bootstrap (tasks fixed), B = 2,000, percentile. H1 primary pairs the 20
  bridge prompts (GLG vs GMB at 600 s, resampled jointly); secondary all 50 (GLG vs GMG at 180 s). H2
  resamples each pool independently. **Supported** iff the one-sided 95% lower bound > 0; **refuted** iff the
  upper bound < 0.01; else inconclusive. If both hold, "supported" with `effect_below_threshold`.
- **H4:** Spearman ρ of unshrunk G main effects, bootstrap CI; reported only.
- **Pool shape (§6.4):** NPMLE per cell under x̄ᵢ ~ N(μᵢ, σ̂²_e/nᵢ) → replay reservoir; upper-tail mass
  (share ≥ median + 0.10) with a prompt-bootstrap CI (B = 200, NPMLE re-fit, execution noise held at the
  cell's estimate). Replay: cap = T, M = 1,000, CRN seeds from 10,000,000,000; T ∈ {50, 100, 200} primary,
  {500, 1000} secondary; regret range over `DEFAULT_K_GRID`; its prompt-bootstrap CI at every primary T
  (B = 200, NPMLE re-estimated, M = 250 — the M = 250 range is biased upward by ≈ 0.001).
- **Classification (§6.5), per cell:**
  - **flat** — regret range < 0.005 at T = 50, 100 and 200 **and** its T = 200 bootstrap upper bound < 0.01;
  - **meaningful** — regret range ≥ 0.01 at ≥ 2 of 3 primary T **and** its T = 200 bootstrap lower bound > 0.005;
  - **moderate** — otherwise.
  - **τ_flat = 0.019166** (`calibrate.tau_flat_from_calibration`: the level-0.6 Beta pool whose regret range at
    T = 200 is 0.005; `calibration.csv` sha256 `48b06726c6a1d566347c74602900bc571414dbf1cb2cfe27a26f0df80758c7d5`).
    Reported beside every class with `tau_bound_below_flat`; **never decisive** (amended: at J = 30–60 a τ = 0
    pool clears it only 9–21% of the time). The Beta calibration curve sits above NPMLE-shaped pools
    (at spread 0.035: Beta 0.0147 vs Pre-reg 9's NPMLE G 0.009) — it is a reference, not a threshold.
- **H3 (§6.6):** in every meaningful cell, prompt-bootstrap contrasts (MEI 0.002):
  `het_scale` = `p3_star` − `fixed_K8` (supported iff upper bound < −MEI), `het_spread` =
  `always_search` − `p3_star` (supported iff lower bound > +MEI), and Pre-reg 9's three for continuity
  (`het_primary`, `het_level`, `het_phi`). **Supported** iff every flat cell has range < 0.005 at every primary T
  and ≥ 1 meaningful cell has `het_scale` and `het_spread` both supported. **Refuted** iff a flat cell shows a
  policy difference > MEI with an interval excluding 0, or a meaningful cell shows either contrast reversed
  beyond MEI — refutation outranks untestable, and only per-cell rows decide (pooled rows are
  supplementary). **Untestable** iff no cell is meaningful.
- **Anchor recovery (§8):** recovered iff `GL_anchor_oracle`'s rate > the 90th percentile of GLG ∪ GLK
  per-prompt rates and `GL_anchor_explorer`'s < the 10th, on the analysis universe. **Not recovered → no
  heterogeneity claim is made.** Gmail baseline drift vs Pre-reg 9 reported.
- **Sensitivities (reported, no verdicts):** discriminating tasks (across-prompt rate in [0.2, 0.8]);
  timeout sensitivity (clock-ended episodes as missing, a prompt that always timed out dropped and counted);
  per-prompt timeout rates; REML / GLMM.
- **Seeds:** verdict bootstraps from 60,000,000,000; calibration from 50,000,000,000 — all disjoint from
  Pre-reg 9. A non-registered `--seed` or `--n-boot` stamps every output `unregistered`.

### Stage 0 (free, run before registration)

| data | τ_main [95%] | τ_set [95%] | note |
|---|---|---|---|
| Gmail G (Pre-reg 9) | 0.018 [0, 0.044] | 0.034 [0.012, 0.052] | matches §13's 0.035 |
| Gmail F (Pre-reg 9) | 0 | 0 | flat |
| old GitLab, 18 arms | 0.125 [0.086, 0.199] | — (no replicates) | |
| old GitLab without oracle and explorer | 0.057 [0.016, 0.111] | — | spec expected ≈ 0.02; inside the interval |

### Gates

- **G2 pilot (GitLab):** missing ≤ 5% counting never-attempted items; 0 watchdog relaunches; all 8 workers
  productive; cost ≤ $0.25 per episode; anchor ordering oracle > explorer on the pilot tasks. Only anchor and
  aggregate cost/infra metrics are computed during the pilot.
- **G3 per profile:** missing ≤ 5% per cell counting never-attempted items; `provider_down` breaker as Pre-reg 9.
- **Budget:** unknown per-episode cost (≈ $810 at $0.10). Hard stop per stage, set by Sanjay; STATUS `budget`
  resumes after a top-up.

### Run

```
scripts/run_empirical_pool.sh --profile gitlab --pilot --budget 60
.venv/bin/python experiments/growing_bandits/empirical/gates.py pilot --profile gitlab
scripts/run_empirical_pool.sh --profile gitlab --budget <B2> --through-index 3059
scripts/run_empirical_pool.sh --profile gmail  --budget <B3>
scripts/run_empirical_pool.sh --profile gitlab --budget <B4>
scripts/run_empirical_pool.sh --profile bridge --budget <B5>
.venv/bin/python experiments/growing_bandits/empirical/gates.py collection --profile gitlab   # and gmail, bridge
.venv/bin/python experiments/growing_bandits/empirical/replay.py estimate --study het
.venv/bin/python experiments/growing_bandits/empirical/replay.py point --study het --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid --study het --workers 12
.venv/bin/python experiments/growing_bandits/empirical/describe.py --study het
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot --study het --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot_kgrid --study het --workers 12
.venv/bin/python experiments/growing_bandits/empirical/het_verdicts.py --tau-flat 0.019165515325428883
.venv/bin/python -c "import sys; sys.path.insert(0, 'experiments/growing_bandits/empirical'); import figures; from pathlib import Path; figures.make_figures(Path('results/growing_bandits/heterogeneity'), Path('results/growing_bandits/heterogeneity/figures'), tau_flat=0.019165515325428883)"
```

### Pre-registration 10 — Amendment 1 (2026-09-29, after a paused pilot, before any analysis)

**Why.** The GitLab pilot was paused at 176 of 390 episodes ($21.26, 0 watchdog relaunches, 0 non-ok
records, $0.12 per episode) because the registered agent step cap has never been enforced.
`src/cold_start/tasks/webarena.py:156` passes `max_steps` to the `browser_use.Agent(...)` constructor,
which accepts arbitrary keyword arguments and ignores that one; the cap belongs to `Agent.run(max_steps=)`,
whose default in browser-use 0.13.1 is 500. The bug dates from the adapter's first commit (`fde1f56`), so
**Pre-registration 9 ran under the same condition**: its "30 steps" was also never applied. In every
collected dataset the wall clock is the only bound on an episode.

**Evidence, measured before this amendment.** Pre-reg 9 (Gmail, 180 s): 152 of 3,330 episodes (4.6%)
exceeded 30 steps without reaching the clock; 53 of them succeeded. GitLab pilot (600 s): 25 of 168
exceeded 30 steps (14.9%, up to 134); 20 succeeded; the pooled success rate would be ≈ 0.70 instead of
0.815 had they been cut at 30. `ended_by` therefore never reports `steps`. Secondary: 5 pilot episodes
ended 10–140 s past the 600 s clock without `timed_out`, because `asyncio.wait_for` cannot cancel a
coroutine inside a blocking call; they are ordinary agent-ended episodes and are kept.

**Disclosure.** While diagnosing, the controller saw the pilot's aggregate success rate (0.815 over 168
ok records), success by step bucket (≤ 30 steps vs > 30), cost and clock-ended counts by task tier
(e/m/h), record counts by pool (GLG 60, GLK 53, anchor 38 at that time), and the same aggregates for
Pre-reg 9. No per-prompt rate, per-arm success, τ, reservoir, K-grid, anchor ordering or contrast was
computed. Sanjay chose this amendment over fixing the cap and rerunning.

**What changes.**
- **Agent:** the "30 steps" clause is struck. The registered agent is `gpt-5.4-mini`, low effort, no vision,
  browser-use 0.13.1 with its default `max_steps` (500, never reached in any collected episode), bounded
  only by the profile wall clock (GitLab 600 s, Gmail 180 s, bridge 600 s). This is what Pre-reg 9 actually
  ran, so H1's "same agent, same 50 G prompts" pairing holds on actual rather than nominal conditions.
- **Code:** the adapter is left as is for the whole Pre-reg 10 collection (fixing it mid-study would split
  the data). The fix (`agent.run(max_steps=self.max_steps)`) is filed for after the collection.
- **Pilot records:** the 176 records collected before the pause count in full; the 8 episodes in flight at
  the pause were not written and are re-attempted on resume. The run resumes from STATUS `paused` with the
  same command and budget; gate G2 is unchanged (0 relaunches so far, `relaunches.log` empty).
- **Timeout sensitivity (§ Statistics):** unchanged — clock-ended episodes are the registered sensitivity;
  no step-based sensitivity is added.
- **Write-up:** §13 gains a note that Pre-reg 9's agent had no step cap; §14 states the same for this study.

## Pre-registration 11 — the knowledge pool under the registered per-episode budget (registered 2026-09-30, before any further paid episode)

*Branch `prompt-heterogeneity`. This is the final planned collection leg. It narrows Pre-registration 10 to
one cell and one question, adds a primary outcome scored at the originally registered 30-step budget, and
fixes the heterogeneity gate and the Stage B thesis test before any of the 1,170 new episodes runs.
Results become §15 of `DEPLOYMENT_RESULTS.md`.*

### Question

Does a real, plausible prompt pool exist whose heterogeneity is large enough that candidate search has
decision value, and if so does the qualitative behaviour the simulation predicts reappear there: too little
search leaves value on the table, too much prevents refinement, and a simple budget-scaled recruit-then-refine
schedule captures most of the attainable value?

### Why this cell, and why the budget (audit before registration, aggregates only)

- Under the as-collected regime (600 s wall clock, no step cap — Pre-reg 10 Amendment 1) block A is a
  degenerate search problem: pooled over the 13 pilot arms, 14 of 30 tasks succeed at ≥ 0.95 for every arm
  and 4 sit at ≤ 0.05; only 7 are in [0.2, 0.8]. `GL_anchor_oracle` − `GL_anchor_baseline` is +0.10 on easy
  and medium tasks and 0.00 on hard ones; `GL_anchor_explorer` scores 0.833 where the June run scored 0.20.
  Given unlimited steps the agent brute-forces what it can solve and the prompt stops mattering.
- The June GitLab run (same model, 180 s clock, ≤ 40 steps) had 19 of 40 discriminating tasks and a true
  spread of 0.057 [0.016, 0.111] among its 16 non-anchor hand-written prompts (Stage 0).
- Re-scoring the pilot at the 30-step budget the original Pre-reg 10 text named (exact from the recorded
  `steps`: an episode that succeeded after step 30 is a failure under the budget; the cap only truncates a
  trajectory) gives 11 of 30 discriminating tasks and restores the anchor ordering the instrument check was
  designed for: oracle 0.867 > baseline 0.733 > explorer 0.633. Pool means: GLG 0.633, GLK 0.647
  (uncapped 0.793 / 0.793). Hard tasks take a median of 29 steps and a 90th percentile of 55; the mechanism
  is efficiency — a prompt that knows where things live finishes inside the budget.
- The 30-step value is not tuned: it is the one the original registration named. Counterfactuals at 20 and
  15 steps were computed while auditing (reported here for transparency: 7 and 10 discriminating tasks,
  more floor tasks) and are not used.
- **Disclosure of what was seen before this registration:** per-task pooled success rates over all 13 pilot
  arms (uncapped and at 30 / 20 / 15 steps); anchor rates by tier and overall in both regimes; GLG and GLK
  pool means by tier, timeout counts and mean steps; cost per episode; step quantiles by tier; the 40 GLK
  prompts' lengths (48–107 words) and feature-area coverage counts (1–5 areas each). **No per-prompt
  success rate, spread, reservoir, K-grid or contrast was computed for any pool.**

### Design (nothing new is built; the frozen Pre-reg 10 queue serves it)

- **Cell:** GLK, the 40 manual-derived GitLab procedural prompts (`pools/GLK.yaml`, sha in the manifest), on
  block A (30 tasks, 10 per tier). Anchors are outside the pool and are instrument checks only.
- **Episodes:** the queue's GLK block-A items with `index ≤ 3059`: 1,200 main cells (150 collected in the
  pilot, 1,050 new) and the 120 GLK block-A replicate cells (indices 2792–3059) — **1,170 new episodes**.
  Collected exactly as Pre-reg 10 collects them (uncapped adapter, 600 s), via the collector's `--pools GLK`
  filter with `--through-index 3059`; the run ends with STATUS `through`.
- **Withdrawn from collection:** the remaining GLG block-A cells, block B, its replicates, the Gmail K pool
  and anchor, and the timeout bridge. Pre-reg 10's H1, H2 and H4 are therefore **untestable and reported as
  not collected**; its H3 is tested on this cell alone. The GLG attribution control (45 prompts × block A) is
  permitted later only if GLK is decision-relevant and the write-up needs to attribute the effect to
  procedural knowledge rather than to the budget; it would be registered separately.
- **Budget:** expected ≈ $163 at the pilot's $0.139 per episode; hard stop `--budget 260` cumulative over the
  gitlab log dir (which already holds $54.13). Gate G3: missing ≤ 5% of the 1,320 GLK block-A items.
- **Operator environment:** the Mac stays on AC power with the lid open (the pilot's single watchdog
  relaunch was host sleep); a relaunch caused by host sleep is reported, not counted against G3.

### Primary outcome and task mixes

- **Primary outcome `success_30`** = `success ∧ steps ≤ 30 ∧ ¬timed_out` on every ok record (main and
  replicate), applied at load time by a study variant (`--study glk30`). **Secondary outcome:** the
  as-collected success (`--study glk`). Every table is produced for both; classification, gate and Stage B
  are read on the primary. Caveat disclosed: browser-use tells an agent when it is on its last permitted
  step, so a live 30-step run could differ marginally from re-scoring; the re-scoring is the registered
  quantity.
- **Primary task mix:** block A, equal weights (the registered stratified block). **Secondary S1:** the
  bank-weighted mix, tiers weighted 20 : 20 : 100 as in the 140-task bank (prompt score = Σ_j w_j y_ij,
  effective n_i = 1 / Σ_j w_j² for the noise term). **Secondary S2:** the discriminating subset — tasks whose
  GLK-pooled `success_30` rate lies in [0.2, 0.8] (spec §6.7). S2 may move a verdict from flat to moderate
  only, never to decision-relevant.
- **Noise:** GLK's own 120 block-A replicate pairs, scored the same way. **Estimation:** Pre-reg 10 §6.1–6.4
  unchanged (ANOVA τ_main / τ_set with MLS intervals, split-half r_SB, NPMLE reservoir, K-grid replay at
  T ∈ {50, 100, 200} with cap = T and M = 1,000, prompt bootstrap B = 200 for the regret-range interval).

### Classification (Pre-reg 10 §6.5 kept; one tier added)

| tier | rule |
|---|---|
| flat | regret range < 0.005 at all three primary T and `rr_hi_T200` < 0.01 |
| meaningful | regret range ≥ 0.01 at ≥ 2 of 3 primary T and `rr_lo_T200` > 0.005 |
| moderate | otherwise |
| **decision-relevant** (sub-tier of meaningful) | regret range at T = 200 ≥ 0.02 **and** `rr_lo_T200` > 0.01 **and** gap(`fixed_K8`) ≥ 0.005 **and** gap(`always_search`) ≥ 0.01 (gaps to the pool's own K-grid minimum at T = 200 on the NPMLE reservoir) **and** the regret range at T = 200 with the top and bottom prompt removed ≥ 0.015 **and** split-half r_SB > 0 at p < 0.05 |

Calibration equivalents (Stage 0, T = 200): flat ≈ true sd < 0.02 (τ_flat 0.0192); decision-relevant ≈ sd
≥ 0.07 on Beta pools (fixed-K8 gap crosses 0.005 between sd 0.05 and 0.075) or ≥ 0.03 on rare-great-arm
mixtures. "Statistically detectable" (τ_set MLS lower bound > 0, or r_SB > 0 at p < 0.05) is reported
beside the tier and decides nothing.

### Gate (Stage A → Stage B)

Stage B's verdict is **read** iff GLK is decision-relevant on the primary mix, or moderate on the primary mix
and decision-relevant on S1. Otherwise the leg ends with the classification and Stage B is run for the
record only (as Pre-reg 9's flatness guard). The gate governs interpretation and any further collection;
Stage B itself costs nothing.

### Stage B — the thesis test (replay on the NPMLE reservoir; fixed before data)

Policies: `fixed_K4`, `fixed_K8`, `p3_star` (the corpus-tuned cap-T constants, unchanged), `always_search`,
and the pool's K-grid minimum per horizon as the ceiling. No learned policy. A live online run is impossible
for `always_search` on a 40-prompt pool (it needs T distinct prompts), so replay is the test, disclosed.
Contrasts at each T ∈ {50, 100, 200}, prompt-bootstrap intervals (B = 200), MEI = 0.002:

- **C1 (too little search loses):** `het_scale` — `p3_star` − `fixed_K8` < −MEI, one-sided lower bound rule.
- **C2 (too much search loses):** `het_spread` — `always_search` − `p3_star` > +MEI.
- **C3 (the schedule captures most of the value):** captured = (max regret over the K grid − regret of
  `p3_star`) / (max regret − regret at the K-grid minimum); holds iff the point estimate ≥ 0.7 and its
  bootstrap 5th percentile ≥ 0.5, or `p3_star`'s gap to the minimum ≤ 0.005.

**Supported:** C1, C2 and C3 all hold at ≥ 2 of the 3 horizons. **Partially supported:** C2 and C3 hold but C1
fails (spread with a thin tail: a small fixed K suffices); or C1 and C2 hold but C3 fails (the corpus-tuned
constants do not transfer). **Contradicted:** C1 or C2 reversed beyond MEI (interval excluding 0 on the wrong
side) at any primary horizon, or captured < 0.5 at ≥ 2 horizons. A model-free secondary at T = 40: replay on
the recorded outcome matrix, prompts drawn without replacement, a random task per pull (`fixed_K4`,
`fixed_K8`, search-all-40-once), reported beside C1.

### Fake-positive diagnostics (all pre-registered; each reported beside the tier)

1. Anchors excluded from the pool; the tier is recomputed with the top and bottom prompt removed and with the
   top four removed (decision-relevant requires the drop-2 range ≥ 0.015, above).
2. Leave-one-prompt-out τ_set²: no prompt may move it by more than 30%.
3. Leave-one-task-out τ_set²: no task may carry more than 30%; split-half r_SB > 0 at p < 0.05 is required
   for decision-relevant (prompt differences must generalise across tasks).
4. Length: Spearman(words, rate) reported; τ_set after regressing prompt rate on length must be ≥ 0.8 × τ_set.
5. Timeouts and steps: per-prompt clock-ended share, mean steps and mean recorded errors, each with its
   Spearman against the rate; τ_set in both regimes side by side (efficiency vs capability).
6. Leak guard: already applied at pool freeze (0 rejections); the 6-gram overlap scan is re-run and reported.
7. Noise: 120 replicate pairs; v̂ with its interval; raw and deconvolved spreads both reported.

### What each outcome means (fixed now)

- **A, decision-relevant and Stage B supported:** the paper gains its positive real cell; the calibration
  curve gets a real point in the regime where search matters; the claim becomes that prompts vary when they
  carry procedural knowledge and the agent works under a per-episode budget, and there the schedule captures
  a stated share of the attainable value while small K and always-search each lose a stated amount.
- **B, decision-relevant but Stage B partial or contradicted:** C1 failing challenges "K should grow with the
  budget" on real pools; C3 failing challenges "a schedule tuned once generalises" (§12.8) and the paper must
  say real pools need per-domain constants; C2 failing most likely indicates a misestimated reservoir, which
  the T = 40 matrix replay exposes.
- **C, flat or moderate under the budget:** stylistic, free-form and manual-derived procedural prompts are
  all below decision relevance under this agent even when budgeted; the paper's conclusion becomes that the
  bottleneck in adaptive prompt search is the candidate reservoir's behavioural diversity, not the search
  algorithm, with the June 0.057 spread reported as the one hint that a more constrained agent shows more
  prompt sensitivity (future work).

### Stop rule

This leg is finished when GLK on block A has been classified on the primary outcome and Stage B has been
replayed and written up (§15), whatever the outcome. Block B, the bridge, the Gmail K pool and any new pool
are not run. The GLG attribution control is the single permitted addition, under the condition above and
its own registration. Any other proposal must name the sentence of the paper that would read differently
depending on its result, or it is future work.

### Run, in order

```bash
scripts/run_empirical_pool.sh --profile gitlab --budget 260 --through-index 3059 --pools GLK
.venv/bin/python experiments/growing_bandits/empirical/gates.py collection --profile gitlab --pools GLK --through-index 3059
.venv/bin/python experiments/growing_bandits/empirical/replay.py estimate --study glk30   # and --study glk
.venv/bin/python experiments/growing_bandits/empirical/replay.py point   --study glk30 --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py kgrid   --study glk30 --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot    --study glk30 --workers 12
.venv/bin/python experiments/growing_bandits/empirical/replay.py boot_kgrid --study glk30 --workers 12
.venv/bin/python experiments/growing_bandits/empirical/describe.py --study glk30
.venv/bin/python experiments/growing_bandits/empirical/glk_gate.py --study glk30 --tau-flat 0.019165515325428883
```

### Pre-registration 11 — Amendment 1 (2026-09-30, after code review, before any Pre-reg 11 episode)

An independent review of the implementation found places where the registered text was ambiguous or the code
did not implement it. These rulings are made **before any of the 1,170 episodes runs and before any analysis
of GLK data**; the only data seen remains the audit aggregates disclosed above.

- **S1's split-half.** S1 re-weights the same 40 prompts on the same 30 tasks, so its split-half r_SB and p are
  the primary mix's. (As first written, S1 carried no split-half and so could never be decision-relevant,
  which made the gate's second branch unreachable.) The ANOVA spread estimates are unweighted and are reported
  as not defined for S1.
- **Diagnostic 1.** "Top four removed" means the four highest-rate prompts (not two top and two bottom). On the
  primary mix the tier is recomputed in full on each drop variant — regret range at T = 50 / 100 / 200, its
  T = 200 bootstrap interval, the two gaps and the split-half on the remaining prompts — and reported
  (`drop2_tier`, `drop4_tier`). The decision-relevance condition is unchanged: the drop-2 range at T = 200 ≥ 0.015.
- **Noise per mix.** A mix's within-cell variance is v_mix = Σ w_t² s_p / Σ w_t² over the replicate pairs on
  tasks the mix weights (s_p = (x₀ − x₁)² / 2). Equal weights on all 30 tasks reproduce the registered v; S2
  uses only its discriminating tasks' pairs, whose noise is higher than the floor and ceiling tasks'. Using the
  all-task v would have understated S2's noise and biased it toward heterogeneity.
- **S2's cap.** S2 may move a verdict from flat to moderate only, so both its class and its tier are capped at
  moderate.
- **Stage B precedence and the gap branch.** A horizon at which C3 holds, including through its small-gap branch,
  never counts toward "captured < 0.5". "Reversed beyond MEI" is Pre-reg 10's rule: for C1 the bootstrap lower
  bound above +MEI, for C2 the upper bound below −MEI. Contradicted takes precedence over supported. When no
  registered pattern holds at two horizons the verdict is `inconclusive`.
- **Gaps on S1 / S2.** The always-search gap is fixed K = T on the K-grid, the same policy as `always_search` at
  cap T. The fixed-K8 gap is K = 8. The primary mix keeps the policy episodes.
- **Matrix replay (T = 40).** A prompt's rate is its replicate-0 rate. A pull draws a task among the prompt's
  observed cells, so a missing cell is never scored as a failure. Refinement is round-robin with a Beta(1, 1)
  pick, not the harness's rule; this is disclosed beside C1.
- **Reported, not decided.** The Stage B table carries each of `fixed_K4`, `fixed_K8`, `p3_star` and
  `always_search`'s gap to the K-grid minimum per T. The leave-one-out diagnostics use τ_set without the
  imputed-cell correction (≤ 5% imputation). Mean recorded errors are capped at 3 per episode by the collector.
- **Stale inputs raise.** The gate refuses replay tables whose reservoir or manifest sha differs from the frozen
  manifest, a bootstrap K-grid missing any replicate at any T, and any missing bootstrap episode file.
- **Collection guard.** A staged launch records its filter in `logs/heterogeneity/gitlab/STAGE.json`. Any later
  launch into that log dir with a different `--pools` / `--through-index` / `--pilot`, or none, refuses before
  touching STATUS or any lock, and the watchdog does not relaunch it. A hand relaunch after `provider_down`
  therefore cannot spend the budget on withdrawn GLG cells.

## Pre-registration 12 — linguistic distance versus behavioural distance (registered 2026-10-05, before any embedding is computed)

*Branch `paper`. An analysis of data that are already collected: no new agent episode is run. The behavioural
results (Pre-reg 9 §13, Pre-reg 11 §15) were known when this was written; the linguistic distances were not.
Results become §16 of `DEPLOYMENT_RESULTS.md` and support claim S4 ("linguistic diversity is not behavioural
diversity") in the paper.*

### Question

Did the prompts in each real pool actually differ in wording and meaning, how much, and does linguistic distance
between two prompts predict behavioural distance between them, beyond what execution noise explains?

### Data

- Pools: G (50, Gmail), F (50, Gmail), GLK (40, GitLab; primary outcome = the 30-step re-score `glk30`, the
  uncapped `glk` as secondary). Anchors are reported as reference points only, never pooled.
- Text: the `prompt_guidance` field (the part that varies; the template is shared). The full rendered prompt is a
  sensitivity check.
- Outcomes: the episode-level outcomes already behind §13 and §15, including their replicate cells. Nothing is
  re-collected.

### Measures

- **Linguistic distance (primary):** cosine distance between OpenAI `text-embedding-3-large` embeddings.
  Secondary: `text-embedding-3-small`, TF-IDF cosine distance, token Jaccard distance. Embeddings are cached
  with model name and text SHA-256. Expected cost: under $0.05.
- **Behavioural distance, two kinds:**
  - *Main effect:* |rate_i − rate_j| over the shared tasks (raw), set against the deconvolved spread already in
    §13 and §15.
  - *Profile:* the per-task disagreement rate between prompts i and j. Its noise floor is the disagreement rate
    between two replicate episodes of the *same* prompt on the same task, taken from the replicate cells.
    Excess disagreement = profile disagreement − noise floor.
- **Linguistic dispersion of a pool:** mean pairwise linguistic distance, with a bootstrap 95% interval over
  prompts. Between-pool and cross-domain (Gmail vs GitLab) distances are reported as reference scales.

### Registered tests

1. **L1 (dispersion ordering).** F's mean pairwise linguistic distance (primary embedding) is ≥ G's. This is a
   descriptive check with a bootstrap interval for the difference.
2. **L2 (no linguistic → behavioural association).** In each pool, a Mantel test (Spearman, 10,000 permutations,
   one-sided for positive association) between the linguistic and behavioural distance matrices, run once for
   main-effect distance and once for excess profile disagreement. The correlation is reported with a bootstrap
   95% interval over prompts.
3. **L3 (excess over noise).** Per pool, mean excess profile disagreement with a bootstrap 95% interval over
   prompts and tasks.

### Decision rules for the paper

- **The S4 sentence stays as written** ("prompts written to differ behave like one"; "linguistic diversity did
  not produce behavioural diversity") if both hold:
  - L1 holds; and
  - in F, the L2 main-effect Mantel test is not significant at 0.05 and the upper end of its correlation interval
    is < 0.3.
- **If L1 fails** (F is not more linguistically dispersed than G), the paper drops "written to differ" and
  reports F's dispersion relative to G and GLK.
- **If L2 is significant and positive in any pool**, the paper reports it in that pool and scopes S4 to the pools
  where it holds.
- **L3** is descriptive. A positive excess with a near-zero main effect is reported as prompt × task interaction,
  consistent with the §13 and §15 variance components. It is not reported as prompt quality.

### Disclosed

- The behavioural results were known before registration. Only the linguistic side is blind.
- The choice of embedding model is a researcher degree of freedom. Secondary measures are reported in full.

### Pre-registration 12 — Amendment 1: power and trajectories (registered 2026-10-05, after L1–L3, before either check is run)

*Added after a validity review of the L1–L3 results. Main problems found:*
- *L2 is close to uninformative when outcomes are flat.*
- *Only 7 (G) and 9 (F) of the 30 Gmail tasks discriminate.*
- *The NPMLE spread for F has a degenerate interval.*
- *"Behavioural" was measured only as pass/fail.*

*Two checks address these. Neither runs a new episode or calls a paid API.*

**P — minimum detectable spread.**
- **Simulation.** For each pool (G, F, GLK30), simulate pools whose true prompt spread is known:
  - Spread τ ∈ {0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10} on the probability scale.
  - Each pool keeps its own number of prompts, tasks and per-task difficulty (fitted on the logit scale from the
    data), its estimated prompt × task interaction, its replicate design, and its noise level.
  - Prompt effects are drawn on the logit scale and scaled so the implied spread on the probability scale equals τ.
  - 200 simulated pools per τ.
- **Estimation.** Each simulated pool goes through the repo's own estimators, unchanged: the NPMLE deconvolved
  spread, the τ interval, the split-half test, and the "detectable" rule Pre-reg 11 used.
- **Reported:** power against τ; the minimum detectable spread (MDS), defined as the smallest τ with power ≥ 0.8;
  and the distribution of the NPMLE spread estimate at each τ, in particular P(estimate ≤ 0.005 | τ).
- **Paper wording.**
  - Point estimates such as "SD 0.001" are replaced by "no detectable spread; a spread of MDS or more would have
    been detected with probability ≥ 0.8".
  - If MDS > 0.05 in a pool, that pool's flatness is reported as uninformative, not as flat.

**T — trajectory distance.**
- **Features.** From the existing episode logs, compute per episode:
  - step count (log scale);
  - the sequence of browser-use action types;
  - the multiset of action types.
- **Distances, in each pool:**
  - (a) Prompt main effect on log steps, tested against the replicate noise (variance components, as in §13/§15,
    plus a permutation test over prompt labels within task).
  - (b) Action distance between two episodes: normalised edit distance between action-type sequences (primary)
    and Jaccard distance between action-type multisets (secondary). Excess = mean distance between different
    prompts on the same task − mean distance between replicates of the same prompt on the same task. Bootstrap
    95% interval over prompts and tasks.
  - (c) A Mantel test between linguistic distance (emb3large) and the per-pair mean action distance, with the
    same settings as L2.
- **Decision rules.**
  - If (b) excess is > 0 with an interval excluding 0 in a pool, the paper says that wording changed *how* the
    agent acted without changing *whether* it succeeded, and the S4 sentence uses "success" rather than
    "behaviour".
  - If (c) is significant and positive, the paper reports that linguistic distance predicts trajectory distance.
  - If (a) and (b) are both null, "behaviour" may be used, scoped to "success and trajectory".

**Limitation, recorded now.** Gmail has no positive control: no anchor that shows prompts *can* move success on this
task bank. This cannot be fixed without paid episodes and is reported as a limitation.

## Pre-registration 13 — does the search's winner beat the default on held-out tasks? (registered 2026-10-10, before any episode)

*Branch `paper`. Confirmatory test of the exploratory finding in `docs/RESEARCH_PATHS_AUDIT_2026-10-09.md` §2:
in every realistic pool, the prompt selected by search did not reliably beat the default prompt on held-out tasks.
The planted answer-key prompt did. This registration is committed before any of its episodes is run. All three
legs use one model (`gpt-5.4-mini`, low reasoning effort), the same agent framework, isolated resets, and the
**30-step cap now actually enforced** (fix `bd3fff5`; it was never enforced before).*

### Leg G (primary): GitLab winner check

- **Arms.**
  - W = `GLK_25`, frozen by a rule applied to old data only: highest GLK30 success on block A, then the tie-breaks
    in the manifest. It had no tie: 0.733 against 0.700 for the next four.
  - D = default `GL_anchor_baseline`.
  - O = planted positive control `GL_anchor_oracle`, which hard-codes task issue IDs.
- **Old block-A rates (30 tasks, 30-step):** W 0.733, D 0.733, O 0.867. The searched winner had **no in-sample lead**
  over the default. This leg therefore tests "the search found nothing better than the default". It does not test
  the winner's curse; Leg M does that.
- **Tasks.** The 110 GitLab tasks the September study never ran (`data/winner_check_gitlab/manifest.json`).
  - **Deviation, signed off by Sanjay 2026-10-10.** The registered "never run by any prompt" rule leaves 0 tasks,
    because the June strong-arm allocation runs covered all 140 tasks.
  - **Selection is unaffected.** No GLK prompt has run on these 110 tasks, and none of them was used to select W.
  - **Exposure in June.** D and O, among other June prompts, ran on them in June under a different regime. No
    selection used those runs.
- **Episodes.** 110 tasks × 3 arms × 2 replicates = 660, in seeded order (seed 20261010). The estimated cost is
  about $92, and the hard stop is `--budget 120`.
- **Primary analysis** (`analyze_winner_check.py`). Each arm's per-task score is the mean of its 2 replicates, and
  d_t = W_t − D_t.
  - **G-primary:** certify "W is not ≥ 10 points better than D" if and only if the one-sided 95% upper bound
    (paired t over tasks) on mean(d) is below 0.10.
  - **G-control, required for G-primary to be informative:** the one-sided 95% lower bound on mean(O − D) is above
    0. If the control fails, the leg is reported as **uninformative**, whatever G-primary says.
- **Expected power.** In simulation, with W truly equal to D, the chance of certifying is about 0.94. That simulation
  ignores prompt × task interaction, so real power may be lower.

### Leg M: Gmail winner's-curse test

- **Arms.** G_14, which was the top G prompt in September at 0.70 against the default's 0.60, an **in-sample lead of
  +10 points**. It is compared with the Gmail default anchor.
- **Tasks.** The 30 Gmail tasks of the 2026-10-09 first-run pilot. None of them was used to select G_14.
  - **G_14 data.** Its 2 pilot runs per task are already collected (uncapped; a frozen file).
  - **New runs.** The default runs 2 × 30 = 60 new episodes (`data/winner_check_gmail/`), estimated at about $5 (hard stop `--budget 8`).
- **Scoring.** Both arms are scored at the 30-step rescore, the same rule as GLK30, because the pilot ran before the
  cap fix.
- **Analysis.** d_t = G14_t − D_t (per-task means of 2 runs each).
  - **"Lead reproduced"** if the one-sided 95% lower bound of mean(d) is at least 0.10.
  - **"Lead shrank below its in-sample size"** if the one-sided 95% upper bound is below 0.10.
  - **Otherwise inconclusive.** With 30 tasks, power is modest; this is disclosed now.
- **Confounds, disclosed now.** The two arms ran one or more days apart. The as-collected (uncapped G_14) analysis is
  reported only as a secondary, labelled confounded.

### Leg N: Gmail noise re-check (drift)

- **Design.** 30 historical G repeat cells, randomly sampled with seed 20261011, each run twice now: 60 episodes (hard stop `--budget 8`),
  about $6 (`data/noise_recheck_gmail/`). In September those cells disagreed in 3 of 30.
- **Purpose.** The pilot's G_14 rerun disagreement was 33%, against 10.3% in September. This leg tests whether
  noise has changed: an exact binomial test of the current disagreement against 0.103, plus a paired per-cell
  comparison.
- **Reading.** It is descriptive only. A raised rate is reported as possible provider drift and goes in the paper's
  limitations.

### Decision rules for the paper

- **G-primary certified and G-control holds:** the claim "in a pre-registered held-out test, the search's best
  prompt was not 10+ points better than the default, while a prompt with a real advantage was detected" moves
  from exploratory to confirmed, for GitLab and this model.
- **Leg M reads "shrank":** the winner's-curse claim is confirmed for Gmail at the +10 scale. **"Reproduced"**
  overturns it, and the paper must say so. **Inconclusive** leaves it exploratory.
- **Every leg is reported in full, whatever the outcome.** No leg is re-run or extended without a new amendment.

### Order and stopping

- **Run order:** Leg N, then Leg M (both on the Gmail server, one worker each, run one after the other to avoid
  load confounds), then Leg G (GitLab, `--workers 4`, run alone).
- **Each leg's `--budget`** is its hard stop. A leg that stops on budget is reported as partial, and its primary
  analysis is not run.

### Pre-registration 13, Amendment 1 — sleep-spanning episodes (registered 2026-10-10, before any Leg N, M or G outcome was examined)

- **What happened.** The MacBook ran on battery and went into clamshell sleep. `pmset` logs a "Clamshell Sleep"
  at 09:55 local time on 2026-10-10, and `caffeinate` does not prevent lid-close sleep. Episodes in flight across
  a sleep fire their clock late.
  - **Leg N:** 6 `ok` records have `wallclock_s` far above the 180 s clock, up to 8,673 s.
  - **Leg M:** one such episode was already caught as an `infra_error` and retried automatically.
  - Only wall-clock times and statuses were inspected to find these. No success, disagreement or arm-level outcome
    was looked at.
- **Rule, for every leg.** An `ok` record with `wallclock_s > timeout_s + 60 s` is an infrastructure artifact.
  - It is moved verbatim to `<log>/archive/worker_<w>.sleep_spanning.jsonl` by
    `experiments/growing_bandits/retrospective/archive_sleep_spanning.py`, and its queue item is re-run.
  - Re-runs come out of the same leg's `--budget`.
  - Archived records are counted and reported, never analysed.
- **Step-cap note, recorded now.** With the cap enforced, browser-use stops at 31 history steps, not 30, because
  its final step is counted too. Nothing ran past 31 in Legs N or M. The registered 30-step rescore (steps ≤ 30)
  scores 31-step episodes as failures in Legs G and M, as registered. Leg N uses raw success.
- **Order change.** Leg G runs next, as the chain already has it. Leg N's re-runs follow Leg G, so that two
  servers never run together. Sleep-spanning records in Leg G are handled the same way after it finishes.

## Pre-registration 14: a positive control that covers every Leg G task (registered 2026-10-10, before any study episode)

*Branch `paper`. Pre-registration 13's Leg G was **uninformative**: its positive control `GL_anchor_oracle` carries
answers for only 20 of the 110 tasks (`docs/growing_bandits/PREREG13_RESULTS.md`). This study re-runs **only the
control**, using an answer key that covers all 110 tasks, against a fresh default arm, on the same 110 tasks. It does
not re-run W and does not change Leg G's registered reading. It adds the missing evidence that this task set and
design could detect a real advantage. Settings are unchanged: `gpt-5.4-mini` at low effort, the same agent framework,
isolated resets, the 30-step cap enforced, 600 s clock.*

### The new control arm (built before registration)

- **`GL_oracle_full`** is generated mechanically by `experiments/growing_bandits/retrospective/build_full_oracle.py`
  (sha256 fe0d44c1…), in three steps:
  1. For every one of the 140 bank tasks, apply the benchmark's own `solve_task_<id>` from `sanity_check_real.py` to
     the seed state.
  2. Diff the result against the seed state.
  3. Write one answer-key line per task: the exact instruction, then the exact item ids and final values.
- **One prompt for both stages.** The same arm, byte-identical pool file (`pools/oraclefull.yaml`, 59,570
  characters, about 14,900 tokens, all 140 entries), is used in the pilot and in the study. No solver failed and no
  diff was empty.
- **Pilot, on the 30 block-A tasks only** (disjoint from the 110; not study data). One episode per task, gate
  ≥ 0.80 at the 30-step rescore. The old oracle scored 0.867 on these tasks and the default 0.733.
  - **v1** (the old oracle's header "Save once, then finish. Do not repeatedly verify", and its vector):
    **0.633, fails** ($1.91; `data/oracle_full_pilot_v1`, `logs/oracle_full_pilot_v1`). The agent ignored the
    key: it searched for items itself and finished without checking.
  - **v2** (`HEADER_V2`: look up the exact line, use the listed ids, reopen and check every value before finishing;
    `VECTOR_V2`: agency 0, verification 2): **0.800 = 24/30, passes the gate exactly** ($2.33).
  - v2 is frozen here. It was revised once, on block-A tasks only. No study task has been run with any O2 version.
  - **Disclosed now:** v2 sits at the gate, and on block A it is about +7 points over the old default rate, not the
    old oracle's +13. On the harder study tasks (the Leg G default is 0.40) there is more room, but the margin is
    not guaranteed.

### Design

- **Arms:**
  - O2 = `GL_oracle_full`.
  - Dnew = `GL_anchor_baseline`, whose prompt sha must match Leg G's.
- **Tasks:** the 110 Leg G tasks (`data/winner_check_gitlab/manifest.json`).
- **Episodes:** 110 tasks × 2 arms × 2 replicates = 440, in seeded order (seed 20261012), with arms shuffled within
  each task (`data/control_check_gitlab/`).
  - **Cost.** The prepare script's conservative estimate is $55, at $0.15 per O2 episode; the pilot measured $0.078.
    Expected spend is about $40–45. The hard stop is `--budget 60`, which is below the script's recommended $75. A
    run that stops on budget is reported as partial, and its primary analysis is not run.
  - The run uses `--workers 4` on the GitLab profile, alone on the machine, with the Mac on AC power and the lid open.
- **Amendment 1 of Pre-registration 13 applies:** sleep-spanning `ok` records are archived and re-run.

### Analysis (`analyze_control_check.py`, sha256 993928c9…, frozen at registration)

- **Scoring:** every record is scored at the 30-step rescore. An arm's per-task score is the mean of its 2
  replicates.
- **PRIMARY:** the one-sided 95% lower bound (paired t over 110 tasks) of mean(O2 − Dnew). **The control holds iff this
  lower bound is > 0.**
- **Combined reading with Leg G**, which was registered here before any O2 study episode:
  - **Control holds, and Leg G's registered upper bound (+0.034) is < 0.10:** the paper may state "on 110 held-out
    GitLab tasks, the search's winner was not ≥ 10 points better than the default (Pre-reg 13), and on the same tasks
    a prompt with a real advantage was detected (Pre-reg 14, next day)." The paper must disclose the two-session
    design.
  - **Control fails:** Leg G stays uninformative and the claim stays exploratory. No further control run without a
    new registration.
- **SECONDARY, reported whatever the outcome:**
  1. A task-cluster bootstrap of the primary.
  2. **Day-to-day drift:** Dnew − Dold, where Dold is Leg G's default arm, as the mean with a 90% CI. If the CI
     excludes 0, the combined reading carries a drift caveat.
  3. **Exploratory:** O2 − Dnew, split by whether the task was in the old oracle's cookbook.
  4. Per-arm operations.
- **Power (stated now):** with D at about 0.40 and a per-task sd of the difference about 0.31 (Leg G's O − D, se 0.029),
  the lower bound exceeds 0 with probability ≈ 0.9 if the true O2 − D ≥ +0.087, and ≈ 0.5 at +0.049.
  - On its 20 covered tasks the old oracle was +0.10. The new key also covers the harder tasks, where the default sits
    at 0.32, so more room is expected there.
  - A control below about +0.05 would likely fail, and that would be reported as is.

## Pre-registration 15: Gmail answer-key headroom (registered 2026-10-11, before any study episode)

*Branch `paper`. Pre-reg 14 found that a full answer key lifts `gpt-5.4-mini` only +5.5 pp on GitLab: there is little
room for any prompt there. For Gmail it is unknown whether the searches found nothing because there was nothing to
find (small headroom) or because they missed real gains (large headroom). This study measures Gmail's answer-key
headroom H: how much a prompt that contains the exact answer to every task beats the default. Settings are Gmail's
registered ones: profile `gmail`, `gpt-5.4-mini` at low effort, 180 s clock, 30-step cap, isolated resets.*

### Arms (built before registration)

- **O = `GM_oracle_full`.** Generated mechanically by `experiments/growing_bandits/retrospective/build_gmail_oracle.py`
  (sha256 0cd0e244…).
  - For each of the 60 Gmail real tasks, the benchmark's own `solve_task_<id>` (`sanity_check_real.py`) is applied to
    the seed state and diffed against it.
  - Each entry is one answer-key line: the exact instruction, then the exact emails (id, subject, sender), labels,
    filters and settings, with their final values.
  - The header is the GitLab v2 procedure adapted to Gmail's URL routes, with the same vector (`VECTOR_V2`).
  - Size: 60 entries, 17,233 characters, about 4,300 tokens. No solver failed and no diff was empty.
  - It renders through `configs/template_gitlab.jinja`, which is byte-identical to `template.jinja` when the guidance
    is empty, so O differs from D only in guidance and vector.
- **D = `GM_anchor_baseline`**, the Gmail default. Its rendered prompt sha256 is ae8c7c7f…, the same prompt as Pre-reg
  13 Leg M's default.
- **Not piloted.** All 60 Gmail real tasks are study tasks, so there is no disjoint set to pilot on. If the agent
  ignores the key (as GitLab v1 did), H **underestimates** the true answer-key headroom. H is therefore a lower
  bound on what a prompt could add.

### Design

- **Episodes:** O and D × the 60 Gmail real tasks × replicates {0, 1} = **240**. Tasks are in a seeded order (seed
  20261013), and a task's 4 episodes are in a seeded order (`data/headroom_gmail/`, manifest sha256 bf2ccf89…;
  `prepare_headroom_gmail.py` sha256 313e8152…).
- **Run:** `--workers 4`, `--budget 30` (hard stop). Estimate $21.60. The Mac is on AC with the lid open. A run that
  stops on budget is reported as partial, and its primary analysis is not run.
- **Amendment 1 of Pre-reg 13 applies:** sleep-spanning `ok` records are archived and re-run.
- **Collector:** gains `agent_overrides` (a manifest may swap the model) and records `llm_model` on every attempt. This
  manifest has no overrides; the agent is the profile's.

### Analysis (`analyze_headroom_gmail.py`, sha256 41fdc32e…, frozen at registration)

- **Scoring:** the 30-step rescore. An arm's per-task score is the mean of its 2 replicates.
- **PRIMARY:** H = mean over the 60 tasks of O_t − D_t, paired t, two-sided 90% CI.
  - **large** iff the lower bound ≥ 0.10;
  - **small** iff the upper bound < 0.10;
  - **unresolved** otherwise.
  - The 0.10 threshold is the margin used throughout: the search winner's in-sample lead (Leg M) and Leg G's
    certification margin.
- **Readings, fixed now:**
  - **large:** Gmail had at least 10 pp of headroom, and the search's winner captured none of it held out (Leg M,
    0.0 pp). The searches *missed* available gains; the pool, not the ceiling, was the problem. This motivates a
    realistic-pool study (step 2) on Gmail.
  - **small:** Gmail, like GitLab, has under 10 pp of headroom for this agent. "There was little to find" holds on
    both apps.
  - **unresolved:** report the point estimate and interval; no claim.
- **SECONDARY, reported whatever the outcome:**
  1. A task-cluster bootstrap of H (10,000 resamples).
  2. **Capture:** Leg M's held-out G_14 lead (0.000) against H on Leg M's 30 tasks.
  3. **Default drift:** D now minus Leg M's default (same prompt) on those 30 tasks, mean with 90% CI. Disclosed:
     Leg M ran 1 worker, this study 4. Both arms here share the load, so H is unaffected.
  4. Per-arm operations and rerun disagreement.
- **Power (stated now):** the per-task sd of O − D is assumed about 0.35 (GitLab's was 0.30), so the se is about
  0.045 and the CI half-width about 0.075.
  - True H = 0.25: P(large) ≈ 0.95. True H = 0.20: P(large) ≈ 0.7.
  - True H = 0.00: P(small) ≈ 0.7. True H = 0.10: neither reading is likely, and the study would most likely end
    unresolved.

## Pre-registration 16: Gmail answer-key headroom under Claude Haiku 4.5 (registered 2026-10-11, before any study episode)

*Branch `paper`. Pre-reg 15 found Gmail's answer-key headroom "small" for `gpt-5.4-mini` (H −1.7 pp). Exploratory
diagnosis: that agent referred to the key in only 33 of 120 episodes. This study asks whether an agent from another
model family uses the answers, and whether that changes the headroom. Everything is Pre-reg 15's except the model.*

### Changes from Pre-reg 15 (only these)

- **Model:** `claude-haiku-4-5-20251001` (provider anthropic), set through the manifest's `agent_overrides`.
  - Run with browser-use's `use_thinking` **off**. With it on, Haiku inlines its action JSON into the thinking field.
    In smoke runs most steps then fail browser-use's output schema, and no tokens are counted, so the budget guard
    would be blind.
  - With it off, every step was valid in 10 smoke episodes, all on non-study tasks: GitLab block A and Gmail
    function-tasks.
  - The adapter (`src/cold_start/tasks/webarena.py`, sha256 181034ef…) gains a `use_thinking` option. Its default
    (None) leaves every earlier configuration unchanged.
- **Queue seed:** 20261014 (`data/headroom_haiku_gmail/`, manifest sha256 6a63246c…; `prepare_headroom_haiku.py`
  sha256 594ffd5b…).
- **Unchanged:**
  - The same two prompt files, byte-identical to Pre-reg 15's (the test asserts this).
  - The same 60 tasks, 2 replicates, profile `gmail`, 180 s clock and 30-step cap: 240 episodes.
- **Run:** `--workers 4`, `--budget 35` (hard stop), estimate about $22, on AC with the lid open. Amendment 1 of
  Pre-reg 13 applies.
- **Disclosed:** Haiku is slower per step than `gpt-5.4-mini` (about 6 s against 4 s in smoke runs), so the 180 s clock
  binds sooner. It binds equally on both arms.

### Analysis (`analyze_headroom_haiku.py`, sha256 b4b214d4…, frozen at registration)

- **PRIMARY:** exactly Pre-reg 15's rule, computed by its frozen analyzer on this study's logs. H is the mean over the
  60 tasks of O − D, with a two-sided 90% paired-t CI.
  - **large** iff the lower bound ≥ 0.10;
  - **small** iff the upper bound < 0.10;
  - **unresolved** otherwise.
- **Readings, fixed now:**
  - **large:** an agent from another family gains at least 10 pp from the answers. The flat Gmail result is specific to
    `gpt-5.4-mini`'s weak adherence, and a realistic-pool study on Haiku becomes the next step.
  - **small:** the low Gmail ceiling holds across two model families.
  - **unresolved:** no claim.
- **SECONDARY, reported whatever the outcome:**
  1. Pre-reg 15's secondaries, computed by its analyzer.
  2. **Model contrast:** H_haiku − H_mini per task, paired over the 60 tasks, mean with a 90% CI. This is
     cross-session and descriptive: D drifted by +15 pp between sessions in Pre-reg 15.
  3. Arm levels (O, D) under each model.
  4. **Key use** in the answer-key arm under each model:
     - the reference rate (model outputs match `/answer[ -]key|the key\b/i`);
     - the direct-URL rate (they match `/#\/email\/\d|#\/settings\//`);
     - success within each group.
     - Haiku has no thinking field, so the reference rate is less sensitive for it. The direct-URL rate is
       behavioural and comparable across models.
