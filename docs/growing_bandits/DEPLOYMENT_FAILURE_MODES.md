# Growing bandits — deployment failure-mode register

Produced by an independent adversarial review of the code, corpus and planned comparison *before* any
deployment run (2026-09-14). Each row names a way the comparison could be wrong or misleading and the
mitigation the study commits to. The post-hoc review (plan M8) re-checks every row against the produced
tables and code.

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

