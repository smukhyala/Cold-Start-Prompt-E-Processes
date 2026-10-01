# Paper plan — adaptive prompt search over a growing reservoir

*Written 2026-10-01, after collection closed (Pre-registration 11, Outcome C). Source of truth for numbers:
`docs/growing_bandits/RESULTS.md` (offline oracle-label study), `docs/growing_bandits/DEPLOYMENT_RESULTS.md`
§1–§13, §15, and the tables named beside each claim. No claim below rests on an experiment that was not run.*

---

## 1. Claim ledger

Format: **Claim** — evidence — quantitative result — scope / qualification — paper section.

### 1.1 Primary contributions

**P1. The SEARCH-versus-REFINE decision reduces to choosing search breadth K.**
- Evidence: K-matched control (§12.2, `capmatch_contrasts.csv`): each policy against front-loaded search capped at
  the policy's own final K, same episodes. Pilot policy benchmark (`PILOT_FINDINGS.md` §3).
- Result: the best learned policy ties its K-matched control, Δ = +0.0001 [−0.0005, +0.0007] (8 environments).
  The validation-tuned gradual schedule is *worse* than front-loading at the same K, +0.0009 [+0.0005, +0.0012],
  p = 0.0004, worse in 8 of 8 environments. In the pilot benchmark, "fill to cap" won 27 of 36 cells, and the
  other 9 were fourth-decimal ties.
- Scope: synthetic reservoirs, T ≤ 200 for the K-matched control; the control's cap is the policy's mean
  final K.
- Section: §4.

**P2. Optimal breadth is interior and grows sublinearly with budget, and a mis-sized breadth dominates every
policy effect.**
- Evidence: fixed-K regret envelope (§12.1, `k_star_envelope.csv`, `k_star_envelope_all33.csv`). Paired cap
  sweep with every constant re-tuned at each cap (§12.7, `capp_policies.csv`, `capp_crosscap.csv`).
- Result: pooled K\* is 24 / 32 / 48 / 128 / 256 at T = 50 / 100 / 200 / 500 / 1000, a log-log slope of about
  0.8. Per environment, K\* at T = 1000 spans 12–400. Opening a new arm every pull is the one rule that always
  loses: uncapped regret at T = 1000 is 0.346, against 0.077 for the tuned schedule. With breadth capped at 64,
  six policies are identical to four decimals (0.0973 at T = 1000). Raising the cap lowers regret by 0.020 for
  every tuned schedule.
- Scope: 0.020 is a pooled point estimate. Its environment-level interval at n = 8 includes zero above cap 128,
  because environments with K\* near 400 gain about 0.05 and those near 48 gain nothing. K\* depends on the
  tail's exponent *and* mass: environments with the same exponent differ up to 8× (§12.8).
- Section: §4.

**P3. A budget-scaled recruit-then-refine schedule captures nearly all attainable value; a learned policy
trained on 87,148 oracle labels reduces to a three-number rule and adds nothing robust.**
- Evidence: registration ladder, Pre-registrations 2–6 (§12.3–12.6, `h1b_*.csv`). Cap headroom (§12.7). Held-out
  family, uncapped, Pre-registration 8 (§12.8, `capc_*.csv`). Offline-vs-deployed (§6, `surrogate_validity.csv`).
- Result: in-distribution at T ≤ 200, the k = 4 model minus the level rule (recruit to
  T^0.75 · exp(4·(0.5 − level)) arms, then refine) is −0.00002 [−0.0007, +0.0007], p = 0.96, over 30
  environments. With headroom, the best rule's edge over a re-tuned schedule is about 0.001–0.002 and not
  significant (level rule minus schedule −0.0014 [−0.0045, +0.0017]). Uncapped, the learned policies over-recruit
  to 380–527 arms, with regret 0.14–0.15 against the level rule's 0.076. Off-family and uncapped, the
  corpus-tuned schedule is within 0.001 of the best single K (−0.0006 [−0.0016, +0.0004]), while the learned
  policy is 0.035 worse. Offline AUC does not predict deployed regret: Spearman +0.005 [−0.53, +0.52] over 22
  variants.
- Scope: "is a three-number rule" holds *in distribution at T ≤ 200*; off-family the model and the rule diverge.
  The held-out family has 3 environments, so its intervals are paired-episode, not environment-level.
- Section: §5.

**P4. The value of search is set by the reservoir's heterogeneity — specifically its upper tail — not by the
search algorithm.**
- Evidence: Stage-0 calibration simulation (`results/growing_bandits/heterogeneity/calibration.csv`). Beta pools
  at level 0.6 with true spread 0.01–0.15, and flat-bulk-plus-rare-great mixtures, at T ∈ {50, 100, 200},
  cap = T, M = 1,000.
- Result: at T = 200, the regret range over K is 0.0013 / 0.0053 / 0.030 / 0.080 / 0.151 at spread
  0.01 / 0.02 / 0.05 / 0.10 / 0.15. Always-search's loss grows 0.002 → 0.188, while the budget schedule stays
  within 0.013 of the ceiling throughout. At *equal* spread the tail decides: a mixture with spread 0.036 and 1%
  of arms at +0.3 has a range of 0.083, against 0.015 for a Beta pool at spread 0.035.
- Scope: one level (0.6), T ≤ 200, two shape families. The flat threshold (spread ≈ 0.019) is defined on Beta
  pools.
- Section: §6.

**P5. Real prompt reservoirs are flatter than any simulated environment, and search breadth barely matters on
them.**
- Evidence: Pre-registration 9 (§13: Gmail, 50 grid and 50 free-form prompts × 30 tasks, 300 replicate cells).
  Pre-registration 11 (§15: GitLab, 40 manual-derived procedural prompts × 30 tasks, 120 replicate cells,
  30-step budget). Stage 0 (`stage0_gmail.csv`, `stage0_gitlab_paired.csv`).
- Result: deconvolved (NPMLE) spread is grid 0.035, free-form 0.001, procedural 0.041 under the budget and 0.009
  as collected. All lie below the corpus minimum of 0.075. Regret range over K at T = 200 is 0.009 / 0.0001 /
  0.008 / 0.001; 0 of 6 Gmail cells are informative, and GitLab is "moderate" on all three task mixes. No pool
  has *detectable* prompt heterogeneity. Split-half reliability is r_SB 0.07 (p = 0.43) on the grid pool and
  ≤ 0 (p = 0.61) on the procedural pool, and the procedural pool's spread has an MLS lower bound of 0. Task
  difficulty dominates: on the Gmail grid pool, task variance is 0.164 against τ_set² ≈ 0.001. Every real pool
  sits *below* the Beta calibration curve at its own spread (grid 0.009 vs 0.015; procedural 0.008 vs 0.021):
  the real reservoirs have no upper tail.
- Scope: one agent (`gpt-5.4-mini`, low reasoning effort, browser-use 0.13.1); two WebArena-style apps
  (`webarena-infinity` Gmail and GitLab clones); 30 tasks per cell, which resolves spread to about ±0.015–0.02;
  40–50 prompts per pool, which cannot exclude a tail of rare excellent prompts at below about 2% frequency —
  exactly the regime P4 shows would matter most.
- Section: §7.

### 1.2 Supporting findings

| # | Claim | Result | Scope | Section |
|---|---|---|---|---|
| S1 | The SEARCH/REFINE decision is predicted by candidate quality, not by e-process evidence, inside the training support | e-process features move balanced accuracy −0.001 / +0.006, against +0.052 for quality features; best e-process feature AUC 0.535 vs 0.706; deployed H2 +0.0001 [−0.0003, +0.0005] | Out of support they help: −0.002 on the held-out family, −0.032 in uncapped cells (95% of the cap-sweep effect). Do not state unconditionally | §3, App. B |
| S2 | The decision is a reservoir-estimation problem | the oracle tail integral scores AUC 0.836 on the oracle label; its observable estimate scores 0.572, chance at T = 1000 | corpus label, k = 16 | §3 |
| S3 | A single action barely matters; a sustained commitment does | decided share 27.5% → 54.0% → 80.4% at k = 1 / 4 / 16; k = 1 and k = 16 agree in sign 52.2% | oracle labels, cap 64 | App. B |
| S4 | Linguistic diversity does not imply behavioral diversity | 50 deliberately diverse free-form instructions: raw spread 0.041 is no larger than measurement noise 0.054; NPMLE spread 0.001 | Gmail, this agent, ±0.015 resolution | §7 |
| S5 | Raw success-rate spreads roughly double the true spread at 30 tasks | grid raw 0.053 vs 0.035; procedural raw 0.062 vs 0.041 | needs replicate pairs to see | §7 |
| S6 | The one apparent spread under a budget is efficiency, not capability | procedural spread 0.036 at 30 steps vs 0.011 uncapped; mean steps vs rate ρ = −0.40; does not replicate across task halves | GitLab | §7 |
| S7 | Adaptive allocation over 12 fixed hand-written prompts did not beat uniform | Gmail 59.2% vs 56.1% over 3 × 120 executions (2 of 3 replicates); Shopping 40.8% vs 45.4%; global null never rejected | small n, not pre-registered | App. G |

### 1.3 Negative results (stated as results, not failures)

- **N1.** Learned SEARCH/REFINE policies do not beat a tuned schedule once breadth is not artificially capped
  (P3), and are harmful outside their training support.
- **N2.** E-process evidence about the leader is not the signal for opening new candidates (S1).
- **N3.** No real prompt pool reached decision relevance. The pre-registered thesis test on real data was
  therefore *not activated* (Pre-registration 11 gate). The paper must say the positive real-data test was
  designed and pre-registered but its precondition failed.
- **N4.** Offline label accuracy is not a surrogate for deployed performance (P3).

### 1.4 Limitations and future work

- One agent model and configuration. Prompt sensitivity under weaker or more constrained agents is future work;
  the June GitLab run of 16 hand-written prompts, 180 s and ≤ 40 steps, no replicates, had spread 0.057
  [0.016, 0.111], recorded only as a hint.
- Two apps from one benchmark family; tasks are UI workflows with verifiable end states.
- 40–50 prompts per pool cannot detect rare excellent prompts below about 2% frequency (see P5).
- Replay treats the deconvolved distribution as an infinite reservoir; no live online run of the schedules on
  real prompts, because always-search at T = 200 needs 200 distinct prompts.
- Simulation reservoirs are parametric families (Beta, tail, mixtures, 33 environments). There is no DP ceiling;
  the fixed-K envelope stands in.
- Protocol deviations, all disclosed and pre-registered as amendments: the 30-step agent cap was never enforced
  in any collected data (Pre-reg 10 Amendment 1); the GLK primary outcome re-scores at 30 steps; host-sleep
  relaunches; one 52-record window during provider latency, covered by the registered timeout sensitivity.

### 1.5 Statements currently stronger than the evidence (flagged, not yet edited)

| Where | Statement | Problem | Suggested wording |
|---|---|---|---|
| `RESULTS.md`, Finding 1 and bottom line | "the e-process evidence … does not drive the SEARCH-versus-REFINE decision" | unqualified; deployed features help off-support (§7.5) | "…inside the training support; outside it they carry some signal (§7.5)" |
| `DEPLOYMENT_RESULTS.md` §1.0 item 1 | "the *timing* of search is worth nothing" | measured at T ≤ 200 on 8 + 3 environments | "…worth nothing at T ≤ 200 on the corpus" |
| §1.0 item 1, §12.7 | "the cap cost 0.021" stated flat | pooled point estimate; cluster CI includes 0 above cap 128 | add "(pooled; environment-level interval includes zero)" |
| §1.0 item 2, §12.6 | "The learned policy is a three-number rule" | true in distribution at T ≤ 200 only; diverges off-family (§12.8) | "…in distribution at T ≤ 200" |
| §13.2 | "fifty differently worded instructions behave … like one prompt" | resolution about ±0.015 at 30 tasks | "…indistinguishable at the ±0.015 resolution of 30 tasks" |
| §15.6 (bold sentence) | "The practical bottleneck … is a candidate reservoir … not the search algorithm" | interpretation; scope sits in a later sentence; rare-tail limitation missing | carry the scope in the sentence, and add the <2% rare-prompt limitation |
| `README.md` | frames the project as e-process evaluation of cold-start prompts | outdated framing; no false claim | update when the paper is posted |
| Paper brief (2026-10-01) | "E-process evidence contributes essentially no useful signal" | same as row 1 | scope to the training support |
| Paper brief | "Simulations show a clear relationship between heterogeneity and the value of choosing K" | spread alone is not sufficient; tail shape decides (P4) | "…between the reservoir's upper tail and…" |

---

## 2. The story (final narrative)

1. **The problem.** Choosing a system prompt for an LLM agent is an expensive evaluation problem. Each trial is a
   full agent episode with a noisy binary outcome, and the candidate set is not given: a designer can always
   write or generate another prompt. Prompt optimizers, from APE and OPRO to DSPy-style compilers and
   evolutionary methods, implicitly answer one question at every step — test a new candidate, or test an
   existing one again? This paper asks when that question matters at all.

2. **The model.** We formalize prompt search as best-arm identification over a growing reservoir: at each pull,
   SEARCH draws a new prompt from an effectively unlimited distribution, and REFINE spends the pull on a
   candidate already held. Simple regret, the gap between the best prompt seen and the one finally chosen,
   decomposes exactly into a discovery term (did we find a good prompt?) and a selection term (can we tell which
   one it was?). Every policy trades one against the other.

3. **What information should drive the decision.** Our starting hypothesis was that anytime-valid evidence —
   e-processes measuring how sure we are the leader is best — should tell an agent when to stop searching. We
   built an oracle that labels 87,148 simulated states with the regret-optimal action by paired replay. The
   decision turns out to be predicted by how *good* the held candidates are relative to a fresh draw, not by how
   *certain* we are about the leader. Inside the training support, evidence adds nothing measurable. The
   decision is, at heart, an estimate of the reservoir's upper tail.

4. **The decision collapses to breadth.** Holding the final number of candidates fixed, the timing of search is
   worthless: front-loading recruitment ties the best learned policy and beats a gradual schedule. Regret is a
   function of how many prompts are evaluated, K, not of when they are introduced. The rich sequential decision
   reduces to choosing one number.

5. **How breadth should scale.** The regret-minimizing K is interior and grows sublinearly with the budget, from
   about 24 at T = 50 to about 256 at T = 1000. Opening a new prompt every pull, which maximizes discovery,
   fails badly because nothing is measured well enough to be selected. Mis-sizing K dwarfs any difference between
   policies: under a breadth cap of 64, every policy we studied was the same policy.

6. **Simple schedules suffice.** A logistic policy trained on the oracle labels beats every fixed schedule — and
   is reproduced to five decimal places by a three-parameter rule that recruits to T^0.75 · exp(4·(0.5 − level))
   arms and then refines. With room to work, that rule's advantage over a plain budget-scaled schedule is within
   noise, and off the training family the schedule carries over while the learned policy does not. A
   recruit-then-refine schedule with two constants captures nearly all of the attainable value.

7. **The moderator.** How much any of this matters depends on the reservoir. On calibrated synthetic pools the
   value of choosing K well rises steeply with heterogeneity: below a true spread of about 0.02 every reasonable
   K ties, while at 0.10 a bad K costs 0.08 of success rate. At equal spread the upper tail decides: a few rare
   excellent prompts make search worth several times more than a symmetric spread does.

8. **Do real reservoirs vary?** Everything above is conditional on prompts actually differing, so we measured
   that directly, with replicate episodes to separate execution noise from true differences and a nonparametric
   deconvolution of the prompt-rate distribution. We used three deliberately different families: structured
   stylistic prompts and deliberately diverse free-form prompts on a Gmail clone, and procedural prompts derived
   from the product manual on a GitLab clone, scored under a per-episode step budget designed to let knowledge
   show through as efficiency.

9. **They largely do not.** All three reservoirs are flatter than every simulated environment. Free-form prompts
   are indistinguishable from one another at our resolution. Neither stylistic nor procedural prompts show
   heterogeneity that replicates across halves of the task set, and the real pools sit below the calibration
   curve because they lack an upper tail. Task difficulty carries over a hundred times more variance than the
   prompt (0.164 against about 0.001 on the Gmail grid pool). Search breadth barely matters: every sensible rule is within a few thousandths of the best K, and
   only opening a new prompt every episode is measurably worse. Our pre-registered test of the simulated policy
   ordering on a heterogeneous real pool could not run, because no pool met its precondition.

10. **Conclusion.** Adaptive prompt search is a breadth-selection problem whose value is set by the candidate
    reservoir. When candidates differ — specifically, when a few are much better — choosing breadth matters, and
    a simple schedule gets nearly all of it. For the agent and applications we measured, generated prompt pools
    did not differ enough for that choice to matter. The practical bottleneck we observe is generating
    candidates that behave differently, not the algorithm that searches among them. Whether weaker or more
    constrained agents are more prompt-sensitive is an open question we leave to future work.

---

## 3. Contributions

**C1 — Search/refine reduces to breadth selection, with a characterization of optimal breadth.**
- *New:* an oracle-labelled study of the SEARCH/REFINE decision in an infinite-armed best-arm setting, and a
  K-matched control that isolates timing from breadth.
- *Evidence:* P1, P2.
- *Why care:* prompt optimizers can drop per-step search heuristics and tune one quantity; capping the candidate
  set silently decides the outcome.
- *Limitation:* synthetic reservoirs; K-matched control at T ≤ 200; no closed-form optimum.

**C2 — A two-constant recruit-then-refine schedule matches sophisticated adaptive policies.**
- *New:* a pre-registered ladder of null models showing a learned policy reduces to an interpretable
  three-parameter rule, which itself adds little over a budget-scaled schedule once breadth has room.
- *Evidence:* P3, N1, N4.
- *Why care:* a strong, simple baseline for adaptive prompt optimization, and a caution that offline label
  accuracy is not a proxy for deployed regret.
- *Limitation:* one family of learned models (logistic and boosted classifiers on 71 features); held-out family
  has 3 environments.

**C3 — Reservoir heterogeneity, and its upper tail in particular, sets the value of search.**
- *New:* a calibration curve mapping a reservoir's deconvolved spread and tail to the regret at stake in
  choosing K, usable to decide whether prompt search is worth running.
- *Evidence:* P4.
- *Why care:* it turns "should we search?" into a measurement that can be made before search.
- *Limitation:* one success-rate level, T ≤ 200, two shape families.

**C4 — A measurement protocol and the finding that real prompt reservoirs are flat.**
- *New:* replicate-identified execution noise plus NPMLE deconvolution, ANOVA decomposition and split-half
  reliability, applied to three qualitatively different real prompt families under pre-registration.
- *Evidence:* P5, S4–S6.
- *Why care:* raw prompt leaderboards overstate prompt differences about twofold at typical task counts, and
  linguistic diversity does not buy behavioral diversity. Evaluation designs should include replicates.
- *Limitation:* one agent, two apps, 30 tasks per pool, 40–50 prompts per pool; rare excellent prompts below
  about 2% frequency cannot be excluded.

---

## 4. Framing

| Option | Fits simulation positives | Fits real-data negatives | Risk |
|---|---|---|---|
| A. Growing-arm bandits for prompt search | strong | real data reads as an afterthought | sounds like a bandit-theory paper without theory |
| **B. When is prompt search worth doing?** | strong (the answer has a "how" part) | strong (the answer has a "when not" part) | must avoid "never" |
| C. Prompt heterogeneity is the bottleneck | weak (simulations become setup) | strong | overclaims from one agent |

**Choice: B.** The simulations answer *how* to search when search is worthwhile, and the real data answers
*whether* it is. Neither half reads as a failure.

- **Working title:** *When Is Prompt Search Worth It? Search Breadth, Reservoir Heterogeneity, and Flat Real
  Prompt Pools*
- **Thesis (one sentence):** Adaptive prompt search reduces to choosing how many candidates to evaluate — a choice
  a simple budget-scaled schedule gets nearly right — and its value is set by how much the candidates actually
  differ, which for the real prompt pools we measured was too little to matter.
- **Abstract-level takeaway:** the search algorithm is the easy part; whether a prompt pool varies, and in its
  upper tail, determines whether search pays, and three real pools did not vary enough.
- **Reader takeaway:** before investing in adaptive prompt search, measure the deconvolved spread of the
  candidate pool on a few dozen tasks with replicates. If it is flat, any reasonable small fixed set suffices. If
  it has a tail, recruit to a budget-scaled K and refine. Never open a new prompt every episode.

---

## 5. Figures and tables

### 5.1 Main paper (5 figures, 2 tables). All new; none of the existing figures is used as is.

| # | Content | Claim | Source tables | Status |
|---|---|---|---|---|
| Fig 1 | Setup: reservoir, SEARCH vs REFINE, recruit-then-refine; regret = discovery + selection | framing | none | draw |
| Fig 2 | (a) regret vs K at T = 50…1000, pooled, with K\* marked; (b) K-matched control: Δ vs front-loaded search at equal K, per policy | P1, P2 | `k_star_envelope.csv`, `capmatch_contrasts.csv` | new script |
| Fig 3 | Simple vs learned: (a) Δ forest over the registration ladder and off-family; (b) regret vs breadth cap at T = 1000 (identical below 64, learned over-recruits above) | P3, P2 | `h1b_*.csv`, `capc_*.csv`, `capp_policies.csv` | new script |
| Fig 4 | Value of search vs heterogeneity: regret range over K and the always-search / K = 8 / schedule gaps vs true spread (Beta curve + mixtures), with the 4 real pools placed by deconvolved spread and measured range | P4, P5 | `calibration.csv`, `emp_kstar.csv`, `emp_pool_location.csv`, `glk*_kstar.csv`, `glk*_pool_location.csv` | new script (adapt `_fig3_value_of_search`) |
| Fig 5 | Real-pool anatomy: (a) variance components (task, prompt, interaction, noise) per pool; (b) raw vs deconvolved per-prompt rates; (c) split-half scatter | P5, S4, S5 | `stage0_gmail.csv`, `glk30_gate_classification.csv`, reservoirs, prompt tables | new script (adapt `_fig1`, `_fig7b`) |
| Table 1 | Policy summary: in distribution (T ≤ 200), with headroom (T = 1000), off-family uncapped — schedule, level rule, learned, always-search, fixed K\* | P3 | `h1b_level*.csv`, `capp_contrasts.csv`, `capc_*.csv` | build |
| Table 2 | Real pools: n prompts × tasks, replicate pairs, raw / deconvolved spread, τ_main [MLS], split-half, regret range T = 50/100/200, gap K8, gap always-search, tier | P5 | `emp_*`, `glk30_*`, `glk_*` | build |

### 5.2 Appendix

| Existing artifact | Decision | Why |
|---|---|---|
| `RESULTS.md` feature-ablation table and single-feature AUCs | Appendix table | S1, the e-process negative result |
| commitment-horizon table (k = 1/4/16) | Appendix table | S3 |
| `offline_vs_deployed_A` | Appendix figure | N4 |
| `reservoir_A` (oracle vs observable tail integral) | Appendix figure | S2 |
| `decomposition_A` | Appendix figure | the discovery/selection trade |
| `regret_vs_T_A`, `dynamics_A_familyA/B` | Appendix, one of each | policy behavior over time |
| `tau_curves_A`, `ood_heatmap_A` | Appendix | deployment mechanics (τ, state shift) |
| Test C / Test D tables (`main_C`, `main_D`) | Appendix tables | generalization detail, horizon transfer |
| calibration mixtures detail, flat-threshold derivation | Appendix | P4 detail |
| GLK secondary task mixes, drop-2/4, diagnostics 1–7, matrix replay, uncapped secondary | Appendix tables | P5 robustness |
| June 12-arm paired sweeps, uniform vs SPRUCE replicates | Appendix table | S7 |
| Pre-registrations 2–11 with amendments | Appendix (summary) + supplementary (full) | protocol transparency |

### 5.3 Remove from the paper (keep in the repository)

`cap_sweep` (unpaired, untuned; superseded by the paired sweep); every `_B` figure (Test B duplicates Test A);
`_smoke`, `_cap`, `_robust` variants of the deployment figures (redundant with A plus the paired sweep);
`_D` dynamics and decompositions; `figures/global_log_e_growth*` (the earlier e-process project); the heterogeneity
module's `fig2_tau_bands`, `fig4`–`fig6` and `fig7c/d` (built for the five-pool Pre-reg 10 design that was not run).

---

## 6. Outline (target: 9 pages main + appendix)

| § | Title | Purpose / main claim | Evidence | Length | Not here |
|---|---|---|---|---|---|
| 1 | Introduction | the question, the two-part answer, contributions C1–C4 | Fig 4 preview sentence | 1.25 p | protocol mechanics, pre-registration history |
| 2 | Related work | prompt optimization (APE, OPRO, DSPy, evolutionary, GEPA), bandit prompt selection, infinite-armed bandits, anytime-valid inference, prompt sensitivity, empirical Bayes | — | 0.6 p | long surveys |
| 3 | Prompt search over a growing reservoir | model, simple regret and its decomposition, recruit-then-refine family, oracle labels; and what decides SEARCH vs REFINE (quality, not evidence) | Fig 1; S1, S2 in one paragraph | 1.25 p | feature lists, label-weighting fixes (App. A, B) |
| 4 | Search breadth is the decision | timing is worthless given K; K\* interior and sublinear in T; always-search fails; the cap lesson | Fig 2, Fig 3b | 1.5 p | Test B / D, unpaired sweep |
| 5 | Simple schedules suffice | learned policy = three-number rule; little gain with headroom; generalization off-family; offline ≠ deployed | Fig 3a, Table 1 | 1.25 p | 25 learned variants, τ selection, OOD analysis (App. C, D) |
| 6 | Heterogeneity sets the value of search | calibration curve; the tail matters at equal spread; a pre-search test | Fig 4 (simulation layer) | 0.75 p | mixture grids (App. E) |
| 7 | Real prompt reservoirs are flat | measurement protocol; three pools; placement on the curve; the budget/efficiency finding; pre-registered gate outcome | Fig 4 (real points), Fig 5, Table 2 | 2 p | secondary mixes, diagnostics, latency window, step-cap deviation (App. F) |
| 8 | Discussion and limitations | what changes for practitioners; scope; rare-tail limitation; constrained agents as future work | — | 0.75 p | new experiments |

Appendix: A setup and harness details; B oracle labels, commitment horizons, e-process ablation; C learned
policies, registration ladder, τ, OOD; D cap sweep and generalization tables; E calibration details; F real-pool
protocol (generation prompts, leak guard, noise model, NPMLE, MLS, split-half, deviations and amendments); G
12-arm uniform vs adaptive runs; H pre-registration index; I reproduction commands.

---

## 7. Repository sequence

State on 2026-10-01: `main` (933a46d) ← PR #5 `fix/post-merge-corrections` (6 commits) ← PR #6
`empirical-reservoir` (30 commits, based on #5) ← `prompt-heterogeneity` (34 commits, ef0c331, no PR) ← `paper`
(this plan). The stack is linear; a simulated merge of `prompt-heterogeneity` into `main` is conflict-free.
Nothing supersedes anything destructively: each later branch *adds* sections (§12.x, §13, §14 note, §15) and
pre-registrations 8–11 with amendments, and every table the paper needs is tracked.

1. Open the PR for `prompt-heterogeneity` with base `empirical-reservoir` (stacked, like #6). *Done: PR #7,
   not merged.*
2. Merge #5 into `main` (merge commit, no squash, so the registration-before-run commit order stays auditable).
3. Retarget #6 to `main`, merge (merge commit).
4. Retarget the `prompt-heterogeneity` PR to `main`, merge (merge commit).
5. Open the `paper` PR against `main` after step 4.
6. Keep every branch until the paper is posted. Delete none: the pre-registration commits are referenced by hash
   in the documents.
7. Untracked local leftovers — `analysis/`, `figures/`, `scratchpad/`, `scripts/generate_global_log_e_growth.py`
   — predate this study. Leave them untracked, or move them to an archive branch; do not commit them to the
   paper.

Steps 2–6 need explicit approval.
