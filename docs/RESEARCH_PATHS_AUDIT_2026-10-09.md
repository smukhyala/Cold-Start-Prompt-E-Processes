# Research-paths audit and direction decision

*2026-10-09. Every new analysis here is **exploratory**: it re-reads data that had already been seen. No new episodes
were run and $0 was spent on inference. The code is in `experiments/growing_bandits/audit/`, the tests are in
`tests/test_audit_*.py` (26 pass), and the outputs are in `results/audit/`.*

## 0. Bottom line

1. **Every method failed for the same reason.** E-process allocation, the learned SEARCH/REFINE policy, held-out
   winner selection and agent portfolios all found little, because every realistic prompt pool we built offers
   little above the default prompt.
2. **Spread is the wrong statistic for deciding whether to search. Mass above the incumbent is the right one.**
   - **This audit adds four pools that had never been put through the deconvolution pipeline:** the June Gmail and
     Shopping 12-arm sweeps, and GitLab with and without its planted arm.
   - **The measured gain:** across six realistic pools in three apps, the held-out training winner never reliably
     beats the default `baseline` prompt. The gain ranges from −5.3 to +8.1 pp, and every 90% split interval includes
     0.
   - **The one exception is planted:** GitLab's `gitlab_oracle_operator` (+19.4 pp) is a synthetic positive control
     that hard-codes task-specific issue IDs (`configs/arms_gitlab_strong.yaml:135`).
   - **Why spread misleads:** pools such as GitLab-without-oracle (NPMLE sd 0.10) and Shopping (0.05) have real
     spread, but it comes from prompts *worse* than baseline. A homogeneity test duly rejects on them, yet selection
     loses to the incumbent.
3. **An anytime-valid test of whether a pool is worth searching is valid and cheap to run, but not yet a headline.**
   - **The homogeneity e-process (E1a):** type-I error is controlled, and it catches the planted arm in a median of
     12 tasks. It answers "do prompts differ?", which, per point 2, is the wrong question.
   - **The incumbent-relative e-process (E1b, the null in Ian's multi-armed testing paper):** it asks the right
     question but needs about 240 tasks at a realistic tail. Every task bank we have holds 30–80.
4. **Recommendation:** write paper **A′**, a sharpened version of the measurement paper. Its estimand is the upper
   tail of the deconvolved pool above the incumbent. Include the anytime-valid test as a supporting section, framed
   by its sample-size requirement (§4). No paid trial is needed to draft it.

## 1. Path ledger

Status key: ✗ failed or null · ◐ survived with caveats · ✓ survived · ○ not run.

| # | Path (dates) | Question | Result | Status: why | What it gives a paper |
|---|---|---|---|---|---|
| 0 | Infrastructure and calibration (Apr–Jun) | Does the framework run end to end? | Yes. First real Gmail run: 60 tasks, Sonnet 4.6, $11 | ✓ tooling | The e-process library, `src/cold_start/inference/` |
| 1 | Paired 12-arm sweeps (Gmail 12×60, Shopping 12×80) | Which hand-written prompt is best? | Gmail 45–70%, with only `junior_reactive` above log-e 2.996. Shopping tops out at 46%, where `rapid` = `baseline` | ◐ became reference data | Two extra pools for §2 (incumbent analysis) |
| 2 | SPRUCE vs uniform allocation, Gmail, 120 trials × 3 reps | Does e-process allocation beat round robin? | 59.2% vs 56.1%. Global null never rejected (max log-e 1.92). Online ranking ρ ≈ 0 against the paired ranking | ✗ too little signal; pool flat above baseline | Appendix S7 |
| 3 | The same comparison on Shopping, 240 trials × 3 reps | Does it replicate? | 43.1% vs 42.8%. Final log-e negative in every run | ✗ null; baseline is the best arm (§2) | Appendix negative |
| 4 | GitLab strong-arm control (18×40 paired, allocation at budget 160) | Can SPRUCE find a planted strong arm? | Oracle arm 87.5%. Online run: 0% identification, 5% of pulls. Tuned offline replay: 57–60% of pulls | ✗ online. Under-powered, and too sensitive to the exploration constant | The positive control used in §2 and §3 |
| 5 | Learned SEARCH/REFINE policy Φ (Sep) | Can a classifier decide when to open a new arm? Do e-process features help? | Beats the continuation policy, not the tuned schedule. E-process features null within the training support. Offline/deployed ρ = 0.005 | ✗ the 64-arm cap made all policies identical | Negative-results section |
| 6 | `level_star` rule (Pre-regs 2–6) | What did Φ learn? | Φ ≡ recruit to T^0.75·exp(4(0.5−level)), then refine (Δ ≈ 0, p = 0.96) | ◐ in distribution only, T ≤ 200; +0.005 worse off-family | C2, the "ladder of nulls". The form is known (UCB-AIR) |
| 7 | Cap sweep and K\* envelope | Optimal number of arms; cost of the cap | K\* = 24 → 256 for T = 50 → 1000 (slope 0.8, implied β ≈ 1.6). Cap cost 0.021 (pooled) | ✓ theory known (Carpentier & Valko 2015) | Measured constants at finite T |
| 8 | K-matched timing control | Does timing matter once K is fixed? | No (Δ ≈ 0) | ✓ at T ≤ 200; known in theory | Finite-T confirmation |
| 9 | Held-out mixtures (Pre-reg 8) | Does the schedule generalise? | Within 0.001 of best K. Learned signals worse | ◐ n = 3 environments | "A simple schedule suffices" |
| 10 | Calibration curve (Stage 0) | How does search value depend on spread and tail? | Regret range 0.001 → 0.15 over sd 0.01 → 0.15. At matched spread the tail dominates (0.083 vs 0.015) | ✓ simulation only; unregistered; one level | **The theoretical core of A′**: the tail, not the spread, sets the value |
| 11 | Real Gmail pools G and F (Pre-reg 9, $334) | Do real pools vary enough for K to matter? | NPMLE sd 0.035 (G) and 0.001 (F). 0 of 6 cells informative | ◐ an informative null; the thesis test never ran | Real-pool measurement |
| 12 | GitLab knowledge pool GLK30 (Pre-regs 10/11, ~$210) | Does procedural knowledge create spread? | sd 0.041, not detectable. Split-half ≤ 0. Gate closed | ◐ null. **The 30-step cap was never enforced** (`webarena.py:156`); GLK30 is a rescore | A third construction answers "wrong axis" |
| 13 | Linguistic vs behavioural distance (Pre-reg 12 + Amendment 1) | Does wording distance predict behaviour? | Wording differs (F about 5× G); behaviour does not (Mantel ≤ 0.06). Minimum detectable spread ≈ 0.03–0.05 | ◐ null; behavioural side not blind; §16 uncommitted | S4, instruction → agent behaviour |
| 14 | Retrospective portfolios, transfer and triage (Oct 6–7) | Second agent? Winner transfer? Retry triage? | Second agent +1.1 to +2.1 pp (CIs include 0). Timeout retry +2.1/+2.6 pp at a higher cost per success. Failure AUC 0.63; retrying on it hurts | ✗ weak and post hoc | Future work only |
| 15 | Gmail first-run pilot | Does a frozen retry rule transfer? | Never ran: sandbox blocked port 8001, $0 | ○ | Not needed for A′ |

## 2. New result: the incumbent, not the spread (E3 and the incumbent analysis)

The script is `experiments/growing_bandits/audit/incumbent_gain.py`, written to `results/audit/e3/incumbent_gain.csv`.
It uses 1000 half-task splits. The winner is chosen on the training half and scored on the held-out half against the
`baseline` prompt; for G and F that is the Gmail anchor on the same tasks.

| Pool | Challengers × tasks | NPMLE sd | Winner − incumbent (pp) | 90% split interval | P(gain > 0) |
|---|---|---|---|---|---|
| G (grid, Gmail) | 50 × 30 | 0.035 | −1.1 | [−13.3, +13.3] | 0.28 |
| F (free-form, Gmail) | 50 × 30 | 0.001 | −3.5 | [−13.3, +6.7] | 0.16 |
| Gmail paired (hand-written, June) | 11 × 60 | 0.055† | +8.1 | [−6.7, +16.7] | 0.84 |
| Shopping paired (June) | 11 × 80 | 0.051† | −4.0 | [−12.5, +2.5] | 0.12 |
| GitLab, without oracle | 16 × 40 | 0.101† | −5.3 | [−25.0, +10.0] | 0.23 |
| GitLab, without oracle or explorer | 15 × 40 | 0.062† | −5.3 | [−25.0, +10.0] | 0.23 |
| *GitLab with planted oracle arm* | 17 × 40 | 0.127† | *+19.4* | [+5.0, +30.0] | 0.97 |

† Noise v borrowed from G/F or GLK, because these pools have no repeat cells. Using the Bernoulli
task-difficulty v changes the sd by ≤ 0.01 and leaves the ordering unchanged. Committed values reproduce:
G 0.0354, F 0.0012 and GLK30 0.0413 (`results/audit/e3/crosscheck_committed.csv`). GLK and GLK30 have no incumbent
run on their tasks.

**Reading.**
- **Gain over the pool mean tracks spread** (Spearman 0.89 across pools, `association.csv`), but that gain
  includes avoiding bad prompts.
- **Against the incumbent, spread buys nothing** unless something sits above baseline. In the GitLab variants, the
  spread comes from prompts below baseline (`explorer` at 0.20, `algorithmic` at 0.43).
- **Shopping's baseline ties for best.**
- **Gmail paired is the only realistic candidate for a real gain.** `junior_reactive` scores 0.70 against baseline
  0.58, which would be +8 pp, but the interval includes 0.

This is the decision-relevant form of the calibration curve (path 10). The value of search comes from the
reservoir's mass above the incumbent, and the realistic pools have none that we can detect.

## 3. Anytime-valid "is this pool worth searching?" tests (E1)

Module: `experiments/growing_bandits/audit/flatness_eprocess.py`. Outputs: `results/audit/e1/`.

**The two tests.**
- **E1a, homogeneity.** H0: prompt labels are exchangeable within each task. Given S successes on a task, the
  success set is uniform over the C(K,S) subsets. The bet is a predictable Fisher-noncentral likelihood ratio,
  with odds exp(η·z_k) built from past scores, its normaliser computed by an elementary-symmetric DP, and a mixture
  over η ∈ {0.25, …, 4}. The test is valid by construction whatever the task difficulties.
- **E1b, beats the incumbent.** H0: no challenger beats the incumbent on discordant tasks. The statistic is a
  uniform mixture of one-sided betting e-processes, using `HedgedCapitalEProcess.log_e_upper` and
  `GlobalNullEProcess(mixture)`. This is the global null of Sandoval, Waudby-Smith & Jordan (2026, arXiv:2603.17925).

**Validity (1000 simulations under two nulls).** At α = 0.05, E1a rejects at most 2.2% of the time and E1b at most
0.1%. Both are conservative.

**Replay (500 task orders, α = 0.05).**

| Pool | E1a reject rate (median tasks to reject) | E1b, baseline incumbent | E1b, random incumbent |
|---|---|---|---|
| G / F / GLK30 / GLK | 0.01 / 0.00 / 0.004 / 0.014 | 0 / 0 / n/a / n/a | 0 everywhere |
| GitLab with oracle | **1.00 (12 tasks = 216 episodes)** | 0 | 0.23 |
| GitLab without oracle | 1.00 (21) | 0 | 0.05 |
| GitLab without oracle or explorer | 0.09 | 0 | 0 |
| Gmail paired | 0.27 (36) | 0 | 0.10 |
| Shopping paired | 0.84 (58) | 0 | 0.11 |

**Power (synthetic pools, level 0.6).**

| Test | K | τ for power 0.8 at n = 30 | at n = 240 |
|---|---|---|---|
| E1a | 12 | ≈ 0.09 | ≈ 0.037 |
| E1a | 40 | ≈ 0.06 | ≈ 0.027 |
| E1b | 40 | not reachable (≤ 0.09) | ≈ 0.085; tail scenario 0.96 |

**Reading.**
- **E1a works mechanically but answers the wrong question.** It rejects on Shopping and on GitLab-without-oracle,
  where §2 shows that search loses to the incumbent.
- **E1b asks the right question but cannot get there at our task counts.**
  - With 40 tasks, the oracle arm is discordant with baseline on only 10 of them (9 wins, 1 loss).
  - Even betting on that arm alone, the most log-capital it can reach is about 2.95. The threshold is log 20 ≈ 3.0,
    and mixing over 17 challengers costs a further log 17.
  - This limit is information-theoretic, not a tuning problem.
- **The sample-size requirement is itself a useful result.** Certifying that a pool is worth searching takes on the
  order of hundreds of tasks. That is larger than most agent benchmarks' per-app task banks.
- **What could cut it:** adaptive challenger sampling, as in the Sandoval–Waudby-Smith–Jordan sampling rule, rather
  than this audit-mode replay. This is the natural Ian-fit follow-up.

**Novelty (E2; `LITERATURE_REVIEW.md` covers the rest).**
- **The method is not new.** It extends the k-sample e-values of Hao et al. (2023, arXiv:2303.00471) and Ian's
  multi-armed betting tests (arXiv:2603.17925).
- **The application appears new.** No anytime-valid stopping rule for prompt or agent-pool search was found. The
  closest stopping rule is Makarova et al. (2022, arXiv:2104.08166) for hyperparameter optimisation, and it is not
  anytime-valid.

## 3b. Futility side, and the clinical-trials framing (E5; added 2026-10-09 after Ian's note)

**Ian's pointer.** On 10-06 Ian suggested looking at early-phase clinical trials. The mapping is in
`docs/paper/CLINICAL_TRIALS_MAPPING.md` and fits closely:

- **Prompts are arms and the incumbent is the control.** "Does any prompt beat the incumbent?" is the efficacy
  null of multi-arm multi-stage (MAMS) trials.
- **"Certify that no prompt beats it by δ" is MAMS futility,** and dropping individual prompts is drop-the-loser.
- **The trials literature already predicts our SPRUCE null.** Adaptive randomisation gains nothing when no arm is
  effective (Wason & Trippa 2014; Thall, Fox & Wathen 2015).

**E5, a pool-level futility e-process.** The code is `experiments/growing_bandits/audit/futility_eprocess.py`, the
outputs are in `results/audit/e5/`, and 5 tests pass.

- **Construction.** Each challenger gets a downward betting e-process for H0_k: E[challenger − incumbent] ≥ δ. The
  pool statistic is min_k E_k, which is valid for the union null "some prompt is at least δ better".
- **Validity.** False-futility rate is at most 0.022 at α = 0.05 when one arm sits exactly δ above the incumbent
  (1000 simulations).

**Replay, 500 orders: P(pool certified "not worth searching").**

| Pool | δ = 0.10 | δ = 0.15 | δ = 0.20 |
|---|---|---|---|
| G, F (30 tasks) | 0 | 0 | 0 |
| Gmail paired (60 tasks; best arm +11.7 pp) | 0 | 0 | 0 |
| Shopping paired (80 tasks) | 0.06 | **1.00 (median 64 tasks)** | **1.00 (45)** |
| GitLab, no oracle (40 tasks) | 0 | 0 | 0.04 |
| *GitLab with oracle* | 0 | 0 | 0 (correct: the oracle is ≈ δ = 0.2 better) |

**Flat-pool power.**

| K | n | Power to certify against δ = 0.15 |
|---|---|---|
| 12 | 120 | 0.61 |
| 12 | 240 | 0.985 |
| 50 | 240 | 0.96 |

δ = 0.10 needs more than 240 tasks.

**Reading.**
- **The two-sided sequential design works.** E1b covers efficacy against the incumbent and E5 covers futility, the
  same pairing as MAMS.
- **Both sides are limited by task-bank size, not by the method.** Certifying a pool against δ = 0.15 takes about
  60–120 tasks; δ = 0.10 or tighter takes more than 240.
- **Shopping is the one real pool certifiably not worth searching.**
- **Caveat.** The replay shuffles a finite bank without replacement, so its guarantee holds for the task population.
  Waudby-Smith & Ramdas's without-replacement confidence sequences would make it exact for the bank.

**What stays open for prompts** (from the mapping, (d)). Three things distinguish this setting from a clinical
trial:
- Candidates are generated, so the number of arms K is a choice.
- Tasks are blocks shared by every arm.
- The incumbent is often the best arm.

The three research questions that fit Ian's work are:
1. A pool-level futility e-process, which E5 now prototypes.
2. Adaptive challenger sampling to cut the roughly 240-task requirement, replayed on the GitLab and Gmail paired
   pools.
3. A trial-style "how many arms to carry forward", compared against the infinite-armed K\* ∝ T^0.8.

## 3c. Option B worked through, steps B1–B5 (2026-10-09; simulation plus replay, exploratory, $0)

**Code.** `experiments/growing_bandits/audit/{adaptive_tests,known_baseline,combined_tests}.py` and their runners.
Outputs are in `results/audit/b/`; the headline is `b5_headline.md`. All 53 audit tests pass.

**The five steps.**

| Step | Question | Result |
|---|---|---|
| B1 | Does adaptive challenger sampling (Thompson, as in Sandoval–Waudby-Smith–Jordan) speed up efficacy? | 1.7–1.9× fewer **episodes**, but about 3.6× more **distinct tasks**. Greedy plateaus; UCB is about the same as audit mode. |
| B2 | Does testing against the default's known per-task rate help? | 1.4–1.8× fewer tasks. Real data cap the gain at about 2×, because only the incumbent's run noise is removed. A plug-in estimate reused across a bank is **invalid** (false rejections up to 0.23 at m = 1); a Hoeffding bank shift fixes it. Futility then needs only about 30 prior default runs per task, efficacy about 100. |
| B3 | Does drop-the-loser speed up futility? | 1.7–2.2× fewer episodes with the same tasks, and 2.5–3.2× on realistic pools that contain bad prompts. Real Shopping: 487 episodes against 780. |
| B4 | Is the finite-bank replay (without replacement) valid? | Yes as is: 0 false calls in 2000 orders per pool; conservative. |
| B5 | Do the two wins combine? | Combined futility (drop-the-loser plus known baseline) uses 2.4–3.7× fewer episodes and 1.4–1.7× fewer tasks than audit mode. Every false-decision rate is ≤ 0.05 within Monte Carlo error. |

**Headline, median episodes / distinct tasks to decide (K = number of prompts).**

| Decision | K | Naive (audit, paired) | Best design | Fits a 30–80-task benchmark? |
|---|---|---|---|---|
| "Not worth searching" (futility), δ = 0.15 | 12 | 1450 / 112 | **602 / 79** | Borderline |
| "Not worth searching" (futility), δ = 0.15 | 40 | 5699 / 139 | **2032 / 94** | No (94) |
| "Not worth searching" (futility), δ = 0.10 | 40 | 12177 / 297 | 3308 / 184 | No |
| "Found a better prompt" (efficacy), Δ = 0.20 | 12 | 2301 / 177 | 1662 / 138 (fewest tasks) or 904 / 904 (fewest episodes) | No |
| "Found a better prompt" (efficacy), Δ = 0.10 | 12 | 8138 / 626 | 3960 / 330 or 2394 / 2394 | No |

**Reading.**
1. **There is an asymmetry.** Certifying "not worth searching" is about 2–4× cheaper than certifying "found a better
   prompt" at the same effect size. At K = 12 and 0.15, futility takes 602 episodes / 79 tasks; efficacy takes
   1258 / 1258 or 2196 / 183. Clinical trials have the same pattern: futility is cheaper than efficacy. It
   also fits A′: real pools look flat, so the practical tool is a futility certificate.
2. **Episodes can be cut 2–4×. Tasks barely move.** Adaptivity saves runs but needs fresh tasks, and benchmark task
   banks, not inference cost, are the binding constraint.
3. **The plan's criterion for B to stand alone is not met.** That criterion was cutting about 240 tasks to about 60
   for efficacy. Efficacy stays infeasible on today's benchmarks under every design. Futility comes close at
   δ = 0.15 (79–94 tasks).
4. **Verdict:** B is a strong, Ian-fit section of A′, titled something like "a cheap, valid certificate that search
   won't pay". It is not a standalone headline paper unless a larger task supply is available, for example
   generated or templated tasks.

## 4. Direction scorecard

| | A′: measurement, incumbent-relative tail | B: anytime-valid test as headline | C: agent portfolios / retry | D: search-vs-refine policy |
|---|---|---|---|---|
| Evidence after this audit | **Strong**: 6 realistic pools, 3 apps, 3 construction methods + planted control | Valid test; the decision-relevant version is under-powered at n ≤ 80 | Weak; all intervals include 0; post hoc | Solid in simulation |
| Novelty margin | Moderate. "Pool, not algorithm" is known; **incumbent-relative tail vs spread** is a sharper claim and reconciles prompt-sensitivity papers that report max−min spreads | Application new, method incremental | Low | Low (known theory) |
| Fit with Ian's agenda | Medium | **High** | Low | Medium |
| New spend needed | $0 | Large task banks for a positive demonstration | Paid fresh trials (roadmap items 1–3) | $0 |
| Time to a full draft | Short: `main.tex` exists; 27 `\todo` | Medium | Long | Already inside A′ |
| **Verdict** | **Headline** | **Supporting section** | Future work | Simulation section of A′ |

The plan's decision rule asked for A + B if E1 showed valid type-I error, early rejection on the strong-arm pool, and
quiet on the flat pools. E1a meets all three. But §2 shows its null is not the decision-relevant one, and the
decision-relevant E1b fails on power. B is therefore demoted to a supporting section.

## 5. Recommended paper: A′

**Working title:** *Search pays only above the incumbent: measuring prompt pools for web agents.*

**Spine.**
1. **Simulation (paths 7–10).** The value of searching for K arms is set by the reservoir's upper tail. The
   K-matched, cap and learned-policy results go here as finite-T confirmations of known theory.
2. **Measurement protocol.** Deconvolved reservoir (NPMLE), replicate noise, split-half check, minimum detectable
   spread.
3. **Real pools (paths 1, 11–13, §2).** Six realistic pools across three apps and three construction methods: hand-
   written personas, stylistic grid, free-form LLM-written, and manual-derived procedural knowledge. None shows
   detectable mass above the default prompt. Spread exists in some, but it lies below the incumbent.
4. **Anytime-valid certification (§3).** A valid sequential test, the reason the homogeneity null misleads, and the
   sample size the incumbent-relative null needs. This is the Ian section.
5. **Limitations.** One agent (`gpt-5.4-mini`, low effort). 30–80 tasks per pool. Prompts rarer than about 2% cannot
   be excluded. The step cap was not enforced. The June pools used a different regime. §2 and §3 are exploratory.

**Fix list for `paper/main.tex` (from `results/audit/claim_ledger.csv`).**
- Replace P5 "real pools are flat" with "no realistic pool has detectable mass above the incumbent". Report G and F
  as "none detectable (MDS ≈ 0.03–0.05)", not "SD 0.001".
- Fix the six flagged overclaims (OVER-1…6):
  - the unconditional e-process null;
  - "timing is worth nothing";
  - "the cap cost 0.021" stated flat;
  - the unqualified three-number rule;
  - "fifty prompts behave like one" without the ±0.015 qualifier;
  - the bottleneck sentence without the rare-tail limitation.
- Add §16, the linguistic-distance result. It is still uncommitted in `DEPLOYMENT_RESULTS.md`.
- Disclose that the 30-step cap was never enforced, and that GLK30 is a rescore.
- Cite Sandoval–Waudby-Smith–Jordan 2026, Hao et al. 2023 and Makarova et al. 2022 in §3. Cite Gao et al. 2026 (p1)
  and the Coin Flip paper as the closest priors to P5.

## 6. Next three steps

1. **Commit the audit and the §16 results.** Then add the incumbent-relative estimand to `PAPER_PLAN.md`'s claim
   ledger, labelled exploratory.
2. **Draft `main.tex` §3–§5** around the spine above, and clear the 27 `\todo` markers. Send the draft to Ian with
   §3 of this audit as the e-process question for him: can adaptive challenger sampling bring the incumbent test
   down to benchmark-sized task banks?
3. **Gmail retry pilot: done 2026-10-09** ($6.85, 90/90 episodes; [FIRST_RUN_PILOT_RESULTS.md](growing_bandits/FIRST_RUN_PILOT_RESULTS.md)).
   - **The frozen uplift rule ties timeout-only retry on success** (0.833 each) at 6% higher cost.
   - **Switching prompts equals rerunning.** G_16 and a G_14 rerun give identical results on the incomplete tasks.
   - **Oracle coverage is 0.90–0.93, but every realized policy is at 0.83.** Choosing which attempt to keep (a
     verifier) is the bottleneck.
   - **Direction C stays future work.** Its only open route is a decision-time verifier.
   - **Rerun noise is 33% disagreement, about 3× the September estimate of 10%,** possibly from provider drift. This
     goes in A′'s limitations. A ≈ $3 recheck of old repeat cells is optional.
   - **Further optional spend:** if a positive real-data demonstration is wanted, the cheapest is one Gmail pool
     built to have an upper tail (task-family procedural prompts that do not leak task IDs). Score it on a fresh
     task bank, because the 30 unused Gmail tasks are now spent on this pilot.
4. **Confirmatory run, Pre-registration 13: done 2026-10-10** ($75.81;
   [PREREG13_RESULTS.md](growing_bandits/PREREG13_RESULTS.md)).
   - **Leg M, Gmail winner's curse: "shrank", as registered.** G_14's +10 pp in-sample lead is 0.0 pp held out
     (90% CI ±9.1 pp). The §2 winner's-curse claim is **confirmed for Gmail at the +10 scale**, narrowly.
   - **Leg G, GitLab winner vs default: "uninformative", as registered.**
     - W − D is −0.9 pp with a one-sided upper bound of +3.4 pp, which would certify.
     - The planted control reached only +3.6 pp, with a lower bound of −1.2 pp.
     - Its cookbook covers just 20 of the 110 fresh tasks; on those 20 it is +10 pp. This is an exploratory
       diagnosis.
     - The §2 "no better than the default" claim **stays exploratory**. A valid control on every task needs a new
       registration, about $45.
   - **Leg N, noise:** rerun disagreement is 20% against 10% in September (p = 0.12). This is possible drift, not significant, and goes in the limitations.
5. **Positive-control re-run, Pre-registration 14: done 2026-10-10** ($50.98 including pilots;
   [PREREG14_RESULTS.md](growing_bandits/PREREG14_RESULTS.md)).
   - **The control holds, narrowly.** A full 140-task answer key beat a same-session default by +5.5 pp, with a
     one-sided lower bound of +0.7 pp.
   - **Leg G is now registered-informative**, as "informative with a next-day control". The §2 "winner no better
     than the default" claim is **confirmed on GitLab** (W − D −0.9 pp, upper bound +3.4 pp), with the two-session
     design disclosed.
   - **Fragile (exploratory):** against Leg G's earlier default, O2 is only +1.8 pp, with a lower bound of
     −3.7 pp. The default dipped 3.6 pp between sessions, which is not significant.
   - **New finding:** even an exact answer key lifts `gpt-5.4-mini` only about 5 pp on GitLab. O2 scores 0.418,
     about the same as the old cookbook's 0.436. Execution, not knowledge, is the ceiling.
6. **Gmail answer-key headroom, Pre-registration 15: done 2026-10-11** ($15.11;
   [PREREG15_RESULTS.md](growing_bandits/PREREG15_RESULTS.md)).
   - **Reading: "small".** A 60-task answer key against the default gives H = −1.7 pp, 90% CI [−8.8, +5.5].
   - **Exploratory:** the agent referred to the key in only 33 of 120 episodes. The answers sitting in the system
     prompt barely steer `gpt-5.4-mini`, so answer-key headroom is a lower bound limited by adherence.
   - **Drift:** the same default scored +15 pp higher than in Leg M on the same 30 tasks (CI [+5.8, +24.2]).
     Session-to-session drift exceeds every prompt effect measured, so only within-session paired comparisons are
     trustworthy.

## Provenance

| Analysis | Model | Code | Output |
|---|---|---|---|
| Exploration: three history inventories | Sonnet | read-only | n/a |
| E0, matrices | Haiku | `audit/export_matrices.py` | `results/audit/matrices/` (11 tests) |
| E1, e-processes | Sonnet; statistics specified and checked by Opus | `audit/flatness_eprocess.py`, `run_e1.py` | `results/audit/e1/` (6 tests) |
| E2, novelty search | Sonnet | n/a | §3 above |
| E3, cross-pool | Sonnet | `audit/cross_pool.py` | `results/audit/e3/` (7 tests) |
| Incumbent gain | Opus | `audit/incumbent_gain.py` | `results/audit/e3/incumbent_gain.csv` (2 tests) |
| E4, claim ledger | Haiku | n/a | `results/audit/claim_ledger.csv` |
| Number audit | Haiku | n/a | §7 below |

## 7. Number audit (Haiku pass)

| Claim | Doc value | Source value | Status |
|---|---|---|---|
| §2 incumbent gain: G | −1.1 pp | −1.073 (incumbent_gain.csv) | OK |
| §2 incumbent gain: F | −3.5 pp | −3.473 (incumbent_gain.csv) | OK |
| §2 incumbent gain: Gmail paired | +8.1 pp | +8.083 (incumbent_gain.csv) | OK |
| §2 incumbent gain: Shopping paired | −4.0 pp | −3.975 (incumbent_gain.csv) | OK |
| §2 incumbent gain: GitLab without oracle | −5.3 pp | −5.335 (incumbent_gain.csv) | OK |
| §2 incumbent gain: GitLab oracle | +19.4 pp | +19.375 (incumbent_gain.csv) | OK |
| §2 NPMLE sd: G | 0.035 | 0.03543 (crosscheck_committed.csv) | OK |
| §2 NPMLE sd: F | 0.001 | 0.00119 (crosscheck_committed.csv) | OK |
| §2 NPMLE sd: GLK30 | 0.041 | 0.04125 (crosscheck_committed.csv) | OK |
| §2 Association Spearman | 0.89 | 0.8929 (association.csv) | OK |
| §3 E1a validity max | 2.2% | 0.022 (validity.csv) | OK |
| §3 E1b validity max | 0.1% | 0.001 (validity.csv) | OK |
| §3 E1a GitLab oracle reject rate | 1.00 | 1.0 (replay_summary.csv) | OK |
| §3 E1a GitLab oracle median tasks | 12 | 12 (replay_summary.csv) | OK |
| §3 E1a GitLab oracle episodes | 216 | 216 (replay_summary.csv) | OK |
| §3 E1a Gmail paired reject rate | 0.27 | 0.266 (replay_summary.csv) | OK |
| §3 E1a Gmail paired median tasks | 36 | 36 (replay_summary.csv) | OK |
| §3 E1a Shopping reject rate | 0.84 | 0.836 (replay_summary.csv) | OK |
| §3 arithmetic log(20) | ≈3.0 | 2.996 | OK |
| §1 Path 1 Gmail max | 70% | 70.0% (paired_sweep_status.md) | OK |
| §1 Path 1 Shopping max | 46% | 46.2% (paired_sweep_shopping_80_status.md) | OK |
| §1 Path 1 Shopping rapid = baseline | 46.2% | both 46.2% (paired_sweep_shopping_80_status.md) | OK |
| §1 Path 2 uniform | 56.1% | 56.1% (uniform_vs_adaptive_120_replicates.md) | OK |
| §1 Path 2 adaptive | 59.2% | 59.2% (uniform_vs_adaptive_120_replicates.md) | OK |
| §1 Path 2 max log-e | 1.92 | 1.922 (uniform_vs_adaptive_120_replicates.md) | OK |
| §1 Path 3 Shopping uniform (3 reps) | 42.8% | 42.78% (aggregate: 308/720) | OK |
| §1 Path 3 Shopping adaptive (3 reps) | 43.1% | 43.06% (aggregate: 310/720) | OK |

**Summary:** 26 numbers checked and verified against source files. All readings match within rounding tolerance. No mismatches detected.
