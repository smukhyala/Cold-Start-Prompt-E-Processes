# Early-phase clinical trials and prompt-pool search: a mapping

*2026-10-09. Prompted by Ian's reply: the problem resembles early-phase trials, where one finds a good variant (dose, schedule) before committing to a full trial. Citations were checked against arXiv, PubMed, publisher or university-repository pages via web search on this date. "(unverified)" means the claim comes from memory or a secondary snippet. Takeaways are one-line paraphrases, not quotes. Nothing here has been run.*

## (a) Translation table

| Trial concept | Our concept | What it predicts or offers |
|---|---|---|
| Treatment arm / dose / schedule | Candidate prompt | Ian's own multi-armed testing paper (arXiv:2603.17925) uses "dosages of a treatment" as its example. |
| Control arm, standard of care | Incumbent (`baseline`) prompt | Every modern design (MAMS, platform) tests against the control. This is exactly our E1b and the audit's "incumbent, not spread" finding. |
| Patient | Task episode | Binary outcome, but our "patients" are cheap and repeatable (no ethics constraint). |
| Stratum / block | Task (difficulty varies) | All arms run on the same tasks, which is a paired or blocked design. Gains power. |
| Phase II screen, then phase III confirm | Search over K prompts, then held-out confirmation | The held-out split we use is a seamless phase II/III. |
| Pick-the-winner | Held-out winner selection | Selection can be right (high PCS) without the winner beating control. |
| Futility stop | Drop a prompt, or abandon the pool | Our pool-level "not worth searching" is a futility stop for the whole phase II. |
| Arms added to a platform | Growing bandit, candidates arriving | Platform-trial theory on added arms, time trends, and FWER. |
| Non-concurrent controls | Incumbent scored on an earlier task batch | Task-bank drift plays the role of time trends. |
| Dose-response monotonicity | Axis-grid pool G has ordinal structure | Parametric or isotonic borrowing could cut K. |
| Response-adaptive randomization (RAR) | SPRUCE / UCB allocation | The RAR critique predicts our "no gain over uniform" result. |
| Phase II go/no-go | E1a / E1b pool certification | The same decision, made with e-processes. |

## (b) Key citations by area

**1. Selection and pick-the-winner**
- Simon, Wittes & Ellenberg (1985), "Randomized phase II clinical trials", *Cancer Treatment Reports* 69(12):1375-1381. Verified bibliographic data. Pick-the-winner designs choose the best arm for further study without a control comparison (content from memory, unverified). A small sample gives a reasonable chance of picking the best arm, not evidence that it beats control.
- Bechhofer (1954), *Ann. Math. Statist.* 25:16-39 (verified). It introduced the indifference-zone formulation: pick the best with probability at least P* whenever the best exceeds the rest by at least delta.
- Sizing (my back-of-envelope, not from the papers): with K around 40, delta 0.10, P* 0.95 and p near 0.5, an unpaired design needs several hundred tasks per arm. The incumbent-relative test needing about 240 tasks is the same order, so this is a fundamental limit rather than a quirk of e-processes.

**2. MAMS, drop-the-loser, seamless II/III**
- Magirr, Jaki & Whitehead (2012), *Biometrika* 99(2):494-501 (verified). Gives a generalized Dunnett test with efficacy and futility boundaries against a shared control, with strong FWER control.
- Wason & Jaki (2012), *Stat. Med.* 31(30):4269-4279 (verified). Optimal MAMS designs minimise expected sample size over a set of effects.
- Sampson & Sill (2005), *Biometrical J.* 47(3):257-268 (verified). Drop-the-losers keeps the best stage-1 arm and takes more data on it.
- Royston, Parmar & Qian (2003), *Stat. Med.* (verified at title level only). The origin of MAMS, with an intermediate outcome that merges phases II and III.
- Wason & Trippa (2014), PMC3981566 (verified). Compares Bayesian adaptive randomization with MAMS. When no arm is effective, MAMS slightly outperforms AR. This matches our null for SPRUCE.

**3. Platform trials with arms entering over time**
- Sydes et al. (2009), *Trials*, STAMPEDE (verified title). A 6-arm, 5-stage MAMS design with arms dropped for lack of benefit and others added later.
- Barker et al. (2009), *Clin. Pharmacol. Ther.*, I-SPY 2 (verified). An adaptive phase II platform that graduates or drops arms. REMAP-CAP and GBM AGILE (unverified; not looked up).
- Burnett, König & Jaki (2020), arXiv:2007.04951 (verified). Adds arms to ongoing MAMS trials via the conditional error approach.
- Lee & Wason (2020), *BMC Med. Res. Methodol.* (verified). Non-concurrent controls bring power but risk bias from time trends.

**4. Dose-finding**
- O'Quigley, Pepe & Fisher (1990), CRM, *Biometrics* (PubMed 2350571, verified). Model-based, treats each patient at the dose closest to target.
- Liu & Yuan (2015), BOIN, *JRSS-C* (verified at title level). A model-assisted interval design.
- Thall & Cook (2004), EffTox, *Biometrics* (verified at title level). Trades efficacy against toxicity through a utility.
- Bretz, Pinheiro & Branson (2005), MCP-Mod, *Biometrics* 61(3):738-748 (verified). Tests for a dose-response signal across candidate monotone shapes, then fits and picks a dose.

**5. Response-adaptive randomization critiques**
- Thall, Fox & Wathen (2015), *Annals of Oncology* (verified). Outcome AR can put more patients on the inferior arm, give less reliable inference, and lose power.
- Korn & Freidlin (2011), *J. Clin. Oncol.* 29:771-776 (verified). Finds no benefit over fixed randomization.
- Caveat: these critiques concern two arms, patient benefit, and time trends. Our setting has none of them, so only the inferential parts transfer.

**6. E-values and anytime-valid methods in trials**
- Grünwald, de Heide & Koolen (2024), "Safe testing", *JRSS-B* (verified). The foundation: e-values stay valid under optional continuation.
- Sokolova & Sokolov (2026), arXiv:2602.06379 (verified). A practitioner guide to e-values for adaptive trials, covering platform trials, arm dropping and graduation (per search snippet).
- Sandoval, Waudby-Smith & Jordan (2026), arXiv:2603.17925 (verified). Sequential testing by betting over multiple arms, with the global null "all arms are null".
- Lin, Ma, Ren & Wei (2026), arXiv:2609.26651 (verified). E-value adaptive sampling that chooses hypotheses to test while controlling FDR. Possibly the missing sample-efficiency tool.
- ter Schure and Grünwald, ALL-IN meta-analysis (unverified details).

**7. Portfolio and "how many candidates" questions**
- Stallard et al., seamless II/III and the optimal number of treatments to carry forward (search surfaced "Optimal choice of the number of treatments to be included in a clinical trial", unverified authors and venue). This is the closest trial analog of K*.
- Burnett, König & Jaki, again: optimal control-to-experimental allocation in platforms.
- Not found: a clean result for pipeline-level success rates and value of screening more compounds. Treat as an open literature search.

**8. Trials meets prompt/ML selection**
- Shi et al. (2024), TRIPLE, arXiv:2402.09723 (verified). Casts prompt selection as fixed-budget best-arm identification.
- Zhou, Ji & Sun (2025), "On Speeding Up Language Model Evaluation", ICLR 2025 (verified at title level). Bandit evaluation that finds the top method with 5-15% of the compute.
- No source found that explicitly calls prompt search a phase II screen with a control arm, or that uses incumbent-relative anytime-valid testing for it. I did not exhaust the search.

## (c) What the literature already answers, and what is open

**Already answered (do not reinvent):**
- Selection without a control is a different, weaker claim than beating control (pick-the-winner versus MAMS).
- Dunnett-style comparisons against a shared control, with futility and efficacy stopping, and FWER control as arms are added.
- RAR gives little for inference and nothing under the null. Our SPRUCE null is the textbook result.
- E-process platform-trial machinery (arm graduation and dropping) already exists.
- Sample size scales with 1/delta^2 times a log K price. The 240-task requirement is expected.

**Open when the treatments are prompts:**
- Evaluations are cheap and repeatable, so rerun noise can be measured and separated from task difficulty. Trials cannot do this.
- Candidates are generated and nearly unlimited, so K is a decision. Trial theory treats K as small and given. Infinite-armed bandits fill this gap.
- Tasks are blocks shared by all arms, with no patient-benefit constraint on allocation. The RAR ethics rationale disappears, so allocation is purely statistical.
- The incumbent is known, and often best. The estimand is the pool's upper tail above the control, not any arm's effect.
- Generation is correlated with the incumbent (rewrites of it), and the realistic prior over candidate quality is below control.

## (d) Three research questions, with zero-cost tests

1. **Pool-level futility stopping.** Can an anytime-valid futility e-process (evidence that no arm exceeds the incumbent by delta) end search early, with an explicit stopping rule for the "all arms below incumbent" pools?
   - Data: `results/audit/e3/incumbent_gain.csv` and `flatness_eprocess.py` replays on G, F, Gmail paired, Shopping, GitLab with and without the oracle.
   - Test: reverse the E1b null and measure tasks to futility stopping, with the GitLab oracle as the false-stop control.

2. **Adaptive challenger sampling for the incumbent-relative test, with shared task blocks.** Ian's sampling rule plus the Lin et al. adaptive-FDR idea may cut 240 tasks toward 80. Compare with the Dunnett/MAMS fixed-sample benchmark.
   - Data: replay on the GitLab oracle pool (10 discordant tasks of 40) and the Gmail paired pool.
   - Test: replay with the sampling rule instead of audit-mode uniform replay, plus the simulated K=40 tail scenario from the E1b power table.

3. **Value of screening K prompts under an incumbent-relative tail, with a trial-style K\*.** Reconcile the infinite-armed result (K\* proportional to T^0.8) with the seamless II/III "number of arms to carry forward" and the indifference-zone price.
   - Data: the Stage-0 calibration curve and the K\* envelope (paths 7 and 10), plus the deconvolved NPMLE tails for G, F and GLK30.
   - Test: compare the simulated optimal K with a Bechhofer-style K under the same budget, and check whether the incumbent-relative tail shifts K\* toward zero for the real pools.

*Possible extras, lower priority:* use MCP-Mod ordinal shapes on pool G to test whether structure lets the grid shrink; and test the platform-trial time-trend analog by checking whether incumbent scores drift across task batches (zero cost, from existing runs).
