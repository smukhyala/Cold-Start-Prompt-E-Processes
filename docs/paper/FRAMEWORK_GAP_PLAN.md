# Making it a framework: what our runs show, and which public data fills the gaps (2026-10-11)

## Update: first public replays (2026-10-12)

The ledger below records the state when this plan was written. Three frozen public-data replays have since been run:

- [Terminal-Bench replay 01](PUBLIC_REPLAY_01_RESULTS.md): the A/A label-swap check crossed 4/2,000 times on 68 complete real-noise tasks. Capy showed a +8.8 pp paired gain and +10.3 pp on reruns, but efficacy crossed in 0/2,000 orders. Three nominated submissions had misleading or unverifiable per-trial model IDs and were excluded; one same-model submission used Vertex routing.
- [PromptEval replay 02](PUBLIC_REPLAY_02_RESULTS.md): across 855 model × subject settings, the best-of-99 search lead averaged +5.02 pp and shrank to +0.54 pp held out. Mean shrinkage was 4.48 pp [4.05, 4.92]; the first sequential boundary was 421 futility, 4 efficacy, 430 unresolved. Row 0 is positional, not a verified default.
- [SWE-bench matched replay 03](PUBLIC_REPLAY_03_RESULTS.md): two prespecified same-date, same-model whole-system comparisons crossed the Bonferroni-adjusted efficacy boundary after 92 and 95 of roughly 500 tasks. Paired gains were +20.0 pp (GPT-4) and +8.9 pp (Claude 3 Opus). This closes the narrow "no real winner detected" gap for historical replays, with different attempt policies and benchmark reuse as limits. It is a fixed-pair check, not the full dated-leaderboard R4.

**Revised next steps.** R1 on our own repeat cells and R2 on measured drift test whether the real-noise result transfers to live sessions. R5 on Mizrahi supplies a natural incumbent before making a general claim about prompt-search payoff. Then R6 can fit a design calculator using these heterogeneous settings. The old [prompt-search manuscript](../../paper/main.tex) has not yet been rewritten around the framework; it should be revised only with these distinctions between prospective study, historical replay, and fixed-benchmark assumptions kept explicit.

**Paper decision after these replays.** Continue developing the framework paper. Its supported central claim is that a single incumbent-relative sequential protocol can certify large gains early, certify practical futility in many settings, and leave underpowered settings unresolved, while naive best-of-K search exaggerates gains. Present the SWE-bench detections as historical whole-system examples; present the web-agent runs as prospective uses with no certified winner. Do not claim that the current data prove a universal error guarantee under drift, that all prompt pools are flat, or that a historical benchmark replay is a prospective deployment result. The older prompt-search manuscript remains a separate draft until the remaining replay checks settle the scope of the framework account.

**The paper's claim, stated generally.** An anytime-valid sequential procedure, modeled on multi-arm clinical trials
(efficacy and futility stopping, drop-the-loser, known-baseline designs), tells a practitioner, for any model, agent,
or benchmark, whether any of K candidate configurations beats their incumbent default. It stops early with guaranteed
error control. Naive best-of-K selection, by contrast, overstates the gain.

Our web-agent studies are the **prospective, pre-registered case study**. Public datasets supply the breadth.

## 1. What our runs already show

| Claim | Shown | Not shown (the gap) |
|---|---|---|
| **C1 Validity** | False-call rates ≤ 0.05 for efficacy, futility, known-baseline and drop-the-loser. Simulation (B5, E1, E5) and 2,000 replay orders per pool (B4). | **Only simulated or replayed on our pools.** No test against real rerun noise, replicates, or session drift. The same default moved +15 pp between sessions (Pre-reg 15). The plug-in known-baseline variant is invalid (false rejections up to 0.23) without the Hoeffding shift. |
| **C2 Power** | A planted winner is caught in 12 tasks (E1a). Futility certified on Shopping in about 64 tasks (δ = 0.15). Futility on GitLab Leg G crossed at task 65. | **No real winner ever detected.** The only positive is a planted, task-leaking prompt, which gave just +5.5 pp when re-run. Incumbent-relative efficacy (E1b) never rejects on any real pool. |
| **C3 Design guidance** | Futility is 2–4× cheaper than efficacy, and tasks, not episodes, are the binding constraint (B1–B5). | **One simulated difficulty distribution (Gmail G), N ≤ 80.** No calculator, and no check on other data. |
| **C4 Practical need** | Winner's curse: Gmail's +10 pp in-sample lead was 0.0 pp held out, confirmed narrowly. GitLab winner ≈ default, confirmed. | Two apps, one model. No estimate of **how often searches in the wild find a real winner**. |
| **C5 Generality** | 3 apps, K = 12–50, N = 30–80. | **One model** (gpt-5.4-mini; the Haiku run is on hold at 16 of 240). Prompts only, binary outcomes only. |

**The gaps that would sink a "framework" paper, worst first:**
1. It never detects a real winner.
2. One model.
3. Validity untested under real noise and drift.
4. Design guidance from one distribution.
5. Prompts only, binary only, small K.

## 2. Public datasets that fill them

| Dataset | What varies | K × N | Incumbent | Repeats | Fills | Access |
|---|---|---|---|---|---|---|
| **Terminal-Bench 2.0 leaderboard** ([HF](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard)) | Agent scaffold × model; the same model appears under many scaffolds | ~100 configs × 89 tasks | Default harness (Terminus 2 / mini-SWE-agent) per model | **Yes, ≥5 trials per task** | Gaps 1, 2, 3, 5 | Apache 2.0 |
| **SWE-bench Verified experiments** ([GitHub](https://github.com/SWE-bench/experiments)) | Whole agent systems, dated 2023–2025 | ~86 × 500 | The previous best at each date | No | Gap 1 (real winners, N = 500), sequential use | Public; license unstated |
| **PromptEval MMLU correctness** ([HF](https://huggingface.co/datasets/PromptEval/PromptEval_MMLU_correctness)) | Prompt templates | 100 templates × 100 examples × 57 subjects × **15 models** | Pre-register one (canonical template id unverified) | No | Gaps 2, 4; C4 at scale | MIT, small |
| **Mizrahi et al. 2024** ([GitHub](https://github.com/SLAB-NLP/Multi-Prompt-LLM-Evaluation)) | Instruction paraphrases | ~175–240 paraphrases × 100 examples × 39 tasks × **16 models** | **Original benchmark template** | No | Gaps 2, 4; C4 with a clean incumbent | CDLAv2; raw responses ~190 MB |
| HAL + Terminal-Bench matrix ([GitHub](https://github.com/fsndzomga/efficient-benchmarking-ai-agents)) | Scaffold × model across 8 agent benchmarks | 33 scaffolds, 70+ models | Pick a baseline scaffold | No | Gap 5 (domains) | MIT, small CSV (derived copy; check against the originals) |
| DOVE / DOVE_Lite ([HF](https://huggingface.co/datasets/nlphuji/DOVE_Lite)) | Thousands of prompt perturbations | 78 benchmarks, many models | Default dimension values (unverified) | Unverified | Breadth (optional) | Gated, ~105 GB |

**Not used:**
- **HELM:** few prompt variants, multi-TB.
- **Aider:** aggregate results only.
- **GEPA / prompt-optimizer logs:** not released. A collaborator ask (Lakshya Agrawal) could get them.
- **BFCL per-entry results:** unverified. A collaborator ask (Huanzhi Mao).

## 3. The replays: each one demonstrates one claim

| # | Replay | Data | Shows | Gap |
|---|---|---|---|---|
| R1 | **A/A validity under real noise.** Split repeated trials of the *same* configuration into two pseudo-arms (a true null with real rerun noise and real task heterogeneity). Run every test in random task orders and report the empirical false-call rate. Repeat on our own repeat cells (G/F/GLK repeats; Pre-reg 13–15 replicate pairs) and on a within-task label-permutation null for the homogeneity test. | Terminal-Bench trials; our repeats | Error control holds on real, noisy data, not only in simulation | 3 |
| R2 | **Drift and the known-baseline trap.** Use the measured drift in a historical default (+15 pp, Pre-reg 15 vs Leg M) to show the plug-in known-baseline test failing, and the concurrent-default and Hoeffding-shifted versions holding. | Our Gmail runs (+ Terminal-Bench job dates, if present) | A concrete design rule: always run the default concurrently, or shift | 3 |
| R3 | **Real winners: power and stopping time.** Use a same-model scaffold search against the default harness, with efficacy plus drop-the-loser in random task orders. Report the detection rate, tasks to detection, and episodes saved against a fixed-n design. | Terminal-Bench | It detects genuine winners, and at what cost | 1, 5 |
| R4 | **The leaderboard as a sequential trial.** Order submissions by date, and test each against the current incumbent. When would the framework have certified each new state of the art, and how many claimed improvements would not certify? | SWE-bench Verified | Real winners at N = 500, in a use case everyone recognises | 1 |
| R5 | **How often does prompt search pay? The winner's curse at scale.** For each (model, task) setting: split examples into search and held-out halves, take best-of-K on the search half, and measure held-out shrinkage against the incumbent. Then run efficacy and futility to report the fraction of settings that certify futility, certify a winner, or stay unresolved, and the evaluation budget saved. | PromptEval (15 models × 57 subjects), Mizrahi (16 models × 39 tasks) | The general version of our Gmail result, across about 1,500 settings | 2, C4 |
| R6 | **Design calculator, fit and validated.** Fit tasks-to-decision as a function of (K, δ, discordance rate, between-arm spread) on half the settings, and predict the other half and our pre-registered studies. Re-check the futility/efficacy cost ratio and "tasks are binding" across datasets. | All of the above | Design guidance that transfers, as a usable tool | 4 |

**Our own studies' role:** they are the only *live, pre-registered* uses of the framework. They show it working in
practice, including the failure modes a practitioner will hit:
- a control that covers only part of the task set;
- session drift;
- machine sleep;
- provider outages and credit exhaustion.

## 4. Order and cost

- **First, about a day each, all $0:**
  - R5 on PromptEval (tiny MIT data).
  - R1 on our own repeats.
  - R3 and R1 on Terminal-Bench.
- **Then:** R4 (SWE-bench), R5 on Mizrahi, R2, R6.
- **Engineering:** one loader per dataset into a common long table (`dataset, setting, arm, task, replicate,
  outcome, is_incumbent`). The existing code (`audit/flatness_eprocess.py`, `futility_eprocess.py`,
  `known_baseline.py`, `adaptive_tests.py`) runs on it unchanged.
  - Fractional outcomes (Terminal-Bench reward, mean over trials) need either the betting tests' [0,1] version or
    binarisation, decided before any replay.
- **Spend: $0.** The Haiku run (Pre-reg 16) is no longer needed for generality and stays on hold; Terminal-Bench and
  PromptEval cover many models.

## 5. Discipline and caveats

- Each replay should be **registered before its data is loaded**, as before. Public data is "seen" by others but not
  by us. Fix the incumbent, δ, α, task-order seeds and readings in advance.
- **Unverified points to check before registering:**
  - PromptEval's canonical template id;
  - Terminal-Bench `result.json` fields and job dates;
  - SWE-bench's license;
  - DOVE's incumbent flag;
  - the arXiv IDs that surfaced only in search.
- **The honest limitation that remains:** most public data has no repeats. The validity-under-noise evidence (R1) rests
  on Terminal-Bench's trials and our own repeats.
