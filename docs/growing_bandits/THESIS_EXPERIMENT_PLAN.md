# Thesis experiment plan: when does another agent add value?

*Draft for discussion with Ian and Keegan, 2026-10-06. Analyses below that use collected outcomes are exploratory: their questions were chosen after inspecting these data. No new paid runs are authorized by this plan.*

## Thesis question and claim ladder

Given a fixed evaluation and deployment budget, when should a system sample more candidate agents, refine existing candidates, or deploy multiple agents? Here an **agent** is a frozen prompt-conditioned instance of the same underlying web agent. The current data identify independent-attempt portfolio coverage and selection on held-out tasks. They do not identify communication, joint execution, or a deployable success verifier.

Claims should advance only as the corresponding evidence is available:

1. **Current-data claim:** Real prompt pools have limited transferable quality and limited *measured* independent-attempt complementarity; synthetic environments show how reservoir quality changes the search decision.
2. **External replication claim:** The same prespecified analysis works on a genuinely independent outcome matrix or fresh task set.
3. **Interactive multi-agent claim:** A controlled experiment with agent communication and matched compute shows an effect beyond independent attempts.

The first claim is enough to write a scoped thesis. Claims 2 and 3 are optional expansions requiring suitable external data or new runs.

## Existing evidence inventory

| Source | Reusable trials | Valid use | Key limit |
|---|---:|---|---|
| Gmail G | 50 prompts × 30 tasks + 145 repeat cells | Selection, two-agent coverage, same-agent baseline, cost/step analysis | One main outcome per cell; 30 task clusters |
| Gmail F | 50 × 30 + 155 repeat cells | Same; G/F cross-family pairs on the same tasks | G and F share tasks, so they are not independent replications |
| GitLab GLK30 | 40 × 30 + 120 repeat cells | Same at a 30-step score cap | Rescore of existing GLK trials, not a new experiment |
| Older GitLab paired study | 17 non-cookbook prompts × 40 tasks | Contrasting pool with stronger task-transfer signal | Different regime, no same-cell repeats; exclude cookbook prompt from primary analysis |
| Synthetic deployment runs | Many saved policy episodes and environment families | Budget/candidate-breadth mechanism and out-of-family checks | Assumed reservoir distributions, not evidence that real prompt pools have rare strong agents |

Use frozen snapshots, manifest checks, and the existing loaders. Retain costs, steps, timeout and task difficulty from live logs for efficiency analysis. Never treat repeated train/test partitions of one task bank as independent experiments.

## Experiment 1: fixed-budget independent-agent coverage (no new trials)

**Question.** With two attempts on the same task, is a different prompt better than a rerun of the same prompt?

**Estimand.** For each randomly replicated prompt-task cell, compare `max(first_a, repeat_a)` with the average `max(first_a, first_b)` over other prompts `b` on that task. Average over sampled cells. Decompose the difference into (i) second-attempt marginal success and (ii) excess disagreement, so a stronger second agent is not mistaken for complementarity.

**Controls.** Same-prompt repeat, random different prompt in pool, and G/F cross-family second prompt. Report two-attempt coverage, gain after first failure, task-cluster bootstrap intervals, and cost/step distributions. Treat the oracle `max` score as *potential coverage*: it assumes correct recognition of success and cannot establish operational team success.

**Current readout.** Different-prompt gains are G +1.4 points [−1.1, +4.2], F +1.1 [−1.4, +3.7], GLK30 +2.1 [−0.5, +4.9] from a 10,000-resample task bootstrap. None is resolved from zero. Cross-family G/F pairs add no clear benefit over within-family pairs. Freeze a reproducible script, source hashes, and full sensitivity table before using this in the thesis.

## Experiment 2: can development-time selection build a better team? (no new trials)

**Question.** Does choosing agents for *complementary* training-task successes improve coverage on held-out tasks more than choosing agents by individual success rate?

**Protocol.** On each stratified task split, spend the same observed training-cell budget for each selector. Compare: random pair; top-two by marginal training rate; greedy coverage pair; and a shrinkage/noise-adjusted coverage selector. Evaluate all fixed pairs on the held-out tasks, for team sizes 1, 2 and 3. Use the same candidate subsets and train budgets across methods. Reserve the older GitLab study as a separate regime check; run a sensitivity with its cookbook prompt clearly labeled as task-informed leakage. Do not tune rules on test outcomes.

**Primary outcome.** Held-out `at least one succeeds` coverage minus the top-two-marginal rule at identical team size and training budget. Secondary: gain per extra agent, baseline-adjusted disagreement, and realized dollar/step cost. Show the hindsight-optimal pair only as an unattainable upper bound. Summarize variation across tasks with task-level intervals; split percentiles describe instability, not independent evidence. Since all three new pools have already been inspected, this remains exploratory even with task holdouts.

**Interpretation.** A stable positive held-out gain supports portfolio selection. A null or negative gain supports the boundary that diversity estimated from small task banks overfits, even when hindsight diversity appears large.

## Experiment 3: can we diagnose when breadth is worth buying? (no new trials)

**Question.** Can a small pilot identify pools where searching more agents or selecting a complementary team pays off?

**Protocol.** Reuse the existing task-transfer setup, but compare candidate pool sizes and *total training episode budgets*, not just counts of prompts. Compute pilot signals using training tasks only: cross-task rank stability, noise-corrected between-agent variance, and same-vs-different disagreement where repeats exist. Predict held-out gain from expanding the candidate set and held-out team coverage. Calibrate with nested task splits and leave-one-task checks. The four substantive pools are too few for a validated pool-level classifier; report descriptive performance and uncertainty, including failure cases.

**Mechanism check.** Place the empirical estimates into the existing synthetic reservoir grid and vary agent quality spread, rare strong agents, task-specific complementarity, evaluation noise, and deployment team size separately. Compare fixed budget-scaled `K`, always-search, and existing adaptive rules using saved simulation machinery. Treat these as model implications, not extra empirical replications.

## Experiment 4: explain success and efficiency from saved trajectories (no new trials)

**Question.** Are apparent agent differences driven by completion speed, task-specific knowledge, timeouts, or genuinely different successful routes?

**Protocol.** Rejoin frozen outcomes to logs by `(pool, arm_id, task_id, replicate)`. On failed-first/successful-second cases, tabulate task tier, timeout, steps, tokens, cost, and verifier result. For GLK, compare the registered 30-step outcome with the uncapped outcome on the **same** trials to test the completion-speed explanation. Review a small, prespecified sample of full traces with a blinded coding rubric, if artifacts are complete enough. Do not interpret post-hoc trace categories as causal mechanisms. Report any missing or clipped traces.

## Optional external validation before buying new API trials

Search for a public dataset with a task × agent outcome matrix, independent repeat attempts, frozen evaluation criteria, and usable provenance. Audit its schema before committing to it. Public multi-agent *code* alone is insufficient: the released [Agent-Scaling repository](https://github.com/SAIL-Research-Lab/Agent-Scaling) provides a runner, while its README does not establish that a ready-to-analyze outcome matrix is included. A compatible public matrix would test generality at no API cost, but differences in tasks and agents require separate reporting rather than pooled estimates.

## Optional fresh-task experiment, only if the thesis needs a stronger claim

Freeze pair-selection rules using the existing data; choose new tasks before seeing outcomes. On every selected task, randomize order and run matched two-attempt conditions: (A) same agent twice, (B) two distinct agents selected for marginal strength, (C) two agents selected for predicted complementarity. Keep model, tools, task reset, time/step cap, and total token budget matched. Predeclare one primary difference: C minus A in verified `at least one succeeds` task coverage; B vs A separates diversity from second-agent strength. Sample size and spending limit must come from a prospective task-cluster power calculation and a small cost pilot, not a guessed episode count. A genuine communication claim needs an additional arm where the second agent sees the first agent's output, with the extra context cost matched and a verifier that selects a usable result.

## Order and decision points

1. **Now:** freeze a provenance ledger and scripts for Experiments 1–4; produce one task-level results table and one figure for each. Draft the thesis around Claim 1.
2. **After the offline readout:** seek advisor feedback on whether independent-agent portfolios are an acceptable multi-agent scope and whether the existing uncertainty is acceptable for the thesis.
3. **If external validity is the main concern:** audit public outcome matrices first. Use one only if its trial-level data support the same estimands.
4. **If interaction is essential:** cost and power a small fresh-task trial, then obtain authorization before any paid collection. Do not spend on more prompts evaluated on the same 30 known tasks merely to tighten a thesis claim about generalization.

The contribution must be positioned against prior work on [infinite-arm simple regret](https://proceedings.mlr.press/v37/carpentier15.html), [agent diversity and scaling](https://arxiv.org/abs/2602.03794), and [interactive multi-agent evaluation](https://aclanthology.org/2025.acl-long.421/). The distinctive result would be a budget-aware empirical account of **when additional prompt-conditioned agents have transferable, complementary value** and when evaluation noise makes searching for that value unproductive.
