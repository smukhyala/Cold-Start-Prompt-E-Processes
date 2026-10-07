# Thesis and experimental roadmap

*Research status: 2026-10-07. This report distinguishes observed results, the prepared Gmail pilot, and experiments that have not yet been run.*

## Executive assessment

The current thesis asks how to allocate a limited evaluation budget among **trying more candidate agents**, **measuring existing candidates more carefully**, and **making another attempt on a deployment task**. In the empirical studies, an agent is a frozen prompt applied to the same underlying web-agent scaffold and model. A two-agent portfolio consists of independent attempts; no collected trial establishes agent communication or coordinated action.

The strongest supported claim is conditional: **search breadth matters when candidate quality differences are large enough to transfer across tasks; when a prompt pool is flat relative to evaluation noise, broad search and elaborate portfolio selection have little demonstrated value.** A second agent has possible task-level coverage value, but using it requires a decision signal available before the hidden success verdict. The existing Gmail traces show that recognizing a timeout is useful; recognizing which apparently completed task is *recoverable* remains unresolved.

This is a defensible empirical thesis, with a clear limitation. A new general bandit algorithm, a validated multi-agent routing rule, and an interacting-agent advantage are **not** current findings. The mathematical search tradeoff overlaps [infinite-arm simple regret](https://proceedings.mlr.press/v37/carpentier15.html); [TRIPLE](https://arxiv.org/abs/2402.09723) already applies fixed-budget best-arm identification to prompt selection; and oracle portfolio coverage is familiar from [algorithm selection](https://link.springer.com/article/10.1007/s10994-022-06161-4). The possible contribution is a careful, budget-matched empirical account of **when these decisions pay off for web agents and when oracle portfolio gains cannot be realized**.

## What is already known

| Question | Current evidence | Defensible conclusion |
|---|---|---|
| Does a training winner transfer? | Three newer G/F/GLK30 prompt pools; an older GitLab pool | Winners and broad audits transfer weakly in the newer pools. The older pool has a stronger selection signal. Reused task splits are exploratory, not independent replications. |
| Does a different second prompt beat a same-prompt rerun? | 145 G, 155 F, 120 GLK30 randomly sampled repeat cells | Oracle two-attempt gain of about **+1.4, +1.1, +2.1 percentage points**, respectively; all task-bootstrap intervals include zero. This is potential coverage, not a deployed selection policy. |
| Can a simple observable rule use a second run? | Saved G/F Gmail logs | Retrying after apparent non-completion raised success **+2.1 to +2.6 points** but also raised cost per successful task. Switching prompts had no reliable advantage over rerunning. |
| Can trace metadata find false completions worth retrying? | 264 completed first-run anchors | A failure predictor reached grouped-task AUC about **0.63**; retrying its highest-risk tasks harmed success. A model targeting retry benefit produced only **two net additional successes**, too little to validate a policy. Generic verification claims and read actions were not reliable verifiers. |
| Is the conditional premise plausible elsewhere? | Public [WorkBench Revisited item outcomes](https://github.com/LucasEtteldorf/workbench) | Large **cross-model** quality differences transferred across held-out task templates; a selected single model covered 97.4% and the top two 98.7%. This is a useful external contrast, not a replication of prompt-level or interacting-agent results. |

The main retrospective analyses and caveats are documented in [agent portfolios](AGENT_PORTFOLIO_RETROSPECTIVE.md), [task transfer](RETROSPECTIVE_TASK_TRANSFER.md), [free external checks](FREE_PORTFOLIO_CHECKS.md), and [first-run triage](FIRST_RUN_TRIAGE.md). G and F share the same 30 tasks. GLK30 is a 30-step rescore of collected GitLab runs, and the older GitLab experiment uses a different regime.

## Immediate experiment: the prepared Gmail pilot

The next batch is **90 isolated runs on 30 Gmail task IDs**: G_14 first, an independent G_14 rerun, and an independent G_16 attempt on every task. Both prompts tied at 70% on the old G task grid and are frozen before the new cells are scored. The 30 IDs were omitted from the main G/F grid, though some had archived pilot results for *other* prompts. No G_14 or G_16 run exists on these IDs. These are new prompt–task cells within familiar task families, not genuinely new task templates.

The prepared queue, prompt hashes, rule coefficients, and analyzer are in [FIRST_RUN_PILOT_READY.md](FIRST_RUN_PILOT_READY.md). Its **primary comparison** is a frozen old-data retry-benefit rule versus retrying the same agent only after observable non-completion. The rule uses first-run steps, cost, elapsed time, response length, and error count. Secondary comparisons are timeout-triggered G_16 versus G_14, stop-after-first, and oracle pair coverage. The analyzer evaluates verified success of the selected trajectory **and cost per task**; it never feeds the hidden success verdict into a decision.

| Possible pilot result | What it would indicate | What it would not establish |
|---|---|---|
| Timeout-triggered retry helps on these 30 IDs | The simple triage signal transfers to new prompt–task cells. | That retry improves successes **per dollar** or works on new task families. |
| G_16 beats G_14 after an incomplete first run | Different prompts can help for this particular frozen pair. | A general diversity advantage; this is one pair and a small sample. |
| Frozen benefit rule beats timeout-only on success **and** an explicit cost criterion | Preliminary evidence that recoverability can be predicted from first-run metadata. | A validated routing rule; a few rescued tasks can move the 30-task result substantially. |
| Benefit rule is null or harmful | The old-data signal does not transport even to these related task IDs. | That no task-specific verifier or better candidate pool could work. |
| Oracle pair coverage rises but policies do not | Choosing successful attempts is the bottleneck. | A realized benefit from deploying two agents. |

The batch is a **feasibility and large-effect pilot**. The observed 1–3 point differences from the older work are smaller than this 30-task design can resolve convincingly. Do not treat one positive point estimate as a publishable confirmation. Report all 30 task rows, paired contrasts, task-bootstrap intervals, retry frequency, total inference spend, and any infrastructure failures. If the collector reaches its cap before all triples are complete, report the partial collection status; do not analyze a convenient subset as the full pilot.

### Relaunch from a normal Terminal

Sanjay authorized the **$12 cumulative cap**. Historical costs for this exact pair imply about **$8.64** for 30 triples; the cap can stop collection early. One worker keeps episode order within each task and limits cap overshoot to an episode already in flight. The prepared batch has not produced any task record: a previous attempt from the Codex sandbox failed before inference because that environment denied the local Gmail server permission to bind port 8001. It reported **$0.00 spent**. The existing `failed` status and empty isolated log directory do not prevent a normal-terminal relaunch.

From the repository root, keep the machine awake and run:

```bash
cd /Users/sanjay/projects/BerkeleyProjects/Research/IanWaudbySmith/coldStartPrompts
./.venv/bin/python experiments/growing_bandits/empirical/collect.py \
  --data data/first_run_pilot_gmail \
  --log-dir logs/first_run_pilot_gmail \
  --budget 12 --workers 1
```

The collector resumes from any terminal task records if interrupted. `STATUS=done` means the queue finished; `STATUS=budget` means it stopped at the cap; `STATUS=provider_down` needs investigation before any restart. The existing analysis command is:

```bash
./.venv/bin/python experiments/growing_bandits/retrospective/analyze_first_run_pilot.py
```

It intentionally refuses to emit a full result unless all 90 required cells have scored terminal records. After the run, bring the status and logs back for a quality audit and interpretation. Do not increase the spending cap automatically if collection stops early.

## Experiments after the pilot, in priority order

### 1. Independent validation of a *decision rule*

**Question.** On genuinely new task templates, can first-run evidence identify cases where rerunning or switching agents improves verified success enough to justify the added cost?

**Design.** Freeze one policy using only prior trials, then collect first and randomized same/different second attempts on a larger task bank with template-level separation between development and evaluation. Include apparently completed first runs, because timeout-only cases cannot test false-completion recovery. Use the same model, scaffold, task reset, step and clock limits, and scorer in every arm. The primary endpoint should be the paired difference in **verified selected-trajectory success and cost per task** versus the timeout-only rule. An explicit value-per-success or cost ceiling is required to declare a policy preferable; more success alone is insufficient when extra runs consume money.

**Interpretation.** A positive held-out policy gain with acceptable cost would support a modest novel agent-allocation result. A null or negative result would support the present measurement thesis: apparent portfolio opportunity can disappear when selection must be made from observable evidence. New task templates and enough task clusters are essential; repeating the current 30 IDs with more seeds would mostly reduce execution noise while leaving generalization unresolved. Size and budget this study from the pilot's observed paired differences, not from the earlier oracle-coverage power estimate.

### 2. A task-specific, decision-time verifier

**Question.** Can the agent's browser state supply evidence of actual completion when the agent says “done”?

**Design.** Define a verifier that can run **before** the hidden benchmark scorer and without reading its ground truth or `verifier_message`. For Gmail, this may require task-specific DOM assertions or an independently specified target-state check; record its runtime and any API cost. Freeze its parser and abstention behavior on development tasks, then evaluate false-positive and false-negative rates on new tasks. Compare stop, timeout-only retry, and verifier-guided retry or result selection at matched budgets. If the verifier only works because it duplicates the benchmark's private scoring rules, label it as a benchmark-specific engineering demonstration.

**Interpretation.** Recovering a meaningful part of the gap between selected-trajectory success and oracle pair coverage would turn a potential portfolio effect into an operational one. If the verifier cannot distinguish false completions, more independent agents alone do not solve the deployment decision.

### 3. A controlled interaction experiment, only if the thesis needs a communication claim

**Question.** Does a second agent **use the first agent's trace** to recover tasks better than an independent rerun or independent different prompt?

**Design.** On new tasks, randomize a matched-cost second attempt among same-agent rerun, independent different agent, and a different agent given a bounded summary of the first trace. Keep first-run prompts and task distribution fixed, cap tokens/steps and wall clock consistently, and use isolated resets or a precisely specified shared-state handoff. Record whether the second agent changes its action because of the trace and whether any first-run side effect persists. Evaluate verified success, cost, and unintended actions.

**Interpretation.** Only an advantage over the independent second-agent control at comparable compute supports an *interaction* claim. The current pilot contains no communication condition and cannot answer this question.

### 4. A pool-level search-breadth experiment, if the search result is the paper's center

**Question.** Can a small pilot predict which candidate pools reward a larger search breadth (K)?

**Design.** Construct multiple independent prompt pools with deliberately varied candidate-generation methods and task families. On each pool, reserve tasks by template, spend an equal development-episode budget at small and large (K), then compare held-out best-prompt performance. Measure stable between-agent variance and rare-high-quality tail mass *before* choosing breadth. The existing three newer pools plus older GitLab contrast are too few and too confounded to validate a pool-level classifier. More episodes in one flat pool would not provide the needed variation across pools.

**Interpretation.** If a prespecified diagnostic predicts which new pools benefit from breadth, that could support a stronger search-versus-refine contribution. If it does not, the existing conditional descriptive result remains the honest claim.

## Decision rule for the thesis draft

Draft now around the conditional empirical finding. The 30-task pilot should be reported as a prospective **check of one frozen pair and one frozen retry policy**, regardless of whether it is positive or negative. Promote the claim only after independent tasks support the appropriate comparison:

1. **Current thesis:** budgeted candidate search and independent-attempt portfolios are useful only when quality or recovery differences are stable and observable; the measured newer prompt pools do not establish that premise.
2. **Stronger allocation paper:** a fixed, decision-time rule improves verified success at a defensible cost on new task templates.
3. **Multi-agent interaction paper:** a trace-conditioned second agent beats independent controls under matched resources.

The immediate pilot can inform step 2 but cannot by itself complete it. Its most valuable outcome may be telling us whether to invest next in **better evidence for recoverability**, **better candidate generation**, or **writing the current scoped thesis without another expensive study**.
