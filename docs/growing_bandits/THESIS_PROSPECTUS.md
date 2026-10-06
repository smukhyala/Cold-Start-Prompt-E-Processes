# Thesis prospectus: budgeted search and deployment of prompt-conditioned agents

*Working draft, 2026-10-06. Scope decision: proceed with independent-agent portfolios using existing trials. Advisor feedback can improve the draft but is not a prerequisite for writing it. All retrospective analyses are labeled exploratory.*

## Working title

**When Does Another Agent Help? Budgeted Search, Selection, and Deployment of Prompt-Conditioned Web Agents**

## Research question

A system can spend a limited budget trying new candidate agents, measuring existing candidates more carefully, or making multiple attempts at deployment. When does each choice improve task success? Which measurable properties of the candidate pool determine whether broader search or a second distinct agent is useful?

An *agent* in the empirical work is one frozen prompt-conditioned instance of the same web-agent scaffold and model. A multi-agent *portfolio* runs such agents independently on the same task. Communication and coordination are outside the empirical estimand of this thesis.

## Proposed thesis claim

The value of a broader agent search depends on stable, transferable differences among candidate agents. Existing synthetic environments show why rare strong candidates make exploration valuable and why an inexpensive search-then-refine schedule can capture much of that value. In three newer real prompt pools, measured differences are small relative to evaluation noise: training winners fail to transfer, broader fixed-budget audits have little clear benefit, and a distinct second agent adds only a small, uncertain amount of potential coverage over a rerun. An older GitLab pool with larger transferable quality differences supplies a contrasting case in which selection helps.

The claim is conditional on the measured candidate pools and tasks. It does not say that agent diversity never helps or that an interacting multi-agent system was tested.

## Contributions

1. **Budgeted decision framework.** Put candidate breadth, evaluation depth, and deployment team size into one resource-allocation problem. Separate the value of discovering a strong candidate from the value of estimating or combining candidates already found.
2. **Real-pool measurement.** Use prompt-by-task outcome matrices and randomized same-cell reruns to measure transferable agent quality, execution noise, and the incremental potential coverage of a distinct agent.
3. **Selection tests.** Evaluate individual and portfolio selectors on held-out tasks at matched training-episode budgets, including random, top-marginal, and coverage-based choices.
4. **Boundary and mechanism.** Show the contrast between three flat newer pools and one older pool with a stronger selection signal; use saved steps, timeouts, costs, and the GLK 30-step rescore to explain what the measured outcome can and cannot mean.

## Core evidence already available

| Question | Evidence | Current readout |
|---|---|
| Does a distinct agent beat a same-agent rerun at two attempts? | 145 G, 155 F, and 120 GLK30 randomly sampled repeat cells | +1.4, +1.1, +2.1 percentage points; all task-bootstrap intervals include zero |
| Does choosing the top two agents transfer? | 500 reused, stratified half-task splits per pool | Top-two minus random pair: G −3.8, F −3.5, GLK30 −1.5, older GitLab +7.2 points |
| Does explicit complementarity selection help? | Same splits and training cells as top-two | Greedy coverage adds 0.0, +1.7, +0.8, −1.1 points over top-two respectively; no robust new-pool advantage established |
| Does more breadth help at a fixed audit budget? | Replayed 48- and 96-episode training audits | No stable large gain in newer pools; older GitLab benefits from selection |
| Does the outcome budget matter? | Same 1,200 GLK episodes rescored at 30 steps | 61.8% at 30 steps versus 75.7% as collected; 166 late successes |

The 500 splits reuse the same task bank and do not represent 500 independent experiments. The older GitLab pool uses a different regime and has no same-cell rerun control. The two-attempt coverage metric uses an oracle `at least one succeeds` score and therefore gives potential coverage, not realized performance of a verifier-equipped team.

## Thesis structure

1. **Introduction.** The practical budget decision and the candidate-pool premise. State the empirical estimands before the results.
2. **Related work.** [Infinite-arm simple regret](https://proceedings.mlr.press/v37/carpentier15.html) and best-arm identification; search-versus-refine rules; [multi-agent diversity and scaling](https://arxiv.org/abs/2602.03794); [interactive multi-agent evaluation](https://aclanthology.org/2025.acl-long.421/). Explain precisely how this thesis differs from each.
3. **Framework and synthetic experiments.** Budget, candidate reservoir, selection rule, deployment team size, and regret/coverage outcomes. Present the existing search-then-refine and heterogeneity results without claiming a new general bandit theorem.
4. **Real web-agent candidate pools.** Task banks, prompt families, scoring caps, frozen outcomes, randomized repeats, noise estimation, and provenance.
5. **Retrospective selection and portfolio experiments.** Task-held-out individual selection, fixed-budget breadth, same-versus-distinct second attempt, team selection, and cost/step diagnostics.
6. **Discussion.** Conditional decision rule, flat-pool boundary, what remains unidentified about communication, independent validation, and candidate generation.

## Figures and tables to draft first

1. One schematic of the two budgets: development trials and deployment attempts.
2. Existing synthetic regret-versus-breadth/heterogeneity plot from the deployment study.
3. [Agent portfolio overview](../../results/growing_bandits/retrospective/agent_portfolio/agent_portfolio_overview.pdf): two-attempt gain, held-out pair selection, and fixed-budget breadth.
4. Table of the four empirical pools, tasks, prompts, repeat counts, and scoring regimes.
5. A concise limitations table identifying which claims need fresh tasks, an actual verifier, or interacting agents.

## Immediate writing sequence

1. Draft Chapters 4 and 5 from the frozen methods and [retrospective report](AGENT_PORTFOLIO_RETROSPECTIVE.md), preserving every exploratory/registered distinction.
2. Draft the framework and synthetic chapter from the already-run [deployment results](DEPLOYMENT_RESULTS.md).
3. Write the introduction around the conditional question and the contrast between flat and heterogeneous pools.
4. Perform a claim-by-claim citation and number audit; export a full thesis draft for review.

No new paid trial is required for this scoped thesis. A public independent outcome matrix would strengthen external validity if one with compatible trial-level data is accessible. A claim about communication would require a separate controlled experiment.
