# Agent portfolios from the trials already collected

*Exploratory analysis, 2026-10-06. No agent run or paid API call was made. This analysis was designed after the local outcomes were known. It uses the same 30 Gmail and GitLab tasks as earlier reports; its many task partitions are repeated views of those tasks.*

## Question and data

When a deployment budget allows two or three attempts on a task, does it help to use distinct prompt-conditioned agents? Can a training-task audit select a useful agent or team, and does a wider candidate search repay its evaluation cost?

Sources are the frozen G and F Gmail grids (50 prompts × 30 tasks each), GLK30 GitLab grid (40 × 30), and the older GitLab paired study (17 non-cookbook prompts × 40 tasks). G/F have 145/155 randomly sampled same-prompt repeat cells; GLK30 has 120. The older GitLab study has no same-cell repeats and uses a different agent regime. The script checks the frozen snapshots against their manifests, joins recorded costs and steps to the live logs, and records hashes of every input file. GLK30 is a 30-step **rescore of the collected GLK episodes**, not another sample. G and F share a task bank, so they are not independent replications.

An agent here is a frozen prompt-conditioned instance of one underlying agent. The `at least one succeeds` outcome is **oracle potential coverage**: it assumes a perfect way to identify a successful attempt. No agents communicated, jointly acted, or used a deployed verifier.

## 1. Two attempts: different agent versus same-agent rerun

For each randomly repeated `(agent, task)` cell, compare the first run plus its recorded rerun with the first run plus a uniformly sampled *different* agent's main outcome on the same task. We average over all eligible second agents, then over repeated cells. The task-bootstrap intervals resample the 29–30 tasks represented among the repeat cells, conditional on these fixed prompt pools and task banks. They are exploratory uncertainty summaries, not confidence intervals for arbitrary future domains.

| Pool | Same agent twice | Two distinct agents | Gain [95% task bootstrap] | Repeated cells |
|---|---:|---:|---:|---:|
| Gmail G | 69.0% | 70.4% | +1.4 pp [−1.1, +4.2] | 145 |
| Gmail F | 71.0% | 72.1% | +1.1 pp [−1.4, +3.8] | 155 |
| GitLab GLK30 | 68.3% | 70.4% | +2.1 pp [−0.6, +4.8] | 120 |

The second agent's marginal success differed from a rerun by only +0.6, +0.3 and −0.9 pp respectively. This decomposition suggests the point gains mainly reflect different failures, but every coverage-gain interval includes zero. A G/F mixed-family second agent gave about 0.2–0.3 pp *less* coverage than a within-family second agent on the same repeated anchors; the intervals include zero. These findings align with the prior same-versus-different disagreement analysis, rather than supplying an independent experiment.

The recorded second-attempt costs are close in the Gmail pools. After an anchor failure, a G repeat cost $0.147 on average versus $0.144 for another G prompt's run on that task; F was $0.158 versus $0.153. These are observed full-run costs, not randomized cost effects. GLK30 costs reflect the uncapped collected run, so they cannot price a real 30-step deployment directly.

## 2. Choosing a team on training tasks

In 500 seeded, difficulty-stratified half-task partitions per pool, every selector sees the **same** full candidate-by-training-task grid. We compare the two highest marginal training success rates, greedy training coverage, and a fixed shrinkage rule that pulls pair coverage toward the independence prediction. We evaluate fixed selected teams of one, two and three agents on the held-out half. The random-team baseline is computed exactly over all teams in the pool. No test outcome enters the selection rule. The 500 partitions are not 500 new trials or a prospective holdout.

| Pool | Random pair coverage | Top-two coverage | Greedy coverage | Top-two minus random | Greedy minus top-two |
|---|---:|---:|---:|---:|---:|
| Gmail G | 69.8% | 66.0% | 66.0% | −3.8 pp | +0.0 pp |
| Gmail F | 69.8% | 66.3% | 68.0% | −3.5 pp | +1.7 pp |
| GitLab GLK30 | 72.3% | 70.9% | 71.6% | −1.5 pp | +0.8 pp |
| Older GitLab17 | 67.7% | 74.8% | 73.7% | +7.2 pp | −1.1 pp |

The shrinkage selector made the same choices as greedy coverage in these runs. Three-agent results follow the same broad pattern: neither selector reliably beats a random triple in the newer pools, whereas marginal selection helps in older GitLab. Because a third agent means another attempt, these team-size curves must not be presented as a diversity advantage over two attempts.

Leaving each task out once and training on all other tasks is a sensitivity check. It makes the newer-pool selection deficits larger in several cases; with only 30 tasks and overlapping training sets, those values are unstable and are not a separate validation sample. The older GitLab positive transfer is directionally consistent with the earlier one-agent [task-transfer analysis](RETROSPECTIVE_TASK_TRANSFER.md). Its different prompt construction and tasks preclude treating it as a controlled domain replication.

## 3. Search breadth at fixed training-episode budgets

For each reused task split, we shuffle candidates once and use nested, difficulty-balanced training-task subsets. A budget of 48 observed training episodes compares `(K, tasks per candidate) = (4, 12), (8, 6), (16, 3)` in three-tier pools. Budget 96 compares `(8, 12), (16, 6), (32, 3)`. Older GitLab has only two tiers and 17 candidates, so only compatible schedules are included. Each policy selects the highest observed training-rate agent or top two, then is evaluated on the held-out tasks. This is a replay over the completed matrix; it excludes candidate-generation cost and cannot simulate new adaptive trajectories.

At budget 96, held-out **single-agent gain over a random candidate** by K was:

| Pool | K=8 | K=16 | K=32 |
|---|---:|---:|---:|
| Gmail G | −0.7 pp | −0.3 pp | +0.3 pp |
| Gmail F | −2.5 pp | −2.8 pp | −3.8 pp |
| GitLab GLK30 | −0.1 pp | +0.5 pp | +2.2 pp |
| Older GitLab17 | +5.7 pp | +6.1 pp | unavailable |

The new pools show no stable, substantial payoff to buying breadth at the same observed episode budget. The older GitLab pool supports search when transferable candidate differences are larger. These are descriptive averages over reused partitions; no interval here should be read as independent-task confirmation. The pilot covariance in the earlier task-transfer report separates the older pool from the newer ones on average but does **not** predict which within-pool split benefits from broadening search. Four known pools cannot validate a deployable diagnostic.

## 4. Saved-run mechanism and efficiency checks

The frozen outcomes reconcile with the live G/F logs; every main trial has a saved `history.json` trace. Failures are far more expensive than successes in all three pools. G/F successful trials average about $0.054 and have median 6 steps; failures average $0.170–$0.171 and have median 23–24 steps. About one third of G/F failures hit the wall clock. This makes *cost per successful outcome*, not just episode count, a relevant follow-up objective.

For GLK, the same 1,200 main episodes score 61.8% at 30 steps and 75.7% as collected: 166 of the as-collected successes occurred too late to count at 30 steps. This supports completion speed as an important explanation of the cap-sensitive GLK result. It does not show that a different prompt causes faster completion, and as-collected costs are not 30-step costs. Among the 50 GLK repeat cells whose first run failed under the cap, the rerun succeeds 24.0% of the time; a different GLK prompt's main run succeeds 29.1% on average. The small number of discordant cases, especially within difficulty tiers, limits trace-level interpretation.

The saved histories make a later blinded qualitative trace review possible. This report uses recorded success, steps, timeouts and costs; it does not assign post-hoc failure-mechanism labels to individual histories.

## What this means for the thesis

The defensible empirical result is **conditional**: in the three newer, flatter prompt pools, a broader search or a more elaborate diversity selector had little transferable value, while an older pool with a stronger quality signal did reward selection. Synthetic results explain *why* a wider search can be valuable when rare strong candidates exist; the real grids show that the premise must be measured rather than assumed. The multi-agent part is a budgeted portfolio of independent prompt-conditioned agents, with a measured same-agent rerun baseline.

This is a stronger thesis narrative than a generic claim that diverse agents help. It remains short of a claim about agent interaction, coordination or a practical team whose successes can be selected online. An independent public outcome matrix would strengthen external validity; a new communication experiment would be required for the interaction claim. The [Agent-Scaling code release](https://github.com/SAIL-Research-Lab/Agent-Scaling) does not by itself provide a ready-to-use outcome matrix, and the promising [CMU trajectory dataset](https://huggingface.co/datasets/cx-cmu/agent_trajectories) is access-gated and filters incomplete attempts, so neither was silently counted as a replication. Public [SWE-bench submissions](https://github.com/SWE-bench/experiments) may support a separate cross-system coverage study, but model/scaffold differences and missing same-system reruns would change the estimand.

## Reproduction

From the repository root:

```bash
./.venv/bin/python experiments/growing_bandits/retrospective/agent_portfolio.py
./.venv/bin/python experiments/growing_bandits/retrospective/plot_agent_portfolio.py
./.venv/bin/pytest -q tests/test_agent_portfolio.py tests/test_task_transfer.py
./.venv/bin/ruff check experiments/growing_bandits/retrospective/agent_portfolio.py experiments/growing_bandits/retrospective/plot_agent_portfolio.py tests/test_agent_portfolio.py
```

Tables, split-level rows, source hashes and the figure are in `results/growing_bandits/retrospective/agent_portfolio/`. The primary visualization is `agent_portfolio_overview.pdf` (and `.png`).
