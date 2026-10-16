# Prospective live-stopping Linear experiment

**Preregistration date: 2026-10-16.** This document, sampler, controller, and
execution wrapper are frozen in Git before drawing the study seed or requesting
any study inference. The realized seed, task list, configuration, package versions,
source hashes, and protocol commit are subsequently frozen in
`data/prospective_live_stopping/{config,manifest,tasks}.json`. This is a local,
timestamped preregistration, not an externally registered clinical trial.
The starting repository revision is `2c49fe228ea86f2cbcef0bd0fb6e80b832a4d912`.
Existing unrelated working-tree changes and `paper/main.tex` are not modified.

## Question and design choice

Can a precommitted sequential comparison stop real agent evaluation at its first
registered statistical boundary and avoid scheduling the remaining executions?
Success means faithful execution, including an unresolved or operationally
incomplete result. The primary outcome is the first statistical decision and its
completed-pair count; secondary outcomes are success rates, paired difference,
discordances, measured resources, and actual planned executions avoided.

There is one fixed baseline and one fixed challenger. Use the complete public
120-task WebArena Infinity Linear account-settings bank as a discrete task
population. Historical feasibility evidence is 64/120 baseline successes and
66/120 challenger successes in the fixed-bank trial. The generated profile-edit
population had only 3/120 and 1/120 successes and is unsuitable for demonstrating
comparison between capable agents. No task is selected, excluded, or weighted
using past success labels. The choice uses capability and implementation
feasibility, not a search over stopping times. No new pilot outcomes are obtained.

## Sampling and paired execution

Sort all 120 source tasks lexicographically by ID. After committing this protocol,
draw one 128-bit seed from operating-system entropy. NumPy PCG64 draws 120
independent uniform integer indices with replacement. Retain duplicates and order;
assign unique `live_001` through `live_120` execution IDs and preserve source IDs,
instructions, verifiers, seed, and draw index. The generator is
`experiments/growing_bandits/empirical/prepare_live_stopping.py`.

These are fresh executions of public task specifications, not novel tasks or a
permutation without replacement. Presampling is outcome-blind; scheduling never
uses future task identities or outcomes. The inferential filtration reveals tasks
and paired outcomes in draw order, rather than conditioning on the whole future
pseudorandom stream. For each draw, evaluate baseline first and challenger second,
each from reset app state. There is one active agent episode and no prefetched
agent jobs. Browser/server recovery must not carry an episode's app state into
the next. The fixed order defines the tested execution regime; temporal order
effects remain a limitation.

## Frozen prompts and inference

Reuse the rendered `data/prospective_linear/pools/fixed_pair.yaml` prompts:

| Role | Arm | SHA-256 of rendered prompt |
|---|---|---|
| Baseline | `linear_baseline` | `ae8c7c7fb6984ed82d52d8dd454e6672e8d7a1bf544cd727affdb35cb539ef82` |
| Challenger | `linear_explorer` | `5e4ee37361c08c0a0282fc548c5d8f6467ea4fed7c4211252b4440241a05c024` |

Vectors, in planning/verification/agency/expertise/format/goal order, are
`(0,0,0,1,0,0)` and `(1,0,2,2,0,0)`. The latter adds implicit planning, proactive
exploration, and senior expertise; its benefit is uncertain. Both receive the same
browser-use system scaffold and limits. Freeze OpenAI
`gpt-5.4-mini-2026-03-17`, low reasoning effort, maximum 4096 completion tokens
per request, default temperature, no vision, headless browser, 30 agent steps,
180-second agent timeout, and browser-use 0.13.1. Use the standard
`https://api.openai.com/v1` endpoint and default service tier; SDK retries are zero.
No prompt/model/limit substitutions after outcomes begin. The dated model name
reduces alias ambiguity; it cannot establish provider/backend invariance. Returned
model identifiers, request IDs, and fingerprints (when supplied) are logged.

## Rewards, errors, and interruption

Reward is exactly the environment verifier's binary success; partial completion
is zero. At the 30-step limit, score the environment verifier; an unsatisfied
verifier is a task failure. A legitimate 180-second agent timeout is a task
failure. A verifier exception, provider error, harness exception, missing binary
outcome, missing usage, model mismatch, or uncertain paid-call completion is an
operational error, never a success or an imputed zero. A provider refusal that
terminates execution is logged separately from task failure. There are **no
automatic infrastructure/provider episode retries** and no replacement draws.
On the first invalid half-pair, stop the experiment with an unresolved/incomplete
operational status and retain the contiguous complete-pair prefix descriptively.

Persist an execution intent before each paid episode and its receipt immediately
after it finishes; persist pair outcomes, full reconstructible statistical state,
and the terminal decision before scheduling anything else. Resume a saved complete
half-pair without paying for it again. If a crash leaves an intent without a
durable receipt, halt for reconciliation; the system cannot promise exactly-once
execution across an unacknowledged external API call. Never silently replay such
a call. A process lock excludes concurrent collectors. Frozen files and prompt
hashes are checked before executing/resuming.

## Hypotheses, evidence, and stopping

Let `D_t = Y_challenger,t - Y_baseline,t`. Fix `N=120`, `K=1`, `delta=0.15`,
`alpha_E=alpha_F=0.025`. Each evidence threshold is 40, or natural log
`3.6888794541139363`.

Efficacy tests the conditional null that a challenger win, given a discordance
and previous pairs, has probability at most 1/2. Under stable iid paired outcomes
this is equivalent to mean gain at most zero. Reuse exactly
`audit/flatness_eprocess.py:e1b_log_paths`, incumbent row 0, challenger row 1,
`m0=.5`, `eps=1e-9`. Only discordances update the one-sided upward capital.
The initial bet is zero; adaptive bet cap is 1. Ties multiply efficacy evidence
by one. The two-sided hedged capital is not substituted.

Practical futility tests the conditional null
`E[D_t | previous pairs] >= .15`. Reuse exactly
`audit/futility_eprocess.py:futility_log_paths`, including every pair. Transform
`X=(D+1)/2`, `m=.575`. Mix five capital components equally: one adaptive downward
bet, initially zero with cap `1.176470588235294`, and four fixed bets
`[.1739130434782609, .3478260869565218, .6956521739130436,
1.3913043478260871]`. The implementation, including its variance update and
`eps=1e-9`, is authoritative and hashed. The thin controller recomputes each saved
prefix using these existing functions; this is an online decision after each
observed pair, not a retrospective run after the full grid.

After saving each complete pair, check both boundaries. Stop immediately at the
first crossing. If both first cross on the same pair, efficacy has label
precedence and both hits are retained. A boundary takes precedence over reaching
the task or dollar budget on that pair. If neither crosses by pair 120, report
unresolved. A dollar-limit or operational stop before 120 without a crossing is
unresolved with its reason, not statistical futility. No later pairs are run to
obtain another decision. Efficacy and futility are compatible when the true gain
is positive but below .15; futility does not establish equality.

## Resource limit and measurements

The maximum study spend is **$75**, below the user's $100 limit. Historical spend
was $16.50530325 for 241 attempts (one retry): scaling the mean to 240 attempts
projects $16.44. This projection is not a guaranteed bound. An uncapped
full-context calculation for 240 x 30 requests is $2292.7104, and auxiliary
requests mean even that is not a rigorous full-run bound. Therefore execution
requires a tested **per-HTTP-request hard budget guard**, not the existing
collector's between-episode spend check. Completion of 120 pairs is not promised.

At frozen standard rates ($0.75/M uncached input, $0.075/M cached input,
$4.50/M output), reserve $0.318432 before every API request: 400,000 input tokens
at the uncached rate plus 4096 output tokens. The declared context/output limits
bound this reservation conservatively. Never send if cumulative measured spend
plus outstanding reservations plus the next reservation exceeds $75. Persist
raw response usage before parsing; parse failures can still cost money. An
unacknowledged or unmetered response retains its reservation and stops execution.
Any violation of the pricing/token assumptions is an audit failure and stops
execution. No other inference provider or billable tool is allowed. Budget
exhaustion within a half-pair leaves it invalid and stops the run. Model limits
and prices were checked against the [official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
on 2026-10-16. Dollar figures are token-derived API charges, not reconciled
invoice totals or taxes.

Record every episode intent, result, status, costs (including failed requests
when known), input/output/cached/reasoning token counts, model metadata, timestamps,
wall time, task IDs, prompts, verifier messages, and first boundary. Report the
240 planned full-budget executions, actual attempted and valid executions, and
`2*(120-tau)` executions avoided by statistical stopping only when a valid online
decision occurs at complete pair `tau`. Operational nonexecution is reported
separately. Project unexecuted dollar costs from mean observed per-episode cost,
explicitly labeling them hypothetical; do not call them realized dollar savings.

## Validity, verification, and planned reporting

Independent draws with replacement from the fixed finite bank support a
population comparison if resets, agent randomization, verifier behavior, and
provider conditions yield the required conditional paired distribution. PRNG
sampling, stable task code, a snapshot name, and proximity in time do not prove
these assumptions. Public benchmark exposure and historical feasibility-based
design choice limit extrapolation to new tasks, applications, or models.
Conditional on each directional null's assumptions, its false-crossing
probability is at most .025; their union is at most .05. No unconditional claim
is made when those assumptions fail. Stopped empirical success rates are
descriptive and are not presented as unbiased population estimates.

Before paid execution, synthetic tests must pass for strong improvement,
inferiority, near equality, a positive submargin gain, no crossing, nearby/both
boundaries, zero discordances, exact offline parity, provider failure in a
half-pair, safe interruption/resume, ambiguous external completion, dollar
exhaustion, and absence of scheduling after a terminal decision. Test the
transport against mocked responses, including parse errors and missing usage.
Check app availability and credentials without a pilot agent call. If a required
gate fails, do not spend money.

Analyze only saved real outcomes into `results/paper/live_stopping/`, with a
natural-log evidence figure, exact counts/costs and machine-readable audit.
Produce `docs/paper/LIVE_STOPPING_RESULTS.md`, a methods paragraph, a results
paragraph, LaTeX table/caption, and an independent final audit. If no valid pairs
are obtained, do not fabricate a real-outcome evidence figure. All departures,
remaining assumptions, and blockers must be explicit. The main manuscript is
not edited.
