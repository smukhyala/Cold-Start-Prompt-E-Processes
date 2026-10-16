# Methods

We preregistered a sequential comparison of fixed baseline and explorer prompts
on independent draws with replacement from the 120-task public Linear bank, using
`gpt-5.4-mini-2026-03-17`. The maximum was 120 pairs, the meaningful improvement
margin was 0.15, and separate efficacy and futility error budgets were 0.025,
with evidence thresholds 40. An infrastructure interruption ended the original
run after 13 valid pairs. A subsequent authorized amendment permitted one retry
of the failed challenger while retaining its saved baseline and all earlier
pairs. The unchanged evidence implementations were updated after each completed
pair, and new scheduling stopped automatically at the first threshold crossing.

# Results for Section 6

The authorized amended continuation first crossed the practical-futility
boundary at 35 completed pairs (evidence 41.266; threshold 40); efficacy evidence
was 1.000. Baseline and challenger successes were 20/35 (57.14%) and 16/35
(45.71%), respectively, a descriptive difference of −11.43 percentage points.
The controller stopped online before pair 36, leaving 170 planned agent
executions unstarted. Across both sessions, 71 attempts produced 70 valid
outcomes and $4.91471 in known API charges, with one original unanswered request
bounded by an additional $0.318432. This demonstrates actual cessation of
inference at a statistical boundary on the amended trajectory; population-level
error control remains conditional on the recovery mechanism preserving the
required null assumptions.

# Limitations addition

The full trajectory was not executed under an unchanged preregistration: the
single retry was authorized after outcomes and a partial failed-episode trace
were available. Reusing the earlier baseline while rerunning only its challenger
can alter the conditional paired-outcome distribution. Host sleep explains the
recovery request but does not establish that the missingness or retry selection
was noninformative. Nominal directional error guarantees require the efficacy
and futility conditional null assumptions to remain valid under this recovery
mechanism, application resets, and provider behavior. We therefore make no
unconditional confirmatory error-control claim for the amended experiment.
Public finite-bank exposure, fixed arm order, temporal separation, unverified
backend stability, and one unresolved request charge further limit inference.

# Figure caption

Natural-log efficacy and practical-futility evidence over all 35 completed
pairs in the authorized amended Linear continuation. The gray region marks the
original 13-pair prefix, retained without restarting evidence. A single authorized
retry completed pair 14 using its saved baseline. The horizontal dotted line is
log(40), with directional error budgets 0.025 and margin 0.15. Futility first
crossed at pair 35 (vertical dash-dot line), where scheduling stopped online;
efficacy did not cross. Curves use saved real outcomes. The amendment followed
an infrastructure stop, so nominal inference additionally requires the stated
recovery assumptions.

# Supported claims

- The controller stopped online at the first futility crossing on the amended trajectory.
- Pair 36 and all later pairs were never scheduled: 170 planned task episodes were avoided.
- The original failed attempt and its authorized retry are both included in the 71 attempts.
- The $11.86985 projected cost of unexecuted work is hypothetical, not realized dollar savings.
- The original preregistered run remains unresolved at its infrastructure endpoint.
