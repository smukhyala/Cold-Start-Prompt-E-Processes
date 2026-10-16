# Methods

We preregistered one prospective comparison of fixed baseline and explorer prompts
using a browser agent with OpenAI `gpt-5.4-mini-2026-03-17`. We drew 120 tasks
independently with replacement from the complete 120-task public Linear
account-settings bank, retaining duplicates and resetting application state before
each episode. Baseline and challenger ran sequentially on identical tasks. After
each valid pair, a durable controller updated the existing discordant-outcome
efficacy process and paired-difference futility mixture, with meaningful gain
0.15, directional error budgets 0.025, and evidence thresholds 40. The controller
was configured to stop new scheduling at the first statistical boundary, the task or $75 budget,
or an invalid episode. Protocol and task commits preceded all study inference;
85 automated tests passed before execution.

# Results

The experiment halted automatically after a request timeout in pair 14, leaving
13 complete pairs. Baseline and challenger successes were 8/13 and 7/13,
respectively, a descriptive difference of −7.69 percentage points. Efficacy and
futility evidence were 1.00 and 2.136, below the threshold of 40; neither boundary
crossed. The run attempted 28 episodes and recorded $1.76231 in known API charges,
with one unacknowledged request bounded by an additional $0.318432 under the
registered pricing assumptions. Host sleep overlapped that request. The remaining
212 planned episodes were not started because of the infrastructure halt. This
operationally incomplete run verifies live evidence updates and automatic failure
handling, but demonstrates no inference savings attributable to statistical
early stopping.

A subsequent resume audit confirmed that the frozen protocol required termination
at the invalid half-pair. No resumed inference was performed, and the 13-pair
trajectory and statistical conclusions are unchanged.

# Figure caption

Natural-log efficacy and practical-futility evidence from the 13 complete pairs
in the preregistered live Linear evaluation. The horizontal dotted line is the
registered threshold log(40), with directional error budgets 0.025 and practical
margin 0.15. Neither process crossed. The vertical line marks the last complete
pair before an infrastructure halt during pair 14; it is not a statistical
stopping boundary. The maximum planned budget was 120 pairs. Curves use saved
real outcomes only; the incomplete pair is not imputed or included.

# Scope of supported claims

- The controller updated existing evidence processes online after every valid pair.
- The implementation passed synthetic first-boundary and no-further-call tests.
- The real run automatically stopped scheduling after an operational error.
- No real statistical crossing or statistical early-stopping savings was demonstrated.
- Any directional validity guarantee requires the registered conditional sampling
  and stability assumptions. This run makes no efficacy or futility rejection.

The exact total charge and total token usage are unknown for one request. Dollar
projections for unexecuted work are hypothetical and must not be called realized savings.

# Limitations addition

The registered infrastructure rule ended the experiment before either a statistical
decision or the maximum evaluation budget. The saved-half resume mechanism did not
permit retrying the finalized invalid challenger episode. Consequently, this run
does not empirically demonstrate statistical early-stopping savings. One request's
charge and token usage remain unknown, and the required conditional sampling and
backend-stability assumptions were not independently established.
