# Authorized continuation after the Linear infrastructure interruption

**Amendment date: 2026-10-16.** This amendment is recorded after the original
13 completed pairs were observed and before any additional study inference.
It does not change or replace `LIVE_STOPPING_PROTOCOL.md`. The original protocol
and its terminal unresolved result remain part of the permanent record.

## Authorization and scope

After the resume review reported that the original no-retry rule prevented
continuation, the user explicitly instructed: “rerun the one it failed on
(my computer went to sleep) and go from there”. This authorizes exactly one
new attempt of the invalid challenger half of pair 14, followed by continuation
of the existing sampled task sequence. It is a post-outcome amendment to the
infrastructure stopping/retry rule, not a new candidate comparison or task sample.

The original run stopped at 2026-10-16T18:50:13.219839Z, with 13 complete pairs,
a valid baseline for pair 14, and an invalid challenger with unknown completion
of request `request_000458`. The baseline and challenger successes among the
complete pairs were 8 and 7, and the evidence values were 1 and 2.1360441229251084.
These outcomes were known when the amendment was made. Host sleep overlapped the
request timeout; the user reports closing/sleeping the computer. This evidence
does not prove that sleep was the only cause of the failure.

## Exactly one authorized retry

1. Preserve the original execution directory, all 28 attempt receipts, request
   ledger, terminal decision, and prior publication/audit artifacts unchanged.
2. Initialize a separate continuation execution directory with the 27 valid
   original episode receipts and intents: 13 complete pairs plus pair 14 baseline.
   The new effective view references the archived invalid challenger; it does not
   erase, recode, or include that invalid outcome in a statistical pair.
3. The next paid episode is only task `live_014`, source `task_h40`, challenger,
   physical attempt `pair_000014_challenger_retry1`. Reuse the valid baseline
   `pair_000014_baseline`; do not execute it again. Reset the application for
   the new challenger attempt, as for every original episode.
4. If valid, combine that challenger with its saved baseline and update both
   evidence processes over all 14 pairs, retaining the original 13-pair prefix.
   There is no evidence restart. Continue `live_015`, then the remaining original
   order, baseline followed by challenger, one active episode at a time.
5. Stop at the first efficacy or futility threshold, with the original efficacy
   precedence; otherwise stop at 120 complete pairs, the cumulative dollar guard,
   or any subsequent infrastructure/validity failure. No further retry is
   authorized by this amendment, including failure of the single retry itself.
6. Durable physical attempt IDs prevent a crash from causing another paid retry.
   A valid saved retry receipt may be reused. An unreceipted request or attempt
   remains ambiguous and halts new execution. No in-flight jobs are prefetched.

## Unchanged design and procedure

The original PCG64 seed `16907f20ad5c78a18cca90759d59838a`, complete 120-task
population, with-replacement draws, task IDs/order, verifiers, prompts and hashes,
model `gpt-5.4-mini-2026-03-17`, low reasoning effort, 4096 output-token limit,
30 agent steps, 180-second agent timeout, 65-second HTTP timeout, zero SDK retries,
browser-use version, and frozen dependencies remain unchanged. No alternative
candidate, outcome-dependent task selection, or additional experiment is allowed.

The maximum is 120 valid task pairs, with delta 0.15, alpha_E = alpha_F = 0.025,
and evidence threshold 40 in each direction. The original `e1b_log_paths` and
`futility_log_paths` implementations and betting settings remain authoritative.
The unchanged core controller recomputes evidence after every complete pair and
must stop scheduling immediately at the first boundary.

## Cumulative budget and accounting

The study-wide cap remains **$75**, covering original and continued inference.
The original known cost is **$1.76230815**. Request 458's entire **$0.318432**
reservation remains reserved, with its actual charge and tokens unknown. Thus
the new transport can authorize at most **$72.91925985** of additional charges
and reservations. An explicit opening-balance record links the new request
ledger to the immutable original ledger. A fresh ledger does not reset spending.
New unknown responses remain reserved and stop the continuation. The original
per-request $0.318432 bound, pricing, and actual-usage logging remain unchanged.

The prior mean among 27 fully accounted valid episodes was $0.064266022222.
The 213 additional episodes needed to reach 120 pairs, including this retry,
project approximately **$13.68866** of further cost. This projection is not a
guarantee; the hard request guard controls actual authorized spending.

Report all original and continued attempts, successful usage responses, tokens,
known charges, unknown-charge bounds, and timestamps. Preserve one failed attempt
and one authorized episode retry in the counts. If stopping at complete pair tau
occurs without another invalid attempt, there are `2*tau` valid episodes and
`2*tau + 1` total attempts. The amended full-budget workload is 241 attempts
(240 originally planned plus one authorized retry). Future task episodes avoided
at a statistical crossing are `2*(120-tau)`, relative to completing that amended
workload. Do not conceal the additional failed attempt or claim infrastructure
nonexecution as statistical savings. Dollar amounts for unexecuted work are
projections, never measured or realized dollar savings.

## Execution protection and verification

Use a persistent execution session wrapped in macOS `caffeinate` to prevent idle
sleep during execution. The laptop must remain open; this assertion cannot
guarantee prevention of lid-triggered sleep. Check current battery/AC status,
credentials, model access, and application/browser availability without a paid
pilot. Do not change global power settings, request timeouts, or inference limits.
Record session boundaries and power assertions, and distinguish active execution
elapsed time from the gap between original and resumed sessions.

Before any new inference, tests must verify saved baseline/prefix reuse, unique
retry identity, one-retry enforcement, exact evidence parity, boundary stopping,
terminal-error behavior, durable crash handling, original-file immutability,
and cumulative spending including the unknown reserve. A dated manifest freezes
the source execution hashes and amendment code/document hashes. The runner checks
the manifest digest and frozen files before scheduling each new episode.

## Interpretation and limitations

Report the original run as preregistered and unresolved at its infrastructure
endpoint, and the subsequent trajectory as an **authorized amended continuation**.
Do not describe the full amended run as an unchanged, wholly preregistered trial.
The operational fact of an online threshold crossing and cessation of scheduling
can be audited independently of its population-level interpretation.

Nominal directional error guarantees for the combined trajectory require the
original conditional null assumptions to hold under this amended execution and
retry mechanism, including the mixed-time pair 14. The selective retry must not
induce outcome-dependent missingness or alter the relevant conditional paired
distribution. An outcome-blind sleep explanation, preserved prompts, and an
unchanged numerical test do not establish these assumptions. The amendment was
made after outcomes and partial failed-episode information were available;
therefore no unconditional confirmatory guarantee is claimed for the amended
experiment. Fixed arm order, temporal separation, public finite-bank exposure,
backend drift, and unresolved billing remain explicit limitations.

No source manuscript changes are authorized or made. Update the separate results,
figures, machine-readable accounting, and paper-ready fragments while retaining
the original audit history and distinguishing the amendment from the original
registration.
