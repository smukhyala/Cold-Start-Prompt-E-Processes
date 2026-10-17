# Independent audit of the prospective live-stopping experiment

Audit completed 2026-10-16T18:55:46.116910+00:00. All 26 listed audit checks passed. No frozen source, protocol, configuration, task, or manuscript file was changed by the audit. Detailed checks and source hashes are in `independent_audit.json`.

## Preregistration and design

Protocol and implementation commit `6ffc20fdd8315fdf20286837094f94e9cfe993a3` at 18:17:57 UTC preceded the seed manifest at 18:18:04.955556 UTC. Task commit `6c2719a9219cd82a6bf2746496d6f06226a282b1` at 18:18:18 UTC preceded controller registration, the first episode intent at 18:18:26.670970 UTC, and the first physical API request at 18:18:34.680545 UTC. All 229 frozen source/dependency/task hashes still match; registered root sources match the protocol commit. All frozen task files match the second commit. The 85 preexecution tests passed before inference.

Seed `16907f20ad5c78a18cca90759d59838a` reproduces all 120 ordered uniform draws with replacement from all 120 source tasks exactly. The plan contains 82 distinct source tasks, with repetitions retained. Baseline and challenger hashes match the historical fixed pair. The complete population, rather than an outcome-selected task subset, was used to avoid the previous profile-edit capability floor. No observed prospective outcome was used to alter the pair, population, margin, error budgets, model, or limits. `paper/main.tex` is byte-identical to the starting commit.

## Statistical result

The 13 complete pairs contain 8 baseline successes (61.5385%) and 7 challenger successes (53.8462%), a paired difference of -7.6923 percentage points. There are one challenger-only win, two baseline-only wins, and ten ties. All 27 valid individual outcomes agree with the saved environment verifier results; the extra valid baseline outcome at pair 14 is excluded from paired inference. The failed challenger outcome is missing, not scored zero.

The complete saved efficacy and futility trajectories reproduce bit-for-bit with the registered offline implementations. Final efficacy evidence is 1.0 (log 0.0); final futility evidence is 2.1360441229251084 (log 0.7589555776847441). Neither reaches 40. The primary result is **unresolved, operationally incomplete**, with no statistical first crossing.

## Execution and accounting

All 28 episode intents have exactly one matching controller receipt and matching runner receipt. They follow baseline then challenger order without episode overlap. The first ambiguous physical API request, `request_000458`, raised `ReadTimeout`; no subsequent physical API request or new episode was scheduled. Terminal status was durably recorded at 18:50:13.219839 UTC. Browser agent attempts to invoke the provider again were blocked locally and did not create paid retries.

The raw journal contains 458 physical request starts, 457 metered responses, and one unacknowledged request. Repricing every response independently from raw usage yields **$1.76230815 in known token-priced API charges**, exactly matching episode, controller, and exported subtotals. Known usage is 4,625,255 input tokens and 173,209 output tokens (4,798,464 total); 3,683,072 input tokens were cached. The unacknowledged request can add unknown usage and a charge between $0 and $0.318432 under the registered pricing/context/output bounds. Thus the bounded token-priced total is **$1.76230815–$2.08074015**, not an exact invoice total. Cost and token accounting correctly remain marked incomplete. Every request reservation was inside the $75 cap.

UTC wall-clock duration from first episode intent to terminal decision is 1,906.548869 seconds (31 minutes 46.548869 seconds). The final episode's UTC duration was 138.182022 seconds versus 95.146065 monotonic seconds. Host sleep/wake events overlap that episode, so the timeout cannot be attributed solely to the provider.

Of 240 planned executions, 28 ran and **212 remained unexecuted because of the infrastructure halt**. Executions avoided by a statistical boundary are **zero**. The experiment does not support a claim of demonstrated inference-cost savings from live statistical stopping. Any estimate for the unexecuted workload would be a hypothetical projection, not measured savings; incomplete usage and the short observed prefix would need to be disclosed.

## Remaining assumptions

All returned model identifiers match the frozen snapshot, but none supplies a usable system fingerprint; provider/backend stability and independent agent randomness cannot be verified. Correct app resets are implemented in the audited task runner but were not independently snapshotted after every reset. The fixed baseline-first order and host suspension limit stability claims. Sampling with replacement supports a conditional comparison on the finite public task population under the registered task, reset, verifier, and outcome assumptions; it does not establish an unconditional anytime-valid guarantee or generalize to unseen task families. The saved 13-pair path is descriptive, and no efficacy or practical-futility conclusion is supported.
