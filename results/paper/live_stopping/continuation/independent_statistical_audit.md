# Independent audit of the authorized live-stopping continuation

Audit time: 2026-10-16T23:55:22.807466+00:00. **All 39 programmed checks passed.** This audit made no provider calls and changed no frozen inputs, statistical code, execution logs, or manuscript.

## Statistical reproduction

Direct calls to the original `e1b_log_paths` and `futility_log_paths` functions reproduce every saved log-evidence value exactly. The combined sequence retains the original 13 pairs and saved baseline of pair 14; the one new challenger retry completes that pair.

- First boundary: **futility at pair 35**, recorded at 2026-10-16T23:51:03.196557+00:00.
- Futility evidence: 37.7355809354342 at pair 34, then **41.2660497150554** at pair 35; threshold **40** (log threshold 3.688879454113936). No earlier efficacy or futility boundary occurred.
- Efficacy evidence at stopping: **1**.
- Baseline: **20/35 = 57.14%**; challenger: **16/35 = 45.71%**; paired difference **−11.43 percentage points**.
- Parameters remain delta **0.15**, alpha_E = alpha_F = **0.025**, maximum **120 pairs**, efficacy precedence, and the original betting functions/settings.

## Execution and accounting

Raw physical intents and receipts show **28 original + 43 continued = 71 attempts**, including **70 valid outcomes**, one invalid original challenger attempt, and exactly one authorized retry (`pair_000014_challenger_retry1`). All valid outcomes match their raw environment verifier records. No baseline was rerun; no task was replaced. All episodes and HTTP requests were serial, and no pair 36 or post-boundary request was scheduled.

There are **1,212 started requests and 1,211 usage-bearing responses**. Independently repricing those responses and summing every physical receipt gives **$4.91471190 known API usage cost**. Original request 458 remains unobserved, with **$0–$0.318432** additional possible charge under the frozen request bound; total accounted cost is therefore **$4.91471190–$5.23314390**, without an invoice-level exact-cost claim. Known usage totals **12,570,744 input + 496,247 output = 13,066,991 tokens**; output includes **162,464 reasoning tokens**. Unknown-request tokens are unavailable. The old failed attempt's known cost and tokens are included exactly once. The cumulative $75 guard retained the original unknown reservation throughout.

The statistical boundary prevented **170 future planned executions**: `2 × (120 − 35)`, equivalently `241 − 71` for the amended full workload including one retry. The 170 unexecuted episodes project to **$11.86985** at the mean known cost among valid completed episodes. This is a projection, **not realized dollar savings**.

## Freeze and provenance

All **229 original frozen files**, **121 source execution/registration files**, and **8 amendment files** match their SHA256 manifests. PCG64 reconstruction exactly reproduces all 120 prospective task draws from the unchanged complete 120-task bank and original seed. Both prompt hashes, all task identities, returned model identifiers and registered inference settings match. The manuscript is byte-identical to its pre-study version.

The amendment/code commit `390b63597038b6045a153c7dc0e0cde5b71b5297` is timestamped **2026-10-16T23:07:18+00:00**; the exact manifest was committed in `2d65d9225c9a58dbe43e4a20236ceb3475532019` at **2026-10-16T23:07:59+00:00**. The first new HTTP request occurred at **2026-10-16T23:08:20.947657+00:00**. All amendment source bytes match the pre-inference commit. This chronology is supported by local Git/log records, not an external timestamp authority.

## Interpretation

The automatic first crossing and cessation of scheduling are directly supported operational findings. The original run remains **unresolved at its preregistered infrastructure endpoint**. The final futility result belongs to an **authorized post-outcome amended continuation**, not an unchanged wholly preregistered trial.

A nominal population-level futility statement requires the original conditional null assumptions to remain valid under the selective retry and mixed-time pair 14. Earlier outcomes, the saved baseline and partial failed trajectory were known when recovery was authorized; the logs do not establish outcome-ignorable missingness. Sleep overlapped the original timeout but is not proven its sole cause. No unconditional confirmatory guarantee is claimed. Futility concerns an improvement of at least 0.15; it does not establish equivalence or exclude smaller gains. Returned model identifiers are stable, but absent fingerprints and the session gap leave backend stability unverified.

The companion JSON records individual check results, precise evidence, request/episode counts, source hashes, timing and limitations.
