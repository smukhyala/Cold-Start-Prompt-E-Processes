# Independent operational audit — authorized live-stopping continuation

Audited 2026-10-16T23:53:22.070286+00:00. **PASS**. All original and continuation API usage receipts were independently repriced using decimal arithmetic; every physical episode cost reconciles to its request ledger.

- Physical attempts: **71**; valid outcomes: **70**; original invalid attempts: **1**. The sole new physical retry is `pair_000014_challenger_retry1`.
- Original known charges: **$1.762308150**. Continuation known charges: **$3.152403750**. Combined known charges: **$4.914711900**.
- The original unknown request retains **$0.318432**. Combined total charges are bounded by **$4.914711900–$5.233143900** under the frozen pricing/token assumptions; exact total cost remains unknown.
- Known tokens: **12,570,744 input**, **496,247 output**, including **162,464 reasoning tokens**. Cached input is **9,994,752**, already included in input.
- Futility was recorded at pair **35**, at **2026-10-16T23:51:03.196557+00:00**. No pair 36 intent or API event after that terminal record exists.
- **170 planned executions** were avoided after the statistical crossing. The failed original attempt adds one physical attempt, giving **169 fewer physical attempts than a 240-execution plan without retries**.
- Wall time measured from each controller's first scheduling intent to its terminal decision: original **1906.548869 s**, continuation **2569.975486 s**, sum **4476.524355 s**. Continuation launch-to-terminal: **2572.283508 s**. The JSON also preserves timings from the runner's later episode intents.
- Runner PID 29933 and caffeinate PID 29935 have exited. The host log records the sleep assertion still present at 23:16:34, 23:31:34, and 23:46:34 UTC, then `ClientDied` at **23:51:04 UTC**; the assertion is now absent.
- System sleep/wake events recorded during the continuation window: **0**. Only sanitized event metadata is retained in the JSON audit.
- Every physical request independently satisfies the global spending bound when its reservation is added, including the original unresolved reserve throughout continuation. At most one API request was awaiting a receipt at a time within either segment.
- The source files pinned by the amendment, including the original ledger and terminal result, are unchanged. The original known charges, unresolved reserve, and new spending allocation reconcile exactly to the global **$75** cap.

The original unknown charge remains unresolved, and the continuation follows an explicit post-outcome amendment. These checks establish operational and accounting consistency; they do not independently establish statistical sampling assumptions, backend stability, or invoice settlement.
