# Gmail first-run pilot: results

*Run 2026-10-09, 22:07–23:26 UTC, from a normal terminal. 90 of 90 episodes scored `ok`. Spend was **$6.85**
against the $12 cap, compared with the $8.64 estimate. There were 0 relaunches, and 6 episodes ended on the clock.
The design is frozen in [FIRST_RUN_PILOT_READY.md](FIRST_RUN_PILOT_READY.md). The analyzer is
`experiments/growing_bandits/retrospective/analyze_first_run_pilot.py`, and its outputs are in
`results/growing_bandits/retrospective/first_run_pilot_gmail/` (bootstrap seed 20261007, worker-log SHA recorded).
This is a 30-task feasibility pilot, not a confirmatory test.*

## Policy results (paired by task, task-bootstrap intervals)

| Policy | Success | Gain vs stop | Gain vs timeout-same | Mean cost / task | Cost / success |
|---|---|---|---|---|---|
| Stop after G_14 | 0.800 | 0 | −0.033 [−0.10, 0] | $0.071 | $0.089 |
| Timeout → rerun G_14 | 0.833 | +0.033 [0, +0.10] | 0 | $0.080 | $0.095 |
| Timeout → switch to G_16 | 0.833 | +0.033 [0, +0.10] | 0 [0, 0] | $0.076 | $0.092 |
| **Frozen uplift rule (primary)** | 0.833 | +0.033 [−0.07, +0.13] | **0 [−0.10, +0.10]** | $0.084 | $0.101 |
| *Oracle coverage, same / different pair* | *0.900 / 0.933* | | | | |

## What happened, task by task

**First runs.** G_14 succeeded on 24 of 30 tasks. Of the 6 failures, 2 were incomplete (timeouts, `task_e18` and
`task_h14`) and 4 were *false completions*: the agent said "done" and the verifier disagreed.

**The timeout rules.**
- Both timeout rules rescue `task_e18` and both miss `task_h14`, so their results are identical.
- Switching to G_16 shows no advantage over rerunning G_14; on this pair the two are indistinguishable.

**The frozen uplift rule** retried 6 tasks, 4 of them after an apparently complete first run.
- **Rescued two tasks:** `task_e18`, a timeout, and `task_e3`, a false completion.
- **Lost one task:** `task_e1`. The first run had correctly starred the email. The retry started from a reset state,
  failed to star it, and the policy kept the retry's result.
- **Net effect:** +1 task, the same as timeout-only, at 6% higher cost per task.
- **The rule caught 1 of the 4 false completions** and threw away 1 true success.

**Oracle coverage versus realized policies.** Oracle coverage for the pair is 0.90–0.93, while every realized policy
reaches 0.83. Choosing which attempt to keep is the binding constraint, not generating more attempts.

## Reading against the pre-registered outcome table

| Possible result (from the readiness doc) | Observed? |
|---|---|
| Timeout-triggered retry helps on these 30 IDs | **Weakly.** +1 task, with an interval reaching 0. Cost per success rises 7%. |
| G_16 beats G_14 after an incomplete first run | **No.** The two are identical on both incomplete tasks. |
| Frozen benefit rule beats timeout-only on success *and* cost | **No.** Equal success, higher cost. |
| Benefit rule is null or harmful | **Null.** The old-data signal does not turn into a gain here. |
| Oracle pair coverage rises but policies do not | **Yes.** 0.90–0.93 oracle coverage against 0.83 realized. |

**Conclusion.** The pilot reproduces the retrospective pattern on new prompt–task cells.
- **Visible non-completion is a usable but small trigger.**
- **Recovering false completions needs a verifier** that can tell a correct "done" from a wrong one. Without one, a
  retry can overwrite a success.
- **Switching prompts adds nothing over rerunning for this frozen pair.**

This supports Roadmap item 2, a decision-time verifier, as the only route to an operational portfolio gain. It does
not support investing further in metadata-based retry rules.

## Unexpected: rerun noise about 3× the historical estimate

- **G_14 against itself.** The first run and the independent rerun disagree on **10 of 30 tasks (33%)**. The first
  run succeeded in 24 of 30, the rerun in 20 of 30, with 7 tasks succeeding only on the first run and 3 only on the
  rerun.
- **Historical comparison.** Repeat cells disagreed in 10.3% of G cells (n = 145) and 17.4% of F cells (n = 155). At
  the G rate, P(≥ 10 of 30) is 0.0005; at the F rate it is 0.015.
- **State carry-over is ruled out.**
  - `webarena-infinity/evaluation/tasks.py` `run_task` calls `reset_state()` before every episode.
  - The verifier messages confirm it. In `task_h3`, email 31, archived by the first G_14 run, is unarchived again
    when the later rerun starts.
- **Task selection is ruled out.** The 30 IDs are the seeded random complement of Pre-reg 9's stratified bank.
- **A possible cause is provider drift.** G_14 now costs $0.076 per episode against $0.102 historically, about 25%
  lower, which suggests the model behind `gpt-5.4-mini` behaves differently from the September runs.
- **Implication for the paper.** The noise model (v) used for every NPMLE reservoir dates from the original
  collection. If current noise is higher, the measured pools are, if anything, flatter relative to noise. But a
  re-collection would not reproduce the old numbers. Add this to the limitations.
- **A cheap check (≈ $3, optional).** Rerun about 30 of the historical G repeat cells now and compare their
  disagreement rate with the 10.3% from September.
