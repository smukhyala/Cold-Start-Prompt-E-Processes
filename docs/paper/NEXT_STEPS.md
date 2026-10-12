# Framework paper: next steps after the dated leaderboard audit

Updated 2026-10-12. This is the operational plan; `FRAMEWORK_GAP_PLAN.md` preserves the original gap inventory and its intermediate updates.

## Completed in this pass

1. Reconstructed the dated SWE-bench Verified score-leader sequence at pinned revision `40f164d`. The registered dense-file rule retained only 12 entries; the post-inspection, source-semantic extension covered 135 itemwise entries and showed 32 score-leader claims, four nominal crossings, and one entrant-spending crossing. The other 47 metadata-only entries did not become score leaders. The protocol, source-format amendment, code, derived rows, and results are recorded in `SWE_LEADERBOARD_{PROTOCOL,RESULTS}.md` and `results/paper/swe_leaderboard/`.
2. Integrated that descriptive audit and a reproducible chronology figure into `paper/main.tex`. Added `METHOD_VALIDITY_NOTE.md` and a manuscript appendix that state the conditional nulls and explain why a fixed-benchmark replay does not inherit a prospective error guarantee.

## Highest-priority remaining evidence

**Run one genuinely prospective, concurrently controlled trial on new tasks.** Freeze an incumbent and one challenger before either sees any outcome on a fresh task cohort. A fixed pair is the cheapest first test: 500 paired tasks require exactly 1,000 arm-task episodes before optional stopping, versus 25,500 for 50 challengers. Use `PROSPECTIVE_TRIAL_TEMPLATE.md` to select the task population, exact system versions, scoring, missing-run policy, target gain, margin, and error allocation before collecting data. Keep each arm's task and timing paired. The result can validly be efficacy, futility, or unresolved; do not choose a different challenger after reading the pilot. A new task cohort and executable systems are external dependencies, so no live run is launched by this paper audit.

## Before external submission

1. Obtain an independent mathematical and implementation review of the efficacy and futility conditional-null assumptions, especially candidate selection, task dependence, adaptive allocation, and the interpretation of a reused benchmark. Resolve any issues in the manuscript and tests before making stronger guarantees.
2. Run a reproducibility pass from clean source snapshots: pinned revisions, result-file hashes, regenerated figures and summaries, and a second extraction check of source status semantics. The current raw public caches are intentionally untracked.
3. Choose a venue and fit its style, page limit, figure requirements, and disclosure rules. The current seven-page article is a working manuscript, not a submission-ready typeset package.

The HAL and DOVE extensions remain optional breadth checks. They cannot replace a prospective validation of the conditional task-sampling claim, and the failed R6 transfer test does not support a universal tasks-to-decision predictor.
