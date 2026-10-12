# Framework paper: next steps after the Linear trial

Updated 2026-10-12. This is the operational plan; `FRAMEWORK_GAP_PLAN.md` preserves the original gap inventory and its intermediate updates.

## Completed in this pass

1. Reconstructed the dated SWE-bench Verified score-leader sequence at pinned revision `40f164d`. The registered dense-file rule retained only 12 entries; the post-inspection, source-semantic extension covered 135 itemwise entries and showed 32 score-leader claims, four nominal crossings, and one entrant-spending crossing. The other 47 metadata-only entries did not become score leaders. The protocol, source-format amendment, code, derived rows, and results are recorded in `SWE_LEADERBOARD_{PROTOCOL,RESULTS}.md` and `results/paper/swe_leaderboard/`.
2. Integrated that descriptive audit and a reproducible chronology figure into `paper/main.tex`. Added `METHOD_VALIDITY_NOTE.md` and a manuscript appendix that state the conditional nulls and explain why a fixed-benchmark replay does not inherit a prospective error guarantee.
3. Completed the registered 120-pair Linear trial frozen at `30fcc9a`: futility for a 15-point gain crossed at pair 93, with an observed 1.67-point gain across all 120 pairs, no missing episodes, and $16.51 spent. The [result and scope](PROSPECTIVE_LINEAR_RESULTS.md), task-level data, and manuscript now report the fixed public-bank limitation.

## Highest-priority remaining evidence

**Test the conditional future-task claim on a genuinely sampled task stream.** The Linear study collected outcomes after registration but used a fixed public bank, so seeded order does not itself justify the e-process null for future tasks. A next study should create or sample tasks prospectively from a defined population, freeze systems and scoring before outcomes, and run the incumbent concurrently. Its protocol should state why the conditional null is credible under the sampling mechanism. The current Linear crossing remains a registered result on its executed cohort, not a future-task population certificate.

## Before external submission

1. Obtain an independent mathematical and implementation review of the efficacy and futility conditional-null assumptions, especially candidate selection, task dependence, adaptive allocation, and the interpretation of a reused benchmark. Resolve any issues in the manuscript and tests before making stronger guarantees.
2. Run a reproducibility pass from clean source snapshots: pinned revisions, result-file hashes, regenerated figures and summaries, and a second extraction check of source status semantics. The current raw public caches are intentionally untracked.
3. Choose a venue and fit its style, page limit, figure requirements, and disclosure rules. The current seven-page article is a working manuscript, not a submission-ready typeset package.

The HAL and DOVE extensions remain optional breadth checks. They cannot replace a prospective validation of the conditional task-sampling claim, and the failed R6 transfer test does not support a universal tasks-to-decision predictor.
