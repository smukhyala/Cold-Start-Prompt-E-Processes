# Framework paper: next steps after the dated leaderboard audit

Updated 2026-10-12. This is the operational plan; `FRAMEWORK_GAP_PLAN.md` preserves the original gap inventory and its intermediate updates.

## Completed in this pass

1. Reconstructed the dated SWE-bench Verified score-leader sequence at pinned revision `40f164d`. The registered dense-file rule retained only 12 entries; the post-inspection, source-semantic extension covered 135 itemwise entries and showed 32 score-leader claims, four nominal crossings, and one entrant-spending crossing. The other 47 metadata-only entries did not become score leaders. The protocol, source-format amendment, code, derived rows, and results are recorded in `SWE_LEADERBOARD_{PROTOCOL,RESULTS}.md` and `results/paper/swe_leaderboard/`.
2. Integrated that descriptive audit and a reproducible chronology figure into `paper/main.tex`. Added `METHOD_VALIDITY_NOTE.md` and a manuscript appendix that state the conditional nulls and explain why a fixed-benchmark replay does not inherit a prospective error guarantee.

## Highest-priority remaining evidence

**Complete the prospective, concurrently controlled Linear trial now underway.** The fixed incumbent/challenger pair, all 120 Linear account-settings tasks, queue, hashes, margin, error allocation, and missing-run rule were frozen at commit `30fcc9a` before collection. See `PROSPECTIVE_LINEAR_PROTOCOL.md`. The 240-episode run started on 2026-10-12 with a $35 collector cap. When it ends, preserve all attempts, produce the complete task-level table with `experiments/growing_bandits/empirical/analyze_linear_trial.py`, check missingness and the registered decision paths, and update `paper/main.tex` with the result, including unresolved or incomplete status if applicable. Do not change the pair after seeing outcomes.

## Before external submission

1. Obtain an independent mathematical and implementation review of the efficacy and futility conditional-null assumptions, especially candidate selection, task dependence, adaptive allocation, and the interpretation of a reused benchmark. Resolve any issues in the manuscript and tests before making stronger guarantees.
2. Run a reproducibility pass from clean source snapshots: pinned revisions, result-file hashes, regenerated figures and summaries, and a second extraction check of source status semantics. The current raw public caches are intentionally untracked.
3. Choose a venue and fit its style, page limit, figure requirements, and disclosure rules. The current seven-page article is a working manuscript, not a submission-ready typeset package.

The HAL and DOVE extensions remain optional breadth checks. They cannot replace a prospective validation of the conditional task-sampling claim, and the failed R6 transfer test does not support a universal tasks-to-decision predictor.
