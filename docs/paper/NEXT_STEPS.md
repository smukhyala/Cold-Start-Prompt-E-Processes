# Framework paper: next steps after the sampled Linear trial

Updated 2026-10-12. This is the operational plan; `FRAMEWORK_GAP_PLAN.md` preserves the original gap inventory and its intermediate updates.

## Completed in this pass

1. Reconstructed the dated SWE-bench Verified score-leader sequence at pinned revision `40f164d`. The registered dense-file rule retained only 12 entries; the post-inspection, source-semantic extension covered 135 itemwise entries and showed 32 score-leader claims, four nominal crossings, and one entrant-spending crossing. The other 47 metadata-only entries did not become score leaders. The protocol, source-format amendment, code, derived rows, and results are recorded in `SWE_LEADERBOARD_{PROTOCOL,RESULTS}.md` and `results/paper/swe_leaderboard/`.
2. Integrated that descriptive audit and a reproducible chronology figure into `paper/main.tex`. Added `METHOD_VALIDITY_NOTE.md` and a manuscript appendix that state the conditional nulls and explain why a fixed-benchmark replay does not inherit a prospective error guarantee.
3. Completed the registered 120-pair Linear trial frozen at `30fcc9a`: futility for a 15-point gain crossed at pair 93, with an observed 1.67-point gain across all 120 pairs, no missing episodes, and $16.51 spent. The [result and scope](PROSPECTIVE_LINEAR_RESULTS.md), task-level data, and manuscript now report the fixed public-bank limitation.
4. Completed the [prospectively sampled profile-edit trial](PROSPECTIVE_SAMPLED_LINEAR_RESULTS.md): 120 new task draws, no missing episodes, $23.60 spent, incumbent 3/120 versus challenger 1/120, and the registered 15-point futility boundary crossed at pair 44. The task distribution was frozen before its seed and the realized bank before outcomes. Both agents were near the capability floor.

## Highest-priority remaining evidence

**Design a broader sampled task trial with nontrivial agent success.** First use an outcome-independent pilot to choose a task generator and agent budget that avoid the profile-edit floor, then freeze the target population, prompt pair or candidate pool, stopping rule, missingness policy, and cost cap before drawing the confirmatory task seed. Evaluate both arms concurrently on each fresh draw and report every attempt. The sampled Linear result supports futility only for its narrow profile-edit distribution under stability assumptions; it does not establish that the framework can discriminate capable agents on general web tasks. Independent review of the conditional assumptions remains necessary before stronger validity claims.

## Before external submission

1. Obtain an independent mathematical and implementation review of the efficacy and futility conditional-null assumptions, especially candidate selection, task dependence, adaptive allocation, and the interpretation of a reused benchmark. Resolve any issues in the manuscript and tests before making stronger guarantees.
2. Run a reproducibility pass from clean source snapshots: pinned revisions, result-file hashes, regenerated figures and summaries, and a second extraction check of source status semantics. The current raw public caches are intentionally untracked.
3. Choose a venue and fit its style, page limit, figure requirements, and disclosure rules. The current article is a working manuscript, not a submission-ready typeset package.

The HAL and DOVE extensions remain optional breadth checks. They cannot replace a prospective validation of the conditional task-sampling claim, and the failed R6 transfer test does not support a universal tasks-to-decision predictor.
