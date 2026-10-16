# Live-stopping experiment: final amended result

**Futility boundary crossed online at pair 35; 170 planned executions avoided.**
Baseline successes: 20/35. Challenger successes: 16/35. Efficacy evidence: 1.000;
futility evidence: 41.266; threshold: 40. Total physical attempts: 71, including
one original failure and one authorized retry. Known API charges: $4.91471190,
plus an unresolved original charge bounded by $0.318432.

The user authorized one retry after the original 13-pair infrastructure halt.
The amended trajectory retains that prefix and its saved pair-14 baseline.
It must not be described as an unchanged wholly preregistered experiment.
See [the full report](../../../docs/paper/LIVE_STOPPING_RESULTS.md) and
[the amendment](../../../docs/paper/LIVE_STOPPING_RESUME_AMENDMENT.md).

## Latest combined artifacts

- `summary.json`, `operational_accounting.json`: final statistics and resources.
- `pairs.{json,csv}`, `evidence.{json,csv}`, `episodes.csv`: all observed pairs and attempts.
- `episode_receipts.json`, `episode_intents.json`, `api_calls.jsonl`: both execution sessions;
  request IDs are namespaced by the `session` field. The invalid attempt is retained.
- `state.json`: effective controller view of 70 valid episode records. Physical attempt
  counts and all-attempt costs, including the original invalid attempt, are in `summary.json`.
- `registration.json`: unchanged original design; `amendment.json` records the retry exception.
- `terminal.json`: actual first-boundary terminal, not a retrospective reconstruction.
- `live_stopping_continuation.{pdf,svg,png}`: complete real-data figure. The older figure
  filenames also point to this updated plot; original figures remain in history.
- `paper_ready.md`, `results_table.tex`, `figure_caption.tex`: manuscript fragments.
- `execution_timeline.json`: original run, blocked review, and later authorized continuation.
- `independent_audit.{json,md}`: final independent verification.
- `continuation/`: self-contained final export, detailed audits, opening balance,
  launch metadata, sleep-prevention evidence, and 101 passing preflight tests.
- `sha256.json`: final artifact checksums.

## Preserved history

- `history/2026-10-16_initial_halt/`: original 13-pair result, receipts, figures,
  report, and its unchanged checksum manifest.
- `history/2026-10-16_blocked_resume_review/`: report and publication fragments
  from the subsequent blocked review, before the user authorized an amendment.
- `resume_review/2026-10-16/`: read-only review showing why the original rule blocked resume.
- Original `preflight*`, `browser_preflight.json`, and `environment_events.json`
  at this directory's root document the initial run; current preflight/audits are under `continuation/`.

Raw execution logs are in `logs/prospective_live_stopping/` (unchanged original)
and `logs/prospective_live_stopping_resume_20261016/` (continued physical attempts).
The original protocol and `paper/main.tex` were not edited. The projected $11.86985
cost of unexecuted work is hypothetical, not realized dollar savings. Population-level
error control requires the conditional recovery assumptions stated in the report.
