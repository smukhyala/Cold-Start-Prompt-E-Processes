# Live-stopping experiment artifacts

Final result: **unresolved, operationally incomplete**, 13 complete pairs,
28 attempted episodes, no statistical crossing. The next pair was not replaced.

Read the [full report](../../../docs/paper/LIVE_STOPPING_RESULTS.md) for the
interpretation, or `paper_ready.md` for manuscript text. Primary files:

- `summary.json`: exact frozen-export statistical findings and known resources.
- `operational_accounting.json`: known-charge interval and hypothetical remaining-work cost.
- `pairs.csv`, `evidence.csv`, `episodes.csv`: portable tabular results.
- `pairs.json`, `evidence.json`, `episode_receipts.json`, `episode_intents.json`:
  complete outcome/evidence/execution metadata.
- `api_calls.jsonl`: physical request ledger, with one unresolved charge.
- `registration.json`, `terminal.json`, `state.json`: frozen design and durable decision.
- `independent_audit.json`, `independent_audit.md`: 26 independent audit checks.
- `environment_events.json`: sleep/wake times overlapping the final request.
- `preflight*`, `browser_preflight.json`: 85 passing tests and zero-inference setup checks.
- `live_stopping_observed_prefix.pdf`: primary publication figure; SVG and PNG also supplied.
- `live_stopping_evidence.pdf`: original full-budget-axis figure from the frozen exporter.
- `results_table.tex`, `figure_caption.tex`: LaTeX fragments.
- `sha256.json`: final artifact checksums.

Exact source, model settings, prompt hashes, seed, and realized tasks are in
`data/prospective_live_stopping/`. The original browser traces are under
`logs/prospective_live_stopping/`. Dollar projections are not realized savings.
No data after the operational halt were collected.

## Resume review, 2026-10-16

Continuation was blocked by the registered terminal infrastructure stop. No new
agent episodes or inference requests were made. The full 13-pair trajectory and
original summary remain unchanged.

- `resume_review/2026-10-16/resume_audit.json`: exact reanalysis, no-callback
  controller replay on temporary copies, unchanged log hashes, and spending balance.
- `resume_review/2026-10-16/README.md`: independent review and the specific blocker.
- `execution_timeline.json`: original run and later review; no resumed inference session.
- `history/2026-10-16_initial_halt/`: original artifacts, report, and checksum manifest.

Reproduce the review without inference:

```bash
.venv/bin/python experiments/growing_bandits/empirical/audit_live_stopping_resume.py \
  --out /tmp/live-stopping-resume-review
```
