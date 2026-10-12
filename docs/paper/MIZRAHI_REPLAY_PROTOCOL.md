# Mizrahi natural-default replay protocol (2026-10-12)

Registered before opening the source's aggregate accuracies or raw response archive. Only the source README, directory names, automatic-paraphrase column names, and `default prompt` flags in two example task files were inspected. Source: `SLAB-NLP/Multi-Prompt-LLM-Evaluation` revision `04f57ef4d5201ef817f54f0e1b09e3fc0c481fb6` (CDLAv2). The raw-response Dropbox folder currently advertises a 405,379,827-byte zip; keep it local, outside tracked results.

## Unit and reference

- Use every task with both an automatic-paraphrase CSV and item-level raw responses in the archive. There are 25 task CSVs in the repository listing: 15 BBH, 10 LMentry. Include every model common to the valid prompt rows for that task. Do not select tasks or models by outcomes.
- Within each task, the incumbent is the row with the smallest numeric `prompt idx` among rows whose `default prompt` flag is true. Other default rows are excluded from challengers. Challengers are rows with false `default prompt` and true `correct`; exclude invalid paraphrases. This choice is made without accuracies.
- Parse raw responses into binary correctness by example, prompt index and model using fields supplied by the archive. If the archive lacks a reliable answer/correctness field, apply only the source's own documented scoring code. Do not invent a new grader. Validate each reconstructed prompt × model accuracy against the repository's corresponding aggregate CSV; retain a task/model only if all checked rows agree within one sample (0.01 for 100 examples). Report missing cells and failures. If fewer than 10 tasks or 8 models pass, stop and report that the item-level replay is infeasible.

## Search and sequential analysis

- For each task, order common example IDs lexically and permute them with NumPy `SeedSequence([20261012, 21, task_index])`, where tasks are lexically ordered. Use the first floor(N/2) examples for search, the remainder held out, with the same split across models. Require N >= 40 common examples per retained setting.
- Select the highest-accuracy challenger on the search half; break ties by smallest prompt index. Report search and held-out gain against the task's original-template incumbent, shrinkage, and share of positive held-out gains. Across settings, use a task-cluster bootstrap with 2,000 resamples from seed `20261016` for mean shrinkage and positive-gain share. Do not treat settings sharing task examples as independent.
- On the search half, run the existing incumbent-relative E1b mixture and the pool futility test (`delta=0.15`), each at `alpha=0.05`, across all registered challengers. Report first efficacy/futility boundary, unresolved share, crossing times, and retrospective full-grid versus stopped episode counts. The inference pertains to one task/model setting; no across-setting familywise claim is made.

## Limits

The source evaluates historical models and tasks; the original benchmark template is a meaningful reference, but not necessarily the best real deployment default. Paraphrase correctness judgments are source annotations. The raw archive may not support itemwise reconstruction despite its advertised availability; that would be a documented feasibility result, not a reason to substitute aggregate accuracies for held-out outcomes.
