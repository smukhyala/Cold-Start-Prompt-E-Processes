# Public replay 03: SWE-bench Verified matched scaffold pairs

Registered 2026-10-12 before opening any `results.json` from the selected submissions. Only the repository README, revision, directory names, and result-file names/sizes were inspected. The repository is public, but no license was found at its root; do not commit its raw results to this repo.

## Frozen source and pairs

- Source: `SWE-bench/experiments` GitHub revision `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`, under `evaluation/verified/`.
- Primary fixed pair: `20240402_rag_gpt4` (reference) versus `20240402_sweagent_gpt4` (challenger).
- Secondary fixed pair: `20240402_rag_claude3opus` (reference) versus `20240402_sweagent_claude3opus` (challenger).
- The two submissions in each pair have the same date and model label. Read their `metadata.yaml` after this registration to verify model identity and any visible configuration deviations. Exclude a pair if the model versions are demonstrably different. Do not substitute another pair.
- Read `results/results.json` for each of the four submissions. Map the repo's instance-level `resolved` status to binary outcomes. Use the intersection of instance IDs per pair, sorted lexically. Include a pair only if at least 400 common instances have unambiguous binary outcomes; otherwise report a feasibility failure. Do not discard an instance based on its outcome. Report any missing, duplicate, or ambiguous records.

## Inference and replay

- For each pair, use a single prespecified task order from NumPy `SeedSequence([20261012, 4, pair_index])`, pair index 0 for GPT-4 and 1 for Claude. Run the repository's incumbent-relative E1b test with one challenger at `alpha=0.025`, providing Bonferroni control for the two fixed efficacy claims. Record first crossing task, final and maximum e-values, and full-grid versus stopped episode counts.
- Report the paired success-rate difference, its task-bootstrap 95% interval (10,000 resamples, seed `20261015 + pair_index`), and discordant win/loss counts. The confidence interval is descriptive and does not replace the registered sequential boundary.
- For stopping-time sensitivity, replay 2,000 task orders per pair from `SeedSequence([20261012, 40, pair_index, replicate])`; report the crossing fraction and first-crossing task distribution. These are reorderings of the same benchmark, not 2,000 independent experiments.
- A pair is a **certified real benchmark winner** only if the corresponding prespecified e-process crosses 1/0.025. The primary headline uses GPT-4; Claude is a fixed secondary replication. If either model identity or per-instance format fails the rules above, report that rather than modify thresholds or pairs.

## Scope

These are historical agent systems on a fixed public benchmark. Same date and model label improve comparability but do not prove that scaffold alone caused a gain; implementation and inference settings may differ. The anytime-valid interpretation assumes tasks representative of a stable population and a conditional null suitable for the paired betting test. Task reordering alone cannot establish validity under new sessions or future tasks. This replay does not implement the full dated-leaderboard analysis R4, which would require an additional protocol for changing incumbents and repeated comparisons.
