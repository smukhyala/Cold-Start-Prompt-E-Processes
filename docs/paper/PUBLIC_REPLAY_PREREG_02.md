# Public replay 02: PromptEval MMLU prompt search

Registered 2026-10-12, before opening any PromptEval correctness matrix. Only its dataset card, README schema, model/subject directory listing, and revision were inspected. This is a retrospective public-data replay.

## Data and estimand

- Source: `PromptEval/PromptEval_MMLU_correctness` at revision `1639d5ea14c362f6964f260ae81bd903af760187` (MIT).
- Use every subject directory and every model parquet file present at that revision. Each matrix has prompt rows and example columns. Retain a setting if it has exactly 100 prompt rows, at least 40 example columns, and only binary correctness entries; report exclusions.
- Row 0 is the **operational reference**, chosen by position before outcomes. It is not known to be a canonical or deployed default. Rows 1–99 are the fixed challengers. Do not call this a real-world incumbent comparison.
- For each subject, use NumPy `SeedSequence([20261012, 5, subject_index])` with subjects in lexical order to permute example columns. The first floor(N/2) examples are search; the rest are held out. The same split applies to every model for a subject.
- Select the challenger with the highest search-half mean correctness, breaking ties by lowest row index. Report its gain over row 0 on the search and held-out halves, and the shrinkage between them. Do not select from held-out outcomes. Across settings, report the mean/median shrinkage, share of positive held-out gains, and subject-cluster bootstrap 95% intervals (2,000 resamples, seed 20261014). Settings are correlated through shared examples and prompts, so do not use an ordinary binomial interval over 855 settings.

## Sequential decisions on the search half

- On the search-half example order above, use the existing incumbent-relative E1b mixture across 99 challengers at `alpha=0.05`. Report the fraction of settings crossing, crossing example, and full-grid versus stopped episode counts.
- Run the existing pool futility test with `delta=0.15`, `alpha=0.05`, also over the search-half example order. Report futility, efficacy, and unresolved fractions. If both boundaries cross, report the first boundary and the count of double crossings. Do not interpret settings as independent repeated tests or claim familywise control across settings.
- The primary claim is the magnitude and direction of held-out shrinkage. The sequential fractions are secondary: half of an MMLU subject can be too small for either boundary.

## Interpretation

The fixed benchmark yields a conditional search/held-out comparison, not fresh deployment trials. Row 0 is arbitrary and may be weaker or stronger than a natural default. The 15 models are older than the live agent study and example outcomes have no rerun replicates. This replay measures selection optimism and cross-model breadth, not rerun-noise validity or current agent performance.
