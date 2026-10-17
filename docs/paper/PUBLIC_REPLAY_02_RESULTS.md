# Public replay 02 results: PromptEval MMLU correctness

Run 2026-10-12 under `PUBLIC_REPLAY_PREREG_02.md`, using revision `1639d5ea14c362f6964f260ae81bd903af760187`. Code: `experiments/growing_bandits/audit/public_replay_02.py`. Tables: `results/paper/public_replay_02/`.

All **855** subject × model matrices were valid: 57 subjects, 15 models, 100 prompt rows per setting, and 100–1,534 example columns. Row 0 was chosen as a positional reference before outcomes were read; its status as a natural default is unknown. The remaining 99 rows formed each search set. Examples were split by subject into search and held-out halves, with the same split across models.

## Best-of-99 selection

| Quantity across 855 settings | Result |
|---|---:|
| Mean selected-prompt lead over row 0 on search half | +5.02 percentage points |
| Mean lead on held-out half | +0.54 points |
| Mean shrinkage (subject-cluster bootstrap 95% interval) | **4.48 [4.05, 4.92] points** |
| Median shrinkage (subject-cluster bootstrap 95% interval) | 3.92 [3.36, 4.17] points |
| Selected prompt has positive held-out gain | 48.3% [45.0%, 51.7%] |

Mean shrinkage is positive for each of the 15 models (model-level means range about 3.0–5.9 points). A search-half lead occurs in 97.3% of settings, but fewer than half of the selected prompts have a strictly positive held-out gain. This is the winner's curse in the fixed PromptEval matrix. It does not estimate the gain over a practitioner-chosen default, because row 0 is arbitrary.

## Sequential decisions on the search half

With `alpha=0.05`, 99 challengers and futility margin `delta=0.15`, the first-boundary decisions were **4 efficacy, 421 futility, 430 unresolved**. Four settings crossed both boundaries eventually; this is logically possible when a small positive gain is detectable while every gain is below 15 points. For the efficacy test alone, stopping saved 130,000 of 10,515,000 arm-example evaluations (1.2%). A separate drop-the-loser futility replay certified 436 settings and used 4,300,530 arm-example evaluations, 59.1% below its full-grid count. These are replay costs on already observed outcomes, not newly saved inference spend.

The selected best-of-99 prompt is not necessarily the arm whose e-process drove an efficacy crossing. Of the four settings with efficacy as the first boundary, one selected prompt had a negative held-out gain. Therefore the four efficacy decisions should not be described as four independently confirmed winners.

## Limits and paper implication

PromptEval supplies broad evidence that search-set selection overstates the gain, across many tasks and models. It does not settle how often a *real* prompt search beats an incumbent: row 0 is positional, the models are older than the live web-agent study, and no rerun trials are available. The median search half contains 86 examples; 430/855 settings remain unresolved at the registered thresholds. The Terminal-Bench replay also did not certify a winner, so the framework paper's top gap remains open. A dataset with a natural incumbent and more paired tasks is the next necessary check.
