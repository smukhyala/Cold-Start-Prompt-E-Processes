# R6 pilot-based design guidance and validation

Run 2026-10-12 under [DESIGN_CALCULATOR_PROTOCOL.md](DESIGN_CALCULATOR_PROTOCOL.md). Code: `experiments/growing_bandits/audit/design_calculator.py`; derived result: `results/paper/design_calculator/summary.json`.

The prediction target was the first efficacy, futility, or unresolved outcome by 50 paired tasks, conditional on no crossing in a ten-task pilot. None of the 1,160 PromptEval and Mizrahi settings crossed by task 10. Features available at that point were candidate count, incumbent success, challenger discordance, largest observed gain, and spread of observed gains. The model is a standardized regularized multinomial logistic regression, selected by training-task group cross-validation.

| Validation | N | Model Brier / log loss | Training-prevalence Brier / log loss | Passes both? |
|---|---:|---:|---:|---|
| Held-out tasks in both datasets | 587 | 0.131 / 0.241 | 0.177 / 0.382 | Yes |
| PromptEval fit → Mizrahi | 305 | 0.390 / 5.276 | 0.378 / 6.916 | **No** |
| Mizrahi fit → PromptEval | 855 | 0.103 / 0.267 | 0.157 / 0.466 | Yes |

The registered transfer criterion fails. At the 50-task horizon, PromptEval contains no efficacy outcomes in its 855 settings, whereas Mizrahi contains 58 in 305. A model trained on PromptEval cannot estimate that class reliably. The contrast reflects a real design difference: PromptEval uses a positional row-0 reference and has relatively small gains; Mizrahi uses an original-template incumbent with many large gains. No general tasks-to-decision calculator is validated by these data. The within-dataset held-out result is exploratory evidence of useful pilot information, not a prospective performance guarantee.

The usable cost calculator is exact: a paired grid of `K` challengers over `N` tasks costs `(K+1)N` arm-task evaluations (or `KN` if a contemporaneous incumbent bank is valid for the intended estimand). `experiments/growing_bandits/audit/design_budget.py` calculates full-grid episodes, cost, and an idealized parallel wall time. For example, `--challengers 50 --tasks 80` requires 4,080 arm-task episodes. Stopping can reduce that total, but a prospective saving needs setting-specific pilot data and validation. In our two historical prompt matrices, decisions differed sharply: PromptEval had 4 efficacy, 421 futility, 430 unresolved over its full search halves; Mizrahi had 58 efficacy, 2 futility, 245 unresolved at 50 tasks. Thus even the efficacy/futility cost ratio does not transfer as a universal constant.
