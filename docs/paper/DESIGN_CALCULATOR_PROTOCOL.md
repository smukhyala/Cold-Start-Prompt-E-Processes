# R6 pilot-based design calculator protocol (2026-10-12)

This is a retrospective exploratory analysis. R5 outcomes in PromptEval and Mizrahi were already seen before this protocol. The feature definitions, split, and validation criteria below are fixed before fitting the calculator.

## Prediction target and inputs

The planning question is: after 10 paired examples with no boundary crossing, what is the chance of an efficacy or pool-futility decision by 50 examples at alpha 0.05 and delta 0.15? Use only information available by task 10: number of challengers K, incumbent accuracy, mean challenger-incumbent discordance, maximum observed challenger gain, and standard deviation of challenger gains. Do not use held-out examples, final gains, or the final boundary path as predictors. A design can also calculate exact full-grid cost as `(K+1)*N` paired arm-example evaluations; this deterministic calculation requires no fit.

Use PromptEval and the accepted Mizrahi task/model matrices. For Mizrahi, restrict to its 50-example search half. For PromptEval, restrict to the first 50 search examples of settings with at least 50. Drop settings already decided by task 10 from the prediction validation but report their count. Predict the three mutually exclusive outcomes at task 50: efficacy first, futility first, or unresolved. Fit regularized multinomial logistic regression with standardized inputs. Choose the regularization value from `C = {0.1, 1, 10}` using group cross-validation on training tasks only, optimizing multiclass log loss.

## Validation

Split task names 50/50 within each dataset by a fixed seeded shuffle (`20261017`), keeping all models of a task on one side. Fit on training tasks from both sources, then evaluate on held-out tasks. Report class counts, multiclass Brier and log loss against the training-prevalence predictor. Also test a PromptEval-only fit on Mizrahi and the reverse; report transfer failure if either benchmark is worse than its training-prevalence baseline. No claim of calibrated prospective probability follows from retrospective data. The calculator is usable for design guidance only if it improves both metrics over prevalence in held-out tasks and in both cross-dataset transfers; otherwise publish the negative validation and the exact budget calculator only.

The own web-agent studies have too few independent tasks and 30--80 task horizons. They can be shown as external context, but not counted as an independent validation sample for a fitted multi-class model.
