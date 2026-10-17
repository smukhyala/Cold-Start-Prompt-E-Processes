# Mizrahi natural-default replay results

Run 2026-10-12 under [MIZRAHI_REPLAY_PROTOCOL.md](MIZRAHI_REPLAY_PROTOCOL.md), using source revision `04f57ef4d5201ef817f54f0e1b09e3fc0c481fb6`. Code: `experiments/growing_bandits/audit/mizrahi_replay.py`; derived setting rows: `results/paper/mizrahi_replay/summary.json`. The raw response archive stays local and untracked.

The 25 task CSVs and item-level response archive yielded **24 tasks, 16 models, and 305 valid task/model settings**. All accepted prompt rows reconciled to the published aggregate accuracy to within one example: 54,672 matched prompt/model rows. Four settings failed this gate (FLAN-T5 base and small on two category tasks); all 16 model settings for `word_not_containing` had only 26 examples, below the registered 40-example minimum. The other 790 mismatched rows belong to the four excluded settings. Across accepted settings, the candidate count is 116--232 valid paraphrases, plus the smallest-index original benchmark template as incumbent.

| Search versus held-out result, 305 settings | Value |
|---|---:|
| Mean best-paraphrase search-half gain | +22.96 percentage points |
| Mean selected-paraphrase held-out gain | +15.04 points |
| Mean shrinkage, task-cluster bootstrap 95% interval | **7.92 [6.24, 9.50] points** |
| Selected paraphrase positive held out, task-cluster bootstrap 95% interval | **78.0% [72.8%, 83.2%]** |

The original template is a meaningful benchmark default. This historical pool contains many paraphrases that outperform it, unlike the arbitrary row-0 comparison in PromptEval. Yet selecting the apparent best of a large pool still overstates the gain by about eight points on average. The held-out split shares examples across models within each task, so uncertainty is clustered by task.

On the 50-example search half, with `alpha=0.05` and a 15-point futility margin, the first boundary was **58 efficacy, 2 futility, and 245 unresolved**. Among the 60 decisions, the median crossing was at 35.5 paired examples. A full grid would evaluate 2,727,500 arm-example outcomes; the counterfactual first-boundary stop uses 2,546,755, a **6.6% reduction**. These are costs reconstructed from a complete historical matrix, not live savings. A pool-level efficacy crossing states that at least one challenger has positive advantage under the sequential assumptions; it does not certify that the search-half argmax is that arm, nor provide familywise control over 305 settings.

This result is deliberately different from PromptEval: it has a source-identified original template, a larger candidate pool and much stronger observed gains. The contrast rules out a universal claim that prompt pools are flat, and makes incumbent choice central to any design recommendation. It remains a historical fixed-benchmark replay without repeat runs or protection against model, task, or session drift.
