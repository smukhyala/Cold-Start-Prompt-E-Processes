# Retrospective repeat and drift audit (2026-10-12)

This protocol fixes a new analysis of **already seen** outcomes. It is not a prospective or blinded preregistration. The pool repeat counts, Pre-registration 15's +15 pp default drift, and the existing historical-baseline code were inspected before writing it. No new episodes or model spend are planned.

## R1: within-cell repeat check

Use `results/audit/matrices/{G,F,GLK30,GLK}.csv`. For each pool, keep exactly the `(arm_id, task_id)` cells with both replicate 0 and 1 and binary outcomes, sorted by `(task_id, arm_id)`. GLK30 is primary; GLK uses the same underlying episodes under another scoring cap and is secondary. Do not combine the four pools as independent experiments.

For 2,000 replays per pool, use NumPy `SeedSequence([20261012, 11, pool_index, replay])` (G/F/GLK30/GLK indices 0/1/2/3) to swap the replicate labels independently within each cell, then permute cell order. Run one-challenger E1b at `alpha=0.05`. Report crossings, an exact binomial interval, median discordance, and unswapped directions. This is a **conditional label-exchangeability** check. Because prompts and tasks recur across cells, the replays are not independent live studies or a proof of a future-session error rate. The `delta=0.15` futility test is under its alternative on A/A cells and has no false-call interpretation here.

## R2: historical baseline bias under session drift

Use the matched 30 Leg-M tasks in `results/headroom_gmail/task_table.csv` and the prior default scores extracted by `analyze_headroom_gmail.leg_m_default_scores()`. Score every episode at the existing 30-step cap. For each task, the historical bank is the mean of Leg-M default replicates 0/1; current default and answer-key arm use their Pre-registration 15 replicates 0/1.

Report the task-mean historical and current default scores and the paired difference. On each of the two current replicate columns, calculate full-path efficacy e-values (`alpha=0.05`) for: (a) current default versus its own historical plug-in bank; (b) current default replicate 0 against replicate 1 as a concurrent A/A diagnostic; (c) current answer-key arm versus historical bank; and (d) current answer-key arm versus concurrent current default. For (a) and (c), use `known_baseline_x` and `efficacy_arm_log_paths`; for (b) and (d), use E1b. Use one fixed task order, sorted by task ID, and 2,000 random-order sensitivities from `SeedSequence([20261012, 12, replay])`. Report first crossing and maximum e-value, not only a final score.

As descriptive sensitivity only, rerun (a) and (c) with the historical bank shifted upward by the **observed** 0.15, clipped to 1, and with per-task Clopper–Pearson upper bounds from two historical replicates at family level 0.025. Neither adjustment has a guaranteed future-session coverage claim here: 0.15 was estimated from these outcomes, and per-task historical intervals do not cover a changing success probability without a drift assumption. If no test crosses in 30 tasks, report the gain distortion and lack of crossing; do not claim to have demonstrated a false rejection.
