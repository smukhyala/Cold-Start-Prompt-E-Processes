# Pre-registration 13: results (run 2026-10-10)

Registration: `docs/growing_bandits/DEPLOYMENT_PLAN.md`, Pre-registration 13 (commit e071393), and Amendment 1
(commit 3b9b620). Both were committed before any episode was run. Every number below comes from the frozen
analyzers' output in `results/winner_check_gitlab/`, `results/winner_check_gmail/` and `results/noise_recheck_gmail/`
(gitignored, regenerable with the commands in each section). Diagnostics labelled **exploratory** were not
registered.

## Bottom line

| Leg | Registered question | Registered reading | One-line result |
|---|---|---|---|
| G (primary, GitLab) | Is the search's winner ≥ 10 points better than the default, on 110 fresh tasks? | **Uninformative**: the positive control failed | W − D = −0.9 pp (one-sided 95% UB +3.4 pp), which would certify. But the planted answer-key prompt was only +3.6 pp (LB −1.2 pp), so the registered rule voids the certification |
| M (Gmail) | Does G_14's +10-point in-sample lead hold on 30 new tasks? | **Shrank** | G_14 − default = 0.0 pp (90% CI ±9.1 pp). The lead went from +10 to 0 |
| N (Gmail) | Has rerun noise changed since September? | Descriptive | Disagreement 20% now against 10% in September (p = 0.12): about 2× higher, but not significant at n = 30 |

**What this means for the paper.**
- **Winner's curse (Leg M): confirmed for Gmail at the +10 scale,** under the registered rule. The margin is narrow:
  the upper bound of +9.1 pp sits just under the +10 threshold. The confounded as-collected comparison has an upper
  bound of +11.2 pp and would not pass.
- **"The search's winner is no better than the default" (Leg G): still exploratory.** The point estimate and bound
  are exactly what the claim predicts. The study, however, did not show that it *could* have detected a real
  winner on these tasks, so by the rule we registered, it cannot confirm anything.
- **Why the control failed (exploratory diagnosis below):** the planted prompt's cookbook was written for the
  original 40-task GitLab set. Only 20 of the 110 fresh tasks are in that set. The registration treated it as a
  control on all 110, and that was a design mistake.

## Operations

| Leg | Episodes | Status | Spent | Cap | Sleep-spanning (Amendment 1) |
|---|---|---|---|---|---|
| N | 60 | all `ok` | $4.82 | $8 | 6 archived and re-run |
| M | 60 (default) + 60 frozen G_14 pilot | all `ok` (1 infra_error retried) | $3.52 | $8 | 0 |
| G | 660 | all `ok` | $67.47 | $120 | 0 |

- **Total spend:** $75.81, against a registered estimate of about $102.
- **Step cap:** enforced (fix bd3fff5). Under the 30-step rescore, 5 Leg G records and 1 Leg M record change from
  recorded success to failure. These are the 31-step episodes disclosed in Amendment 1.
- **Run timing:** Leg N took 10.8 h because the Mac slept (Amendment 1). Leg M took 3.8 h and Leg G 3.7 h with 4
  workers.

## Leg G: GitLab winner check (primary)

`./.venv/bin/python experiments/growing_bandits/retrospective/analyze_winner_check.py`

**Arms:** W = GLK_25, the old-data GLK winner; D = GL_anchor_baseline; O = GL_anchor_oracle. **Data:** 110 tasks × 2
replicates, 30-step rescore.

| | Mean success | vs D (mean, paired over tasks) | One-sided 95% bound | Registered test |
|---|---|---|---|---|
| D | 0.400 | — | — | — |
| W | 0.391 | −0.009 (se 0.026) | UB **+0.034** | < 0.10 → certify ✓ |
| O | 0.436 | +0.036 (se 0.029) | LB **−0.012** | > 0 → control ✗ |

- **Registered reading:** G-control fails, so the leg is **uninformative**, "whatever G-primary says". The
  analyzer reports `study_informative: false`, `certification_valid: false`.
- **Task-cluster bootstrap (secondary):** it agrees. W − D has UB +0.032 and 0% of resamples are ≥ 0.10. O − D has
  LB −0.009, and 12% of resamples are ≤ 0.
- **Anytime-valid reference (secondary), in queue order:**
  - E5 futility against δ = 0.10 crosses log 20 at task **65 of 110**; against δ = 0.15 it crosses at task **49**.
  - E1b efficacy of O vs D never crosses: the maximum log-e is 0.93, against a threshold of 3.0.
- **Rerun disagreement, replicate 0 vs 1:** 16% for W, 16% for D and 18% for O, over 110 tasks.

**Why the positive control failed (exploratory, not registered).**
- O is a cookbook prompt. It hard-codes routes and ids for the tasks of the June 40-task GitLab strong-arm set.
- Of the 110 fresh tasks, 20 are in that set and 90 are not:

| Subset | n | O | D | W | O − D (se) | W − D (se) |
|---|---|---|---|---|---|---|
| In O's cookbook (June 40-set) | 20 | 0.850 | 0.750 | 0.650 | +0.100 (0.069) | −0.100 (0.069) |
| Not in O's cookbook | 90 | 0.344 | 0.322 | 0.333 | +0.022 (0.032) | +0.011 (0.027) |

- On the tasks it has answers for, O keeps roughly its old edge: +10 pp here, against +13 pp on the old block A.
- On the other 90 it is an ordinary prompt.
- The registration should have used only cookbook-covered tasks for the control, or built a control whose
  advantage holds on every task. That was not caught before the run. **It is recorded here, and the registered
  reading is not changed.**
- The 90 uncovered tasks are also much harder (D 0.32 against 0.75), so this split confounds coverage with
  difficulty.

## Leg M: Gmail winner's-curse test

`./.venv/bin/python experiments/growing_bandits/retrospective/analyze_winner_check_gmail.py`

| | G_14 | default | d = G_14 − default | 90% CI (paired t) | Registered reading |
|---|---|---|---|---|---|
| September, in sample (30 Pre-reg 9 tasks, 1 run each) | 0.70 | 0.60 | **+0.10** | — | (the selected lead) |
| **Primary:** held out, 30-step rescore (30 pilot tasks × 2 runs) | 0.717 | 0.717 | **0.000** | [−0.091, +0.091] | UB < 0.10 → **shrank** |
| Secondary, confounded: as collected (G_14 uncapped) | 0.733 | 0.717 | +0.017 | [−0.079, +0.112] | (not registered) |

- **Task bootstrap (primary):** the 90% interval is [−0.083, +0.083]; 4.8% of resamples have d ≥ 0.10.
- **Margin:** the registered "shrank" reading holds, but narrowly (UB 0.091). Thirty tasks give low power, as the
  registration disclosed.
- **Confounds, disclosed in the registration:**
  - The G_14 runs are from the 2026-10-09 pilot, with no step cap; the default ran on 2026-10-10 with the cap.
  - The analyzer's disclosure string says the default ran "after 2026-10-12". That is a stale date in its text: the
    runs are dated 2026-10-10 in the logs. The analysis is unaffected.
- **Rerun disagreement, run 1 vs run 2 on the same task:** G_14 37% (30-step) and default 23%, against 10.3% for
  September's G repeat cells. Leg N tests whether that rise is real.

## Leg N: Gmail noise re-check

`./.venv/bin/python experiments/growing_bandits/retrospective/analyze_noise_recheck.py`

| | Now (2026-10-10) | September, same 30 cells | September, all 145 G repeat cells |
|---|---|---|---|
| Disagreement (the 2 runs differ) | **6/30 = 20%** (95% CP 7.7%–38.6%) | 3/30 = 10% | 15/145 = 10.3% |
| Mean success | 0.50 | 0.58 | — |
| Cost per episode | $0.075 | $0.096 (ratio 0.78) | — |

- **Exact binomial test of 6/30 against 0.103:** p = 0.12, two-sided.
- **Paired McNemar on the same cells:** 5 cells disagree now only, 2 disagree in September only, and 1 disagrees in
  both. p = 0.45.
- **Reading (descriptive, as registered).** Disagreement is about twice September's, but it is **not significantly
  raised** at n = 30. The pilot's 33% does not replicate as a clear shift.
  - Leg M's 23–37% is on different cells and arms, so it is not directly comparable.
  - Success and cost per episode are both lower now, which is consistent with a provider-side change and with the
    step cap now being enforced. This cannot separate the two.
- **For the paper's limitations:** "rerun disagreement on re-sampled September cells was 20% [8%, 39%] against 10%
  in September (p = 0.12); provider drift between runs is possible and is not controlled"

## What would make Leg G informative (needs a new registration; not run)

- **Option 1, cheapest.** A new control arm whose advantage holds on every one of the 110 tasks. For example, a
  cookbook generated from each task's verifier, run alongside a fresh D. This needs about 440 episodes, roughly
  $45, plus a new registration.
- **Option 2.** Re-score Leg G on only the 20 cookbook-covered tasks. This is post hoc and badly underpowered, so it
  is not recommended as a confirmation.
- **Option 3.** Accept Leg G as uninformative. The paper reports the result in full, as registered: the point
  estimate and the futility e-process are consistent with "no better than the default", but this is not confirmed.
