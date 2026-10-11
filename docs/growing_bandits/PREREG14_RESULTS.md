# Pre-registration 14: results (run 2026-10-10)

Registration: `docs/growing_bandits/DEPLOYMENT_PLAN.md`, Pre-registration 14 (commit 67c58ed), committed before any
study episode. The frozen files match their registered hashes: `analyze_control_check.py` (sha256 993928c9…) and
`build_full_oracle.py` (fe0d44c1…). Every number below comes from the frozen analyzer's output in
`results/control_check_gitlab/` (gitignored; regenerate with
`PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_control_check.py`). Diagnostics
labelled **exploratory** were not registered.

## Bottom line

| Registered question | Registered rule | Result | Registered reading |
|---|---|---|---|
| Does a prompt with a real advantage (the full answer key, O2) beat the default on Leg G's 110 tasks? | One-sided 95% LB of mean(O2 − Dnew) > 0 | +0.055 (se 0.028), **LB +0.007** | **Control holds** |
| Combined with Pre-reg 13 Leg G (W − D UB +0.034 < 0.10) | Control holds and UB < 0.10 | Both true | **"Leg G certification is informative with a next-day control"** |

**The paper may now state, as registered:** "on 110 held-out GitLab tasks, the search's winner was not ≥ 10 points
better than the default (Pre-reg 13), and on the same tasks a prompt with a real advantage was detected (Pre-reg 14,
next session)." It must disclose the two-session design.

**How strong this is, honestly:**
- **The pass is narrow.** The lower bound clears zero by 0.7 points. The registration put power at about 0.5 for a
  true +5 points, and the observed effect is +5.5. The task-cluster bootstrap agrees: LB +0.009, with 2.8% of
  resamples ≤ 0.
- **It leans on the same-session default (exploratory, below).** Against Leg G's earlier default, O2 is only
  +0.018 (LB −0.037). The registered drift check does *not* trigger a caveat, because its 90% CI includes 0, but
  the new default scored 3.6 points lower than the old one. Part of the margin is that dip.
- **Read it as:** the design *can* detect an advantage of about +5 points on these tasks, against a concurrent
  default. It is not strong evidence of power at smaller effects.

## Operations

| | Episodes | Status | Spent |
|---|---|---|---|
| Pilot v1 (block A, not study data) | 30 | gate failed (0.633) | $1.91 |
| Pilot v2 (block A, not study data) | 30 | gate passed (0.800) | $2.33 |
| Study | 440 | all `ok` | $46.74 |
| **Total** | | | **$50.98** |

- **Budget:** $46.74 against the registered hard stop of $60 and an expected $40–45. It did not stop on budget.
- **Sleep (Amendment 1 of Pre-reg 13):** the lid was closed on battery from 19:12 to 22:07 PDT, against the
  registered run condition (AC, lid open).
  - 4 in-flight records had wall-clock times of about 10,600 s against a 600 s limit. They were archived to
    `logs/control_check_gitlab/archive/` and re-run with the same command ($0.61).
  - A second archive pass found 0. All 440 records are now `ok`.
- **Step cap:** 0 episodes ended by the step cap and 0 by the clock. 4 records change from recorded success to
  failure under the 30-step rescore.
- **Per arm:** O2 averaged $0.108 and 18.7 steps per episode; D averaged $0.102 and 21.5 steps.

## Primary

| | Mean success | vs Dnew (paired over 110 tasks) | One-sided 95% LB | Registered test |
|---|---|---|---|---|
| Dnew (`GL_anchor_baseline`) | 0.364 | — | — | — |
| O2 (`GL_oracle_full`) | 0.418 | **+0.055** (sd 0.298, se 0.028) | **+0.007** | > 0 → control holds ✓ |

## Secondary (registered)

1. **Task-cluster bootstrap** (10,000 resamples, seed 20261012): LB +0.009, 90% interval [+0.009, +0.100]. 2.8% of
   resamples are ≤ 0.
2. **Day-to-day drift** (Dnew − Dold, the same default prompt in Leg G and now): −0.036 (se 0.031), 90% CI
   [−0.087, +0.015]. The CI includes 0, so the registered rule adds **no drift caveat**. Dold is 0.400 and Dnew is
   0.364.
3. **Leg G, recomputed from its logs:** W − Dold = −0.009, UB +0.034. It still certifies.
4. **Old-cookbook split (exploratory):**

| Subset | n | O2 | Dnew | Dold | O2 − Dnew (se) |
|---|---|---|---|---|---|
| In the old oracle's cookbook | 20 | 0.750 | 0.750 | 0.750 | 0.000 (0.063) |
| Not in the cookbook | 90 | 0.344 | 0.278 | 0.322 | +0.067 (0.032) |

## Exploratory diagnostics (not registered)

| Comparison | Mean | One-sided 95% LB | Note |
|---|---|---|---|
| O2 − Dold (Leg G's default, earlier session) | +0.018 | −0.037 | Would not pass against the earlier default |
| O2 − mean(Dnew, Dold) | +0.036 | −0.008 | Would not pass against the pooled default |
| O2 − W (Leg G's winner) | +0.027 | (se 0.032) | — |

- **The full answer key is no stronger than the old 40-task cookbook.**
  - Overall, O2 scores 0.418 and the old O scored 0.436 in Leg G.
  - On the 90 tasks the old cookbook did not cover, both score 0.344.
  - Giving `gpt-5.4-mini` the exact ids and target values for every task lifts it only about 5 points over the
    default. On GitLab, the binding constraint is execution (navigating, editing and saving within 30 steps), not
    knowing the answer.
- **Rerun disagreement:** replicate 0 and replicate 1 differ on 23.6% of tasks for O2 and 21.8% for Dnew. In Leg G it
  was 16–18%. That is consistent with the Leg N noise rise (20% against 10% in September), and is not tested.
- **The pilot does not transfer.** O2 scored 0.80 on block A and 0.42 on the study tasks, because the study tasks are
  much harder (the default is about 0.36–0.40 here against 0.73 on block A).

## What this means for the paper

- **Leg G moves from exploratory to registered-informative**, under the combined reading registered in advance.
  The claim "the search's winner on GitLab is not ≥ 10 points better than the default" is now confirmed on 110
  held-out tasks, with W − D = −0.9 pp (UB +3.4 pp).
- **Required disclosures:**
  - the two-session design;
  - the narrow control margin (LB +0.7 pp);
  - that the control would not pass against the earlier session's default (exploratory);
  - the machine sleep and the re-run of 4 episodes;
  - the pilot gate passed exactly, on the second header version.
- **Substantive finding worth a sentence:** even a perfect answer key buys only about 5 points on these tasks. The
  ceiling on what any prompt can do for this agent on GitLab is low, which supports the flat-pool story directly.
