# Novelty and progress update — 2026-10-12

## Positioning after the literature check

The paper is an **empirical application and comparison of established sequential methods**, not a claim to have invented sequential agent evaluation. [TRIPLE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b46bc1449205888e1883f692aff1a252-Abstract-Conference.html) already studies budgeted prompt selection; [PACE](https://arxiv.org/abs/2606.08106) already uses a paired, anytime-valid test to accept agent changes; [SERPANT](https://proceedings.mlr.press/v306/gu26l.html) uses e-processes for LLM ranking; and [Stop Guessing When to Stop Testing](https://arxiv.org/abs/2607.08522) proposes sequential model evaluation with practical margins and futility. The winner's curse in LLM self-improvement has also been measured directly in [Hu and Tang](https://arxiv.org/abs/2610.09239). The three arXiv-only papers cited here are preprints as of this update.

The defensible contribution is the *joint empirical accounting* of incumbent-relative efficacy, formally tested practical futility, and unresolved decisions across live web-agent studies and several public prompt and agent matrices. It shows how the incumbent and task population alter the answer, and distinguishes fresh sampled tasks from historical benchmark replays. This is a specific evidence and scope contribution. A literature search does not establish priority for the exact combination, and the draft should make no first-of-its-kind claim.

## What has been completed and what failed

- Registered live runs show practical-futility decisions and unresolved searches. The fixed public Linear bank crossed 15-point futility at pair 93 with 64/120 versus 66/120 successes; the later newly sampled profile-edit trial crossed at pair 44 with 3/120 versus 1/120 successes. The latter addresses fresh sampling for its narrow population but sits at a severe capability floor.
- Public replays show both efficacy crossings and unresolved cases, while held-out prompt evaluations quantify how the best search score can overstate a gain. Those replays are descriptive because they reuse public outcomes and cannot inherit a prospective future-task error guarantee merely from a registered analysis order.
- The pilot-based tasks-to-decision predictor failed its cross-dataset transfer check. The manuscript retains exact full-grid cost arithmetic but makes no universal stopping-time or episode-saving promise.
- A measured historical-default shift reversed one comparison's sign. It showed bias and drift risk, not a false sequential rejection.

## Decision for the next phase

Freeze new empirical collection for this draft. The immediate work is to finish the related-work comparison, obtain an independent mathematical and implementation review of the conditional-null claims, run the reproducibility pass, and prepare the submission. A broader sampled-task experiment is a possible response to a concrete reviewer or venue requirement for general agent-discrimination evidence; it is not an automatic prerequisite for the scoped empirical paper.
