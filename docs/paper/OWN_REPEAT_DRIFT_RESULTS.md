# Live-repeat and drift audit results

Analysed 2026-10-12 under `OWN_REPEAT_DRIFT_PROTOCOL.md`. These outcomes were known before the protocol was written; the audit is retrospective. Code: `experiments/growing_bandits/audit/own_repeat_drift.py`; derived table: `results/paper/own_repeat_drift/summary.json`.

## R1: same prompt, same task, two live episodes

| Pool | Paired repeat cells | Discordant cells | E1b crossings in 2,000 label-swap/order replays |
|---|---:|---:|---:|
| G | 145 | 15 | 12 (0.6%) |
| F | 155 | 27 | 43 (2.15%) |
| GLK30 | 120 | 19 | 15 (0.75%) |
| GLK, alternate cap | 120 | 23 | 38 (1.9%) |

Every observed fraction is below the nominal 5% efficacy threshold; neither unswapped replicate direction crossed in any pool. The repeat cells span 49, 49 and 38 prompts (38 for both GitLab scorings), with only 1–7 repeat tasks per prompt. These are conditional label-swap checks on a fixed set of episodes. They do not establish a future-session false-call rate. GLK and GLK30 rescore the same episodes and are not independent replications. Same-configuration repeats lie on the alternative side of the 15-point futility null, so no futility error-rate claim follows.

## R2: what the observed default drift changes

Across 30 matched Gmail tasks, the same default scored **71.7%** in Leg M and **86.7%** in Pre-registration 15, a **+15.0-point** shift. The current answer-key arm scored 73.3%. Its estimated gain is **+1.7 points against the historical plug-in default** and **−13.3 points against the concurrent default**. Reusing the old bank changes the answer by exactly the observed 15 points and reverses its sign.

No efficacy process crossed 20 in the fixed task order or in any of 2,000 task-order replays across the registered scenarios. The largest fixed-order plug-in e-value for the *same* current default against its older estimates was 5.39; the concurrent A/A comparison never exceeded 1. These 30 tasks demonstrate substantial effect-estimate bias from a nonconcurrent control, **not** a false efficacy declaration in this sample.

The sensitivity adjustments also did not cross. Adding the observed 0.15 to each old task estimate and clipping at one raised the bank's average only from 0.717 to 0.777, because many tasks were already at one; it cannot be treated as a guaranteed correction for the 0.15 aggregate drift. Per-task Clopper–Pearson upper bounds from only two old replicates had mean 0.995, too conservative to be useful here. More fundamentally, historical confidence bounds cover estimation error around historical task probabilities, not an unbounded change in those probabilities. A valid known-baseline claim across sessions needs either a prospectively justified bound on drift or a concurrently evaluated default.

## Claim revision

The framework plan's proposed R2 wording—"show the plug-in test failing, and the concurrent-default and Hoeffding-shifted versions holding"—is too strong for these data. The defensible result is a 15-point bias and sign reversal, with no false rejection observed. The live-repeat checks support conditional calibration in the measured cells, while the drift example shows why concurrent controls are the safer default design.
