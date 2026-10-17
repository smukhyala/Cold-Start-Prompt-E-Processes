# Public replay 01 results: Terminal-Bench 2.0

Run 2026-10-12 under `PUBLIC_REPLAY_PREREG_01.md` and its model-ID amendment, at source revision `572b2614be2c0cb2527e14f5b1e4026f1072e6c1`. Code: `experiments/growing_bandits/audit/public_replay_01.py`. Outcome and replay tables: `results/paper/public_replay_01/`.

## Feasibility and exclusions

The leaderboard repository contains per-trial `result.json` files and five task attempts per listed submission. Six submissions were nominated before reading outcomes, with Terminus 2 as the incumbent. The per-trial configurations forced three exclusions: Crux identifies Claude Opus 4.5, Droid `aurora-01-21`, and WozCode only `opus`. Capy and Terminus 2 identify `anthropic/claude-opus-4-6`; Terminus-KIRA identifies `vertex_ai/claude-opus-4-6` and is retained under Amendment 1. Vertex routing could cause differences beyond scaffold design.

Complete two-trial task cells: Terminus 2 76/89, Capy 79/89, Terminus-KIRA 77/89. The common intersection is **68 tasks**. Verifier exceptions caused the missing cells (72, 60, 73 attempted trials respectively). Their records were treated as missing, not failures. Each arm's first two valid trials per task were selected by full path before any reward-based choice.

## R1: A/A under actual rerun noise

Terminus 2's first two valid trials were assigned to two pseudo arms by independent within-task label swaps. Over 2,000 swaps and task orders, E1b crossed at `alpha=0.05` in **4 (0.2%)**, exact binomial 95% interval [0.05%, 0.51%]. Neither unswapped direction crossed. This supports calibration for this conditional label-swap replay. It does not establish a 0.2% future-task error rate, and it does not test futility type I error: identical configurations are on the alternative side of the `delta=0.15` futility null.

## R3: observed gains, no efficacy certification

| Scaffold | Trial 0 paired gain vs Terminus 2 | Task-bootstrap 95% interval | Trial 1 paired gain |
|---|---:|---:|---:|
| Capy | +8.8 pp | [+1.5, +17.6] pp | +10.3 pp |
| Terminus-KIRA | +5.9 pp | [-2.9, +14.7] pp | +10.3 pp |

Across 2,000 random task orders, the fixed two-challenger E1b mixture crossed **0 times** at `alpha=0.05`. In the lexicographically first order the largest e-value was 3.45, below 20. Every replay used all 68 tasks and 204 arm-task episodes; no early-stop saving was observed. Neither challenger crossed its `delta=0.15` futility boundary in the fixed task order. The preregistered independent rerun winner check did not trigger because no primary efficacy crossing selected a winner.

The positive point gains are encouraging and repeat across trial indices, especially for Capy. The registered sequential procedure does **not** certify a winner at this task count. The bootstrap interval describes fixed-benchmark task variation after the candidate set was fixed; it is not a substitute for the registered anytime-valid decision. Leaderboard submissions were run at different times and can differ in settings, so these gains are associations, not isolated scaffold effects.

## Implication for the framework plan

R1 supplies a real-noise calibration check, with the stated conditional interpretation. R3 has not filled the paper's top gap: a certified real winner. A larger paired task set or a more powerful prespecified design is needed before writing that claim. The registered 68-task sample is materially smaller than the planned 89-task benchmark because exceptions occurred in different task cells across submissions.
