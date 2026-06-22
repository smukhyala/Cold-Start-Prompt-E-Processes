# Run summary — `webarena_gmail_adaptive_spruce_120_rep3_trial0_2026-06-22T02-26-11Z_resume_from_119`

- T = 120
- alpha = 0.05
- per-arm null m_0 = 0.5
- global log-e = -0.026 (e = 0.975); global null rejected = **False**
- τ_α (first global rejection) = — (never crossed log 1/α)

## Per-arm final state

| arm | a* | pulls | successes | μ̂_a | Δ_a | log_e | CS low | CS high | rejected |
|---|---|---|---|---|---|---|---|---|---|
| algorithmic |  | 11 | 7 | 0.636 | 0.114 | -0.385 | 0.215 | 0.985 | False |
| balanced |  | 8 | 3 | 0.375 | 0.375 | -0.950 | 0.015 | 0.877 | False |
| baseline |  | 13 | 9 | 0.692 | 0.058 | 0.382 | 0.292 | 0.985 | False |
| cautious |  | 7 | 3 | 0.429 | 0.321 | -1.383 | 0.015 | 0.938 | False |
| domain_expert |  | 8 | 4 | 0.500 | 0.250 | -1.030 | 0.062 | 0.969 | False |
| exploratory |  | 10 | 6 | 0.600 | 0.150 | -0.958 | 0.154 | 0.985 | False |
| explorer | ★ | 16 | 12 | 0.750 | 0.000 | 1.688 | 0.400 | 0.985 | False |
| junior_reactive |  | 9 | 5 | 0.556 | 0.194 | -1.193 | 0.077 | 0.954 | False |
| overthinker |  | 8 | 5 | 0.625 | 0.125 | -0.964 | 0.123 | 0.985 | False |
| planner |  | 8 | 3 | 0.375 | 0.375 | -0.929 | 0.015 | 0.877 | False |
| rapid |  | 11 | 5 | 0.455 | 0.295 | 0.000 | 0.077 | 0.877 | False |
| verifier |  | 11 | 6 | 0.545 | 0.205 | -0.377 | 0.154 | 0.938 | False |

Empirical best arm a* = **explorer** (μ̂ = 0.750).

## First-rejection times

- algorithmic: —
- balanced: —
- baseline: —
- cautious: —
- domain_expert: —
- exploratory: —
- explorer: —
- junior_reactive: —
- overthinker: —
- planner: —
- rapid: —
- verifier: —

## Arm geometry — prompt-distance matrix d(a,b)

Weighted, range-normalized Manhattan over the six axes (Mukhyala &
Waudby-Smith § 3.2). 0 on the diagonal; larger values = more dissimilar.

| | algorithmic | balanced | baseline | cautious | domain_expert | exploratory | explorer | junior_reactive | overthinker | planner | rapid | verifier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **algorithmic** | 0.000 | 1.833 | 4.833 | 1.833 | 1.500 | 3.500 | 4.167 | 3.333 | 0.333 | 1.500 | 4.500 | 2.667 |
| **balanced** | 1.833 | 0.000 | 3.000 | 1.667 | 2.000 | 1.667 | 2.333 | 1.500 | 2.167 | 1.000 | 2.667 | 1.500 |
| **baseline** | 4.833 | 3.000 | 0.000 | 3.667 | 4.333 | 2.333 | 1.667 | 2.167 | 5.167 | 3.333 | 1.333 | 2.833 |
| **cautious** | 1.833 | 1.667 | 3.667 | 0.000 | 2.000 | 3.333 | 4.000 | 2.167 | 1.500 | 2.000 | 4.333 | 0.833 |
| **domain_expert** | 1.500 | 2.000 | 4.333 | 2.000 | 0.000 | 2.000 | 2.667 | 3.500 | 1.833 | 1.667 | 3.000 | 2.833 |
| **exploratory** | 3.500 | 1.667 | 2.333 | 3.333 | 2.000 | 0.000 | 1.333 | 2.500 | 3.833 | 2.000 | 1.667 | 3.167 |
| **explorer** | 4.167 | 2.333 | 1.667 | 4.000 | 2.667 | 1.333 | 0.000 | 3.167 | 4.500 | 2.667 | 0.333 | 3.167 |
| **junior_reactive** | 3.333 | 1.500 | 2.167 | 2.167 | 3.500 | 2.500 | 3.167 | 0.000 | 3.667 | 2.500 | 3.500 | 1.333 |
| **overthinker** | 0.333 | 2.167 | 5.167 | 1.500 | 1.833 | 3.833 | 4.500 | 3.667 | 0.000 | 1.833 | 4.833 | 2.333 |
| **planner** | 1.500 | 1.000 | 3.333 | 2.000 | 1.667 | 2.000 | 2.667 | 2.500 | 1.833 | 0.000 | 3.000 | 1.833 |
| **rapid** | 4.500 | 2.667 | 1.333 | 4.333 | 3.000 | 1.667 | 0.333 | 3.500 | 4.833 | 3.000 | 0.000 | 3.500 |
| **verifier** | 2.667 | 1.500 | 2.833 | 0.833 | 2.833 | 3.167 | 3.167 | 1.333 | 2.333 | 1.833 | 3.500 | 0.000 |

