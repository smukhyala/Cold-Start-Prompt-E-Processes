# Run summary — `webarena_gmail_adaptive_spruce_120_trial0_2026-06-20T23-35-23Z_resume_from_113`

- T = 120
- alpha = 0.05
- per-arm null m_0 = 0.5
- global log-e = -0.184 (e = 0.832); global null rejected = **False**
- τ_α (first global rejection) = — (never crossed log 1/α)

## Per-arm final state

| arm | a* | pulls | successes | μ̂_a | Δ_a | log_e | CS low | CS high | rejected |
|---|---|---|---|---|---|---|---|---|---|
| algorithmic |  | 9 | 5 | 0.556 | 0.178 | -0.869 | 0.123 | 0.985 | False |
| balanced | ★ | 15 | 11 | 0.733 | 0.000 | 1.282 | 0.369 | 0.985 | False |
| baseline |  | 10 | 6 | 0.600 | 0.133 | -0.813 | 0.185 | 0.985 | False |
| cautious |  | 10 | 6 | 0.600 | 0.133 | -0.632 | 0.169 | 0.985 | False |
| domain_expert |  | 11 | 7 | 0.636 | 0.097 | -0.309 | 0.215 | 0.985 | False |
| exploratory |  | 8 | 5 | 0.625 | 0.108 | -0.874 | 0.108 | 0.985 | False |
| explorer |  | 8 | 4 | 0.500 | 0.233 | -0.800 | 0.092 | 0.969 | False |
| junior_reactive |  | 8 | 4 | 0.500 | 0.233 | -1.118 | 0.015 | 0.938 | False |
| overthinker |  | 8 | 5 | 0.625 | 0.108 | -0.755 | 0.123 | 0.985 | False |
| planner |  | 12 | 8 | 0.667 | 0.067 | -0.351 | 0.246 | 0.985 | False |
| rapid |  | 10 | 6 | 0.600 | 0.133 | -0.120 | 0.169 | 0.985 | False |
| verifier |  | 11 | 5 | 0.455 | 0.279 | 0.000 | 0.077 | 0.877 | False |

Empirical best arm a* = **balanced** (μ̂ = 0.733).

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

