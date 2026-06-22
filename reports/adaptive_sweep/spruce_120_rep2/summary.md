# Run summary — `webarena_gmail_adaptive_spruce_120_rep2_trial0_2026-06-21T07-50-52Z_resume_from_120`

- T = 120
- alpha = 0.05
- per-arm null m_0 = 0.5
- global log-e = 0.721 (e = 2.06); global null rejected = **False**
- τ_α (first global rejection) = — (never crossed log 1/α)

## Per-arm final state

| arm | a* | pulls | successes | μ̂_a | Δ_a | log_e | CS low | CS high | rejected |
|---|---|---|---|---|---|---|---|---|---|
| algorithmic |  | 11 | 7 | 0.636 | 0.153 | -0.391 | 0.215 | 0.985 | False |
| balanced |  | 12 | 8 | 0.667 | 0.123 | 0.066 | 0.262 | 0.985 | False |
| baseline |  | 9 | 4 | 0.444 | 0.345 | -0.613 | 0.062 | 0.938 | False |
| cautious |  | 10 | 7 | 0.700 | 0.089 | -0.153 | 0.231 | 0.985 | False |
| domain_expert | ★ | 19 | 15 | 0.789 | 0.000 | 2.904 | 0.462 | 0.985 | False |
| exploratory |  | 7 | 4 | 0.571 | 0.218 | -0.983 | 0.046 | 0.985 | False |
| explorer |  | 9 | 6 | 0.667 | 0.123 | -0.884 | 0.154 | 0.985 | False |
| junior_reactive |  | 8 | 3 | 0.375 | 0.414 | -0.693 | 0.015 | 0.862 | False |
| overthinker |  | 9 | 6 | 0.667 | 0.123 | -0.349 | 0.185 | 0.985 | False |
| planner |  | 8 | 3 | 0.375 | 0.414 | -0.929 | 0.015 | 0.877 | False |
| rapid |  | 10 | 6 | 0.600 | 0.189 | -0.710 | 0.185 | 0.985 | False |
| verifier |  | 8 | 4 | 0.500 | 0.289 | -0.867 | 0.077 | 0.985 | False |

Empirical best arm a* = **domain_expert** (μ̂ = 0.789).

## First-rejection times

- algorithmic: —
- balanced: —
- baseline: —
- cautious: —
- domain_expert: 96
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

