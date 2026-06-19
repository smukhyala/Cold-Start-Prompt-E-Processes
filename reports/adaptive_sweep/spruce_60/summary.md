# Run summary — `webarena_gmail_adaptive_spruce_60_trial0_2026-06-19T21-46-42Z_resume_from_44`

- T = 60
- alpha = 0.05
- per-arm null m_0 = 0.5
- global log-e = 0.152 (e = 1.16); global null rejected = **False**
- τ_α (first global rejection) = — (never crossed log 1/α)

## Per-arm final state

| arm | a* | pulls | successes | μ̂_a | Δ_a | log_e | CS low | CS high | rejected |
|---|---|---|---|---|---|---|---|---|---|
| algorithmic |  | 5 | 3 | 0.600 | 0.275 | -0.575 | 0.015 | 0.985 | False |
| balanced |  | 4 | 2 | 0.500 | 0.375 | -0.706 | 0.015 | 0.985 | False |
| baseline |  | 8 | 6 | 0.750 | 0.125 | 0.827 | 0.262 | 0.985 | False |
| cautious |  | 4 | 3 | 0.750 | 0.125 | -0.389 | 0.015 | 0.985 | False |
| domain_expert |  | 4 | 2 | 0.500 | 0.375 | -0.706 | 0.015 | 0.985 | False |
| exploratory |  | 7 | 5 | 0.714 | 0.161 | 0.236 | 0.169 | 0.985 | False |
| explorer |  | 4 | 2 | 0.500 | 0.375 | -0.706 | 0.015 | 0.985 | False |
| junior_reactive |  | 3 | 2 | 0.667 | 0.208 | -0.693 | 0.015 | 0.985 | False |
| overthinker |  | 4 | 1 | 0.250 | 0.625 | -0.693 | 0.015 | 0.985 | False |
| planner | ★ | 8 | 7 | 0.875 | 0.000 | 1.740 | 0.323 | 0.985 | False |
| rapid |  | 5 | 3 | 0.600 | 0.275 | -0.687 | 0.015 | 0.985 | False |
| verifier |  | 4 | 2 | 0.500 | 0.375 | -0.706 | 0.015 | 0.985 | False |

Empirical best arm a* = **planner** (μ̂ = 0.875).

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

