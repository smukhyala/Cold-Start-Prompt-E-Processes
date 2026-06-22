# Run summary — `webarena_gmail_uniform_multiarm_120_rep3_trial0_2026-06-21T21-04-04Z_resume_from_103`

- T = 120
- alpha = 0.05
- per-arm null m_0 = 0.5
- global log-e = -0.627 (e = 0.534); global null rejected = **False**
- τ_α (first global rejection) = — (never crossed log 1/α)

## Per-arm final state

| arm | a* | pulls | successes | μ̂_a | Δ_a | log_e | CS low | CS high | rejected |
|---|---|---|---|---|---|---|---|---|---|
| algorithmic |  | 10 | 6 | 0.600 | 0.100 | -0.848 | 0.154 | 0.985 | False |
| balanced |  | 10 | 6 | 0.600 | 0.100 | -1.115 | 0.138 | 0.969 | False |
| baseline | ★ | 10 | 7 | 0.700 | 0.000 | 0.193 | 0.246 | 0.985 | False |
| cautious |  | 10 | 5 | 0.500 | 0.200 | -0.306 | 0.108 | 0.938 | False |
| domain_expert |  | 10 | 6 | 0.600 | 0.100 | -0.848 | 0.154 | 0.985 | False |
| exploratory |  | 10 | 6 | 0.600 | 0.100 | -0.848 | 0.154 | 0.985 | False |
| explorer |  | 10 | 4 | 0.400 | 0.300 | -0.867 | 0.015 | 0.831 | False |
| junior_reactive |  | 10 | 5 | 0.500 | 0.200 | -1.175 | 0.062 | 0.908 | False |
| overthinker |  | 10 | 4 | 0.400 | 0.300 | -1.133 | 0.015 | 0.831 | False |
| planner |  | 10 | 6 | 0.600 | 0.100 | -1.095 | 0.138 | 0.969 | False |
| rapid |  | 10 | 6 | 0.600 | 0.100 | -0.769 | 0.169 | 0.985 | False |
| verifier |  | 10 | 3 | 0.300 | 0.400 | 0.000 | 0.015 | 0.769 | False |

Empirical best arm a* = **baseline** (μ̂ = 0.700).

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

