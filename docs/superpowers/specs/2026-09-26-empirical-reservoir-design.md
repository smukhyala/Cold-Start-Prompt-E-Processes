# Empirical reservoirs — design (2026-09-26)

*Branch `empirical-reservoir`, cut from `fix/post-merge-corrections` at `a6f780e`. Registers as
Pre-registration 9 in `docs/growing_bandits/DEPLOYMENT_PLAN.md`; results become §13 of
`DEPLOYMENT_RESULTS.md`.*

## 1. Why

Every result in the growing-bandits study (§1.0, §12.1–12.8) was measured on synthetic reservoirs: parametric
distributions of arm means (Beta, power-tail, mixtures; `src/cold_start/growing/reservoirs.py`) with Bernoulli
pulls. The paper's claims — search-vs-refine reduces to choosing a final arm count K; K\* grows like T^0.75;
the level rule `T^0.75 · exp(4 · (0.5 − level))` reproduces the learned policy; a corpus-tuned schedule
transfers — have never touched a real prompt. This study measures real prompt reservoirs on WebArena Gmail and
re-runs the registered contrasts on them.

**Success** = (a) we know where real Gmail prompt pools sit on the simulation's map (level, spread, tail gap,
K\*), and (b) the Pre-registration 9 contrasts return a verdict — including the pre-registered verdict
"uninformative: K barely matters on real prompts", which is itself a scoping result for the paper.

## 2. What the existing logs already say (measured 2026-09-26)

Gmail only (1,631 deduplicated episodes, 12 hand-written arms, 60 tasks, `gpt-5.4-mini`, low effort):

- Arm rates 0.49–0.66. After subtracting binomial noise the **true between-prompt SD is ≈ 0.028**.
- **Task difficulty dominates**: per-task rates span 0.00–1.00; most tasks are near-deterministic.
- Within-(arm, task) outcome variance ≈ 0.093, so a prompt's mean over 60 fixed tasks run once each carries
  noise SD ≈ **0.039** (0.056 at 30 tasks). Measurement noise is comparable to the signal; raw per-prompt
  rates would roughly double the apparent spread. This is why §4 deconvolves.
- The simulation corpus rejects environments with spread (q99 − q01) < 0.05 (`MIN_SPREAD`). A real pool may
  lie outside the corpus's support; that is reported, not hidden.

(The earlier "0.45–0.58" figure quoted in conversation pooled Gmail with Shopping and is wrong for Gmail.)

## 3. Data collection

### 3.1 Pools (frozen and committed, with content hashes, before any paid run — gate G0)

- **G (grid):** 50 prompt vectors sampled uniformly without replacement from the 2,304-point grid of
  `configs/axes.yaml` (4·4·3·4·4·3), fixed seed recorded in the pool file, rendered with the existing
  `configs/template.jinja`.
- **F (free-form):** 50 agent instructions written by Claude from one fixed generation prompt asking for
  diverse guidance for a browser agent doing email tasks. The generator never sees the task bank. Stored as
  `Arm.prompt_guidance`, rendered by a new `configs/template_freeform.jinja` that emits the guidance alone. No
  adapter change.
- **Anchor:** the hand-written `baseline` arm, all 60 tasks, as a drift check against its historical 0.66.

Pool files: `data/empirical_pool/pool_G.yaml`, `pool_F.yaml` (arm id, vector or guidance text, sha256).

### 3.2 Episodes

- Every prompt × every one of the 60 Gmail `real-tasks`, once: 2 × 50 × 60 = 6,000, plus 60 anchor.
- **Replicates:** 300 (prompt, task) pairs drawn uniformly at random (fixed seed) from the 6,000 are run a
  second time, to estimate within-cell outcome variance directly (§4.1).
- Agent settings identical to the historical Gmail runs: `llm_provider: openai`, `llm_model: gpt-5.4-mini`,
  `llm_reasoning_effort: low`, `max_agent_steps: 30`, `timeout_s: 180`, `use_vision: false`, headless.
- **Order:** all 6,360 episodes are shuffled once by a fixed seed into a single global queue, so temporal
  drift (API, server) spreads evenly over prompts and pools.

### 3.3 Execution

- New runner `experiments/growing_bandits/empirical/collect.py`: N = 8 workers, worker w owns a WebArena
  server on port 8001 + w and its own artifacts dir; each appends to `logs/empirical_pool/worker_<w>.jsonl`
  in the existing JSONL schema plus `pool`, `replicate` and `status` fields.
- **Resumable:** on start it reads all worker logs and skips (arm, task, replicate) triples with a terminal
  status. A watchdog restarts dead workers (pattern of `scripts/watchdog_*.py`).
- **Infra failures ≠ task failures.** Server crash, API error, or harness exception → `status: infra_error`,
  retried up to 2 times; still failing → `status: missing`, excluded (never scored 0). If `RunResult` cannot
  currently distinguish these, a minimal adapter change makes it do so (tested).
- **Budget stop:** the runner sums logged `cost_usd` and halts all workers at **$260**.

## 4. Pool estimation

### 4.1 Noise model

A prompt's score is a sum of 60 heterogeneous Bernoullis, not a binomial; the binomial variance would
over-correct and flatten the pool. Measurement variance for prompt i is estimated as
σ̂ᵢ² = v̂ / nᵢ, where nᵢ is its non-missing task count and v̂ is the within-cell outcome variance from the 300
replicates (pooled; per-pool v̂ reported). If the two pools' v̂ differ by more than their bootstrap SE, each
pool uses its own.

### 4.2 Deconvolution (primary pools)

The mixing distribution of true prompt rates μᵢ is estimated by the **NPMLE** under
x̄ᵢ ~ N(μᵢ, σ̂ᵢ²) (Kiefer–Wolfowitz; EM on a fixed 400-point grid on [0, 1], convergence tolerance 1e−8 on
the log-likelihood). The output is a discrete distribution (atoms, weights): `G*` and `F*`, the **primary
reservoirs**.

### 4.3 Sensitivity reservoirs (reported, no verdicts)

- **Raw:** atoms = observed x̄ᵢ, equal weights. Over-dispersed by construction — an upper bound on how much
  search can matter.
- **Parametric:** the study's own families (Beta, `TailReservoir`, `MixtureReservoir`) fit by maximum
  marginal likelihood under the same noise model; best by AIC. Smooth tail for long horizons.

### 4.4 Simulator integration

New `EmpiricalReservoir(atoms, weights)` in `reservoirs.py`, registered like the others, implementing
`_icdf` (discrete step inverse) and `_survival`; construction with `validate=False` and the
`validate_reservoir` outcome recorded, since a real pool may fail `MIN_SPREAD`. Everything downstream —
CRN streams, `run_cell`, recommenders, decomposition, `registered_contrast.py` — is unchanged; a replay is a
new `CellSpec` whose reservoir spec points at a frozen pool file.

## 5. Replay cells and policies

- **Cells:** {G, F} × {primary, raw, parametric} × T ∈ {50, 100, 200, 500, 1000}, cap = T, M = 1,000
  CRN-paired episodes, test id `emp`.
- **Policies, all frozen from the corpus — no constant is tuned on real data:**
  - `fixed_K` over `DEFAULT_K_GRID` = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 400), K ≤ T
    (traces the U-curve and each pool's K\*);
  - `fixed_K_star`: the corpus-selected K(T) at cap = T (Pre-registration 7);
  - `p3_star`: the corpus-tuned schedule with cap-T constants (Pre-registration 7). Horizons for which
    Pre-registration 7 selected no cap-T constants get them by the same procedure **on the corpus only**,
    committed before G2;
  - `level_star` (α = 0.75, c = 1.0, b = 4; Pre-registration 6);
  - `phi_k4` at its selected τ;
  - `fixed_K` 64 (the inherited cap), for pricing the cap.

## 6. Pre-registration 9 (committed before the pilot)

Primary panel: the two **primary** reservoirs × T ∈ {50, 100, 200} = 6 cells. MEI = 0.002, as in
Pre-registration 8.

**Interval of record — prompt bootstrap.** The dominant uncertainty is which 50 prompts were sampled. B = 200
replicates: resample the 50 prompts of each pool with replacement (with all their task outcomes and any
replicates), re-estimate v̂ and the NPMLE, rebuild the reservoir, rerun the cells with fresh CRN seeds;
95% percentile interval of the pooled Δ. The episode-paired CI is reported beside it.

**Flatness guard.** A primary cell is *informative* iff max − min of `fixed_K` regret over the K-grid
(K ≤ T) exceeds 5 × MEI = 0.01, judged on the full-sample point estimates (not per bootstrap replicate). If fewer than 2 of the 6 primary cells are informative, every contrast's
verdict is **"uninformative: K barely matters on real Gmail prompts"**, and that is the headline.
Otherwise contrasts are evaluated on the informative cells only, and the count is reported.

**Contrasts** (pooled over informative primary cells):

1. **Primary — the schedule transfers to real prompts (non-inferiority).** Δ = `p3_star` − `fixed_K_star`.
   *Supported* iff the bootstrap interval's upper bound < +MEI; *refuted* iff its lower bound ≥ +MEI;
   otherwise *inconclusive*.
2. **Secondary — the signals do not help on real prompts (not-better).** Δ = `level_star` − `p3_star` and
   Δ = `phi_k4` − `p3_star`. *Supported* iff lower bound > −MEI; *refuted* iff upper bound ≤ −MEI; otherwise
   *inconclusive*. Reported, uncorrected.
3. **Secondary — the level rule's cross-pool prediction.** Let ℓ_G, ℓ_F be the pools' means. Predicted:
   the lower-level pool has the larger K\*(T) at each primary T. Reported: the sign agreement (k of 3
   horizons) and the ratio K\*_low / K\*_high against the rule's exp(4 · (ℓ_high − ℓ_low)).
4. **Descriptive.** Each pool's level, SD, q99 − mean, and K\*(T) against the 33 corpus environments
   (`k_star_envelope_all33.csv`); the U-curves; each rule's gap to the pool's own K\* (in-sample argmin, never
   a deployable policy); the cap-64 cost at T ∈ {500, 1000}, labelled *extrapolation beyond 50 observed
   prompts*.
5. **Sensitivity.** Contrasts 1–2 on the raw and parametric reservoirs; reported without verdicts.

Registered in `registered_contrast.py` as `emp_primary`, `emp_level`, `emp_phi`, with a `bootstrap: prompt`
interval option.

## 7. Budget

| Item | Episodes | Cost at $0.036 |
|---|---|---|
| 2 pools × 50 prompts × 60 tasks | 6,000 | $216 |
| Anchor (`baseline` × 60) | 60 | $2 |
| Noise replicates | 300 | $11 |
| Claude generation of pool F | — | < $1 |
| **Total** | **6,360** | **≈ $230** (hard stop $260); ≈ 10.5 h at 8 workers |

The pilot (§8, G2) runs the 660 queue entries belonging to 5 prompts per pool (chosen by the pool seed) plus
the anchor, in their queue order, before anything else; if nothing changes after it, those episodes count
toward the total.

## 8. Gates (any failure stops the run and reports)

- **G0 — registration.** Pre-registration 9 committed; `pool_G.yaml`, `pool_F.yaml` and the replicate list
  frozen and committed with hashes; any missing cap-T constants selected on the corpus and committed.
- **G1 — rehearsal (free).** The full §4–§6 pipeline on synthetic collection data: prompts drawn from known
  reservoirs (a flat SD-0.03 pool at level 0.6, and a wider corpus-like pool), per-task outcomes generated
  with the logged task-difficulty spread and within-cell variance 0.093, 50 prompts × 60 tasks, 300
  replicates. Pass iff the NPMLE's SD is within ±0.01 of truth on both, the raw SD is visibly inflated, and
  the estimated K\*(T=200) is within one grid step of the true K\* on the wide pool. Fail → redesign before
  any spend.
- **G2 — pilot ($24).** Pass iff cost/episode ≤ $0.05, `missing` ≤ 5%, all 8 workers stable, anchor rate
  within its historical binomial 95% interval (else flag drift and ask before continuing).
- **G3 — collection.** ≤ 5% of pairs `missing`.
- **G4 — analysis.** Contrasts only through `registered_contrast.py`; every regenerated table diffed
  cell-by-cell against HEAD before commit.

## 9. Components and tests (pytest, written first)

| Unit | Location | Tests |
|---|---|---|
| `EmpiricalReservoir` | `src/cold_start/growing/reservoirs.py` | icdf/survival agree; `sample_from_uniforms` deterministic under CRN; mean = Σ w·atom; builds from a pool file |
| Noise model + NPMLE | `src/cold_start/growing/empirical.py` | recovers a known discrete G; heteroscedastic σᵢ handled; raw SD > NPMLE SD on noisy data |
| Parametric fit | `src/cold_start/growing/empirical.py` | recovers Beta parameters on synthetic data within tolerance |
| Pool builders | `experiments/growing_bandits/empirical/make_pools.py` | G sample deterministic by seed; F file round-trips; hashes stable |
| Collector | `experiments/growing_bandits/empirical/collect.py` | shuffle deterministic; resume skips done triples; infra error → retry → `missing`, never 0; budget stop fires (all on the `mock` model) |
| Replay cells | `experiments/growing_bandits/deploy/cells.py` (`--test emp`) | cell specs point at frozen pools; CRN pairing assert passes |
| Prompt bootstrap | `experiments/growing_bandits/deploy/registered_contrast.py` | interval reproducible by seed; flatness guard logic |
| End-to-end | — | mock model, 2 prompts × 3 tasks → pool → replay → contrast table |

## 10. Out of scope

- Live (non-replay) bandit runs on WebArena.
- Other apps (Shopping, GitLab) — the same pipeline applies later.
- Re-tuning any constant on real data.
- Idea C (engineered wide-K\* family), still parked.

## 11. Before this branch starts

The 6 commits `5596821..a6f780e` on `origin/fix/post-merge-corrections` get their follow-up PR first, so
this branch's PR diff contains only this study.
