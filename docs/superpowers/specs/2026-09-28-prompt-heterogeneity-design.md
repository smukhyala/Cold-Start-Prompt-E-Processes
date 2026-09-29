# Where does prompt heterogeneity come from? — design (2026-09-28)

*Branch `prompt-heterogeneity`, cut from `empirical-reservoir` at `64ecd00` (it reuses that study's
collector, estimation, replay and contrast code). Registers as **Pre-registration 10** in
`docs/growing_bandits/DEPLOYMENT_PLAN.md`; results become §14 of `DEPLOYMENT_RESULTS.md`.*

## 1. Why

Pre-registration 9 (§13) found both real Gmail prompt pools flatter than every simulated environment
(deconvolved spread: grid 0.035, free-form 0.001), so K barely mattered and every registered contrast was
*uninformative*. The paper's emerging claim is that **prompt heterogeneity is a prerequisite for adaptive
prompt search to be useful**. One flat environment cannot test that claim: it needs cells that differ in
heterogeneity, a measurement shown to detect heterogeneity when it exists, and a demonstration that the value
of search rises with it.

Re-reading the project's own data sharpens the design (noise-corrected true spread of arm success rates):

| data | arms × tasks | true spread |
|---|---|---|
| Gmail, 12 hand-written arms | 12 × 60 | 0.028 |
| Gmail pool G (Pre-reg 9) | 50 × 30 | 0.035 |
| Shopping, 12 hand-written arms | 12 × 80 | 0.048 |
| GitLab `gitlab_strong_arm/paired`, 18 arms | 18 × 40 | 0.116 |
| the same GitLab run without `gitlab_oracle_operator` (0.875) and `explorer` (0.20) | 16 × 40 | **0.020** |

GitLab's apparent heterogeneity is two outliers: an arm whose prompt carries a benchmark-specific route
cookbook, and one whose prompt induces a harmful behavior. The stylistic bulk is as flat as Gmail. Hence
two hypotheses the Gmail study could not separate — its pools were style-only by construction (the F
generator was told not to mention any specific email, person, label or task) — and the design below:
**environment × prompt content**, with known-effect anchors as instrument checks.

**Success** = every cell classified flat / moderate / meaningful by pre-registered rules; H1–H4 each
supported, refuted or inconclusive; and a value-of-search-versus-heterogeneity figure with every empirical
pool placed on a simulated calibration curve.

## 2. Research question and hypotheses

**Question.** Is adaptive prompt search useful only when the candidate pool is heterogeneous, and does that
heterogeneity come from the environment, from what the prompts contain, or both?

- **H1 (environment).** The same 50 grid prompts are more heterogeneous in GitLab than in Gmail:
  τ(GitLab, G) − τ(Gmail, G) > 0.
- **H2 (content).** Within each environment, prompts carrying app-specific procedural knowledge are more
  heterogeneous than stylistic prompts: τ(env, K) − τ(env, G) > 0, for env ∈ {Gmail, GitLab}.
- **H3 (the thesis).** The value of adaptive search rises with heterogeneity: in flat cells all policies tie;
  in meaningful cells the simulation's ordering reappears.
- **H4 (portability, secondary).** A grid prompt's effect in Gmail predicts its effect in GitLab.

τ is the standard deviation of prompts' true success rates (the prompt main effect, §6.1).

## 3. What stays identical to Pre-registration 9

- Agent: `gpt-5.4-mini`, `llm_reasoning_effort: low`, `max_agent_steps: 30`, `use_vision: false`,
  headless, the same browser-use version and adapter.
- The 50 G prompts, **byte-identical** (`data/empirical_pool/pool_G.yaml`, same sha256), so H1 is paired
  by prompt.
- The collector, its infra/timeout classification, `provider_down` breaker, watchdog, gates G2/G3
  (thresholds below), budget stop, and CRN-seeded replay; the estimation pipeline (replicate-pair noise
  model, NPMLE, raw and parametric sensitivity), K-grid, MEI 0.002, flatness threshold 0.01, B = 200.
- Pre-registration committed before any paid episode; the pilot is judged on cost and infrastructure only.

**Deliberate differences, each measured:** the wall-clock timeout (§4.4) and the task count in GitLab (60
rather than 30; §4.3).

## 4. Design

### 4.1 Cells

| cell | pool | prompts | tasks | episodes |
|---|---|---|---|---|
| Gmail × G | existing (Pre-reg 9) | 50 | 30 (existing subset) | 0 new |
| **Gmail × K** | new, from the Gmail manual | 40 | the same 30 | 1,200 |
| **GitLab × G** | the same 50 G prompts | 50 | 60 | 3,000 |
| **GitLab × K** | new, from the GitLab manual | 40 | 60 | 2,400 |
| GitLab anchors | `baseline`, `explorer`, `gitlab_oracle_operator` | 3 | 60 | 180 |
| Gmail anchor | `baseline` (drift) | 1 | 30 | 30 |
| Timeout bridge (§4.4) | 20 G prompts in Gmail at 600 s | 20 | 30 | 600 |
| Noise replicates | second run of random (prompt, task) cells: 300 GitLab G, 240 GitLab K, 120 Gmail K | — | — | 660 |
| **total** | | | | **8,070** |

Optional tier (not part of the core registration): GitLab × F (40 free-form prompts regenerated with Pre-reg
9's generation prompt, app description swapped) — 2,400 episodes.

### 4.2 Pools

- **G** — the 50 existing grid prompts, rendered with `configs/template.jinja` (app-agnostic). In GitLab
  they are sent unchanged.
- **K (knowledge-bearing)** — 40 per app, written by `claude-opus-4-7` from a **frozen manual bundle** for
  that app, never from task text:
  - GitLab bundle: `apps/gitlab-plan-and-track/APP_DESCRIPTION.md` (2,790 words) plus the user-manual pages
    covering the features that app implements (issues, labels, milestones, boards, epics, iterations,
    search), capped at ~25,000 words.
  - Gmail bundle: the `apps/user-manuals/gmail/` pages for organize-and-manage, compose-and-send and
    settings-and-configuration, capped at ~25,000 words.
  - Bundles are frozen by sha256 before generation. One generation call per app asks for 50 system-prompt
    extensions that give **procedural guidance for this application** (where features live, how workflows
    run, shortcuts, how to confirm a change took effect), varied in scope, emphasis, structure and length
    (40–250 words); the first 40 valid, distinct ones are kept. The generator is not asked to write bad
    guidance.
  - Leak guard: a K prompt is rejected if it shares any 8-token sequence with any task instruction of its
    app (all 60 Gmail / all 140 GitLab tasks), or names an entity (person, email subject, issue title, label
    name) that appears in a task instruction.
  - Rendered with `configs/template_freeform.jinja`.
- **Anchors** (outside every pool; they validate the instrument, they are not hypotheses):
  `baseline` (drift, both apps), `explorer` (known bad in GitLab) and `gitlab_oracle_operator` (known good
  in GitLab; its prompt contains a benchmark-specific route cookbook and is labelled synthetic).
- Arm ids namespaced by app and pool: `GL_G_00`, `GL_K_00`, `GM_K_00`, `GL_anchor_oracle`, ….

### 4.3 Tasks

- **GitLab:** 60 of the 140 `real-tasks`, stratified 20 easy / 20 medium / 20 hard (the bank is 20/20/100),
  drawn by `make_pools.select_tasks` with seed 20260928. Ordered into two stratified blocks of 30 (10/10/10
  each); the queue runs block A for every arm before block B, so a budget stop leaves a coherent 30-task
  design.
- **Gmail:** the existing 30-task Amendment-1 subset, so Gmail × K is directly comparable to Gmail × G.

### 4.4 Timeouts

Gmail timed out 12.7% of episodes at a 180 s wall clock under 8-way parallel load, partly measuring API
latency rather than agent behavior. GitLab runs **600 s wall clock with the 30-step limit binding**. Gmail ×
K keeps 180 s so that within-Gmail comparisons (H2) match Gmail × G. The **timeout bridge** re-runs 20
randomly chosen G prompts on the 30 Gmail tasks at 600 s; it measures how much the timeout changes Gmail's
τ and is used to put H1 (Gmail 180 s vs GitLab 600 s) on a common footing (§6.3). Every episode records
whether the step limit or the clock ended it.

### 4.5 Collection, pilot, budget

- Collector as in Pre-reg 9 with an `app` dimension (port pools per app; one WebArena server per worker per
  app). Queue order: pilot, then GitLab block A, Gmail cells, GitLab block B, bridge, replicates last
  within each app.
- **Pilot** (gate G2): GitLab only — 5 G + 5 K + the 3 anchors on block A (13 × 30 = 390 episodes).
  Thresholds: missing ≤ 5% counting never-attempted items; 0 watchdog relaunches; all 8 workers productive;
  cost ≤ $0.25 per episode; anchor ordering oracle > explorer on the pilot tasks (instrument sanity, not a
  hypothesis). Only anchor and aggregate cost/infra metrics are computed during the pilot.
- **Budget:** cost per GitLab episode under this agent is unknown (the old GitLab run cost $0.03–0.05 per
  episode, possibly with another model; Gmail cost $0.10). At $0.10 the core is ≈ $810. Hard stop set per
  stage by Sanjay; the run stops with STATUS `budget` and resumes after a top-up.
- **G3:** missing ≤ 5% per cell counting never-attempted items.

## 5. Stage 0 — free analyses before any spend

1. **Gmail reanalysis** (existing Pre-reg 9 data): variance decomposition (§6.1) and split-half reliability
   (§6.2) for G and F. Reported in §14 regardless of the new data.
2. **Old GitLab reanalysis** (`gitlab_strong_arm/paired/paired_results.csv`, 18 arms × 40 tasks, task-level):
   the same decomposition with and without the two outliers, and the upper-tail statistic (§6.4).
3. **Calibration simulation**: the simulator on synthetic pools at level 0.6 — Beta pools with true spread
   0.01–0.15, and "flat bulk + rare great arm" mixtures (bulk spread 0.02; 1%, 2%, 5% of arms +0.10,
   +0.20, +0.30) — at T ∈ {50, 100, 200}, cap = T, M = 1,000: regret range over the K-grid, and the gaps of
   `p3_star`, `fixed_K8` and `always_search` to the ceiling. This is the calibration curve for figure 3.
   It is run and committed **before** the pre-registration so it cannot move the thresholds.

## 6. Analysis

### 6.1 Variance decomposition (primary)

Per cell, on success (0/1):

y_ijr = μ + a_i + b_j + (ab)_ij + e_ijr

- Var(a_i) = τ² — the prompt main effect (what the bandit optimizes over the task mix);
- Var(b_j) — task difficulty;
- Var(ab)_ij — prompt × task interaction;
- Var(e) — execution noise, identified by the replicate cells.

Estimator: the method-of-moments (ANOVA / expected-mean-squares) estimator for the balanced crossed design,
with Var(e) from the replicate pairs; negative variance estimates are truncated at 0 and reported as such.
Sensitivity: REML linear mixed model and a logistic GLMM with crossed prompt and task effects. **Interval for
τ:** the Graybill–Wang modified-large-sample (MLS) interval for σ²_A = (MS_prompt − MS_resid)/J (two-sided
95%, and the one-sided 95% upper bound used by §6.5), validated by a simulation coverage test. *Amended before
registration:* the two-way (prompt × task) bootstrap first specified here inflates τ² by about MS_resid/J —
duplicated task columns enter every prompt mean — covering τ = 0 only ~20% of the time, so it is not used.
Missing cells are filled additively with the residual degrees of freedom and the prompt-mean residual term
corrected for the fill; more than 5% missing cells in a cell is an error. τ is reported in success-rate units.

### 6.2 Split-half reliability (model-free)

Split the cell's tasks into two halves stratified by difficulty (fixed seed); compute each prompt's success
rate on each half; Pearson r across prompts; Spearman–Brown corrected r_SB = 2r/(1+r). Test r > 0 by
permuting prompt labels within one half (10,000 permutations). A significantly positive r_SB means prompt
differences generalize across tasks; it needs no noise model and no NPMLE.

### 6.3 Contrasts for H1, H2, H4

Intervals for the contrasts come from a **prompt-only bootstrap** (B = 2,000; percentile) with each cell's
task set held fixed — τ is defined over the study's fixed task mix. H1 resamples the 20 bridge prompts
jointly in both apps; H2 resamples each pool's prompts independently.

- **H1:** Δ₁ = τ(GitLab, G) − τ(Gmail, G). Primary: GitLab at 600 s vs the bridge (Gmail at 600 s, the
  same 20 prompts are a subset of the 50) — computed on the 20 bridge prompts in both apps, paired by prompt,
  with a prompt-and-task bootstrap. Secondary: all 50 G prompts, Gmail at 180 s. Supported iff the primary
  one-sided 95% lower bound > 0; refuted iff the upper bound < 0.01; otherwise inconclusive.
- **H2:** Δ₂(env) = τ(env, K) − τ(env, G), for each env; supported per env by the same rule.
- **H4:** Spearman ρ between the 50 G prompts' estimated main effects in Gmail and GitLab (BLUPs from 6.1),
  with a bootstrap CI; reported, not decided.

### 6.4 Pool shape and decision quantity

- NPMLE per cell (as in Pre-reg 9) → reservoir for replay; **upper-tail mass**: estimated share of prompts
  with true rate ≥ median + 0.10, with a prompt-bootstrap CI.
- Replay every cell (cap = T; T ∈ {50, 100, 200} primary, {500, 1000} secondary) over the K-grid:
  **regret range** over K, with a prompt-bootstrap CI (B = 200, NPMLE re-estimated per resample).

### 6.5 Classification (pre-registered)

Per cell:

| class | rule |
|---|---|
| **flat** | regret range < 0.005 at all three primary T **and** the one-sided upper 95% MLS bound of τ < τ_flat |
| **practically meaningful** | regret range ≥ 0.01 at ≥ 2 of 3 primary T (Pre-reg 9's guard) **and** the bootstrap lower bound of the regret range at T = 200 > 0.005 |
| **moderate** | otherwise |

τ_flat is fixed in Pre-registration 10, before any data, from the Stage-0 calibration: the true spread of a
level-0.6 Beta pool whose regret range at T = 200 equals 0.005. *(Amended before registration: a fixed 0.03 is
unreachable at J = 30 even when τ = 0 — the mean upper bound there is about 0.04 — which would make "flat"
structurally impossible for Gmail cells.)*

### 6.6 H3 — the thesis

Replay contrasts (paired, prompt-bootstrap intervals, MEI = 0.002) in every **meaningful** cell:

1. `p3_star` (budget-scaled schedule) − `fixed_K8` (small fixed K) < −MEI — scaling K with the budget pays;
2. `always_search` − `p3_star` > +MEI — spreading every pull over new prompts loses;
3. Pre-reg 9's `emp_primary` / `emp_level` / `emp_phi` contrasts, for continuity.

**H3 is supported** iff (a) every flat cell has regret range < 0.005 at every primary T, and (b) at least one
meaningful cell exists in which contrasts 1 and 2 both hold. **Refuted** iff a flat cell shows a policy
difference > MEI with a bootstrap interval excluding 0, or a meaningful cell shows contrast 1 or 2 reversed
beyond MEI. **Untestable** (reported as such, not as support) iff no cell is meaningful.

### 6.7 Discriminating-task sensitivity

All primary analyses use all tasks (the bandit's objective is the task mix). Secondary: repeat 6.1–6.2 on
tasks whose across-prompt success rate lies in [0.2, 0.8], where prompt effects are not compressed by floor
or ceiling (Gmail had 13 of 60 tasks at 1.0 and 6 at 0.0).

### 6.8 Timeout sensitivity

Repeat 6.1 treating clock-ended episodes as missing rather than 0; report per-prompt timeout rates (a prompt
that makes the agent slow is a real effect on cost, reported separately from success).

## 7. Figures for the paper

1. Variance decomposition per cell (task / prompt / prompt × task / noise).
2. τ with 95% CIs per cell over the flat / moderate / meaningful bands.
3. **Value of search vs heterogeneity**: the Stage-0 calibration curve with every empirical pool as a point
   (Gmail G, F, K; GitLab G, K; historical Shopping and GitLab hand-written arms, labelled historical).
4. Regret-vs-K curves for each empirical pool at T ∈ {50, 100, 200}.
5. Policy gaps to the ceiling in meaningful cells.
6. Portability scatter: G prompt effects, Gmail vs GitLab.
7. Supplement: NPMLE densities, split-half scatter, per-prompt timeout rates, anchor recovery.

## 8. What each outcome licenses

| outcome | claim |
|---|---|
| GitLab × G meaningful | heterogeneity is environment-dependent; the value of search tracks it (H1, H3) |
| G flat in both apps, K meaningful in ≥ 1 | style is a weak lever; task-relevant knowledge content creates heterogeneity; adaptive search pays over knowledge-bearing candidates (H2, H3) |
| all cells flat, anchors recovered | for this agent, hand-written and manual-derived prompt variation is a small lever on these benchmarks; H3 untestable here; the follow-up (§9) is required |
| anchors not recovered (oracle not above the bulk, explorer not below) | the instrument is too weak; no heterogeneity claim is made |

## 9. Follow-up if every cell is flat (not part of this registration)

Arms produced by a **real prompt optimizer**: a GEPA-style reflective optimizer seeded from G mutates prompts
from logged failures; the growing arm set *is* the optimizer's candidate stream, so "how many candidates to
evaluate" is the question the optimizer faces. Fallbacks: arms as agent configurations (model × reasoning
effort × prompt family); a weaker agent model.

## 10. Gmail confounds this design fixes

| confound in Pre-reg 9 | fix here |
|---|---|
| 30 tasks (per-prompt noise ±0.056) | 60 GitLab tasks (±0.034); Gmail × K stays at 30 for comparability |
| timeouts 12.7% at 180 s wall clock | 600 s in GitLab with the step limit binding; bridge measures the effect in Gmail; 6.8 |
| dependence on NPMLE | primary τ from the moment estimator with a two-way bootstrap; split-half reliability; NPMLE only shapes the replay reservoir |
| per-pool noise difference (F 0.087 vs G 0.052) | ~120–300 replicate pairs per new cell; noise fit per cell |
| style-only pools | K pools carry procedural knowledge |
| prompt × task interaction never estimated | estimated in 6.1 |
| floor / ceiling tasks | 6.7 |
| pool-level success rates seen during the pilot | pilot judged on cost/infra and anchors only |

## 11. Components (for the implementation plan)

| unit | location | change |
|---|---|---|
| multi-app pools and queue | `experiments/growing_bandits/empirical/make_pools.py` | app dimension; manual-bundle freezing; K generation + leak guard; blocked queue |
| collector | `experiments/growing_bandits/empirical/collect.py` | per-arm app and timeout; workers serve both apps |
| variance decomposition, split-half, bootstrap | `src/cold_start/growing/heterogeneity.py` (new) | 6.1, 6.2, 6.7, 6.8 |
| calibration simulation | `experiments/growing_bandits/empirical/calibrate.py` (new) | Stage 0 item 3 |
| replay and describe | `replay.py`, `describe.py` | pools beyond ("G", "F"); cells keyed by app × pool |
| contrasts | `registered_contrast.py` | H1, H2, H3 registrations |
| figures | `experiments/growing_bandits/empirical/figures.py` (new) | §7 |

## 12. Out of scope

GitLab × F (optional tier); other apps; other models; the §9 follow-up; live bandit runs.
