# Growing Bandits — Deploying the Learned SEARCH-vs-REFINE Policy

*Deployment study of the plan frozen in `DEPLOYMENT_PLAN.md`. Branch `deploy-policy` (base `b45f07d`,
HEAD `6b68b5c`). Runs of 2026-09-14, with the Test-D baselines re-tuned and every test re-analysed on
2026-09-19.*

**How to read the numbers.** Every figure below carries its source table in parentheses, with the row keys
needed to find it again, e.g. `(primary_contrasts_A_primary.csv, H1b, level=pooled)`. All tables live in
`results/growing_bandits/deploy/tables/`, all figures in `results/growing_bandits/deploy/figures/`. Unless a
sentence says otherwise, every number is on the **primary recommendation rule, `posterior_mean_shrunk`**
(`src/cold_start/growing/deploy/recommenders.py:32`), whose tables carry the `_primary` suffix.
`_lcb`, `_oracle_prior`, `_posterior_mean` and `_empirical` are **secondary** rules and are always named as
such. Three conventions matter throughout:

- **Paired CI** = 95% percentile bootstrap over episodes, which are CRN-paired across policies within a cell.
  **Cluster CI** = 95% percentile bootstrap over *environments*. Where a row carries
  `cluster_degenerate=True` (n_envs ≤ 3) the "interval" is arithmetically the minimum and maximum
  per-environment mean, and the only statement it supports is *every environment agrees in sign*. Those rows
  are phrased that way here, never as "95% CI excludes zero".
- **Regret** is `R_T = μ*_ep − Q_T`: the best of the episode's first `T` reservoir draws (attainable, CRN-paired)
  minus the true mean of the arm the policy recommends at the end. Lower is better; a negative Δ favours the
  learned policy. It decomposes as `R_T = R_disc + R_sel` (discovery + selection).
- **Multiplicity**: only H1a, H1b and H2 are pre-registered. Everything else is exploratory and uncorrected.
  Section 3.5 states what survives correction, and it is not much.
- **The cluster interval is the Student-t on the environment means** (`cluster_lo` / `cluster_hi`,
  `cluster_method = env_mean_t`). The percentile cluster bootstrap this document first shipped under-covers
  at the environment counts the study has — 0.719 / 0.796 / 0.863 / 0.893 at n_envs = 3 / 4 / 6 / 8 against
  nominal 0.95, measured on the 30-environment population; the t interval covers 0.918 / 0.922 / 0.935 /
  0.943 — so it was replaced (NEXT-STEPS 2.1). Every "cluster" interval below is the t interval; its
  two-sided p is `cluster_p`, the exact sign-flip p is `cluster_p_sign`, and the old percentile interval is
  still on every row as `cluster_pct_lo` / `cluster_pct_hi` for comparison. Where the swap changed a
  conclusion the text says so in place. Every point estimate, paired CI and flag is bit-identical to the
  previous version of this document.
- **Two bootstrap runs of record.** `primary_contrasts_<test>_primary.csv` and `main_<test>_primary.csv`
  re-bootstrap the same paired contrasts independently at n_boot = 10000, so their point estimates and their
  (deterministic) t cluster bounds are bit-identical while their *paired* CI bounds differ by Monte-Carlo
  noise of about 1% of interval width. Both are quoted below with their source file. No CI in this document
  comes from `summary_<test>.csv`, which is a third, coarser bootstrap (n_boot = 2000).

---

## 1. Bottom line

Deploying the learned k=16 policy Φ lowers simple regret against the continuation policy its labels were
defined against, and does **not** lower it against a validation-tuned growth schedule.

- **H1a (Φ vs `cp0`)**: pooled Δ = **−0.038165**, paired [−0.039006, −0.037310], environment-cluster
  [−0.070636, −0.005693], p = 0.027, and negative in every one of the eight environments (exact sign
  p = 0.0078); cluster-significant pooled and at every horizon, i.e. in the 6 of 18 Test-A strata that have
  eight environments — the 12 `family` / `family_horizon` strata have four, where a t interval on three
  degrees of freedom includes zero although all four environments agree in sign
  (`primary_contrasts_A_primary.csv`, H1a). The necessary condition holds, conditional on the 64-arm cap the
  labels were harvested under: in the cap sweep pooled H1a is −0.013909 with a cluster interval
  [−0.047846, +0.020029] that includes zero (n_envs = 4), and at T = 1000 it reverses in sign to
  **+0.009645**, cluster [−0.005667, +0.024956], also including zero (`primary_contrasts_cap_primary.csv`,
  H1a, level=pooled and level=horizon, horizon=1000).
- **H1b (Φ vs the tuned power schedule P3\*)**: pooled Δ = **−0.001389**, paired [−0.001885, −0.000898],
  cluster **[−0.003899, +0.001121] — includes zero** (same file, H1b, level=pooled). In none of the six
  tests is Φ cluster-significantly better than P3\*, and on both off-distribution tests it is worse.
- **H2 (e-process features)**: pooled Δ(`phi_k16` − `phi_k16_quality`) = **+0.000114**, cluster
  [−0.000277, +0.000505] — a null on Test A, unmeasured on Tests B and D, and negative (the features help)
  only outside the training support.
- **H3 (offline surrogate)**: Spearman ρ(offline OOF AUC, deployed pooled regret) = **+0.005082**,
  [−0.527026, +0.523659], n_variants = 22 (`surrogate_validity.csv`, test=A,
  recommender=posterior_mean_shrunk, offline=oof_auc, deployed=pooled).

The mechanism is the 64-arm cap: P3\* is already a cap-filling policy at T ≥ 200 (k_final = 64.0 at
T = 200/500/1000), and Φ joins it only at T = 1000 (k_final 44.2 at T=200, 57.7 at T=500, 64.0 at T=1000;
`main_A_primary.csv`, level=horizon). So H1b at T = 200 and T = 500 is "Φ against fill-the-cap", and at
T = 1000 it is not a test of a learned rule at all.

---

## 2. What was run

**Harness.** One `Simulator` per (env, horizon, cap) cell, warm-started with 2 arms, run to the horizon.
Every policy in a cell shares the cell's `base_seed`, so arm means and reward prefixes are bit-identical
across policies and all differences are paired (failure-mode register #3; `harness.py:468`, per-policy RNG
isolated at `harness.py:224-239`, runtime guard at `:287`). The comparator is the per-episode
`μ*_ep = max` of the first T reservoir draws in the episode's own CRN stream, so discovery regret is
non-negative by construction. All five recommendation rules are scored from the same final state, which is
free because the recommendation does not affect the trajectory; every table is emitted per rule.

**Splits and seeds.** Schedule constants `c` and `K0` are tuned on the tune block
(`base_seed = 1_000_000 + cell_id·1_000`), model thresholds τ and the schedule selection on the validation
block (`5_000_000 + …`), and every number in this document is measured on the test block
(`10_000_000 + …`); the on-policy harvest of section 8 uses a fourth block (`15_000_000 + …`). A startup
assertion checks these are disjoint from the corpus seed set (12,960 seeds in [20,262,460, 343,348,312]) and
from the old benchmark's 11 seeds. `select_thresholds.py` refuses `--split test` without an explicit flag;
no file or log carries the `_TESTSPLIT` suffix that flag would produce.

**Panels.** Counted from the shipped cell tables and manifests:

| test | cells | envs | horizons | caps | policies | episodes/cell | rows |
|---|---|---|---|---|---|---|---|
| A (main) | 40 | 8 | 50/100/200/500/1000 | 64 | 36 | 2000 | 1440 |
| B (held-out generating regime) | 40 | 8 | as A | 64 | 5 | 2000 | 200 |
| C (held-out mixture family) | 15 | 3 | as A | 64 | 34 | 2000 | 510 |
| D (horizon transfer) | 19 | 8 | 200/1000/2000 | 64 | 11 | 2000 (1000 at T=2000) | 200 |
| cap sweep | 32 | 4 | 200/1000 | 32/64/128/T | 6 | 1000 | 192 |
| robustness | 150 | 30 | as A | 64 | 6 | 500 | 900 |

(`cells_<test>_primary.csv`; `manifest_<test>.jsonl` has one line per (cell, policy), **3,448 in total**.
The `rows` column above sums to 3,442 because `manifest_D.jsonl` has 206 lines for 200 pairs: the six
re-tuned T=2000 baseline items were appended when they were re-run.)

**Policy table and how each arm was tuned.** The schedule grid is Ruling 10's 48-point geometric grid
`c ∈ [0.25, 64]` at four exponents, tuned on the tune split at M=500 and selected on validation
(`baseline_params.json:meta`). The selected P3\* is per horizon: T=50 (α=1/4, c=7.654), T=100 (α=1/3,
c=7.654), T=200 (α=2/3, c=3.771), T=500 (α=1/3, c=15.535), T=1000 (α=1/4, c=15.535), T=2000 (α=2/3,
c=0.915) — all interior to the grid, so no schedule is boundary-limited. `refine_after_init` takes K0 = 8 at
every horizon, the grid maximum of {2,4,8}. Learned policies take a single global τ per model, chosen as the
argmin of mean validation regret over τ ∈ {0.3,…,0.7} (`tau_curves.csv`, source=validation): τ = 0.5 for the
pre-registered `phi_k16` (validation regret 0.109076 at τ=0.5 against 0.109793 at 0.4 and 0.110587 at 0.6),
0.4 for `phi_k4`, 0.4 for `phi_k1`, 0.5 for `phi_k16_quality`, 0.4 for `phi_k16_clock`. This asymmetry — one
global scalar for Φ against a per-horizon (α, c) for the schedule — is pre-registered
(`DEPLOYMENT_PLAN.md:237`) and runs against the learned policy. The three horizon-holdout models select τ on
the 32 non-held-out cells only (Ruling 9): `phi_k16_noT1000` τ=0.4, `phi_k16_noT200` τ=0.5,
`phi_sf_k16_noT1000` τ=0.5.

**Reproduction gate.** Before any protocol change, the new trainer had to reproduce the offline study's
headline: `reproduction_gate.json` records `bal_acc_05_fold_mean = 0.6472` against a target of 0.647 and
`auc_fold_mean = 0.7405` against 0.7405, tolerance 0.01, `ok: true`.

**Feature hygiene.** Feature groups are explicit column lists, not substring selectors
(`deploy/feature_groups.py`): the deployed `clock_quality_evidence` set is 62 columns, `clock_quality` 35,
`clock` 11, `all71` 71, the scale-free clock variant 56, the reservoir hand rule 7
(`offline_vs_deployed.csv`, `n_features`). No `oracle_*`, `label_*` or `meta_*` column can enter a policy;
a grep of `src/cold_start/growing/deploy/*.py` for `state.mu|reservoir|oracle_|meta_` hits only docstrings,
the harness-side comparator, the harness's `_hidden_truth_*` helpers and the forbidden-prefix list itself —
`features_vec.py`, `model_policy.py` and `rules.py` contain zero live reads.

**Feature parity.** The vectorized features the policy actually consumes were checked against the scalar
`extract_features` on 737,280 logged on-policy states: `onpolicy_parity.csv` has 71 rows, `passed=True` on
all 71, Σ`n_fail` = 0, worst column `f_log_e_pair` at max |Δ| = 1.757e−06 (inside rtol 1e−5). The analysis
refuses to write main tables if any column fails.

**Tests and lint at the analysed commit.** `make test` = **970 passed, exit 0** at `da41273`
(`logs/deploy/m8_make_test.log`), 980 passed after the fix wave added ten tests; the 5 GB
`data/pairwise_tables` cache was fingerprinted before and after and is byte-unchanged. `make lint` is **red
repo-wide, with 68 ruff errors — every one of them in a file this branch never touched**; the intersection of
the failing-file set with `git diff --name-only b45f07d HEAD` is empty and there are zero errors in
`src/cold_start/growing/deploy/` or `tests/test_deploy_*.py` (Ruling 18). That is the scope of the "lint
clean" verification item: this branch's own surface, not the repository.

---

## 3. Pre-registered results (Test A)

Test A is the in-distribution panel: 8 corpus environments × 5 horizons × 2000 episodes at cap 64, 36
policies, all CRN-paired. The pre-registered table is exactly 54 rows — 3 hypotheses × 18 strata — every one
`pre_registered=True`, `status=ok`, `n_boot=10000` (`primary_contrasts_A_primary.csv`).

### 3.1 H1a — Φ beats `cp0`, pooled and at every horizon

| stratum | Δ | paired CI | cluster CI |
|---|---|---|---|
| pooled | **−0.038165** | [−0.039006, −0.037310] | [−0.070636, −0.005693] |
| family A | −0.043414 | [−0.044695, −0.042104] | [−0.119764, +0.032936] |
| family B | −0.032916 | [−0.033984, −0.031866] | [−0.086742, +0.020910] |
| T=50 | −0.040014 | [−0.042487, −0.037583] | [−0.075587, −0.004440] |
| T=100 | −0.040590 | [−0.042839, −0.038340] | [−0.078060, −0.003121] |
| T=200 | −0.040170 | [−0.042038, −0.038337] | [−0.078463, −0.001877] |
| T=500 | −0.039152 | [−0.040630, −0.037706] | [−0.070866, −0.007438] |
| T=1000 | −0.030898 | [−0.031990, −0.029807] | [−0.053539, −0.008257] |

(`primary_contrasts_A_primary.csv`, H1a rows; paired win rate 0.604281 pooled.) The pooled row and the five
`horizon` rows — the six strata with n_envs = 8 — have cluster CIs excluding zero (p = 0.027 pooled; exact
sign p = 0.0078, all eight environments negative). The two `family` rows and the ten `family_horizon` rows
not shown have n_envs = 4, and there a t interval on three degrees of freedom includes zero for every one of
them (p between 0.09 and 0.21) even though all four environments agree in sign in each (sign p = 0.125,
the floor at n = 4). The previous version of this document reported all 18 as cluster-significant on the
percentile interval; that interval under-covers at n = 4 (§3.5). The effect is large relative to
everything else in this study: Φ's pooled regret is 0.110106 against `cp0`'s 0.148271
(`main_A_primary.csv`, level=pooled), and roughly 86% of `cp0`'s regret is discovery
(`regret_disc` 0.128180 of 0.148271) — `cp0` simply does not search enough. The necessary condition is met:
the labels are one improvement step from `cp0`, and deploying the model trained on them beats `cp0`.

### 3.2 H1b — Φ does not beat the tuned schedule

| stratum | Δ | paired CI | cluster CI |
|---|---|---|---|
| pooled | **−0.001389** | [−0.001885, −0.000898] | [−0.003899, **+0.001121**] |
| T=50 | −0.002841 | [−0.004472, −0.001214] | [−0.009548, +0.003867] |
| T=100 | −0.004067 | [−0.005740, −0.002451] | [−0.009466, +0.001333] |
| T=200 | −0.000387 | [−0.001179, +0.000404] | [−0.003375, +0.002600] |
| T=500 | **+0.000350** | [+0.000131, +0.000570] | [−0.000876, +0.001577] |
| T=1000 | −7.42e−07 | [−4.6e−06, +2.3e−06] | [−4.0e−06, +2.0e−06] |

(`primary_contrasts_A_primary.csv`, H1b rows.) The pooled cluster CI includes zero (p = 0.23; sign p = 0.23,
five of eight environments negative). **No stratum has a cluster CI excluding zero.** The two that did on the
percentile interval, both at T=100 — `level=horizon` (−0.004067, now cluster [−0.009466, +0.001333],
p = 0.12) and `family=A, horizon=100` (−0.006837, cluster [−0.019037, +0.005362], n_envs = 4) — do not on
the t interval. At T=500 the schedule is better and the paired CI excludes zero. At T=1000 the difference
is seven decimal places from zero, because both policies are doing the same thing (§4).
The honest statement of H1b is: **on the corpus environments Φ and a validation-tuned power schedule are
indistinguishable, with a short-horizon advantage to Φ at T=100 that the environment-level interval does not
confirm, and a small advantage to the schedule at T=500.**

### 3.3 H2 — the e-process features are a null here

| stratum | Δ(`phi_k16` − `phi_k16_quality`) | paired CI | cluster CI |
|---|---|---|---|
| pooled | **+0.000114** | [−0.000040, +0.000265] | [−0.000277, +0.000505] |
| T=100 | +0.000587 | [+0.000025, +0.001165] | [−0.000475, +0.001648] |
| family A, T=100 | +0.001354 | [+0.000431, +0.002349] | [−0.000937, +0.003645] |
| family A, T=500 | −0.000340 | [−0.000529, −0.000157] | [−0.001389, +0.000709] |
| T=1000 | −9.4e−07 | [−2.4e−06, +3.6e−07] | [−3.0e−06, 0.000000] |

(`primary_contrasts_A_primary.csv`, H2 rows.) A positive Δ means the model *with* the e-process columns is
slightly worse. Pooled, the interval straddles zero. No Test-A stratum has a cluster CI excluding zero. The
one that did on the percentile interval, the `family_horizon` row family A × T=100 at +0.001354, reads
cluster [−0.000937, +0.003645] on the t interval (n_envs = 4; §9.4 item 14) — the evidence features appear
to cost about a thousandth of a regret unit there, and the paired CI [+0.000431, +0.002349] says so, but
the environment-level interval does not. The same is true of H1b's family A × T=100 row (−0.006837,
cluster [−0.019037, +0.005362]) and its `horizon`-level T=100 row, the two H1b strata discussed in §3.2.
(The post-hoc review attributed that significance to the `horizon=100` row; the shipped
table's `horizon=100` cluster CI is [−0.000475, +0.001648] and includes zero. The table is what is quoted
here.) A cleaner version of the same ablation is the single-column contrast: `phi_k16` (with `f_log_e_pair`)
against `phi_k16_cs` (confidence-sequence bounds only) differ by **1.0e−06** in pooled regret
(0.110106 vs 0.110105, `main_A_primary.csv`, level=pooled). The e-process column changes nothing in
this regime. What H2 means across the other five tests is in §7.5 — it is not a uniform null, and it must
never be stated flatly in either direction.

### 3.4 The τ=0.5 twin, and the recommendation rule

The register asked for τ=0.5 to be reported as its own row. `phi_k16_tau05` is present in every Test-A table
and is **numerically identical to `phi_k16`** (pooled regret 0.110106, k_final 45.494225, search_frac
0.244064 in both; `main_A_primary.csv`, level=pooled), because the validation selection chose τ = 0.5 for
that model anyway. The τ→regret curve is flat near the optimum: 0.109793 / 0.109076 / 0.110587 at
τ = 0.4 / 0.5 / 0.6 on validation, i.e. a spread of ~0.0015 across the plausible range, with search_frac
moving 0.286 → 0.244 → 0.208 (`tau_curves.csv`, variant=clock_quality_evidence_k16, source=validation).

Does the conclusion change under another recommendation rule? No. Pooled pre-registered contrasts under all
five rules (`primary_contrasts_A_<rec>.csv`, level=pooled):

| rule | H1a | H1b | H2 |
|---|---|---|---|
| `posterior_mean_shrunk` (primary) | −0.038165 cl[−0.070636, −0.005693] | −0.001389 cl[−0.003899, +0.001121] | +0.000114 cl[−0.000277, +0.000505] |
| `lcb` (secondary) | −0.038218 cl[−0.070741, −0.005695] | −0.001427 cl[−0.003958, +0.001103] | +0.000108 cl[−0.000280, +0.000496] |
| `oracle_prior` (secondary, not deployable) | −0.038196 cl[−0.070717, −0.005675] | −0.001422 cl[−0.003909, +0.001065] | +0.000101 cl[−0.000272, +0.000473] |
| `posterior_mean` (secondary) | −0.037593 cl[−0.069648, −0.005538] | −0.000933 cl[−0.003357, +0.001491] | +0.000076 cl[−0.000279, +0.000431] |
| `empirical` (secondary, the naive control) | −0.034256 cl[−0.064432, −0.004081] | −0.001471 cl[−0.007493, +0.004551] | +0.000992 cl[−0.000508, +0.002492] |

H1a is cluster-significant under every rule; H1b and H2 are cluster-null under every rule. The feared
inflation from the oracle-prior recommender is empirically nil at cap 64: primary and `oracle_prior` pooled
H1a differ by 3e−5. The four non-naive rules also agree on the *ranking* of policies (pooled Kendall τ
0.948–0.995 for every pair among `lcb`, `oracle_prior`, `posterior_mean`, `posterior_mean_shrunk`), while the
naive `empirical` rule is much looser against all of them (0.752–0.782) — it is the study's pre-declared
degenerate control, reported but never averaged into anything (`recommender_kendall.csv`, level=pooled).

### 3.5 Multiplicity, stated explicitly

No multiplicity correction is applied anywhere in the shipped tables, and none is claimed. Computed after the
fact by Holm on the shipped `cluster_p` (the two-sided p of the environment-mean t interval). Every number
in this section changed when the t interval replaced the percentile bootstrap, in the direction of fewer
survivors, because the percentile interval was too narrow (header note; NEXT-STEPS 2.1):

- Within the 54-row Test-A pre-registered family, **no row survives**. The previous version's single
  survivor, H1a at `level=horizon`, `horizon=1000` (adjusted 0.0264 on the percentile interval), has
  p = 0.0145 on the t interval and adjusted p = 0.78; pooled H1a (p = 0.027) is adjusted to 1.0.
- Across the full 282-row pre-registered family (A 54 + B 54 + C 21 + D 36 + cap 63 + robust 54), 171 rows
  carry a cluster interval at all — Tests B and D's `not_evaluated` / `missing` rows (18 each), all 21 Test-C
  rows (n_envs = 3, §7.2), and the cap sweep's 12 `untuned_reference` and 42 n_envs = 2 rows do not. Of those
  171, **12 survive, every one an H1a row of the 30-environment robustness sweep** (pooled, the five
  `horizon` rows, and family B's six rows; adjusted p ≤ 0.0032, smallest 1e−4). Nothing from Test A, B, C,
  D or the cap sweep survives. (On the percentile interval this read 14, with one cap-sweep H2 row and one
  more robustness row; both are gone.)
- On the defensible one-row-per-hypothesis-per-test family (13 testable pooled rows — Test C's three have no
  cluster interval at n_envs = 3, §7.2), **one survives: the robustness sweep's H1a** (n_envs = 30,
  p = 3e−7). Test A's pooled H1a (p = 0.027), Test B's (the same row relabelled) and Test D's (p = 0.026)
  are each adjusted to 0.31; the study's only environment-level survivor of a family of any size is the one
  row with thirty environments behind it. **None points against the thesis**. (An earlier version of this
  section listed four survivors, and before that two that pointed against. Test C H1b (+0.00368, "adjusted 0.0476")
  rested on the n_envs = 3 [min, max] range read as a 95% interval, which the tables no longer emit as one;
  its *paired* CI [+0.002793, +0.004572] still excludes zero and §7.2 reports it as such — worse, but not
  cluster-tested. Cap H1b at +0.0193 was an artefact of comparing against a schedule constant tuned at a
  different cap; once those cells are refused it is +0.000387 with a CI including zero — a null, not evidence
  either way. See §4.)

The exploratory set is larger still: `secondary_contrasts_A_primary.csv` alone is 1,854 rows, every one
`pre_registered=False`. Wherever this document lists "policies whose cluster CI excludes zero", the expected
number of false positives at α = 0.05 over 25 learned policies is exactly 25 × 0.05 = **1.25** by linearity
of expectation — unchanged by the strong positive dependence CRN induces. About one of any such seven is an
expected chance finding before bootstrap noise is considered.

### 3.6 The exploratory ladder

With that discount in place: of the 25 learned variants in Test A, **two** have pooled cluster CIs excluding
zero against P3\* — `phi_k4` (−0.003445, cl[−0.006266, −0.000624], p = 0.023, negative in all eight
environments) and `phi_k1` (−0.003429, cl[−0.006280, −0.000578], p = 0.025, likewise eight of eight) — and
the **pre-registered `phi_k16` is not one of them** (−0.001389, cl[−0.003899, +0.001121])
(`main_A_primary.csv`, level=pooled). On the percentile interval this document previously counted seven,
adding `phi_k16_perstep` (−0.003386, now cl[−0.006988, +0.000217], p = 0.062), `phi_k16_onpolicy_union`
(−0.001936, cl[−0.004382, +0.000509], p = 0.10), `phi_k16_noT200` (−0.001895, cl[−0.004382, +0.000593],
p = 0.11), `phi_k16_notrunc` (−0.001886, cl[−0.003834, +0.000062], p = 0.056) and `phi_k16_nopolicy`
(−0.001283, cl[−0.002869, +0.000303], p = 0.097); each of those five loses it on the t interval. Two
qualifications are required and neither is optional. (i) The count is uncorrected, with E[false positives]
= 1.25 as above — so two observed against 1.25 expected is not evidence of anything. (ii) Under Holm across
those 25 rows **nothing** survives: the smallest adjusted p is 0.585 (`phi_k4`). The previous version's
"`phi_k4` and `phi_k1` survive at 0.0374" was computed from the under-covering interval. (The old count
was also unstable in its bootstrap seed — 6/7/8 significant policies in 32/141/127 of 300 re-draws; the t
interval has no seed to re-draw.)

The one robust pattern in the ladder is that **short commitment wins**. `phi_k4` (0.108050), `phi_k1`
(0.108066) and `phi_k16_perstep` (0.108110 — the k=16 model re-evaluated every step, which is the same
commitment axis) are a tie at the top, spread 5.9e−05, each separated from rank 4 by at least 1.45e−03,
and each beats base `phi_k16` on a matched (policy, reference) contrast: −0.002056 [−0.002437, −0.001671],
−0.002040 [−0.002448, −0.001631] and −0.001997 [−0.002395, −0.001606] respectively
(`secondary_contrasts_A_primary.csv`, reference=phi_k16, level=pooled). Their *internal* ordering is not
stable and is not reported: under the naive `empirical` rule `phi_k16_perstep` leads and `phi_k1` falls to
fifth. All 25 learned variants lie within 0.0077 pooled regret of one another (0.108050 to 0.115748).

*Figure: `regret_vs_T_A.png` — pooled regret against horizon per family, with tied policies merged into a
single label (the pre-fix version fanned exact ties into a spurious ranking; read the companion CSV, not the
label order).*

---

## 4. The cap is the policy

The single most important fact about this deployment is that the 64-arm cap, inherited from the corpus, is
doing most of the work the policy appears to be doing.

**At T = 1000 and cap 64, Φ is operationally `always_search`.** Twenty of the 25 learned variants — every
one except `phi_k16_famA_only` (k_final 63.857), `phi_k16_hgb` (63.993), `phi_k16_weighted` (63.881),
`phi_sf_k16_noT1000` (63.831) and `phi_k16_onpolicy_only` (47.050) — finish with `k_final = 64.0` and
`search_frac = 62/998 = 0.062124` in all eight environments, at regret 0.097277–0.097279: within 1.2e−6 of
`always_search` (0.097277) and `p3_star` (0.097278), and within 3e−5 of `uniform` (0.097250), which fills
the cap too. Four of the five exceptions fall marginally short of the cap in two to three environments
(per-cell `k_final` 62.90–63.99) and land within 2e−5 of the same regret (0.097273–0.097290); only
`phi_k16_onpolicy_only` is genuinely different (`main_A_primary.csv`, level=horizon, horizon=1000;
`cells_A_primary.csv`, horizon=1000). In Family A the per-cell regrets are bit-identical across `phi_k16`,
`phi_k16_cs`, `phi_k16_quality`, `phi_k16_all71`, `phi_k16_guard` and `always_search`; in Family B they are
indistinguishable (|Δ| ≤ 8e−6, e.g. `tail_b0.5_mu1.0_c1.0` 0.001909 vs 0.001917), and `n_demoted` and
`t_cap_hit` differ across variants in all eight environments (`cells_A_primary.csv`, horizon=1000). The
identity is an identity of *outcomes and search decisions*, not of internal state.

**What Φ actually wants is far more search than it gets.** At T=1000, `phi_k16` decides SEARCH on **82.4% of
its fresh decisions** (8-environment mean; 41.7% in `beta_good_common`, 92.4% in `beta_mostly_mediocre`,
97.9% in `tail_b2.0`, and 100.0% in `beta_rare_excellent`, `beta_skewed` and `tail_b8.0`), and a mean of
**1,525,503** cap demotions per environment-cell convert that into the realised `search_frac` of 0.062124
(`cap_demotion_A.csv`, policy=phi_k16, horizon=1000, all 8 rows). `cap_hit_frac` is 1.0 in every
environment, and the median cap-hit time is 6.4–8.0% of the horizon. The cap binds early and then the
policy is irrelevant.

**Uncapped, that same rule is catastrophic.** In the cap sweep at cap = T = 1000, `phi_k16` runs to
K = 987.2 (`search_frac` 0.987) for regret **0.287859** against P3\*'s 0.103364 — Δ = **+0.184494** — and
`cp0` (0.153703) beats it easily; `always_search` is worse still at 0.375018, with 1000 singleton arms and
`n_rec` = 1.0 (`cap_sweep.csv`, mean over the 4 sweep environments). The degenerate-baseline prediction of
register row 13 is confirmed exactly: at cap = T the recommendation is a one-pull tie-break.

**A stopping notion exists at short horizons and vanishes at long ones.** At T=200 with cap 128, `phi_k16`
stops at K = 82.4 (cap_hit 0.389) for regret 0.132055, against `always_search` 0.140941 and P3\* 0.142790
(both at K = 128) — Φ is genuinely not `always_search` relabelled there. At T=1000 with cap 128, by contrast,
`phi_k16` is **bit-identical to `always_search`** (both 0.095835 at K = 128, cap_hit 1.0), so the −0.010749
against P3\* there says nothing about a learned stopping rule. That comparison is plateau-decided and must be
read as such. P3\*'s `c` is selected on a tuning objective that at T ≥ 200 is structurally blind to it: at
T=200, 30 grid values lie within one pooled SE (1.57e−3) of the selected c=3.771, and 5 of them
(c ∈ [2.09, 3.35]) imply K = 72–115 at cap 128 — at or below Φ's 82.4 — which can flip the sign of the
−0.010735. At T=1000 the tuning regret spread over c ∈ [12.27, 64.0] is 3.89e−6 against a pooled SE of
1.15e−3, and 9 of the 15 within-1-SE values saturate cap 128, collapsing the −0.010749 to ≈ 0
(`schedule_tuning.csv`, split=tune, cap=64; Ruling 20). Neither cap-128 number is evidence about a learned
rule.

**Cap 128 is not uniformly better, and the comparison is not like-for-like.** At T=200 `always_search` goes
0.124932 → 0.140941 and P3\* 0.124932 → 0.142790 from cap 64 to cap 128, and at T=1000 `always_search` goes
0.118209 → 0.095835 the other way.

*Read cross-cap differences with care: the cap axis is not CRN-paired.* `base_seed` is a function of
(split, env, horizon, cap) (`cells.py`, `make_cell`), so the four caps draw four different environments'
worth of randomness rather than re-running the same draws under a different budget. The size of that noise is
directly measurable from the two policies whose behaviour cannot depend on the cap at all: over `cp0` and
`refine_after_init`, the pooled cross-cap spread reaches 0.005909 and the per-environment spread reaches
0.018652 (`cap_sweep.csv`, means over the 4 environments). The `always_search` (+0.016010) and P3\*
(+0.017858) changes above clear that floor; `phi_k16`'s (+0.006344, 0.125711 → 0.132055) does not, so this
document does not claim that *every* policy is worse at cap 128 — for the learned policy that difference is
within the unpaired-seed noise. Contrasts computed *within* a single cap (every Δ vs P3\* quoted in this
section) share a `base_seed` and are properly paired; only statements comparing one cap against another are
affected.

And P3\*'s `c` was tuned at cap 64 only (Ruling 20), so the cap-128 column compares a learned policy
extrapolating outside its training support against a schedule extrapolating outside its tuning support.
τ has the same problem in mirror image: all 4,880 rows of `threshold_selection.csv` are cap 64, and τ = 0.5 is deployed unchanged at
every cap. No conclusion about "the value of a bigger cap" is drawn here.

**The pre-registered hypotheses move under the cap.** Over the whole sweep, pooled H1a is −0.013909
(cluster [−0.047846, +0.020029], n_envs=4 — the interval includes zero; on the percentile interval its
upper bound was −0.000245), and at T=1000 it **reverses in sign** to +0.009645 (cluster
[−0.005667, +0.024956], also including zero): Φ loses to `cp0` there, on the point estimate, because `cp0`
never over-recruits. With four environments neither the pooled deficit nor the T=1000 reversal is
cluster-significant; what the sweep establishes is that the sign of H1a is cap-dependent, not its size. Pooled H1b is **+0.000387**, paired [−0.000300, +0.001072], cluster [−0.002035, +0.002809] — a null
(`primary_contrasts_cap_primary.csv`, H1b, level=pooled, n_cells=8, n_envs=4). It is computed on 8 cells
rather than 32 because the analysis now refuses a contrast whose reference baseline carries a constant tuned
at a different cap (`params_cap`; status `untuned_reference`). The +0.019327 this document previously reported
over all 32 cells was that refused comparison: P3\*'s `c`, tuned at cap 64, deployed at caps 32/128/T, and
most of the effect came from the cap = T cells where it is furthest from its tuning point. The honest reading
is that the cap sweep says **nothing** about H1b, not that Φ is much worse there. Pooled H2 is −0.008423 (cluster [−0.023526, +0.006679]); see §7.5 for why that is
not a licence to say the e-process features help (`primary_contrasts_cap_primary.csv`). Note that all 18
`family` and `family_horizon` rows of that table have n_envs = 2 and `cluster_degenerate=True`, where the
"cluster CI" is just the two environment means and can be up to 24× *narrower* than the paired CI; only the
pooled and `horizon` rows (n_envs = 4) carry a non-degenerate interval, and at n_envs = 4 those are
directional evidence, not tests.

**K > 64 is out of support and the figure now says so.** The corpus dropped 12.7% of its states at the cap,
policy-correlated, so the models never saw K ≥ 64.

*Figure: `cap_sweep.png` — regret against cap at T ∈ {200, 1000}, with `cap > 64` shaded as outside the
learned models' training support.*

---

## 5. Why — the mechanism

### 5.1 The regret decomposition

`R_T = R_disc + R_sel` holds on every row of every decomposition table to 2.2e−16
(`decomposition_A_primary.csv`, 2,088 rows). Pooled over Test A (`main_A_primary.csv`, level=pooled):

| policy | R_T | R_disc | R_sel | K_final | search_frac |
|---|---|---|---|---|---|
| `cp0` | 0.148271 | 0.128180 | 0.020091 | 16.25 | 0.0567 |
| `refine_after_init` | 0.182053 | 0.164814 | 0.017239 | 8.00 | 0.0469 |
| `phi_k16` | 0.110106 | 0.054643 | 0.055463 | 45.49 | 0.2441 |
| `p3_star` | 0.111495 | 0.058207 | 0.053288 | 49.80 | 0.2485 |
| `always_search` | 0.135747 | 0.043349 | 0.092398 | 61.20 | 0.4265 |
| `phi_k4` | 0.108050 | 0.051322 | 0.056728 | 51.37 | 0.2730 |

The trade is legible. `cp0` and `refine_after_init` lose almost everything in discovery: they never find a
good arm. `always_search` has the lowest discovery regret of any policy and the highest selection regret —
it finds the good arm and cannot tell which one it is, because at T=50 every arm has one pull
(`n_singletons_final` 28.18 pooled, 50.0 at T=50, where its regret is 0.249848 against Φ's 0.132650). Φ and
P3\* sit at the balance point, and they sit at almost the same place. The horizon structure is the whole
story: discovery dominates as T grows (at T=1000, `phi_k16` is 0.086381 discovery against 0.010896
selection) and selection dominates at T=50 (0.028993 against 0.103658).

*Figure: `decomposition_A.png` — stacked R_disc (bright, lower) + R_sel (dim, upper) per policy per cell.*

### 5.2 Dynamics

The per-step traces (`dynamics_A.csv`, 50-point t/T grid; figures `dynamics_A_familyA.png`,
`dynamics_A_familyB.png`) show the same three regimes the summary numbers imply: at T=50 nobody reaches the
cap and the policies differ in how many arms they open; at T=200 Φ hits the cap in 43% of episodes
(`cap_demotion_A.csv`, cap_hit_frac 0.431250) at a median of 48% of the horizon; at T=1000 every episode
hits the cap at ~7% of the horizon and the remaining 93% of the budget is pure LUCB refinement. The guard
variant is the cleanest evidence that the learned rule is not self-limiting: `phi_k16_guard` (Φ with `cp0`'s
affordability guard) records **1,250,693** guard vetoes at T=1000 summed over the eight Test-A cells —
and 190,912 / 379,119 / 698,611 / 915,219 at T = 50/100/200/500 — against exactly **0** for `phi_k16`,
and still ends at the same K = 64 (`cap_demotion_A.csv`, `n_guard_vetoes`).

### 5.3 The on-policy state shift

The deployed state distribution is strongly separable from the training corpus, and this is not a clock
artefact. Under Ruling 12 the primary OOD columns exclude the 24 fixed logging times (which alone give a
domain AUC of 0.85–0.999; `ood_A.csv`, `domain_auc_CLOCK`, the five non-hand-rule policies), and even so
the clock-excluded domain AUC is **0.980–0.9996** for `phi_k16`, with
per-cell `ood_frac` (fraction of on-policy states beyond the corpus self-kNN 95th percentile) at median
0.094 and mean 0.168; for `phi_k16_all71`, which carries the nine history "fingerprint" features, it is
median 0.582 and mean 0.600 (`ood_A.csv`, per-policy over 40 cells). 130 (cell, policy) rows are flagged above the 0.10 threshold
(`ood_flags.csv`, 130 rows; `ood_A.csv` holds 240 rows over 6 policies × 40 cells, of which the 40
clock-only rows carry no measurement at all). Two caveats: this covers five of the six logged
policies — the clock-only `phi_k16_clock` has `n_features_nonclock = 0` and therefore NaN `ood_frac` and
`domain_auc` in all 40 of its cells, so its shift is characterised only by the clock-confounded
`domain_auc_all` (0.969–0.999), a missing measurement rather than a clean bill; and the OOD analysis was run
for Tests A and C only, never for the cap sweep, which is exactly where the extrapolation is largest.

*Figure: `ood_heatmap_A.png` — per-feature coverage and OOD fraction per cell; the right panel's "flag >
0.10" marking is faint, so read `ood_flags.csv`.*

### 5.4 The reservoir tail: the decision is an estimation problem, not an inference problem

The offline ceiling is unambiguous. Scoring the corpus's own k=16 label with the *oracle* reservoir integral
`I_t` gives AUC **0.8357** over all 82,738 decided rows, while the *observable* estimate `Î_t` gives
**0.5718** (`reservoir_diagnostics_offline.csv`, scope=all). Per horizon the oracle/observable pairs are
0.8556/0.6187, 0.8946/0.6228, 0.9172/0.6449, 0.8975/0.6126 and 0.7558/**0.5032** at T = 50/100/200/500/1000
(scope=T=…). The T=1000 observable figure must be quoted as *chance*: its raw AUC is 0.4968 with
`sign = −1.0`, i.e. the oriented 0.5032 is a sign-flip of a below-chance score, not a weak positive. So a
policy that could see the tail would have a large edge, and the estimator we can actually compute recovers
almost none of it. That is a reservoir-estimation problem, which is what the offline study concluded, and it
is not an inference problem: this is why the e-process columns buy nothing where the policy was trained to
operate (§3.3). They acquire value only outside that support — on the held-out mixture family and in the
uncapped cells of the cap sweep — and the reservoir ceiling says nothing about why (§7.5).

Does Φ act on the tail? At short horizons, yes: the AUC of Φ's own probability for its realised decision is
0.7485 at T=50, falling to 0.6865 (T=100), 0.6231 (T=200), 0.5776 (T=500) and **0.3241** at T=1000
(`reservoir_A.csv`, policy=phi_k16, mean over the 8 cells per horizon). The below-chance value at T=1000 is a
cap artefact, not a statement about the rule: the scored action is the one the simulator took *after*
demotion, and among states at K ≥ 64 the realised search rate is exactly zero while Φ's probability is at
its highest. The hand-built reservoir rule deployed as its own policy (`rule_reservoir`) is poor — pooled
regret 0.123252, Δ vs P3\* +0.011757 (`main_A_primary.csv`, level=pooled) — and the learned logistic version
(`phi_reservoir_rule`, 7 features) is +0.002803, also worse than the schedule. Knowing that the tail is what
matters has not yet produced a rule that exploits it.

*Figure: `reservoir_A.png` — oracle `I_t` against observable `Î_t` and against Φ's decisions, per cell.*

---

## 6. Offline skill versus deployed regret

H3 is the pre-registered check that offline model quality predicts deployed performance. It does not.

Spearman ρ between offline OOF AUC (grouped by environment) and deployed pooled regret across all 22 deployed
learned variants is **+0.005082**, 95% bootstrap CI [−0.527026, +0.523659]; the balanced-accuracy arm is
ρ = −0.036702 [−0.548984, +0.489986] (`surrogate_validity.csv`, test=A, recommender=posterior_mean_shrunk).
The pre-registered statistic is the AUC arm (`DEPLOYMENT_PLAN.md:181`); the sign of the bal-acc arm flips
against it, and the bal-acc levels themselves carry an undischarged optimism — the reported `oof_bal_acc` is
`bal_acc_tau_off`, whose threshold was chosen on the same out-of-fold predictions, +0.0296 mean optimism over
the deployment-relevant `bal_acc_05`, at a τ (0.57–0.69) that is never the deployed τ (0.4–0.5) (Ruling 4).
Recomputed at τ = 0.5 the correlation is −0.005082: the same ≈ 0 conclusion.

The ranking makes the point more vividly than the correlation (`offline_vs_deployed.csv`, primary rule):

| variant | offline OOF AUC | deployed pooled regret |
|---|---|---|
| `clock_quality_evidence_k16_noambig` | **0.770199** (best offline) | 0.110572 (15th of 22) |
| `clock_quality_evidence_k16_hgb` | 0.765680 | 0.110873 (17th) |
| `clock_quality_evidence_k16` (pre-registered Φ) | 0.750147 | 0.110106 (11th) |
| `clock_quality_evidence_k4` | 0.693039 (19th of 22) | **0.108050** (best deployed) |
| `clock_quality_evidence_k1` | 0.637177 (**last** of 22) | **0.108066** (2nd) |

The two best deployed policies rank 19th and 22nd of 22 offline; the best offline model ranks 15th deployed. The commitment horizon k, which the offline
metric treats as just another modelling choice, is the axis that actually matters once the rule is run.

Two related observations. First, the commitment story is consistent across every test: k=1 and k=4 beat k=16
on Test A (§3.6) and are the only variants that generalize to the held-out family (§7.2). Second, the τ story
runs the same way: the offline-optimal τ_off is 0.57–0.69 for every variant, while the deployment-optimal
τ_val is 0.4–0.5 for all of them (`tau_curves.csv`; `offline_vs_deployed.csv`, `tau_off`) — deployment wants
consistently *more* search than the offline balanced-accuracy optimum. An offline threshold would have made
the deployed policy worse.

*Figure: `offline_vs_deployed_A.png` — offline AUC against deployed regret, one point per variant.*

---

## 7. Generalization

### 7.1 Test B is not independent evidence

Test B trains on a corpus excluding the `aggressive`, `random` and `epsilon` generating policies and deploys
on Test A's cells. All 36 `status=ok` rows of `primary_contrasts_B_primary.csv` are **bit-identical** to Test
A's on delta, bounds, se, win rate and counts, because the pre-registered B contrast deploys `phi_k16` on
Test A's cells (`DEPLOYMENT_PLAN.md:222`); the other 18 rows are H2 and read
`status=not_evaluated:phi_k16_quality`, `evaluated=False`, `n_cells=0`, because P7 was never deployed in that
policy set. Test B therefore contributes **no independent pre-registered evidence**, and it is not a
replication of Test A.

Its exploratory rows are the informative part, and they are flat: three variants inside 2.7e−04 of one
another, none with a cluster interval excluding zero (`phi_k16_nopolicy`'s did so marginally on the
percentile interval, at an upper endpoint of −0.000027; on the t interval it is +0.000303). The rows are `phi_k16_all71` −0.001553 (cluster [−0.003861, +0.000755]), `phi_k16` −0.001389
and `phi_k16_nopolicy` −0.001283 (cluster [−0.002869, +0.000303]), at pooled regret
0.109942 / 0.110106 / 0.110212 (`main_B_primary.csv`, level=pooled). Dropping the fingerprint-heavy
generating policies and the history features costs nothing and gains nothing. Do not read this off the Test-B line, decomposition, dynamics or
tau-curve figures: those default to the robustness sweep's six policies, whose intersection with Test B is
`{cp0, p3_star, phi_k16}`, so Test B's ablation arms are simply absent from them.

### 7.2 Test C — the held-out mixture family, where Φ loses

Test C deploys on three `mix_*` environments that were held out of the corpus entirely. The pre-registered Φ
is **worse** than the tuned schedule there: pooled Δ = **+0.003679**, paired [+0.002793, +0.004572], with the
three mixture environments agreeing in sign and their per-environment means spanning
[+0.000319, +0.005428] (`primary_contrasts_C_primary.csv`, H1b, level=pooled). That bracket is *not* a 95%
cluster interval — all 21 Test-C rows have n_envs = 3 and `cluster_degenerate=True`, where the percentile
bootstrap returns exactly the minimum and maximum environment mean. The effect is concentrated at T=100
(+0.005883) and T=200 (+0.011684) and is exactly zero at T=1000, where the cap again equalises everything.

H1a survives the family change and gets larger: −0.074116, paired [−0.075838, −0.072399], all three
environments agreeing in sign with means spanning [−0.098692, −0.059994].

Only the short-commitment variants generalize (`main_C_primary.csv`, level=pooled): `phi_k1` −0.004023 and
`phi_k4` −0.003236 against P3\*, against `phi_k16` +0.003679, `phi_k16_quality` +0.005725,
`phi_k16_perstep` +0.004989 and the Family-A-only model +0.008219. Note that `phi_k16_perstep`, which ties
for first on Test A, is firmly in the losing half here — the "short commitment wins" pattern is about k, and
it does not transfer through the per-step variant.

### 7.3 Test D — horizon transfer, and the headline that was an artefact

Test D deploys horizon-holdout models at T=200 and T=1000 and extrapolates to T=2000 on three environments.
**Its T=2000 baselines were untuned placeholders** (`p3_star` at α=0.5, c=1.0; `refine_after_init` at K0=4)
deployed under the tuned labels, while every other horizon's baseline came from the full protocol. A cost
probe measured one (c, env) evaluation at T=2000 at 0.60 s, implying ~17.5 min single-core for the whole
protocol, so the schedules were tuned properly: the actual run took **137 s** and produced α=2/3, c=0.9153
for P3\* and K0=8 for `refine_after_init`, with `baseline_params.json` byte-identical elsewhere once the six
new T=2000 keys are removed.

The effect of that correction is the headline finding of the fix wave:

| quantity | untuned placeholder | protocol-tuned |
|---|---|---|
| `p3_star` at T=2000: regret / k_final | 0.124022 / 45.0 | **0.106148 / 64.0** (cap-filling) |
| Test-D Δ at T=2000 vs P3\* (all learned + `always_search`) | −0.017889 | **−0.000015** |
| pooled Δ `phi_sf_k16` | −0.003354 | −0.000532 (cluster [−0.001104, +0.000040], includes zero) |
| pooled Δ `phi_k16_cs` | −0.002970 | −0.000148 (cluster [−0.001503, +0.001042], includes zero) |
| pooled Δ `phi_sf_k16_noT1000` | −0.003210 | −0.000388 (cluster [−0.001620, +0.000743], includes zero) |
| pooled Δ `phi_k16_clock` | −0.002220 | **+0.000603** (sign flip) |
| `refine_after_init` at T=2000 | 0.313886 (K0=4) | 0.240173 (K0=8) |

(`main_D_primary.csv`, level=horizon/pooled, `d_regret_vs_p3_star`.) "Horizon transfer works at T=2000" was
an artefact of comparing against a badly-chosen constant. Against a schedule tuned by the same protocol as
every other horizon, the learned policies are 1.5e−05 better, which is nothing; and all four of them are
identical to `always_search` there on every statistic the table carries (regret 0.106133, K = 64,
search_frac 0.031031), so the row says nothing about a learned stopping rule either way. The T=2000 row also has n_envs = 3 and `cluster_degenerate=True`:
its bracket [−0.000029, −0.000002] means all three environments agree in sign, at a magnitude of 1e−5.

What remains of horizon transfer is the T=200 holdout, and it is modest: the model that never saw T=200
(`phi_k16_noT200`) scores Δ = **−0.001542**, paired [−0.002203, −0.000881], **cluster [−0.003831, +0.000747],
which includes zero**, against −0.001889 for the model that *did* see T=200 (`phi_k16_noT1000`)
(`main_D_primary.csv`, level=horizon, horizon=200). Holding out the horizon costs nothing measurable and
buys nothing measurable. The scale-free clock variant demanded by register row 12 is deployed:
`phi_sf_k16` scores pooled Δ = −0.000532, cluster [−0.001104, +0.000040] (`main_D_primary.csv`,
level=pooled). That is an exploratory, uncorrected contrast — H1b is pre-registered only for `phi_k16`,
whose Test-D pooled Δ is −0.000194 with a cluster CI that includes zero — and it is not the largest or the
only one: `phi_k16_noT1000` is lower in pooled regret (0.100824 against 0.101140) at Δ = −0.000945,
cluster [−0.002101, +0.000212], and at T=200 both `phi_k16_noT1000` (−0.001889,
cluster [−0.004203, +0.000425]) and `phi_sf_k16` (−0.001257, cluster [−0.002701, +0.000187]) have
paired CIs excluding zero and cluster CIs that do not (they cleared zero on the percentile interval).
These are not comparable to one another in any case: `phi_sf_k16`'s pooled contrast runs on 19 cells and
`phi_k16_noT1000`'s on 16 (`d_regret_vs_p3_star_n_cells`). Every one of these is a 5e−04 to 2e−03 effect
against a tuned schedule; none is a pre-registered Test-D result.

Two structural facts about Test D that a reader will otherwise get wrong. First, its pre-registered rows
duplicate Test A: 12 of its 18 `ok` rows are bit-identical to Test A's — eight `family_horizon` rows and
four `horizon` rows, at T ∈ {200, 1000} — and its H2 rows are all `not_evaluated:phi_k16_quality` (P7 was never deployed in the D policy set), while the T=2000 rows
read `missing:phi_k16` — the three log-e policies were withdrawn from T=2000 under Ruling 16 because the
exact chunked pairwise evaluator costs hours per item there. Second, **the pooled and family levels now
exclude T=2000 entirely**: the common-support rule averages each stratum's levels over the cells every policy
ran in, which is 16 of 19 for Test D. A subtlety follows from that and is visible in the table: the *levels*
are on 16 cells while a *paired contrast* runs on the two policies' intersection, up to 19
(`main_D_primary.csv` carries both `n_cells` and `d_regret_vs_p3_star_n_cells`), so
`regret(phi_sf_k16) − regret(p3_star) = −0.000628` while the row's `d_regret_vs_p3_star` is −0.000532. Use
the contrast column, not the difference of levels.

### 7.4 Robustness sweep — 30 environments, marginal

All 30 corpus environments × 5 horizons at M=500. Pooled: `phi_k16_quality` −0.000980
(cluster [−0.002009, +0.000050]), `phi_k16` −0.000891 (cluster [−0.001906, +0.000124]), against `cp0`
+0.031917 and `always_search` +0.025593 (`main_robust_primary.csv`, level=pooled; H1a pooled −0.032808
cluster [−0.042996, −0.022621], cluster-significant in 16 of 18 strata). These 30 environments **are exactly
the 30 training-corpus environments**. This is a robustness sweep over a wider environment grid, not
generalization evidence; Test C is the generalization evidence, and there Φ is worse.

### 7.5 H2 across all six tests

H2 is not a uniform null and must be reported test-conditionally
(`primary_contrasts_<test>_primary.csv`, H2, level=pooled):

| test | H2 (Φ − quality-only) | reading |
|---|---|---|
| A | +0.000114, cluster [−0.000277, +0.000505] | null |
| B | **no measurement** — `not_evaluated:phi_k16_quality`, n_cells = 0 | — |
| C (held-out family) | −0.002046, paired [−0.002476, −0.001622], 3-env range [−0.003446, −0.000079] | the features **help**, three of three environments agreeing in sign |
| D | **no measurement** — `not_evaluated:phi_k16_quality`, n_cells = 0 | — |
| cap sweep | −0.008423, paired [−0.009103, −0.007768], cluster [−0.023526, +0.006679] (n_envs=4) | the features help on the point estimate; the cluster interval includes zero — see below |
| robustness (30 corpus envs) | +0.000089, cluster [−0.000132, +0.000309] | null |

So: inside the training support the e-process column is worth nothing; two of the six tests never measured it
at all; outside the support it is worth something. Both of the "helps" need their conditions attached. On
Test C the entire effect lives at T ∈ {100, 200} (T=500 is −0.000063, T=1000 exactly 0, T=50 is
*positive* at +0.000437), and the bracket is a three-environment range, not a 95% interval. On the cap sweep
the effect is concentrated in the uncapped cells. By cap, the mean Δ over the four environments and two
horizons is **−0.000240 (cap 32), −0.000091 (cap 64), −0.001416 (cap 128) and −0.031947 (cap = T)**, whose
average is the pooled −0.008423 (`cap_sweep.csv`, `regret` of `phi_k16` minus `phi_k16_quality` per cell) —
so **94.8% of the pooled effect comes from the eight cap = T cells**, the regime the corpus, which dropped
every K ≥ 64 state, never contained. The defensible statement is: **the e-process features are worth
nothing where the policy was trained to operate, and acquire value only in a regime the training corpus
excludes by construction.**

Similarly for H1b across tests: A/B −0.001389 (cluster includes zero), robustness −0.000891 (cluster includes
zero), D −0.000194 (cluster includes zero), C **+0.003679** (worse), cap **+0.000387** (cluster includes zero,
on the 8 cells whose reference is tuned at the cap it is deployed at — §4). In none of the six tests is the
learned policy cluster-significantly better than the validation-tuned schedule, and in one — the held-out
mixture family — it is worse.
H1a is negative in all of them — A/B −0.038165, C −0.074116, D −0.035534, cap −0.013909 (cluster
[−0.047846, +0.020029], including zero at n_envs = 4), robustness −0.032808 — with the cap-sweep reversal
at T=1000 as the one place its sign fails.

---

## 8. On-policy relabelling (M9)

The register's first failure mode is that the labels are off-policy: `A_t` is one policy-improvement step
away from `cp0`, computed with `cp0` as the continuation. M9 is the direct test. 11,301 states were harvested
from the deployed Φ on Test-A environments with fresh seeds (own seed block, `--early-times 4` so that the
pre-cap decision window is represented — at T ≥ 500 Φ reaches the cap by t ≈ 64–81 and at-cap states have no
SEARCH counterfactual), relabelled at k=16 with **Φ itself as the continuation**, and two new models were
trained: Φ′(union) on corpus + on-policy rows, Φ′(only) on the on-policy rows alone. Both were deployed on
the full Test-A panel.

**The labels do shift, and at long horizons they stop existing.** Of the 11,301 harvested states, 3,675 are
undefined at the cap. At T=1000, **63.6% of the labels are exact ties**, mean |A| = 4.8e−5, and Φ scores
balanced accuracy 0.5086 and AUC 0.4907 on its own continuation labels — chance. At T=500, 31.8% ties and
0.5655 balanced accuracy. At T=50 and T=100 the labels are informative and Φ retains AUC 0.7683 and 0.7291.
P(SEARCH | decided) on-policy is 0.546 and 0.584 at T=50/100 against 0.600 and 0.634 for the corpus's
schedule-generated rows. Overall: on-policy 0.640 agreement / 0.648 balanced accuracy / 0.697 AUC, against
0.709 / 0.662 / 0.757 on the corpus (`onpolicy_label_diagnostics.csv`).

**One policy-iteration step does not materially change the deployed result.** Φ′(union) scores pooled Δ vs
P3\* = −0.001936 (cluster [−0.004382, +0.000509]) against base Φ's −0.001389 (cluster [−0.003899, +0.001121])
— about 0.0005 of pooled regret (`main_A_primary.csv`, level=pooled). Matched directly against base Φ, the
gain is −0.000547 pooled with cluster [−0.001414, +0.000320], i.e. the cluster interval includes zero, and it
is bought entirely at two horizons: −0.001753 at T=100 (cluster [−0.003753, +0.000247]) and −0.000816 at
T=200 (cluster [−0.002339, +0.000707]); at T=500 and T=1000 the two policies are indistinguishable
(Δ = +9e−06 and +1e−06, both CIs straddling zero) (`secondary_contrasts_A_primary.csv`,
policy=phi_k16_onpolicy_union, reference=phi_k16). This is an
exploratory contrast whose own cluster interval against P3\* includes zero (p = 0.10); under Holm across
the 25 pooled learned rows it is adjusted to 1.0.

**Training on on-policy data alone is harmful at long horizons.** Φ′(only) vs base Φ: −0.002774 at T=50
(better), then +0.002583 at T=500 (cluster [−0.000580, +0.005747]) and **+0.029427 at T=1000**
(cluster [+0.007461, +0.051393]), pooled +0.005642. At T=1000 it ends at K = 47.050 with
`search_frac` 0.045141, against base Φ's 64.0 and 0.062124 (`main_A_primary.csv`, level=horizon,
horizon=1000) — it under-searches exactly where its label set is 64% ties. Pooled it ends at K = 46.2 with
`search_frac` 0.275149, above base Φ's 0.244064, so the deficit is a long-horizon phenomenon and not a
global one. Its coefficients confirm the mechanism: agreement with base Φ falls to 0.553 on corpus rows and
its clock coefficients move drastically, one of them flipping sign (`f_K_over_T` from −0.008 to −0.897,
`f_log_t` from +0.784 to −0.497), while Φ′(union)'s coefficients stay close to Φ's and it agrees with Φ on 97.1% of corpus rows and 93.1% of
on-policy rows (`onpolicy_model_shift.csv`).

**Caveats.** The on-policy labels keep the corpus's oracle-prior recommender, so the only things that changed
relative to the corpus are the state distribution and the continuation policy (Ruling 13) — the labels are
not "deployable-recommender consistent", the same caveat the corpus already carries. The continuation
restarts Φ's 16-step commitment cycle at the branch point rather than resuming the deployed phase. And
`--early-times 4` deliberately over-represents early states, which is the only window where the decision is
live at long horizons.

---

## 9. What is and is not established

### 9.1 Established

1. **H1a.** The learned k=16 policy has lower mean paired regret than `cp0` in the pooled result of every
   test that deployed both: cluster-significant pooled and at every horizon in Test A (the six strata with
   eight environments; the twelve with four are negative in every environment but a t interval on three
   degrees of freedom includes zero), in 16 of the 18 strata of the 30-environment robustness sweep, and
   negative in all seven Test-C strata (where, at n_envs = 3, that means all three mixture environments
   agree in sign). It is the only pre-registered result that survives a conservative view — and only where
   there are thirty environments behind it: under Holm, no Test-A row survives in any family (pooled H1a
   p = 0.027 is adjusted to 1.0 in the 54-row family and to 0.31 in the 13 one-row-per-hypothesis-per-test
   rows), and the single survivor of every family is the robustness sweep's pooled H1a (p = 3e−7) with its
   own horizon and family-B strata (§3.5). It is conditional on the 64-arm cap: in the cap sweep at
   T = 1000 it reverses in sign to +0.009645, though at four environments that reversal is not itself
   cluster-significant.
2. **The harness is sound.** CRN holds exactly across all 296 cells; the tables reproduce from the episode
   parquet (all 54 pre-registered rows to max |Δ| = 9.4e−17); levels, decompositions, manifests and summaries
   reconcile across A, B, C, cap and robust to 1e−12; the parity gate passes on all 71 feature columns over
   737,280 states; no deployed policy reads hidden truth; no selection touched the test split.
3. **The cap is the dominant mechanism at T ≥ 200**, and at T = 1000 cap 64 the learned policy is
   operationally `always_search` (§4).
4. **Offline skill does not predict deployed regret** (§6), and the deployment optimum for τ is systematically
   below the offline optimum.
5. **The decision is a reservoir-estimation problem**: the oracle tail integral scores AUC 0.836 on the
   corpus label where the observable estimate scores 0.572 (§5.4).

### 9.2 Not established, and said plainly

- **DP validation.** Register row 14: a dynamic-programming ceiling was declared out of scope in the plan and
  **was not computed**. Nothing in this study is validated against an exact optimum. `dp.py` remains a
  label sanity check only.
- **That Φ beats a tuned schedule.** H1b's pooled cluster CI includes zero, no stratum is
  cluster-significant, at T=500 the schedule wins, and at T ≥ 200 P3\* is itself a cap-filling policy, so the
  comparison there is "Φ vs fill-the-cap", not "Φ vs a schedule".
- **That the policy generalizes.** On the one genuinely held-out environment family, the pre-registered Φ is
  worse than the schedule.
- **That a bigger cap is better, or that Φ's extrapolation beyond K = 64 is beneficial.** P3\*'s `c` and Φ's
  τ are both tuned at cap 64 only (Ruling 20 and §4), so the cap-128 column is not like-for-like — and the cap
  axis is not CRN-paired (§4), so cross-cap differences carry a seed-noise floor of 0.005909 pooled and
  0.018652 per environment, measured on the two cap-independent policies.
- **Any claim about T = 2000 beyond "the tuned schedule and the learned policies are the same policy there"**.

### 9.3 Every ruling taken during this study, with its cost if wrong

1. **Ruling 1** — Wave-1 implementers ran in parallel in the main tree with strict file ownership and
   commit-by-pathspec rather than in worktrees, because the editable install resolves `cold_start` to the main
   `src` and `data/` is gitignored. *Cost if wrong:* a commit race (retry) or a cross-file edit (caught in
   task review).
2. **Ruling 2** — the user's pre-staged `analysis/adaptive_framework_analysis/*` files are not part of this
   work and must never be committed by this branch; every commit uses an explicit pathspec. *Cost if wrong:*
   ten unrelated files land in the branch and must be reverted.
3. **Ruling 3** — the lead wrote `deploy/feature_groups.py` and `deploy/artifacts.py` before dispatch so four
   parallel implementers shared one definition. *Cost if wrong:* a column misclassification, caught by the
   partition test and the parity tests.
4. **Ruling 4** — `bal_acc_tau_off` is an in-sample-τ optimistic number; the deployed τ is chosen on
   validation episodes, so offline τ_off is reported but never used for the primary result. *Cost if wrong:*
   none for the primary result; the H3 bal-acc arm carries the +0.0296 optimism note (§6).
5. **Ruling 5** — `phi_k16_perstep` and `phi_k16_guard` reuse the base artifact's validation τ, being
   deployment modes of the same model. *Cost if wrong:* a slightly suboptimal τ for two secondary variants.
6. **Ruling 6** — the hand rule `rule_reservoir` gets its own τ grid {0.125,…,8}, since its τ has a different
   scale (selected: 0.25). *Cost if wrong:* the P11 hand rule is mis-tuned; it finishes near the bottom of
   the ladder either way.
7. **Ruling 7** — Test D at T=2000 kept the log-e policies via the chunked exact evaluator but ran them last,
   with state logging restricted to six policies. *Cost if wrong:* the on-policy parity gate covers six
   policies rather than all (and, as recorded below, was never recomputed for the two M9 variants).
8. **Ruling 8** — the on-policy parity gate (register #5) is implemented on logged snapshots and the analysis
   refuses to write main tables if any column fails. *Cost if wrong:* a silent vectorized/scalar divergence;
   the gate passed on all 71 columns.
9. **Ruling 9** — horizon-holdout models deploy `tau_val_excl_heldout` everywhere, so neither the model nor
   its τ saw the held-out horizon. *Cost if wrong:* a slightly different τ for three secondary variants;
   measured as costless — the excluded-horizon τ equals the all-horizon τ for all three.
10. **Ruling 10** — re-tune the schedules on a 48-point grid `c ∈ [0.25, 64]` so the selected constant is
    never boundary-limited. *Cost if wrong:* two minutes of compute. It worked: all five selected constants
    are interior.
11. **Ruling 11** — add cap = 128 to the cap sweep so the 64 → T jump has an intermediate point. *Cost if
    wrong:* ~10 minutes of compute.
12. **Ruling 12** — the OOD kNN and domain-classifier primary columns exclude clock features, because the 24
    fixed logging times are a clock fingerprint worth domain AUC 0.85–0.999 on their own
    (`ood_A.csv`, `domain_auc_CLOCK`, the five non-hand-rule policies); the all-feature
    versions are kept and labelled `clock_confounded`. *Cost if wrong:* the OOD flags understate a genuine
    clock shift — mitigated by per-feature coverage rows, but it also leaves the clock-only policy with no
    primary OOD measurement at all (§5.3).
13. **Ruling 13** — on-policy labels keep the corpus's oracle-prior recommender, so the only changes versus
    the corpus are the state distribution and the continuation policy. *Cost if wrong:* the labels are not
    deployable-recommender consistent — the same caveat the corpus already carries.
14. **Ruling 14** — register the two M9 variants in the trainer's `VARIANTS` with a subset marker the
    corpus-only trainer skips, so the policy table stays valid and the runner test stays structural. *Cost if
    wrong:* a runner test that asserts on variant keys breaks.
15. **Ruling 15** — harvest with `--early-times 4` so the T=500/1000 on-policy states include the pre-cap
    decision window. *Cost if wrong:* the on-policy corpus over-represents early states — stated in §8.
16. **Ruling 16** — Test D at T=2000 drops the three log-e policies (`phi_k16`, `phi_k16_noT1000`,
    `phi_k16_noT200`): the chunked exact pairwise evaluator costs hours per item there, not the ten minutes
    estimated. `phi_k16_cs`, whose decisions are bit-identical to `phi_k16` on every Test-A cell, stands in.
    *Cost if wrong:* the T=2000 transfer row for the log-e models is absent — it is, and the tables say
    `missing:phi_k16`.
17. **Ruling 17** — six orphan T=2000 parquet files (the three withdrawn policies in two of the three
    environments), written by workers that outlived a killed parent, were quarantined to
    `episodes_quarantine/D_orphaned_T2000/`, and the D analysis and figure re-run. *Cost if
    wrong:* none; reproducible. The quarantined files were verified byte-identical to the retained
    `phi_k16_cs` on regret, k_final, search_frac and mu_star.
18. **Ruling 18** — `make lint` is red repo-wide with 68 pre-existing ruff errors, none in a file this branch
    touched; the plan's "lint clean" item is satisfied for this branch's surface and cleaning the rest is out
    of scope here. *Cost if wrong:* a reader takes "lint clean" as repo-wide — mitigated by saying so here
    and in the ledger.
19. **Ruling 19** — `run_deployment.py --test D --resume` tries to undo Ruling 16: nothing persists the fact
    that the three T=2000 cells deliberately exclude `phi_k16` / `phi_k16_noT1000` / `phi_k16_noT200`, so a
    plain resume schedules those nine items (hours each). The requirement to pass `--policies` *and* a
    `--cells` list that omits the T=2000 cells is documented in the RUNBOOK and here rather than fixed in
    code; the durable fix (persisting the exclusion in the
    manifest) is **not done**. *Cost if wrong:* someone re-runs Test D and burns hours re-deriving cells
    Ruling 16 deliberately dropped — and, worse, silently changes the published numbers (see Ruling 22).
20. **Ruling 20** — B2 is **not** repaired: P3\*'s `c` is still tuned at cap 64 only, so the cap-128 column is
    not a like-for-like comparison. The figure's training-support shading makes the extrapolation visible; it
    does not fix it. *Cost if wrong:* the report over-claims the cap result — mitigated by the standing
    prohibition in §9.4.
21. **Ruling 21** — F1 is disclosure, not repair: the aggregation guard remains `nunique() >= 2`, so Test C's
    and the cap sweep's intervals still ship, now carrying `cluster_degenerate` (21 of 21 Test-C rows, 18 of
    27 cap rows). Every such row is phrased here as "all environments agree in sign". *Cost if wrong:* a
    degenerate interval is read as a 95% CI.
22. **Ruling 22 (executed)** — nine further orphan parquet files (the same three policies in all three T=2000
    cells, written 19:07–19:34 on 2026-09-19 by workers that outlived the killed `--resume` of Ruling 19,
    absent from `manifest_D.jsonl`) were quarantined to
    `episodes_quarantine/D_orphaned_T2000_20260919/`. `load_cells` globs `*.parquet` with no manifest
    reconciliation, so the next Test-D analysis would have lifted the common support from 16 to 19 cells,
    silently undone Ruling 16 and moved the headline delta and the H3 companions again. The shipped D tables
    (written 18:36) predate the orphans and were never contaminated — but they were not reproducible from the
    tree until this move. Each of the three T=2000 cells now holds exactly the eight parquet files its
    manifest records. *Cost if wrong:* a published number that cannot be regenerated.
23. **Ruling 23 (parked)** — the common-support intersection can empty a stratum: with disjoint coverage,
    `main_table` and `decomposition_table` write **no rows** for that stratum, with only a log warning and
    nothing visible in the CSV. Benign on today's data (no stratum is empty; Test D is the only ragged test,
    at 16 of 19). *Cost if wrong:* a future run ships a table with a stratum missing rather than flagged.

**Carry-forward items** (recorded here because they are recorded nowhere else):

- `phi_k16_clock`'s Test-D pooled delta **flipped sign** with the tuned T=2000 baseline, −0.002220 →
  +0.000603. Anything previously written about the clock-only variant beating P3\* on Test D is void.
- Test D's pooled and family **levels**, and therefore its H3 companion correlations, now exclude T=2000
  entirely. A reader will otherwise assume the longest horizon is in the pooled number. It is not.
- The Test-D H3 family-A sign flip (+0.071 → −0.393 for AUC, +0.286 → −0.607 for bal-acc) was caused by the
  common-support fix, not by the re-tuning: four variants had been compared on 19 cells against three
  variants' 16, and on the common 16 they converge. With n = 7 the Spearman CI is roughly [−1, +0.96], so it
  carries almost no information. The pre-registered H3 lives on Test A, where nothing moved.
- `tau_curves_{B,C,D,robust,smoke}.csv` each gained 10 rows in the re-run (the two M9 variants, whose
  artifacts postdate the tables' previous generation). Shared rows are bit-identical. This was pre-existing
  staleness, now corrected.
- The parity gate that licensed the Test-A tables was computed on the six `--log-policies` snapshot set
  (`phi_k16`, `phi_k16_quality`, `phi_k16_clock`, `phi_k16_all71`, `phi_k16_cs`, `rule_reservoir`; 240 items,
  737,280 states) and was **never recomputed for the two M9 variants**.
- `n_undefined` in `onpolicy_label_diagnostics.csv` (879 for `corpus_all`, 0 for `corpus_schedule`) is a
  lower bound: undefined states after a shard's last emitted row leave no row to carry the count. The 3,675
  on-policy figure is exact, being sourced from the harvest.
- Two bootstraps of record exist for every paired delta CI: `summary_<test>.csv` is the runner's raw output
  (n_boot 2000) and `tables/` is the analysis's (n_boot 10000). They agree exactly on point estimates and
  differ by ~1% of interval width on bounds. All CIs quoted in this document come from `tables/`.
- `decomposition_<test>_primary.csv` writes one `n_cells` from the level aggregate while its `d_*_vs_cp0`
  columns run over the full paired intersection, and unlike `main_*` it emits no per-contrast `n_cells`. On
  ragged strata (Test D) the level difference and the contrast column will not agree.

### 9.4 Claims this study does not make

These are the adversarial reviewer's "would not sign" list, reproduced as explicit non-claims. None of them
is stated anywhere above, in any paraphrase.

1. **Not claimed:** "On Test D the learned policy beats the tuned schedule, pooled Δ = −0.0034
   [−0.0057, −0.0008]." 84% of that effect came from three cells whose `p3_star` was the untuned placeholder;
   with the baselines tuned the pooled figure is −0.000532 for `phi_sf_k16` and smaller or null for the rest.
2. **Not claimed:** "Horizon transfer works: at T=2000 the learned policy reaches 0.106 vs the tuned
   schedule's 0.124 (Δ = −0.0179)." No tuned schedule existed at T=2000 when that was measured; tuned, it
   scores 0.106148 and Δ = −0.000015. All four learned policies at T=2000 are byte-identical to
   `always_search`.
3. **Not claimed:** "At T=2000 `refine_after_init` scores 0.3139." That was the placeholder K0 = 4; at the
   tuned K0 = 8 it scores 0.240173.
4. **Not claimed:** "At cap=128 Φ's out-of-support extrapolation is beneficial at both horizons (−0.011 both)
   because more arms genuinely help up to 128." At T=1000 the −0.010749 is bit-identical to `always_search`'s
   and rests on a P3\* constant its own selection objective cannot identify; and at T=200 cap 128 is worse
   than cap 64 for `always_search` (+0.016010) and P3\* (+0.017858), though Φ's own +0.006344 sits inside the
   0.005909 unpaired-seed floor, so "every policy" is not claimed either (§4).
5. **Not claimed:** "Φ′(union)'s cluster CI excludes zero." It does not on the t interval (p = 0.10); only
   `phi_k4` and `phi_k1` do, and those are exploratory and uncorrected, not confirmatory.
6. **Not claimed:** "Test D: T=200 `noT200` −0.0019." That is `noT1000`'s number. `noT200` is −0.001542 and
   its cluster CI includes zero.
7. **Not claimed:** "H3: Spearman(offline OOF AUC, deployed regret) = +0.089 [−0.49, +0.60]." That is a stale
   ledger line predating the M9 variants. The table gives +0.005082 [−0.527026, +0.523659] over 22 variants,
   with the bal-acc arm at −0.036702 and the opposite sign.
8. **Not claimed:** any pooled or family-level raw `regret` from `main_D_primary.csv` read as if all policies
   shared a cell set. Per-horizon Test-D rows are safe; the pooled levels are on the 16-cell common support.
9. **Not claimed:** "at T=1000 H1b is exactly zero" or "every learned variant is bit-identical per episode at
   T=1000" without scope. True in Family A; Family B is −1.48e−6, and `t_cap_hit`/`n_demoted` differ across
   variants in all eight environments.
10. **Not claimed:** "Φ beats the validation-tuned schedule" as an unconditional headline. Every statement of
    H1b here carries its horizon and cap conditions.
11. **Not claimed:** "the 30-environment sweep shows the policy generalizes." Those 30 environments are the
    training-corpus environments.
12. **Not claimed:** "Test B independently replicates Test A." Test B's `ok` rows are bit-identical to Test
    A's; the same applies to Test D's T=200 and T=1000 cells.
13. **Not claimed:** "the offline surrogate predicts deployed performance."
14. **Not claimed:** that the cluster-bootstrap CIs are conservative at small n_envs. At n_envs = 4 a
    percentile cluster bootstrap under-covers substantially; family-level cluster CIs are directional
    evidence, not tests. At n_envs ≤ 3 they are degenerate, as stated throughout.
15. **Not claimed:** anything resting on a secondary recommender table without naming it as secondary.
16. **Not claimed:** "DP validation supports this." It is out of scope and absent (§9.2).
17. **Sign trap, noted for the next reader:** `secondary_contrasts_A_primary.csv` contains 18 rows of
    (policy=`phi_k16_quality`, reference=`phi_k16`) that are the exact negation of the primary H2 rows
    (pooled −0.000114 against the primary table's +0.000114). Match on the (policy, reference) **pair**,
    never on policy name alone.
18. **Not claimed:** "cluster-significant" for any Test-C row. All 21 have n_envs = 3.
19. **Not claimed:** "cluster-significant" for any cap-sweep `family` or `family_horizon` row. Those have
    n_envs = 2, where the interval is the two environment means and can be 24× narrower than the paired CI.
20. **Not claimed:** "228 pre-registered rows outside Test A corroborate the result." 48 are untestable NaNs
    (B 18, D 18, and the cap sweep's 12 `untuned_reference` refusals), 48 are Test A's rows relabelled (all 36
    Test-B `ok` rows and 12 of Test D's 18), and of the 132 that are new measurements (C 21, cap 51, robust
    54, and Test D's 6 family/pooled rows) only 69 carry a cluster interval that is an interval — the 21
    Test-C rows and 42 of the cap rows sit below `CLUSTER_MIN_ENVS` and ship [min, max] ranges instead.
21. **Not claimed:** "Test B's pre-registered contrast confirms Test A."
22. **Not claimed:** "H2 was measured on all six tests." Tests B and D have no H2 row.
23. **Not claimed:** "the e-process features help" as an unconditional reading of the cap sweep's −0.008423.
24. **Not claimed:** that any pre-registered result survives multiplicity correction on the cluster interval,
    except H1a on the 30-environment robustness sweep (pooled, its five horizon rows and family B's six);
    no Test-A, B, C, D or cap-sweep row survives in any family (§3.5).
25. **Not claimed:** "seven learned policies beat P3\* significantly." Two do on the uncorrected t interval,
    against 1.25 expected by chance, and none survives Holm.
26. **Not claimed:** "Φ′(union) is significant." Its cluster interval includes zero and its Holm-adjusted
    p is 1.0.
27. **Not claimed:** any internal ordering of `phi_k4` / `phi_k1` / `phi_k16_perstep`. The spread is 5.9e−05
    and the order changes with the recommendation rule.
28. **Not claimed:** any policy ranking read off `regret_vs_T_*.png` at the right-hand edge. Exact ties are
    now merged into a single label rather than fanned into a spurious ladder, but the CSV is the source.
29. **Not claimed:** anything read off the decomposition figures' old caption. R_disc is the bright, lower
    segment and R_sel the dim, upper one; the caption and the two-swatch key now say so.
30. **Not claimed:** anything about Test B's ablation read from a Test-B line, decomposition, dynamics or
    tau-curve figure. Those default to the robustness sweep's six policies.
31. **Not claimed:** any delta CI quoted from `summary_<test>.csv`.
32. **Not claimed:** the reservoir AUC pair 0.836 / 0.572 as a per-horizon result, or 0.503 as a weak
    positive. Those are `scope=all`; the per-horizon pairs are in §5.4 and the T=1000 observable is chance
    (raw 0.497, sign −1).
33. **Not claimed:** "Φ learns to stop searching at long horizons." At T ≥ 500 the cap is the policy.

### 9.5 Deferred minors that could matter

Five were dispositioned in the completeness round; two change a quoted number. **(a)** τ is selected only on
cap-64 data and the selector never conditions on cap, structurally the same censoring as P3\*'s `c` — but far
better identified (the τ grid spans 9.6 SE pooled over the 40 selection cells, against 0.0034 SE for `c` at
T=1000), and the residual exposure is small: τ = 0.5 beats τ = 0.4 pooled by 1.04 SE, with T=50 supplying
108% of that margin. If the cap-sweep's `c` is ever re-tuned per cap, τ must be re-selected in the same pass
or the asymmetry simply reverses. **(b)** The quoted OOD fractions are medians; the cell means are higher
(§5.3). The other three are inert: the cross-recommender disk backfill did not fire (all five per-recommender
tables per test were written in one contiguous wave with identical policy sets), the pairwise-table race was
discharged by building the tables in the parent before forking, and the on-policy undefined counter is a
shard-scoped lower bound, not a per-chunk restart.

---

## 10. Errata for `RESULTS.md`

The offline study's document remains broadly correct, and its central negative finding — that the e-process
evidence does not drive the decision — is *confirmed* by this deployment (§3.3, §7.5, §5.4). Seven items
need correcting, and one of its predictions can now be closed out.

1. **"The clock + all the e-process evidence" (Finding 1 table, k=1 0.5410 / k=16 0.5853) is not all the
   e-process evidence.** That row is ablation D, which the offline selector implemented as clock + the single
   column `f_log_e_pair` (`fit_models.py:50-64`, substring selector). The deployment study defines feature
   groups by explicit column list: the deployed clock+quality+evidence set is 62 columns against 35 for
   clock+quality and 11 for clock (`offline_vs_deployed.csv`, `n_features`). The finding survives the
   correction — the full-EVIDENCE deployed model and the CS-only model differ by 1.0e−06 in pooled regret —
   but the row as printed understates what was ablated.
2. **`f_challenger_ucb`'s single-feature AUC of 0.706 (Finding 1, second table) is a CS-derived feature and
   is sign-flipped.** Its raw AUC is 0.294: a high challenger UCB predicts REFINE, not SEARCH. Listing it at
   the top of a table headed by quality features conflates the two groups.
3. **"the k=1 and k=16 answers agree on direction only 52.2% of the time" (Finding 3) counts ties.** Among
   decided pairs the agreement is **70.4%**. The qualitative point — that k=1 and k=16 are different control
   problems — stands, and the deployment supports it: k=1 and k=4 beat k=16 when deployed (§3.6, §6).
4. **The six-variable `(z,u,k,τ,r,h)` form of Φ at k=1 is at chance (balanced accuracy 0.5065).** Any
   suggestion that the legacy `LearnedPolicy` form is a usable deployment path should be struck; the deployed
   models here are the fitted sklearn pipelines over explicit 11–71-column feature sets.
5. **`PILOT_FINDINGS.md` §4's "the recommendation rule does NOT drive the benchmark" (0.1528 / 0.1528 /
   0.1527) was measured on a single policy.** The deployment measures it properly across 36 policies and 40
   cells: the four non-naive rules agree on the policy ranking at pooled Kendall τ 0.948–0.995 and change no
   pre-registered conclusion (§3.4), but the naive `empirical` rule agrees at only 0.752–0.782 and is strictly
   worse for all 36 policies. The claim is right in substance and was under-evidenced as stated.
6. **"~100k labelled states" (RUNBOOK §1) is 87,148**, of which 12,692 states (12.7%) were *dropped* at the
   64-arm cap (`DEPLOYMENT_PLAN.md:100`), and the drop is policy-correlated. At k=16, 82,738 rows are
   decided, 5.1% are exact ties and 15.7% are truncated or cap-demoted (`offline_diagnostics.csv`,
   `label_n`, `label_fraction`). The cap-exclusion is not a rounding detail: it is why K ≥ 64 is out of the models' support (§4).
7. **Three numbers in the shipped documents could not be reproduced** and should be marked as such rather
   than quoted: the schedule-honesty triple "0.5889 → 0.5854 → 0.5218" (`RESULTS.md`, "Mistake 3"), the
   `fill_to_cap` 36-cell grid ("won 27/36", `PILOT_FINDINGS.md` §3), and the recommender-agnostic 0.1528
   triple of item 5.
8. **`RESULTS.md`'s open question can now be closed.** It says: *"That a fitted rule would actually beat a
   schedule when deployed … has not been run with a fitted Φ yet."* It has now. The answer is that a fitted Φ
   beats the label's continuation policy `cp0` decisively and does **not** beat a validation-tuned growth
   schedule (H1b pooled cluster CI includes zero; worse on the held-out family and in the cap sweep), and that
   the offline margin was indeed an upper bound — offline AUC and deployed regret are uncorrelated (ρ ≈ 0).

---

## 11. Reproduction

All commands from the repository root with `.venv/bin/python`, in this order. Seed blocks: tune
`1_000_000 + cell_id·1_000`, validation `5_000_000 + …`, test `10_000_000 + …`, on-policy harvest
`15_000_000 + …`, where `cell_id` enumerates (env, T, cap); a startup assertion checks disjointness from the
corpus block [20,262,460, 343,348,312] and the old benchmark block [4,504, 14,111].

This section reproduces the deployment study only. It assumes the labelled corpus of RUNBOOK.md §1 already
exists at `data/oracle_labels` (`experiments/growing_bandits/label_states.py`, ~3-4 h on 9 workers, resumable
through `manifest.jsonl`); Family C stays held out, so `--include-mixtures` is not passed. RUNBOOK.md Part II
carries the same sequence with wall times, resume semantics and the seed-block table.

```bash
# 1. Train every policy variant from the corpus (writes models/, offline_metrics.csv,
#    offline_diagnostics.csv, feature_hygiene.csv, reservoir_diagnostics_offline.csv,
#    reproduction_gate.json).
.venv/bin/python experiments/growing_bandits/deploy/train_policies.py --n-jobs 12

# 2. Tune the schedule constants on the tune split, select P3* and K0 on validation.
#    Ruling 10's grid; the horizon list must include every horizon, T=2000 included.
.venv/bin/python experiments/growing_bandits/deploy/tune_baselines.py \
    --horizons 50 100 200 500 1000 2000 --c-min 0.25 --c-max 64 --n-c 48 --workers 12

# 3. Select tau per learned model on the validation split (writes threshold_selection.csv,
#    thresholds.json, and the full tau -> regret curve).
.venv/bin/python experiments/growing_bandits/deploy/select_thresholds.py --workers 12

# 4. Smoke run end-to-end before the long jobs.
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test smoke --workers 12

# 5. Deployment runs, in this order. Each is resumable through manifest_<test>.jsonl.
# Test A before M9 deploys the 34 corpus-trained policies; the two on-policy variants
# of step 8 have no artifact yet and the runner exits on a missing model.
A_POLICIES=$(.venv/bin/python -c "import sys; sys.path.insert(0,'experiments/growing_bandits/deploy'); import policy_table as pt; print(','.join(pt.CORPUS_POLICIES))")
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test A      --workers 12 \
    --policies "$A_POLICIES" --log-states \
    --log-policies phi_k16,phi_k16_quality,phi_k16_clock,phi_k16_all71,phi_k16_cs,rule_reservoir
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test C      --workers 12 \
    --log-states \
    --log-policies phi_k16,phi_k16_quality,phi_k16_clock,phi_k16_all71,phi_k16_cs,rule_reservoir
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test B      --workers 12
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test cap    --workers 12
# cap=128 (Ruling 11) is not in CAP_SWEEP_CAPS; it is driven by an explicit cell list
# (M=1000 comes from DEFAULT_REPLICATES['cap'], the same as the rest of the sweep).
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test cap    --workers 12 \
    --cells beta_good_common:200:128,beta_good_common:1000:128,beta_rare_excellent:200:128,beta_rare_excellent:1000:128,tail_b2.0_mu1.0_c1.0:200:128,tail_b2.0_mu1.0_c1.0:1000:128,tail_b8.0_mu1.0_c1.0:200:128,tail_b8.0_mu1.0_c1.0:1000:128
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test robust --workers 12

# Test D runs in two passes. See the warning below.
# Pass 1 — the eight policies Ruling 16 keeps at every horizon (the full D grid;
# the T=2000 cells keep their M=1000 override).
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test D --workers 10 \
    --policies cp0,p3_star,refine_after_init,always_search,phi_k16_cs,phi_k16_clock,phi_sf_k16,phi_sf_k16_noT1000

# Pass 2 — the three exact chunked log-e variants, on the sixteen T<=1000 cells ONLY
# (Ruling 16: each costs hours per item at T=2000). Do NOT drop the --cells list.
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test D --workers 10 --resume \
    --policies phi_k16,phi_k16_noT1000,phi_k16_noT200 \
    --cells beta_good_common:200:64,beta_rare_excellent:200:64,beta_mostly_mediocre:200:64,beta_skewed:200:64,tail_b0.5_mu1.0_c1.0:200:64,tail_b2.0_mu1.0_c1.0:200:64,tail_b8.0_mu1.0_c1.0:200:64,tail_b1.0_mu0.9_c2.0:200:64,beta_good_common:1000:64,beta_rare_excellent:1000:64,beta_mostly_mediocre:1000:64,beta_skewed:1000:64,tail_b0.5_mu1.0_c1.0:1000:64,tail_b2.0_mu1.0_c1.0:1000:64,tail_b8.0_mu1.0_c1.0:1000:64,tail_b1.0_mu0.9_c2.0:1000:64

# 6. Analysis (all recommenders; B/cap/robust/D take Test A's parity gate).
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test A      --recommender all
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test C      --recommender all
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test B      --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test cap    --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test robust --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test D      --recommender all --gate-from A

# 7. Figures.
for t in A B C D cap robust; do
  .venv/bin/python experiments/growing_bandits/deploy/make_deploy_figures.py --test $t
done

# 8. On-policy relabelling (M9), then redeploy the two new variants on Test A.
.venv/bin/python experiments/growing_bandits/deploy/relabel_onpolicy.py harvest --early-times 4 --workers 12
.venv/bin/python experiments/growing_bandits/deploy/relabel_onpolicy.py label   --workers 12
.venv/bin/python experiments/growing_bandits/deploy/relabel_onpolicy.py train
.venv/bin/python experiments/growing_bandits/deploy/select_thresholds.py \
    --variants phi_k16_onpolicy_union,phi_k16_onpolicy_only --workers 12
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test A --resume --workers 12 \
    --policies phi_k16_onpolicy_union,phi_k16_onpolicy_only
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test A --recommender all --skip-snapshots
.venv/bin/python experiments/growing_bandits/deploy/make_deploy_figures.py --test A
```

> **Warning — re-running Test D requires `--policies` *and* a `--cells` list (Ruling 19).** Nothing in the
> code or the manifest records that the three T=2000 cells deliberately exclude `phi_k16`, `phi_k16_noT1000`
> and `phi_k16_noT200` (Ruling 16: each costs hours through the exact chunked log-e evaluator). A plain
> `--resume` schedules those nine items. `--policies` alone is not enough — `--policies` is global and is
> crossed with every cell of the grid, so the three excluded variants must be given a `--cells` list that
> omits the T=2000 cells. Never pass the three of them without `--cells`, and never pass `--cells` for a
> T=2000 cell (`parse_cell_overrides` returns `m_override=None`, so the cell would silently run at
> `DEFAULT_REPLICATES['D']` = 2000 instead of `T2000_REPLICATES` = 1000). Worse, if the run is then killed, the pool workers outlive the parent and write complete parquet
> files that never enter `manifest_D.jsonl`, and `load_cells` globs `*.parquet` with no manifest
> reconciliation — so the next analysis silently lifts the common support from 16 cells to 19, undoes
> Ruling 16 and moves the headline delta and the H3 correlations. This has happened **twice** (Rulings 17 and
> 22); both batches are quarantined under `results/growing_bandits/deploy/episodes_quarantine/`. After any
> Test-D run, check that each `*_T2000_cap64` cell holds exactly eight parquet files and that every file has a
> manifest record.

**Verification.** `make test` (980 passed at HEAD, including the ten added by the fix wave);
`.venv/bin/ruff check experiments/growing_bandits/deploy/ tests/test_deploy_*.py` (clean — the repository-wide
`make lint` is red on 68 pre-existing errors in untouched files, Ruling 18); the parity gate
(`onpolicy_parity.csv`, 71/71 passed) and the CRN identity test gate the analysis itself.

**Provenance of the numbers in this document.** Tables: `results/growing_bandits/deploy/tables/*_primary.csv`
plus the cross-recommender `primary_contrasts_A_{lcb,oracle_prior,posterior_mean,empirical}.csv` and
`main_A_empirical.csv`,
`cap_sweep.csv`, `cap_demotion_A.csv`, `ood_A.csv`, `ood_flags.csv`, `reservoir_A.csv`, `tau_curves.csv`,
`surrogate_validity.csv`, `offline_vs_deployed.csv`, `recommender_kendall.csv`, `onpolicy_parity.csv`.
Root-level provenance: `baseline_params.json`, `schedule_tuning.csv`, `schedule_selection.csv`,
`threshold_selection.csv`, `offline_metrics.csv`, `offline_diagnostics.csv`,
`reservoir_diagnostics_offline.csv`, `onpolicy_label_diagnostics.csv`, `onpolicy_model_shift.csv`,
`reproduction_gate.json`, and the six `manifest_<test>.jsonl`. `results/` is gitignored wholesale
(`.gitignore:55`), so these are committed by targeted negation, never by `git add -A`.
