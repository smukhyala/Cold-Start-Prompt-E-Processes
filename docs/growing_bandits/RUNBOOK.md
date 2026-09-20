# Growing bandits — how to reproduce this study

All commands from the repo root, using `.venv/bin/python`.

## 1. Generate and label the corpus  (M4)

```
.venv/bin/python experiments/growing_bandits/label_states.py \
    --horizons 50 100 200 500 1000 --states-per-horizon 20000 \
    --trajectories 8 --snapshots 8 --max-live-arms 64 \
    --target-se 3e-4 --max-replicates 4096 --commit-steps 1 4 16 \
    --workers 9 --chunk-shards 8 --out data/oracle_labels
```

1560 shards, ~100k labelled states, ~3-4 h on 9 workers. **Resumable**: the run
appends completed shard ids to `data/oracle_labels/manifest.jsonl`, and re-running the
same command skips them. Parquet chunks are written atomically (`tmp` + `os.replace`),
so an interrupted run never leaves a footer-less file that would make the whole
directory unreadable.

Family C (the mixture reservoirs) is **held out** unless `--include-mixtures` is passed.
It is the out-of-distribution test set for section 15.

## 2. Inspect before modelling  (M5 gate)

```
.venv/bin/python experiments/growing_bandits/review_dataset.py --data data/oracle_labels
```

Reports label precision, design coverage, where the signal sits by remaining budget,
how the answer moves with the commitment horizon, and a verdict listing any blocking
issues. Read this before fitting anything.

## 3. Fit the decision function  (M6)

```
.venv/bin/python experiments/growing_bandits/fit_models.py --data data/oracle_labels
.venv/bin/python experiments/growing_bandits/fit_models.py --data data/oracle_labels --label label_A_k16
```

Climbs the model ladder on the six theoretical variables, runs ablations A-F with the
model class held fixed, and tests generalization by holding out whole groups
(environment, family, horizon, behavioural policy) rather than rows. Writes JSON + CSV
to `results/growing_bandits/tables/`.

## 4. Benchmark the learned policy  (section 17)

```
.venv/bin/python experiments/growing_bandits/evaluate_policy.py \
    --phi results/growing_bandits/tables/phi.json --horizons 100 200 500 --seeds 2000
```

`phi.json` is `{"coef": {...}, "intercept": float}` over the six variables
`z, u, k, tau, r, h` (interactions as `"a:b"`, squares as `"a^2"`). Without `--phi` it
benchmarks the baselines alone.

## 5. Figures  (section 16)

```
.venv/bin/python experiments/growing_bandits/make_figures.py --data data/oracle_labels
```

PNG+PDF pairs plus the CSV each figure was drawn from, in
`results/growing_bandits/figures/`.

## Validation

```
make test                      # 810 tests
make lint
.venv/bin/python experiments/growing_bandits/run_pilot.py     # end-to-end + section 21 contrasts
```

The toy dynamic program in `src/cold_start/growing/dp.py` is the only ground truth in
the study — it solves small two-point-reservoir environments exactly by backward
induction, and is what the Monte Carlo labels can be checked against.

---

# Part II — the deployment study  (`DEPLOYMENT_PLAN.md` → `DEPLOYMENT_RESULTS.md`)

Everything above reproduces the *offline* study: the labelled corpus and the fitted decision function.
This part reproduces the *deployment* study — running a fitted Φ from scratch as a real algorithm against
the baselines. New code lives in `src/cold_start/growing/deploy/` (library) and
`experiments/growing_bandits/deploy/` (runnable scripts); output goes to
`results/growing_bandits/deploy/` (`episodes/`, `tables/`, `figures/`, `models/`, plus the root provenance
files). All commands are from the repo root with `.venv/bin/python`, and every stage is resumable.

Two figures in Part I are corrected in DEPLOYMENT_RESULTS.md §10: §1's "~100k labelled states" is 87,148
rows, of which 12,692 (12.7%) were dropped at the 64-arm cap.

## Seed blocks

`cell_id` enumerates (env, T, cap); `CELL_STRIDE = 1000`.

| block | base seed | used by |
|---|---|---|
| tune | `1_000_000 + cell_id·1_000` | schedule constant `c`, `K0` (`tune_baselines.py`) |
| validation | `5_000_000 + cell_id·1_000` | P3\* selection, τ selection (`select_thresholds.py`) |
| **test** | `10_000_000 + cell_id·1_000` | every reported number (`run_deployment.py`) |
| on-policy | `15_000_000 + cell_id·1_000` | M9 harvest + labelling (`relabel_onpolicy.py`) |

Disjoint by construction from the corpus block `[20_262_460, 343_348_312]` and the old benchmark block
`[4_504, 14_111]`; `cells.assert_seed_disjointness()` is called at the start of `run_deployment.py`,
`tune_baselines.py`, `select_thresholds.py` and `relabel_onpolicy.py` and will abort on a collision.
Within a cell, every policy shares the cell's `base_seed` (common random numbers); policy-internal
randomness is `default_rng([base_seed, crc32(policy_name)])`.

## 1. Train the policy variants  (M4)

```
.venv/bin/python experiments/growing_bandits/deploy/train_policies.py --n-jobs 12
```

≈ 6 min. Writes `models/<variant>.joblib` (`{pipeline, feature_list, k, tau, meta}`), `offline_metrics.csv`,
`offline_diagnostics.csv`, `feature_hygiene.csv`, `reservoir_diagnostics_offline.csv` and
`reproduction_gate.json`. The gate asserts the new trainer reproduces the offline headline
(bal-acc 0.6472 vs target 0.647; AUC 0.7405 vs 0.7405, tol 0.01) before any protocol change is allowed.

## 2. Tune the baselines  (M5)

```
.venv/bin/python experiments/growing_bandits/deploy/tune_baselines.py \
    --horizons 50 100 200 500 1000 2000 --c-min 0.25 --c-max 64 --n-c 48 --workers 12
```

≈ 2–3 min at 12 workers (137 s for the six-horizon grid). Ruling 10's 48-point geometric grid keeps the
selected constant off the boundary. **List every horizon you want in `baseline_params.json`**: the selection
re-derives the ones already on disk, so a short list silently drops them. Writes `schedule_tuning.csv`
(tune split), `schedule_selection.csv` and `schedule_oracle_tuned.csv` (validation), and
`baseline_params.json`. An untuned horizon is not a silent failure any more — the runner stamps
`params_tuned: False` plus the substituted constants into the manifest, and `analyze_deployment.py` refuses
to pool a contrast whose reference is untuned, labelling it `status=untuned_reference` instead.

## 3. Select the thresholds  (M5)

```
.venv/bin/python experiments/growing_bandits/deploy/select_thresholds.py --workers 12
```

≈ 4 min. τ per learned model = argmin mean validation regret over τ ∈ {0.3,…,0.7}, one global scalar per
model; the horizon-holdout models get `tau_val_excl_heldout`, selected without the held-out horizon
(Ruling 9). Writes `threshold_selection.csv` (the full curve) and `thresholds.json`. The script refuses
`--split test` unless `--allow-test-split` is passed, and suffixes any such file `_TESTSPLIT`.

## 4. Smoke, then the deployment runs  (M6)

```
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test smoke --workers 12

# Test A before M9 deploys the 34 corpus-trained policies; the two on-policy variants
# of §6 have no artifact yet and the runner exits on a missing model.
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

# Test D, pass 1 — the eight policies Ruling 16 keeps at every horizon (the full D grid;
# the T=2000 cells keep their M=1000 override).
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test D      --workers 10 \
    --policies cp0,p3_star,refine_after_init,always_search,phi_k16_cs,phi_k16_clock,phi_sf_k16,phi_sf_k16_noT1000

# Test D, pass 2 — the three exact chunked log-e variants, on the sixteen T<=1000 cells
# ONLY (Ruling 16: each costs hours per item at T=2000). Do NOT drop the --cells list.
.venv/bin/python experiments/growing_bandits/deploy/run_deployment.py --test D      --workers 10 --resume \
    --policies phi_k16,phi_k16_noT1000,phi_k16_noT200 \
    --cells beta_good_common:200:64,beta_rare_excellent:200:64,beta_mostly_mediocre:200:64,beta_skewed:200:64,tail_b0.5_mu1.0_c1.0:200:64,tail_b2.0_mu1.0_c1.0:200:64,tail_b8.0_mu1.0_c1.0:200:64,tail_b1.0_mu0.9_c2.0:200:64,beta_good_common:1000:64,beta_rare_excellent:1000:64,beta_mostly_mediocre:1000:64,beta_skewed:1000:64,tail_b0.5_mu1.0_c1.0:1000:64,tail_b2.0_mu1.0_c1.0:1000:64,tail_b8.0_mu1.0_c1.0:1000:64,tail_b1.0_mu0.9_c2.0:1000:64
```

Wall times on 12 workers: A 23 min (1360 items in this pass; manifest_A reaches 1440 after the M9 pass of
§6), C 18 min (510), B 7 min (200), cap 3 min (144 + 48 = 192), robust 6 min (900), D ~20 min (200).
Per-episode results go to
`episodes/<test>/<cell>/<policy>.parquet` with a `summary_<test>.csv` and a resumable
`manifest_<test>.jsonl`. `--resume` re-runs only items whose params, artifact hash or code fingerprint
changed. The pairwise log-e tables are built in the parent process before the pool forks; T=1000 is a
~4 GB memmap under `data/pairwise_tables/`.

> ### ⚠ Re-running Test D requires `--policies` and `--cells`  (ledger Ruling 19)
>
> Nothing persists the fact that the three T=2000 cells **deliberately exclude** `phi_k16`,
> `phi_k16_noT1000` and `phi_k16_noT200`. Ruling 16 withdrew them because the exact chunked log-e evaluator
> costs *hours per item* at T=2000 (the original 10-minute estimate was wrong; nine items stalled for 57
> minutes). The stand-in is `phi_k16_cs`, whose decisions are bit-identical to `phi_k16` on every Test-A
> cell, which is why the eight-policy set at T=2000 still carries a Φ arm. A plain
> `run_deployment.py --test D --resume` will schedule those nine items again.
>
> The second-order failure is worse than the wasted compute. If you then kill the run, the pool workers
> outlive the parent and each finishes writing its parquet file — complete, valid, and **absent from
> `manifest_D.jsonl`**, because the manifest is written by the parent. `load_cells` globs `*.parquet` with
> no manifest reconciliation, so the next `analyze_deployment.py --test D` picks them up, lifts the
> common support from 16 cells to 19, silently undoes Ruling 16, and moves the pooled delta, the family
> levels, `offline_vs_deployed_D` and the H3 correlations. **This has happened twice** — see Rulings 17 and
> 22; both batches of orphans are quarantined under
> `results/growing_bandits/deploy/episodes_quarantine/D_orphaned_T2000*/`.
>
> So: `--policies` alone is not enough — `--policies` is global and is crossed with every cell of the grid,
> so the three excluded variants must be given a `--cells` list that omits the T=2000 cells. Never pass the
> three of them without `--cells`, and never pass `--cells` for a T=2000 cell: `parse_cell_overrides` returns
> `m_override=None`, so the cell would silently run at `DEFAULT_REPLICATES['D']` = 2000 instead of
> `T2000_REPLICATES` = 1000. After any Test-D run verify that each `*_T2000_cap64` cell directory holds
> **exactly eight** parquet files and that every parquet has a matching `manifest_D.jsonl` record before
> running the analysis. The durable fix — persisting the per-horizon exclusion and making `load_cells`
> refuse an unmanifested parquet — is **not implemented**.

## 5. Analysis and figures  (M7)

```
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test A      --recommender all
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test C      --recommender all
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test B      --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test cap    --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test robust --recommender all --gate-from A
.venv/bin/python experiments/growing_bandits/deploy/analyze_deployment.py --test D      --recommender all --gate-from A

for t in A B C D cap robust; do
  .venv/bin/python experiments/growing_bandits/deploy/make_deploy_figures.py --test $t
done
```

Always run the analysis with `--recommender all`: the cross-recommender frames have no freshness check, so a
single-recommender run can leave stale sibling tables behind. Test A recomputes the on-policy feature-parity
gate from its logged snapshots (737,280 states, 71 columns) and **refuses to write the main tables if any
column fails**; the tests that log no snapshots take A's gate with `--gate-from A`. Outputs land in
`tables/` (the bootstrap of record — never quote a CI from `summary_<test>.csv`, which is the runner's own
lower-`n_boot` draw) and `figures/` (PNG + PDF + the CSV each figure was drawn from).

## 6. On-policy relabelling, and the second Test-A pass  (M9)

```
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

≈ 1–2 h end to end. `harvest` draws states from the deployed Φ on its own seed block with a per-record seed
check; `--early-times 4` is required (Ruling 15) because Φ reaches the cap by t ≈ 64–81 at T ≥ 500 and
at-cap states have no SEARCH counterfactual. `label` re-runs the oracle label with **Φ itself as the
continuation**; `train` fits `phi_k16_onpolicy_union` (corpus + on-policy) and `phi_k16_onpolicy_only`.
Note that the Test-A parity gate is not recomputed for these two variants — the `--skip-snapshots` pass
reuses the original gate artifact.

## 7. Validation for this part

```
make test                                                             # 980 passed at HEAD
.venv/bin/ruff check experiments/growing_bandits/deploy/ tests/test_deploy_*.py
```

`make lint` is red repo-wide on 68 pre-existing ruff errors, every one in a file this branch never touched
(ledger Ruling 18); the branch's own surface is clean, which is what the command above checks.
`results/` is gitignored wholesale (`.gitignore:55`), so tables are committed by targeted negation —
never `git add -A`.
