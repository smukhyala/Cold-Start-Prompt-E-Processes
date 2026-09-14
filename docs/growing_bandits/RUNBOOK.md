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
