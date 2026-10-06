# Retrospective task-transfer analysis of prompt selection

*Exploratory reanalysis, 2026-10-06. No new agent episodes or paid API calls. The
research questions were formed after the outcome data were known. Code:
`experiments/growing_bandits/retrospective/task_transfer.py`; local outputs:
`results/growing_bandits/retrospective/`. This document is separate from the
registered verdicts in `DEPLOYMENT_RESULTS.md`.*

## Question and design

Does the prompt with the highest success rate on one group of tasks succeed more
often on other tasks? The comparison is to a prompt sampled uniformly from the
same candidate pool. It is a measure of the *value of selecting a prompt using
these tasks*, not a comparison of live search policies at a fixed episode budget.

The four primary pools use one recorded outcome per prompt-task cell:

| pool | prompts | tasks | source | qualification |
|---|---:|---:|---|---|
| G | 50 | 30 | Gmail grid `outcomes_snapshot.jsonl` | same task bank as the registered real-pool study |
| F | 50 | 30 | Gmail free-form `outcomes_snapshot.jsonl` | same task bank |
| GLK30 | 40 | 30 | GitLab procedural `outcomes_snapshot.jsonl` | outcome rescored at 30 steps |
| GitLab17 | 17 | 40 | older GitLab paired `paired_results.csv` | excludes `gitlab_oracle_operator`, whose prompt contains a task-specific cookbook |

The fifth pool, GitLab18, retains the cookbook prompt as a **sensitivity check
only**. The older GitLab study has a different task mix, prompt construction and
agent regime, and is not an independent replication of the newer GitLab pool.

For each pool, 2,000 seeded random splits put half the tasks in training and
half in testing, separately within each difficulty tier. The training winner
has the highest mean success rate on training tasks; ties are broken randomly.
The outcome is its test rate minus the average test rate of all prompts in that
pool. Training optimism is the analogous training difference. Every arm shares
every task, and no task appears on both sides of a split. We also omit each
task once and repeat 300 splits to check sensitivity to a single task.

## Results

All values below are percentage points. The 5th–95th range is the distribution
**across reused task splits**, not a confidence interval for future tasks.
The leave-one-task range is a sensitivity check, not a confidence interval.

| pool | training winner's apparent gain | held-out gain | 5th–95th across splits | splits with gain > 0 | leave-one-task range |
|---|---:|---:|---:|---:|---:|
| G | +13.5 | **−2.5** | −10.5 to +5.6 | 30.7% | −3.3 to −0.9 |
| F | +14.2 | **−4.6** | −14.5 to +4.3 | 19.3% | −5.2 to −2.5 |
| GLK30 | +16.9 | **−2.0** | −13.7 to +7.5 | 38.1% | −3.2 to +0.3 |
| GitLab17 | +20.0 | **+8.6** | −4.4 to +18.8 | 86.8% | +6.5 to +11.7 |
| GitLab18, cookbook included | +32.7 | **+32.0** | +21.7 to +42.2 | 99.3% | +29.0 to +34.9 |

In the three newer pools, the apparently best prompt looks 13–17 points better
than the pool average on the tasks used to select it, but is slightly *worse*
on held-out tasks. The result is directionally robust to leaving out one task
for G and F; GLK30 changes sign after one omission, so its negative point
estimate is less stable. The older GitLab pool shows that the same procedure
can find transferable differences when a broader prompt-quality gap is present.
Its cookbook sensitivity is large but does not support a general prompt-search
claim because that prompt contains explicit instructions for the benchmark's
tasks.

## Pilot diagnostic: does a small audit predict the value of selection?

For each outer split, eight prompts are sampled as a pilot. Twelve training
tasks, balanced across difficulty tiers, are divided into two disjoint halves.
The diagnostic is the cross-prompt covariance of success rates on those halves:
a positive covariance is evidence that pilot prompts rank similarly on two
task samples. The test outcomes play no role in that calculation.

| pool | mean pilot covariance | share of positive pilot estimates | held-out gain from selecting among only eight pilot prompts | extra held-out gain from selecting among all prompts |
|---|---:|---:|---:|---:|
| G | +0.0002 | 44.5% | −0.9 points | −1.6 points |
| F | −0.0009 | 37.6% | −2.8 points | −1.8 points |
| GLK30 | +0.0003 | 49.6% | −0.6 points | −1.4 points |
| GitLab17 | +0.0094 | 74.5% | +8.3 points | +0.3 points |

The pilot signal separates GitLab17 from the three newer pools *on average*.
It does **not** predict which splits within a pool will benefit: mean held-out
gain conditional on a positive pilot estimate is nearly the same as when the
estimate is nonpositive in every pool. There are only four substantive pools,
all already studied, so this is **not validation of a deployable pre-test**.
The eight-versus-all comparison also gives the larger search a larger
evaluation cost; it must not be called a fixed-budget policy comparison.

## Interpretation and limits

This adds a concrete consequence to the registered real-pool measurements:
selection by noisy observed success rates can overfit the task bank, even when
the training winner appears substantially better. The older GitLab contrast
shows that the held-out procedure is capable of finding genuine transferable
differences in at least one previously collected setting. It does not establish
that prompt search usually helps in GitLab or usually harms in Gmail.

The same finite tasks are reused across the 2,000 splits. Their split
percentiles are not 2,000 independent experimental replicates. All four pools
were examined before this analysis was specified, and candidate construction
and task difficulty differ across studies. The held-out tasks are unseen by
the selector within each split but are not a new external benchmark. The data
still cannot rule out rare excellent prompts outside the sampled pools.

## Reproduction and checks

From the repository root:

```bash
./.venv/bin/python experiments/growing_bandits/retrospective/task_transfer.py
./.venv/bin/pytest tests/test_task_transfer.py
./.venv/bin/ruff check experiments/growing_bandits/retrospective/task_transfer.py tests/test_task_transfer.py
```

The script refuses duplicate or missing prompt-task cells and nonbinary
outcomes, records source SHA-256 hashes and seeds in `provenance.json`, and
writes split-level, summary and leave-one-task CSVs. The older paired input
and the four small retrospective outputs are force-tracked despite the
repository's general `results/` ignore rule, so a clean checkout has the
inputs and saved outputs needed to audit this analysis.
