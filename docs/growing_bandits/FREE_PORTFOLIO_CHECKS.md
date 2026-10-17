# No-cost checks of agent portfolio value

*Exploratory analysis, 2026-10-06. This used only previously saved runs and a public outcome file. No model inference or paid trial was run.*

## Questions and sources

1. Can a rule available at deployment choose the better of two recorded Gmail agent runs?
2. Does candidate selection transfer across unseen task types in an independent public agent benchmark?

For (1), we used the frozen G and F Gmail pools (50 prompt-conditioned agents × 30 tasks each) and their 145 and 155 randomly sampled same-agent reruns. Success labels came from the frozen `data/empirical_pool/reservoirs/outcomes_snapshot.jsonl`, checked against the original analysis. Runtime information came from `logs/empirical_pool/worker_*.jsonl`. For (2), we used [WorkBench Revisited's item-level file](https://github.com/LucasEtteldorf/workbench/blob/main/data/results/item_level_results.csv.gz), limited to its 24 `revisited_2026` model runs using the same v2 ground truth and all-tools configuration. This gives a complete 24-model × 690-task outcome matrix across 69 base task templates. Its source file and MIT license are copied into the result directory; the manifest fixes its SHA-256 hash.

These are distinct questions. The Gmail agents differ by prompt under one agent setup. The WorkBench agents are different models, evaluated in a common benchmark scaffold. Neither dataset contains agent communication, a deployed verifier, or a live portfolio policy.

## 1. A deployment-observable choice rule on Gmail runs

The fixed rule chooses the second run **only if** the first did not finish with a nonempty final response and the second did. Otherwise it chooses the first. It reads `is_done`, `timed_out`, and `final_result`; it never reads success, the benchmark's `verifier_message`, or the task's ground truth. The rule is evaluated on the same randomly repeated anchor cells as the earlier oracle-coverage analysis. A distinct-agent comparison averages over every other prompt's saved main run on the anchor task. Because those main runs were not paired in the same randomized rerun schedule, the distinct-agent estimate is more model-dependent than the same-agent estimate.

| Pool | First run | Same-agent choice | Same-agent perfect-choice ceiling | Distinct-agent choice | Distinct-agent perfect-choice ceiling |
|---|---:|---:|---:|---:|---:|
| Gmail G, 145 anchors / 29 tasks | 63.4% | 65.5% | 69.0% | 65.8% | 70.4% |
| Gmail F, 155 anchors / 30 tasks | 61.9% | 64.5% | 71.0% | 64.0% | 72.1% |

Same-agent gain over the first run was **+2.1 percentage points** in G (95% task bootstrap: 0.0 to +5.3) and **+2.6 points** in F (+0.6 to +5.0). The rule rescued 3 of the 8 G anchors and 4 of the 14 F anchors that an oracle could rescue. It made no harmful switches in these saved pairs because no timed-out first run was benchmark-correct. The 95% task bootstraps resample the 29–30 represented tasks, conditional on the fixed pool and this selected set of rerun cells; they are not prospective guarantees.

The key failure is among apparently completed runs. Both runs appeared complete in 117 G and 126 F pairs. Of those, 8 and 16 pairs had discordant correctness. In 5 G and 10 F pairs, the first agent looked complete but was wrong while the second was right. The rule cannot identify them and leaves the remaining oracle gain unrealized. Some agent responses confidently reported completion despite a failed benchmark verdict. A practical result selector would need task-specific evidence from the environment, not just the agent's declaration.

Two runs still incur two runs' cost. These are isolated sandbox episodes with independent resets. Choosing a successful *trajectory* after both have executed is an offline construct; for state-changing Gmail tasks, simply selecting its final text would not apply that trajectory's state changes to a shared live environment. This study therefore supports a **timeout triage signal**, not a working multi-agent deployment.

## 2. Public WorkBench transfer check

For each of 1,000 seeded partitions, entire **base templates** were assigned to the training or test side. Each method saw the same full model × training-task grid. We selected either the highest training success rate model, the two highest-rate models, or the highest-rate model plus a second model chosen for training-task coverage. Results below average test-task coverage over reused partitions. Random single- and two-model baselines are computed exactly within each test set. The 5th–95th split percentiles in the CSV express sensitivity to partition choice; these are **not** confidence intervals from 1,000 independent experiments.

| Method | Heldout task coverage | Gain over random team of same size | Gain over selected single model |
|---|---:|---:|---:|
| Selected single model | 97.4% | +23.5 pp | — |
| Two highest-rate models | 98.7% | +10.5 pp | +1.3 pp |
| Greedy coverage pair | 98.5% | +10.3 pp | +1.1 pp |

Across splits, the perfect-choice ceiling over all 24 models averaged 99.9%. The top single was Claude Fable 5 in 920/1,000 partitions. WorkBench therefore provides an **external positive example of transferable candidate quality**: broad model differences are learnable on one set of templates and persist on others. But the best single model already approaches the ceiling. A second model contributes only about 1.3 points of *oracle coverage*, and greedy diversity selection did not beat picking the two best marginal models. WorkBench measures correctness and unwanted side effects; all `correct=True` rows in this cohort had `unwanted_side_effects=False`, but running two agents in one real environment could still cause side effects before any choice is made.

This does not replicate the Gmail prompt-diversity effect: model quality, model cost, task family, and benchmark scaffold all differ. The random-model comparison also spans old and modern systems with a very wide quality range. Thus the large advantage over a random model is evidence that **some candidate pools have strong transferable differences**, not evidence that the Gmail pools do or that a second agent is cost-effective.

## Thesis implication and remaining empirical gap

The combined evidence supports a conditional framing: candidate search pays off when quality differences transfer across tasks; prompt pools that are flat and noisy may not repay a broad audit. The WorkBench matrix gives a clean external contrast to our flat prompt pools. The Gmail saved runs show that the potential gain from two attempts is larger than what a simple observable completion rule can recover. The remaining publishable question is how to obtain reliable, affordable **task-level evidence** for selecting or stopping agents, especially when both claim to have completed a state-changing task. The existing records cannot establish coordination, communication, or live shared-state recovery.

## Reproduction

From the repository root:

```bash
./.venv/bin/python experiments/growing_bandits/retrospective/free_checks.py
./.venv/bin/ruff check experiments/growing_bandits/retrospective/free_checks.py
```

Code: `experiments/growing_bandits/retrospective/free_checks.py`. Input hashes, source file, license, cell-level Gmail rows, WorkBench split rows, and summaries: `results/growing_bandits/retrospective/free_checks/`. The copied WorkBench file is under its bundled MIT license and preserves the authors' attribution.
