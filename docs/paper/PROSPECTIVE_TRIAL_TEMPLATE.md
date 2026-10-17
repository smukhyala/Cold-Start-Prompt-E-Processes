# Prospective fixed-pair trial template — unregistered until filled and committed

This is an implementation-ready checklist, **not** a completed preregistration. Fill every bracketed field and commit the completed protocol before observing any result from the new task cohort.

## Frozen design

- Target task population and sampling frame: `[source, inclusion rule, sampling seed, task IDs or sealed manifest]`. The tasks must be new to the candidate-selection process. Record known overlap with public benchmarks or model training when assessable.
- Concurrent incumbent: `[system/prompt/model versions and hashes]`.
- One challenger: `[system/prompt/model versions and hashes]`. No substitution after pilot outcomes.
- Paired outcome: `[binary success definition, grader version, treatment of partial credit]`.
- Task order: `[seed and deterministic generator]`; run both arms on each task within the same session window where possible.
- Operational failures: `[timeout, provider error, retry count, missing pair, and exclusion rules]`. Freeze handling before outcomes. Preserve attempted and excluded records.
- Maximum paired tasks `N`: `[integer]`. Exact full-grid cost for this fixed pair is `2N` episodes. Add expected episode price and wall-time budget using `experiments/growing_bandits/audit/design_budget.py --challengers 1 --tasks N`.

## Inference

- Familywise budget: `alpha_total=0.05`, allocated `alpha_eff=0.025` and `alpha_fut=0.025` if both directional claims will be made. If only efficacy is primary, specify its allocation before collection.
- Meaningful gain: `delta=0.15` unless the application justifies a different value before seeing results.
- Efficacy: the repository's incumbent-relative E1b path. Futility: the repository's downward paired-difference process at the frozen margin. Record first crossing, maximum e-value, final e-value, and paired task count. If neither crosses by `N`, report unresolved.
- Any pilot is part of the same monitored path. It may estimate discordance and operational cost, but cannot be used to change the incumbent, challenger, outcome, or target tasks without a new protocol and new task cohort.
- Report paired success rates, discordant wins/losses, costs, missingness, and session timestamps alongside the sequential result. Keep outcome-blind audit logs and all exclusions.

## Launch gate

Before a live run, fill the fields above, verify that the task cohort is independent of candidate selection, execute a tiny outcome-blind harness check, and record the committed protocol hash. A scientific result requires the complete task-level outcome table, not only aggregate success counts. Selection of a fresh cohort and runnable systems remains open in the current repository plan.
