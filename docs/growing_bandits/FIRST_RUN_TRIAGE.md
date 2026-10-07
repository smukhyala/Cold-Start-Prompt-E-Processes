# First-run triage: completed no-cost analysis and prospective protocol

*2026-10-07. No model calls or new episodes were run. This is an exploratory replay of saved Gmail trials, followed by a protocol for an independent trial.*

## Decision being tested

After one prompt-conditioned agent runs, choose **stop**, **rerun the same agent**, or **try a different agent** using only information available at that point. The practical objective is verified task success at a stated cost, not oracle coverage from two independent attempts. A choice must work for state-changing tasks: each saved episode used a separate sandbox reset, so offline replay selects a complete trajectory, not just a final response to paste into an already modified environment.

## Saved-trial comparison

The G and F Gmail pools have 145 and 155 randomly sampled same-agent rerun anchors on 29 and 30 represented tasks. The first and same-agent second outcomes are observed at those anchors. For a different second agent, the analysis averages over all 49 other prompts' main runs on the same task; this is **not** a randomized paired second-agent trial. Success is checked against the frozen outcome matrix. The decision rule sees only `is_done`, `timed_out`, and whether `final_result` is nonempty. It never uses `success` or `verifier_message` to decide.

The fixed **timeout rule** accepts a first run that appears complete. Otherwise it resets and executes a second run, accepting that run's trajectory. We compare it with stopping after the first, and with always paying for a second run while using the same completion-based choice rule. The latter has the same selected outcome but tests whether triage saves cost.

| Pool | Policy | Retry rate | Verified success | Mean cost/task | Cost/verified success |
|---|---|---:|---:|---:|---:|
| G | Stop | 0% | 63.4% | $0.095 | $0.150 |
| G | Retry same if incomplete | 13.8% | 65.5% | $0.125 | $0.191 |
| G | Try different if incomplete | 13.8% | 65.8% | $0.124 | $0.189 |
| F | Stop | 0% | 61.9% | $0.089 | $0.143 |
| F | Retry same if incomplete | 10.3% | 64.5% | $0.111 | $0.172 |
| F | Try different if incomplete | 10.3% | 64.0% | $0.107 | $0.168 |

The same-agent retry adds +2.1 points in G (95% task bootstrap: 0.0 to +5.3) and +2.6 in F (+0.6 to +5.1). Its extra inference cost per anchor is $0.030 in G and $0.023 in F, about **$1.46 and $0.87 per additional successful task** on these anchors. The different-agent estimate adds +2.3 and +2.1 points, for about $1.24 and $0.91 per additional success. These ratios are descriptive point estimates; small gains make them unstable. The cost per success across the whole task stream *increases* under either retry policy. Thus a retry is sensible only if the value of rescuing a particular failed task outweighs its extra cost; under a pure successes-per-dollar objective, stopping dominates in this sample.

The different-versus-same success contrast after timeout is +0.3 points in G (task bootstrap −0.7 to +1.5) and −0.5 in F (−1.5 to +0.5). There is **no reliable switch-versus-rerun advantage** here. Always paying for a second run gives the same selected successes under this completion rule, but costs roughly $0.181–$0.189 per task. The extra run on apparently complete first attempts buys nothing unless a better verifier or selection rule is supplied.

The task bootstrap resamples the 29–30 represented tasks, conditional on the fixed prompt and task pools. G and F share tasks, so they are not independent replications. The replay was designed after observing related analyses of these same trials; its intervals are descriptive. In particular, a completed first run can be wrong. The timeout rule intentionally leaves those errors uncorrected.

## Probe of apparently completed runs

To see whether saved runtime metadata offers a stronger trigger, we fit one regularized logistic model to predict first-run failure from step count, cost, elapsed time, final-response length, nontrivial error count, and G/F pool. A separate regularized linear model predicted the observed gain from a same-agent rerun using those same features. Predictions were cross-fitted in five folds grouped by **task ID across both pools**; no task ID appears in both the training and test side of a fold. Neither model receives the hidden verdict or second-run outcome as an input at decision time. These models and the 20% retry fraction were exploratory choices made after the saved trials existed.

Among 264 apparently completed anchor runs, 76 first runs were wrong. The failure model's grouped-CV AUC was 0.63 overall (G 0.60, F 0.64), with only a small Brier-score improvement over predicting training-fold prevalence. Retrying the 20% of completed runs it judged most likely to be wrong **reduced** success in both pools: choosing the same-agent second run changed success by −12 percentage points among selected G cases and −14 points among selected F cases. Predicting failure selected hard tasks that the second agent often also failed.

The model aimed directly at *retry benefit* did better descriptively: it selected 25 G and 28 F completed anchors and produced **one net additional success in each pool**. Across all 264 completed anchors that is +0.76 points (task bootstrap 0.0 to +2.3), at $0.0062 extra per completed task. Two net rescues on the same reused task banks are far too little evidence for a deployed rule; changing the feature set or the next tasks could reverse the result. This probe sharpens the next-trial target: a useful policy must predict **recoverability after retry**, rather than just the chance that the first attempt is wrong.

I also inspected the saved browser-action histories for three fixed observable cues: whether the final answer *claimed* verification, whether a read-style action appeared anywhere, and whether one appeared after the final click. Among completed anchors, claimed verification appeared in 176/264 runs; success was 72.2% there versus 69.3% without it. A read-style action after the last click appeared in only 11/264 runs, and those runs succeeded less often (36.4% versus 72.7%). In this scaffold, extra inspection often accompanies a hard task, while many successful short tasks end directly after a click. These crude cues do not constitute a reliable verifier. The summary is descriptive and was chosen after the outcomes were available.

## What the next independent trial must test

The saved data reveal a simple timeout rule but cannot validate a stronger rule for **apparently completed but incorrect** first runs. That is the first-run research question that remains open. The independent trial should be frozen as follows before inference:

1. **Task split:** hold out new tasks, preferably new task templates, before fitting a rule. Do not randomly split repeated instances of the same template between training and test. State the inference budget and stop date in advance.
2. **First run:** choose the anchor prompt and task distribution in advance. Record its trace, elapsed time, step count, cost, final response, and any task-specific UI evidence available without querying the hidden benchmark scorer.
3. **Second-run assignment:** randomize eligible tasks 1:1 to a same-agent rerun or a different frozen prompt. Include a prespecified sample of apparently completed first runs; otherwise the central false-completion question is unidentifiable. Reset the sandbox for each isolated trajectory. Retain all failed, timed-out, and infrastructure cases in an intention-to-treat table, with infrastructure failures flagged separately.
4. **Rules and baselines:** freeze at least three policies: stop; retry after observable timeout; and a candidate trace-based rule that may also retry apparently completed runs. Compare same versus different second attempts at equal cost and matched retry rate. If a task-specific verifier is proposed, count its cost and prohibit access to the hidden scoring verdict at decision time.
5. **Primary endpoint:** verified success of the *chosen trajectory* per task together with mean cost per task. Report the success-cost frontier and incremental dollars per added success. Score any safety-relevant unintended actions too. Use task or template clusters for uncertainty, and evaluate the final rule once on the heldout set.

The 30 unused Gmail tasks could check feasibility but are underpowered for a 2–3 point gain. The earlier [cost estimate](FRESH_TRIAL_COST_ESTIMATE.md) put a three-run pass on those 30 tasks near $8–9 per Gmail pool; a randomized two-run design would have a different, likely lower cost. No such run was started because the standing constraint is no new spending. A larger independent task bank is needed before claiming that a learned rule generalizes.

## Reproduction

Run `./.venv/bin/python experiments/growing_bandits/retrospective/first_run_triage.py`, `./.venv/bin/python experiments/growing_bandits/retrospective/first_run_trace_probe.py`, and `./.venv/bin/python experiments/growing_bandits/retrospective/history_evidence_probe.py` from the repository root. The code, cell-level anchors, grouped-CV predictions, history cues, summaries, and input hashes are in those scripts and `results/growing_bandits/retrospective/first_run_triage/`.
