# GitLab Strong-Arm Positive-Control Experiments

GitLab is used as a positive control because software-engineering browser tasks
have clear stateful workflows: inspect the repository or issue, change a small
piece of state, then verify the final state. That structure makes it plausible
that prompt strategy can create a real, measurable gap between arms.

The suite adds several GitLab-specialized strategies, but it does not assume any
of them wins. The paired benchmark runs every arm on the same task sequence and
defines `true_best_arm` as the arm with the highest empirical paired success
rate. If a generic arm wins, the adaptive experiments use that generic arm as
the reference.

Adaptive allocation should begin with broad exploration and then reduce
allocation entropy as evidence accumulates. Entropy is useful because it shows
whether a policy is still spreading budget uniformly or concentrating future
trials on a smaller set of promising prompt strategies.

Figures:

- `budget_success_rate`: whether adaptive policies improve with more budget.
- `budget_regret`: cumulative loss relative to the paired empirical best arm.
- `budget_identification_probability`: how often the true best arm is stably
  most-pulled by the end of a run.
- `entropy_time`: exploration-to-exploitation dynamics. Uniform should remain
  near maximum entropy; adaptive policies should decline if they exploit.
- `best_arm_share_time`: whether adaptive policies concentrate trials on the
  strongest empirical arm.
- `final_entropy_budget`: how concentration changes as budget increases.
- `cost_per_success_budget`: cost efficiency of each allocation rule.
- `scaling_success_rate` and `scaling_regret`: behavior as the prompt pool grows.

These experiments complement the Gmail and Shopping sweeps by providing a
positive-control setting where specialized software-engineering behavior should
be easier for adaptive policies to discover and exploit.
