# GitLab 160 Early Identification Analysis

- True best arm from paired sweep: `gitlab_oracle_operator`.
- Paired success rate for true best: `0.875`.
- Best non-oracle paired success rate: `0.700`.
- E-process certification threshold: `log(1 / alpha) = 2.996` with `alpha=0.05`.
- Empirical leader metrics require at least `3` pulls for an arm.

Key definitions:

- `first_oracle_eprocess_beats_runner_up_t`: first task index where the oracle arm's one-sided hedged-capital e-process rejects the null that its mean is no better than the paired runner-up mean.
- `stable_oracle_empirical_leader_t`: first task index after which the oracle remains the empirical success-rate leader among arms with enough pulls.
- `stable_oracle_policy_score_leader_t`: first task index after which the oracle remains the top SPRUCE score arm, when policy scores are logged.
- `*_tasks_saved`: remaining task budget if we committed to the oracle at that decision time.
- `*_projected_final_successes_if_commit_oracle`: successes so far plus remaining budget times the paired oracle success rate.

Current run-level summary:

| policy  | replicate | observed_timesteps | complete | successes_observed | final_oracle_pulls | first_oracle_eprocess_beats_runner_up_t | stable_oracle_empirical_leader_t | stable_oracle_policy_score_leader_t |
| ------- | --------- | ------------------ | -------- | ------------------ | ------------------ | --------------------------------------- | -------------------------------- | ----------------------------------- |
| spruce  | 0         | 160                | True     | 56                 | 8                  |                                         |                                  |                                     |
| uniform | 0         | 160                | True     | 43                 | 8                  |                                         | 134.000                          |                                     |
| spruce  | 1         | 160                | True     | 40                 | 8                  |                                         |                                  |                                     |
| uniform | 1         | 160                | True     | 58                 | 8                  |                                         | 127.000                          |                                     |
| spruce  | 2         | 160                | True     | 55                 | 9                  |                                         | 160.000                          | 160.000                             |
| uniform | 2         | 160                | True     | 33                 | 8                  |                                         | 65.000                           |                                     |

Policy-level rollup:

| policy  | success_rate_observed_mean | final_oracle_pull_share_mean | stable_oracle_empirical_leader_t_mean | stable_oracle_policy_score_leader_t_mean | first_oracle_eprocess_beats_runner_up_t_mean | first_oracle_eprocess_beats_runner_up_t_rate | first_oracle_cs_best_certified_t_rate | stable_oracle_empirical_leader_t_rate | stable_oracle_policy_score_leader_t_rate |
| ------- | -------------------------- | ---------------------------- | ------------------------------------- | ---------------------------------------- | -------------------------------------------- | -------------------------------------------- | ------------------------------------- | ------------------------------------- | ---------------------------------------- |
| spruce  | 0.315                      | 0.052                        | 160.000                               | 160.000                                  |                                              | 0.000                                        | 0.000                                 | 0.333                                 | 0.333                                    |
| uniform | 0.279                      | 0.050                        | 108.667                               |                                          |                                              | 0.000                                        | 0.000                                 | 1.000                                 | 0.000                                    |

Generated artifacts:

- `identification_timeseries.csv`
- `identification_summary.csv`
- `identification_summary_by_policy.csv`
- `plots/`
