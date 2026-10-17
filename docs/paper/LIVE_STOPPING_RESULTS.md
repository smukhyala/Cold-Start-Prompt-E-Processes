# Live stopping: futility boundary at 35 pairs in an authorized amended continuation

**Executed 2026-10-16.** The controller automatically stopped at the first
practical-futility crossing after **35 complete pairs**, before scheduling pair
36. This prevented **170 remaining planned agent executions**. The result is from
an authorized amendment after an infrastructure interruption; the original
13-pair preregistered run remains unresolved at its recorded operational endpoint.
The original protocol was not edited, and this is not a new candidate comparison.

## Design and amendment

The [original protocol](LIVE_STOPPING_PROTOCOL.md) fixed the Linear baseline and
explorer prompts, `gpt-5.4-mini-2026-03-17`, independent uniform draws with
replacement from the complete 120-task bank, maximum 120 pairs, delta 0.15,
alpha_E = alpha_F = 0.025, and threshold 40. Seed, task order, verifiers, prompts,
model, reasoning effort, execution limits, betting implementations, and the
study-wide **$75 cap** remained unchanged.

The original run ended with an invalid challenger in pair 14 after a request
timeout overlapping host sleep. A later resume audit correctly found that the
frozen no-retry rule prohibited continuation. The user then explicitly authorized:
“rerun the one it failed on (my computer went to sleep) and go from there”. The
[dated amendment](LIVE_STOPPING_RESUME_AMENDMENT.md) records this single exception,
including the fact that the first 13 outcomes and failed partial trajectory were
already known. It was committed before additional inference.

The continuation reused all 13 complete pairs and the valid baseline of pair 14.
Only its challenger was rerun, under physical ID
`pair_000014_challenger_retry1`, from reset app state. The retry yielded a valid
failure (0), paired with the saved baseline failure (0). The original invalid
attempt remains null, with its charges retained. Pairs 15–35 then ran in the
original order, baseline followed by challenger. No evidence process was restarted
and no further retry was made.

## Statistical results

| Quantity | Final amended trajectory |
|---|---:|
| Decision / stopping reason | Practical-futility boundary / statistical crossing |
| Complete pairs / maximum | 35 / 120 |
| Baseline successes | 20/35 = 57.142857% |
| Challenger successes | 16/35 = 45.714286% |
| Challenger-minus-baseline difference | −4/35 = −11.428571 percentage points |
| Challenger-only wins / baseline-only wins / ties | 1 / 5 / 29 |
| Efficacy evidence | 1.0000000000 |
| Futility evidence at pair 34 | 37.7355809354 |
| Futility evidence at pair 35 | 41.2660497151 |
| Natural log futility at pair 35 | 3.7200401211 |
| Threshold / natural log threshold | 40 / 3.6888794541 |
| First statistical crossing | Futility, pair 35 |
| Terminal timestamp | 2026-10-16T23:51:03.196557Z |

The saved outcomes exactly reproduce the online evidence using the original
`e1b_log_paths` and `futility_log_paths`. No boundary crossed at pairs 1–34.
The controller recorded its terminal state 18.308 milliseconds after the final
challenger core receipt, with `recovered_before_scheduling=false`: the boundary
was acted on during execution. No pair 36 intent, evaluation, or HTTP request
exists, and no API requests occurred after the terminal timestamp.

Under the additional assumptions below, practical futility concerns an
improvement of at least 0.15. It does not establish equality. The empirical rates
and difference are descriptive stopped-sample quantities, not unbiased population
estimates or confidence intervals.

## Actual resources and avoided work

| Quantity | Measured or bounded result |
|---|---:|
| Original / continued physical episode attempts | 28 / 43 |
| Total attempts / valid outcomes | 71 / 70 |
| Authorized episode retries / SDK retries | 1 / 0 |
| API requests started / responses with usage | 1,212 / 1,211 |
| Known original / continuation API charges | $1.76230815 / $3.15240375 |
| Total known token-priced API charges | **$4.91471190** |
| Unknown original request charge bound | $0–$0.318432 |
| Total charge interval under frozen bounds | **$4.91471190–$5.23314390** |
| Known input / output tokens | 12,570,744 / 496,247 |
| Known total tokens | **13,066,991** |
| Cached input, included in input above | 9,994,752 |
| Reasoning tokens, included in output above | 162,464 |
| Original controller session | 1,906.548869 s = 31 min 46.55 s |
| Continued controller session | 2,569.975486 s = 42 min 49.98 s |
| Sum of controller session wall times | **4,476.524355 s = 74 min 36.52 s** |
| Original planned episodes | 240 |
| Amended full-budget attempts, including one retry | 241 |
| Future planned episodes not executed after crossing | **170** |

All known original charges, including the invalid episode's $0.02712555, are
included. The unanswered original request's full $0.318432 reservation remains
outstanding; no zero charge was imputed. There were no new provider errors,
missing outcomes, or missing usage responses during the continuation. Exact total
cost and tokens remain unknown for that original request. Charges are calculated
from logged provider usage at frozen prices, not reconciled invoices or taxes.

The counts reconcile as `71 = 2*35 + 1`, including the original invalid attempt,
and `241 - 71 = 2*(120-35) = 170`. These are actual unscheduled future task
episodes after an online boundary, not episodes omitted because of infrastructure
failure. Relative to the amended full workload, 170 planned task executions
were avoided; the extra attempt is explicitly counted in both resource accounting
and the full-budget comparison.

The 70 valid episodes averaged $0.0698226621 in known charges. Applying that mean
to the 170 unexecuted episodes projects **$11.86985256** in additional workload,
and **$16.78456446–$17.10299646** for the hypothetical full amended workload when
combined with the observed cost interval. These are projections under comparable
future costs, **not realized dollar savings** or guaranteed full-budget costs.

Elapsed times above run from the first controller episode intent to the terminal
record within each session. The frozen exporter also records 4,469.645306 s from
first physical episode intents, excluding 6.879049 s of startup included in the
controller measure. These are different explicit time origins, not inconsistent
outcomes. The full calendar span was 19,956.525587 s (5 h 32 min 36.53 s), including
a 15,480.001232 s gap between sessions. Review/reporting time is not charged to
experimental execution time, while host sleep within the original session remains
part of its wall time.

## Execution timeline and safeguards

| UTC on 2026-10-16 | Event |
|---|---|
| 18:17:57 | Original protocol/code commit `6ffc20f` |
| 18:18:18 | Original realized task schedule commit `6c2719a` |
| 18:18:26.670970 | Original first controller intent |
| 18:50:13.219839 | Original infrastructure terminal after 13 pairs |
| 19:41:05–19:41:06 | Read-only resume review: original rule prohibited continuation |
| 23:07:18 | Amendment/code commit `390b635` |
| 23:07:32.863702 | Amendment manifest frozen after explicit retry authorization |
| 23:07:59 | Manifest/preflight commit `2d65d92` |
| 23:08:10.913049 | Detached process launched under `caffeinate -i` |
| 23:08:13.221071 | Authorized challenger retry controller intent |
| 2026-10-16T23:08:20.947657+00:00 | First new inference request |
| 23:51:03.196557 | First futility crossing, automatic terminal at pair 35 |

The new request allocation was $72.91925985: original $75 less $1.76230815 known
charges and the unchanged $0.318432 uncertainty reservation. Every new request
used the original conservative reservation and zero-retry transport. The
continuation ran in a detached session, with idle-sleep prevention verified.
The model-access and browser preflight used no generation requests. Before paid
continuation, **101 selected tests passed** (98 execution/statistical tests and
3 continuation-accounting tests). No frozen source or parameter was changed
after new outcomes began.

## Scientific interpretation and limitations

The full trajectory was not executed under an unchanged preregistration: the
single retry was authorized after outcomes and a partial failed-episode trace
were available. Reusing the earlier baseline while rerunning only its challenger
can alter the conditional paired-outcome distribution. Host sleep explains the
recovery request but does not establish that the missingness or retry selection
was noninformative. Nominal directional error guarantees require the efficacy
and futility conditional null assumptions to remain valid under this recovery
mechanism, application resets, and provider behavior. We therefore make no
unconditional confirmatory error-control claim for the amended experiment.
Public finite-bank exposure, fixed arm order, temporal separation, unverified
backend stability, and one unresolved request charge further limit inference.

The central operational finding is directly supported: the same evidence code
was updated during real evaluation, its first boundary stopped scheduling, and
170 remaining task executions did not occur. The inferential claim must retain
the recovery qualification. The amendment cannot retroactively turn the original
13-pair terminal run into an unchanged completed preregistered trial.

## Independent audit and reproducibility

Independent reviews checked exact evidence reproduction, first-crossing timing,
all original/continued physical attempts, cumulative costs, immutable original
records, task/model/prompt identities, and absence of work after stopping.
All 121 source-record hashes, 8 amendment-file hashes, and 229 original frozen
file hashes matched. The source protocol and `paper/main.tex` remain unchanged.

Primary artifacts are under `results/paper/live_stopping/continuation/`:

- `summary.json`, `operational_accounting.json`: statistical and full-attempt resource results.
- `pairs.json`, `evidence.json`, and corresponding CSV files: all 35 observed pairs.
- `episode_receipts.json`, `episode_intents.json`, `api_calls.jsonl`: both sessions,
  with session namespaces for request IDs and the original invalid attempt retained.
- `execution_timeline.json`, `session_launch.json`, `opening_balance.json`: execution provenance.
- `live_stopping_continuation.pdf`, `.svg`, `.png`: complete real-data trajectory.
- `independent_statistical_audit.*`, `independent_operational_audit.*`: independent checks.
- `paper_ready.md`, `results_table.tex`, `figure_caption.tex`: publication fragments.

The original halt artifacts are preserved in `history/2026-10-16_initial_halt/`;
the blocked resume review is preserved in `history/2026-10-16_blocked_resume_review/`
and `resume_review/2026-10-16/`. Root-level numerical files and figures now mirror the latest combined result;
the original numerical files and figures remain preserved in the history directories. The frozen amendment manifest is
`data/prospective_live_stopping_resume/amendment.json`, SHA-256
`016e10492d77035580426a7dacbf070c5dc0e7be5fab263f03958947ec133248`.

Reproduce the combined analysis and figure without inference:

```bash
.venv/bin/python experiments/growing_bandits/empirical/analyze_live_stopping_continuation.py \
  --manifest data/prospective_live_stopping_resume/amendment.json \
  --out /tmp/live-stopping-continuation-analysis
.venv/bin/python experiments/growing_bandits/empirical/plot_live_stopping_continuation.py \
  --results /tmp/live-stopping-continuation-analysis
```

## Paper-ready Results paragraph

The authorized amended continuation first crossed the practical-futility
boundary at 35 completed pairs (evidence 41.266; threshold 40); efficacy evidence
was 1.000. Baseline and challenger successes were 20/35 (57.14%) and 16/35
(45.71%), respectively, a descriptive difference of −11.43 percentage points.
The controller stopped online before pair 36, leaving 170 planned agent
executions unstarted. Across both sessions, 71 attempts produced 70 valid
outcomes and $4.91471 in known API charges, with one original unanswered request
bounded by an additional $0.318432. This demonstrates actual cessation of
inference at a statistical boundary on the amended trajectory; population-level
error control remains conditional on the recovery mechanism preserving the
required null assumptions.
