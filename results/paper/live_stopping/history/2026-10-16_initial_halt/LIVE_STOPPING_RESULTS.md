# Prospective live-stopping experiment: unresolved after an infrastructure halt

**Executed 2026-10-16.** The controller updated evidence during real evaluation and
automatically halted after an unacknowledged API request in the challenger half of
pair 14. Thirteen valid pairs completed. Neither statistical boundary crossed.
This is an **operationally incomplete, unresolved experiment**, not a completed
120-pair unresolved trial and not a demonstration of statistical early-stopping
savings. No replacement task, episode retry, or second experiment was run.

## Registration and execution

The [protocol](LIVE_STOPPING_PROTOCOL.md), controller, sampler, runner, analysis,
and tests were committed at `6ffc20fdd8315fdf20286837094f94e9cfe993a3` at
18:17:57 UTC. The seed `16907f20ad5c78a18cca90759d59838a` was drawn after that
commit; the manifest is timestamped 18:18:04.955556 UTC. The realized sample and
preflight records were committed at `6c2719a` at 18:18:18 UTC. Controller
registration occurred at 18:18:26.646048 UTC, the first episode intent at
18:18:26.670970 UTC, and the first API request at 18:18:34.680545 UTC.

All 120 draws were uniform with replacement from the complete 120-task public
Linear account-settings bank. There were 82 distinct source tasks; repetitions
were retained, with a maximum multiplicity of five. The sample included 17 easy,
22 medium, and 81 hard tasks. No outcome-based filtering occurred. The baseline
and explorer prompt hashes match the prior fixed pair. Both used the frozen
`gpt-5.4-mini-2026-03-17` snapshot, low reasoning effort, 30 steps, 180-second
agent limit, and the same browser environment. Baseline always ran first.

The full correctness gate passed **85 tests** before paid evaluation, including
exact online/offline evidence parity, all registered synthetic outcome regimes,
adjacent boundaries, interruption/resume, uncertain external completion,
provider errors, cost exhaustion, and prevention of calls after stopping.
Nonbillable model access and browser/app checks also passed. No pilot inference
was used to select the experiment.

## Statistical observations

Only the contiguous 13 complete pairs enter the evidence processes. Pair 14's
baseline was retained as an unpaired valid outcome; the challenger was invalid
and was not scored zero despite its unsatisfied environment verifier.

| Quantity | Observed value |
|---|---:|
| Final status | Unresolved; infrastructure halt |
| Complete pairs / maximum | 13 / 120 |
| Baseline success | 8/13 = 61.5385% |
| Challenger success | 7/13 = 53.8462% |
| Paired challenger-minus-baseline difference | −1/13 = −7.6923 percentage points |
| Challenger-only wins / baseline-only wins / ties | 1 / 2 / 10 |
| Efficacy evidence at last complete pair | 1.0000000000 |
| Futility evidence at last complete pair | 2.1360441229 |
| Natural log efficacy / futility evidence | 0 / 0.7589555777 |
| Registered threshold, each direction | 40; log threshold 3.6888794541 |
| First efficacy crossing | None |
| First futility crossing | None |
| Meaningful improvement threshold | 0.15 |
| Directional error budgets | 0.025 efficacy; 0.025 futility |

Saved episode receipts exactly reproduce the paired table and both evidence
trajectories. Efficacy remained 1 throughout the observed prefix; futility's
maximum was its final value, 2.1360441229. These data support neither efficacy
nor practical futility. The stopped empirical rates are descriptive, not
unbiased estimates or confidence limits for the task population.

## Operational outcome and resource accounting

| Quantity | Observed value |
|---|---:|
| Actual agent episode attempts | 28 |
| Valid outcomes | 27: 26 in complete pairs, one unpaired baseline |
| Invalid outcomes | 1 challenger episode |
| Physical API requests started / usage receipts returned | 458 / 457 |
| Known API usage cost | $1.76230815 |
| Outstanding maximum charge for the unacknowledged request | $0.318432 |
| Total charge interval under frozen pricing/token bounds | $1.76230815–$2.08074015 |
| Known input tokens | 4,625,255 |
| Known output tokens, including reasoning | 173,209 |
| Known total tokens | 4,798,464 |
| Cached input tokens, already included above | 3,683,072 |
| Wall time, first episode intent to terminal decision | 1,906.548869 seconds = 31 min 46.55 s |
| Maximum planned agent executions | 240 |
| Planned executions not started because of infrastructure halt | 212 |
| Executions avoided by a statistical boundary | **0 demonstrated** |

Known costs include all returned usage, including six successful requests within
the failed episode. The last request's cost and tokens are unknown. Its full
reservation remains outstanding; no zero charge is imputed. These dollar figures
are computed from provider usage and frozen standard prices, not reconciled
invoice totals. An exact total API cost or token count cannot be reported.

The 27 valid episodes with complete accounting cost $1.73518260, a mean of
$0.0642660222 per episode. Applying that mean to the 212 unstarted episodes
projects **$13.62439671 of hypothetical additional workload**, assuming comparable
future costs. This is not realized dollar savings and is not attributable to a
statistical decision. The registered formula `2*(120-tau)` does not apply here:
there was no statistical stopping time `tau`, and both episodes of incomplete
pair 14 had already been attempted. Simply substituting 13 would incorrectly
claim 214 unexecuted episodes.

## What stopped the run

Request `request_000458` began at 18:48:37.153498 UTC during
`live_014` / source `task_h40` / challenger. It ended with a transport
`ReadTimeout` at 18:50:11.592023 UTC and no usage response. The invocation guard
latched the error, so subsequent internal agent attempts were rejected locally
without additional HTTP requests. The controller saved a null outcome and
terminal `infrastructure_error` at 18:50:13.219839 UTC. No pair 15 was scheduled,
and no API request was sent after the timeout.

Read-only power-log inspection identified **host sleep during this episode**:
clamshell sleep at 18:48:19 UTC, maintenance sleeps at 18:48:43 and 18:48:56,
intermediate dark wakes, and user wake at 18:49:25. The last request overlapped
these events. This is evidence of a local execution interruption; it does not
establish that the provider itself was unavailable or resolve the missing charge.
The final episode's UTC interval and monotonic duration also differ, so UTC
timestamps define the reported overall wall time. Sanitized sleep/wake events
are saved in `environment_events.json`. No power settings were changed afterward
to resume the registered experiment.

## Interpretation, assumptions, and departures

The central research question remains **unresolved by this experiment**. Real
episodes were evaluated sequentially, evidence was updated online, and the
registered operational stop prevented additional scheduling. The ability to halt
at statistical boundaries is established by synthetic correctness tests; a real
statistical crossing with avoided inference was not observed.

Sampling with replacement targets the finite public Linear task distribution.
Under the corresponding conditional discordant-win null for efficacy and
conditional mean-difference null for futility, with stable agent/application/reset
behavior, the separate directional crossing probabilities are bounded by .025;
their union is bounded by .05. This run makes no rejection claim. Public task
exposure, historical feasibility-based design choice, fixed arm execution order,
and the narrow application limit generalization. A snapshot identifier and
matching source hashes do not prove backend stability or independent execution
noise. All returned model identifiers match the registered snapshot; no system
fingerprint was provided. Host suspension means uninterrupted execution conditions
were not maintained through the final incomplete pair.

There were **no changes** to candidates, task distribution, seed/order, margin,
error budgets, evidence implementations, or execution settings after outcomes.
The unplanned sleep/timeout caused the preregistered infrastructure stop. The
experiment did not reach its planned statistical or maximum-pair endpoint, and
provider accounting remains incomplete for one request. These are limitations,
not grounds to reinterpret missingness or continue with a replacement. A
presentation-only zoom figure was added after execution; it uses the saved real
evidence without changing statistical calculations. The original frozen
exporter's full-budget-axis figure is also retained.

## Audit and reproducibility artifacts

- `data/prospective_live_stopping/`: immutable prompts, source bank, sampled tasks,
  configuration, seed, hashes, package versions, and registration manifest.
- `results/paper/live_stopping/summary.json`: frozen export's numerical findings.
- `results/paper/live_stopping/{pairs,evidence,episodes}.csv`: tabular outcomes,
  evidence, and episode-level costs; corresponding JSON retains full metadata.
- `results/paper/live_stopping/api_calls.jsonl`: every physical request intent,
  returned usage, unknown completion, and invocation-error event.
- `results/paper/live_stopping/operational_accounting.json`: incomplete-charge
  interval and explicitly hypothetical cost projection.
- `results/paper/live_stopping/independent_audit.{json,md}`: independent final
  checks, including preregistration timing and lack of calls after the halt.
- `results/paper/live_stopping/live_stopping_observed_prefix.{pdf,svg,png}`:
  publication figure of the 13 observed pairs with the operational halt labeled.
- `results/paper/live_stopping/live_stopping_evidence.{pdf,svg,png}`: original
  frozen export showing the full planned 120-pair axis.
- `results/paper/live_stopping/{paper_ready.md,results_table.tex,figure_caption.tex}`:
  manuscript-ready text and LaTeX fragments. **`paper/main.tex` was not edited.**
- Raw browser traces remain in `logs/prospective_live_stopping/`; all statistics
  and cost receipts needed for the reported findings are exported above.

Reproduce the numerical audit and original figure in the frozen environment:

```bash
.venv/bin/python experiments/growing_bandits/empirical/analyze_live_stopping.py \
  --log-dir logs/prospective_live_stopping --out results/paper/live_stopping
.venv/bin/python experiments/growing_bandits/empirical/plot_live_stopping.py \
  --results results/paper/live_stopping
```

The strict runtime audit uses absolute paths from this registration. For an
independent installation, the exported paired binary outcomes and original
`e1b_log_paths` / `futility_log_paths` functions suffice to reproduce the numerical
trajectories; source/path provenance must be mapped separately. The result
directory's `sha256.json` records final artifact digests.

## Paper-ready methods paragraph

We preregistered one prospective comparison of fixed baseline and explorer prompts
using a browser agent with OpenAI `gpt-5.4-mini-2026-03-17`. We drew 120 tasks
independently with replacement from the complete 120-task public Linear
account-settings bank, retaining duplicates and resetting application state before
each episode. Baseline and challenger ran sequentially on identical tasks. After
each valid pair, a durable controller updated the existing discordant-outcome
efficacy process and paired-difference futility mixture, with meaningful gain
0.15, directional error budgets 0.025, and evidence thresholds 40. The controller
was configured to stop new scheduling at the first statistical boundary, the task or $75 budget,
or an invalid episode. Protocol and task commits preceded all study inference;
85 automated tests passed before execution.

## Paper-ready results paragraph

The experiment halted automatically after a request timeout in pair 14, leaving
13 complete pairs. Baseline and challenger successes were 8/13 and 7/13,
respectively, a descriptive difference of −7.69 percentage points. Efficacy and
futility evidence were 1.00 and 2.136, below the threshold of 40; neither boundary
crossed. The run attempted 28 episodes and recorded $1.76231 in known API charges,
with one unacknowledged request bounded by an additional $0.318432 under the
registered pricing assumptions. Host sleep overlapped that request. The remaining
212 planned episodes were not started because of the infrastructure halt. This
operationally incomplete run verifies live evidence updates and automatic failure
handling, but demonstrates no inference savings attributable to statistical
early stopping.
