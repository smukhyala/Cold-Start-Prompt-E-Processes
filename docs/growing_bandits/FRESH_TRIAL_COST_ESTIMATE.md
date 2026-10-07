# Cost and sample-size estimate for fresh agent trials

*Planning estimate, 2026-10-06. No API calls or new episodes were run. The calculations use actual costs and outcome discordance from the frozen G/F/GLK30 trials. They are not a spending authorization or a guaranteed price quote.*

## Minimal fresh-task design priced here

Freeze an anchor prompt `a` and a different prompt `b` before seeing outcomes on each **new** task. Reset the environment between runs. Collect three independent episodes on that task: first `a`, rerun `a`, and `b`. Score the paired oracle-potential coverage difference

`D_t = max(Y_a, Y_b) - max(Y_a, Y_a_repeat)`.

This design uses three model episodes per task and measures the *incremental coverage of a distinct independent agent*. It does not test communication or a deployed verifier. Randomize run order and balance task difficulty. Each distinct task, rather than each agent episode, is the unit of generalization.

The candidate pools, model, scaffold, task scoring, time limit, and prompt-selection rule must remain frozen. A different selected pair could have a different effect and variance from the **random different prompt** contrast used for the estimates below.

## Historical unit cost and variance

The live logs record GPT-5.4 Mini costs. Its current official token rates are [listed by OpenAI](https://developers.openai.com/api/docs/models/gpt-5.4-mini): $0.75 input, $0.075 cached input, and $4.50 output per million tokens. Historical observed totals are more useful for this web agent because step count, screenshots, caching, and timeouts make token use variable.

For each existing randomly repeated cell, the third-run outcome and cost are averaged over all other prompts on the same task. The observed standard deviation treats a randomly chosen second prompt as one future run. Approximate sample size uses a two-sided 5% test with 80% power: `n ≈ (1.96 + 0.84)^2 × SD(D)^2 / target_effect^2`. The estimates assume independent fresh tasks, one anchor-pair triple per task, and similar cost and outcome variability to the old task bank. They do not include task creation or personnel time.

| Pool | Mean observed episode cost | Mean three-run cost/task | P90 observed three-run cost/task | Estimated SD(D) |
|---|---:|---:|---:|---:|
| Gmail G | $0.099 | $0.267 | $0.701 | 0.228 |
| Gmail F | $0.098 | $0.287 | $0.662 | 0.275 |
| GitLab GLK30 | $0.135 as collected | $0.407 as collected | $0.997 | 0.354 |

The GLK30 score is a 30-step rescore of longer collected runs; the recorded $0.407 triple cost is therefore **not** the price of enforcing a true 30-step stop. It is the defensible as-collected planning anchor. A genuine cap may cost less, but needs a small pilot to establish that. The p90 row is the cost of an unusually expensive *individual* triple; multiplying it by the number of tasks would be an excessive budget forecast.

## What different budgets buy

| Target for a two-attempt coverage advantage | G tasks / mean cost | F tasks / mean cost | GLK30 tasks / mean cost |
|---|---:|---:|---:|
| Detect **5 percentage points** with ~80% power | 163 / **$43** | 238 / **$68** | 393 / **$160** |
| Detect **3 points** | 452 / **$121** | 660 / **$190** | 1,090 / **$444** |
| Detect **2 points** | 1,016 / **$271** | 1,484 / **$426** | 2,453 / **$998** |

These mean-cost figures are only inference charges. A prospective spending cap should allow for differences in task mix and longer trajectories; a **50% reserve** puts the 5-point studies at approximately $65 (G), $100 (F), and $240 (GLK), before task collection and setup. The power figures are rough because the standard deviations were learned from only 29–30 task clusters and the true effect may differ from the target. A stricter multiplicity correction, multiple domains, or a smaller effect requires more tasks.

The bank presently has **30 unused Gmail tasks** and **110 unused GitLab tasks** relative to the primary G/F and GLK30 grids. One three-run pass would cost roughly $8 for G, $9 for F, or $45 for GLK at historical means. If all unused tasks behave similarly, the smallest effect these bank sizes would have about 80% power to detect is roughly **12 points (G), 14 points (F), or 9 points (GLK)**. Running both Gmail prompt pools on the same 30 tasks would cost about $17, but would still supply only 30 new task clusters. To resolve a 1–2 point gain, the limiting input is hundreds or thousands of genuinely new tasks, not merely model charges.

## Costs for the other unconfirmed claims

| Claim to strengthen | Smallest defensible next step | Approximate API cost | What it would establish |
|---|---|---:|---|
| Generalization beyond the inspected 30-task banks | Run the frozen three-run contrast on the unused 30 Gmail or 110 GitLab tasks | ~$8–9 per Gmail pool; ~$45 GitLab | Independent task-set check for large effects; underpowered for the observed 1–2 points |
| Usable result selection | First audit whether saved final outputs support a blinded human or deterministic verifier; test it offline on existing trials | **$0 model cost**, plus human time if used | Whether an operational selector can recover some oracle potential coverage on these tasks |
| Actual agent communication | On new tasks, add one second-agent run conditioned on the first agent's output to the three-run design, while matching total token/step budget | At least ~$0.10 per Gmail task or ~$0.135 per GitLab task **extra**, likely more for longer context; a 30-task feasibility pilot is roughly $12–20 for all four runs | Feasibility and a preliminary interaction contrast; 30 tasks do not settle small effects |
| A predictor of which candidate pools reward breadth | Create genuinely independent prompt pools and task sets; a minimal 16 prompts × 24 tasks = 384 episodes/pool, plus repeats | ~$38–52 per pool, before repeats; 10 pools ~$380–520, and ten may still be too few | Pool-level association only if candidate construction and task regimes vary by design; not a cheap confirmatory validation |

The communication estimate assumes a fourth run costs like prior episodes. It excludes any paid verifier call, context inflation, or new environment/tool charges. A 20–50% contingency is sensible for a *pilot*; a final power study needs its own observed cost and variance estimates.

## Recommendation under a no-spend preference

The existing data already support a scoped thesis on independent-agent portfolio choice. Use a public external outcome matrix if a compatible one is accessible, and audit a verifier against saved outputs without model calls. If fresh evidence is essential, use the unused task bank as a small feasibility/generalization pilot with a hard spending cap. Its value is checking design and catching a large effect; it cannot be sold as a decisive test of the observed small gain. Do not buy thousands of episodes to chase a 1–2 point effect unless that claim is necessary to the thesis.
