# Research Process: Uniform Sweep Then Adaptive Sampling

## Goal

Evaluate whether adaptive prompt-arm allocation can identify strong cold-start
prompts faster and more cheaply than a full uniform sweep, while preserving
anytime-valid evidence through e-processes.

The experiment has two phases:

1. Full paired uniform benchmark: every prompt arm sees the same 60 Gmail tasks.
2. Adaptive online run: the allocation policy uses observed rewards/e-process
   state to concentrate pulls on promising arms.

## Current Cost Baseline

The surviving historical 60-task WebArena Gmail run used `claude-sonnet-4-6`
through browser-use:

- Tasks: 60
- Total cost: $11.41
- Mean cost per task: $0.190
- Total wall time: 4,419 s = 73.7 min
- Mean wall time per task: 73.7 s
- Total input tokens: 3,168,536
- Total output tokens: 172,146
- Total cache-read tokens: 3,031,893

Cost projections:

| Run | Tasks | Estimated cost | Sequential wall time |
|---|---:|---:|---:|
| One 60-task run total | 60 | ~$11-$13 | ~1.2 h |
| Full paired uniform sweep, 12 arms x 60 tasks | 720 | ~$137-$152 | ~14.7 h |
| One adaptive 60-task run | 60 | ~$11-$13 | ~1.2 h |
| Full paired sweep + adaptive run | 780 | ~$148-$165 | ~16 h |

For the next paired sweep, the infrastructure is configured to use
`gpt-5.4-mini` through browser-use with `llm_reasoning_effort: low`. This
should be much cheaper than the historical Claude run, but a 4-task smoke test
should be run first to verify model access, structured output compatibility,
and token accounting.

## Phase 1: Full Paired Uniform Sweep

Run each of the 12 prompt arms on the same 60 Gmail tasks. This produces the
clean comparison table for prompt quality because each arm faces the same task
set.

Design:

- Arms: all 12 prompt vectors in `configs/arms_initial.yaml`
- Tasks per arm: 60
- Total task executions: 720
- Allocation: deterministic paired/block design, not adaptive
- Model: `gpt-5.4-mini`
- Reasoning effort: `low`
- Reward: binary verifier success

Primary outputs:

- Per-arm success rate over 60 tasks
- Per-arm success rate by difficulty
- Per-arm token cost
- Per-arm wall time
- Per-arm mean steps
- Per-arm e-process trajectory against the chosen baseline/null

This phase answers: which prompt arm is best on the benchmark when task
assignment is fair?

## Current Uniform Sweep Results

As of 2026-06-18, the first four paired arms have completed all 60 Gmail
tasks. The remaining eight arms are still pending.

| rank | arm | success | success rate | cost | wall time | log-e | CS |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | baseline | 35/60 | 58.3% | $2.6262 | 58.6 min | -0.582 | 0.400-0.800 |
| 2 | cautious | 33/60 | 55.0% | $2.6031 | 74.2 min | -1.626 | 0.369-0.769 |
| 3 | explorer | 33/60 | 55.0% | $2.9363 | 71.8 min | -1.117 | 0.369-0.754 |
| 4 | planner | 27/60 | 45.0% | $2.6365 | 96.9 min | -1.311 | 0.292-0.646 |

Difficulty breakdown:

| arm | easy | medium | hard |
|---|---:|---:|---:|
| baseline | 15/20 | 13/20 | 7/20 |
| cautious | 15/20 | 13/20 | 5/20 |
| explorer | 16/20 | 13/20 | 4/20 |
| planner | 15/20 | 8/20 | 4/20 |

Paired comparison against baseline:

| arm | wins vs baseline | losses vs baseline | ties | net success delta |
|---|---:|---:|---:|---:|
| planner | 3 | 11 | 46 | -8 |
| cautious | 3 | 5 | 52 | -2 |
| explorer | 5 | 7 | 48 | -2 |

Current interpretation:

- Baseline is the best completed arm so far by success rate and wall time.
- Cautious and explorer are close on total success, but both trail baseline by
  2 paired successes.
- Planner is clearly worse on this benchmark: lower success, more steps, and
  substantially longer wall time.
- All completed arms are cheap under `gpt-5.4-mini` low effort: roughly
  $2.60-$2.94 per 60-task arm.

## Phase 2: Adaptive Sampling Run

Run a 60-task online adaptive allocation policy, such as UCB or SPRUCE. At each
step, the policy chooses which prompt arm to use based on prior observations.

Design:

- Total task executions: 60
- Allocation: adaptive policy
- Model: `gpt-5.4-mini`
- Reasoning effort: `low`
- Reward: binary verifier success
- Inference: upward-capital per-arm e-process and global linear-mixture null

Primary outputs:

- Arm selected at each time step
- Per-step reward
- Per-step and cumulative token cost
- Per-step and cumulative wall time
- Per-arm and global log-e trajectories
- First time the adaptive run identifies or heavily concentrates on the best
  arm from Phase 1
- First global rejection time, if any

This phase answers: can adaptive allocation find the strong prompt(s) faster
and with less token/time expenditure than evaluating every arm uniformly?

## Metrics To Track In Every Run

Every JSONL record should preserve:

- `t`
- `task_id`
- `task_meta.difficulty`
- `arm_id`
- `success`
- `reward`
- `steps`
- `wallclock_s`
- `tokens.input`
- `tokens.output`
- `tokens.cache_read`
- `tokens.cache_write`
- `tokens.cost_usd`
- `tokens.invocations`
- `policy.type`
- `policy.scores`
- `per_arm_state`
- `global_e.log_e`
- `global_e.rejected`

Derived run-level metrics:

- Cumulative cost by time step
- Cumulative wall time by time step
- Cumulative tokens by time step
- Cost per success
- Wall time per success
- Time-to-best-arm concentration
- Time-to-global-rejection, if reached
- Regret relative to the best arm from the paired sweep

## Metrics By Phase

For the 60 x 12 paired uniform sweep, collect metrics that answer: which prompt
arm is best when every arm sees the same task set?

- Success rate per arm
- Success rate by difficulty: easy, medium, hard
- Per-task success table showing which arms solved which exact tasks
- Mean steps per arm
- Mean wall-clock time per arm
- Total wall-clock time per arm
- Total token use per arm: input, output, cache-read, and cache-write tokens
- Total API cost per arm
- Cost per success per arm
- Time per success per arm
- Per-arm e-process trajectories
- Global e-process trajectory
- First rejection time, if any
- Confidence sequence bounds per arm

For the adaptive run, collect the same raw execution metrics plus online
allocation metrics that answer: did adaptivity find the good prompt faster or
cheaper?

- Arm chosen at each step
- Policy score for each arm at each step
- Pull count per arm over time
- Cumulative success rate over time
- Cumulative tokens over time
- Cumulative cost over time
- Cumulative wall-clock time over time
- Task index where the policy starts concentrating on the best arm from the
  paired sweep
- Number of pulls spent on weak arms
- Regret relative to the best arm from the paired sweep
- Time-to-global-rejection, if reached
- Tokens-to-global-rejection, if reached
- Cost-to-global-rejection, if reached

The headline efficiency comparison is:

| Question | Uniform paired sweep | Adaptive run |
|---|---|---|
| Prompt quality | Best arm after exhaustive paired evaluation | Arm identified by online allocation |
| Task cost | 720 task executions | 60 task executions |
| Token cost | Total tokens for all arms on all tasks | Tokens used before concentration/rejection |
| Dollar cost | Total API spend for exhaustive sweep | Spend before concentration/rejection |
| Wall time | Total time for exhaustive sweep | Time before concentration/rejection |
| Waste | All arms fully evaluated, including weak arms | Pulls spent on weak arms before learning |

## Interpretation

The paired uniform sweep is the benchmark-quality comparison. It is expensive
but clean: every arm gets the same task set.

The adaptive run is the efficiency demonstration. It should not be treated as
an unbiased final ranking of every arm, because it intentionally samples arms
unequally. Its value is showing whether e-process-guided allocation reaches
useful evidence earlier, spends fewer tokens, or avoids wasting time on weak
arms.

A strong paper claim would be:

> The full paired sweep identified the best cold-start prompt after 720 task
> executions. The adaptive e-process run concentrated on the same prompt within
> a 60-task budget, using substantially fewer tokens and less wall-clock time
> than exhaustive uniform evaluation.

## Practical Recommendation

Before the full 720-task sweep, run a smaller paired pilot:

- 12 arms x 15 tasks = 180 tasks
- Estimated cost: ~$34-$38

Use the pilot to validate:

- task ordering and pairing,
- CSV/report generation,
- token and wall-time accounting,
- adaptive comparison plots,
- resume behavior after browser-use failures.

Then run the full paired sweep once the pipeline is stable.
