# Ready-to-run Gmail first-run pilot

*Prepared 2026-10-07. No new agent episode or billable API call has been started.*

## Exact batch

- **30 task IDs:** the 10 easy, 10 medium, and 10 hard Gmail tasks omitted from the primary G/F grid. Some of these task IDs had archived pre-amendment pilot runs with other prompts. Neither selected prompt has a recorded run on any of the 30 IDs, so the planned **prompt–task cells** are new; this is not an untouched task bank or new task-template family.
- **Frozen pair:** G_14 as the first agent and same-agent rerun; G_16 as the different-agent alternative. Both tied at 70% on the old 30-task G grid and differ in prompt vector. They were chosen from old data before this batch's outcomes.
- **90 episodes:** one G_14 main, one independent G_14 rerun, and one independent G_16 run per task, each after a sandbox reset. The single-worker queue places G_14 main before its two alternatives, whose order is randomized per task. The alternatives do not read the first agent's trace, so this tests independent recovery rather than communication.
- **Frozen policies:** stop after G_14; retry G_14 after observable non-completion; switch to G_16 after observable non-completion; and an exploratory old-data ridge rule that also retries some apparently complete first runs. The rule's coefficients and 80th-percentile threshold are saved in `data/first_run_pilot_gmail/uplift_rule.json`. Hidden success labels are used only after a policy decision.

The primary comparison is the frozen ridge policy against timeout-only same-agent retry, reporting verified selected-trajectory success and cost per task. Timeout different versus timeout same is secondary. Oracle `at least one succeeds` coverage is secondary and cannot be called a realized policy outcome. Every contrast is paired by task. With 30 tasks, this is a **pilot**, not a decisive test of a 2–3 percentage-point difference.

## Cost and launch boundary

On the old 30 tasks, G_14 cost $0.10184 per episode on average and G_16 cost $0.08427. Two G_14 episodes plus one G_16 imply **$8.64 for 30 triples** at those historical means. The proposed collector cap is **$12 cumulative**, using one worker to limit overshoot to the episode in flight. The cap may stop the batch before all 90 episodes if tasks cost more than expected; the analyzer refuses to present a complete-trial result until all required cells have terminal, scored records. These are historical usage-based estimates, not a current-price guarantee.

The OpenAI key is configured in the repository's local `.env` (its value was not read or printed). The existing collector validates the frozen queue and prompt hashes, isolates output in `logs/first_run_pilot_gmail/`, and can resume from its terminal records. It must **not** be launched through `scripts/run_empirical_pool.sh`, which points at the earlier study's log directory and has no custom data-path option.

Prepared input: `data/first_run_pilot_gmail/` (manifest, 90-item queue, frozen prompt files, rule). The preparation script is `experiments/growing_bandits/retrospective/prepare_first_run_pilot.py`; the analysis script is `experiments/growing_bandits/retrospective/analyze_first_run_pilot.py`. The collector's queue and prompt checks passed, and the analyzer passed a synthetic 90-cell dry run. Neither check invokes the agent model.

**The billable launch command, after a spending limit is authorized, is:**

```bash
./.venv/bin/python experiments/growing_bandits/empirical/collect.py \
  --data data/first_run_pilot_gmail \
  --log-dir logs/first_run_pilot_gmail \
  --budget 12 --workers 1
```

Then, after all 90 scored episodes finish:

```bash
./.venv/bin/python experiments/growing_bandits/retrospective/analyze_first_run_pilot.py
```

## What this pilot can settle

It can check whether the old timeout signal and the frozen exploratory benefit rule behave similarly on new prompt–task cells, whether switching to G_16 beats a G_14 rerun for this selected pair, and whether the collector/cost assumptions hold. It cannot establish a general multi-agent decision rule, especially for small gains, different task templates, or interacting agents. If the pilot's gain is small or unstable, a larger independent task set is needed before making that claim.
