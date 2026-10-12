# Prospective Linear fixed-pair trial — frozen before collection

Registered 2026-10-12. This protocol and `data/prospective_linear/` are committed before any Linear trial outcome is collected. The candidate pair was chosen from `configs/arms_initial.yaml` without viewing Linear outcomes. The task bank is public, so possible model training exposure is unknown; the claim concerns this previously unused local cohort and this fixed pair, not novel task creation.

## Design

- Population: all 120 tasks in the sibling WebArena Infinity `apps/linear-account-settings/real-tasks.json`; no filtering by instruction, difficulty, or observed result. The bank SHA-256 and exact IDs/order are in `data/prospective_linear/manifest.json`. WebArena Infinity revision at registration: `b3015d794b6ed8bc6fb48b95626e96d28c483904` (its `uv.lock` has a local modification; application and grader files are clean).
- Incumbent: `linear_baseline`, vector `(planning=0, verification=0, agency=0, expertise=1, format=0, goal=0)`. Challenger: `linear_explorer`, vector `(1,0,2,2,0,0)`. Both use `configs/template.jinja`; rendered prompt hashes are frozen in the manifest and pool. These are the existing catalog's baseline and explorer choices, selected before Linear outcomes.
- Agent: existing WebArena browser-use adapter and collector; OpenAI `gpt-5.4-mini`, low reasoning effort, no vision, headless, maximum 30 steps, 180-second episode timeout. Both arms share settings and task grader. Model name is an API alias, so an unannounced provider-side revision is possible.
- Outcome: the app's existing task verifier binary success, one episode per arm per task. Partial credit is zero. A genuine agent timeout is failure. The collector retries harness/provider errors up to three attempts, then records `missing` rather than failure; provider outage stops collection. All attempts and costs remain in logs.
- Order: NumPy `default_rng(20261012)` permutes sorted task IDs, then independently permutes the two arms inside each task. The resulting 240-item queue is frozen in `data/prospective_linear/linear/queue.jsonl`. Four workers share the queue in fixed round-robin partitions. Each arm starts from a reset app state.
- Maximum: 120 paired tasks, 240 episodes. Historical working estimate $0.08–$0.10 per episode ($19.20–$24.00 full grid); hard collector spend cap $35, with only in-flight episode overshoot possible. Four workers, idealized wall time at the 180-second cap 10,800 seconds (3 hours) plus setup/recovery.

## Decision

- Challenger minus incumbent paired success difference `D_t ∈ {-1,0,1}`. Meaningful gain `delta=0.15`.
- Efficacy tests `H0: E[D_t] <= 0` with repository E1b, one challenger, `alpha_eff=0.025` (threshold 40). Futility tests `H0: E[D_t] >= 0.15` with the repository downward paired-difference mixture, `alpha_fut=0.025` (threshold 40). Report first crossing on either path, maxima, final e-values, number of completed pairs, paired rates, discordant counts, cost, time and all exclusions. If both cross, report both without selecting a favorable label. If neither crosses, report unresolved.
- Analysis uses the frozen task order and only a contiguous prefix of fully observed pairs. No result from a later task is moved ahead of a missing/incomplete pair. If any attempted episode becomes terminal `missing`, the primary inferential result is unresolved; complete-case paths may be shown descriptively. A budget or outage stop before all 120 pairs is reported as incomplete, with no positive scientific claim. Pilot outcomes, if any, belong to this one path; arms, tasks, margin, and tests will not change.
- Interpretation assumes task-level conditional nulls and sufficiently stable operation across the collection session. Public task reuse and possible training exposure limit external generalization. Concurrent arms control session drift within tasks imperfectly because workers can finish at different times.

## Execution and provenance

Prepare with `./.venv/bin/python experiments/growing_bandits/empirical/prepare_linear_trial.py`; it refuses changed frozen files. Run `./.venv/bin/python experiments/growing_bandits/empirical/collect.py --profile linear --data data/prospective_linear --log-dir logs/prospective_linear --budget 35 --workers 4`. Before starting paid collection, verify the committed protocol and manifest hashes. Preserve raw worker records and the final task-level table. No outcome-based revision of this protocol is allowed.
