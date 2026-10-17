# Public replay 01: Terminal-Bench repeated trials

Registered 2026-10-12, before any trial outcome from this dataset was opened. Dataset listing, README, directory names, and `metadata.yaml` model identities were inspected; `result.json` contents were not. This is a retrospective analysis of a public leaderboard, not a prospective experiment.

## Frozen source and comparison

- Dataset: `harborframework/terminal-bench-2-leaderboard` at Hugging Face revision `572b2614be2c0cb2527e14f5b1e4026f1072e6c1` (Apache 2.0).
- Incumbent: `Terminus2__Claude-Opus-4.6`.
- Challengers, in this fixed order: `Capy__Claude-Opus-4.6`, `Crux__Claude-Opus-4.6`, `Droid__Claude-Opus-4.6`, `Terminus-KIRA__Claude-Opus-4.6`, `WozCode__Claude-Opus-4.6`.
- Every listed submission's metadata names `anthropic/claude-opus-4-6`. Exclude any job or trial whose actual configuration identifies a different model, a model mixture, nondefault task resources/timeouts, or a nonbinary reward. Report the exclusion counts. If a submission has fewer than two valid trials per task, omit that submission and report it; do not replace it after seeing scores.
- Collect only per-trial `result.json` files. If jobs are shards, combine them within a submission. Deduplicate identical trial IDs, then sort trials within `(submission, task)` by the full relative path. Use the first two valid trials. The primary analysis uses index 0; index 1 is an independent rerun check. A task is included only if the incumbent and every retained challenger have both trials. Do not select tasks by reward. If fewer than 40 tasks remain, stop and report a feasibility failure rather than modify the design.
- Use the binary verifier reward, `1` for a passed task and `0` otherwise. Do not use a mean or fractional reward. Treat missing/verifier-error trials as missing, never as a failure; report their counts by submission.

## R1: A/A replay under real rerun noise

Use the incumbent's index-0 and index-1 trials as pseudo arms. For each of 2,000 replicates, independently swap their labels within each task using NumPy `SeedSequence([20261012, 1, replicate])`, then randomly order the included tasks with the same generator. Run the incumbent-relative E1b efficacy test at `alpha=0.05` (one challenger) and record whether it ever crosses, its first crossing task, and the observed discordance. Report the empirical crossing fraction with a binomial 95% interval. Also report the two unswapped directions separately as drift diagnostics. The label-swap distribution is a conditional randomization check on these observed trials; it is not an estimate of error over future tasks or future sessions. Same-config A/A data are an alternative for the `delta=0.15` futility test, so its crossing rate is **not** a futility false-call rate.

## R3: Same-model scaffold search

Run the fixed challenger set against the Terminus 2 incumbent on index-0 trials. For each of 2,000 random task orders from `SeedSequence([20261012, 3, replicate])`, run the existing E1b uniform-mixture efficacy process at `alpha=0.05`. Report the fraction of orders that cross, first crossing task distribution, which challengers have positive full-task paired gains, and the full-grid episode count `(K+1)N` versus stopping at crossing. Noncrossing orders consume all `N` tasks. The primary gain estimate and paired task bootstrap interval (10,000 task resamples, seed `20261013`) are for every challenger, including nonwinners.

If index-0 efficacy crosses, select the challenger with the largest individual log capital at the first crossing in the lexicographically first task order. Test that single selected challenger against the incumbent on index-1 trials using E1b at `alpha=0.05` over the same task order, and report its paired gain and crossing time. Index-1 is a rerun check of the selected comparison, not a new search. If index-0 never crosses, report that and do not select a winner. Report futility/drop-the-loser at `delta=0.15`, `alpha=0.05` as a secondary budget analysis only; do not call its crossing under these data a type-I-error check.

## Interpretation limits

Random reordering of a fixed benchmark yields a distribution of *replay stopping times*, not independent experiments. The anytime-valid population interpretation additionally needs task draws that satisfy the test's conditional null and stable agent configurations. We will state this assumption and the nonconcurrent leaderboard runs explicitly. A certified result on the benchmark does not establish a causal scaffold effect, because submissions can differ in implementation, dates, and other settings. If no efficacy crossing occurs, the result is a power limitation or a negative finding, not proof that no scaffold helps.

## Feasibility contingencies

Inspect a small number of `result.json` files only after this file is committed. If schema or directory structure differs from the documented layout, adapt the parser without changing the analysis choices. Document every mapping. If a job-level result contains a complete task/trial list, it may be used as an index, but per-trial results remain the outcome source. No candidate, task, replicate, alpha, delta, or seed may be chosen using outcomes.

## Amendment 1: model identifiers (2026-10-12, before aggregate analysis)

The initial per-trial schema audit found that metadata model labels can differ from the actual `config.agent.model_name`. All Crux trials checked identify `claude-opus-4-5`; Droid identifies `aurora-01-21`; WozCode identifies only `opus`. These three submissions are excluded by the original model-identity rule. Terminus-KIRA identifies `vertex_ai/claude-opus-4-6`, an explicit Opus 4.6 version string routed through Vertex AI. It is retained as the same model version, with the provider-route difference stated as a limitation. Capy and Terminus 2 identify `anthropic/claude-opus-4-6`. No aggregate task rewards or arm gains were inspected before this amendment. The same fixed task-intersection, trial, seed, and test rules apply to the retained arms.
