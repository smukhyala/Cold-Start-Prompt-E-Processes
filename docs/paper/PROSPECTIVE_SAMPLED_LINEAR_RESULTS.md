# Prospectively sampled Linear profile-edit trial result

The [task distribution, prompt pair, and decision rule](PROSPECTIVE_SAMPLED_LINEAR_PROTOCOL.md) were committed at `f4d91f51288521a7e434ff10d4e29c42399568d7` before the task seed was drawn. The seed `fa2b3af816156387ac1feb8a576a848e`, all 120 realized tasks, verifiers, and queue were committed at `b22825d52d0e89042d364222266a769927b74324` before the first agent outcome. The generated task bank SHA-256 is `d496b6e80cdf095a94e816e9d3d549ecd4b8c0fddc0f5f05c08784a9d440302e`. The realized family counts were 34 name+username, 29 username+email, 33 name+email, and 24 all-three tasks; draws were retained in their original order.

## Registered result

All 120 pairs completed. The baseline incumbent succeeded on **3/120 (2.50%)** and the explorer challenger on **1/120 (0.83%)**, a paired challenger-minus-incumbent difference of **-1.67 percentage points**. There was one challenger-only win, three incumbent-only wins, and 116 ties.

At the registered 15-point margin and `alpha_fut=0.025`, the futility e-process first crossed its threshold of 40 at pair **44** (e-value 40.31). It ended at 55,268.30, also its maximum. The efficacy process did not cross its `alpha_eff=0.025` threshold; its maximum and final e-values were both 1.00. Under the protocol's conditional task-sampling and stable-agent assumptions, this is a futility decision against a challenger gain of at least 15 points on the defined randomized profile-edit population. It does not show that the two prompts are equal or that a smaller gain is impossible.

Collection ran from 2026-10-12 23:25:40 through 2026-10-13 01:06:30 UTC. All 240 episodes were recorded as `ok`; there were no retries, terminal missing episodes, verifier exceptions, or clock timeouts. Spend was **$23.60** under the $35 cap. The actual run completed the full 240 episodes. A strictly sequential stop at pair 44 would have required 88 terminal episodes before in-flight work, so its episode saving is counterfactual.

## Interpretation and limits

This trial uses new task instances drawn after registration from an explicit four-family distribution, unlike the earlier fixed public Linear bank. The task-sampling design supports a conditional population interpretation if the model and app behavior remain stable across the session. The 128-bit seeded pseudorandom draw, concurrent arms, and isolated resets improve that design but cannot themselves prove provider stability; the API model name is an alias.

The trial also has a severe capability floor. Raw task results show that neither agent changed any required profile field in **230/240** episodes. The baseline reached at least 30 recorded steps on 114/120 episodes, and the explorer on 117/120; browser-use can report 31 steps because it includes the initial navigation. Only four episodes met every required field. The e-process result is therefore evidence about this narrow, difficult profile-edit distribution, not evidence that the framework distinguishes close, capable systems across general web tasks. A broader sampled task distribution with nontrivial success remains needed for that claim.

## Audit files

- `results/paper/prospective_sampled_linear/tasks.csv`: task-order outcomes and timestamps, SHA-256 `3f5bb37800e0e73c0f604b0da3e5ee57ff403cfc0eecec8c6d3c8ffc15b6d673`.
- `results/paper/prospective_sampled_linear/attempts.csv`: all 240 episode records, costs, prompt hashes, and timestamps, SHA-256 `a352efaee5daca782920993f8ee77d0aa635e0ce22c8d73714801bffca4eb589`.
- `results/paper/prospective_sampled_linear/summary.json`: counts and registered paths, SHA-256 `ccb077c7e52e0c392aa16ed537485ab40d9910e0e5cd81c504c7428c50d5d1c6`.
- Raw worker JSONL files remain in `logs/prospective_sampled_linear/`; their SHA-256 values by worker 0–3 are `41436611c4880847a4e299d58d863a7d94fd26d78f64c91deb6967cda42d3448`, `115b8d54d7968ee077c9a0837cc6352fc1a459ed3d31a8b70d3d1b9766702230`, `27b0590c0fceea580187e6950ee501254b653bc63e4cacee05f0c7971343d0a1`, and `aaeed2f619f749fd57749d0030936e0979fdb7d55692d054a575724352e827a0`.

Recreate the derived files with `./.venv/bin/python experiments/growing_bandits/empirical/analyze_sampled_linear.py --table results/paper/prospective_sampled_linear/tasks.csv --attempts results/paper/prospective_sampled_linear/attempts.csv > results/paper/prospective_sampled_linear/summary.json`.
