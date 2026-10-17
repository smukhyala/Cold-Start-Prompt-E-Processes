# Prospective Linear fixed-pair trial result

The protocol, candidate pair, 120-task order, and prompt hashes were committed at `30fcc9a858d83d28a19b1d0acae80d181a403877` before the first outcome. See [the frozen protocol](PROSPECTIVE_LINEAR_PROTOCOL.md). Collection ran on 2026-10-12, from 19:23:56 to 21:21:03 UTC, using WebArena Infinity revision `b3015d7` and OpenAI `gpt-5.4-mini` through browser-use 0.13.1. The provider model name is an alias and does not pin its backend weights.

## Registered decision

All 120 task pairs completed. The baseline incumbent succeeded on 64/120 tasks (53.33%) and the explorer challenger on 66/120 (55.00%), a paired gain of 2/120 = **1.67 percentage points**. There were 11 challenger-only wins, nine incumbent-only wins, and 100 ties.

The 15-point futility process first crossed its allocated `alpha_fut=0.025` boundary of 40 at task **93**, with e-value 43.25; it ended at 88.52 and reached a maximum of 169.87. The efficacy process did not cross its `alpha_eff=0.025` boundary; its maximum was 1.00 and final value 0.134. Thus the registered path reaches **futility for a gain of at least 15 points**, under the protocol's conditional task-sampling null. This allows a smaller positive gain, and the observed difference does not establish efficacy.

The full collection used 240 successful terminal episodes plus one provider-error attempt that was retried successfully. There were no terminal missing episodes or exclusions. Recorded spend, including the retry, was **$16.51**, below the $35 cap. Had a strictly sequential runner stopped after task 93, it would have needed 186 instead of 240 terminal episodes before in-flight work; the actual run completed all 240 and incurred the full $16.51 cost.

## Scope of the inference

The pair and local task outcomes were evaluated prospectively, but the cohort is a fixed, public 120-task bank. The tasks were previously unused by this project's candidate selection, but model training exposure cannot be ruled out. A seeded permutation of a fixed bank does **not** by itself establish the conditional null required for the e-process; task difficulty and arm differences can vary across the fixed order. The crossing is a registered boundary result and evidence about this executed cohort. A 2.5% error guarantee for future tasks requires a justified conditional task-sampling or stability model, which this run does not independently verify. It should not be described as a fresh-task population certificate. Both arms were collected in the same session, with separate isolated app resets and worker timing that could differ within a pair.

## Audit files

- `results/paper/prospective_linear/tasks.csv`: all 120 pairs in frozen task order, with scores and timestamps; SHA-256 `e37a1deb2360a9ec14fa117d6ef811da822e8736270bd34106f403b1027cb9a5`.
- `results/paper/prospective_linear/attempts.csv`: 241 attempt rows, including the provider retry, prompt hashes, costs, and status; SHA-256 `4e31b4bbf2880fad60ef23ffbfa85b43c98204f582b63a30abc3af3fea6906b4`.
- `results/paper/prospective_linear/summary.json`: aggregate counts and registered paths; SHA-256 `68385849a7ac208b157e45c534eb9c5f22c3feb49ecccd0d43098adc0aaed613`.
- Raw worker JSONL files remain under `logs/prospective_linear/`; their SHA-256 values by worker 0–3 are `68e33853a5d31f6f94f129651091ef3314a831d86698a74bfd4ad8cb1264f5a8`, `c22a38573d2fdef2dd7495c04240d77e4ceb8de7b01e1d38b47f9a1477ee465f`, `a8f0c012bb4b9c2a71f7e56290d896aee0d7a2025b6b0920e874920c19018daa`, and `5eab90c63e513af513ec1dbfed9de9780778bc85a0af77601afb4b46ddeec7d5`.

Reproduce the derived files with `./.venv/bin/python experiments/growing_bandits/empirical/analyze_linear_trial.py --table results/paper/prospective_linear/tasks.csv --attempts results/paper/prospective_linear/attempts.csv > results/paper/prospective_linear/summary.json`.
On a clean checkout, `./.venv/bin/python experiments/growing_bandits/empirical/verify_linear_result.py` verifies the tracked tables against the frozen task order, prompt hashes, registered e-process crossings, and recorded spend without the raw logs.
