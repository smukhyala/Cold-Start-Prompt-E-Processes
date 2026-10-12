# Public replay 03 results: matched SWE-bench Verified systems

Run 2026-10-12 under `PUBLIC_REPLAY_PREREG_03.md`, using `SWE-bench/experiments` revision `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`. Code: `experiments/growing_bandits/audit/public_replay_03.py`; derived summary: `results/paper/public_replay_03/summary.json`. The source repository has no root license statement that we could find, so its raw result files were read from GitHub and not copied into this repository.

The metadata verifies exact within-pair model versions: `gpt-4-1106-preview` and `claude-3-opus-20240229`. Both pairs compare submissions dated 2024-04-02. Each reference RAG system records `2+` attempts per issue and each SWE-agent system records `1`, so this is a comparison of whole systems with different operating rules, not an isolated scaffold causal effect.

| Pair (RAG → SWE-agent) | Common tasks | Resolved, reference → challenger | Paired gain [task-bootstrap 95%] | Discordant wins:losses | First E1b crossing, fixed order | 2,000-order sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| GPT-4 (1106) | 495 | 13 → 112 | **+20.0 pp** [16.6, 23.6] | 100:1 | **92 tasks** | 2,000/2,000 cross; median 53 tasks |
| Claude 3 Opus | 496 | 35 → 79 | **+8.9 pp** [5.6, 12.1] | 58:14 | **95 tasks** | 2,000/2,000 cross; median 140 tasks |

Both prespecified E1b processes crossed the Bonferroni-adjusted `alpha=0.025` boundary in the fixed task order. The GPT-4 comparison used 184 paired episodes at the crossing against 990 for a full grid (81.4% fewer); Claude used 190 against 992 (80.8% fewer). These are stopping-time counterfactuals on already available task outcomes. They demonstrate that this procedure *can* detect large real system gains on a paired benchmark. The two comparisons were fixed by names and metadata before their outcomes were read.

The GPT-4 challenger contains four duplicate `no_generation` status entries, which were deduplicated as instance IDs; none creates an ambiguous resolved status. The per-pair intersections omit five GPT-4 and four Claude instances absent from the challengers' result files. Those omissions were based on record presence, not success, and were reported under the registered common-instance rule.

These results close the narrow **"never detects a real winner"** gap in `FRAMEWORK_GAP_PLAN.md` for historical benchmark replays. They do not show a prospective certification, a causal scaffold effect, validity under session drift, or efficacy for small gains. A fixed public benchmark and systems developed against it also limit the population interpretation of the anytime-valid threshold. Terminal-Bench's +8.8-point Capy gain did not cross at 68 tasks, illustrating the power limit alongside these positive cases.
