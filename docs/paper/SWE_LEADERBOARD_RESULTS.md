# R4 dated SWE-bench Verified leaderboard audit

Analysed 2026-10-12 from `SWE-bench/experiments` revision `40f164d5b8f1d249bf95a6df8b74b577fd8e519d`. Protocol and post-inspection amendment: [SWE_LEADERBOARD_PROTOCOL.md](SWE_LEADERBOARD_PROTOCOL.md). Code: `experiments/growing_bandits/audit/swe_leaderboard_replay.py`; derived summaries: `results/paper/swe_leaderboard/{strict,official}.json`. Source result files remain local and untracked. The directory prefix is used as a date stamp; it is not independently verified as publication or first-use date.

## Coverage and source semantics

The pinned source has 182 date-stamped Verified folders. Of these, 135 contain item-level `results/results.json`; 47 provide only a score in metadata. The initial registered rule required at least 490 distinct listed IDs in a result file, leaving **12 eligible entries**, 11 comparisons, 8 score-leader claims, 3 nominal and 3 entrant-spending crossings. That strict result describes only the early, densely enumerated files.

The source's [result-generation code](https://github.com/SWE-bench/experiments/blob/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/analysis/get_results.py) writes `resolved` IDs plus a few failure categories and may omit ordinary unresolved issues. Its [leaderboard code](https://github.com/SWE-bench/experiments/blob/40f164d5b8f1d249bf95a6df8b74b577fd8e519d/analysis/get_leaderboard.py) scores an entry as `len(resolved)/500`. Thus the original 490-ID requirement was incompatible with the later sparse format; a missing ID from the result lists is an unsuccessful issue for the published score, not automatically a missing benchmark task. This was learned after the strict pass, so the source-semantic extension is explicitly **post-inspection and exploratory**.

All 135 result files are well-formed under the documented `resolved`-list semantics and their ID union is the expected 500 Verified issues. The 47 metadata-only entries have scores but no released itemwise resolved list. A score-only chronology confirms that none of the 47 displaces the running best score; consequently the 135 result-bearing entries determine the score-leader sequence in this pinned snapshot. The final score leader among them is `20251205_sonar-foundation-agent_claude-opus-4-5` at 396/500 (79.2%).

## Historical score-leader sequence

Sort entries by the directory date and full name, keep the highest 500-task score seen so far, and compare every result-bearing entrant to the prior leader. A strictly higher score updates the displayed leader even when its E1b path does not cross. The same fixed, seeded order of 500 paired issues is used for nominal `alpha=0.05` and entrant-index spending `alpha_j=0.05/[j(j+1)]`.

| Measure | Source-semantic extension |
|---|---:|
| Itemwise result entries / comparisons | 135 / 134 |
| Reported-score leader claims | **32** |
| Claims crossing nominal 5% efficacy | **4** |
| Claims crossing entrant-index spending threshold | **1** |
| Median first crossing among nominal hits | 147 issues |
| Median first crossing among spending hits | 214 issues |
| Paired episodes saved across the 32 claims, nominal / spending | 2,596 / 572 out of 32,000 |
| Non-leader entries with a nominal efficacy crossing | 0 |

The four nominal crossings are `20240402_sweagent_claude3opus` (+44/500), `20240402_sweagent_gpt4` (+33/500), `20240612_MASAI_gpt4o` (+35/500), and `20240824_gru` (+23/500). Only the Claude 3 Opus entry also crosses its entrant-index spending threshold. An earlier fixed-pair replay found stronger detection for GPT-4 and Claude because it compared each SWE-agent system to a same-model RAG system; this dated replay compares each entrant to the current *highest-scoring whole system*, which is a different reference. Even the 4 nominal crossings are descriptive historical flags, not four fresh certifications of future-task improvement.

The date prefix orders entries only to a day. In a post hoc 2,000-shuffle sensitivity check of entries within the same date, the number of displayed leader increases ranges from 30 to 36 (median 32), against 32 in the frozen lexicographic order. The crossing counts above apply to that frozen order; they were not reinterpreted as invariant to unknown within-day submission time.

## Interpretation

The exercise answers a narrow leaderboard question: many displayed score increases are too small to cross the frozen task-order boundary, and a stringent entrant-index threshold crosses only once. It cannot establish a prospective familywise error rate. All submissions reuse the same public 500 issues; the running leader is selected using those outcomes, and later systems may have been developed in response to prior leaderboard feedback. A nominal E1b threshold or alpha-spending arithmetic does not remove those adaptive-data and benchmark-reuse problems. The source also mixes models, agent implementations, and attempt policies, so the contrasts describe whole systems rather than a causal effect of a model or scaffold.
