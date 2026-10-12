# Prospectively sampled Linear profile-edit trial — registration before task draw

Registered 2026-10-12, before drawing this study's task seed or observing an agent outcome on its generated tasks. The implementation is `experiments/growing_bandits/empirical/prepare_sampled_linear.py` and the `sampled_linear` collector profile. The earlier fixed-bank Linear result may have influenced the decision to reuse these two arms, but this study's task instances are independently drawn after the pair is frozen.

## Target population and random sampling

Each task is an independent draw with replacement from this defined distribution of Linear account profile edits. Choose one of four families uniformly: change name and username; username and email; name and email; or all three. Independently choose a first name uniformly from the 12-entry `FIRST` list, a surname uniformly from the 12-entry `LAST` list, and an integer suffix uniformly from 1000 through 9999. Construct a full name, username, and `@example.net` email as in the committed generator. All targets differ from the app's reset values. Duplicate draws are retained. There is no filtering based on task difficulty or agent outcomes. This population is intentionally narrow; conclusions do not automatically cover all Linear tasks or other web apps.

After committing this protocol and generator, draw one 128-bit seed from operating-system entropy and generate the 120-task suite once. The seed and every realized instruction, task family, verifier, and queue item will be committed before any paid episode. NumPy's PCG64 stream determines task draws; a separately initialized stream randomizes the two arm positions within each task. The queue preserves draw order for monitoring. The app state is reset before each arm-task episode.

## Fixed pair, outcome, and operations

- Incumbent: `linear_baseline`; challenger: `linear_explorer`. Reuse the exact rendered prompt hashes and vectors from the frozen 2026-10-12 Linear pool. No substitution after the task draw or pilot outcomes.
- Agent: the same browser-use 0.13.1 adapter, OpenAI `gpt-5.4-mini` with low reasoning effort, no vision, headless, 30-step cap, and 180-second episode timeout. The model API alias may change backend weights without notice.
- Outcome: binary verifier success. Each generated verifier reads the app's `currentUser` state and requires every target field to equal its drawn value. Partial completion is zero. The generator and a verifier false-before/true-after state check were tested on a separate all-zero preflight seed, without an agent call; those preflight task instances are excluded from the study.
- Run all 120 pairs (240 terminal episodes) through four workers in one session. Historical same-app cost was $16.51 for 240 episodes; the hard collector cap is $35, with only in-flight overshoot possible. Preserve every attempt, app/task hashes, prompt hashes, costs, timestamps, and task-level outcomes.
- A genuine agent timeout is a failure. Harness or provider errors are retried up to three attempts and then marked `missing`; a provider outage stops collection. If any task has a terminal missing arm, or the cap/outage prevents all 120 pairs, the primary result is incomplete and no confirmatory claim is made. Descriptive complete-pair paths may still be reported.

## Registered sequential decision

For task $t$, set $D_t=Y_{\mathrm{explorer},t}-Y_{\mathrm{baseline},t}$. The meaningful margin is $\delta=0.15$. Use the repository's E1b discordant-task efficacy process for $H^+_0: \mathbb E[D_t\mid\mathcal F_{t-1}]\leq0$ and its downward paired-difference mixture for $H^-_0: \mathbb E[D_t\mid\mathcal F_{t-1}]\geq0.15$. Allocate $\alpha_\mathrm{eff}=0.025$ and $\alpha_\mathrm{fut}=0.025$, each with e-value threshold 40. Report first crossing, maximum and final e-values, paired success rates, discordant counts, missingness, spend, and timestamps. If neither crosses, report unresolved; if both cross, report both. The full run continues to 120 pairs even if an early boundary crosses, preserving a complete outcome table. No change to the pair, margin, tasks, or tests is permitted after outcomes begin.

The random draw of task specifications supports an iid task-population interpretation if the agent, verifier, resets, and provider behavior remain stable enough that the conditional null holds. Pseudorandom sampling, a fixed API model alias, and same-session concurrency do not prove that stability. The claim is limited to this randomized profile-edit distribution and the recorded execution conditions.
