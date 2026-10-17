# Pre-registration 15: results (run 2026-10-11)

Registration: `docs/growing_bandits/DEPLOYMENT_PLAN.md`, Pre-registration 15 (commit 8e71efb), committed before any
study episode. Every number below comes from the frozen analyzer's output in `results/headroom_gmail/` (gitignored;
regenerate with `PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_headroom_gmail.py`).
Diagnostics labelled **exploratory** were not registered.

## Bottom line

| Registered question | Result | Registered reading |
|---|---|---|
| How much does a prompt containing the exact answer to every Gmail task beat the default? | H = **−1.7 pp**, 90% CI [−8.8, +5.5] | **small** (upper bound < 10 pp) |

- **As registered:** Gmail, like GitLab, has under 10 pp of answer-key headroom for this agent. "There was little to
  find" holds on both apps.
- **The registered caveat bites (exploratory).** The registration said H underestimates the true headroom if the
  agent ignores the key. It largely did: the agent referred to the key in only **33 of 120** answer-key episodes (28%).
  In the rest it solved the task from scratch, as if the key were not there.
- **So the precise finding is:** putting the literal answers in `gpt-5.4-mini`'s system prompt does not raise its
  Gmail success, because the agent mostly does not act on them. That is a statement about how little the prompt
  steers this agent. It is not a measurement of what a perfectly followed prompt could add.

## Operations

- **Episodes:** 240 of 240 `ok`. **Spend: $15.11**, against an estimate of $21.60 and a hard stop of $30.
- **Run:** 4 workers. Started 17:00 PDT on AC power (the launch waited for AC), finished 17:50.
- **Sleep:** 0 sleep-spanning records. **Step cap:** 2 records change under the 30-step rescore. **Timeouts:** 0 in
  either arm.

| Arm | Success | Mean cost | Mean steps | Rerun disagreement |
|---|---|---|---|---|
| O, `GM_oracle_full` | 0.708 | $0.059 | 9.9 | 15% |
| D, `GM_anchor_baseline` | 0.725 | $0.067 | 11.4 | 28% |

## Primary

- H = mean over the 60 tasks of O_t − D_t = **−0.017** (sd 0.331, se 0.043).
- The 90% CI is [−0.088, +0.055]. Its upper bound is < 0.10, so the reading is **small**.
- Per task: O is better on 9 tasks, worse on 11, and tied on 40.

## Secondary (registered)

1. **Task-cluster bootstrap:** 90% interval [−0.083, +0.050], reading small. 0.3% of resamples are ≥ 0.10.
2. **Capture:** on Leg M's 30 tasks, H = **−0.133** (90% CI [−0.224, −0.043]). The answer key did *worse* than the
   default there, so a capture fraction is undefined. The search winner's held-out lead was 0.0 pp, the same as the
   answer key's or better.
3. **Default drift:** on the same 30 tasks, the same default prompt scored **0.867 now against 0.717 in Leg M**
   (+15.0 pp, 90% CI [+5.8, +24.2]). The CI excludes 0.
   - Leg M ran with 1 worker on 2026-10-10; this study ran with 4 workers on 2026-10-11.
   - Session-to-session movement of the *same prompt* is larger than any prompt-to-prompt difference measured in
     this project.
4. **Operations:** see the table above. Rerun disagreement for D (28%) is again above September's 10%.

## Exploratory diagnostics (not registered)

**Did the agent use the key?** A keyword pass over every O episode's reasoning looked for "answer key" or "the key".

| O episodes | n | Success |
|---|---|---|
| Reasoning refers to the answer key | 33 | 19/33 = 58% |
| Reasoning never refers to it | 87 | 66/87 = 76% |

- Direct-URL navigation (`#/email/<id>`, `#/settings/...`), which the key's procedure prescribes, appears in 32 of
  120 episodes.
- Failure pattern: in typical failures (`task_h3`, `task_h17`, `task_h14`, `task_m12`), the agent finishes in 3–7
  steps and declares success without the reopen-and-verify step the header requires. For example, it selects 4
  emails, the app reports "1 conversation(s) archived", and it finishes.
- Episodes that use the key do worse. This is confounded: the agent may turn to the key on harder tasks. It does
  show that following the key, when the agent does, does not reliably produce the target state.

## What this means for the paper

- **The registered reading supports the flat-pool story:** on Gmail, as on GitLab, even a prompt containing the
  answers does not beat the default by 10 pp.
- **The mechanism is sharper than "low ceiling".** On GitLab the key needed a rewritten header to be followed at all,
  and then it gave +5.5 pp. On Gmail, with the same procedure, the agent used it in about a quarter of episodes and
  gained nothing.
  - What is in the system prompt barely governs this agent's behaviour, even when it is literally the answer.
  - That fits Pre-reg 12 (wording does not predict behaviour) and explains why searching over prompts finds nothing.
- **Limitation, stated plainly:** answer-key headroom is a *lower bound*, limited by how well the agent follows its
  prompt. It cannot show that a prompt the agent *did* follow would not help.
- **Drift is now a first-order limitation.** The same default moved +15 pp between consecutive sessions (CI excludes
  0). Any comparison across sessions, including Pre-reg 14's two-session control, must be read with that in mind.
  Within-session paired comparisons, which all registered primaries use, are unaffected.
