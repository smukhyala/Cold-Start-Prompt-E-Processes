# Growing Bandits — Results

*87,148 oracle-labelled states. Generated 2026-09-13, analysed 2026-09-14.*

---

## What this study was trying to find out

The setup, in plain terms. You are evaluating prompts (or drugs, or ad variants — the maths
does not care). At every step you face one choice:

- **REFINE** — take one of the candidates you already have and test it once more, to learn
  more about how good it really is.
- **SEARCH** — throw away nothing, but spend your next test on a *brand-new* candidate you
  have never tried, drawn from an effectively unlimited pool.

You have a fixed budget of tests. At the end you must name one candidate as your answer, and
you are scored on how good that candidate truly is.

The question: **is there a simple rule that tells you which of the two to do, at any moment?**
And in particular — the reason this project exists — **does the anytime-valid e-process
machinery help you decide?** That machinery is what tells you, with rigorous statistical
guarantees, how confident you are that your current leader really is the best. The hypothesis
was that "how settled is the evidence?" should be the thing that drives the choice.

### How we answered it

For each of 87,148 saved situations, we ran the future twice:

1. Force **SEARCH** on the next step, then play out the rest of the budget with a fixed
   sensible strategy. Record how good the finally-chosen candidate truly was.
2. Rewind to exactly the same situation. Force **REFINE** instead. Play out the rest the
   same way. Record the same thing.

Repeat both hundreds or thousands of times with different random luck, and average. The
difference is the label:

> **A_t = (average final quality if you SEARCH) − (average final quality if you REFINE)**
>
> **A_t > 0 means searching was the better move. A_t < 0 means refining was.**

This is why it is called an *oracle* label: we can only compute it because we are in a
simulation and can replay the same situation twice. A real algorithm cannot. The whole point
is to learn a rule that predicts `A_t` from things a real algorithm *can* see, so that the
rule can then be deployed.

### How to read the numbers below

Two scores appear throughout. Both are worth understanding before reading on.

**Balanced accuracy.** The fraction of decisions you get right, but corrected so that always
guessing the more common answer scores exactly 0.500. So:

- **0.500 = a coin flip.** No skill whatsoever.
- **0.600 = you are right 60% of the time** in a fair comparison.
- **1.000 = perfect.**

The correction matters here because SEARCH is the better move in about 60% of situations. A
lazy rule that says "always SEARCH" would be right 60% of the time on raw accuracy, which
sounds decent and means nothing. Balanced accuracy scores that lazy rule at 0.500, correctly.

**AUC.** Roughly: pick one situation where searching was truly better and one where refining
was truly better, at random. AUC is the probability your rule rates the first one as more
search-worthy than the second. Again 0.5 is chance and 1.0 is perfect. AUC is useful because
it does not depend on where you draw the cutoff — it measures whether you have *ranked*
situations correctly, which is a more basic question than whether you picked a good threshold.

A rule of thumb for what follows: differences below about **0.01** are noise. Differences
around **0.05** are real but modest. Differences above **0.10** are large.

---

## Finding 1 — The e-process evidence does not help. This is the main result.

This was the study's central question, so it gets stated first and plainly.

We held everything constant — same model, same cross-validation, same data — and changed only
which features the model was allowed to see:

| What the model can see | k=1 | k=16 |
|---|---|---|
| Just the clock: time spent `t`, candidates so far `K_t` | 0.5423 | 0.5791 |
| **The clock + all the e-process evidence** | 0.5410 | **0.5853** |
| The clock + how good your current candidates look | **0.5714** | **0.6308** |

**Reading this:** adding every e-process feature we compute moves the score by **−0.001** in
one case and **+0.006** in the other. Both are inside the noise band. Meanwhile, swapping in
features that describe *how good your existing candidates are* moves it by **+0.052** — about
eight times as much, and far outside noise.

Looking at features one at a time tells the same story. This is the AUC of each single feature,
used alone as the entire decision rule:

| Feature | What it measures | AUC |
|---|---|---|
| `f_challenger_ucb` | how good your second-place candidate might plausibly be | **0.706** |
| `f_second_best_mean` | how good your second-place candidate looks on average | **0.700** |
| `f_leader_ucb` | how good your best candidate might plausibly be | 0.691 |
| `est_quantile_0.9` | how good your better candidates are, generally | 0.685 |
| `f_log_e_pair` | **the pairwise e-process evidence** | **0.535** |

The best single e-process feature scores **0.535** — barely distinguishable from a coin flip.
The best "how good is my current pool" feature scores **0.706** on its own, which is better
than every e-process feature *combined*.

**Why this might be.** The e-process answers *"how sure am I that my leader is the best of the
candidates I have?"* But that turns out not to be the question that matters. What matters is
*"is my current best actually any good, compared to what a fresh draw would give me?"* Being
extremely confident that your best candidate scores 0.3 does not make 0.3 a good outcome — it
just means you should go looking for something better. Confidence and quality are different
things, and it is quality that drives the decision.

**How much to trust this.** This is the one conclusion that survived every reframing we tried:
regression versus classification, weighted versus unweighted scoring, pooled versus
regime-by-regime, and at both commitment horizons. It did not move.

---

## Finding 2 — The decision *is* predictable, but the margin is modest

The rule does beat a growth schedule — the classical approach of "keep the number of
candidates near `c · t^(1/3)`" — but by less than the headline numbers suggest.

| | k=1 | k=16 |
|---|---|---|
| Growth schedule `K < c·t^(1/3)` | 0.5467 | 0.5964 |
| A model using all 71 observable features | **0.5736** | **0.6473** |
| **Margin** | +0.027 | +0.051 |

So at the longer commitment horizon the learned rule is right about 65% of the time versus
about 60% for the schedule. Real, but not dramatic.

### Two reasons the pooled numbers flatter the result

Both of these were found by adversarial review and then measured directly. They are the kind
of thing that is easy to miss and important to state.

**Reason 1: the model is partly recognising the situation type, not judging the situation.**

The answer varies a lot depending on where the state came from:

- Across horizons, the fraction of states where SEARCH wins rises steadily: **0.567 → 0.723**
  as the budget grows from 50 to 1000.
- Across the behavioural policies that generated the states, it ranges **0.474 → 0.761**.

So a model can score well partly by working out "this looks like a long-horizon state, and
those usually favour searching" — without actually judging *this particular* situation. To
measure how much of the score was that, we recomputed AUC *within* cells of
(horizon × allocation rule × generating policy × remaining budget), where the base rate is
held fixed:

> Pooled AUC **0.750** → within-regime AUC **0.702**

So about **0.048 of the headline was regime recognition**, and roughly 0.20 of genuine
above-chance skill remains. Still a real signal — just smaller than it first appeared.

**Reason 2: the baseline we were beating is itself contaminated.**

This one is subtle. The growth schedule looks at `K_t` versus `t^(1/3)` — but the number of
candidates you have at a given time is close to a *fingerprint of which strategy generated the
state*. An aggressive searcher has many candidates; a conservative one has few.

We tested this directly by building a deliberately silly rule: **"SEARCH if and only if the
state was generated by policy X"** — a rule that looks at nothing about the actual situation,
only at which strategy produced it. That rule scores **0.5596**.

The real schedule scores 0.5964. Both are measured against a 0.500 coin flip, so:

- schedule's skill above chance: 0.5964 − 0.500 = **0.0964**
- what policy identity alone gives you: 0.5596 − 0.500 = **0.0596**
- **62% of the schedule's apparent skill is just recognising which strategy made the state.**

That is an artefact of how our corpus was built — we deliberately generated states from eight
different behavioural policies for coverage. A schedule deployed for real would not get this
advantage. So the schedule is a *weaker* baseline than its number suggests, which cuts against
reading our margin over it as meaningful.

---

## Finding 3 — A single decision barely matters; a sustained commitment does

`A_t` as originally defined asks: what if I force **one** action, then go back to playing
normally? The answer, most of the time, is "almost nothing happens" — a good strategy simply
absorbs one forced move.

So we also recorded what happens if you commit to the forced action for **k rounds**:

| Commitment length | Typical size of the effect | How often the answer is clear-cut | SEARCH preferred |
|---|---|---|---|
| k = 1 round | 0.0022 | 27.5% | 41.0% |
| k = 4 rounds | 0.0085 | 54.0% | 54.5% |
| k = 16 rounds | 0.0252 | **80.4%** | 60.0% |

Committing for 16 rounds produces an effect roughly **ten times larger** than a single action,
and three times as many situations have a clear answer.

There is also a genuinely surprising detail: **the k=1 and k=16 answers agree on direction only
52.2% of the time** — essentially a coin flip. They are not noisy versions of each other; they
are answering different questions. "Is this one move worth it?" and "is it worth committing to
this for a while?" have genuinely different answers. That is why all three horizons are stored,
and why neither should be treated as a proxy for the other.

### When does the decision matter at all?

| Budget remaining | 0–10% | 10–25% | 25–50% | 50–75% | 75–100% |
|---|---|---|---|---|---|
| Clear-cut answers | **56.2%** | 47.3% | 31.8% | 23.1% | **8.9%** |

Near the end of your budget, more than half of situations have a definite right answer. Early
on, with most of your budget ahead of you, barely 1 in 11 does — because whatever you do now,
you have plenty of time to correct it.

---

## Finding 4 — Weak candidates get abandoned quickly, as intended

The research brief specifically warned against an error model of the form `sqrt(K/t)`, which
assumes every candidate you have ever tried needs to be measured to comparable precision. That
assumption is badly pessimistic if your strategy is any good, because a good strategy stops
spending on candidates that are clearly out of the running.

Measured on a 1000-test budget with 32 candidates:

| | Candidates ruled out | Candidates still in contention |
|---|---|---|
| How many | 4.5 of 32 | 27.5 |
| Tests spent on them | **31 tests (3.1%)** | 969 tests (96.9%) |
| Tests each | **7.0** | 35.1 |

Ruled-out candidates consume **3% of the budget in total** — not 3% each. So the pessimistic
assumption the brief flagged does not apply here, confirmed directly.

One honest caveat: this only really kicks in at longer budgets. Below about 500 tests, almost
nothing gets ruled out, because we are using a *genuinely rigorous* confidence sequence rather
than a convenient approximation, and rigour means you need real evidence before you are allowed
to discard anything.

---

## Three mistakes I made and fixed

These are documented because each one changed the answer, and because any of them could have
been reported as a result.

**Mistake 1 — a few hundred rows were drowning out the other 87,000.**

Labels were weighted by their precision, as `1 / SE²`, so that better-measured labels count
more. But 28% of rows have `SE` of exactly zero: both branches gave the identical answer every
single time, so there was no variance at all. Those rows got essentially infinite weight.

Result: **the top 1% of rows carried 80% of the total weight**, and since a tie counts as
"not SEARCH", the apparent SEARCH rate collapsed from 41% to 17%. Every model was being fitted
to a few hundred uninformative ties.

Fixed by flooring the weights at a sensible quantile and capping them at 20× the median. Ties
are now excluded from decision metrics entirely — if both actions give the same answer, there
is no right answer to score.

**Mistake 2 — I predicted the wrong thing, and reported a false null because of it.**

I fitted a model to predict the *size* of `A_t`, then said "search if the prediction is
positive." That sounds reasonable and is wrong. Squared-error fitting is dominated by the
handful of situations with huge `A_t` — the top 5% of rows account for 43% of the total error
being minimised. So the model spent all its effort on getting *magnitudes* right in extreme
cases, and was then asked a *direction* question it had never optimised for.

Predicting the direction directly lifts AUC from **0.694 to 0.751**.

On the back of this error I reported to you that nothing predicts the decision. That was wrong,
and it was my analysis, not the data.

One refinement worth recording: I initially thought the fix was "use a classifier instead of a
regression." It is not. A plain ridge regression fitted to the *direction* scores AUC 0.7165,
essentially identical to logistic regression's 0.7194 — same estimator, same weights, only the
target changed. **The lesson is "throw away the magnitude", not "use a classifier."**

**Mistake 3 — the baseline got to cheat.**

The growth schedule's threshold `c` was tuned on the full dataset and then scored on that same
data, while every learned model was properly cross-validated. That is an unfair comparison in
the baseline's favour. Tuned honestly, the schedule drops from 0.5889 to 0.5854, and to 0.5218
when held out by generating policy.

---

## What you can and cannot conclude

**Solid — backed by a theorem.** The confidence sequences and e-processes are genuinely
anytime-valid. You may look at them at any time, as often as you like, and stop whenever you
want, without invalidating the error guarantee. Verified empirically at α = 0.05 across three
different sampling rules.

**Solid — a measurement with error bars.** Every `A_t` value, each with its own standard error.
These are Monte Carlo estimates, not statements about what is optimal.

**Reasonably solid — observed on this corpus.** Findings 1 through 4. They hold across
holdouts by environment, reservoir family, horizon and generating policy. They do depend on
choices we made: the continuation strategy, the 64-candidate cap, and which reservoir families
were sampled.

**NOT established — and this matters most.** That a fitted rule would actually beat a schedule
*when deployed*. Predicting the oracle label well and achieving low regret when you actually
run the thing are two different claims, and only the second one matters practically. Both
contaminations in Finding 2 mean the offline margin is an **upper bound** on the deployed one.

The experiment that settles it is `evaluate_policy.py`, which runs a fitted rule from scratch as
a real algorithm against the baselines. **It has not been run with a fitted Φ yet.** That is the
obvious next step, and until it is done, "the learned rule is better" is a hypothesis, not a
result.

---

## The bottom line

If you take one thing from this: **the e-process evidence, which this whole line of work was
built around, does not drive the SEARCH-versus-REFINE decision.** What drives it is simply how
good your current candidates are. Confidence about your leader and quality of your leader are
different things, and it is quality that matters.

That is a negative result for the original hypothesis, but a clean and useful one — and it
points somewhere specific: a deployable rule should be built around estimating *how good a
fresh draw is likely to be relative to what you hold*, which is a reservoir-estimation problem,
not an inference problem.
