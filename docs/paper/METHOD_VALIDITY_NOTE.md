# What the sequential error claims require

This note audits the implementation used in the framework manuscript. It distinguishes its prospective conditional guarantees from the descriptive boundaries drawn on historical benchmark matrices.

## Efficacy

Fix an incumbent and `K` challengers before monitoring. For binary paired outcome `Y_kt, Y_bt`, let `C_kt` indicate discordance and let `W_kt=Y_kt` on discordance. For the implemented E1b bet, the sufficient sequential null is

`P(W_kt=1 | F_(t-1), C_kt=1) <= 1/2`

for every challenger and task with positive discordance probability. An admissible predictable bet `lambda_kt` multiplies challenger capital by `1 + lambda_kt*(W_kt - 1/2)` on discordance and by one on ties. Under that conditional null, each capital is a nonnegative supermartingale. The equal mixture over the `K` challenger capitals is also a nonnegative supermartingale under the intersection null. Ville's inequality bounds its chance of ever crossing `1/alpha` by `alpha`.

For independent identically distributed paired tasks, the conditional discordant-win inequality is equivalent to a nonpositive mean paired gain. For adaptively selected or nonstationary tasks, the unconditional mean-gain null alone does not give the required conditional inequality. The implementation keeps each bet predictable by updating it only after the current observation.

## Futility

For challenger `k`, set `D_kt=Y_kt-Y_bt` and `X_kt=(D_kt+1)/2`. The margin null is the conditional lower bound `E[D_kt | F_(t-1)] >= delta`, equivalently `E[X_kt | F_(t-1)] >= m_delta=(1+delta)/2`. An admissible predictable downward bet multiplies capital by `1-lambda_kt*(X_kt-m_delta)`, whose conditional expectation is at most one under the arm's margin null. The implementation mixes one adaptive and four fixed admissible bets for each arm.

The pool null says that *some* challenger has gain at least `delta`. Simultaneous crossing by **all** arm futility processes implies that this true arm crosses. Its Ville bound therefore controls a false pool-futility declaration at level `alpha`; no independence across arms is required. The minimum over arms is a convenient crossing statistic, but its own supermartingale status need not be asserted. Efficacy and futility use separate level-`alpha` statements; a joint familywise 5% claim for both would require a further allocation.

## Where historical replay differs

Randomly permuting a fixed benchmark is a sensitivity check, not a new sequence of independent tasks. Even a fixed population with total paired gain zero can violate the conditional null: with one challenger win and one loss, drawing the loss first makes a win certain next. Reusing a public benchmark also lets a candidate or the running leaderboard incumbent depend on earlier scores from the same tasks. A historical crossing is consequently a descriptive counterfactual unless fresh-task sampling and fixed-pool conditions are justified separately. Reusing a historical default across sessions requires a prospective drift bound; a confidence interval for its old mean covers only old-session estimation error.

The clean future-task design freezes the candidate pool, samples or prespecifies new tasks representing the target population, evaluates the incumbent concurrently on each task, records exclusions before outcomes, and uses a separately allocated error budget for each desired claim family. New candidates proposed after reading monitored outcomes need a new inference design rather than automatic inclusion in the existing equal-weight mixture.
