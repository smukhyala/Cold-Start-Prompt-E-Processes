#!/usr/bin/env python3
"""EXPLORATORY (data already seen): B5 combined adaptive-allocation + known-baseline designs.

Combines the two wins found earlier:
  * B3 / B1 adaptive allocation (drop-the-loser for futility, Thompson for efficacy), and
  * B2 known-incumbent statistic X = (Y - b_t + 1) / 2 (no incumbent episodes at all).

Combined FUTILITY (B5a / B5b).  Each step: one fresh task, every still-active challenger is run
once (cost = number of active arms, NO incumbent episode).  Per-challenger downward E5 mixture on
X against (1 + delta) / 2.  An arm is dropped once its e-process >= 1/alpha and never reactivated.
The pool is certified when every arm has been dropped.
  (a) b_t known exactly.
  (b) b_t estimated once per task of a FINITE bank of N tasks from m prior default runs; tasks are
      revisited (drawn with replacement from the bank).  B2's Hoeffding bank-shift fix: run the
      test with delta' = delta - eps, eps = sqrt(log(2/alpha) / (2 m N)), at level alpha / 2
      (threshold 2/alpha).  Validity: with probability >= 1 - alpha/2 the bank offset
      mean_t(q_t - bhat_t) is >= -eps, so each null arm's factor has conditional mean <= 1 and
      Ville adds at most alpha/2.

Combined EFFICACY (B5a).  Each step: Thompson picks ONE challenger, one fresh task, one episode.
Upward aGRAPA capital of the pulled arm on X vs 1/2 (non-pulled arms frozen); global e-value is the
average of the K capitals.  Validity: the pulled arm is predictable; under the null
E[X - 1/2 | past, arm] <= 0 because b_t is the true per-task rate; the average of supermartingales
is a supermartingale.

Everything is batched over simulations (axis 0).  No API calls.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.growing_bandits.audit import adaptive_tests as ad
from experiments.growing_bandits.audit import futility_eprocess as fu

EPS = 1e-9


# ----------------------------------------------------------------------------
# efficacy
# ----------------------------------------------------------------------------
class KnownEfficacyState(ad.EfficacyState):
    """EfficacyState whose pulled arms ALWAYS update on X in [0, 1] (no incumbent discordance
    filter).  wins / disc hold the fractional win count and pull count for Thompson."""

    def update(self, pulled: np.ndarray, x: np.ndarray, y_inc=None) -> None:
        upd = pulled
        m0 = self.m0
        term = 1.0 + self.lam * (x - m0)
        self.log_kp = self.log_kp + np.where(upd, np.log(np.maximum(term, 1e-300)), 0.0)
        cnt_new = self.cnt + 1.0
        sum_x_new = self.sum_x + x
        mu = (0.5 + sum_x_new) / (1.0 + cnt_new)
        sum_sq_new = self.sum_sq + (x - mu) ** 2
        sig2 = (0.25 + sum_sq_new) / (1.0 + cnt_new)
        lam_new = np.clip((mu - m0) / np.maximum(sig2, EPS), 0.0, self.cap)
        self.cnt = np.where(upd, cnt_new, self.cnt)
        self.sum_x = np.where(upd, sum_x_new, self.sum_x)
        self.sum_sq = np.where(upd, sum_sq_new, self.sum_sq)
        self.lam = np.where(upd, lam_new, self.lam)
        self.disc = self.disc + upd
        self.wins = self.wins + np.where(upd, x, 0.0)


def simulate_known_efficacy(rule: str, K: int, shifts: np.ndarray, q_grid: np.ndarray, budget: int,
                            alpha: float, rng: np.random.Generator, lo: float = 0.01,
                            hi: float = 0.99) -> dict:
    """rule 'all': every challenger on every task (K episodes per task, B2 known design).
    rule 'thompson': one challenger per step (1 episode per step, B5a combined).
    Returns ep_rej / tasks_rej (inf if no rejection within budget), and episode accounting."""
    B = shifts.shape[0]
    per_step = K if rule == "all" else 1
    max_steps = budget // per_step
    thr = math.log(1.0 / alpha)
    st = KnownEfficacyState(B, K)
    rows = np.arange(B)
    done = np.zeros(B, dtype=bool)
    ep_rej = np.full(B, np.inf)
    tasks_rej = np.full(B, np.inf)
    incumbent_episodes = 0
    for t in range(max_steps):
        q = q_grid[rng.integers(len(q_grid), size=B)]
        if rule == "all":
            qk = np.clip(q[:, None] + shifts, lo, hi)
            y = (rng.random((B, K)) < qk).astype(float)
            x = (y - q[:, None] + 1.0) / 2.0
            pulled = np.ones((B, K), dtype=bool)
        else:
            score = rng.beta(st.wins + 1.0, st.disc - st.wins + 1.0)
            arm = score.argmax(axis=1)
            qa = np.clip(q + shifts[rows, arm], lo, hi)
            ya = (rng.random(B) < qa).astype(float)
            x = np.zeros((B, K))
            x[rows, arm] = (ya - q + 1.0) / 2.0
            pulled = np.zeros((B, K), dtype=bool)
            pulled[rows, arm] = True
        st.update(pulled, x)
        newly = (st.log_global() >= thr) & ~done
        if newly.any():
            ep_rej[newly] = (t + 1) * per_step
            tasks_rej[newly] = t + 1
            done = done | newly
            if done.all():
                break
    return {"ep_rej": ep_rej, "tasks_rej": tasks_rej, "episodes_per_step": per_step,
            "incumbent_episodes": incumbent_episodes}


# ----------------------------------------------------------------------------
# futility
# ----------------------------------------------------------------------------
def hoeffding_eps(m: int, N: int, alpha: float) -> float:
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * m * N))


def simulate_known_futility(mode: str, K: int, shifts: np.ndarray, task_q: np.ndarray, delta: float,
                            budget: float, alpha: float, rng: np.random.Generator,
                            baseline: str = "exact", m: int = 30, trace: bool = False,
                            lo: float = 0.01, hi: float = 1.0) -> dict:
    """mode 'drop' (B5) or 'all' (B2 known-baseline only: run every arm each task, no dropping,
    certify when min_k E_k >= 1/alpha).  No incumbent episodes in either mode.

    baseline 'exact': task_q is the difficulty grid, tasks drawn i.i.d. from it (fresh each step).
    baseline 'bank' : task_q is the bank (N,) of true per-task rates; tasks are drawn with
                      replacement; bhat = Binomial(m, q)/m fixed per bank task and simulation;
                      Hoeffding shift delta' = delta - eps and threshold 2/alpha.
    Returns episodes / distinct tasks at certification (inf if none) and accounting arrays.
    """
    B = shifts.shape[0]
    N = len(task_q)
    if baseline == "bank":
        eps_shift = hoeffding_eps(m, N, alpha)
        delta_eff = delta - eps_shift
        thr = math.log(2.0 / alpha)
        bhat = rng.binomial(m, task_q[None, :], size=(B, N)).astype(float) / m
        visited = np.zeros((B, N), dtype=bool)
    else:
        eps_shift = 0.0
        delta_eff = delta
        thr = math.log(1.0 / alpha)
        bhat = None
        visited = None
    st = ad.FutilityState(B, K, delta_eff)
    rows = np.arange(B)
    active = np.ones((B, K), dtype=bool)
    ep = np.zeros(B)
    done = np.zeros(B, dtype=bool)
    ep_cert = np.full(B, np.inf)
    tasks_cert = np.full(B, np.inf)
    steps_used = np.zeros(B)
    incumbent_episodes = np.zeros(B)
    history = []
    t = 0
    while True:
        if mode == "drop":
            cost = active.sum(axis=1).astype(float)
        else:
            cost = np.full(B, float(K))
        alive = ~done & (ep + cost <= budget) & (cost > 0)
        if not alive.any():
            break
        idx = rng.integers(N, size=B)
        q = task_q[idx]
        qk = np.clip(q[:, None] + shifts, lo, hi)
        y = (rng.random((B, K)) < qk).astype(float)
        if baseline == "bank":
            b = bhat[rows, idx]
            visited[rows[alive], idx[alive]] = True
        else:
            b = q
        x = (y - b[:, None] + 1.0) / 2.0
        if mode == "drop":
            mask = active & alive[:, None]
        else:
            mask = np.ones((B, K), dtype=bool) & alive[:, None]
        st.update(mask, x)
        ep = ep + np.where(alive, cost, 0.0)
        steps_used = steps_used + alive
        log_e = st.log_e()
        if mode == "drop":
            newly_dropped = active & (log_e >= thr) & alive[:, None]
            active = active & ~newly_dropped
            cert = ~active.any(axis=1)
            if trace:
                history.append(active.copy())
        else:
            cert = log_e.min(axis=1) >= thr
        newly = cert & alive
        if newly.any():
            ep_cert[newly] = ep[newly]
            if baseline == "bank":
                tasks_cert[newly] = visited.sum(axis=1)[newly]
            else:
                tasks_cert[newly] = t + 1
            done = done | newly
        t += 1
    out = {"ep_cert": ep_cert, "tasks_cert": tasks_cert, "ep_used": ep, "steps_used": steps_used,
           "incumbent_episodes": incumbent_episodes, "delta_eff": delta_eff, "eps_shift": eps_shift}
    if trace:
        out["active_history"] = history
    return out


def make_bank(q_grid: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Bank of N tasks whose difficulties are drawn (with replacement) from the G pool means."""
    return q_grid[rng.integers(len(q_grid), size=N)]


# ----------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------
def med(values: np.ndarray) -> float:
    return float(np.median(values))


def decided_frac(values: np.ndarray) -> float:
    return ad.rate_finite(values)
