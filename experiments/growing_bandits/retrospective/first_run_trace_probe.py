"""Exploratory grouped-CV probe for first-run false-completion signals.

Uses only saved first-run metadata as predictor inputs. Held-out task IDs are
shared across G and F folds. The second-run outcomes are used only for scoring.
"""

from __future__ import annotations

import csv

import numpy as np
from first_run_triage import OUT, anchor_rows
from free_checks import saved_rows
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(row: dict, raw: dict) -> list[float]:
    errors = raw.get("errors") or []
    return [
        np.log1p(row["first_steps"]),
        np.log1p(100 * row["first_cost_usd"]),
        np.log1p(float(raw["wallclock_s"])),
        np.log1p(len(raw["final_result"] or "")),
        np.log1p(sum(str(error).lower() not in ("none", "", "null") for error in errors)),
        float(row["pool"] == "F"),
    ]


def crossfit(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logs = saved_rows()
    x = np.asarray([features(r, logs[r["pool"], r["arm_id"], r["task_id"], 0]) for r in rows])
    fail = np.asarray([1 - r["first_success"] for r in rows])
    groups = np.asarray([r["task_id"] for r in rows])
    pred = np.full(len(rows), np.nan)
    base = np.full(len(rows), np.nan)
    uplift = np.full(len(rows), np.nan)
    observed_uplift = np.asarray([r["same_success"] - r["first_success"] for r in rows])
    for train, test in GroupKFold(n_splits=5).split(x, fail, groups):
        assert not set(groups[train]) & set(groups[test])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(x[train], fail[train])
        pred[test] = model.predict_proba(x[test])[:, 1]
        base[test] = float(fail[train].mean())
        benefit_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        benefit_model.fit(x[train], observed_uplift[train])
        uplift[test] = benefit_model.predict(x[test])
    return pred, base, uplift


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in anchor_rows() if r["first_complete"]]
    pred, base, uplift = crossfit(rows)
    fail = np.asarray([1 - r["first_success"] for r in rows])
    selected = np.zeros(len(rows), dtype=bool)
    benefit_selected = np.zeros(len(rows), dtype=bool)
    for pool in ("G", "F"):
        indices = np.flatnonzero([r["pool"] == pool for r in rows])
        n = round(0.2 * len(indices))
        selected[indices[np.argsort(-pred[indices])[:n]]] = True
        benefit_selected[indices[np.argsort(-uplift[indices])[:n]]] = True
    for i, r in enumerate(rows):
        r["cv_failure_risk"] = float(pred[i])
        r["cv_baseline_risk"] = float(base[i])
        r["selected_top_20pct"] = int(selected[i])
        r["cv_predicted_same_retry_benefit"] = float(uplift[i])
        r["benefit_selected_top_20pct"] = int(benefit_selected[i])
    with (OUT / "trace_probe_completed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    rng = np.random.default_rng(20261007)
    for pool in ("G", "F", "combined"):
        idx = np.flatnonzero([pool == "combined" or r["pool"] == pool for r in rows])
        subset = [rows[i] for i in idx]
        y = fail[idx]
        s = selected[idx]
        bs = benefit_selected[idx]
        # Exact random-allocation expectation at the matched selected fraction.
        q = float(s.mean())
        score = {
            "pool": pool,
            "n_completed_anchors": len(idx),
            "n_first_failures": int(y.sum()),
            "task_group_cv_auc": float(roc_auc_score(y, pred[idx])),
            "task_group_cv_brier": float(brier_score_loss(y, pred[idx])),
            "fold_prevalence_brier": float(brier_score_loss(y, base[idx])),
            "retry_fraction": q,
        }
        task_names = sorted({r["task_id"] for r in subset})
        task_to_indices = {
            task: np.asarray([j for j, r in enumerate(subset) if r["task_id"] == task])
            for task in task_names
        }
        for kind, outcome_key, cost_key in (
            ("same", "same_success", "same_cost_usd"),
            ("different", "different_success_expectation", "different_cost_expectation_usd"),
        ):
            gains = np.asarray([r[outcome_key] - r["first_success"] for r in subset])
            costs = np.asarray([r[cost_key] for r in subset])
            score[f"risk_selected_{kind}_gain"] = float(np.mean(gains[s]))
            score[f"benefit_selected_{kind}_gain"] = float(np.mean(gains[bs]))
            score[f"benefit_selected_{kind}_gain_per_completed_task"] = float(np.mean(gains * bs))
            bootstrap = []
            for _ in range(5000):
                sampled = np.concatenate(
                    [
                        task_to_indices[t]
                        for t in rng.choice(task_names, len(task_names), replace=True)
                    ]
                )
                bootstrap.append(float(np.mean((gains * bs)[sampled])))
            score[f"benefit_selected_{kind}_gain_task_boot_lo"] = float(
                np.quantile(bootstrap, 0.025)
            )
            score[f"benefit_selected_{kind}_gain_task_boot_hi"] = float(
                np.quantile(bootstrap, 0.975)
            )
            score[f"random_{kind}_gain_expectation"] = float(np.mean(gains))
            score[f"risk_selected_{kind}_extra_cost_per_completed_task_usd"] = float(
                np.mean(costs * s)
            )
            score[f"random_{kind}_extra_cost_per_completed_task_usd"] = float(q * costs.mean())
            score[f"benefit_selected_{kind}_extra_cost_per_completed_task_usd"] = float(
                np.mean(costs * bs)
            )
        summary.append(score)
    with (OUT / "trace_probe_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
