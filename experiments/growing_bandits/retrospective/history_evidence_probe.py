"""Audits observable evidence in saved first-run histories, without model calls.

Features read only agent output and tool observations recorded before scoring.
Associations with hidden outcomes are exploratory and are never inputs to features.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from first_run_triage import OUT, anchor_rows
from free_checks import saved_rows

EVIDENCE_WORDS = re.compile(
    r"\b(verif(?:y|ied)|confirm(?:ed)?|check(?:ed)?|shows?|display(?:s|ed)?|observed)\b", re.I
)
READ_ACTIONS = {"evaluate", "screenshot", "extract_content", "get_page_content"}


def history_features(raw: dict) -> dict[str, int]:
    path = Path(raw["task_dir"]) / "history.json"
    history = json.loads(path.read_text())["history"]
    actions = []
    for step in history:
        output = step.get("model_output") or {}
        for action in output.get("action") or []:
            actions.extend(action)
    last_click = max((i for i, action in enumerate(actions) if action == "click"), default=-1)
    read_after_click = (
        any(action in READ_ACTIONS for action in actions[last_click + 1 :])
        if last_click >= 0
        else False
    )
    return {
        "claims_verification": int(bool(EVIDENCE_WORDS.search(raw["final_result"] or ""))),
        "read_action_after_last_click": int(read_after_click),
        "has_read_action": int(any(action in READ_ACTIONS for action in actions)),
        "n_actions": len(actions),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logs = saved_rows()
    rows = []
    for anchor in anchor_rows():
        if not anchor["first_complete"]:
            continue
        raw = logs[anchor["pool"], anchor["arm_id"], anchor["task_id"], 0]
        rows.append({**anchor, **history_features(raw)})
    with (OUT / "history_evidence_completed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for pool in ("G", "F", "combined"):
        selected = [r for r in rows if pool == "combined" or r["pool"] == pool]
        for feature in (
            "claims_verification",
            "read_action_after_last_click",
            "has_read_action",
        ):
            for value in (0, 1):
                subset = [r for r in selected if r[feature] == value]
                if not subset:
                    continue
                summary.append(
                    {
                        "pool": pool,
                        "feature": feature,
                        "feature_value": value,
                        "n_completed_anchors": len(subset),
                        "first_success_rate": float(np.mean([r["first_success"] for r in subset])),
                        "same_retry_gain_if_replaced": float(
                            np.mean([r["same_success"] - r["first_success"] for r in subset])
                        ),
                        "distinct_retry_gain_if_replaced": float(
                            np.mean(
                                [
                                    r["different_success_expectation"] - r["first_success"]
                                    for r in subset
                                ]
                            )
                        ),
                    }
                )
    with (OUT / "history_evidence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
