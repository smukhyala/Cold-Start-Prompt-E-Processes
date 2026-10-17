#!/usr/bin/env python3
"""Retrospective R6 pilot prediction, per DESIGN_CALCULATOR_PROTOCOL.md."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths

LABELS = ["efficacy", "futility", "unresolved"]
FEATURES = ["K", "pilot_incumbent", "pilot_discordance", "pilot_best_gain", "pilot_spread"]


def hit(path):
    idx = np.flatnonzero(path >= np.log(20))
    return int(idx[0] + 1) if len(idx) else None


def prompt_eval():
    base = ROOT / "results/paper/public_replay_02"
    listing = json.loads((base / "index.json").read_text())
    subjects = listing["subjects"]
    records = []
    for path in listing["files"]:
        subject = path.split("/", 1)[0]
        y = pd.read_parquet(base / "parquet" / path).to_numpy()
        if y.shape[0] != 100 or y.shape[1] < 100 or not np.isin(y, [0, 1]).all():
            continue
        order = np.random.default_rng(np.random.SeedSequence([20261012, 5, subjects.index(subject)])).permutation(y.shape[1])
        search = y[:, order[:50]]
        pilot = search[:, :10].astype(float)
        pgain = pilot[1:].mean(axis=1) - pilot[0].mean()
        e = hit(e1b_log_paths(search[None], np.array([0]), np.array([[False] + [True] * 99]))[0])
        f = hit(futility_log_paths((search[1:] - search[0])[None], .15)[0].min(axis=0))
        stop = min([x for x in (e, f) if x is not None], default=50)
        decision = "efficacy" if e == stop and f != stop else "futility" if f == stop and e != stop else "unresolved"
        records.append({"dataset": "PromptEval", "task": subject, "model": path.split("/", 1)[1],
                        "K": 99, "pilot_incumbent": pilot[0].mean(),
                        "pilot_discordance": (pilot[1:] != pilot[0]).mean(),
                        "pilot_best_gain": pgain.max(), "pilot_spread": pgain.std(),
                        "pilot_decided": decision != "unresolved" and stop <= 10,
                        "decision": decision})
    return records


def mizrahi():
    summary = json.loads((ROOT / "results/paper/mizrahi_replay/summary.json").read_text())
    return [{"dataset": "Mizrahi", "task": s["task"], "model": s["model"],
             **{k: s[k] for k in FEATURES + ["pilot_decided", "decision"]}}
            for s in summary["setting_rows"]]


def split_tasks(rows):
    rng = np.random.default_rng(20261017)
    train = set()
    for data in sorted({r["dataset"] for r in rows}):
        tasks = sorted({r["task"] for r in rows if r["dataset"] == data})
        rng.shuffle(tasks)
        train.update((data, t) for t in tasks[:len(tasks) // 2])
    return [r for r in rows if (r["dataset"], r["task"]) in train], [r for r in rows if (r["dataset"], r["task"]) not in train]


def X(rows):
    return np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)


def aligned_prob(model, x):
    raw = model.predict_proba(x)
    out = np.zeros((len(x), len(LABELS)))
    for j, label in enumerate(model.classes_):
        out[:, LABELS.index(label)] = raw[:, j]
    return np.maximum(out, 1e-12) / np.maximum(out, 1e-12).sum(axis=1, keepdims=True)


def metrics(rows, probs):
    yy = np.array([LABELS.index(r["decision"]) for r in rows])
    target = np.eye(len(LABELS))[yy]
    return {"n": len(rows), "counts": dict(Counter(r["decision"] for r in rows)),
            "brier": float(np.mean(np.sum((probs - target)**2, axis=1))),
            "log_loss": float(log_loss(yy, probs, labels=range(len(LABELS))))}


def fit(train):
    yy = [r["decision"] for r in train]
    group = [r["dataset"] + "/" + r["task"] for r in train]
    best = (float("inf"), None)
    for c in (.1, 1, 10):
        losses = []
        for tr, val in GroupKFold(n_splits=3).split(X(train), yy, group):
            rtr, rval = [train[i] for i in tr], [train[i] for i in val]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300))
            model.fit(X(rtr), [r["decision"] for r in rtr])
            # A fold may have no examples of one class; evaluate with a full class vector.
            p = aligned_prob(model[-1], model[0].transform(X(rval)))
            losses.append(metrics(rval, p)["log_loss"])
        if np.mean(losses) < best[0]:
            best = (float(np.mean(losses)), c)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=300))
    model.fit(X(train), yy)
    return model, {"C": best[1], "cv_log_loss": best[0]}


def evaluate(train, test):
    model, params = fit(train)
    p = aligned_prob(model[-1], model[0].transform(X(test)))
    base = np.array([np.mean([r["decision"] == label for r in train]) for label in LABELS])
    b = np.repeat(base[None], len(test), axis=0)
    actual = metrics(test, p)
    naive = metrics(test, b)
    return {"train_n": len(train), "test_n": len(test), "params": params,
            "model": actual, "prevalence": naive,
            "improves_both": actual["brier"] < naive["brier"] and actual["log_loss"] < naive["log_loss"]}


def main():
    all_rows = prompt_eval() + mizrahi()
    early = Counter(r["dataset"] for r in all_rows if r["pilot_decided"])
    remaining = [r for r in all_rows if not r["pilot_decided"]]
    tr, te = split_tasks(remaining)
    results = {"registered_target": "decision by 50 after unresolved pilot at 10",
               "total_rows": len(all_rows), "pilot_decided": dict(early),
               "remaining": len(remaining), "heldout_tasks": evaluate(tr, te),
               "prompt_eval_to_mizrahi": evaluate([r for r in remaining if r["dataset"] == "PromptEval"],
                                                      [r for r in remaining if r["dataset"] == "Mizrahi"]),
               "mizrahi_to_prompt_eval": evaluate([r for r in remaining if r["dataset"] == "Mizrahi"],
                                                      [r for r in remaining if r["dataset"] == "PromptEval"])}
    results["validated_for_design"] = all(results[k]["improves_both"] for k in
                                         ("heldout_tasks", "prompt_eval_to_mizrahi", "mizrahi_to_prompt_eval"))
    out = ROOT / "results/paper/design_calculator/summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
