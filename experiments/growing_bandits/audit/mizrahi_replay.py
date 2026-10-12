#!/usr/bin/env python3
"""Retrospective natural-default replay registered in MIZRAHI_REPLAY_PROTOCOL.md."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.futility_eprocess import (
    first_crossing, futility_log_paths, pool_log_paths,
)


def rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def flags(path: Path):
    rr = rows(path)
    defaults = sorted(int(r["prompt idx"]) for r in rr if r["default prompt"].lower() == "true")
    if not defaults:
        raise ValueError(f"no default: {path}")
    challengers = sorted(int(r["prompt idx"]) for r in rr
                         if r["default prompt"].lower() == "false" and r["correct"].lower() in {"1", "true"})
    return defaults[0], challengers


def model_from_member(member: str, task: str, bench: str):
    stem = Path(member).stem
    prefix = ("big_bench_hard_" if bench == "BBH" else "lmentry_") + task + "_"
    if not stem.startswith(prefix):
        return None
    model = stem[len(prefix):].split("_temp")[0]
    return model.replace("t0-3b", "t0_3b")


def outcomes(z: zipfile.ZipFile, info, prompt_ids):
    data = json.load(z.open(info))
    matrix = {}
    for example_id, groups in data["samples"].items():
        merged = {int(p): row for group in groups.values() for p, row in group.items()}
        vals = []
        for p in prompt_ids:
            v = merged.get(p, {}).get("auto_validation")
            vals.append(v if v in (0, 1) and type(v) is int else None)
        matrix[str(example_id)] = vals
    return matrix


def first_hit(path, alpha=.05):
    hit = first_crossing(path, alpha)
    return int(hit[0] + 1) if hit[0] >= 0 else None


def analyze(y, task_index):
    # y: incumbent, then ascending challenger indices, in common lexicographic example order.
    n = y.shape[1]
    perm = np.random.default_rng(np.random.SeedSequence([20261012, 21, task_index])).permutation(n)
    search = y[:, perm[:n // 2]]
    held = y[:, perm[n // 2:]]
    gains = search[1:].mean(axis=1) - search[0].mean()
    best = int(np.argmax(gains)) + 1
    search_gain = float(gains[best - 1])
    held_gain = float(held[best].mean() - held[0].mean())
    log_e = e1b_log_paths(search[None], np.array([0]),
                            np.array([[False] + [True] * (len(y) - 1)]))[0]
    log_f = futility_log_paths((search[1:] - search[0])[None], .15)[0]
    eff = first_hit(log_e[None])
    fut = first_hit(pool_log_paths(log_f[None]))
    # Earliest of two decisions; if both on same task, call it ambiguous.
    decision = "unresolved"
    stop = search.shape[1]
    if eff is not None or fut is not None:
        stop = min(x for x in (eff, fut) if x is not None)
        decision = "efficacy" if eff == stop and fut != stop else "futility" if fut == stop and eff != stop else "both"
    pilot = search[:, :10].astype(float)
    pilot_gains = pilot[1:].mean(axis=1) - pilot[0].mean()
    return {"selected_row": best, "search_gain": search_gain, "heldout_gain": held_gain,
            "shrinkage": search_gain - held_gain, "positive_heldout": held_gain > 0,
            "efficacy_task": eff, "futility_task": fut, "decision": decision,
            "stop_task": stop, "search_tasks": search.shape[1], "K": len(y) - 1,
            "full_episodes": int(search.size), "stopped_episodes": int(stop * len(y)),
            "pilot_decided": stop <= 10 and decision != "unresolved",
            "pilot_incumbent": float(pilot[0].mean()),
            "pilot_discordance": float((pilot[1:] != pilot[0]).mean()),
            "pilot_best_gain": float(pilot_gains.max()),
            "pilot_spread": float(pilot_gains.std())}


def bootstrap(settings, nboot=2000):
    groups = defaultdict(list)
    for s in settings:
        groups[s["task"]].append(s)
    tasks = sorted(groups)
    rng = np.random.default_rng(20261016)
    vals = np.empty((nboot, 2))
    for b in range(nboot):
        picked = rng.choice(tasks, len(tasks), replace=True)
        ss = [s for task in picked for s in groups[task]]
        vals[b] = [np.mean([s["shrinkage"] for s in ss]), np.mean([s["positive_heldout"] for s in ss])]
    return {"shrinkage_ci95": np.quantile(vals[:, 0], [.025, .975]).tolist(),
            "positive_heldout_ci95": np.quantile(vals[:, 1], [.025, .975]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "results/paper/mizrahi_raw")
    ap.add_argument("--out", type=Path, default=ROOT / "results/paper/mizrahi_replay/summary.json")
    a = ap.parse_args()
    source = a.data / "source/data"
    paras = source / "automatic paraphrases"
    accs = source / "paraphrases accuracies"
    archive = next((a.data / "inner").glob("*gold*.zip"))
    task_paths = sorted(paras.rglob("*.csv"), key=lambda p: (p.parent.name, p.stem))
    settings = []
    diagnostics = Counter()
    checks = Counter()
    examples = {}
    exclusions = []
    with zipfile.ZipFile(archive) as z:
        infos = defaultdict(list)
        for info in z.infolist():
            if info.filename.endswith(".json"):
                parts = Path(info.filename).parts
                infos[(parts[1], parts[2])].append(info)
        for task_index, path in enumerate(task_paths):
            bench, task = path.parent.name, path.stem
            inc, chal = flags(path)
            ids = [inc] + chal
            agg_path = accs / bench / (task + ".csv")
            if not agg_path.exists():
                diagnostics["missing_aggregate_task"] += 1
                continue
            agg = {int(r["paraphrase idx"]): r for r in rows(agg_path)}
            for info in infos.get((bench, task), []):
                model = model_from_member(info.filename, task, bench)
                if model is None or model + "_acc" not in next(iter(agg.values())):
                    diagnostics["missing_aggregate_model"] += 1
                    exclusions.append({"task": bench + "/" + task, "model": model, "reason": "missing_aggregate_model"})
                    continue
                raw = outcomes(z, info, ids)
                valid_ids = [x for x, vals in raw.items() if all(v is not None for v in vals)]
                if len(valid_ids) < 40:
                    diagnostics["insufficient_common_examples"] += 1
                    exclusions.append({"task": bench + "/" + task, "model": model, "reason": "insufficient_common_examples", "n": len(valid_ids)})
                    continue
                yy = np.array([raw[x] for x in sorted(valid_ids)], dtype=np.int8).T
                # Published accuracies use all item responses; compare against the full
                # raw matrix before removing incomplete examples.
                mismatch = False
                for j, prompt in enumerate(ids):
                    raw_values = [v[j] for v in raw.values() if v[j] is not None]
                    published = agg.get(prompt, {}).get(model + "_acc", "")
                    if not raw_values or published == "" or abs(np.mean(raw_values) - float(published)) > .01000001:
                        mismatch = True
                        checks["row_mismatch"] += 1
                    else:
                        checks["row_match"] += 1
                if mismatch:
                    diagnostics["aggregate_mismatch_setting"] += 1
                    exclusions.append({"task": bench + "/" + task, "model": model, "reason": "aggregate_mismatch"})
                    continue
                s = analyze(yy, task_index)
                s.update({"task": bench + "/" + task, "model": model, "incumbent": inc,
                          "selected_challenger": chal[s.pop("selected_row") - 1],
                          "n_examples": len(valid_ids)})
                settings.append(s)
                examples[s["task"]] = len(valid_ids)
            print(f"{bench}/{task}: {sum(s['task'] == bench + '/' + task for s in settings)} settings", flush=True)
    distinct_models = sorted({s["model"] for s in settings})
    feasible = len(examples) >= 10 and len(distinct_models) >= 8
    result = {"source_revision": "04f57ef4d5201ef817f54f0e1b09e3fc0c481fb6",
              "feasible": feasible, "tasks": len(examples), "models": len(distinct_models),
              "settings": len(settings), "diagnostics": dict(diagnostics),
              "aggregate_checks": dict(checks), "task_examples": examples, "exclusions": exclusions}
    if feasible:
        decisions = Counter(s["decision"] for s in settings)
        result.update({"decisions": dict(decisions),
                       "mean_search_gain": float(np.mean([s["search_gain"] for s in settings])),
                       "mean_heldout_gain": float(np.mean([s["heldout_gain"] for s in settings])),
                       "mean_shrinkage": float(np.mean([s["shrinkage"] for s in settings])),
                       "positive_heldout_share": float(np.mean([s["positive_heldout"] for s in settings])),
                       "median_decision_task": float(np.median([s["stop_task"] for s in settings if s["decision"] != "unresolved"])) if any(s["decision"] != "unresolved" for s in settings) else None,
                       "full_episodes": sum(s["full_episodes"] for s in settings),
                       "stopped_episodes": sum(s["stopped_episodes"] for s in settings),
                       **bootstrap(settings)})
    result["setting_rows"] = settings
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "setting_rows"}, indent=2))


if __name__ == "__main__":
    main()
