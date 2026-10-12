"""Frozen Terminal-Bench public replay 01; see docs/paper/PUBLIC_REPLAY_PREREG_01.md.

Fetches only trial result.json files. No agent trajectories or large LFS artifacts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths

REV = "572b2614be2c0cb2527e14f5b1e4026f1072e6c1"
DATASET = "harborframework/terminal-bench-2-leaderboard"
ROOT = "submissions/terminal-bench/2.0"
ARMS = (
    "Terminus2__Claude-Opus-4.6",
    "Capy__Claude-Opus-4.6",
    "Crux__Claude-Opus-4.6",
    "Droid__Claude-Opus-4.6",
    "Terminus-KIRA__Claude-Opus-4.6",
    "WozCode__Claude-Opus-4.6",
)
RETAINED = (ARMS[0], ARMS[1], ARMS[4])
OUT = Path("results/paper/public_replay_01")


def get_json(url: str):
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(min(15 + 10 * attempt, 60))
            elif attempt == 9:
                raise
            else:
                time.sleep(min(0.5 * 2**attempt, 10))
        except (OSError, json.JSONDecodeError):
            if attempt == 9:
                raise
            time.sleep(min(0.5 * 2**attempt, 10))


def tree(path: str):
    quoted = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REV}/{quoted}?recursive=false&limit=1000"
    rows = get_json(url)
    if len(rows) == 1000:
        raise RuntimeError(f"Possible truncated tree: {path}")
    return rows


def result(path: str):
    quoted = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REV}/{quoted}/result.json"
    return get_json(url)


def index_paths():
    out = defaultdict(list)
    for arm in ARMS:
        root = f"{ROOT}/{arm}"
        for job in tree(root):
            if job["type"] != "directory":
                continue
            for trial in tree(job["path"]):
                if trial["type"] != "directory":
                    continue
                task = trial["path"].rsplit("/", 1)[-1].split("__", 1)[0]
                out[(arm, task)].append(trial["path"])
    return {k: sorted(v) for k, v in out.items()}


def valid_record(path: str, obj: dict):
    cfg = obj.get("config") or {}
    agent = cfg.get("agent") or {}
    model = str(agent.get("model_name", "")).lower().replace(".", "-")
    if model not in ("anthropic/claude-opus-4-6", "vertex_ai/claude-opus-4-6"):
        return None, "model"
    keys = ("override_timeout_sec", "max_timeout_sec", "override_setup_timeout_sec")
    if any(agent.get(k) is not None for k in keys):
        return None, "timeout_override"
    for section in ("task", "environment", "verifier"):
        details = cfg.get(section) or {}
        if any(v is not None for k, v in details.items() if k.startswith("override_") or k.startswith("max_timeout")):
            return None, "resource_or_timeout_override"
    if obj.get("exception_info"):
        return None, "exception"
    reward = ((obj.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if isinstance(reward, bool) or reward not in (0, 0.0, 1, 1.0):
        return None, "nonbinary_or_missing_reward"
    task = path.rsplit("/", 1)[-1].split("__", 1)[0]
    if obj.get("task_name") != task:
        return None, "task_mismatch"
    return {"path": path, "task": task, "reward": int(reward), "trial_id": obj.get("id")}, None


def collect_one(item):
    (arm, task), paths = item
    kept, reasons = [], Counter()
    seen = set()
    for path in paths:
        try:
            row, reason = valid_record(path, result(path))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise
            row, reason = None, f"http_{exc.code}"
        except Exception:
            row, reason = None, "download_or_parse_error"
        if reason:
            reasons[reason] += 1
            continue
        if row["trial_id"] in seen:
            reasons["duplicate_trial_id"] += 1
            continue
        seen.add(row["trial_id"])
        row.update(arm=arm, replicate=len(kept))
        kept.append(row)
        if len(kept) == 2:
            break
    if len(kept) < 2:
        reasons["fewer_than_two_valid"] += 1
    return kept, {"arm": arm, "task": task, **reasons}


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = index_paths()
    jobs = sorted(paths.items())
    cache = OUT / "cells.jsonl"
    done = set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            item = json.loads(line)
            done.add((item["arm"], item["task"]))
    pending = [(key, value) for key, value in jobs if key not in done]
    with cf.ThreadPoolExecutor(max_workers=4) as pool, cache.open("a") as handle:
        futures = {pool.submit(collect_one, item): item[0] for item in pending}
        for i, future in enumerate(cf.as_completed(futures), 1):
            key = futures[future]
            got, count = future.result()
            handle.write(json.dumps({"arm": key[0], "task": key[1], "records": got,
                                     "exclusions": count}) + "\n")
            handle.flush()
            if i % 50 == 0:
                print(f"fetched {i}/{len(pending)} remaining arm-task cells", flush=True)
    cells = [json.loads(line) for line in cache.read_text().splitlines()]
    records = [row for cell in cells for row in cell["records"]]
    pd.DataFrame(records).to_csv(OUT / "trials.csv", index=False)
    pd.DataFrame([cell["exclusions"] for cell in cells]).fillna(0).to_csv(OUT / "exclusions.csv", index=False)
    print("records", len(records), "cells", len(cells), "source", REV)


def e1_path(y):
    a, n = y.shape
    inc = np.array([0])
    chal = np.ones((1, a), dtype=bool)
    chal[:, 0] = False
    return e1b_log_paths(y[None], inc, chal)[0]


def first_cross(log_path, alpha=0.05):
    hits = np.flatnonzero(log_path >= np.log(1 / alpha))
    return int(hits[0] + 1) if len(hits) else 0


def analyze():
    df = pd.read_csv(OUT / "trials.csv")
    df = df[df.arm.isin(RETAINED)]
    counts = df.groupby(["arm", "task"]).replicate.nunique()
    tasks = sorted(t for t in df.task.unique() if all(counts.get((arm, t), 0) == 2 for arm in RETAINED))
    if len(tasks) < 40:
        raise RuntimeError(f"Only {len(tasks)} complete tasks; feasibility failure")
    y = np.empty((2, len(RETAINED), len(tasks)), dtype=np.int8)
    for row in df.itertuples():
        if row.task in tasks:
            y[int(row.replicate), RETAINED.index(row.arm), tasks.index(row.task)] = int(row.reward)
    bootstrap = np.random.default_rng(20261013).integers(0, len(tasks), (10000, len(tasks)))
    gain_rows = []
    for ai, arm in enumerate(RETAINED):
        diff = y[0, ai].astype(float) - y[0, 0]
        draws = diff[bootstrap].mean(axis=1)
        gain_rows.append({"arm": arm, "trial0_mean": y[0, ai].mean(),
                          "trial1_mean": y[1, ai].mean(), "trial0_gain": diff.mean(),
                          "trial0_gain_ci_low": np.quantile(draws, 0.025),
                          "trial0_gain_ci_high": np.quantile(draws, 0.975),
                          "trial1_gain": (y[1, ai] - y[1, 0]).mean()})
    pd.DataFrame(gain_rows).to_csv(OUT / "gains.csv", index=False)
    rows = []
    for b in range(2000):
        rng = np.random.default_rng(np.random.SeedSequence([20261012, 1, b]))
        aa = y[:, 0].copy()
        swaps = rng.integers(0, 2, len(tasks)).astype(bool)
        aa[:, swaps] = aa[::-1, swaps]
        order = rng.permutation(len(tasks))
        rows.append({"replicate": b, "cross_task": first_cross(e1_path(aa[:, order])),
                     "discordance": int(np.count_nonzero(aa[0] != aa[1]))})
    pd.DataFrame(rows).to_csv(OUT / "r1_aa.csv", index=False)
    replay = []
    first_order = None
    for b in range(2000):
        rng = np.random.default_rng(np.random.SeedSequence([20261012, 3, b]))
        order = rng.permutation(len(tasks))
        if b == 0:
            first_order = order
        c = first_cross(e1_path(y[0, :, order]))
        replay.append({"replicate": b, "cross_task": c, "episodes": (len(RETAINED) * (c or len(tasks)))})
    pd.DataFrame(replay).to_csv(OUT / "r3_replay.csv", index=False)
    r1_hits = sum(row["cross_task"] > 0 for row in rows)
    r3_hits = sum(row["cross_task"] > 0 for row in replay)
    r1_ci = binomtest(r1_hits, len(rows)).proportion_ci(confidence_level=0.95)
    first_crossing = replay[0]["cross_task"]
    confirmation = {"primary_cross_task": first_crossing, "selected_arm": None,
                    "rerun_cross_task": 0, "rerun_gain": None}
    if first_crossing:
        individual = [e1_path(y[0, [0, ai]][:, first_order]) for ai in range(1, len(RETAINED))]
        selected_ai = 1 + int(np.argmax([path[first_crossing - 1] for path in individual]))
        confirmation.update(selected_arm=RETAINED[selected_ai],
                            rerun_cross_task=first_cross(e1_path(y[1, [0, selected_ai]][:, first_order])),
                            rerun_gain=float((y[1, selected_ai] - y[1, 0]).mean()))
    # Secondary futility: every arm on every task, delta=0.15.
    d = y[0, 1:] - y[0, :1]
    flogs = futility_log_paths(d[None].astype(float), 0.15)[0]
    pd.DataFrame({"arm": RETAINED[1:], "futility_first_cross":
                  [first_cross(flogs[k]) for k in range(len(RETAINED)-1)]}).to_csv(OUT / "r3_futility.csv", index=False)
    summary = {"revision": REV, "tasks": len(tasks), "arms": list(RETAINED),
               "r1_crossings": r1_hits, "r1_replays": len(rows),
               "r1_fraction": r1_hits / len(rows),
               "r1_binomial_ci": [r1_ci.low, r1_ci.high],
               "r1_unswapped_cross_tasks": [first_cross(e1_path(y[[0, 1], 0])),
                                             first_cross(e1_path(y[[1, 0], 0]))],
               "r3_crossings": r3_hits, "r3_replays": len(replay),
               "r3_fraction": r3_hits / len(replay),
               "r3_median_cross_task_among_hits": float(np.median([r["cross_task"] for r in replay if r["cross_task"]])) if r3_hits else None,
               "r3_full_grid_episodes": len(RETAINED) * len(tasks),
               "r3_mean_episodes": float(np.mean([r["episodes"] for r in replay])),
               "confirmation": confirmation}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fetch", "analyze"])
    args = parser.parse_args()
    {"fetch": fetch, "analyze": analyze}[args.mode]()
