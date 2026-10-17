#!/usr/bin/env python3
"""Retrospective dated SWE-bench Verified leaderboard replay; see SWE_LEADERBOARD_PROTOCOL.md."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.growing_bandits.audit.public_replay_01 import e1_path

REV = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
BASE = "evaluation/verified"
CACHE = ROOT / "results/paper/swe_leaderboard_raw"
OUT = ROOT / "results/paper/swe_leaderboard"
ADVERSE = ("no_generation", "install_fail", "reset_failed", "test_errored")


def get_json(url):
    for attempt in range(7):
        try:
            with urllib.request.urlopen(url, timeout=50) as f:
                return json.load(f)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == 6:
                raise
            time.sleep(min(2**attempt, 20))


def get_text(url):
    with urllib.request.urlopen(url, timeout=50) as f:
        return f.read().decode()


def inventory():
    tree = get_json(f"https://api.github.com/repos/SWE-bench/experiments/git/trees/{REV}?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing truncated")
    items = tree["tree"]
    folders = sorted(x["path"].split("/")[2] for x in items
                     if x["type"] == "tree" and x["path"].startswith(BASE + "/")
                     and len(x["path"].split("/")) == 3)
    result_paths = {x["path"] for x in items if x["type"] == "blob" and
                    x["path"].startswith(BASE + "/") and x["path"].endswith("/results/results.json")}
    names = [name for name in folders if f"{BASE}/{name}/results/results.json" in result_paths]
    if len(names) != len(result_paths):
        raise RuntimeError("Unexpected nested results path")
    return {"revision": REV, "folders": folders, "result_names": names,
            "missing_result_names": sorted(set(folders) - set(names))}


def fetch_one(name):
    path = CACHE / (name + ".json")
    if path.exists():
        return
    url = f"https://raw.githubusercontent.com/SWE-bench/experiments/{REV}/{BASE}/{name}/results/results.json"
    data = get_json(url)
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(data))
    temp.replace(path)


def fetch_metadata(name):
    path = CACHE / (name + ".yaml")
    if path.exists():
        return
    base = f"https://raw.githubusercontent.com/SWE-bench/experiments/{REV}/{BASE}/{name}/metadata"
    try:
        raw = get_text(base + ".yaml")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        raw = get_text(base + ".yml")
    path.write_text(raw)


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    inv = inventory()
    (CACHE / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n")
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(fetch_one, inv["result_names"]), 1):
            if i % 25 == 0:
                print(f"cached {i}/{len(inv['result_names'])} result files", flush=True)
        list(pool.map(fetch_metadata, inv["missing_result_names"]))
    print(f"folders={len(inv['folders'])}, result files={len(inv['result_names'])}")


def parse_result(name, coverage):
    data = json.loads((CACHE / (name + ".json")).read_text())
    if not isinstance(data, dict) or "resolved" not in data:
        raise ValueError("missing_resolved_status")
    for key, vals in data.items():
        if not isinstance(vals, list) or not all(isinstance(v, str) for v in vals):
            raise ValueError(f"malformed_status_{key}")
    ids = set().union(*(set(v) for v in data.values()))
    resolved = set(data["resolved"])
    adverse = set().union(*(set(data.get(k, [])) for k in ADVERSE))
    if resolved & adverse:
        raise ValueError("resolved_overlaps_adverse")
    if coverage == "strict" and len(ids) < 490:
        raise ValueError(f"fewer_than_490_ids_{len(ids)}")
    duplicates = sum(len(v) - len(set(v)) for v in data.values())
    return {"name": name, "ids": ids, "resolved": resolved, "duplicates": duplicates,
            "reported_ids": len(ids)}


def first_cross(path, alpha):
    hits = np.flatnonzero(path >= np.log(1 / alpha))
    return int(hits[0] + 1) if len(hits) else None


def analyze(coverage):
    inv = json.loads((CACHE / "inventory.json").read_text())
    eligible, excluded = [], []
    for name in inv["result_names"]:
        try:
            eligible.append(parse_result(name, coverage))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            excluded.append({"name": name, "reason": str(exc)})
    eligible.sort(key=lambda r: (r["name"][:8], r["name"]))
    universe = sorted(set().union(*(r["ids"] for r in eligible)))
    if len(universe) != 500:
        raise RuntimeError(f"Expected exactly 500 Verified IDs; found {len(universe)}")
    universe_set = set(universe)
    for row in eligible:
        if not row["ids"] <= universe_set:
            raise RuntimeError(f"Out-of-universe IDs in {row['name']}")
        row["missing_ids"] = 500 - len(row["ids"])
        row["score"] = len(row["resolved"] & universe_set)
        row["y"] = np.array([int(t in row["resolved"]) for t in universe], dtype=np.int8)
    if len(eligible) < 2:
        raise RuntimeError("Fewer than two eligible entries")
    leader = eligible[0]
    comparisons = []
    for j, entrant in enumerate(eligible[1:], 1):
        y = np.stack([leader["y"], entrant["y"]])
        rng = np.random.default_rng(np.random.SeedSequence([20261012, 44, j]))
        path = e1_path(y[:, rng.permutation(500)])
        nominal = first_cross(path, .05)
        spent_alpha = .05 / (j * (j + 1))
        spent = first_cross(path, spent_alpha)
        gain = int(entrant["score"] - leader["score"])
        claim = gain > 0
        comparisons.append({"entrant_index": j, "directory_date": entrant["name"][:8],
                            "entrant": entrant["name"], "prior_leader": leader["name"],
                            "entrant_resolved": entrant["score"], "leader_resolved": leader["score"],
                            "reported_gain_tasks": gain, "paired_gain": gain / 500,
                            "leader_claim": claim,
                            "discordant_wins": int(((y[1] == 1) & (y[0] == 0)).sum()),
                            "discordant_losses": int(((y[1] == 0) & (y[0] == 1)).sum()),
                            "nominal_cross_task": nominal, "spent_cross_task": spent,
                            "spent_alpha": spent_alpha, "max_log_e": float(path.max()),
                            "final_log_e": float(path[-1]),
                            "full_grid_episodes": 1000,
                            "nominal_stop_episodes": 2 * (nominal or 500),
                            "spent_stop_episodes": 2 * (spent or 500),
                            "entrant_missing_ids": entrant["missing_ids"],
                            "leader_missing_ids": leader["missing_ids"]})
        if claim:
            leader = entrant
    claims = [r for r in comparisons if r["leader_claim"]]
    # Post hoc chronology sensitivity: the source dates only to a day, so the
    # lexicographic within-day order is a deterministic convention.
    day_groups = defaultdict(list)
    for row in eligible:
        day_groups[row["name"][:8]].append(row)
    tie_rng = np.random.default_rng(20261012)
    same_day_counts = []
    for _ in range(2000):
        highest, n_claims = -1, 0
        for day in sorted(day_groups):
            group = day_groups[day].copy()
            tie_rng.shuffle(group)
            for row in group:
                if row["score"] > highest:
                    n_claims += highest >= 0
                    highest = row["score"]
        same_day_counts.append(n_claims)
    metadata_only = []
    for name in inv["missing_result_names"]:
        path = CACHE / (name + ".yaml")
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        value = (data.get("info") or {}).get("resolved")
        if isinstance(value, (float, int)):
            metadata_only.append({"name": name, "resolved_percent": float(value)})
    # Score-only audit: metadata-only entries cannot enter itemwise tests, but
    # they might have displaced the leader before a later result-file entrant.
    score_rows = {r["name"]: (r["score"] / 5, "result") for r in eligible}
    score_rows.update({r["name"]: (r["resolved_percent"], "metadata") for r in metadata_only})
    score_leader, score_best = None, -1.0
    metadata_leader_claims = 0
    result_with_metadata_leader = 0
    for name in sorted(score_rows):
        score, kind = score_rows[name]
        if kind == "result" and score_leader is not None and score_rows[score_leader][1] == "metadata":
            result_with_metadata_leader += 1
        if score > score_best:
            score_best, score_leader = score, name
            if kind == "metadata":
                metadata_leader_claims += 1
    nominal_hits = [r["nominal_cross_task"] for r in claims if r["nominal_cross_task"] is not None]
    spent_hits = [r["spent_cross_task"] for r in claims if r["spent_cross_task"] is not None]
    summary = {"revision": REV, "coverage_policy": coverage, "folders": len(inv["folders"]),
               "with_results": len(inv["result_names"]),
               "missing_results": len(inv["missing_result_names"]),
               "eligible": len(eligible), "excluded": len(excluded),
               "exclusion_reasons": dict(Counter(r["reason"].split("_")[0] for r in excluded)),
               "initial_leader": eligible[0]["name"], "final_leader": leader["name"],
               "universe_ids": len(universe), "eligible_with_missing_ids": sum(r["missing_ids"] > 0 for r in eligible),
               "max_missing_ids": max(r["missing_ids"] for r in eligible),
               "duplicate_status_records": sum(r["duplicates"] for r in eligible),
               "metadata_only_with_score": len(metadata_only),
               "metadata_only_max_percent": max((r["resolved_percent"] for r in metadata_only), default=None),
               "metadata_only_at_least_final_subset_leader": sum(r["resolved_percent"] >= leader["score"] / 5
                                                                 for r in metadata_only),
               "metadata_only_leader_claims": metadata_leader_claims,
               "result_entries_with_metadata_prior_leader": result_with_metadata_leader,
               "score_only_final_leader": score_leader,
               "comparisons": len(comparisons), "leader_claims": len(claims),
               "same_day_order_claim_count_range": [min(same_day_counts), max(same_day_counts)],
               "same_day_order_claim_count_median": float(np.median(same_day_counts)),
               "nominal_crossing_claims": len(nominal_hits), "spent_crossing_claims": len(spent_hits),
               "nominal_median_cross_task": float(np.median(nominal_hits)) if nominal_hits else None,
               "spent_median_cross_task": float(np.median(spent_hits)) if spent_hits else None,
               "nominal_claim_episode_saving": sum(1000 - r["nominal_stop_episodes"] for r in claims),
               "spent_claim_episode_saving": sum(1000 - r["spent_stop_episodes"] for r in claims),
               "nonclaim_nominal_crossings": sum(r["nominal_cross_task"] is not None for r in comparisons if not r["leader_claim"]),
               "missing_result_names": inv["missing_result_names"],
               "metadata_only_rows": metadata_only, "exclusions": excluded,
               "comparison_rows": comparisons}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (coverage + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ("missing_result_names", "metadata_only_rows", "exclusions", "comparison_rows")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("fetch", "analyze"))
    ap.add_argument("--coverage", choices=("strict", "official"), default="strict")
    args = ap.parse_args()
    fetch() if args.mode == "fetch" else analyze(args.coverage)
