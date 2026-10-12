"""PromptEval public replay 02; see docs/paper/PUBLIC_REPLAY_PREREG_02.md."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.growing_bandits.audit.public_replay_01 import e1_path, first_cross
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths

DATASET = "PromptEval/PromptEval_MMLU_correctness"
REV = "1639d5ea14c362f6964f260ae81bd903af760187"
OUT = Path("results/paper/public_replay_02")
CACHE = OUT / "parquet"


def get(url: str) -> bytes:
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(min(15 + 10 * attempt, 60))
            elif attempt == 9:
                raise
            else:
                time.sleep(min(0.5 * 2**attempt, 10))
        except OSError:
            if attempt == 9:
                raise
            time.sleep(min(0.5 * 2**attempt, 10))


def tree(path: str = ""):
    suffix = "/" + urllib.parse.quote(path, safe="/") if path else ""
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REV}{suffix}?recursive=false&limit=1000"
    return json.loads(get(url))


def index():
    subjects = sorted(x["path"] for x in tree() if x["type"] == "directory")
    files = []
    for subject in subjects:
        files.extend(x["path"] for x in tree(subject) if x["type"] == "file" and x["path"].endswith(".parquet"))
    return subjects, sorted(files)


def fetch_one(path: str):
    destination = CACHE / path
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REV}/{urllib.parse.quote(path, safe='/')}"
    raw = get(url)
    temp = destination.with_suffix(".part")
    temp.write_bytes(raw)
    temp.replace(destination)


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    subjects, files = index()
    (OUT / "index.json").write_text(json.dumps({"revision": REV, "subjects": subjects, "files": files}, indent=2) + "\n")
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_one, path) for path in files]
        for i, future in enumerate(cf.as_completed(futures), 1):
            future.result()
            if i % 100 == 0:
                print(f"downloaded {i}/{len(files)} matrices", flush=True)
    print("downloaded", len(files), "matrices across", len(subjects), "subjects")


def analyze():
    listing = json.loads((OUT / "index.json").read_text())
    subjects, files = listing["subjects"], listing["files"]
    rows, exclusions = [], []
    for i, path in enumerate(files, 1):
        subject = path.split("/", 1)[0]
        model = path.split("/", 1)[1].removesuffix("-00000-of-00001.parquet")
        try:
            data = pd.read_parquet(CACHE / path)
            y = data.to_numpy()
            if y.shape[0] != 100 or y.shape[1] < 40 or not np.isin(y, [0, 1]).all():
                exclusions.append({"path": path, "reason": f"shape_or_values_{y.shape}"})
                continue
            n = y.shape[1]
            order = np.random.default_rng(np.random.SeedSequence([20261012, 5, subjects.index(subject)])).permutation(n)
            n_search = n // 2
            search, holdout = y[:, order[:n_search]], y[:, order[n_search:]]
            search_rates = search.mean(axis=1)
            winner = 1 + int(np.argmax(search_rates[1:]))
            search_gain = float(search_rates[winner] - search_rates[0])
            held_gain = float(holdout[winner].mean() - holdout[0].mean())
            e_cross = first_cross(e1_path(search))
            arm_futility_paths = futility_log_paths((search[1:].astype(float) - search[0].astype(float))[None], 0.15)[0]
            futility_path = arm_futility_paths.min(axis=0)
            f_cross = first_cross(futility_path)
            arm_f_cross = np.array([first_cross(path) for path in arm_futility_paths])
            drop_pool_cross = int(arm_f_cross.max()) if np.all(arm_f_cross > 0) else 0
            drop_horizon = drop_pool_cross or n_search
            drop_episodes = int(drop_horizon + np.minimum(np.where(arm_f_cross > 0, arm_f_cross, n_search), drop_horizon).sum())
            if e_cross and f_cross:
                decision = "efficacy" if e_cross < f_cross else "futility" if f_cross < e_cross else "tie"
            else:
                decision = "efficacy" if e_cross else "futility" if f_cross else "unresolved"
            rows.append({"subject": subject, "model": model, "n_examples": n,
                         "n_search": n_search, "winner_row": winner,
                         "reference_search": search_rates[0], "winner_search": search_rates[winner],
                         "search_gain": search_gain, "heldout_gain": held_gain,
                         "shrinkage": search_gain - held_gain,
                         "efficacy_cross": e_cross, "futility_cross": f_cross,
                         "drop_futility_cross": drop_pool_cross,
                         "drop_futility_episodes": drop_episodes,
                         "decision": decision, "full_episodes": 100*n_search,
                         "efficacy_stop_episodes": 100*(e_cross or n_search)})
        except Exception as exc:
            exclusions.append({"path": path, "reason": type(exc).__name__ + ": " + str(exc)[:100]})
        if i % 100 == 0:
            print(f"analyzed {i}/{len(files)} matrices", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT / "settings.csv", index=False)
    pd.DataFrame(exclusions, columns=["path", "reason"]).to_csv(OUT / "exclusions.csv", index=False)
    if data.empty:
        raise RuntimeError("No valid matrices")
    groups = [data[data.subject == subject] for subject in sorted(data.subject.unique())]
    rng = np.random.default_rng(20261014)
    boots = []
    for _ in range(2000):
        picked = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[j] for j in picked], ignore_index=True)
        boots.append((sample.shrinkage.mean(), sample.shrinkage.median(),
                      (sample.heldout_gain > 0).mean()))
    boot = np.array(boots)
    summary = {"revision": REV, "settings": len(data), "subjects": data.subject.nunique(),
               "models": data.model.nunique(), "excluded": len(exclusions),
               "mean_shrinkage": data.shrinkage.mean(),
               "mean_shrinkage_ci": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
               "median_shrinkage": data.shrinkage.median(),
               "median_shrinkage_ci": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
               "positive_heldout_share": (data.heldout_gain > 0).mean(),
               "positive_heldout_share_ci": np.quantile(boot[:, 2], [0.025, 0.975]).tolist(),
               "decisions": data.decision.value_counts().to_dict(),
               "double_crossings": int(((data.efficacy_cross > 0) & (data.futility_cross > 0)).sum()),
               "full_episodes": int(data.full_episodes.sum()),
               "efficacy_stopped_episodes": int(data.efficacy_stop_episodes.sum()),
               "drop_futility_certifications": int((data.drop_futility_cross > 0).sum()),
               "drop_futility_episodes": int(data.drop_futility_episodes.sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fetch", "analyze"])
    mode = parser.parse_args().mode
    {"fetch": fetch, "analyze": analyze}[mode]()
