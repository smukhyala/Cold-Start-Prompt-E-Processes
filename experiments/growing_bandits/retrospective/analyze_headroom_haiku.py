"""Pre-registration 16: Gmail answer-key headroom under Claude Haiku 4.5 (frozen analyzer).

    PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_headroom_haiku.py
    PYTHONPATH=. ./.venv/bin/python experiments/growing_bandits/retrospective/analyze_headroom_haiku.py --synthetic

PRIMARY: exactly Pre-registration 15's (``analyze_headroom_gmail.analyze``, frozen, reused unmodified) on this study's
data and logs: H = mean over the 60 tasks of O_t - D_t (30-step rescore, mean of 2 replicates), paired t, two-sided
90% CI; large iff lower bound >= 0.10, small iff upper bound < 0.10, else unresolved.

SECONDARY (reported whatever the outcome):
    1. Pre-reg 15's secondaries as that analyzer computes them (bootstrap, Leg M capture/drift, operations).
    2. Model contrast: dH_t = (O_t - D_t) under Haiku minus (O_t - D_t) under gpt-5.4-mini (Pre-reg 15), paired over
       the 60 tasks, mean with 90% CI. Cross-session (descriptive; Pre-reg 15 showed +15 pp session drift in D).
    3. Arm levels by model: O and D means under each model.
    4. Key use (manipulation check), per model, over the answer-key arm's episodes:
       a. reference rate: the episode's serialized model outputs match /answer[ -]key|the key\\b/i;
       b. direct-URL rate: they match /#\\/email\\/\\d|#\\/settings\\//;
       with the arm's success rate within each group. Haiku runs with use_thinking off (no thinking field), so (a) is
       less sensitive for Haiku than for gpt-5.4-mini; (b) is behavioural and comparable.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT))

import analyze_headroom_gmail as ahg  # noqa: E402
import analyze_winner_check as awc  # noqa: E402

DATA = ROOT / "data" / "headroom_haiku_gmail"
LOGS = ROOT / "logs" / "headroom_haiku_gmail"
RESULTS = ROOT / "results" / "headroom_haiku_gmail"
MINI_RESULTS_TABLE = ROOT / "results" / "headroom_gmail" / "task_table.csv"
MINI_DATA = ROOT / "data" / "headroom_gmail"
MINI_LOGS = ROOT / "logs" / "headroom_gmail"
O_ARM = "GM_oracle_full"
REFERENCE = re.compile(r"answer[ -]key|the key\b", re.IGNORECASE)
DIRECT_URL = re.compile(r"#/email/\d|#/settings/")


def key_use(log_dir: Path) -> dict:
    """Manipulation check over the answer-key arm's terminal episodes (artifacts/w*/<arm>/<task>/r<rep>_a<attempt>)."""
    scored = {}
    for path in sorted(Path(log_dir).glob("worker_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("arm_id") != O_ARM or rec.get("status") != "ok":
                continue
            ok = int(bool(rec["success"]) and not bool(rec["timed_out"]) and int(rec["steps"]) <= ahg.SCORE_CAP)
            scored[(str(rec["task_id"]), int(rec["replicate"]), int(rec["attempt"]))] = ok
    groups = {"references_key": [], "no_reference": [], "direct_url": [], "no_direct_url": []}
    pattern = str(Path(log_dir) / "artifacts" / "w*" / O_ARM / "*" / "r*_a*" / "history.json")
    for hist in sorted(glob.glob(pattern)):
        parts = Path(hist).parts
        task = parts[-3]
        tag = parts[-2]
        rep = int(tag[1:tag.index("_")])
        attempt = int(tag[tag.index("_a") + 2:])
        key = (task, rep, attempt)
        if key not in scored:
            continue
        doc = json.loads(Path(hist).read_text())
        items = doc.get("history", doc) if isinstance(doc, dict) else doc
        outputs = []
        for step in items:
            outputs.append(step.get("model_output") or {})
        text = json.dumps(outputs)
        if REFERENCE.search(text):
            groups["references_key"].append(scored[key])
        else:
            groups["no_reference"].append(scored[key])
        if DIRECT_URL.search(text):
            groups["direct_url"].append(scored[key])
        else:
            groups["no_direct_url"].append(scored[key])
    out = {}
    total = len(groups["references_key"]) + len(groups["no_reference"])
    out["episodes"] = total
    for name in groups:
        vals = groups[name]
        out[name] = {"n": len(vals), "rate": (len(vals) / total) if total else None,
                     "success": (float(np.mean(vals)) if vals else None)}
    return out


def read_table(path: Path) -> dict:
    import csv
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["task_id"]] = {"O": float(row["O"]), "D": float(row["D"]), "H": float(row["O_minus_D"])}
    return out


def analyze(data_dir: Path = DATA, log_dir: Path = LOGS, out_dir: Path = RESULTS, allow_synthetic: bool = False,
            mini_table: Path = MINI_RESULTS_TABLE, mini_logs: Path = MINI_LOGS) -> dict:
    base = ahg.analyze(data_dir, log_dir, out_dir, allow_synthetic=allow_synthetic)
    haiku = read_table(Path(out_dir) / "task_table.csv")
    contrast = {"available": False}
    levels = {"haiku": {"O": base["primary"]["arm_means"]["O"], "D": base["primary"]["arm_means"]["D"]}}
    if Path(mini_table).exists():
        mini = read_table(mini_table)
        diffs = []
        o_m = []
        d_m = []
        for task in haiku:
            diffs.append(haiku[task]["H"] - mini[task]["H"])
            o_m.append(mini[task]["O"])
            d_m.append(mini[task]["D"])
        t = awc.t_summary(np.asarray(diffs), ahg.ALPHA)
        contrast = {"available": True, "quantity": "mean over tasks of H_haiku_t - H_mini_t (cross-session)",
                    "mean": t["mean"], "se": t["se"], "interval_two_sided_90": t["interval_two_sided_90"]}
        levels["gpt-5.4-mini"] = {"O": float(np.mean(o_m)), "D": float(np.mean(d_m))}
    extra = {"synthetic": bool(allow_synthetic),
             "model_contrast_H": contrast,
             "arm_levels_by_model": levels,
             "key_use": {"haiku": key_use(log_dir) if not allow_synthetic else {"skipped": "synthetic"},
                         "gpt-5.4-mini": key_use(mini_logs) if Path(mini_logs).is_dir() else {"available": False}}}
    (Path(out_dir) / "secondary_haiku.json").write_text(json.dumps(extra, indent=2) + "\n")
    return {"primary": base["primary"], "secondary": base["secondary"], "secondary_haiku": extra}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="fabricated outcomes in a temp dir (pipeline check)")
    args = ap.parse_args(argv)
    if args.synthetic:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="headroom_haiku_synthetic_"))
        ahg.make_synthetic_log(DATA, tmp / "logs")
        result = analyze(DATA, tmp / "logs", tmp / "results", allow_synthetic=True)
    else:
        result = analyze()
    print(json.dumps(result["primary"], indent=2))
    print(json.dumps(result["secondary_haiku"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
