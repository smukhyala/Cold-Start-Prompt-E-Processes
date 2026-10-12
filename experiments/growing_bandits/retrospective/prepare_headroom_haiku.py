"""Freeze Pre-registration 16 (Gmail answer-key headroom under Claude Haiku 4.5) without launching any paid episode.

    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_headroom_haiku.py --dry-run
    ./.venv/bin/python experiments/growing_bandits/retrospective/prepare_headroom_haiku.py

Same arms, tasks and design as Pre-registration 15 (``prepare_headroom_gmail``, frozen and reused, not modified):
O = ``GM_oracle_full`` and D = ``GM_anchor_baseline`` x the 60 Gmail real tasks x replicates {0, 1} = 240 episodes,
profile ``gmail`` (180 s clock, 30-step cap). The only differences: the agent's model
(``claude-haiku-4-5-20251001``, provider anthropic, with browser-use's ``use_thinking`` off; with it on, Haiku inlines
its action JSON into the thinking field and fails browser-use's output schema) and a fresh queue seed (20261014).
The manifest's ``agent_overrides`` carries the model; ``collect.agent_settings`` applies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import prepare_headroom_gmail as phg  # noqa: E402
from make_pools import QueueItem  # noqa: E402

OUT = ROOT / "data" / "headroom_haiku_gmail"
LOG_NAME = "headroom_haiku_gmail"
SEED = 20261014
AGENT_OVERRIDES = {"llm_provider": "anthropic", "llm_model": "claude-haiku-4-5-20251001",
                   "llm_reasoning_effort": None, "use_thinking": False}
COST_O_USD = 0.10
COST_D_USD = 0.08
BUDGET_USD = 35
STUDY = ("Pre-registration 16 -- Gmail answer-key headroom under Claude Haiku 4.5: full answer key (O) vs the "
         "default prompt (D), 60 tasks")


def build_queue(task_ids: list[str]) -> list[QueueItem]:
    """`prepare_headroom_gmail.build_queue`'s scheme with this study's seed."""
    rng = np.random.default_rng(SEED)
    task_order = rng.permutation(sorted(task_ids))
    items = []
    for task in task_order:
        episodes = []
        for pool, arm in ((phg.O_POOL, phg.O_ARM), (phg.D_POOL, phg.D_ARM)):
            for rep in phg.REPLICATES:
                episodes.append((pool, arm, rep))
        order = rng.permutation(len(episodes))
        for k in order:
            pool, arm, rep = episodes[int(k)]
            items.append((pool, arm, str(task), rep))
    queue = []
    for i in range(len(items)):
        pool, arm, task, rep = items[i]
        queue.append(QueueItem(index=i, pool=pool, arm_id=arm, task_id=task, replicate=rep, pilot=False))
    return queue


def prepare(out: Path = OUT) -> dict:
    out = Path(out)
    o_bytes = phg.O_SOURCE.read_bytes()
    if o_bytes != (ROOT / "data" / "headroom_gmail" / phg.O_REL).read_bytes():
        raise ValueError("answer-key pool differs from Pre-reg 15's frozen copy; refusing")
    if phg.rebuilt_oracle_bytes() != o_bytes:
        raise ValueError("answer-key pool differs from a fresh build_gmail_oracle build; refusing")
    d_bytes = phg.D_SOURCE.read_bytes()
    phg.check_arms(o_bytes, d_bytes)
    task_ids = []
    for task in phg.bgo.load_bank():
        task_ids.append(str(task["id"]))
    queue = build_queue(task_ids)
    qtext = phg.queue_text(queue)
    files = {phg.QUEUE_REL: hashlib.sha256(qtext.encode()).hexdigest(),
             phg.O_REL: hashlib.sha256(o_bytes).hexdigest(),
             phg.D_REL: hashlib.sha256(d_bytes).hexdigest()}
    manifest = phg.build_manifest(queue, files, o_bytes)
    estimate = 0.0
    for item in queue:
        if item.arm_id == phg.O_ARM:
            estimate += COST_O_USD
        else:
            estimate += COST_D_USD
    agent = dict(manifest["agent"])
    for key in AGENT_OVERRIDES:
        agent[key] = AGENT_OVERRIDES[key]
    manifest.update({
        "study": STUDY,
        "design": ("O (GM_oracle_full) and D (GM_anchor_baseline) x the 60 Gmail real tasks x replicates {0, 1}; "
                   "seeded task order and within-task episode order; isolated resets; profile gmail (180 s), 30-step "
                   "cap; agent claude-haiku-4-5-20251001 with use_thinking off"),
        "seed": SEED,
        "agent": agent,
        "agent_overrides": AGENT_OVERRIDES,
        "cost_per_episode_assumed_usd": {"O": COST_O_USD, "D": COST_D_USD},
        "cost_estimate_usd": round(estimate, 2),
        "budget_usd": BUDGET_USD,
        "generator_sha256": {"build_gmail_oracle.py": phg.sha(Path(phg.bgo.__file__)),
                             "prepare_headroom_gmail.py": phg.sha(Path(phg.__file__)),
                             "prepare_headroom_haiku.py": phg.sha(Path(__file__))},
        "comparison": {"pre_reg_15_manifest_sha256": phg.sha(ROOT / "data" / "headroom_gmail" / "manifest.json")},
    })
    contents = {phg.QUEUE_REL: qtext.encode(), phg.O_REL: o_bytes, phg.D_REL: d_bytes,
                "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    for rel in contents:
        dest = out / rel
        if dest.exists() and dest.read_bytes() != contents[rel]:
            raise ValueError(f"existing {rel} differs; refusing to overwrite")
    for rel in contents:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(contents[rel])
    return {"queue": queue, "manifest": manifest}


def launch_command() -> str:
    rel = OUT.relative_to(ROOT)
    return (f"caffeinate -dims ./.venv/bin/python experiments/growing_bandits/empirical/collect.py --profile "
            f"{phg.PROFILE} --data {rel} --log-dir logs/{LOG_NAME} --budget {BUDGET_USD} --workers {phg.WORKERS}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build + validate in a temp dir; write nothing")
    args = ap.parse_args(argv)
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="headroom_haiku_dry_"))
        try:
            result = prepare(tmp / "data")
            for line in phg.validate(tmp / "data"):
                print(line)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print("DRY RUN (nothing written)")
    else:
        result = prepare(OUT)
        for line in phg.validate(OUT):
            print(line)
    m = result["manifest"]
    print(f"{m['n_episodes']} episodes on {m['n_tasks']} tasks; estimate ${m['cost_estimate_usd']}; "
          f"budget ${m['budget_usd']}")
    print(launch_command())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
