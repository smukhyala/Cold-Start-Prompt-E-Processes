"""Freeze one outcome-blind, with-replacement Linear trial after protocol commit.

No provider calls. Refuses to overwrite an existing registration. A public task
may recur, but each draw receives a distinct ID and fresh executions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import secrets
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.growing_bandits.empirical import collect  # noqa: E402

OUT = ROOT / "data/prospective_live_stopping"
WA = ROOT.parent / "webarena-infinity"
APP = WA / "apps/linear-account-settings"
N = 120
MODEL = "gpt-5.4-mini-2026-03-17"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_tasks(bank: list[dict], seed: int, n: int = N) -> list[dict]:
    """Uniform independent draws with replacement; never filter by outcome."""
    bank = sorted(bank, key=lambda task: task["id"])
    rng = np.random.Generator(np.random.PCG64(seed))
    return [{**bank[int(j)], "source_task_id": bank[int(j)]["id"],
             "id": f"live_{i + 1:03d}", "task_id": f"live_{i + 1:03d}",
             "draw_index": i + 1, "seed_hex": f"{seed:032x}"}
            for i, j in enumerate(rng.integers(0, len(bank), size=n))]


def freeze(protocol_commit: str, out: Path = OUT) -> dict:
    out = out.resolve()
    if out.exists():
        raise ValueError(f"refusing to redraw or replace registration: {out}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != protocol_commit:
        raise ValueError("protocol_commit must be the current commit")
    protocol = ROOT / "docs/paper/LIVE_STOPPING_PROTOCOL.md"
    committed = subprocess.check_output(["git", "show", f"{head}:docs/paper/LIVE_STOPPING_PROTOCOL.md"], cwd=ROOT)
    if committed != protocol.read_bytes():
        raise ValueError("protocol differs from committed registration")
    root_sources = [protocol, Path(__file__),
                    ROOT / "experiments/growing_bandits/empirical/live_stopping.py",
                    ROOT / "experiments/growing_bandits/empirical/live_stopping_runner.py",
                    ROOT / "experiments/growing_bandits/empirical/analyze_live_stopping.py",
                    ROOT / "experiments/growing_bandits/empirical/collect.py",
                    ROOT / "experiments/growing_bandits/empirical/make_pools.py",
                    ROOT / "experiments/growing_bandits/audit/flatness_eprocess.py",
                    ROOT / "experiments/growing_bandits/audit/futility_eprocess.py",
                    ROOT / "src/cold_start/tasks/webarena.py",
                    ROOT / "configs/axes.yaml", ROOT / "configs/template.jinja"]
    for source in root_sources:
        committed_source = subprocess.check_output(
            ["git", "show", f"{head}:{source.relative_to(ROOT)}"], cwd=ROOT)
        if committed_source != source.read_bytes():
            raise ValueError(f"source differs from protocol commit: {source}")
    bank_path = APP / "real-tasks.json"
    bank = json.loads(bank_path.read_text())
    if len(bank) != 120 or len({x["id"] for x in bank}) != 120:
        raise ValueError("expected full 120-task Linear bank")
    prompts = collect.load_prompts(ROOT / "data/prospective_linear", "linear")
    # Entropy is drawn exactly once, only after checking the committed protocol.
    seed = secrets.randbits(128)
    tasks = draw_tasks(bank, seed)
    out.mkdir(parents=True)
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (out / "source_bank.json").write_bytes(bank_path.read_bytes())
    prompt_config = {}
    for role, arm_id in (("baseline", "linear_baseline"), ("challenger", "linear_explorer")):
        p = prompts[arm_id]
        prompt_config[role] = {"arm_id": arm_id, "text": p.text,
                               "sha256": p.sha256, "vector": asdict(p.arm.vector)}
        (out / f"{role}.txt").write_text(p.text)
    sources = [*root_sources,
               WA / "evaluation/tasks.py", WA / "evaluation/agents.py", WA / "evaluation/server.py"]
    import browser_use
    import openai
    bu_root = Path(browser_use.__file__).parent
    sources.extend(bu_root / rel for rel in
                   ("agent/service.py", "agent/views.py", "agent/prompts.py",
                    "llm/openai/chat.py", "llm/openai/serializer.py", "tokens/service.py", "tools/service.py"))
    sources.append(Path(openai.__file__).parent / "_base_client.py")
    sources.extend((bu_root / "agent/system_prompts").glob("*.md"))
    app_files = subprocess.check_output(["git", "ls-files", "apps/linear-account-settings"], cwd=WA, text=True).splitlines()
    sources.extend(WA / rel for rel in app_files if (WA / rel).is_file())
    sources.extend(out.iterdir())
    config = {"study": "Prospective live stopping on iid public-bank Linear draws",
              "registered_utc": dt.datetime.now(dt.UTC).isoformat(),
              "protocol_commit": protocol_commit, "seed_hex": f"{seed:032x}",
              "max_pairs": N, "delta": .15, "alpha_e": .025, "alpha_f": .025,
              "model": MODEL, "max_usd": 75.0, "budget_mode": "transport",
              "minimum_request_reservation_usd": .318432,
              "prompts": prompt_config, "web_app": "apps/linear-account-settings",
              "task_suite": str(out / "tasks"), "port": 8601,
              "max_steps": 30, "timeout_s": 180, "reasoning_effort": "low",
              "max_completion_tokens": 4096, "max_retries": 0,
              "use_vision": False, "headless": True,
              "provider_base_url": "https://api.openai.com/v1",
              "webarena_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WA, text=True).strip(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("browser-use", "openai", "httpx", "numpy", "scipy", "pydantic")},
              "frozen_files": {str(p.resolve()): digest(p) for p in sorted(set(sources))}}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"config_sha256": digest(out / "config.json"),
                "tasks_sha256": digest(out / "tasks.json"),
                "protocol_commit": head, "registered_utc": config["registered_utc"],
                "seed_hex": config["seed_hex"], "n_pairs": N,
                "source_bank_sha256": digest(bank_path)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.protocol_commit, args.out), indent=2))
