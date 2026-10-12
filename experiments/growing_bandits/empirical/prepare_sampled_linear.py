"""Draw a fresh iid Linear profile-edit task sample after the protocol is committed.

The task distribution and verifier code are fixed in this source. Pass a newly drawn
128-bit hex seed; the seed, realized tasks, verifiers, prompts, and queue are then frozen
before any agent episode. An optional --out supports outcome-blind harness checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect  # noqa: E402
import make_pools  # noqa: E402

PROFILE = "sampled_linear"
OUT = ROOT / "data/prospective_sampled_linear"
APP = ROOT.parent / "webarena-infinity/apps/linear-account-settings"
POOL_SOURCE = ROOT / "data/prospective_linear/pools/fixed_pair.yaml"
POOL_REL = "pools/fixed_pair.yaml"
QUEUE_REL = "sampled_linear/queue.jsonl"
FAMILIES = ("name_username", "username_email", "name_email", "all_three")
FIRST = ("Mira", "Noah", "Leila", "Owen", "Amara", "Eli", "Nadia", "Theo", "Rina", "Caleb", "Tara", "Hugo")
LAST = ("Chen", "Patel", "Rivera", "Kim", "Okafor", "Shah", "Bennett", "Alvarez", "Ibrahim", "Sato", "Nolan", "Park")
N = 120
ARMS = ("linear_baseline", "linear_explorer")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_once(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"refusing to replace frozen file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def draw_task(rng: np.random.Generator, index: int, out: Path) -> tuple[dict, bytes]:
    """Draw one task independently from the four-family distribution."""
    family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    first = FIRST[int(rng.integers(len(FIRST)))]
    last = LAST[int(rng.integers(len(LAST)))]
    suffix = int(rng.integers(1000, 10000))
    name = f"{first} {last}"
    username = f"{first.lower()}{last.lower()}{suffix}"
    email = f"{username}@example.net"
    if family == "name_username":
        instruction = f"Set your account name to '{name}' and your username to '{username}'."
        expected = {"fullName": name, "username": username}
    elif family == "username_email":
        instruction = f"Change your username to '{username}' and your account email to '{email}'."
        expected = {"username": username, "email": email}
    elif family == "name_email":
        instruction = f"Update your account name to '{name}' and your account email to '{email}'."
        expected = {"fullName": name, "email": email}
    else:
        instruction = (f"Update your account name to '{name}', username to '{username}', "
                       f"and account email to '{email}'.")
        expected = {"fullName": name, "username": username, "email": email}
    task_id = f"generated_{index:03d}"
    verifier = out / "verifiers" / f"{task_id}.py"
    verify_rel = os.path.relpath(verifier, APP)
    code = ("import requests\n\n"
            f"EXPECTED = {expected!r}\n\n"
            "def verify(server_url: str) -> tuple[bool, str]:\n"
            "    response = requests.get(f'{server_url}/api/state', timeout=10)\n"
            "    response.raise_for_status()\n"
            "    user = response.json().get('currentUser', {})\n"
            "    wrong = {key: {'expected': value, 'actual': user.get(key)}\n"
            "             for key, value in EXPECTED.items() if user.get(key) != value}\n"
            "    return (not wrong, 'profile targets matched' if not wrong else str(wrong))\n")
    return ({"id": task_id, "difficulty": "sampled", "family": family,
             "instruction": instruction, "verify": verify_rel}, code.encode())


def prepare(seed_hex: str, protocol_commit: str, out: Path = OUT) -> dict:
    if len(seed_hex) != 32:
        raise ValueError("seed must be exactly 128 bits (32 hex digits)")
    seed = int(seed_hex, 16)
    out = Path(out).resolve()
    task_rng = np.random.default_rng(seed)
    order_rng = np.random.default_rng(seed ^ 0xE7B56F5AFA41C8C122530875A12CB7D1)
    tasks = []
    verifiers = {}
    for i in range(N):
        task, code = draw_task(task_rng, i, out)
        tasks.append(task)
        rel = f"verifiers/{task['id']}.py"
        verifiers[rel] = code
    bank_bytes = (json.dumps(tasks, indent=2) + "\n").encode()
    pool_bytes = POOL_SOURCE.read_bytes()
    queue = []
    for task in tasks:
        for arm in order_rng.permutation(ARMS):
            queue.append(make_pools.QueueItem(index=len(queue), pool="fixed_pair", arm_id=str(arm),
                                              task_id=task["id"], replicate=0, pilot=False))
    queue_bytes = "".join(json.dumps(asdict(item)) + "\n" for item in queue).encode()
    manifest = {"study": "Prospectively sampled Linear profile-edit pair", "profile": PROFILE,
                "protocol_commit": protocol_commit,
                "seed_hex": seed_hex, "distribution": "uniform over four families; independent names and suffix per task",
                "n_tasks": N, "n_episodes": len(queue), "task_ids": [t["id"] for t in tasks],
                "pool_files": [POOL_REL], "arms": list(ARMS),
                "task_bank_sha256": sha(bank_bytes),
                "verifier_sha256": {rel: sha(code) for rel, code in verifiers.items()},
                "generator_sha256": sha(Path(__file__).read_bytes()),
                "files": {QUEUE_REL: sha(queue_bytes), POOL_REL: sha(pool_bytes)}}
    planned = {"tasks.json": bank_bytes, QUEUE_REL: queue_bytes, POOL_REL: pool_bytes,
               **verifiers, "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    for rel, data in planned.items():
        put_once(out / rel, data)
    if out == OUT:
        collect.verify_queue(out, PROFILE)
        prompts = collect.load_prompts(out, PROFILE)
        collect.check_queue_arms(queue, prompts)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, help="new 128-bit random seed as 32 hex digits")
    parser.add_argument("--protocol-commit", required=True, help="commit freezing this generator and protocol")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = prepare(args.seed, args.protocol_commit, args.out)
    print(json.dumps({"tasks": manifest["n_tasks"], "episodes": manifest["n_episodes"],
                      "bank_sha256": manifest["task_bank_sha256"],
                      "family_counts": {family: sum(t["family"] == family for t in json.loads((args.out / "tasks.json").read_text()))
                                        for family in FAMILIES}}, indent=2))


if __name__ == "__main__":
    main()
