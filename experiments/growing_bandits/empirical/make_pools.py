"""Build and freeze the real prompt pools (spec section 3.1) and the episode queue (3.2).

    .venv/bin/python experiments/growing_bandits/empirical/make_pools.py

writes, under ``data/empirical_pool/``: ``pool_G.yaml`` (50 grid prompts), ``pool_F.yaml``
(50 Claude-written prompts), ``pool_anchor.yaml`` (the hand-written ``baseline``),
``f_generation_raw.json`` (the generator's request and full response), ``queue.jsonl``
(6,360 episodes, pilot first) and ``manifest.json`` (sha256 of each file). It refuses to
overwrite an existing pool unless ``--force``: the pools are frozen by Pre-registration 9.

Every pool file records the sha256 of each arm's *rendered* prompt; `load_pool`
re-renders and refuses a mismatch, so the collector can only ever send the frozen text.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.prompts.axes import AXIS_NAMES, AxesSpec, load_axes  # noqa: E402
from cold_start.prompts.template import render_prompt  # noqa: E402
from cold_start.types import Arm, PromptVector  # noqa: E402

log = logging.getLogger("empirical.make_pools")

SEED = 20260926
N_PER_POOL = 50
N_REPLICATES = 300
PILOT_PER_POOL = 5
#: Amendment 1: the paid pilot's real cost (~$0.104/episode) forced halving the task bank to a
#: seeded, difficulty-stratified 30 of the 60 Gmail tasks. `N_TASKS` must divide evenly by
#: `len(STRATA)`.
N_TASKS = 30
#: Difficulty strata, by the letter after `task_` in a bank task id (`task_e3`, `task_m12`, ...),
#: processed in this order by `select_tasks`.
STRATA: tuple[str, ...] = ("e", "m", "h")
AMENDMENT_1_LABEL = "1: 30-task stratified subset"
_TASK_ID_RE = re.compile(r"^task_([emh])(\d+)$")
DATA_DIR = ROOT / "data" / "empirical_pool"
AXES_PATH = ROOT / "configs" / "axes.yaml"
GRID_TEMPLATE = ROOT / "configs" / "template.jinja"
FREEFORM_TEMPLATE = ROOT / "configs" / "template_freeform.jinja"
BASELINE_VECTOR = {"planning": 0, "verification": 0, "agency": 0, "expertise": 1, "format": 0, "goal": 0}
ANCHOR_ARM_ID = "anchor_baseline"

GENERATOR_MODEL = "claude-opus-4-7"
GENERATOR_MAX_TOKENS = 16_000
N_REQUEST = 60
MIN_WORDS, MAX_WORDS = 40, 250
GENERATION_PROMPT = """You are helping build a benchmark of system-prompt instructions for an AI agent that \
operates a web-based email client through a browser: reading, searching, starring, labeling, archiving, \
replying to, forwarding and composing emails.

Write {n} distinct instructions that could be appended to such an agent's system prompt to guide how it \
works. Make them genuinely different from one another: vary the strategy, tone, level of detail, emphasis \
(speed, caution, verification, planning, exploration, precision), structure (prose, bullet rules, \
numbered procedures) and persona. Write each as a real practitioner might, not as a caricature, and not \
deliberately bad.

Constraints for every instruction:
- 40 to 250 words.
- Addressed to the agent in the second person.
- Do not mention any specific email, person, label, or task.
- Plain text only: no Markdown headings, no code fences.

Return only a JSON array of {n} strings."""


# ---- prompts ------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolArm:
    pool: str
    arm: Arm
    template: str
    text: str
    sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def all_grid_vectors(axes: AxesSpec) -> list[PromptVector]:
    ranges = [range(axes[name].max + 1) for name in AXIS_NAMES]
    return [PromptVector(**dict(zip(AXIS_NAMES, combo, strict=True))) for combo in itertools.product(*ranges)]


def sample_grid_vectors(axes: AxesSpec, n: int, seed: int) -> list[PromptVector]:
    grid = all_grid_vectors(axes)
    idx = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [grid[int(i)] for i in idx]


def grid_arm(arm_id: str, vec: PromptVector) -> Arm:
    return Arm(arm_id=arm_id, name=arm_id, vector=vec, prompt_guidance="")


def freeform_arm(arm_id: str, text: str) -> Arm:
    # The vector is a placeholder the free-form template never reads.
    return Arm(arm_id=arm_id, name=arm_id, vector=PromptVector(**BASELINE_VECTOR), prompt_guidance=text)


def render(arm: Arm, template: Path, axes: AxesSpec) -> str:
    return render_prompt(arm.vector, axes, template, prompt_guidance=arm.prompt_guidance,
                         arm_id=arm.arm_id, arm_name=arm.name)


def parse_freeform(text: str, n: int) -> list[str]:
    """The first `n` valid, distinct instructions in the response's JSON array."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in the generator's response")
    items = json.loads(text[start : end + 1])
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item)
        norm = " ".join(s.split())
        if not MIN_WORDS <= len(norm.split()) <= MAX_WORDS or norm in seen:
            continue
        seen.add(norm)
        out.append(s)
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} valid distinct instructions; need {n}")


def generate_freeform(client, n: int = N_PER_POOL) -> tuple[list[str], dict]:
    """One generator call; returns the instructions and the full request/response record."""
    prompt = GENERATION_PROMPT.format(n=N_REQUEST)
    response = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=GENERATOR_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise ValueError(f"the generator's response was cut off at max_tokens={GENERATOR_MAX_TOKENS}; "
                         "refusing to freeze a truncated pool F")
    raw = {"model": GENERATOR_MODEL, "max_tokens": GENERATOR_MAX_TOKENS, "prompt": prompt,
           "stop_reason": stop_reason, "response_text": text}
    return parse_freeform(text, n), raw


# ---- pool files ---------------------------------------------------------------------


def write_pool(path: Path, pool: str, arms: list[Arm], template: Path, axes_path: Path, meta: dict) -> None:
    axes = load_axes(axes_path)
    entries = []
    for arm in arms:
        text = render(arm, template, axes)
        entries.append({
            "arm_id": arm.arm_id,
            "name": arm.name,
            "vector": arm.vector.as_dict(),
            "prompt_guidance": arm.prompt_guidance,
            "prompt_sha256": sha256_text(text),
        })
    doc = {"pool": pool, "template": str(Path(template).relative_to(ROOT)) if Path(template).is_relative_to(ROOT)
           else str(template), "meta": meta, "arms": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True, width=10_000)


def load_pool(path: Path, axes_path: Path = AXES_PATH) -> list[PoolArm]:
    """Every arm of a pool file, re-rendered; raises if a rendered prompt's sha256 moved."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    template = Path(doc["template"])
    if not template.is_absolute():
        template = ROOT / template
    axes = load_axes(axes_path)
    out: list[PoolArm] = []
    for entry in doc["arms"]:
        arm = Arm(arm_id=entry["arm_id"], name=entry["name"], vector=PromptVector(**entry["vector"]),
                  prompt_guidance=entry["prompt_guidance"])
        text = render(arm, template, axes)
        sha = sha256_text(text)
        if sha != entry["prompt_sha256"]:
            raise ValueError(f"{path.name}:{arm.arm_id}: rendered prompt sha256 {sha} != frozen "
                             f"{entry['prompt_sha256']}")
        out.append(PoolArm(pool=doc["pool"], arm=arm, template=str(template), text=text, sha256=sha))
    return out


# ---- queue --------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueItem:
    index: int
    pool: str
    arm_id: str
    task_id: str
    replicate: int
    pilot: bool


def build_queue(
    arms_by_pool: dict[str, list[str]],
    task_ids: list[str],
    *,
    n_replicates: int,
    pilot_arms: set[str],
    seed: int,
) -> list[QueueItem]:
    """Every (arm, task) once, plus `n_replicates` repeats of G/F cells; pilot first, each part shuffled."""
    rng = np.random.default_rng(seed)
    main = [(pool, arm, task) for pool in sorted(arms_by_pool) for arm in arms_by_pool[pool] for task in task_ids]
    eligible = [m for m in main if m[0] in ("G", "F")]
    rep_idx = sorted(int(i) for i in rng.choice(len(eligible), size=n_replicates, replace=False))
    items = [(p, a, t, 0) for p, a, t in main] + [(*eligible[i], 1) for i in rep_idx]
    pilot = [it for it in items if it[1] in pilot_arms and it[3] == 0]
    rest = [it for it in items if not (it[1] in pilot_arms and it[3] == 0)]
    ordered = [pilot[int(i)] for i in rng.permutation(len(pilot))] + [rest[int(i)] for i in rng.permutation(len(rest))]
    return [QueueItem(index=k, pool=p, arm_id=a, task_id=t, replicate=r, pilot=k < len(pilot))
            for k, (p, a, t, r) in enumerate(ordered)]


def write_queue(path: Path, queue: list[QueueItem]) -> None:
    with open(path, "w") as fh:
        for item in queue:
            fh.write(json.dumps(asdict(item)) + "\n")


def read_queue(path: Path) -> list[QueueItem]:
    with open(path) as fh:
        return [QueueItem(**json.loads(line)) for line in fh if line.strip()]


def _task_stratum_and_suffix(task_id: str) -> tuple[str, int]:
    m = _TASK_ID_RE.match(task_id)
    if not m:
        raise ValueError(f"{task_id!r}: expected task_<{'|'.join(STRATA)}><digits>")
    return m.group(1), int(m.group(2))


def select_tasks(task_ids: list[str], n: int, seed: int) -> list[str]:
    """A seeded, difficulty-stratified subset of `n` of `task_ids` (amendment 1's 30-task bank).

    Stratifies by the difficulty letter after ``task_`` (``e``, ``m``, ``h``) and takes
    ``n // len(STRATA)`` from each with one ``np.random.default_rng(seed)``, strata processed
    in the order ``e, m, h``, each stratum sorted by its task id's integer suffix before
    sampling without replacement. Returns the chosen ids in bank order (`task_ids`'s order).
    """
    if n % len(STRATA) != 0:
        raise ValueError(f"n={n} is not divisible by the {len(STRATA)} strata {STRATA}")
    per_stratum = n // len(STRATA)
    by_stratum: dict[str, list[str]] = {s: [] for s in STRATA}
    for task_id in task_ids:
        letter, _ = _task_stratum_and_suffix(task_id)
        by_stratum[letter].append(task_id)
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for letter in STRATA:
        ids = by_stratum[letter]
        if len(ids) < per_stratum:
            raise ValueError(f"stratum {letter!r} has only {len(ids)} tasks; need {per_stratum}")
        ordered = sorted(ids, key=lambda tid: _task_stratum_and_suffix(tid)[1])
        idx = rng.choice(len(ordered), size=per_stratum, replace=False)
        chosen.update(ordered[int(i)] for i in idx)
    return [task_id for task_id in task_ids if task_id in chosen]


def requeue(out: Path) -> dict:
    """Rebuild ``queue.jsonl``/``manifest.json`` for `N_TASKS`-task stratified subset (amendment 1).

    Loads (and thereby verifies the frozen per-arm sha256s of) the three pool files with
    `load_pool`, then checks the pool files' and ``f_generation_raw.json``'s whole-file
    sha256 against `manifest.json`'s -- refusing if either moved. Never calls the generator
    and never rewrites a pool file: it only rewrites ``queue.jsonl`` and ``manifest.json``.
    """
    out = Path(out)
    pool_paths = {p: out / f"pool_{p}.yaml" for p in ("G", "F", "anchor")}
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    loaded = {p: load_pool(path) for p, path in pool_paths.items()}  # verifies rendered sha256s

    frozen = [*pool_paths.values(), out / "f_generation_raw.json"]
    for path in frozen:
        got = file_sha256(path)
        expected = manifest["files"].get(path.name)
        if expected is None or got != expected:
            raise ValueError(f"{path.name}: sha256 {got} != manifest's frozen {expected}; refusing to requeue")

    task_ids: list[str] = manifest["task_ids"]  # all 60, bank order
    seed: int = manifest["seed"]
    subset = select_tasks(task_ids, N_TASKS, seed)

    arms_by_pool = {p: [a.arm.arm_id for a in loaded[p]] for p in pool_paths}
    pilot = set(arms_by_pool["G"][:PILOT_PER_POOL] + arms_by_pool["F"][:PILOT_PER_POOL] + [ANCHOR_ARM_ID])
    queue = build_queue(arms_by_pool, subset, n_replicates=N_REPLICATES, pilot_arms=pilot, seed=seed)
    write_queue(out / "queue.jsonl", queue)

    manifest["files"]["queue.jsonl"] = file_sha256(out / "queue.jsonl")
    manifest["task_subset"] = subset
    manifest["n_tasks"] = N_TASKS
    manifest["amendment"] = AMENDMENT_1_LABEL
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("requeued %d items (%d pilot) over %d tasks under %s",
             len(queue), sum(q.pilot for q in queue), len(subset), out)
    return manifest


def gmail_task_ids() -> list[str]:
    from cold_start.tasks.webarena import _import_webarena, _webarena_root

    _, _, tasks_mod = _import_webarena()
    tasks = tasks_mod.load_tasks(str(_webarena_root() / "apps/gmail"), "real-tasks")
    return [str(t["id"]) for t in tasks]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing pool files")
    ap.add_argument("--requeue", action="store_true",
                    help="amendment 1: rebuild queue.jsonl/manifest.json for the N_TASKS-task "
                         "stratified subset from the existing frozen pools; ignores --force, "
                         "never calls the generator or rewrites a pool file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out: Path = args.out
    if args.requeue:
        requeue(out)
        return
    pools = {p: out / f"pool_{p}.yaml" for p in ("G", "F", "anchor")}
    if any(p.exists() for p in pools.values()) and not args.force:
        raise SystemExit(f"pool files exist under {out}; they are frozen (pass --force to rebuild)")

    axes = load_axes(AXES_PATH)
    g_arms = [grid_arm(f"G_{i:02d}", v) for i, v in enumerate(sample_grid_vectors(axes, N_PER_POOL, args.seed))]

    # Generate (and validate) F before writing ANY file: a failed or truncated generation must
    # leave the directory untouched, not a half-frozen set that needs --force to rebuild.
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    texts, raw = generate_freeform(anthropic.Anthropic(), N_PER_POOL)
    f_arms = [freeform_arm(f"F_{i:02d}", t) for i, t in enumerate(texts)]

    write_pool(pools["G"], "G", g_arms, GRID_TEMPLATE, AXES_PATH, meta={"seed": args.seed, "grid_size": 2304})
    (out / "f_generation_raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    write_pool(pools["F"], "F", f_arms, FREEFORM_TEMPLATE, AXES_PATH, meta={"generator": GENERATOR_MODEL})

    anchor = [grid_arm(ANCHOR_ARM_ID, PromptVector(**BASELINE_VECTOR))]
    write_pool(pools["anchor"], "anchor", anchor, GRID_TEMPLATE, AXES_PATH, meta={"historical_arm": "baseline"})

    task_ids = gmail_task_ids()
    if len(task_ids) != 60:
        raise SystemExit(f"expected 60 Gmail real-tasks, found {len(task_ids)}")
    arms_by_pool = {"G": [a.arm_id for a in g_arms], "F": [a.arm_id for a in f_arms], "anchor": [ANCHOR_ARM_ID]}
    pilot = set(arms_by_pool["G"][:PILOT_PER_POOL] + arms_by_pool["F"][:PILOT_PER_POOL] + [ANCHOR_ARM_ID])
    queue = build_queue(arms_by_pool, task_ids, n_replicates=N_REPLICATES, pilot_arms=pilot, seed=args.seed)
    write_queue(out / "queue.jsonl", queue)

    files = [*pools.values(), out / "f_generation_raw.json", out / "queue.jsonl"]
    manifest = {"seed": args.seed, "task_ids": task_ids, "files": {p.name: file_sha256(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d pool files, %d queue items (%d pilot) under %s",
             len(pools), len(queue), sum(q.pilot for q in queue), out)


if __name__ == "__main__":
    main()
