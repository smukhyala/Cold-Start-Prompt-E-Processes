"""Pool builders for the prompt-heterogeneity study (spec section 4).

    .venv/bin/python experiments/growing_bandits/empirical/make_het_pools.py

writes, under ``data/heterogeneity/``:

- ``bundles/{gitlab,gmail}.md`` -- frozen manual bundles (documentation only, capped at
  25,000 words), used both as the K generator's context and as the leak guard's stoplist
  source (an entity that already appears in the bundle is not a leak).
- ``k_generation_{gitlab,gmail}.json`` -- the K generator's full request/response record,
  including every rejected candidate and why.
- ``pools/GLG.yaml`` -- the 50 Pre-reg 9 grid ("G") prompts, re-id'd ``GLG_00..49``; the
  arm id never enters ``template.jinja``'s render, so each arm's rendered-prompt sha256 is
  asserted equal to the frozen Pre-reg 9 sha (``data/empirical_pool/pool_G.yaml``).
- ``pools/GLK.yaml`` / ``pools/GMK.yaml`` -- 40 Claude-written "knowledge" prompts per app,
  generated from that app's manual bundle and passed through the leak guard.
- ``pools/GMB.yaml`` -- 20 of the 50 GLG arms (chosen by ``np.random.default_rng(seed)``),
  re-id'd ``GMB_00..19``, for the Gmail "bridge" profile (same rendered prompt as its GLG
  source, run against Gmail tasks; ``manifest.json``'s ``bridge_source`` records the pairing
  for the H1 analysis).
- ``pools/anchors_gitlab.yaml`` -- ``GL_anchor_baseline``, ``GL_anchor_explorer`` (vectors
  from ``configs/arms_initial.yaml``) and ``GL_anchor_oracle`` (vector + prompt guidance
  from ``configs/arms_gitlab_strong.yaml``'s ``gitlab_oracle_operator``), all three rendered
  with ``configs/template_gitlab.jinja``.
- ``pools/anchors_gmail.yaml`` -- ``GM_anchor_baseline``, Pre-reg 9's anchor re-id'd and
  rendered with ``template.jinja``; asserted equal to ``data/empirical_pool/pool_anchor.yaml``.
- ``{gitlab,gmail,bridge}/queue.jsonl`` -- per-profile episode queues (see ``gitlab_queue``,
  ``gmail_queue``, ``bridge_queue``).
- ``manifest.json`` -- sha256 of every file above, task subsets, seeds, and ``bridge_source``.

Refuses to overwrite an existing profile file unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_pools import (  # noqa: E402
    AXES_PATH,
    FREEFORM_TEMPLATE,
    GRID_TEMPLATE,
    QueueItem,
    file_sha256,
    freeform_arm,
    grid_arm,
    load_pool,
    select_tasks,
    sha256_text,
    write_pool,
    write_queue,
)
from make_pools import ROOT as MP_ROOT  # noqa: E402

from cold_start.prompts.axes import load_axes  # noqa: E402
from cold_start.prompts.template import render_prompt  # noqa: E402
from cold_start.tasks.webarena import _webarena_root  # noqa: E402
from cold_start.types import Arm, PromptVector  # noqa: E402

assert MP_ROOT == ROOT

log = logging.getLogger("empirical.make_het_pools")

# ---- app registry ---------------------------------------------------------------------

GITLAB_TEMPLATE = ROOT / "configs" / "template_gitlab.jinja"
ARMS_INITIAL_PATH = ROOT / "configs" / "arms_initial.yaml"
ARMS_GITLAB_STRONG_PATH = ROOT / "configs" / "arms_gitlab_strong.yaml"
PREREG9_DIR = ROOT / "data" / "empirical_pool"
OUT_DIR = ROOT / "data" / "heterogeneity"
SEED = 20260928

#: Section 4: what each app's manual bundle is built from. ``manual_globs`` are resolved
#: under ``<webarena-root>/apps/user-manuals/<app>``; ``app_description`` (when set) is a
#: single file, relative to the webarena-infinity root, prefixed before the sorted manuals.
APPS: dict[str, dict] = {
    "gitlab": {
        "web_app": "apps/gitlab-plan-and-track",
        "manual_globs": [
            "user/project/issues/**/*.md",
            "user/project/labels.md",
            "user/project/milestones/**/*.md",
            "user/project/issue_board.md",
            "user/group/epics/**/*.md",
            "user/group/iterations/**/*.md",
        ],
        "app_description": "apps/gitlab-plan-and-track/APP_DESCRIPTION.md",
    },
    "gmail": {
        "web_app": "apps/gmail",
        "manual_globs": [
            "organize-and-manage/*.md",
            "compose-and-send/*.md",
            "settings-and-configuration/*.md",
        ],
        "app_description": None,
    },
}

PILOT_PER_POOL = 5
N_REPLICATES_GITLAB = {"GLG": 300, "GLK": 240}
N_REPLICATES_GMK = 120
N_BRIDGE = 20
N_K = 40
N_TASKS_GITLAB_BANK = 60
N_TASKS_BLOCK = 30


# ---- manual bundles ---------------------------------------------------------------------


def _manual_paths(app: str) -> list[Path]:
    """The app's manual files, in sorted path order."""
    spec = APPS[app]
    manual_root = _webarena_root() / "apps" / "user-manuals" / app
    paths: set[Path] = set()
    for pattern in spec["manual_globs"]:
        matched = list(manual_root.glob(pattern))
        if not matched:
            raise ValueError(f"{app!r}: manual glob {pattern!r} under {manual_root} matched nothing")
        paths.update(matched)
    return sorted(paths)


def build_bundle(app: str, max_words: int = 25_000) -> str:
    """The app's frozen manual bundle: app description (if any) then manuals, sorted path
    order, each section prefixed ``## <path relative to the webarena-infinity root>``,
    truncated at ``max_words`` at whole-file granularity -- the file that would push the
    running total over the cap is itself cut down to exactly fill the remaining budget,
    rather than being dropped or left whole.
    """
    webarena_root = _webarena_root()
    spec = APPS[app]
    ordered_paths: list[Path] = []
    if spec.get("app_description"):
        ordered_paths.append(webarena_root / spec["app_description"])
    ordered_paths.extend(_manual_paths(app))

    sections: list[str] = []
    used_words = 0
    for path in ordered_paths:
        remaining = max_words - used_words
        if remaining <= 0:
            break
        rel = path.relative_to(webarena_root).as_posix()
        body = path.read_text(encoding="utf-8", errors="ignore")
        words = body.split()
        if len(words) > remaining:
            words = words[:remaining]
            sections.append(f"## {rel}\n\n" + " ".join(words))
            used_words += remaining
            break
        sections.append(f"## {rel}\n\n" + body)
        used_words += len(words)
    return "\n\n".join(sections)


# ---- task instructions / entities --------------------------------------------------------


def _real_tasks_path(app: str) -> Path:
    return _webarena_root() / APPS[app]["web_app"] / "real-tasks.json"


def _load_real_tasks(app: str) -> list[dict]:
    with open(_real_tasks_path(app), encoding="utf-8") as fh:
        return json.load(fh)


def task_ids(app: str) -> list[str]:
    return [str(t["id"]) for t in _load_real_tasks(app)]


def task_texts(app: str) -> list[str]:
    """Every task instruction of `app`'s ``real-tasks.json``, in bank order."""
    return [str(t["instruction"]) for t in _load_real_tasks(app)]


def _norm_entity(s: str) -> str:
    """Whitespace-normalized, lowercased, with a bare leading/trailing quote mark and a
    trailing possessive "'s" stripped -- a capitalized-name run's trailing word character
    class matches the apostrophe of a possessive ("Chen's") or of an adjacent closing quote
    ("Initiative'"), neither of which is part of the entity itself.
    """
    norm = " ".join(s.split()).lower().strip("'\"")
    if norm.endswith("'s"):
        norm = norm[:-2].strip()
    return norm


#: A quote delimiter: not immediately preceded/followed by a word character, so a possessive
#: or contraction apostrophe (``don't``, ``Chen's``, ``O'Brien``) is never mistaken for one.
_SINGLE_QUOTE_RE = re.compile(r"(?<!\w)'([^'\n]{2,60}?)'(?!\w)")
_DOUBLE_QUOTE_RE = re.compile(r'(?<!\w)"([^"\n]{2,60}?)"(?!\w)')
#: 2-4 consecutive capitalized-initial words -- a capitalized multi-word name.
_CAP_RUN_RE = re.compile(r"[A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){1,3}")


def _extract_entities(text: str) -> set[str]:
    ents: set[str] = set()
    for regex in (_SINGLE_QUOTE_RE, _DOUBLE_QUOTE_RE):
        for m in regex.finditer(text):
            norm = _norm_entity(m.group(1))
            if norm:
                ents.add(norm)
    # Lower the sentence-initial word so it never itself starts a false 2-word capitalized
    # run (e.g. "Star Sarah Chen's..." must not yield "star sarah" as well as "sarah chen").
    desensitized = text[:1].lower() + text[1:] if text else text
    for m in _CAP_RUN_RE.finditer(desensitized):
        ents.add(_norm_entity(m.group(0)))
    return ents


def task_entities(app: str) -> set[str]:
    """Quoted strings and capitalized multi-word names from `app`'s task instructions,
    minus any that already appear in the app's manual bundle (documented, so not a leak).
    """
    ents: set[str] = set()
    for text in task_texts(app):
        ents |= _extract_entities(text)
    bundle_lower = build_bundle(app).lower()
    return {e for e in ents if e and e not in bundle_lower}


# ---- leak guard ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NGRAM_N = 8


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngram_windows(tokens: list[str], n: int = _NGRAM_N) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def leak_violations(text: str, task_texts: list[str], entities: set[str]) -> list[str]:  # noqa: F811
    """Reasons `text` leaks a task: a shared 8-token window with a task instruction, or a
    case-insensitive, word-boundary occurrence of a task entity.
    """
    violations: list[str] = []
    prompt_windows = _ngram_windows(_tokens(text))
    if prompt_windows:
        for task in task_texts:
            shared = prompt_windows & _ngram_windows(_tokens(task))
            for window in sorted(shared):
                violations.append(f"8-gram overlap with a task instruction: {' '.join(window)!r}")
    lowered = text.lower()
    for entity in sorted(e for e in entities if e):
        if re.search(r"\b" + re.escape(entity) + r"\b", lowered):
            violations.append(f"entity leak: {entity!r}")
    return violations


# ---- K generation ---------------------------------------------------------------------

GENERATOR_MODEL = "claude-opus-4-7"
GENERATOR_MAX_TOKENS = 16_000
K_MIN_WORDS, K_MAX_WORDS = 40, 250
N_REQUEST_SLACK = 10

KNOWLEDGE_PROMPT = """You are writing system-prompt extensions for an AI agent that operates the web \
application described in the documentation below, through a browser. Each extension should give the agent \
practical procedural knowledge about THIS application: where features live, how its workflows run, useful \
shortcuts or navigation routes, and how to confirm that a change actually took effect.

Write {n} distinct extensions. Vary their scope (one workflow vs. broad coverage), emphasis, structure \
(prose, rules, numbered procedures) and length. Each must be 40 to 250 words, addressed to the agent in the \
second person, plain text only, and must not invent features the documentation does not describe.

Return only a JSON array of {n} strings.

<documentation>
{bundle}
</documentation>"""


def generate_knowledge(client, app: str, bundle: str, n: int = N_K) -> tuple[list[str], dict]:
    """Ask the generator for `n` + 10 knowledge extensions, drop leaks/out-of-range/dupes,
    keep the first `n`, and raise if fewer than `n` survive. Every rejection (including its
    reason) is recorded in the returned dict's ``"rejected"`` list.
    """
    n_request = n + N_REQUEST_SLACK
    # `.replace`, not `.format`: the manual bundle routinely contains literal `{`/`}` (Hugo
    # shortcodes, JSON snippets) that would raise inside `str.format`.
    prompt = KNOWLEDGE_PROMPT.replace("{n}", str(n_request)).replace("{bundle}", bundle)
    response = client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=GENERATOR_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "max_tokens":
        raise ValueError(f"the generator's response was cut off at max_tokens={GENERATOR_MAX_TOKENS}; "
                         "refusing to use a truncated knowledge batch")

    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("no JSON array in the generator's response")
    items = json.loads(text[start : end + 1])

    texts_for_app = task_texts(app)
    ents = task_entities(app)

    kept: list[str] = []
    seen: set[str] = set()
    rejected: list[dict] = []
    for raw_item in items:
        item = str(raw_item)
        norm = " ".join(item.split())
        n_words = len(norm.split())
        if not K_MIN_WORDS <= n_words <= K_MAX_WORDS:
            rejected.append({"text": item, "reason": f"word count {n_words} outside [{K_MIN_WORDS}, {K_MAX_WORDS}]"})
            continue
        if norm in seen:
            rejected.append({"text": item, "reason": "duplicate"})
            continue
        violations = leak_violations(norm, texts_for_app, ents)
        if violations:
            rejected.append({"text": item, "reason": "; ".join(violations)})
            continue
        seen.add(norm)
        kept.append(item)
        if len(kept) == n:
            break

    if len(kept) < n:
        raise ValueError(f"only {len(kept)} valid, non-leaking, distinct knowledge extensions out of "
                         f"{len(items)} candidates; need {n}")

    raw = {
        "model": GENERATOR_MODEL,
        "max_tokens": GENERATOR_MAX_TOKENS,
        "app": app,
        "n_requested": n_request,
        "n": n,
        "prompt": prompt,
        "stop_reason": stop_reason,
        "response_text": text,
        "rejected": rejected,
    }
    return kept, raw


# ---- per-profile queues ---------------------------------------------------------------


def gitlab_queue(arms: dict[str, list[str]], block_a: list[str], block_b: list[str], seed: int) -> list[QueueItem]:
    """gitlab profile: pilot = (first 5 GLG + first 5 GLK + all anchors) x block A, shuffled;
    then the rest of block A shuffled; then block B shuffled; then 300 GLG + 240 GLK
    replicate cells (drawn without replacement from their own main cells) shuffled.
    """
    rng = np.random.default_rng(seed)
    all_tasks = block_a + block_b
    main = [(pool, arm, task) for pool in sorted(arms) for arm in arms[pool] for task in all_tasks]

    pilot_arms = set(arms["GLG"][:PILOT_PER_POOL]) | set(arms["GLK"][:PILOT_PER_POOL]) | set(arms["anchor"])
    block_a_set = set(block_a)

    pilot = [m for m in main if m[1] in pilot_arms and m[2] in block_a_set]
    rest_a = [m for m in main if m[2] in block_a_set and m[1] not in pilot_arms]
    block_b_items = [m for m in main if m[2] not in block_a_set]

    pilot = [pilot[int(i)] for i in rng.permutation(len(pilot))]
    rest_a = [rest_a[int(i)] for i in rng.permutation(len(rest_a))]
    block_b_items = [block_b_items[int(i)] for i in rng.permutation(len(block_b_items))]

    replicates: list[tuple[str, str, str]] = []
    for pool, n_rep in N_REPLICATES_GITLAB.items():
        pool_main = [m for m in main if m[0] == pool]
        idx = sorted(int(i) for i in rng.choice(len(pool_main), size=n_rep, replace=False))
        replicates.extend(pool_main[i] for i in idx)
    replicates = [replicates[int(i)] for i in rng.permutation(len(replicates))]

    n_pilot = len(pilot)
    ordered = [(*m, 0) for m in pilot] + [(*m, 0) for m in rest_a] + [(*m, 0) for m in block_b_items] \
        + [(*m, 1) for m in replicates]
    return [QueueItem(index=k, pool=p, arm_id=a, task_id=t, replicate=r, pilot=k < n_pilot)
            for k, (p, a, t, r) in enumerate(ordered)]


def gmail_queue(arms: dict[str, list[str]], tasks: list[str], seed: int) -> list[QueueItem]:
    """gmail profile: GMK x tasks + GM_anchor_baseline x tasks, shuffled; then 120 GMK
    replicate cells (drawn without replacement from GMK's main cells) shuffled.
    """
    rng = np.random.default_rng(seed)
    main = [(pool, arm, task) for pool in sorted(arms) for arm in arms[pool] for task in tasks]
    main = [main[int(i)] for i in rng.permutation(len(main))]

    gmk_main = [m for m in main if m[0] == "GMK"]
    idx = sorted(int(i) for i in rng.choice(len(gmk_main), size=N_REPLICATES_GMK, replace=False))
    replicates = [gmk_main[i] for i in idx]
    replicates = [replicates[int(i)] for i in rng.permutation(len(replicates))]

    ordered = [(*m, 0) for m in main] + [(*m, 1) for m in replicates]
    return [QueueItem(index=k, pool=p, arm_id=a, task_id=t, replicate=r, pilot=False)
            for k, (p, a, t, r) in enumerate(ordered)]


def bridge_queue(arms: dict[str, list[str]], tasks: list[str], seed: int) -> list[QueueItem]:
    """bridge profile: GMB x tasks, shuffled. No pilot, no replicates."""
    rng = np.random.default_rng(seed)
    main = [(pool, arm, task) for pool in sorted(arms) for arm in arms[pool] for task in tasks]
    main = [main[int(i)] for i in rng.permutation(len(main))]
    return [QueueItem(index=k, pool=p, arm_id=a, task_id=t, replicate=0, pilot=False)
            for k, (p, a, t) in enumerate(main)]


# ---- profile assembly (never run from tests) -------------------------------------------


def build_bundle_files(out: Path) -> tuple[str, str]:
    gitlab_bundle = build_bundle("gitlab")
    gmail_bundle = build_bundle("gmail")
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    (out / "bundles" / "gitlab.md").write_text(gitlab_bundle, encoding="utf-8")
    (out / "bundles" / "gmail.md").write_text(gmail_bundle, encoding="utf-8")
    return gitlab_bundle, gmail_bundle


def build_profiles(seed: int = SEED, out: Path = OUT_DIR, force: bool = False) -> None:
    """Build and freeze the prompt-heterogeneity study's pools and queues under `out`.

    Never calls the generator on an already-frozen output directory: refuses to overwrite
    unless `force`. Controller-only -- never run from tests.
    """
    out = Path(out)
    targets = [
        out / "bundles" / "gitlab.md", out / "bundles" / "gmail.md",
        out / "k_generation_gitlab.json", out / "k_generation_gmail.json",
        *(out / "pools" / f"{name}.yaml" for name in
          ("GLG", "GLK", "GMK", "GMB", "anchors_gitlab", "anchors_gmail")),
        out / "gitlab" / "queue.jsonl", out / "gmail" / "queue.jsonl", out / "bridge" / "queue.jsonl",
        out / "manifest.json",
    ]
    if any(p.exists() for p in targets) and not force:
        raise SystemExit(f"heterogeneity profile files exist under {out}; frozen (pass --force to rebuild)")

    axes = load_axes(AXES_PATH)

    # ---- GLG: the 50 Pre-reg 9 G arms, re-id'd; render sha must match the frozen sha. ----
    g_pool = load_pool(PREREG9_DIR / "pool_G.yaml")
    if len(g_pool) != 50:
        raise ValueError(f"expected 50 Pre-reg 9 G arms, found {len(g_pool)}")
    glg_arms: list[Arm] = []
    for i, pa in enumerate(g_pool):
        if pa.arm.prompt_guidance:
            raise ValueError(f"pool_G.yaml:{pa.arm.arm_id}: expected empty prompt_guidance for a grid arm")
        arm = grid_arm(f"GLG_{i:02d}", pa.arm.vector)
        rendered = render_prompt(arm.vector, axes, GRID_TEMPLATE, prompt_guidance=arm.prompt_guidance,
                                 arm_id=arm.arm_id, arm_name=arm.name)
        sha = sha256_text(rendered)
        if sha != pa.sha256:
            raise ValueError(f"GLG_{i:02d}: re-id'd render sha256 {sha} != Pre-reg 9 sha256 {pa.sha256}")
        glg_arms.append(arm)
    write_pool(out / "pools" / "GLG.yaml", "GLG", glg_arms, GRID_TEMPLATE, AXES_PATH,
               meta={"source": "data/empirical_pool/pool_G.yaml (Pre-reg 9)", "seed": seed})

    # ---- GMB: 20 of the 50 GLG arms, chosen by seed; bridge_source pairs each with its GLG id.
    bridge_idx = np.random.default_rng(seed).choice(50, N_BRIDGE, replace=False)
    gmb_arms: list[Arm] = []
    bridge_source: dict[str, str] = {}
    for j, i in enumerate(bridge_idx):
        src = glg_arms[int(i)]
        gmb_id = f"GMB_{j:02d}"
        gmb_arms.append(grid_arm(gmb_id, src.vector))
        bridge_source[gmb_id] = src.arm_id
    write_pool(out / "pools" / "GMB.yaml", "GMB", gmb_arms, GRID_TEMPLATE, AXES_PATH,
               meta={"source": "GLG subset (bridge)", "seed": seed})

    # ---- K generation: gitlab, gmail. ----
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()

    gitlab_bundle, gmail_bundle = build_bundle_files(out)

    gitlab_texts, gitlab_raw = generate_knowledge(client, "gitlab", gitlab_bundle, n=N_K)
    (out / "k_generation_gitlab.json").write_text(json.dumps(gitlab_raw, indent=2, ensure_ascii=False), encoding="utf-8")
    glk_arms = [freeform_arm(f"GLK_{i:02d}", t) for i, t in enumerate(gitlab_texts)]
    write_pool(out / "pools" / "GLK.yaml", "GLK", glk_arms, FREEFORM_TEMPLATE, AXES_PATH,
               meta={"generator": GENERATOR_MODEL, "app": "gitlab"})

    gmail_texts, gmail_raw = generate_knowledge(client, "gmail", gmail_bundle, n=N_K)
    (out / "k_generation_gmail.json").write_text(json.dumps(gmail_raw, indent=2, ensure_ascii=False), encoding="utf-8")
    gmk_arms = [freeform_arm(f"GMK_{i:02d}", t) for i, t in enumerate(gmail_texts)]
    write_pool(out / "pools" / "GMK.yaml", "GMK", gmk_arms, FREEFORM_TEMPLATE, AXES_PATH,
               meta={"generator": GENERATOR_MODEL, "app": "gmail"})

    # ---- anchors. ----
    initial_arms = {a["arm_id"]: a for a in yaml.safe_load(ARMS_INITIAL_PATH.read_text())["arms"]}
    strong_arms = {a["arm_id"]: a for a in yaml.safe_load(ARMS_GITLAB_STRONG_PATH.read_text())["arms"]}

    gl_baseline = grid_arm("GL_anchor_baseline", PromptVector(**initial_arms["baseline"]["vector"]))
    gl_explorer = grid_arm("GL_anchor_explorer", PromptVector(**initial_arms["explorer"]["vector"]))
    oracle_src = strong_arms["gitlab_oracle_operator"]
    gl_oracle = Arm(arm_id="GL_anchor_oracle", name="GL_anchor_oracle",
                    vector=PromptVector(**oracle_src["vector"]),
                    prompt_guidance=oracle_src.get("prompt_guidance", ""))
    write_pool(out / "pools" / "anchors_gitlab.yaml", "anchor", [gl_baseline, gl_explorer, gl_oracle],
               GITLAB_TEMPLATE, AXES_PATH,
               meta={"sources": ["configs/arms_initial.yaml", "configs/arms_gitlab_strong.yaml"]})

    gm_baseline = grid_arm("GM_anchor_baseline", PromptVector(**initial_arms["baseline"]["vector"]))
    prereg9_anchor = load_pool(PREREG9_DIR / "pool_anchor.yaml")[0]
    gm_rendered = render_prompt(gm_baseline.vector, axes, GRID_TEMPLATE, prompt_guidance=gm_baseline.prompt_guidance,
                                arm_id=gm_baseline.arm_id, arm_name=gm_baseline.name)
    if sha256_text(gm_rendered) != prereg9_anchor.sha256:
        raise ValueError("GM_anchor_baseline: render sha256 != Pre-reg 9 anchor sha256")
    write_pool(out / "pools" / "anchors_gmail.yaml", "anchor", [gm_baseline], GRID_TEMPLATE, AXES_PATH,
               meta={"historical_arm": "anchor_baseline (Pre-reg 9)"})

    # ---- queues. ----
    gitlab_bank = task_ids("gitlab")
    sixty = select_tasks(gitlab_bank, N_TASKS_GITLAB_BANK, seed)
    block_a = select_tasks(sixty, N_TASKS_BLOCK, seed + 1)
    block_b = [t for t in sixty if t not in set(block_a)]

    gitlab_arms_by_pool = {
        "GLG": [a.arm_id for a in glg_arms],
        "GLK": [a.arm_id for a in glk_arms],
        "anchor": [gl_baseline.arm_id, gl_explorer.arm_id, gl_oracle.arm_id],
    }
    gl_queue = gitlab_queue(gitlab_arms_by_pool, block_a, block_b, seed=seed)
    (out / "gitlab").mkdir(parents=True, exist_ok=True)
    write_queue(out / "gitlab" / "queue.jsonl", gl_queue)

    prereg9_manifest = json.loads((PREREG9_DIR / "manifest.json").read_text())
    gmail_task_subset: list[str] = prereg9_manifest["task_subset"]

    gmail_arms_by_pool = {"GMK": [a.arm_id for a in gmk_arms], "anchor": [gm_baseline.arm_id]}
    gm_queue = gmail_queue(gmail_arms_by_pool, gmail_task_subset, seed=seed)
    (out / "gmail").mkdir(parents=True, exist_ok=True)
    write_queue(out / "gmail" / "queue.jsonl", gm_queue)

    bridge_arms_by_pool = {"GMB": [a.arm_id for a in gmb_arms]}
    br_queue = bridge_queue(bridge_arms_by_pool, gmail_task_subset, seed=seed)
    (out / "bridge").mkdir(parents=True, exist_ok=True)
    write_queue(out / "bridge" / "queue.jsonl", br_queue)

    # ---- manifest. ----
    files = [
        out / "bundles" / "gitlab.md", out / "bundles" / "gmail.md",
        out / "k_generation_gitlab.json", out / "k_generation_gmail.json",
        out / "pools" / "GLG.yaml", out / "pools" / "GLK.yaml", out / "pools" / "GMK.yaml",
        out / "pools" / "GMB.yaml", out / "pools" / "anchors_gitlab.yaml", out / "pools" / "anchors_gmail.yaml",
        out / "gitlab" / "queue.jsonl", out / "gmail" / "queue.jsonl", out / "bridge" / "queue.jsonl",
    ]
    manifest = {
        "seed": seed,
        "gitlab_task_bank": gitlab_bank,
        "gitlab_task_subset_60": sixty,
        "gitlab_block_a": block_a,
        "gitlab_block_b": block_b,
        "gmail_task_subset_30": gmail_task_subset,
        "bridge_source": bridge_source,
        "files": {str(p.relative_to(out)): file_sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote heterogeneity profiles under %s", out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing profile files")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_profiles(seed=args.seed, out=args.out, force=args.force)


if __name__ == "__main__":
    main()
