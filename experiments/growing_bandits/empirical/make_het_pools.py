"""Pool builders for the prompt-heterogeneity study (spec section 4).

    .venv/bin/python experiments/growing_bandits/empirical/make_het_pools.py

writes, under ``data/heterogeneity/``:

- ``bundles/{gitlab,gmail}.md`` -- frozen manual bundles (documentation only, capped at
  25,000 words), used both as the K generator's context and as the leak guard's stoplist
  source (an entity that already appears in the bundle is not a leak). The app description
  has any "Seed Data Summary"-style section stripped first (that section names every seeded
  user/epic/board by name, which would otherwise exempt them from the leak guard); manuals
  are assembled in a fixed per-app priority order (see ``APPS``), not plain alphabetical
  order, and a page byte-identical (or near-identical modulo its own self-referential
  "Source: <url>" line) to one already included is skipped.
- ``k_generation_{gitlab,gmail}.json`` -- the K generator's full request/response record,
  including every rejected candidate and why, and the frozen entity set used to judge them.
  Written even when generation fails (``generate_knowledge`` raises
  ``KnowledgeGenerationError`` carrying the same record), so a failed paid call is never
  silently lost.
- ``pools/GLG.yaml`` -- the 50 Pre-reg 9 grid ("G") prompts, re-id'd ``GLG_00..49``; the
  arm id never enters ``template.jinja``'s render, so each arm's rendered-prompt sha256 is
  asserted equal to the frozen Pre-reg 9 sha (``data/empirical_pool/pool_G.yaml``).
- ``pools/GLK.yaml`` / ``pools/GMK.yaml`` -- 40 Claude-written "knowledge" prompts per app,
  generated from that app's manual bundle and passed through the leak guard. Generation is
  skip-if-present: if a pool and its ``k_generation_<app>.json`` already exist and verify,
  the generator is not called again for that app, so an interrupted build (e.g. GitLab
  succeeded, Gmail failed) resumes without paying for GitLab a second time. ``--force``
  ignores any cached pool and regenerates both apps unconditionally.
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
- ``manifest.json`` -- sha256 of every file above (plus each bundle's included-file list
  with hashes, each app's real-tasks.json sha256, its frozen entity-set hash, the bundle
  word cap, and the derived block seed), task subsets, seeds, and ``bridge_source``.

A *complete* prior build (``manifest.json`` present) is frozen: refuses to rebuild unless
``--force``. An *incomplete* one (e.g. an interrupted/failed run) is always resumable without
``--force`` -- see the K-generation skip-if-present note above.
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
BUNDLE_MAX_WORDS = 25_000

#: Section 4: what each app's manual bundle is built from. ``manual_globs`` is a *priority*
#: order (fix round 1, issue 2): patterns are resolved and appended to the candidate list in
#: the order given, sorted within each pattern; a path already yielded by an earlier pattern
#: is skipped when the later, broader glob matches it again. Resolved under
#: ``<webarena-root>/apps/user-manuals/<app>``; ``app_description`` (when set) is a single
#: file, relative to the webarena-infinity root, prefixed before the priority-ordered manuals.
APPS: dict[str, dict] = {
    "gitlab": {
        "web_app": "apps/gitlab-plan-and-track",
        "manual_globs": [
            "user/project/labels.md",
            "user/project/issue_board.md",
            "user/project/milestones/**/*.md",
            "user/project/issues/managing_issues.md",
            "user/project/issues/**/*.md",
            "user/group/epics/**/*.md",
            "user/group/iterations/**/*.md",
        ],
        "app_description": "apps/gitlab-plan-and-track/APP_DESCRIPTION.md",
    },
    "gmail": {
        "web_app": "apps/gmail",
        "manual_globs": [
            "settings-and-configuration/*.md",
            "organize-and-manage/*.md",
            "compose-and-send/*.md",
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

#: A markdown heading line, e.g. ``## Seed Data Summary``: group 1 is the ``#`` run (its
#: level), group 2 is the heading text.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)


def _strip_seed_data_sections(text: str) -> str:
    """Remove every section whose heading matches /seed data/i (fix round 1, issue 1): from
    that heading up to, but not including, the next heading of the same or higher level
    (i.e. a ``#`` run no longer than the matched heading's), or end of file if there is none.
    An app description's "Seed Data Summary" names every seeded user/epic/board/label by
    name; leaving it in the bundle would exempt all of them from the leak guard, unlike
    Gmail (whose manuals carry no such section).
    """
    while True:
        matches = list(_HEADING_RE.finditer(text))
        target = None
        for i, m in enumerate(matches):
            if re.search(r"seed data", m.group(2), re.IGNORECASE):
                target = (i, m)
                break
        if target is None:
            return text
        i, m = target
        level = len(m.group(1))
        end = len(text)
        for m2 in matches[i + 1 :]:
            if len(m2.group(1)) <= level:
                end = m2.start()
                break
        text = text[: m.start()] + text[end:]


#: A manual page's own self-referential "Source: <url ending in this file's doc id>" line --
#: stripped before hashing for *dedup* purposes only (fix round 1, issue 2), so two mirrored
#: copies of the same support page (different doc ids under different section folders) are
#: recognized as duplicates despite not being byte-identical. The section's recorded
#: ``sha256`` (for the manifest) is still the true, unmodified content hash.
_SOURCE_LINE_RE = re.compile(r"^Source:\s*\S+\s*$", re.MULTILINE)


def _dedup_key(body: str) -> str:
    return sha256_text(_SOURCE_LINE_RE.sub("", body))


def _assemble_bundle(items: list[tuple[str, str]], max_words: int = BUNDLE_MAX_WORDS) -> list[dict]:
    """Ordered, deduplicated, capped bundle sections from `items` = ``[(rel_path, body), ...]``
    already in priority order. A `rel_path` already used, or a `body` that is a near-
    duplicate (by ``_dedup_key``) of one already included, is skipped. Each section is
    headed ``## <rel_path>``; the running total -- including that header -- is capped at
    `max_words`, with the section that would cross the cap truncated to exactly fill what's
    left (fix round 1, issue 2: the header itself now counts against the cap).
    """
    sections: list[dict] = []
    seen_paths: set[str] = set()
    seen_content: set[str] = set()
    used_words = 0
    for rel_path, body in items:
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        key = _dedup_key(body)
        if key in seen_content:
            continue
        seen_content.add(key)

        remaining = max_words - used_words
        if remaining <= 0:
            break
        header = f"## {rel_path}\n\n"
        header_words = len(header.split())
        body_words = body.split()
        total_words = header_words + len(body_words)
        content_sha = sha256_text(body)
        if total_words > remaining:
            body_budget = max(remaining - header_words, 0)
            truncated_body = body_words[:body_budget]
            text = header + " ".join(truncated_body)
            included = header_words + len(truncated_body)
            sections.append({"path": rel_path, "sha256": content_sha, "words": included,
                             "truncated": True, "text": text})
            used_words += included
            break
        text = header + body
        sections.append({"path": rel_path, "sha256": content_sha, "words": total_words,
                         "truncated": False, "text": text})
        used_words += total_words
    return sections


def bundle_sections(app: str, max_words: int = BUNDLE_MAX_WORDS) -> list[dict]:
    """`app`'s bundle sections: the (seed-data-stripped) app description first, if any, then
    its manuals in ``APPS[app]['manual_globs']`` priority order. Each entry is
    ``{"path", "sha256", "words", "truncated", "text"}``.
    """
    webarena_root = _webarena_root()
    spec = APPS[app]
    items: list[tuple[str, str]] = []
    seen: set[Path] = set()

    if spec.get("app_description"):
        p = webarena_root / spec["app_description"]
        seen.add(p)
        body = _strip_seed_data_sections(p.read_text(encoding="utf-8", errors="ignore"))
        items.append((p.relative_to(webarena_root).as_posix(), body))

    manual_root = webarena_root / "apps" / "user-manuals" / app
    for pattern in spec["manual_globs"]:
        matched = sorted(manual_root.glob(pattern))
        if not matched:
            raise ValueError(f"{app!r}: manual glob {pattern!r} under {manual_root} matched nothing")
        for p in matched:
            if p in seen:
                continue
            seen.add(p)
            items.append((p.relative_to(webarena_root).as_posix(),
                         p.read_text(encoding="utf-8", errors="ignore")))

    return _assemble_bundle(items, max_words)


def build_bundle(app: str, max_words: int = BUNDLE_MAX_WORDS) -> str:
    """The app's frozen manual bundle text (see `bundle_sections`)."""
    return "\n\n".join(s["text"] for s in bundle_sections(app, max_words))


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
#: An email address (fix round 1, issue 3).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
#: An internal possessive boundary within a matched capitalized run: "Chen's " (apostrophe-s)
#: or "Williams' " (bare apostrophe after a trailing "s"). Used to split a run at the
#: boundary rather than emit the whole thing (and its possessive marker) as one entity.
_POSSESSIVE_S_RE = re.compile(r"(?<=\w)'s(?=\s|$)")
_POSSESSIVE_BARE_RE = re.compile(r"(?<=s)'(?=\s|$)")


def _split_possessive_run(run: str) -> list[str]:
    """A matched capitalized run split at its first internal possessive boundary (fix round
    1, issue 3), e.g. ``"Marcus Williams' Design System"`` ->
    ``["Marcus Williams", "Design System"]`` -- so a leak of just the person's name (without
    the rest of the run) is still caught, instead of only the whole run being a single,
    much-less-likely-to-recur entity.
    """
    candidates = [m for m in (_POSSESSIVE_S_RE.search(run), _POSSESSIVE_BARE_RE.search(run)) if m]
    if not candidates:
        return [run]
    m = min(candidates, key=lambda mm: mm.start())
    left, right = run[: m.start()].strip(), run[m.end() :].strip()
    return [part for part in (left, right) if part]


def _extract_entities(text: str) -> set[str]:
    ents: set[str] = set()
    for regex in (_SINGLE_QUOTE_RE, _DOUBLE_QUOTE_RE):
        for m in regex.finditer(text):
            norm = _norm_entity(m.group(1))
            if norm:
                ents.add(norm)
    for m in _EMAIL_RE.finditer(text):
        norm = _norm_entity(m.group(0))
        if norm:
            ents.add(norm)
    # Lower the sentence-initial word so it never itself starts a false 2-word capitalized
    # run (e.g. "Star Sarah Chen's..." must not yield "star sarah" as well as "sarah chen").
    desensitized = text[:1].lower() + text[1:] if text else text
    for m in _CAP_RUN_RE.finditer(desensitized):
        for part in _split_possessive_run(m.group(0)):
            norm = _norm_entity(part)
            if norm:
                ents.add(norm)
    return ents


def task_entities(app: str) -> set[str]:
    """Quoted strings, email addresses, and capitalized multi-word names (possessive runs
    split into their component names) from `app`'s task instructions, minus any that already
    appear in the app's manual bundle (documented, so not a leak).
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
    case-insensitive occurrence of a task entity not immediately adjacent to a word
    character on either side. Fix round 1, issue 3: matched with ``(?<!\\w)``/``(?!\\w)``
    rather than ``\\b`` -- ``\\b`` never matches next to a *non*-word boundary character
    (e.g. an entity that itself starts with ``$`` or ends with a digit followed by more
    punctuation), silently letting such entities leak undetected.
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
        if re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", lowered):
            violations.append(f"entity leak: {entity!r}")
    return violations


# ---- K generation ---------------------------------------------------------------------

GENERATOR_MODEL = "claude-opus-4-7"
#: Fix round 1, issue 4: 16,000 was too tight for 60 candidates x <=250 words; raised to
#: 32,000 and requested via the SDK's streaming helper rather than a single blocking call.
GENERATOR_MAX_TOKENS = 32_000
K_MIN_WORDS, K_MAX_WORDS = 40, 250
#: Fix round 1, issue 3: raised from 10 to 20 (n + 20 = 60 for the default n=40) to absorb
#: the extra rejections from the now-fixed (more sensitive) leak guard.
N_REQUEST_SLACK = 20

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


class KnowledgeGenerationError(ValueError):
    """Raised by `generate_knowledge` when the generator's response can't be used (cut off,
    unparsable, or too few candidates survive the filters). Carries the same ``raw`` record
    `generate_knowledge` would otherwise have returned, so a caller (`build_profiles`) can
    persist the paid call's full record before re-raising, instead of losing it.
    """

    def __init__(self, message: str, raw: dict) -> None:
        super().__init__(message)
        self.raw = raw


def generate_knowledge(client, app: str, bundle: str, n: int = N_K) -> tuple[list[str], dict]:
    """Ask the generator for `n` + 20 knowledge extensions, drop leaks/out-of-range/dupes,
    keep the first `n`, and raise `KnowledgeGenerationError` (carrying the raw record so far)
    if fewer than `n` survive. Every rejection (including its reason) and the frozen entity
    set used to judge them are recorded in the returned dict. Writes nothing itself --
    `build_profiles` is responsible for persisting the record, on success or failure alike.
    """
    n_request = n + N_REQUEST_SLACK
    # `.replace`, not `.format`: the manual bundle routinely contains literal `{`/`}` (Hugo
    # shortcodes, JSON snippets) that would raise inside `str.format`.
    prompt = KNOWLEDGE_PROMPT.replace("{n}", str(n_request)).replace("{bundle}", bundle)

    with client.messages.stream(
        model=GENERATOR_MODEL,
        max_tokens=GENERATOR_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    stop_reason = getattr(response, "stop_reason", None)

    raw: dict = {
        "model": GENERATOR_MODEL,
        "max_tokens": GENERATOR_MAX_TOKENS,
        "app": app,
        "n_requested": n_request,
        "n": n,
        "prompt": prompt,
        "stop_reason": stop_reason,
        "response_text": text,
        "rejected": [],
        "entities": [],
    }

    if stop_reason == "max_tokens":
        raise KnowledgeGenerationError(
            f"the generator's response was cut off at max_tokens={GENERATOR_MAX_TOKENS}; "
            "refusing to use a truncated knowledge batch", raw)

    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise KnowledgeGenerationError("no JSON array in the generator's response", raw)
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise KnowledgeGenerationError(f"malformed JSON in the generator's response: {exc}", raw) from exc

    texts_for_app = task_texts(app)
    ents = task_entities(app)
    raw["entities"] = sorted(ents)

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

    raw["rejected"] = rejected

    if len(kept) < n:
        raise KnowledgeGenerationError(
            f"only {len(kept)} valid, non-leaking, distinct knowledge extensions out of "
            f"{len(items)} candidates; need {n}", raw)

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


# ---- profile assembly (never run from tests except with an injected fake client) -------


def build_bundle_files(out: Path) -> tuple[str, str]:
    gitlab_bundle = build_bundle("gitlab")
    gmail_bundle = build_bundle("gmail")
    (out / "bundles").mkdir(parents=True, exist_ok=True)
    (out / "bundles" / "gitlab.md").write_text(gitlab_bundle, encoding="utf-8")
    (out / "bundles" / "gmail.md").write_text(gmail_bundle, encoding="utf-8")
    return gitlab_bundle, gmail_bundle


def _knowledge_pool_paths(out: Path, app: str) -> tuple[Path, Path]:
    pool_name = "GLK" if app == "gitlab" else "GMK"
    return out / "pools" / f"{pool_name}.yaml", out / f"k_generation_{app}.json"


def _reuse_knowledge_pool(pool_path: Path, k_gen_path: Path) -> tuple[list[str], dict] | None:
    """The already-built K pool and its generation record, if both exist and the pool's
    frozen per-arm sha256 still verifies; ``None`` if either is missing, unreadable, or
    doesn't verify, so the caller falls back to calling the generator.
    """
    if not (pool_path.exists() and k_gen_path.exists()):
        return None
    try:
        pool_arms = load_pool(pool_path)
        raw = json.loads(k_gen_path.read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError):
        return None
    if len(pool_arms) != N_K:
        return None
    return [pa.arm.prompt_guidance for pa in pool_arms], raw


def _knowledge_pool(client, app: str, bundle: str, out: Path, force: bool) -> tuple[list[str], dict]:
    """`app`'s K-generation texts and record: reused from disk when not `force` and a
    verified pool + record already exist (fix round 1, issue 4's resume); otherwise
    generated, with the record persisted to ``k_generation_<app>.json`` before any
    `KnowledgeGenerationError` propagates, so a failed paid call is never lost.
    """
    pool_path, k_gen_path = _knowledge_pool_paths(out, app)
    if not force:
        reused = _reuse_knowledge_pool(pool_path, k_gen_path)
        if reused is not None:
            log.info("%s: reusing verified %s and %s; skipping the generator",
                     app, pool_path.name, k_gen_path.name)
            return reused
    try:
        texts, raw = generate_knowledge(client, app, bundle, n=N_K)
    except KnowledgeGenerationError as exc:
        k_gen_path.parent.mkdir(parents=True, exist_ok=True)
        k_gen_path.write_text(json.dumps(exc.raw, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    k_gen_path.parent.mkdir(parents=True, exist_ok=True)
    k_gen_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return texts, raw


def build_profiles(seed: int = SEED, out: Path = OUT_DIR, force: bool = False, client=None) -> None:
    """Build and freeze the prompt-heterogeneity study's pools and queues under `out`.

    `client` defaults to a real ``anthropic.Anthropic()`` (constructed lazily, after loading
    ``.env``, only if actually needed) but can be injected -- a fake for tests, or a
    pre-configured client for the controller. A *complete* prior build (``manifest.json``
    present) is frozen and refuses to rebuild unless `force`; an incomplete one always
    resumes (see `_knowledge_pool`). `force` also makes K-generation unconditional (both
    apps regenerated even if a verified pool already exists).
    """
    out = Path(out)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        raise SystemExit(f"a completed heterogeneity profile already exists under {out}; "
                         "frozen (pass --force to rebuild)")

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

    # ---- K generation: gitlab, gmail (each resumable; see _knowledge_pool). ----
    if client is None:
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        client = anthropic.Anthropic()

    gitlab_bundle, gmail_bundle = build_bundle_files(out)

    gitlab_texts, gitlab_raw = _knowledge_pool(client, "gitlab", gitlab_bundle, out, force)
    glk_arms = [freeform_arm(f"GLK_{i:02d}", t) for i, t in enumerate(gitlab_texts)]
    write_pool(out / "pools" / "GLK.yaml", "GLK", glk_arms, FREEFORM_TEMPLATE, AXES_PATH,
               meta={"generator": GENERATOR_MODEL, "app": "gitlab"})

    gmail_texts, gmail_raw = _knowledge_pool(client, "gmail", gmail_bundle, out, force)
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
    block_seed = seed + 1
    sixty = select_tasks(gitlab_bank, N_TASKS_GITLAB_BANK, seed)
    block_a = select_tasks(sixty, N_TASKS_BLOCK, block_seed)
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

    def _sections_for_manifest(app: str) -> list[dict]:
        return [{k: v for k, v in s.items() if k != "text"} for s in bundle_sections(app)]

    manifest = {
        "seed": seed,
        "block_seed": block_seed,
        "bundle_max_words": BUNDLE_MAX_WORDS,
        "real_tasks_sha256": {app: file_sha256(_real_tasks_path(app)) for app in APPS},
        "entity_set_sha256": {app: sha256_text(json.dumps(sorted(task_entities(app)))) for app in APPS},
        "bundles": {app: _sections_for_manifest(app) for app in APPS},
        "gitlab_task_bank": gitlab_bank,
        "gitlab_task_subset_60": sixty,
        "gitlab_block_a": block_a,
        "gitlab_block_b": block_b,
        "gmail_task_subset_30": gmail_task_subset,
        "bridge_source": bridge_source,
        "files": {str(p.relative_to(out)): file_sha256(p) for p in files},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
