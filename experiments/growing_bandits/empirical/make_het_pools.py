"""Pool builders for the prompt-heterogeneity study (spec section 4).

    .venv/bin/python experiments/growing_bandits/empirical/make_het_pools.py

writes, under ``data/heterogeneity/``:

- ``bundles/{gitlab,gmail}.md`` -- frozen manual bundles (user-manual pages only -- fix round
  2: GitLab's ``APP_DESCRIPTION.md`` is excluded entirely, like Gmail, since it describes
  this app *instance* -- its seeded users, epics, milestones, boards and labels -- which
  would otherwise exempt those names from the leak guard), capped at 25,000 words with an
  equal word budget per feature area (``APPS[app]["manual_areas"]``; unused budget rolls
  forward to the next area, never backward), used both as the K generator's context and as
  the leak guard's stoplist source (an entity that already appears in the bundle is not a
  leak). Within an area, files are in sorted-path priority order; a page byte-identical (or
  near-identical modulo its own self-referential "Source: <url>" line) to one already
  included is skipped.
- ``k_generation_{gitlab,gmail}.json`` -- the K generator's full request/response record
  (including every rejected candidate and why, the frozen entity set and bundle hash used to
  judge/identify it, and -- on success -- the kept texts), written only for a *successful*
  generation. Fix round 2: a failed generation is archived as
  ``k_generation_{gitlab,gmail}.failed-<n>.json`` instead, so the canonical path never holds
  an unreusable failure record and a later success is never blocked from being written there.
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
``--force`` -- see the K-generation skip-if-present note above. Fix round 2: any build that
proceeds (forced or resuming) unlinks ``manifest.json`` first and writes it only at
successful completion, so a failed ``--force`` run never leaves a stale ``manifest.json``
blocking a later resume; ``--force`` also unconditionally invalidates (deletes) each app's
canonical ``k_generation_<app>.json`` before attempting it, so a forced attempt that fails
never leaves behind a stale-but-still-verifiable cache that a later plain resume would
silently reuse.
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

#: Section 4 (amended before registration, spec sec. 4.2): what each app's manual bundle is
#: built from. ``manual_areas`` is a *priority* list of feature *areas*, each area a list of
#: one or more glob patterns sharing one word budget (fix round 2, ruling 2): resolved under
#: ``<webarena-root>/apps/user-manuals/<app>``, sorted within a pattern, in the order given; a
#: path already yielded by an earlier pattern (in this area or an earlier one) is skipped.
#: GitLab's ``issues/managing_issues.md`` and the rest of ``issues/**`` are one area (per the
#: ruling). Neither app has an ``app_description`` any more (fix round 2, finding 1): the
#: bundle is user-manual pages only, like Gmail always was -- GitLab's ``APP_DESCRIPTION.md``
#: describes this app *instance* (its seeded users/epics/milestones/boards/labels), which
#: would otherwise exempt those names from the leak guard. ``app_description``, when set, is
#: a single file relative to the webarena-infinity root, treated as its own (first) area.
APPS: dict[str, dict] = {
    "gitlab": {
        "web_app": "apps/gitlab-plan-and-track",
        "manual_areas": [
            ["user/project/labels.md"],
            ["user/project/issue_board.md"],
            ["user/project/milestones/**/*.md"],
            ["user/project/issues/managing_issues.md", "user/project/issues/**/*.md"],
            ["user/group/epics/**/*.md"],
            ["user/group/iterations/**/*.md"],
        ],
        "app_description": None,
    },
    "gmail": {
        "web_app": "apps/gmail",
        "manual_areas": [
            ["settings-and-configuration/*.md"],
            ["organize-and-manage/*.md"],
            ["compose-and-send/*.md"],
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

#: A manual page's own self-referential "Source: <url ending in this file's doc id>" line --
#: stripped before hashing for *dedup* purposes only (fix round 1, issue 2), so two mirrored
#: copies of the same support page (different doc ids under different section folders) are
#: recognized as duplicates despite not being byte-identical. The section's recorded
#: ``sha256`` (for the manifest) is still the true, unmodified content hash.
_SOURCE_LINE_RE = re.compile(r"^Source:\s*\S+\s*$", re.MULTILINE)


def _dedup_key(body: str) -> str:
    return sha256_text(_SOURCE_LINE_RE.sub("", body))


def _assemble_bundle(areas: list[list[tuple[str, str]]], max_words: int = BUNDLE_MAX_WORDS) -> list[dict]:
    """Ordered, deduplicated, per-area-budgeted, capped bundle sections. `areas` is a
    priority-ordered list of areas, each already in within-area (sorted) file order as
    ``[(rel_path, body), ...]``. A `rel_path` already used (in this area or an earlier one),
    or a `body` that near-duplicates one already included (by ``_dedup_key``), is skipped.

    Fix round 2, ruling 2: each area gets an equal share of the total budget
    (``max_words // len(areas)``, floor); words an area doesn't use roll forward to the next
    area (never backward, so an early thin area doesn't starve everything after it, and a fat
    one doesn't get to starve everything after it either -- it only spends its own share).
    Each section is headed ``## <rel_path>``; the header counts against its area's (and hence
    the overall) budget. The file that would cross an area's remaining budget is truncated to
    exactly fill what's left, and that area is then done (any further files in it are
    dropped, its budget having already been exhausted). By construction the grand total never
    exceeds `max_words` (each area's own budget is drawn only from `max_words // len(areas)`
    plus unspent rollover from earlier areas).
    """
    n_areas = len(areas)
    if n_areas == 0:
        return []
    budget_each = max_words // n_areas

    sections: list[dict] = []
    seen_paths: set[str] = set()
    seen_content: set[str] = set()
    rollover = 0

    for items in areas:
        area_budget = budget_each + rollover
        area_used = 0
        for rel_path, body in items:
            if rel_path in seen_paths:
                continue
            seen_paths.add(rel_path)
            key = _dedup_key(body)
            if key in seen_content:
                continue
            seen_content.add(key)

            remaining = area_budget - area_used
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
                area_used += included
                break
            text = header + body
            sections.append({"path": rel_path, "sha256": content_sha, "words": total_words,
                             "truncated": False, "text": text})
            area_used += total_words
        rollover = area_budget - area_used

    return sections


def bundle_sections(app: str, max_words: int = BUNDLE_MAX_WORDS) -> list[dict]:
    """`app`'s bundle sections: the app description as its own (first) area, if set, then its
    manuals in ``APPS[app]['manual_areas']`` priority order, each area budgeted per
    `_assemble_bundle`. Each entry is ``{"path", "sha256", "words", "truncated", "text"}``.
    """
    webarena_root = _webarena_root()
    spec = APPS[app]
    manual_root = webarena_root / "apps" / "user-manuals" / app
    seen: set[Path] = set()
    areas: list[list[tuple[str, str]]] = []

    if spec.get("app_description"):
        p = webarena_root / spec["app_description"]
        seen.add(p)
        areas.append([(p.relative_to(webarena_root).as_posix(),
                       p.read_text(encoding="utf-8", errors="ignore"))])

    for globs in spec["manual_areas"]:
        area_items: list[tuple[str, str]] = []
        for pattern in globs:
            matched = sorted(manual_root.glob(pattern))
            if not matched:
                raise ValueError(f"{app!r}: manual glob {pattern!r} under {manual_root} matched nothing")
            for p in matched:
                if p in seen:
                    continue
                seen.add(p)
                area_items.append((p.relative_to(webarena_root).as_posix(),
                                   p.read_text(encoding="utf-8", errors="ignore")))
        areas.append(area_items)

    return _assemble_bundle(areas, max_words)


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

    Fix round 2, finding 4: `entities` (and `bundle_sha256`) are computed up front, before the
    network call and before the response is parsed, so *every* raw record -- including one
    from an early failure (cut off, no JSON array, malformed JSON) -- carries them; a caller
    checking whether a cached record is still valid for reuse needs them regardless of which
    stage produced the record. `raw["error"]` is `None` on success and the failure message
    otherwise; `raw["kept"]` (the final kept texts, in order) is set only on success, after
    the `n`-th one is kept, so it always matches the pool that gets written from it.
    """
    n_request = n + N_REQUEST_SLACK
    # `.replace`, not `.format`: the manual bundle routinely contains literal `{`/`}` (Hugo
    # shortcodes, JSON snippets) that would raise inside `str.format`.
    prompt = KNOWLEDGE_PROMPT.replace("{n}", str(n_request)).replace("{bundle}", bundle)
    ents = task_entities(app)

    raw: dict = {
        "model": GENERATOR_MODEL,
        "max_tokens": GENERATOR_MAX_TOKENS,
        "app": app,
        "n_requested": n_request,
        "n": n,
        "prompt": prompt,
        "bundle_sha256": sha256_text(bundle),
        "stop_reason": None,
        "response_text": None,
        "rejected": [],
        "entities": sorted(ents),
        "error": None,
    }

    with client.messages.stream(
        model=GENERATOR_MODEL,
        max_tokens=GENERATOR_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    stop_reason = getattr(response, "stop_reason", None)
    raw["stop_reason"] = stop_reason
    raw["response_text"] = text

    if stop_reason == "max_tokens":
        raw["error"] = (f"the generator's response was cut off at max_tokens={GENERATOR_MAX_TOKENS}; "
                        "refusing to use a truncated knowledge batch")
        raise KnowledgeGenerationError(raw["error"], raw)

    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raw["error"] = "no JSON array in the generator's response"
        raise KnowledgeGenerationError(raw["error"], raw)
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raw["error"] = f"malformed JSON in the generator's response: {exc}"
        raise KnowledgeGenerationError(raw["error"], raw) from exc

    texts_for_app = task_texts(app)

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
        raw["error"] = (f"only {len(kept)} valid, non-leaking, distinct knowledge extensions out of "
                        f"{len(items)} candidates; need {n}")
        raise KnowledgeGenerationError(raw["error"], raw)

    raw["kept"] = kept
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


def _archive_failed_generation(k_gen_path: Path, raw: dict) -> Path:
    """Persist a failed generation `raw` record as ``<k_gen_path.stem>.failed-<n>.json`` (`n`
    = the next free integer) rather than the canonical ``k_generation_<app>.json`` path (fix
    round 2, ruling 3). The canonical path is thereby only ever written on a genuine success,
    so it can never be mistaken for a reusable record, and a later success is never blocked
    from being written there.
    """
    k_gen_path.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = k_gen_path.with_name(f"{k_gen_path.stem}.failed-{n}.json")
        if not candidate.exists():
            break
        n += 1
    candidate.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return candidate


def _reuse_knowledge_pool(pool_path: Path, k_gen_path: Path, bundle: str, app: str) -> tuple[list[str], dict] | None:
    """The already-built K pool and its generation record, if reusable; ``None`` if either
    file is missing/unreadable, the pool's frozen per-arm sha256 doesn't verify, the record is
    a failure (``raw["error"]`` set), or the record's kept texts don't match the pool's
    ``prompt_guidance`` texts in order (fix round 2, ruling 3) -- any of these means "not a
    resumable prior success," so the caller falls back to calling the generator.

    If the record otherwise verifies (no error, kept texts == pool texts) but its
    `bundle_sha256`/`entities` no longer match the *current* `bundle`/`task_entities(app)`
    (fix round 2, finding 4) -- e.g. the manual bundle or task instructions changed since it
    was generated -- this raises `SystemExit` rather than silently reusing stale content or
    silently paying to regenerate it: the drift needs an explicit `--force`.
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
    if raw.get("error") is not None:
        return None
    pool_texts = [pa.arm.prompt_guidance for pa in pool_arms]
    if raw.get("kept") != pool_texts:
        return None

    expected_bundle_sha = sha256_text(bundle)
    expected_entities = sorted(task_entities(app))
    if raw.get("bundle_sha256") != expected_bundle_sha or raw.get("entities") != expected_entities:
        raise SystemExit(
            f"{app}: {k_gen_path} verifies against {pool_path} but its bundle hash or entity "
            "set no longer matches the current inputs (the manual bundle or task instructions "
            "changed since it was generated); pass --force to regenerate")
    return pool_texts, raw


def _knowledge_pool(client, app: str, bundle: str, out: Path, force: bool) -> tuple[list[str], dict]:
    """`app`'s K-generation texts and record: reused from disk when not `force` and a
    verified pool + record already exist (fix round 1, issue 4's resume; fix round 2,
    ruling 3's stricter verification); otherwise generated, with the record persisted to
    ``k_generation_<app>.json`` on success or archived (never overwriting the canonical path)
    on failure, so a failed paid call is never lost and never masquerades as reusable.

    `force` unconditionally invalidates (deletes) the canonical record before attempting --
    "force" means "regenerate, discard whatever is cached" -- so a forced attempt that fails
    never leaves a stale-but-still-verifiable cache behind for a later plain resume to
    silently reuse.
    """
    pool_path, k_gen_path = _knowledge_pool_paths(out, app)
    if force:
        k_gen_path.unlink(missing_ok=True)
    else:
        reused = _reuse_knowledge_pool(pool_path, k_gen_path, bundle, app)
        if reused is not None:
            log.info("%s: reusing verified %s and %s; skipping the generator",
                     app, pool_path.name, k_gen_path.name)
            return reused
    try:
        texts, raw = generate_knowledge(client, app, bundle, n=N_K)
    except KnowledgeGenerationError as exc:
        archived = _archive_failed_generation(k_gen_path, exc.raw)
        log.warning("%s: knowledge generation failed; archived the failed record to %s "
                    "(canonical %s left untouched)", app, archived.name, k_gen_path.name)
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
    if manifest_path.exists():
        if not force:
            raise SystemExit(f"a completed heterogeneity profile already exists under {out}; "
                             "frozen (pass --force to rebuild)")
        # Fix round 2, ruling 3: unlink before proceeding, and only ever write it again at
        # successful completion -- otherwise a failed --force run leaves the *previous*
        # manifest.json in place, which then wrongly blocks a later plain resume.
        manifest_path.unlink()

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

    # Fix round 2, finding 4: the entity-set hash is computed from the generation record
    # *actually used* for each app's K pool (`raw["entities"]`, frozen when that record was
    # generated/reused) rather than a fresh `task_entities(app)` call, so the manifest
    # reflects what actually gated the K pool, not whatever the entity heuristic would
    # compute right now.
    k_raw_by_app = {"gitlab": gitlab_raw, "gmail": gmail_raw}

    manifest = {
        "seed": seed,
        "block_seed": block_seed,
        "bundle_max_words": BUNDLE_MAX_WORDS,
        "real_tasks_sha256": {app: file_sha256(_real_tasks_path(app)) for app in APPS},
        "entity_set_sha256": {app: sha256_text(json.dumps(k_raw_by_app[app]["entities"])) for app in APPS},
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
