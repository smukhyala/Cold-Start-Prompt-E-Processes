"""Pool builders for the prompt-heterogeneity study: bundles, K generation with a leak guard,
G/anchor reuse, and per-profile queues."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
import make_het_pools as mh  # noqa: E402
import make_pools as mp  # noqa: E402

# ---- fake Anthropic client -------------------------------------------------------------
#
# `generate_knowledge` calls `client.messages.stream(...)` as a context manager and reads
# `.get_final_message()`; these fakes support both `.stream` and `.create` (fix round 1,
# issue 4) so a single fake works regardless of which the implementation calls.


class _FakeStream:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._response


class _FakeMessages:
    def __init__(self, response_text: str, stop_reason: str = "end_turn"):
        self._response_text = response_text
        self._stop_reason = stop_reason

    def _response(self):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._response_text)],
                               stop_reason=self._stop_reason)

    def create(self, **kw):
        assert "temperature" not in kw
        return self._response()

    def stream(self, **kw):
        assert "temperature" not in kw
        return _FakeStream(self._response())


class _FakeClient:
    def __init__(self, response_text: str, stop_reason: str = "end_turn"):
        self.messages = _FakeMessages(response_text, stop_reason)


def test_leak_guard_rejects_task_ngrams_and_entities():
    tasks = ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label."]
    ents = {"sarah chen", "q1 product roadmap"}
    clean = "Use the star icon beside a message, then confirm the label list shows your change. " * 3
    leaky_ngram = "Star Sarah Chen's Q1 product roadmap email and move it to the inbox."
    leaky_entity = "When a message from Sarah Chen arrives, archive it after reading."
    assert mh.leak_violations(clean, tasks, ents) == []
    assert any("8-gram" in v for v in mh.leak_violations(leaky_ngram, tasks, ents))
    assert any("entity" in v for v in mh.leak_violations(leaky_entity, tasks, ents))


def test_leak_guard_matches_entities_with_punctuation_boundaries():
    """Fix round 1, issue 3: `\\b` never matches next to a non-word boundary character, so
    an entity that itself starts/ends with punctuation (e.g. a leading "$") could never be
    detected under the old `\\bENTITY\\b` matching. `(?<!\\w)`/`(?!\\w)` fixes this.
    """
    text = "Reply to the note about the $5,000,000 inheritance right away."
    violations = mh.leak_violations(text, [], {"$5,000,000 inheritance"})
    assert any("entity" in v for v in violations)


def test_generate_knowledge_filters_and_records(monkeypatch):
    good = [f"Guidance {i}: open the sidebar, pick the feature, confirm the banner. " + "word " * 40 for i in range(45)]
    bad = ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label. " + "word " * 40]

    monkeypatch.setattr(mh, "task_texts", lambda app: ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label."])
    monkeypatch.setattr(mh, "task_entities", lambda app: {"sarah chen"})
    client = _FakeClient(json.dumps(bad + good))
    texts, raw = mh.generate_knowledge(client, "gmail", "bundle text", n=40)
    assert len(texts) == 40 and all("Sarah" not in t for t in texts)
    assert raw["rejected"] and raw["rejected"][0]["reason"]
    assert raw["entities"] == ["sarah chen"]


def test_generate_knowledge_raises_when_too_few(monkeypatch):
    monkeypatch.setattr(mh, "task_texts", lambda app: [])
    monkeypatch.setattr(mh, "task_entities", lambda app: set())
    client = _FakeClient(json.dumps(["too short"] * 60))
    with pytest.raises(mh.KnowledgeGenerationError) as excinfo:
        mh.generate_knowledge(client, "gmail", "b", n=40)
    # Fix round 1, issue 4: the raw record (including every rejection) is attached to the
    # exception so a failed paid call is never lost.
    assert len(excinfo.value.raw["rejected"]) == 60
    assert all(r["reason"] for r in excinfo.value.raw["rejected"])


def test_generate_knowledge_uses_the_streaming_helper(monkeypatch):
    """Fix round 1, issue 4: `generate_knowledge` must call `client.messages.stream(...)`
    (not `.create`), so a fake exposing only `.stream` still works.
    """
    good = [f"Guidance {i}: open the sidebar, pick the feature, confirm the banner. " + "word " * 40 for i in range(40)]

    class _StreamOnlyMessages:
        def stream(self, **kw):
            assert kw["max_tokens"] == mh.GENERATOR_MAX_TOKENS
            return _FakeStream(SimpleNamespace(
                content=[SimpleNamespace(type="text", text=json.dumps(good))], stop_reason="end_turn"))

    monkeypatch.setattr(mh, "task_texts", lambda app: [])
    monkeypatch.setattr(mh, "task_entities", lambda app: set())
    client = SimpleNamespace(messages=_StreamOnlyMessages())
    texts, raw = mh.generate_knowledge(client, "gmail", "bundle text", n=40)
    assert len(texts) == 40
    assert mh.GENERATOR_MAX_TOKENS == 32_000


def test_bundle_is_capped_and_deterministic():
    a = mh.build_bundle("gmail", max_words=3000)
    assert len(a.split()) <= 3000 + 200 and a == mh.build_bundle("gmail", max_words=3000)
    assert a.startswith("## ")


def _gitlab_design():
    arms = {"GLG": [f"GLG_{i:02d}" for i in range(50)], "GLK": [f"GLK_{i:02d}" for i in range(40)],
            "anchor": ["GL_anchor_baseline", "GL_anchor_explorer", "GL_anchor_oracle"]}
    block_a = [f"task_e{i}" for i in range(10)] + [f"task_m{i}" for i in range(10)] + [f"task_h{i}" for i in range(10)]
    block_b = [f"task_e{i}" for i in range(10, 20)] + [f"task_m{i}" for i in range(10, 20)] + [f"task_h{i}" for i in range(10, 20)]
    return arms, block_a, block_b


def test_gitlab_queue_blocks_and_pilot():
    arms, block_a, block_b = _gitlab_design()
    q = mh.gitlab_queue(arms, block_a, block_b, seed=1)
    n_pilot = sum(i.pilot for i in q)
    assert n_pilot == 13 * 30 and all(i.pilot for i in q[:n_pilot])
    main = [i for i in q if i.replicate == 0]
    assert len(main) == (50 + 40 + 3) * 60
    first_b = next(k for k, i in enumerate(q) if i.task_id in block_b)
    assert all(i.task_id in block_a for i in q[:first_b])
    reps = [i for i in q if i.replicate == 1]
    assert sum(i.pool == "GLG" for i in reps) == 300 and sum(i.pool == "GLK" for i in reps) == 240
    assert all(i.index == k for k, i in enumerate(q))


def test_gitlab_queue_stratifies_replicates_by_block():
    # I1 ruling: half of each pool's replicate cells from block-A main cells, queued right after the rest of
    # block A; half from block B, queued after block B. The pilot stays first.
    arms, block_a, block_b = _gitlab_design()
    q = mh.gitlab_queue(arms, block_a, block_b, seed=1)
    a, b = set(block_a), set(block_b)
    n_pilot = sum(i.pilot for i in q)
    n_a_main = (50 + 40 + 3) * 30
    n_a_reps = (300 + 240) // 2
    seg_rest_a = q[n_pilot:n_a_main]
    seg_reps_a = q[n_a_main:n_a_main + n_a_reps]
    seg_b = q[n_a_main + n_a_reps:n_a_main + n_a_reps + n_a_main]
    seg_reps_b = q[n_a_main + n_a_reps + n_a_main:]
    assert all(i.replicate == 0 and i.task_id in a for i in seg_rest_a)
    assert all(i.replicate == 1 and i.task_id in a for i in seg_reps_a)
    assert all(i.replicate == 0 and i.task_id in b for i in seg_b)
    assert all(i.replicate == 1 and i.task_id in b for i in seg_reps_b) and len(seg_reps_b) == n_a_reps
    for seg in (seg_reps_a, seg_reps_b):
        assert sum(i.pool == "GLG" for i in seg) == 150 and sum(i.pool == "GLK" for i in seg) == 120
        assert not any(i.pool == "anchor" for i in seg)
    # every replicate repeats a main cell of its own pool, without replacement
    main = {(i.pool, i.arm_id, i.task_id) for i in q if i.replicate == 0}
    reps = [(i.pool, i.arm_id, i.task_id) for i in q if i.replicate == 1]
    assert set(reps) <= main and len(set(reps)) == len(reps)
    # deterministic
    assert q == mh.gitlab_queue(arms, block_a, block_b, seed=1)
    assert q != mh.gitlab_queue(arms, block_a, block_b, seed=2)


def test_gitlab_stage_ends_names_the_through_index_of_block_a_and_its_replicates():
    arms, block_a, block_b = _gitlab_design()
    q = mh.gitlab_queue(arms, block_a, block_b, seed=1)
    ends = mh.gitlab_stage_ends(q, block_a)
    assert ends["pilot"] == 13 * 30 - 1
    assert ends["block_a_with_replicates"] == (50 + 40 + 3) * 30 + 270 - 1
    assert ends["last"] == len(q) - 1
    upto = q[:ends["block_a_with_replicates"] + 1]
    assert all(i.task_id in set(block_a) for i in upto) and q[ends["block_a_with_replicates"] + 1].task_id in set(block_b)
    # a queue whose block-A items are not one prefix is refused (the stage boundary would be meaningless)
    broken = [q[-1], *q[:-1]]
    with pytest.raises(ValueError, match="prefix"):
        mh.gitlab_stage_ends(broken, block_a)


# ---- fix round 2, finding 1: APP_DESCRIPTION.md excluded entirely (manual-only bundle) -


def test_gitlab_bundle_excludes_app_description_and_frees_seed_names():
    """The old seed-data-stripping approach (fix round 1) left APP_DESCRIPTION.md's non-
    "Seed Data Summary" sections (e.g. "Boards") in the bundle, which kept exempting some
    seeded instance names (like the "Bug Triage" board) from the leak guard. The ruling for
    this round instead drops APP_DESCRIPTION.md from the bundle entirely -- GitLab's bundle
    is user-manual pages only, exactly like Gmail's always was.
    """
    bundle = mh.build_bundle("gitlab")
    names = ("Platform Redesign", "Security Hardening", "Bug Triage", "Sprint 26",
             "v4.0 - Platform Redesign", "AcmeCorp", "platform-v4")
    for name in names:
        assert name not in bundle, name

    ents = mh.task_entities("gitlab")
    tasks = mh.task_texts("gitlab")
    occurs_in_a_task = [n for n in names if any(n.lower() in t.lower() for t in tasks)]
    assert occurs_in_a_task  # sanity: real-tasks.json still names these seed-data entities

    # "Sprint 26" is a real occurring seed name that the *entity extractor* (unchanged this
    # round; its capitalized-run regex requires every word of a run to start with a letter)
    # cannot represent on its own -- a numeral-suffixed name was never extractable, with or
    # without APP_DESCRIPTION.md in the bundle. That is a pre-existing extractor scope limit
    # (the same category as fix round 1's noted 4-word cap-run limit), not something this
    # finding's ruling (which is about the APP_DESCRIPTION.md exemption, not extractor
    # coverage) asked to fix -- so it is excluded from the flagged-by-leak_violations check
    # below rather than asserted falsely.
    checked = [n for n in occurs_in_a_task if n != "Sprint 26"]
    assert checked  # sanity: still exercises the fix (e.g. "Bug Triage", "Platform Redesign")
    for name in checked:
        assert any("entity" in v for v in mh.leak_violations(f"Update the {name} item.", [], ents)), name


# ---- fix round 2, ruling 2: per-area word budgets (rolling forward) --------------------


def test_assemble_bundle_single_area_priority_order_dedup_and_hard_cap():
    areas = [[
        ("a.md", "Alpha " * 5),
        ("b.md", "Beta " * 5),
        ("a.md", "Alpha duplicate path, must be skipped regardless of its content " * 5),
        ("c.md", "Beta " * 5),  # byte-identical body to b.md -> deduped by content
        ("d.md", "Delta " * 500),
    ]]
    sections = mh._assemble_bundle(areas, max_words=20)
    paths = [s["path"] for s in sections]
    assert paths == ["a.md", "b.md", "d.md"]
    assert "c.md" not in paths
    assert sum(s["words"] for s in sections) <= 20
    assert sum(len(s["text"].split()) for s in sections) <= 20
    assert sections[-1]["truncated"] is True


def test_assemble_bundle_multi_area_budget_and_rollover():
    """budget = cap // number_of_areas per area, in priority order; an area's unused budget
    rolls forward (never backward), so the small early areas here let the last, larger area
    use more than its own bare 1/3 share before being truncated.
    """
    areas = [
        [("a.md", "Alpha " * 3)],
        [("b.md", "Beta " * 3)],
        [("c.md", "Charlie " * 100)],
    ]
    sections = mh._assemble_bundle(areas, max_words=30)
    paths = [s["path"] for s in sections]
    assert paths == ["a.md", "b.md", "c.md"]
    assert sections[0]["truncated"] is False
    assert sections[1]["truncated"] is False
    assert sections[2]["truncated"] is True
    assert sum(s["words"] for s in sections) <= 30


def test_assemble_bundle_dedups_path_across_areas():
    areas = [[("a.md", "Alpha " * 5)], [("a.md", "Alpha duplicate path, must be skipped " * 5)]]
    sections = mh._assemble_bundle(areas, max_words=1000)
    assert [s["path"] for s in sections] == ["a.md"]


def test_assemble_bundle_dedups_near_identical_content_ignoring_source_line():
    """The webarena manual pages end in a self-referential "Source: <this page's own doc
    id>" line, so two mirrored copies of the same page are near- but not byte-identical.
    """
    body_a = "Some shared procedural content about a feature.\n\nSource: https://example.com/111\n"
    body_b = "Some shared procedural content about a feature.\n\nSource: https://example.com/222\n"
    assert body_a != body_b  # not byte-identical
    sections = mh._assemble_bundle([[("a.md", body_a), ("b.md", body_b)]], max_words=1000)
    assert [s["path"] for s in sections] == ["a.md"]


def test_gitlab_bundle_every_area_represented_and_priority_order():
    """Ruling 2: every area of the priority list is represented in the real bundle (labels,
    issue_board, milestones, issues, epics, iterations), in that order, under the default
    25,000-word cap and its equal-per-area budget with rollover -- and APP_DESCRIPTION.md
    (finding 1) never appears.
    """
    sections = mh.bundle_sections("gitlab")
    paths = [s["path"] for s in sections]

    def idx(suffix):
        return next(i for i, p in enumerate(paths) if p.endswith(suffix))

    for suffix in ("user/project/labels.md", "user/project/issue_board.md",
                   "user/project/milestones/_index.md", "user/project/issues/managing_issues.md",
                   "user/group/epics/_index.md", "user/group/iterations/_index.md"):
        assert any(p.endswith(suffix) for p in paths), suffix

    assert idx("user/project/labels.md") < idx("user/project/issue_board.md")
    assert idx("user/project/issue_board.md") < idx("user/project/milestones/_index.md")
    assert idx("user/project/milestones/_index.md") < idx("user/project/issues/managing_issues.md")
    assert idx("user/project/issues/managing_issues.md") < idx("user/group/epics/_index.md")
    assert idx("user/group/epics/_index.md") < idx("user/group/iterations/_index.md")

    assert not any(p.endswith("APP_DESCRIPTION.md") for p in paths)
    assert sum(s["words"] for s in sections) <= mh.BUNDLE_MAX_WORDS


def test_gmail_bundle_every_area_represented_including_compose_and_send():
    """Ruling 2: under the old plain-priority-order cap (fix round 1), Gmail's compose-and-
    send area was dropped entirely once settings-and-configuration and organize-and-manage
    consumed the whole 25,000-word cap. With an equal per-area budget it now always gets a
    (possibly truncated) share.
    """
    sections = mh.bundle_sections("gmail")
    paths = [s["path"] for s in sections]
    assert any("settings-and-configuration/" in p for p in paths)
    assert any("organize-and-manage/" in p for p in paths)
    assert any("compose-and-send/" in p for p in paths)
    first_settings = next(i for i, p in enumerate(paths) if "settings-and-configuration/" in p)
    first_organize = next(i for i, p in enumerate(paths) if "organize-and-manage/" in p)
    first_compose = next(i for i, p in enumerate(paths) if "compose-and-send/" in p)
    assert first_settings < first_organize < first_compose
    assert sum(s["words"] for s in sections) <= mh.BUNDLE_MAX_WORDS


def test_gmail_bundle_dedups_near_identical_pages():
    # A cap well beyond gmail's ~53k raw-total word count so nothing is truncated away,
    # isolating the dedup effect (fix round 1, issue 2's two cited near-duplicate pairs).
    sections = mh.bundle_sections("gmail", max_words=80_000)
    paths = [s["path"] for s in sections]
    mail_merge = [p for p in paths if p.endswith("send-personalized-emails-with-mail-merge.md")]
    branded = [p for p in paths if p.endswith("create-branded-emails-with-customized-layouts.md")]
    assert len(mail_merge) == 1
    assert len(branded) == 1


def test_bundle_hard_cap_counts_header_words():
    a = mh.build_bundle("gitlab", max_words=500)
    assert len(a.split()) <= 500


def test_bundle_determinism_at_default_cap():
    assert mh.build_bundle("gitlab") == mh.build_bundle("gitlab")
    assert mh.build_bundle("gmail") == mh.build_bundle("gmail")


# ---- fix round 1, issue 3: possessive splitting + email entities -----------------------


def test_split_possessive_run_emits_both_sides():
    assert mh._split_possessive_run("Marcus Williams' Design System") == ["Marcus Williams", "Design System"]
    assert mh._split_possessive_run("Sarah Chen's Roadmap") == ["Sarah Chen", "Roadmap"]
    assert mh._split_possessive_run("Frontend Modernization") == ["Frontend Modernization"]


def test_extract_entities_splits_possessive_run_and_leak_guard_catches_the_name_alone():
    text = "Mark Marcus Williams' Design System as read."
    ents = mh._extract_entities(text)
    assert "marcus williams" in ents
    assert "design system" in ents
    # The whole un-split run must no longer be the only representation.
    assert "marcus williams' design system" not in ents

    # And the leak guard now catches a leak of just the person's name.
    leak_text = "Reassign the ticket to Marcus Williams before end of day."
    assert any("entity" in v for v in mh.leak_violations(leak_text, [], ents))


def test_extract_entities_extracts_email_addresses():
    ents = mh._extract_entities("Unblock the sender annoying@daily-deals.biz.")
    assert "annoying@daily-deals.biz" in ents


def test_generate_knowledge_requests_n_plus_20():
    assert mh.N_REQUEST_SLACK == 20


# ---- fix round 2, finding 4: entities/bundle hash computed before parsing --------------


def test_generate_knowledge_raw_carries_entities_and_bundle_hash_even_on_early_failure(monkeypatch):
    """`entities` and `bundle_sha256` are computed before the response is parsed, so a raw
    record from an EARLY failure (here: no JSON array at all) still carries them --
    previously only a late failure (too-few-survive, past the parsing/filtering step) did.
    """
    monkeypatch.setattr(mh, "task_texts", lambda app: ["Reassign this to Sarah Chen today."])
    monkeypatch.setattr(mh, "task_entities", lambda app: {"sarah chen"})
    client = _FakeClient("not json at all, no brackets here")
    with pytest.raises(mh.KnowledgeGenerationError) as excinfo:
        mh.generate_knowledge(client, "gmail", "bundle text", n=40)
    assert excinfo.value.raw["entities"] == ["sarah chen"]
    assert excinfo.value.raw["bundle_sha256"] == mh.sha256_text("bundle text")
    assert excinfo.value.raw["error"]


def test_generate_knowledge_records_error_none_and_kept_on_success(monkeypatch):
    good = [f"Guidance {i}: open the sidebar, pick the feature, confirm the banner. " + "word " * 40 for i in range(40)]
    monkeypatch.setattr(mh, "task_texts", lambda app: [])
    monkeypatch.setattr(mh, "task_entities", lambda app: set())
    client = _FakeClient(json.dumps(good))
    texts, raw = mh.generate_knowledge(client, "gmail", "bundle text", n=40)
    assert raw["error"] is None
    assert raw["kept"] == texts
    assert raw["bundle_sha256"] == mh.sha256_text("bundle text")


# ---- fix round 2, ruling 3: reuse verification, failure archiving, safe resume ---------


def test_reuse_knowledge_pool_accepts_a_verified_matching_record(tmp_path, monkeypatch):
    monkeypatch.setattr(mh, "task_entities", lambda app: {"x"})
    texts = [f"Guidance {i} about the feature. " + "word " * 40 for i in range(40)]
    arms = [mh.freeform_arm(f"GMK_{i:02d}", t) for i, t in enumerate(texts)]
    pool_path = tmp_path / "GMK.yaml"
    mh.write_pool(pool_path, "GMK", arms, mh.FREEFORM_TEMPLATE, mh.AXES_PATH, meta={})
    k_gen_path = tmp_path / "k_generation_gmail.json"
    k_gen_path.write_text(json.dumps({
        "kept": texts, "error": None, "bundle_sha256": mh.sha256_text("bundle"), "entities": ["x"],
    }))
    reused = mh._reuse_knowledge_pool(pool_path, k_gen_path, "bundle", "gmail")
    assert reused is not None
    got_texts, got_raw = reused
    assert got_texts == texts
    assert got_raw["entities"] == ["x"]


def test_reuse_knowledge_pool_rejects_a_failure_record(tmp_path, monkeypatch):
    monkeypatch.setattr(mh, "task_entities", lambda app: {"x"})
    texts = [f"Guidance {i} about the feature. " + "word " * 40 for i in range(40)]
    arms = [mh.freeform_arm(f"GMK_{i:02d}", t) for i, t in enumerate(texts)]
    pool_path = tmp_path / "GMK.yaml"
    mh.write_pool(pool_path, "GMK", arms, mh.FREEFORM_TEMPLATE, mh.AXES_PATH, meta={})
    k_gen_path = tmp_path / "k_generation_gmail.json"
    k_gen_path.write_text(json.dumps({
        "kept": texts, "error": "boom", "bundle_sha256": mh.sha256_text("bundle"), "entities": ["x"],
    }))
    assert mh._reuse_knowledge_pool(pool_path, k_gen_path, "bundle", "gmail") is None


def test_reuse_knowledge_pool_rejects_mismatched_kept_texts(tmp_path, monkeypatch):
    monkeypatch.setattr(mh, "task_entities", lambda app: {"x"})
    texts = [f"Guidance {i} about the feature. " + "word " * 40 for i in range(40)]
    arms = [mh.freeform_arm(f"GMK_{i:02d}", t) for i, t in enumerate(texts)]
    pool_path = tmp_path / "GMK.yaml"
    mh.write_pool(pool_path, "GMK", arms, mh.FREEFORM_TEMPLATE, mh.AXES_PATH, meta={})
    k_gen_path = tmp_path / "k_generation_gmail.json"
    k_gen_path.write_text(json.dumps({
        "kept": ["totally different text"] * 40, "error": None,
        "bundle_sha256": mh.sha256_text("bundle"), "entities": ["x"],
    }))
    assert mh._reuse_knowledge_pool(pool_path, k_gen_path, "bundle", "gmail") is None


def test_reuse_knowledge_pool_refuses_loudly_when_inputs_drifted(tmp_path, monkeypatch):
    monkeypatch.setattr(mh, "task_entities", lambda app: {"x"})
    texts = [f"Guidance {i} about the feature. " + "word " * 40 for i in range(40)]
    arms = [mh.freeform_arm(f"GMK_{i:02d}", t) for i, t in enumerate(texts)]
    pool_path = tmp_path / "GMK.yaml"
    mh.write_pool(pool_path, "GMK", arms, mh.FREEFORM_TEMPLATE, mh.AXES_PATH, meta={})
    k_gen_path = tmp_path / "k_generation_gmail.json"
    k_gen_path.write_text(json.dumps({
        "kept": texts, "error": None, "bundle_sha256": "a-stale-hash", "entities": ["x"],
    }))
    with pytest.raises(SystemExit, match="force"):
        mh._reuse_knowledge_pool(pool_path, k_gen_path, "bundle", "gmail")


def test_archive_failed_generation_uses_next_free_integer_without_touching_canonical(tmp_path):
    k_gen_path = tmp_path / "k_generation_gmail.json"
    p1 = mh._archive_failed_generation(k_gen_path, {"error": "first"})
    p2 = mh._archive_failed_generation(k_gen_path, {"error": "second"})
    assert p1.name == "k_generation_gmail.failed-1.json"
    assert p2.name == "k_generation_gmail.failed-2.json"
    assert not k_gen_path.exists()
    assert json.loads(p1.read_text())["error"] == "first"
    assert json.loads(p2.read_text())["error"] == "second"


# ---- fix round 1, issue 4: end-to-end build_profiles with a fake client + resume -------


class _ResumeFakeClient:
    """Records each generation call's app (detected from the bundle's own `## <path>`
    headers -- only the gitlab bundle's paths ever contain "gitlab") and how many times
    each app has been called; an app in `fail_first_for` raises (too few valid candidates)
    on its first call and succeeds on every call after, simulating an interrupted build
    that gets resumed.
    """

    def __init__(self, good_json: str, bad_json: str, fail_first_for: set[str]):
        self.messages = self
        self._good = good_json
        self._bad = bad_json
        self._fail_first_for = set(fail_first_for)
        self.calls: dict[str, int] = {"gitlab": 0, "gmail": 0}

    @staticmethod
    def _app_of(prompt: str) -> str:
        return "gitlab" if "gitlab" in prompt.lower() else "gmail"

    def stream(self, **kw):
        assert "temperature" not in kw
        prompt = kw["messages"][0]["content"]
        app = self._app_of(prompt)
        self.calls[app] += 1
        text = self._bad if (app in self._fail_first_for and self.calls[app] == 1) else self._good
        response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")
        return _FakeStream(response)


def _safe_knowledge_batch(n: int = 60) -> str:
    """Generic, entity-free filler that satisfies word-count bounds and is vanishingly
    unlikely to 8-gram-collide with any real gitlab/gmail task instruction.
    """
    return json.dumps([
        f"Guidance note {i}: open the relevant panel, locate the applicable control, apply "
        "the requested change, then reopen the page to confirm the update is reflected. "
        + "detail " * 30
        for i in range(n)
    ])


def test_build_profiles_end_to_end_with_fake_client_and_resume(tmp_path):
    good = _safe_knowledge_batch(60)
    bad = json.dumps(["short"] * 5)
    client = _ResumeFakeClient(good, bad, fail_first_for={"gmail"})
    out = tmp_path / "het"

    # First run: gitlab succeeds, gmail fails (too few valid candidates).
    with pytest.raises(mh.KnowledgeGenerationError):
        mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client)

    assert (out / "pools" / "GLK.yaml").exists()
    assert (out / "k_generation_gitlab.json").exists()
    # Fix round 2, ruling 3: the failed record is archived, not written to the canonical
    # path -- the canonical path only ever holds a genuine success.
    assert (out / "k_generation_gmail.failed-1.json").exists()
    assert not (out / "k_generation_gmail.json").exists()
    assert not (out / "manifest.json").exists()
    assert client.calls == {"gitlab": 1, "gmail": 1}

    # Resume (no --force): gmail succeeds this time; gitlab must NOT be called again.
    mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client)
    assert client.calls == {"gitlab": 1, "gmail": 2}

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == mh.SEED
    assert manifest["block_seed"] == mh.SEED + 1
    assert manifest["bundle_max_words"] == mh.BUNDLE_MAX_WORDS
    assert set(manifest["real_tasks_sha256"]) == {"gitlab", "gmail"}
    assert set(manifest["entity_set_sha256"]) == {"gitlab", "gmail"}
    assert set(manifest["bundles"]) == {"gitlab", "gmail"}
    assert set(manifest["bridge_source"]) == {f"GMB_{i:02d}" for i in range(20)}
    assert all(v.startswith("GLG_") for v in manifest["bridge_source"].values())

    assert len(mp.load_pool(out / "pools" / "GLG.yaml")) == 50
    assert len(mp.load_pool(out / "pools" / "GMB.yaml")) == 20
    assert len(mp.load_pool(out / "pools" / "GLK.yaml")) == 40
    assert len(mp.load_pool(out / "pools" / "GMK.yaml")) == 40
    assert len(mp.load_pool(out / "pools" / "anchors_gitlab.yaml")) == 3
    assert len(mp.load_pool(out / "pools" / "anchors_gmail.yaml")) == 1

    gl_queue = mp.read_queue(out / "gitlab" / "queue.jsonl")
    assert sum(i.replicate == 0 for i in gl_queue) == (50 + 40 + 3) * 60
    assert sum(i.pool == "GLG" and i.replicate == 1 for i in gl_queue) == 300
    assert sum(i.pool == "GLK" and i.replicate == 1 for i in gl_queue) == 240

    gm_queue = mp.read_queue(out / "gmail" / "queue.jsonl")
    assert sum(i.replicate == 0 for i in gm_queue) == (40 + 1) * 30
    assert sum(i.pool == "GMK" and i.replicate == 1 for i in gm_queue) == 120

    br_queue = mp.read_queue(out / "bridge" / "queue.jsonl")
    assert len(br_queue) == 20 * 30

    # I1/I6: the manifest names block A (the fallback universe) and the gitlab stage boundaries
    assert len(manifest["gitlab_block_a"]) == 30 and len(manifest["gitlab_block_b"]) == 30
    assert manifest["gitlab_stage_ends"] == mh.gitlab_stage_ends(gl_queue, manifest["gitlab_block_a"])
    # minors: the agent package and the environment repo are recorded
    assert manifest["browser_use_version"] == importlib.metadata.version("browser-use")
    wa = manifest["webarena_infinity"]
    assert set(wa) == {"root", "commit", "dirty", "note"}
    assert (wa["commit"] is None) == (wa["note"] is not None)

    # A second call without --force, now that the build is complete, is refused.
    with pytest.raises(SystemExit):
        mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client)
    assert client.calls == {"gitlab": 1, "gmail": 2}

    # --force rebuilds everything unconditionally, including a fresh call for gitlab too.
    mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client, force=True)
    assert client.calls == {"gitlab": 2, "gmail": 3}


class _ScriptedFakeClient:
    """Like `_ResumeFakeClient`, but scripted per call number rather than "first call only":
    `script[app]` is a list of bools (True = succeed) indexed by that app's call count so
    far (1-based); once exhausted, later calls succeed. Needed for fix round 2's ruling-3
    test, which drives a specific app through succeed -> (forced) fail -> succeed across
    three separate `build_profiles` calls, not just "fails once, ever".
    """

    def __init__(self, good_json: str, bad_json: str, script: dict[str, list[bool]]):
        self.messages = self
        self._good = good_json
        self._bad = bad_json
        self._script = script
        self.calls: dict[str, int] = {"gitlab": 0, "gmail": 0}

    @staticmethod
    def _app_of(prompt: str) -> str:
        return "gitlab" if "gitlab" in prompt.lower() else "gmail"

    def stream(self, **kw):
        assert "temperature" not in kw
        prompt = kw["messages"][0]["content"]
        app = self._app_of(prompt)
        self.calls[app] += 1
        n = self.calls[app]
        outcomes = self._script.get(app, [])
        ok = outcomes[n - 1] if n - 1 <= len(outcomes) - 1 else True
        text = self._good if ok else self._bad
        response = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")
        return _FakeStream(response)


def test_build_profiles_failed_force_run_invalidates_only_the_forced_app_and_resume_regenerates_it(tmp_path):
    """Fix round 2, ruling 3's core scenario: a completed build, then a `--force` rebuild
    where gitlab succeeds again but gmail fails, then a plain resume. Asserts (a) no
    `manifest.json` remains after the failed forced run, (b) gitlab's freshly-forced cache is
    reused on resume (no 3rd call), and (c) gmail -- whose cache was invalidated by `--force`
    and then failed to regenerate -- is regenerated on resume rather than silently reusing
    either the archived failure or a stale pre-force record.
    """
    good = _safe_knowledge_batch(60)
    bad = json.dumps(["short"] * 5)
    out = tmp_path / "het"
    client = _ScriptedFakeClient(good, bad, script={"gitlab": [True, True], "gmail": [True, False, True]})

    # Round 1: a clean, fully successful build (no force needed -- `out` is empty).
    mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client)
    assert (out / "manifest.json").exists()
    assert client.calls == {"gitlab": 1, "gmail": 1}

    # Round 2: --force. gitlab succeeds again (its 2nd call); gmail fails (its 2nd call).
    with pytest.raises(mh.KnowledgeGenerationError):
        mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client, force=True)
    assert not (out / "manifest.json").exists()  # unlinked before proceeding, never rewritten
    assert client.calls == {"gitlab": 2, "gmail": 2}
    assert (out / "k_generation_gitlab.json").exists()  # gitlab's fresh (round-2) record
    assert (out / "k_generation_gmail.failed-1.json").exists()
    # --force invalidates the canonical record up front, so the failed attempt leaves none
    # behind -- not the archived failure, and not a stale pre-force success either.
    assert not (out / "k_generation_gmail.json").exists()

    # Round 3: resume, no --force. GitLab's just-refreshed cache is reused (no 3rd call);
    # Gmail has no canonical cache to reuse, so it is regenerated -- never silently reusing
    # the archived failure record.
    mh.build_profiles(out=out, log_root=tmp_path / "logs", client=client)
    assert client.calls == {"gitlab": 2, "gmail": 3}
    assert (out / "manifest.json").exists()
    gmail_raw = json.loads((out / "k_generation_gmail.json").read_text())
    assert gmail_raw["error"] is None
    assert len(mp.load_pool(out / "pools" / "GMK.yaml")) == 40


def test_build_profiles_refuses_while_heterogeneity_logs_exist(tmp_path):
    # I5 ruling: rebuilding pools/queues under collected data would silently re-key un-recollectable
    # episodes; refuse -- even with force -- before touching anything or calling the generator.
    logs = tmp_path / "logs"
    (logs / "gitlab").mkdir(parents=True)
    (logs / "gitlab" / "worker_0.jsonl").write_text("{}\n")
    client = _ResumeFakeClient(_safe_knowledge_batch(60), "[]", fail_first_for=set())
    for force in (False, True):
        with pytest.raises(SystemExit, match="worker_0.jsonl"):
            mh.build_profiles(out=tmp_path / "het", log_root=logs, client=client, force=force)
    assert client.calls == {"gitlab": 0, "gmail": 0} and not (tmp_path / "het").exists()


def test_print_stages_reads_the_frozen_queue(tmp_path, capsys):
    arms, block_a, block_b = _gitlab_design()
    q = mh.gitlab_queue(arms, block_a, block_b, seed=3)
    (tmp_path / "gitlab").mkdir()
    mp.write_queue(tmp_path / "gitlab" / "queue.jsonl", q)
    (tmp_path / "manifest.json").write_text(json.dumps({"gitlab_block_a": block_a}))
    mh.main(["--out", str(tmp_path), "--print-stages"])
    got = json.loads(capsys.readouterr().out)
    assert got == mh.gitlab_stage_ends(q, block_a)
