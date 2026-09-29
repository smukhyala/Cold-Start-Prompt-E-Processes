"""Pool builders for the prompt-heterogeneity study: bundles, K generation with a leak guard,
G/anchor reuse, and per-profile queues."""

from __future__ import annotations

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


def test_gitlab_queue_blocks_and_pilot():
    arms = {"GLG": [f"GLG_{i:02d}" for i in range(50)], "GLK": [f"GLK_{i:02d}" for i in range(40)],
            "anchor": ["GL_anchor_baseline", "GL_anchor_explorer", "GL_anchor_oracle"]}
    block_a = [f"task_e{i}" for i in range(10)] + [f"task_m{i}" for i in range(10)] + [f"task_h{i}" for i in range(10)]
    block_b = [f"task_e{i}" for i in range(10, 20)] + [f"task_m{i}" for i in range(10, 20)] + [f"task_h{i}" for i in range(10, 20)]
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


# ---- fix round 1, issue 1: seed-data stripping -----------------------------------------


def test_strip_seed_data_sections_removes_matched_section_only():
    text = (
        "# App\n\n"
        "## Summary\n"
        "This app has issues and boards.\n\n"
        "## Seed Data Summary\n\n"
        "### Users (2)\n"
        "Sarah Chen (Owner), Bob Jones (Dev)\n\n"
        "### Boards (1)\n"
        "Bug Triage\n\n"
        "## Navigation Structure\n"
        "Issues, boards, and labels are reachable from the sidebar.\n"
    )
    stripped = mh._strip_seed_data_sections(text)
    assert "Sarah Chen" not in stripped
    assert "Seed Data Summary" not in stripped
    assert "### Users" not in stripped and "### Boards" not in stripped
    assert "## Summary" in stripped and "## Navigation Structure" in stripped
    assert "sidebar" in stripped


def test_strip_seed_data_sections_noop_without_a_match():
    text = "# App\n\n## Summary\nNothing to strip here.\n"
    assert mh._strip_seed_data_sections(text) == text


def test_gitlab_bundle_strips_seed_data_and_frees_sarah_chen_entity():
    bundle = mh.build_bundle("gitlab")
    assert "Sarah Chen" not in bundle
    ents = mh.task_entities("gitlab")
    assert "sarah chen" in ents
    # NOTE: "Bug Triage" is *not* asserted absent here -- unlike "Sarah Chen", it is also
    # named outside the Seed Data Summary section (APP_DESCRIPTION.md's "Boards" section
    # documents it as one of three fixed pre-configured board names), so it correctly
    # remains exempt from the leak guard even after seed-data stripping. See the fix-round
    # report for the full explanation.


# ---- fix round 1, issue 2: priority order, dedup, hard cap -----------------------------


def test_assemble_bundle_priority_order_dedup_and_hard_cap():
    items = [
        ("a.md", "Alpha " * 5),
        ("b.md", "Beta " * 5),
        ("a.md", "Alpha duplicate path, must be skipped regardless of its content " * 5),
        ("c.md", "Beta " * 5),  # byte-identical body to b.md -> deduped by content
        ("d.md", "Delta " * 500),
    ]
    sections = mh._assemble_bundle(items, max_words=20)
    paths = [s["path"] for s in sections]
    assert paths == ["a.md", "b.md", "d.md"]
    assert "c.md" not in paths
    assert sum(s["words"] for s in sections) <= 20
    assert sum(len(s["text"].split()) for s in sections) <= 20
    assert sections[-1]["truncated"] is True


def test_assemble_bundle_dedups_near_identical_content_ignoring_source_line():
    """The webarena manual pages end in a self-referential "Source: <this page's own doc
    id>" line, so two mirrored copies of the same page are near- but not byte-identical.
    """
    body_a = "Some shared procedural content about a feature.\n\nSource: https://example.com/111\n"
    body_b = "Some shared procedural content about a feature.\n\nSource: https://example.com/222\n"
    assert body_a != body_b  # not byte-identical
    sections = mh._assemble_bundle([("a.md", body_a), ("b.md", body_b)], max_words=1000)
    assert [s["path"] for s in sections] == ["a.md"]


def test_gitlab_bundle_priority_order_and_previously_dropped_files():
    sections = mh.bundle_sections("gitlab")
    paths = [s["path"] for s in sections]

    def idx(suffix):
        return next(i for i, p in enumerate(paths) if p.endswith(suffix))

    # labels.md, issue_board.md, milestones/*, and issues/managing_issues.md were dropped
    # by the old plain-alphabetical-then-cut ordering; under the ruled priority order they
    # now survive the 25,000-word cap.
    for suffix in ("APP_DESCRIPTION.md", "user/project/labels.md", "user/project/issue_board.md",
                   "user/project/milestones/_index.md", "user/project/issues/managing_issues.md"):
        assert any(p.endswith(suffix) for p in paths), suffix

    assert idx("APP_DESCRIPTION.md") < idx("user/project/labels.md")
    assert idx("user/project/labels.md") < idx("user/project/issue_board.md")
    assert idx("user/project/issue_board.md") < idx("user/project/milestones/_index.md")
    assert idx("user/project/milestones/_index.md") < idx("user/project/issues/managing_issues.md")
    assert idx("user/project/issues/managing_issues.md") < idx("user/project/issues/_index.md")


def test_gmail_bundle_priority_order_settings_first():
    sections = mh.bundle_sections("gmail")
    paths = [s["path"] for s in sections]
    assert paths[0].startswith("apps/user-manuals/gmail/settings-and-configuration/")
    first_organize = next(i for i, p in enumerate(paths) if "organize-and-manage/" in p)
    first_compose = next((i for i, p in enumerate(paths) if "compose-and-send/" in p), None)
    last_settings = max(i for i, p in enumerate(paths) if "settings-and-configuration/" in p)
    assert last_settings < first_organize
    if first_compose is not None:
        assert first_organize < first_compose


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
        mh.build_profiles(out=out, client=client)

    assert (out / "pools" / "GLK.yaml").exists()
    assert (out / "k_generation_gitlab.json").exists()
    assert (out / "k_generation_gmail.json").exists()  # the failed record was persisted
    assert not (out / "manifest.json").exists()
    assert client.calls == {"gitlab": 1, "gmail": 1}

    # Resume (no --force): gmail succeeds this time; gitlab must NOT be called again.
    mh.build_profiles(out=out, client=client)
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

    # A second call without --force, now that the build is complete, is refused.
    with pytest.raises(SystemExit):
        mh.build_profiles(out=out, client=client)
    assert client.calls == {"gitlab": 1, "gmail": 2}

    # --force rebuilds everything unconditionally, including a fresh call for gitlab too.
    mh.build_profiles(out=out, client=client, force=True)
    assert client.calls == {"gitlab": 2, "gmail": 3}
