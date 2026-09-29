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


def test_leak_guard_rejects_task_ngrams_and_entities():
    tasks = ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label."]
    ents = {"sarah chen", "q1 product roadmap"}
    clean = "Use the star icon beside a message, then confirm the label list shows your change. " * 3
    leaky_ngram = "Star Sarah Chen's Q1 product roadmap email and move it to the inbox."
    leaky_entity = "When a message from Sarah Chen arrives, archive it after reading."
    assert mh.leak_violations(clean, tasks, ents) == []
    assert any("8-gram" in v for v in mh.leak_violations(leaky_ngram, tasks, ents))
    assert any("entity" in v for v in mh.leak_violations(leaky_entity, tasks, ents))


def test_generate_knowledge_filters_and_records(monkeypatch):
    good = [f"Guidance {i}: open the sidebar, pick the feature, confirm the banner. " + "word " * 40 for i in range(45)]
    bad = ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label. " + "word " * 40]

    class Msg:
        def create(self, **kw):
            assert "temperature" not in kw
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(bad + good))],
                                   stop_reason="end_turn")

    monkeypatch.setattr(mh, "task_texts", lambda app: ["Star Sarah Chen's Q1 product roadmap email and move it to the Projects label."])
    monkeypatch.setattr(mh, "task_entities", lambda app: {"sarah chen"})
    texts, raw = mh.generate_knowledge(SimpleNamespace(messages=Msg()), "gmail", "bundle text", n=40)
    assert len(texts) == 40 and all("Sarah" not in t for t in texts)
    assert raw["rejected"] and raw["rejected"][0]["reason"]


def test_generate_knowledge_raises_when_too_few(monkeypatch):
    class Msg:
        def create(self, **kw):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(["too short"] * 60))],
                                   stop_reason="end_turn")
    monkeypatch.setattr(mh, "task_texts", lambda app: [])
    monkeypatch.setattr(mh, "task_entities", lambda app: set())
    with pytest.raises(ValueError):
        mh.generate_knowledge(SimpleNamespace(messages=Msg()), "gmail", "b", n=40)


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
