"""Pool construction: deterministic, frozen by hash, and exact about the text it froze."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import make_pools as mp  # noqa: E402

from cold_start.prompts.axes import load_axes  # noqa: E402

AXES = ROOT / "configs" / "axes.yaml"


def test_grid_has_2304_points_and_the_sample_is_deterministic():
    axes = load_axes(AXES)
    assert len(mp.all_grid_vectors(axes)) == 2304
    a = mp.sample_grid_vectors(axes, 50, seed=20260926)
    b = mp.sample_grid_vectors(axes, 50, seed=20260926)
    assert a == b and len(set(a)) == 50
    assert a != mp.sample_grid_vectors(axes, 50, seed=1)


def _texts(k, words=60, start=0):
    return [f"Prompt {i}: " + " ".join(["careful"] * words) for i in range(start, start + k)]


def test_parse_freeform_takes_the_first_n_valid():
    # start=3 keeps this batch's texts distinct from the first _texts(3) call: this test
    # is about skipping the invalid "too short" entry while collecting n valid ones, not
    # about de-duplication (that is test_parse_freeform_drops_duplicates, below), so the
    # two batches must not accidentally collide on the same "Prompt 0"/"Prompt 1" text.
    raw = "Here you go:\n" + json.dumps(_texts(3) + ["too short"] + _texts(2, start=3)) + "\nthanks"
    assert len(mp.parse_freeform(raw, 4)) == 4
    with pytest.raises(ValueError, match="valid"):
        mp.parse_freeform(json.dumps(_texts(2)), 4)


def test_parse_freeform_drops_duplicates():
    texts = _texts(2)
    assert len(mp.parse_freeform(json.dumps(texts + texts + _texts(1)), 2)) == 2
    with pytest.raises(ValueError):
        mp.parse_freeform(json.dumps(texts + texts), 3)


def test_generate_freeform_calls_the_client_once():
    calls = []

    class Messages:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(_texts(60)))])

    texts, raw = mp.generate_freeform(SimpleNamespace(messages=Messages()), n=50)
    assert len(texts) == 50 and len(calls) == 1
    assert calls[0]["model"] == mp.GENERATOR_MODEL and "temperature" not in calls[0]
    assert json.loads(raw["response_text"]) == _texts(60)


def test_freeform_text_round_trips_exactly(tmp_path):
    text = 'Use {{ braces }} and {% raw %}, "quotes", naïve unicode ✓,\n\ttabs and\nnewlines.  '
    path = tmp_path / "pool_F.yaml"
    mp.write_pool(path, "F", [mp.freeform_arm("F_00", text)], ROOT / "configs" / "template_freeform.jinja",
                  AXES, meta={"seed": 1})
    arms = mp.load_pool(path, AXES)
    assert arms[0].text == text
    assert arms[0].arm.prompt_guidance == text


def test_load_pool_refuses_a_changed_text(tmp_path):
    path = tmp_path / "pool_F.yaml"
    mp.write_pool(path, "F", [mp.freeform_arm("F_00", "Be precise. " * 10)],
                  ROOT / "configs" / "template_freeform.jinja", AXES, meta={})
    path.write_text(path.read_text().replace("Be precise.", "Be sloppy."))
    with pytest.raises(ValueError, match="sha256"):
        mp.load_pool(path, AXES)


def _queue():
    arms = {"G": [f"G_{i:02d}" for i in range(50)], "F": [f"F_{i:02d}" for i in range(50)],
            "anchor": ["anchor_baseline"]}
    tasks = [f"task_{i}" for i in range(60)]
    pilot = set(arms["G"][:5] + arms["F"][:5] + arms["anchor"])
    return mp.build_queue(arms, tasks, n_replicates=300, pilot_arms=pilot, seed=20260926), pilot


def test_queue_counts_order_and_pilot():
    q, pilot = _queue()
    assert len(q) == 6000 + 60 + 300
    assert [item.index for item in q] == list(range(len(q)))
    n_pilot = sum(item.pilot for item in q)
    assert n_pilot == 660
    assert all(item.pilot for item in q[:660]) and not any(item.pilot for item in q[660:])
    assert all(item.arm_id in pilot and item.replicate == 0 for item in q[:660])
    keys = {(i.arm_id, i.task_id, i.replicate) for i in q}
    assert len(keys) == len(q)


def test_replicates_repeat_existing_g_or_f_cells():
    q, _ = _queue()
    main = {(i.arm_id, i.task_id) for i in q if i.replicate == 0}
    reps = [i for i in q if i.replicate == 1]
    assert len(reps) == 300
    assert all((i.arm_id, i.task_id) in main and i.pool in ("G", "F") for i in reps)


def test_queue_is_deterministic_and_round_trips(tmp_path):
    a, _ = _queue()
    b, _ = _queue()
    assert a == b
    path = tmp_path / "queue.jsonl"
    mp.write_queue(path, a)
    assert mp.read_queue(path) == a
