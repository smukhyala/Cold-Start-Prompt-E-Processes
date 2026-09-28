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


def test_generate_freeform_refuses_a_response_cut_off_at_max_tokens():
    class Messages:
        def create(self, **kw):
            return SimpleNamespace(stop_reason="max_tokens",
                                   content=[SimpleNamespace(type="text", text=json.dumps(_texts(60)))])

    with pytest.raises(ValueError, match="max_tokens"):
        mp.generate_freeform(SimpleNamespace(messages=Messages()), n=50)


def test_generate_freeform_records_the_stop_reason():
    class Messages:
        def create(self, **kw):
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text=json.dumps(_texts(60)))])

    _, raw = mp.generate_freeform(SimpleNamespace(messages=Messages()), n=50)
    assert raw["stop_reason"] == "end_turn"


def test_a_failed_f_generation_writes_no_pool_file(tmp_path, monkeypatch):
    import anthropic

    def boom(client, n):
        raise ValueError("only 12 valid distinct instructions; need 50")

    monkeypatch.setattr(anthropic, "Anthropic", lambda: object())
    monkeypatch.setattr(mp, "generate_freeform", boom)
    with pytest.raises(ValueError, match="valid"):
        mp.main(["--out", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []  # nothing frozen: a retry needs no --force


# ---- amendment 1: the 30-task stratified subset -----------------------------------------------


def _task_bank(n_e=20, n_m=20, n_h=20):
    """A fake bank in bank order: all `e`s, then all `m`s, then all `h`s."""
    ids = []
    for letter, n in (("e", n_e), ("m", n_m), ("h", n_h)):
        ids += [f"task_{letter}{i:02d}" for i in range(n)]
    return ids


def test_select_tasks_stratifies_ten_ten_ten():
    chosen = mp.select_tasks(_task_bank(), 30, seed=1)
    assert len(chosen) == 30
    counts = {"e": 0, "m": 0, "h": 0}
    for t in chosen:
        counts[t[len("task_")]] += 1
    assert counts == {"e": 10, "m": 10, "h": 10}
    assert len(set(chosen)) == 30


def test_select_tasks_is_deterministic_by_seed():
    ids = _task_bank()
    a = mp.select_tasks(ids, 30, seed=42)
    b = mp.select_tasks(ids, 30, seed=42)
    assert a == b
    assert a != mp.select_tasks(ids, 30, seed=43)


def test_select_tasks_returns_ids_in_bank_order():
    ids = _task_bank()
    chosen = mp.select_tasks(ids, 30, seed=1)
    assert chosen == [t for t in ids if t in set(chosen)]

    shuffled = ids[::-1]  # a bank order that is not sorted by letter or suffix
    chosen2 = mp.select_tasks(shuffled, 30, seed=1)
    assert chosen2 == [t for t in shuffled if t in set(chosen2)]
    assert set(chosen2) == set(mp.select_tasks(ids, 30, seed=1))  # same picks, different order


def test_select_tasks_refuses_n_not_divisible_by_the_strata():
    with pytest.raises(ValueError, match="divisible"):
        mp.select_tasks(_task_bank(), 31, seed=1)


def test_select_tasks_refuses_a_stratum_too_small():
    with pytest.raises(ValueError, match="stratum 'e'"):
        mp.select_tasks(_task_bank(n_e=5), 30, seed=1)


def test_select_tasks_refuses_an_unrecognized_task_id():
    with pytest.raises(ValueError, match="task_"):
        mp.select_tasks(["task_x1", *_task_bank()[1:]], 30, seed=1)


def test_select_tasks_uses_one_rng_in_e_m_h_order_sorted_by_suffix():
    """Reference re-derivation of the exact algorithm the brief specifies, against a fresh
    ``np.random.default_rng``: one rng, strata in order e/m/h, each sorted by integer suffix."""
    import numpy as np

    ids = _task_bank(n_e=12, n_m=9, n_h=15)
    n, seed = 12, 7
    per_stratum = n // 3
    rng = np.random.default_rng(seed)
    expected: set[str] = set()
    for letter in ("e", "m", "h"):
        ordered = sorted((t for t in ids if t.startswith(f"task_{letter}")),
                          key=lambda t: int(t[len(f"task_{letter}"):]))
        idx = rng.choice(len(ordered), size=per_stratum, replace=False)
        expected.update(ordered[int(i)] for i in idx)
    chosen = mp.select_tasks(ids, n, seed=seed)
    assert set(chosen) == expected
    assert chosen == [t for t in ids if t in expected]


def _fake_pool_set(tmp_path, monkeypatch, n_arms=6, n_replicates=8, n_tasks_subset=6):
    """A tiny frozen pool set + manifest, like `make_pools.main()` writes but small enough for
    a fast test: `n_arms` arms each in G and F, one anchor arm, and a 12-task bank (4 per
    stratum) with `N_TASKS`/`N_REPLICATES` monkeypatched down so a full `build_queue` is cheap."""
    monkeypatch.setattr(mp, "N_TASKS", n_tasks_subset)
    monkeypatch.setattr(mp, "N_REPLICATES", n_replicates)

    axes, tmpl_grid, tmpl_free = (AXES, ROOT / "configs" / "template.jinja",
                                  ROOT / "configs" / "template_freeform.jinja")
    out = tmp_path / "data"
    out.mkdir()
    g_arms = [mp.grid_arm(f"G_{i:02d}", mp.PromptVector(**mp.BASELINE_VECTOR)) for i in range(n_arms)]
    f_arms = [mp.freeform_arm(f"F_{i:02d}", f"Be careful, number {i}. " * 10) for i in range(n_arms)]
    anchor = [mp.grid_arm(mp.ANCHOR_ARM_ID, mp.PromptVector(**mp.BASELINE_VECTOR))]
    mp.write_pool(out / "pool_G.yaml", "G", g_arms, tmpl_grid, axes, meta={"seed": 1})
    mp.write_pool(out / "pool_F.yaml", "F", f_arms, tmpl_free, axes, meta={})
    mp.write_pool(out / "pool_anchor.yaml", "anchor", anchor, tmpl_grid, axes, meta={})
    (out / "f_generation_raw.json").write_text(json.dumps({"model": "x", "response_text": "[]"}))

    task_ids = _task_bank(4, 4, 4)  # 12-task bank, in bank order
    arms_by_pool = {"G": [a.arm_id for a in g_arms], "F": [a.arm_id for a in f_arms], "anchor": [mp.ANCHOR_ARM_ID]}
    pilot = set(arms_by_pool["G"][:mp.PILOT_PER_POOL] + arms_by_pool["F"][:mp.PILOT_PER_POOL] + [mp.ANCHOR_ARM_ID])
    seed = 20260926
    full_queue = mp.build_queue(arms_by_pool, task_ids, n_replicates=mp.N_REPLICATES, pilot_arms=pilot, seed=seed)
    mp.write_queue(out / "queue.jsonl", full_queue)

    files = [out / "pool_G.yaml", out / "pool_F.yaml", out / "pool_anchor.yaml",
             out / "f_generation_raw.json", out / "queue.jsonl"]
    manifest = {"seed": seed, "task_ids": task_ids, "files": {p.name: mp.file_sha256(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out, arms_by_pool, task_ids, seed


def _refuse_generator(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("--requeue must never call the generator")

    monkeypatch.setattr(mp, "generate_freeform", boom)


def test_requeue_rewrites_queue_and_manifest_leaving_pools_byte_identical(tmp_path, monkeypatch):
    out, arms_by_pool, task_ids, seed = _fake_pool_set(tmp_path, monkeypatch)
    _refuse_generator(monkeypatch)
    pool_bytes_before = {n: (out / f"pool_{n}.yaml").read_bytes() for n in ("G", "F", "anchor")}
    manifest_before = json.loads((out / "manifest.json").read_text())

    mp.main(["--out", str(out), "--requeue"])

    for name, before in pool_bytes_before.items():
        assert (out / f"pool_{name}.yaml").read_bytes() == before  # frozen pools untouched

    manifest = json.loads((out / "manifest.json").read_text())
    for key, value in manifest_before.items():
        if key == "files":
            for fname, sha in value.items():
                if fname != "queue.jsonl":
                    assert manifest["files"][fname] == sha  # every other file hash kept
        else:
            assert manifest[key] == value  # every existing key kept

    assert manifest["n_tasks"] == 6
    assert manifest["amendment"] == "1: 30-task stratified subset"
    subset = manifest["task_subset"]
    assert subset == mp.select_tasks(task_ids, 6, seed)
    assert manifest["files"]["queue.jsonl"] == mp.file_sha256(out / "queue.jsonl")
    assert manifest["files"]["queue.jsonl"] != manifest_before["files"]["queue.jsonl"]

    queue = mp.read_queue(out / "queue.jsonl")
    assert {q.task_id for q in queue} == set(subset)
    n_arms = 2 * len(arms_by_pool["G"]) + 1  # G + F arms, plus the one anchor arm
    assert sum(q.replicate == 0 for q in queue) == n_arms * 6
    assert sum(q.replicate == 1 for q in queue) == 8  # n_replicates from _fake_pool_set
    n_pilot_arms = 2 * mp.PILOT_PER_POOL + 1
    assert sum(q.pilot for q in queue) == n_pilot_arms * 6


def test_requeue_ignores_force_and_never_touches_pools(tmp_path, monkeypatch):
    out, *_ = _fake_pool_set(tmp_path, monkeypatch)
    _refuse_generator(monkeypatch)
    pool_bytes_before = {n: (out / f"pool_{n}.yaml").read_bytes() for n in ("G", "F", "anchor")}

    mp.main(["--out", str(out), "--requeue", "--force"])  # --force must be ignored, not error

    for name, before in pool_bytes_before.items():
        assert (out / f"pool_{name}.yaml").read_bytes() == before


def test_requeue_refuses_if_a_pool_files_hash_moved(tmp_path, monkeypatch):
    out, *_ = _fake_pool_set(tmp_path, monkeypatch)
    _refuse_generator(monkeypatch)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["files"]["pool_G.yaml"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="pool_G.yaml.*sha256"):
        mp.main(["--out", str(out), "--requeue"])
    assert json.loads((out / "manifest.json").read_text()) == manifest  # untouched by the refusal


def test_requeue_refuses_if_f_generation_raw_hash_moved(tmp_path, monkeypatch):
    out, *_ = _fake_pool_set(tmp_path, monkeypatch)
    _refuse_generator(monkeypatch)
    (out / "f_generation_raw.json").write_text(json.dumps({"model": "x", "response_text": "[1]"}))

    with pytest.raises(ValueError, match="f_generation_raw.json.*sha256"):
        mp.main(["--out", str(out), "--requeue"])
