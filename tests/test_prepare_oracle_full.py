"""prepare_oracle_full: queue sizes, seeded determinism, refuse-to-overwrite, and the collector's own checks.

Needs node (the seed state of the GitLab app) and the sibling webarena-infinity checkout, like the build itself;
no model call, no server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import build_full_oracle as bfo  # noqa: E402
import collect  # noqa: E402
import make_pools as mp  # noqa: E402
import prepare_oracle_full as prep  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return bfo.build_all()


def test_one_pool_covers_all_140_tasks_in_sorted_order_with_no_problems(built):
    assert list(built["pools"]) == [bfo.POOL]
    info = built["pools"][bfo.POOL]
    assert info["arm_id"] == "GL_oracle_full" and len(info["entries"]) == 140
    assert info["task_ids"] == sorted(info["task_ids"]) and len(set(info["task_ids"])) == 140
    assert built["problems"] == []
    assert len(built["block_a"]) == 30 and len(built["study"]) == 110
    assert set(built["block_a"]).isdisjoint(set(built["study"]))
    assert "cookbook below" not in info["guidance"]
    assert bfo.HEADER_V2 in info["guidance"] and bfo.ANSWER_KEY_LINE_V2 in info["guidance"]


def test_pilot_queue_is_30_episodes_of_the_pilot_arm_replicate_zero(built):
    queue = prep.build_queue("pilot", built["block_a"])
    assert len(queue) == 30
    assert {q.arm_id for q in queue} == {bfo.ARM}
    assert {q.pool for q in queue} == {bfo.POOL}
    assert {q.replicate for q in queue} == {0}
    assert len({q.task_id for q in queue}) == 30
    assert [q.index for q in queue] == list(range(30))


def test_study_queue_is_440_episodes_four_per_task(built):
    queue = prep.build_queue("study", built["study"])
    assert len(queue) == 440
    cells = {}
    for item in queue:
        cells.setdefault(item.task_id, set()).add((item.arm_id, item.replicate))
    assert len(cells) == 110
    for task in cells:
        assert cells[task] == {(bfo.ARM, 0), (bfo.ARM, 1), (prep.D_ARM, 0), (prep.D_ARM, 1)}
    # a task's four episodes are adjacent (tasks are the outer, seeded order)
    seen = []
    for item in queue:
        if len(seen) == 0 or seen[-1] != item.task_id:
            seen.append(item.task_id)
    assert len(seen) == 110


def test_queues_are_seeded_and_deterministic(built):
    ids = built["study"]
    first = prep.queue_text(prep.build_queue("study", ids))
    again = prep.queue_text(prep.build_queue("study", list(reversed(ids))))
    assert first == again  # input order does not matter, only the seed
    assert prep.STAGES["pilot"]["seed"] == 20261011 and prep.STAGES["study"]["seed"] == 20261012
    other = prep.queue_text(prep.build_queue("pilot", built["block_a"]))
    assert other != first


def test_prepare_pilot_is_idempotent_and_validates_with_the_collectors_checks(built, tmp_path):
    out = tmp_path / "pilot"
    first = prep.prepare("pilot", out, built)
    before = (out / "manifest.json").read_bytes()
    prep.prepare("pilot", out, built)
    assert (out / "manifest.json").read_bytes() == before
    manifest = json.loads(before)
    assert manifest["n_episodes"] == 30 and manifest["max_steps"] == 30 and manifest["timeout_s"] == 600
    assert manifest["cost_estimate_usd"] == 4.5 and manifest["recommended_budget_usd"] == 8
    assert manifest["pool_files"] == ["pools/oraclefull.yaml"]
    assert "0.80" in manifest["gate"]["statement"]
    assert first["manifest"]["files"] == manifest["files"]
    lines = prep.validate(out)
    assert any("check_queue_arms" in line for line in lines)
    prompts = collect.load_prompts(out, "gitlab")
    assert set(prompts) == {bfo.ARM}


def test_prepare_refuses_to_overwrite_a_different_queue_pool_or_manifest(built, tmp_path):
    out = tmp_path / "pilot"
    prep.prepare("pilot", out, built)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["seed"] = 1
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        prep.prepare("pilot", out, built)

    out2 = tmp_path / "pilot2"
    prep.prepare("pilot", out2, built)
    queue = out2 / prep.QUEUE_REL
    lines = queue.read_text().splitlines()
    lines.reverse()
    queue.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="queue differs"):
        prep.prepare("pilot", out2, built)

    out3 = tmp_path / "pilot3"
    prep.prepare("pilot", out3, built)
    pool = out3 / "pools" / "oraclefull.yaml"
    pool.write_text(pool.read_text() + "# tampered\n")
    with pytest.raises(ValueError, match="pool file differs"):
        prep.prepare("pilot", out3, built)


def test_tampered_queue_fails_the_collectors_sha_check(built, tmp_path):
    out = tmp_path / "pilot"
    prep.prepare("pilot", out, built)
    queue = out / prep.QUEUE_REL
    lines = queue.read_text().splitlines()
    lines.reverse()
    queue.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="refusing to start"):
        collect.verify_queue(out, "gitlab")


def test_study_dir_carries_the_leg_g_default_prompt_byte_for_byte(built, tmp_path):
    out = tmp_path / "study"
    prep.prepare("study", out, built)
    leg_g = ROOT / "data" / "winner_check_gitlab"
    assert (out / prep.ANCHORS_REL).read_bytes() == (leg_g / prep.ANCHORS_REL).read_bytes()
    mine = collect.load_prompts(out, "gitlab")
    old = collect.load_prompts(leg_g, "gitlab")
    assert mine[prep.D_ARM].sha256 == old[prep.D_ARM].sha256
    assert json.loads((out / "manifest.json").read_text())["n_episodes"] == 440
    study_manifest = json.loads((out / "manifest.json").read_text())
    assert study_manifest["cost_estimate_usd"] == 55.0 and study_manifest["recommended_budget_usd"] == 75
    prep.validate(out)


def test_a_different_anchors_pool_is_refused(built, tmp_path):
    fake = tmp_path / "anchors.yaml"
    fake.write_text((ROOT / "data" / "winner_check_gitlab" / prep.ANCHORS_REL).read_text() + "# changed\n")
    with pytest.raises(ValueError, match="not be the same prompt"):
        prep.prepare("study", tmp_path / "study", built, anchors_source=fake)


def test_older_data_dirs_are_unaffected_by_the_manifest_pool_files_override():
    assert collect.pool_files(ROOT / "data" / "winner_check_gitlab", "gitlab") == collect.profile_spec("gitlab")["pools"]
    assert mp.file_sha256(ROOT / "data" / "winner_check_gitlab" / "manifest.json")


def test_pilot_and_study_use_the_byte_identical_oracle_pool(built, tmp_path):
    prep.prepare("pilot", tmp_path / "p", built)
    prep.prepare("study", tmp_path / "s", built)
    pilot_pool = (tmp_path / "p" / "pools" / "oraclefull.yaml").read_bytes()
    assert pilot_pool == (tmp_path / "s" / "pools" / "oraclefull.yaml").read_bytes()
    p = collect.load_prompts(tmp_path / "p", "gitlab")["GL_oracle_full"]
    s = collect.load_prompts(tmp_path / "s", "gitlab")["GL_oracle_full"]
    assert p.sha256 == s.sha256 and p.text == s.text
