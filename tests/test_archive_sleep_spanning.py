import json

from experiments.growing_bandits.retrospective.archive_sleep_spanning import archive_sleep_spanning, is_sleep_spanning


def _rec(task, wall, status="ok"):
    return {"arm_id": "A", "task_id": task, "replicate": 0, "status": status, "wallclock_s": wall, "timeout_s": 180}


def test_rule():
    assert is_sleep_spanning(_rec("t", 8673.5))
    assert not is_sleep_spanning(_rec("t", 180.0))
    assert not is_sleep_spanning(_rec("t", 239.0))
    assert not is_sleep_spanning(_rec("t", 9994.0, status="infra_error"))


def test_archive_moves_only_sleep_spanning_and_is_idempotent(tmp_path):
    lines = [_rec("a", 50.0), _rec("b", 8673.5), _rec("c", 180.0)]
    (tmp_path / "worker_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    counts = archive_sleep_spanning(tmp_path)
    assert counts["archived"] == 1 and counts["kept"] == 2
    kept = [json.loads(l)["task_id"] for l in (tmp_path / "worker_0.jsonl").read_text().splitlines()]
    assert kept == ["a", "c"]
    archived = (tmp_path / "archive" / "worker_0.sleep_spanning.jsonl").read_text().splitlines()
    assert [json.loads(l)["task_id"] for l in archived] == ["b"]
    assert archive_sleep_spanning(tmp_path)["archived"] == 0
