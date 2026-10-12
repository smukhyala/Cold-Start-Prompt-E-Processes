"""Pre-reg 16 Amendment 2: credit-refused records are archived verbatim; everything else is kept."""

import json

from experiments.growing_bandits.retrospective.archive_credit_exhausted import archive_credit_exhausted


def test_archives_only_credit_refusals(tmp_path):
    good = {"status": "ok", "arm_id": "A", "task_id": "t1", "replicate": 0, "error_tail": [None]}
    bad = {"status": "ok", "arm_id": "A", "task_id": "t2", "replicate": 0,
           "error_tail": [None, "Error code: 400 - ... Your credit balance is too low to access the Anthropic API."]}
    (tmp_path / "worker_0.jsonl").write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    counts = archive_credit_exhausted(tmp_path)
    assert counts == {"kept": 1, "archived": 1, "files": 1}
    kept = (tmp_path / "worker_0.jsonl").read_text().splitlines()
    assert [json.loads(x)["task_id"] for x in kept] == ["t1"]
    archived = (tmp_path / "archive" / "worker_0.credit_exhausted.jsonl").read_text().splitlines()
    assert json.loads(archived[0]) == bad
    assert archive_credit_exhausted(tmp_path) == {"kept": 1, "archived": 0, "files": 1}
