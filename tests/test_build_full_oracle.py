"""build_full_oracle: diff + render on a tiny fake state with fake solve functions; the build's failure modes.

No model, no server, no node: the seed state, the solvers and the bank are all fabricated in-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import build_full_oracle as bfo  # noqa: E402


def fake_state():
    return {
        "users": [{"id": "usr_1", "name": "Ann Lee"}, {"id": "usr_2", "name": "Bo Chen"}],
        "labels": [{"id": "lbl_1", "title": "bug"}, {"id": "lbl_2", "title": "ux"}],
        "milestones": [{"id": "ms_1", "title": "v1", "status": "active"}, {"id": "ms_2", "title": "v2", "status": "active"}],
        "issues": [
            {"id": "issue_1", "title": "Crash on start", "status": "open", "assignees": ["usr_1"], "labels": ["lbl_1"],
             "milestoneId": "ms_1", "weight": 1, "timeEstimate": 7200, "updatedAt": "a", "closedAt": None},
            {"id": "issue_2", "title": "Slow page", "status": "open", "assignees": [], "labels": [],
             "milestoneId": "ms_2", "weight": None, "timeEstimate": 0, "updatedAt": "a", "closedAt": None},
        ],
        "epics": [], "iterations": [], "iterationCadences": [], "boards": [],
        "todos": [{"id": "todo_1", "targetType": "issue", "targetId": "issue_2", "action": "assigned",
                   "status": "pending"}],
        "timelogs": [{"id": "tl_1", "issueId": "issue_2", "userId": "usr_1", "timeSpent": 3600}],
        "_nextIssueIid": 3,
    }


def solve_close(state):
    issue = state["issues"][0]
    issue["status"] = "closed"
    issue["closedAt"] = "NOW"
    issue["updatedAt"] = "NOW"
    state["_nextIssueIid"] = 99


def solve_labels(state):
    issue = state["issues"][0]
    issue["labels"] = ["lbl_2"]
    issue["assignees"].append("usr_2")
    issue["timeEstimate"] = 14400
    issue["milestoneId"] = "ms_2"


def solve_delete_issue(state):
    state["issues"] = [i for i in state["issues"] if i["id"] != "issue_2"]
    state["todos"] = []
    state["timelogs"] = []


def solve_delete_milestone(state):
    state["milestones"] = [m for m in state["milestones"] if m["id"] != "ms_2"]
    state["issues"][1]["milestoneId"] = None


def solve_create(state):
    state["issues"].append({"id": "issue_3", "title": "New thing", "status": "open", "assignees": ["usr_2"],
                            "labels": ["lbl_1"], "weight": 0, "createdAt": "x", "confidential": True})


def solve_nothing(state):
    state["issues"][0]["updatedAt"] = "NOW"
    state["_nextIssueIid"] = 5


def solve_raises(state):
    raise KeyError("boom")


def test_status_change_renders_old_to_new_and_ignores_bookkeeping():
    before = fake_state()
    after = fake_state()
    solve_close(after)
    changes = bfo.diff_states(before, after)
    assert len(changes) == 1
    line = bfo.render_entry("Close the crash bug.", changes)
    assert line == '- "Close the crash bug." → issue_1 "Crash on start": status open → closed.'


def test_label_assignee_milestone_and_seconds_are_mapped_to_names():
    before = fake_state()
    after = fake_state()
    solve_labels(after)
    text = bfo.render_changes(bfo.diff_states(before, after))
    assert "labels: add lbl_2 ux; remove lbl_1 bug" in text
    assert "assignees: add usr_2 Bo Chen" in text
    assert "time estimate 2h → 4h" in text
    assert "milestone ms_1 v1 → ms_2 v2" in text


def test_deleted_entity_lists_id_and_title_and_hides_cascades():
    before = fake_state()
    after = fake_state()
    solve_delete_issue(after)
    changes = bfo.diff_states(before, after)
    assert len(changes) == 1
    assert bfo.render_changes(changes) == 'DELETE issue_2 "Slow page".'


def test_clearing_a_reference_to_a_deleted_entity_is_a_side_effect():
    before = fake_state()
    after = fake_state()
    solve_delete_milestone(after)
    assert bfo.render_changes(bfo.diff_states(before, after)) == 'DELETE ms_2 "v2".'


def test_created_entity_lists_the_fields_that_matter():
    before = fake_state()
    after = fake_state()
    solve_create(after)
    text = bfo.render_changes(bfo.diff_states(before, after))
    assert text.startswith('CREATE issue "New thing" with ')
    assert "assignees=[usr_2 Bo Chen]" in text
    assert "labels=[lbl_1 bug]" in text
    assert "confidential=true" in text
    assert "weight" not in text and "createdAt" not in text and "status" not in text


def test_identical_changes_on_several_entities_are_grouped():
    before = fake_state()
    after = fake_state()
    for issue in after["issues"]:
        issue["status"] = "closed"
    text = bfo.render_changes(bfo.diff_states(before, after))
    assert text == 'issue_1 "Crash on start", issue_2 "Slow page": status open → closed.'


def test_empty_diff_and_raising_solver_are_reported_and_fail_the_build():
    solvers = {"t_ok": solve_close, "t_empty": solve_nothing, "t_boom": solve_raises}
    changes, problems = bfo.solve_all(["t_ok", "t_empty", "t_boom", "t_missing"], fake_state(), solvers, {})
    assert len(changes["t_ok"]) == 1
    joined = " | ".join(problems)
    assert "t_empty: EMPTY diff" in joined
    assert "t_boom: solve function raised" in joined
    assert "t_missing: no solve function" in joined
    assert len(problems) == 3


def test_allow_list_waives_an_empty_diff_only_for_that_task():
    solvers = {"t_empty": solve_nothing, "t_empty2": solve_nothing}
    _, problems = bfo.solve_all(["t_empty", "t_empty2"], fake_state(), solvers, {"t_empty": "no-op by design"})
    assert len(problems) == 1 and problems[0].startswith("t_empty2")


def _fake_bank(n):
    bank = []
    for k in range(n):
        bank.append({"id": f"t{k}", "instruction": f"Do thing {k}."})
    return bank


def test_build_all_fails_on_wrong_block_sizes_and_on_empty_diffs():
    bank = _fake_bank(5)
    solvers = {}
    for task in bank:
        solvers[task["id"]] = solve_close
    with pytest.raises(bfo.BuildError, match="exactly 30 block-A"):
        bfo.build_all(fake_state(), solvers, bank, ["t0", "t1"], "HEADER", {"planning": 0, "verification": 0,
                      "agency": 2, "expertise": 3, "format": 0, "goal": 0}, {})


def test_split_requires_30_and_110():
    ids = []
    for k in range(140):
        ids.append(f"t{k}")
    block_a, study = bfo.split_task_ids(ids, ids[30:])
    assert len(block_a) == 30 and len(study) == 110
    assert set(block_a).isdisjoint(set(study))
    with pytest.raises(bfo.BuildError):
        bfo.split_task_ids(ids, ids[31:])


def test_prompt_is_header_plus_answer_key_and_header_drops_the_old_cookbook():
    guidance = ("Intro line: match the visible task instruction to the cookbook below and execute the\n"
                "listed route.\n\nDefault component: act.\n\nExact issue/epic/milestone/iteration ids:\nissue_5 x\n\n"
                "Per-task cookbook:\n01 old stuff")
    header = bfo.header_from_guidance(guidance)
    assert "issue_5" not in header and "old stuff" not in header and header.endswith("act.")
    assert "cookbook below" not in header and bfo.NEW_ROUTING_SENTENCE in header
    tasks = [{"id": "a", "instruction": "Close it."}]
    changes = {"a": bfo.diff_states(fake_state(), _closed())}
    text, entries = bfo.build_prompt(header, tasks, changes)
    assert bfo.ANSWER_KEY_LINE in text
    assert len(entries) == 1 and entries[0] in text
    with pytest.raises(bfo.BuildError):
        bfo.header_from_guidance("no marker here")
    with pytest.raises(bfo.BuildError, match="routing sentence"):
        bfo.header_from_guidance("Intro.\n\nExact issue/epic/milestone/iteration ids:\nx")


def _closed():
    state = fake_state()
    solve_close(state)
    return state
