"""build_gmail_oracle: diff + render on tiny synthetic Gmail states, header, ordering, template machinery.

No model, no server. Only ``test_real_seed_*`` needs node (skipped when missing).
"""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))

import build_gmail_oracle as bgo  # noqa: E402
import make_pools  # noqa: E402


def fake_state():
    return {
        "emails": [
            {"id": 1, "threadId": "thread_1", "from": {"name": "Ann Lee", "email": "ann@x.io"},
             "to": [{"name": "Me", "email": "me@g.com"}], "cc": [], "bcc": [], "subject": "Roadmap review",
             "snippet": "snip", "body": "hello", "date": "2026-02-24T09:15:00Z", "isRead": False,
             "isStarred": False, "starType": None, "isImportant": True, "labels": ["INBOX", "label_1"],
             "isTrashed": False, "isSpam": False, "isArchived": False, "isDraft": False, "isSent": False},
            {"id": 2, "threadId": "thread_2", "from": {"name": "Bo Chen", "email": "bo@y.io"},
             "to": [{"name": "Me", "email": "me@g.com"}], "cc": [], "bcc": [], "subject": "Lunch?",
             "snippet": "s", "body": "b", "date": "2026-02-25T09:15:00Z", "isRead": False, "isStarred": False,
             "starType": None, "isImportant": False, "labels": ["INBOX"], "isTrashed": False, "isSpam": False,
             "isArchived": False, "isDraft": False, "isSent": False},
            {"id": 3, "threadId": "thread_3", "from": {"name": "Cy Dee", "email": "cy@z.io"},
             "to": [], "cc": [], "bcc": [], "subject": "Junk", "snippet": "s", "body": "b",
             "date": "2026-02-26T09:15:00Z", "isRead": True, "isStarred": False, "starType": None,
             "isImportant": False, "labels": ["SPAM"], "isTrashed": False, "isSpam": True,
             "isArchived": False, "isDraft": False, "isSent": False},
        ],
        "labels": [
            {"id": "INBOX", "name": "Inbox", "type": "system", "color": None, "visible": True, "parentId": None,
             "messageCount": 2, "unreadCount": 2},
            {"id": "label_1", "name": "Work", "type": "user", "color": {"background": "#111", "text": "#fff"},
             "visible": True, "parentId": None, "messageCount": 1, "unreadCount": 1},
            {"id": "label_2", "name": "Reference", "type": "user", "color": {"background": "#222", "text": "#fff"},
             "visible": True, "parentId": None, "messageCount": 0, "unreadCount": 0},
        ],
        "filters": [
            {"id": "filter_1", "criteria": {"from": "a@b.com", "to": "", "subject": "", "hasWords": "",
                                             "doesntHave": "", "hasAttachment": False, "size": None,
                                             "sizeUnit": "MB", "sizeComparison": "greater"},
             "actions": {"label": "label_1", "archive": False, "markRead": False, "star": False,
                         "forward": None, "delete": False},
             "enabled": True, "createdAt": "2025-06-10T09:00:00Z"},
        ],
        "settings": {"theme": "default", "undoSendDelay": 5, "categoriesEnabled": {"social": True, "forums": True},
                     "multipleInboxSections": [{"query": "", "name": "Section 1", "maxMessages": 25},
                                               {"query": "", "name": "Section 2", "maxMessages": 25}]},
        "contacts": [{"id": "contact_01", "name": "Ann Lee", "email": "ann@x.io", "isFrequent": True}],
        "blockedSenders": [{"email": "old@spam.com", "blockedAt": "2025-11-15T10:00:00Z"}],
        "currentUser": {"name": "Me"},
        "_nextEmailId": 200, "_nextLabelId": 30, "_nextFilterId": 20,
    }


def solve_star_trash(state):
    state["emails"][0]["isStarred"] = True
    state["emails"][0]["starType"] = "yellow-star"
    state["emails"][0]["date"] = "CHANGED"
    state["emails"][0]["snippet"] = "CHANGED"
    state["emails"][1]["isTrashed"] = True
    state["emails"][1]["labels"].remove("INBOX")
    state["_nextEmailId"] = 999
    state["labels"][0]["messageCount"] = 77


def solve_send(state):
    state["emails"].append({
        "id": 200, "threadId": "thread_new", "from": {"name": "Me", "email": "me@g.com"},
        "to": [{"name": "Ann Lee", "email": "ann@x.io"}], "cc": [{"name": "", "email": "cc@x.io"}], "bcc": [],
        "subject": "Quarterly update", "snippet": "x", "date": "2026-03-01T00:00:00Z",
        "body": "Hi Ann, the numbers are in.\nThanks, Me", "isSent": True, "isDraft": False, "isRead": True,
        "labels": ["SENT"], "isStarred": False, "isTrashed": False, "isSpam": False, "isArchived": False})


def solve_labels_filters(state):
    state["labels"].append({"id": "label_30", "name": "Q1 Priority", "type": "user",
                            "color": {"background": "#cc3333", "text": "#fff"}, "visible": True,
                            "parentId": "label_1", "messageCount": 0, "unreadCount": 0})
    state["emails"][0]["labels"].append("label_30")
    state["labels"][2]["name"] = "Resources"
    state["filters"][0]["actions"]["archive"] = True
    state["filters"].append({"id": "filter_20", "criteria": {"from": "", "to": "", "subject": "invoice",
                                                             "hasWords": "", "doesntHave": "",
                                                             "hasAttachment": False, "size": None,
                                                             "sizeUnit": "MB", "sizeComparison": "greater"},
                             "actions": {"label": "label_2", "archive": False, "markRead": True, "star": False,
                                         "forward": None, "delete": False},
                             "enabled": True, "createdAt": "NOW"})
    state["blockedSenders"].append({"email": "new@spam.com", "blockedAt": "NOW"})
    state["blockedSenders"] = [b for b in state["blockedSenders"] if b["email"] != "old@spam.com"]
    state["contacts"].append({"id": "contact_99", "name": "Dee Fox", "email": "dee@fox.io", "isFrequent": False})


def solve_settings(state):
    state["settings"]["theme"] = "dark"
    state["settings"]["categoriesEnabled"]["social"] = False
    state["settings"]["multipleInboxSections"][1]["query"] = "is:important"


def solve_empty_spam(state):
    state["emails"] = [e for e in state["emails"] if not e["isSpam"]]


def solve_only_volatile(state):
    state["emails"][0]["date"] = "X"
    state["labels"][0]["messageCount"] = 5
    state["_nextLabelId"] = 31


def solve_delete_label(state):
    state["labels"] = [l for l in state["labels"] if l["id"] != "label_1"]
    state["emails"][0]["labels"].remove("label_1")
    state["filters"] = []


def render_all(solver):
    before = fake_state()
    after = copy.deepcopy(before)
    solver(after)
    return bgo.render_changes(bgo.diff_states(before, after))


# ---- diff and render --------------------------------------------------------------------------


def test_email_state_changes_identify_email_and_use_friendly_fields():
    text = render_all(solve_star_trash)
    assert 'email 1 "Roadmap review" from Ann Lee <ann@x.io>' in text
    assert 'email 2 "Lunch?" from Bo Chen <bo@y.io>' in text
    assert "starred false → true" in text
    assert 'star type none → "yellow-star"' in text
    assert "in trash false → true" in text
    assert 'labels: remove INBOX "Inbox"' in text


def test_no_volatile_fields_rendered():
    text = render_all(solve_star_trash) + render_all(solve_labels_filters) + render_all(solve_send)
    for volatile in ("CHANGED", "snippet", "2026-02-24", "createdAt", "blockedAt", "messageCount", "NOW",
                     "_next", "999", "threadId", "unreadCount"):
        assert volatile not in text


def test_only_volatile_changes_give_an_empty_diff():
    before = fake_state()
    after = copy.deepcopy(before)
    solve_only_volatile(after)
    assert bgo.diff_states(before, after) == []


def test_created_email_body_is_verbatim_and_complete():
    text = render_all(solve_send)
    assert "CREATE email (compose and SEND it)" in text
    assert "to=[Ann Lee <ann@x.io>]" in text
    assert "cc=[cc@x.io]" in text
    assert 'subject="Quarterly update"' in text
    assert 'body="Hi Ann, the numbers are in.\\nThanks, Me"' in text
    assert "200" not in text.replace("2026", "")  # the auto id is not rendered


def test_created_draft_is_labelled_as_draft():
    def solve(state):
        solve_send(state)
        state["emails"][-1]["isSent"] = False
        state["emails"][-1]["isDraft"] = True
    assert "draft email (save it as a draft, do not send)" in render_all(solve)


def test_labels_filters_blocked_contacts():
    text = render_all(solve_labels_filters)
    assert 'CREATE label "Q1 Priority", nested under label_1 "Work", color background #cc3333, text #fff' in text
    assert 'labels: add "Q1 Priority" (the new label)' in text
    assert 'label_2 "Reference": name "Reference" → "Resources"' in text
    assert 'filter_1 (from="a@b.com"): action archive false → true' in text
    assert 'CREATE filter with criteria [subject="invoice"], actions [apply label label_2 "Resources", markRead]' in text
    assert "BLOCK sender new@spam.com" in text
    assert "UNBLOCK sender old@spam.com" in text
    assert 'CREATE contact with name="Dee Fox", email=dee@fox.io' in text


def test_settings_nested_and_tabs():
    text = render_all(solve_settings)
    assert 'setting theme (Settings > general tab): "default" → "dark"' in text
    assert "setting categoriesEnabled social (Settings > inbox tab): true → false" in text
    assert 'setting multipleInboxSections item 2 query (Settings > inbox tab): empty → "is:important"' in text


def test_deleted_emails_and_labels():
    assert "PERMANENTLY DELETE email 3" in render_all(solve_empty_spam)
    text = render_all(solve_delete_label)
    assert 'DELETE label label_1 "Work"' in text
    assert "DELETE filter filter_1" in text
    assert "labels: remove" not in text  # stripping a deleted label from a mail is a side effect


def test_identical_changes_are_grouped():
    def solve(state):
        for email in state["emails"][:2]:
            email["isRead"] = True
    text = render_all(solve)
    assert text.count("read false → true") == 1
    assert "email 1" in text and "email 2" in text


# ---- solve_all, build, header ------------------------------------------------------------------


def fake_bank():
    return [{"id": "task_m2", "instruction": "Star the roadmap."}, {"id": "task_e1", "instruction": "Send it."},
            {"id": "task_e10", "instruction": "Dark mode."}]


def fake_solvers():
    return {"task_m2": solve_star_trash, "task_e1": solve_send, "task_e10": solve_settings}


def test_every_task_gets_an_entry_in_sorted_order_keyed_by_instruction():
    built = bgo.build_all(seed_state=fake_state(), solvers=fake_solvers(), bank=fake_bank(), expected_tasks=3)
    assert built["task_ids"] == ["task_e1", "task_e10", "task_m2"]
    assert len(built["entries"]) == 3
    assert built["entries"][0].startswith('- "Send it." → ')
    assert built["entries"][1].startswith('- "Dark mode." → ')
    assert built["entries"][2].startswith('- "Star the roadmap." → ')
    assert built["empty"] == [] and built["failed"] == []


def test_empty_diff_and_failed_solver_fail_the_build_unless_allowed():
    solvers = fake_solvers()
    solvers["task_e10"] = solve_only_volatile
    with pytest.raises(bgo.BuildError, match="EMPTY diff"):
        bgo.build_all(seed_state=fake_state(), solvers=solvers, bank=fake_bank(), expected_tasks=3)

    def boom(state):
        raise KeyError("nope")
    solvers["task_e10"] = boom
    with pytest.raises(bgo.BuildError, match="raised"):
        bgo.build_all(seed_state=fake_state(), solvers=solvers, bank=fake_bank(), expected_tasks=3)
    built = bgo.build_all(seed_state=fake_state(), solvers=solvers, bank=fake_bank(), expected_tasks=3,
                          allow={"task_e10": "test"})
    assert built["failed"] == ["task_e10"]


def test_wrong_task_count_fails():
    with pytest.raises(bgo.BuildError, match="expected 60"):
        bgo.build_all(seed_state=fake_state(), solvers=fake_solvers(), bank=fake_bank())


def test_header_has_the_five_steps_and_routes():
    header = bgo.HEADER
    for n in range(1, 6):
        assert f"\n{n}. " in header
    assert "identical to your task" in header
    assert "<start host>/#/email/15" in header
    for route in ("#/settings/filters", "#/settings/labels", "#/settings/general", "#/settings/inbox",
                  "#/label/label_17", "#/trash"):
        assert route in header
    assert "reopen each listed item" in header
    assert "Finish only when every listed value matches" in header


def test_guidance_layout_and_vector():
    built = bgo.build_all(seed_state=fake_state(), solvers=fake_solvers(), bank=fake_bank(), expected_tasks=3)
    guidance = built["guidance"]
    assert guidance.startswith(bgo.HEADER)
    assert bgo.KEY_TITLE in guidance
    assert guidance.index(bgo.HEADER) < guidance.index(bgo.KEY_TITLE) < guidance.index(built["entries"][0])
    assert built["arm"].vector.as_dict() == {"planning": 0, "verification": 2, "agency": 0, "expertise": 3,
                                             "format": 0, "goal": 0}
    assert built["arm_id"] == "GM_oracle_full" and built["pool"] == "gmoraclefull"


# ---- same template machinery as the Gmail default arm -------------------------------------------------


def test_arm_renders_through_the_gmail_default_template(tmp_path):
    built = bgo.build_all(seed_state=fake_state(), solvers=fake_solvers(), bank=fake_bank(), expected_tasks=3)
    path = bgo.pool_path(tmp_path)
    bgo.write_oracle_pool(path, built)
    doc = yaml.safe_load(path.read_text())
    anchors = yaml.safe_load((ROOT / "data" / "heterogeneity" / "pools" / "anchors_gmail.yaml").read_text())
    assert anchors["template"] == "configs/template.jinja"
    assert doc["template"] == "configs/template_gitlab.jinja"  # template.jinja never prints prompt_guidance
    assert doc["pool"] == "gmoraclefull"
    loaded = make_pools.load_pool(path)  # re-renders and checks the frozen sha256
    assert len(loaded) == 1 and loaded[0].arm.arm_id == "GM_oracle_full"
    assert "Domain-specific operating guidance:\n" + built["guidance"] in loaded[0].text
    assert bgo.HEADER in loaded[0].text and built["entries"][0] in loaded[0].text

    # With the default arm's vector and empty guidance BOTH templates give exactly the default prompt (so only
    # guidance + vector differ), and template.jinja really would drop the key.
    axes = make_pools.load_axes(bgo.AXES_PATH)
    default = anchors["arms"][0]
    twin = bgo.make_arm("twin", "", default["vector"])
    assert make_pools.sha256_text(make_pools.render(twin, bgo.TEMPLATE, axes)) == default["prompt_sha256"]
    plain = ROOT / "configs" / "template.jinja"
    assert make_pools.sha256_text(make_pools.render(twin, plain, axes)) == default["prompt_sha256"]
    keyed = bgo.make_arm("keyed", built["guidance"], default["vector"])
    assert built["entries"][0] not in make_pools.render(keyed, plain, axes)
    assert built["entries"][0] in make_pools.render(keyed, bgo.TEMPLATE, axes)


# ---- the real benchmark (needs node) ------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to load the Gmail seed state")
def test_real_seed_all_60_tasks_have_nonempty_diffs():
    try:
        built = bgo.build_all()
    except FileNotFoundError:
        pytest.skip("webarena-infinity checkout not available")
    assert len(built["entries"]) == 60
    assert built["empty"] == [] and built["failed"] == []
    assert built["task_ids"] == sorted(built["task_ids"])
    bank = bgo.load_bank()
    for task in bank:
        assert f'- "{task["instruction"]}" → ' in built["guidance"]
