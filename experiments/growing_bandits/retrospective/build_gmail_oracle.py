"""Build the full-coverage positive-control arm ("answer key" oracle) for the 60 Gmail tasks, mechanically.

    ./.venv/bin/python experiments/growing_bandits/retrospective/build_gmail_oracle.py --dry-run   # report only
    ./.venv/bin/python experiments/growing_bandits/retrospective/build_gmail_oracle.py             # write the pool

Same idea as ``build_full_oracle.py`` (GitLab, FROZEN; we only import helpers from it): the benchmark's own
``webarena-infinity/apps/gmail/sanity_check_real.py`` defines ``solve_task_<id>(state)`` for each of the 60 real
tasks (it mutates the seed state into the expected end-state). We load the seed state once (``load_seed_state``
evaluates ``js/data.js`` with node), deep-copy it, run each solver, and DIFF before/after into concrete,
human-readable changes: emails (id + subject + sender, changed fields, label ids mapped to names), created emails
(to / cc / bcc / subject / full body), labels, filters (criteria + actions), settings, contacts and blocked
senders. One answer-key line per task, keyed by the task's exact instruction, goes into ONE pool file
(pool ``gmoraclefull``, arm ``GM_oracle_full``) rendered with ``configs/template_gitlab.jinja``: the Gmail default
arm's ``template.jinja`` plus a block that prints ``prompt_guidance`` (``template.jinja`` ignores guidance). With
empty guidance the two render identically, so the only differences from the default arm are the guidance and the
vector.

Volatile fields (dates, timestamps, auto ids, label counters, ``_next*`` counters) are never reported. A task whose
diff is EMPTY or whose solver raises fails the build loudly unless it is in ``ALLOW_LIST`` with a reason.
No model is called, no server is started.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "retrospective"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import make_pools  # noqa: E402
from build_full_oracle import VECTOR_V2, file_sha256, make_arm, webarena_root  # noqa: E402

APP = "gmail"
#: ``template.jinja`` (the Gmail default arm's template) NEVER renders ``prompt_guidance``, so an answer key put
#: through it would be silently dropped. ``template_gitlab.jinja`` is the same file plus a guidance block; with
#: empty guidance it renders byte-for-byte what ``template.jinja`` renders (the default arm's prompt sha256 is
#: unchanged -- asserted in the tests), so using it keeps "only guidance + vector differ".
TEMPLATE = ROOT / "configs" / "template_gitlab.jinja"
AXES_PATH = ROOT / "configs" / "axes.yaml"
DEFAULT_OUT_DIR = ROOT / "data" / "headroom" / "pools"

POOL, ARM = "gmoraclefull", "GM_oracle_full"
N_TASKS = 60
VECTOR = dict(VECTOR_V2)

#: task id -> reason. A task listed here may have an empty diff / a raising solver without failing the build.
ALLOW_LIST: dict[str, str] = {}

HEADER = """This is a synthetic positive-control arm. Below is an answer key: one line per task, giving the
task's exact instruction in quotes, then the exact items (emails by numeric id with their subject and sender,
labels by id and name, filters by id) and the exact final values to produce.

Procedure, every time:
1. Before any other action, find the ONE answer-key line whose quoted instruction is identical to your task.
   In your first reasoning step, copy that line's target state.
2. Use exactly the emails, labels and filters it lists, and all of them. Do not search for or guess which items
   are meant: the key already names them.
3. Open each email directly by URL on the same host and port as the start page, e.g. <start host>/#/email/15
   (the number is the email id). Other pages: #/inbox, #/starred, #/snoozed, #/sent, #/drafts, #/important,
   #/allmail, #/spam, #/trash, #/label/label_17 (the mail with that label), and the settings pages
   #/settings/general, #/settings/labels, #/settings/inbox and #/settings/filters (filters and blocked
   addresses). Opening an email marks it as read.
4. Make every listed change and save (send any email you are told to compose).
5. Before you finish, reopen each listed item and check every listed value against the key. If any value differs,
   fix it and save again. Finish only when every listed value matches."""
KEY_NOTES = ("Answer key lines are matched on the quoted instruction, word for word. In a quoted text value, "
             "\\n is a line break; \"none\" and \"empty\" mean the field is blank.")
KEY_TITLE = "Answer key (per-task cookbook):"

#: Fields that are bookkeeping or volatile, never reported.
IGNORED_FIELDS = ("id", "threadId", "date", "snippet", "createdAt", "updatedAt", "blockedAt",
                  "messageCount", "unreadCount")
#: Friendlier names for email flags as the UI shows them.
DISPLAY_NAMES = {
    "isRead": "read", "isStarred": "starred", "starType": "star type", "isImportant": "important",
    "isTrashed": "in trash", "isArchived": "archived", "isSpam": "in spam", "isSnoozed": "snoozed",
    "snoozeUntil": "snooze until", "isMuted": "muted", "isDraft": "draft", "isSent": "sent",
    "category": "category tab", "hasAttachments": "has attachments",
}
#: Which settings tab holds a settings key (for the answer key's "where").
SETTING_TABS = {
    "inboxType": "inbox", "categoriesEnabled": "inbox", "importanceMarkers": "inbox", "previewPane": "inbox",
    "multipleInboxSections": "inbox", "multipleInboxPosition": "inbox",
}
LONG_SUBJECT = 45
FILTER_SKIP_DEFAULTS = ("sizeUnit", "sizeComparison")


class BuildError(RuntimeError):
    """The answer key cannot be built as specified (empty diff, failed solver, wrong task count, ...)."""


# ---- ground truth ------------------------------------------------------------------------------


def app_dir() -> Path:
    return webarena_root() / "apps" / APP


def bank_path() -> Path:
    return app_dir() / "real-tasks.json"


def load_bank() -> list[dict]:
    return json.loads(bank_path().read_text(encoding="utf-8"))


def load_sanity_module():
    """``sanity_check_real.py`` imported from its path (it needs ``requests``; its seed loader needs node)."""
    path = app_dir() / "sanity_check_real.py"
    spec = importlib.util.spec_from_file_location("gmail_sanity_check_real_for_oracle", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_seed(module) -> dict:
    """The seed state; ``load_seed_state`` calls ``sys.exit`` when node fails, which we turn into BuildError."""
    try:
        return module.load_seed_state()
    except SystemExit as err:
        raise BuildError(f"could not load the Gmail seed state via node (exit {err.code})") from err


# ---- rendering helpers -------------------------------------------------------------------------


def _is_list(value) -> bool:
    return hasattr(value, "append")


def _is_dict(value) -> bool:
    return hasattr(value, "keys")


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def quote(text) -> str:
    """A text value in double quotes, JSON-escaped (a newline is written ``\\n``), non-ASCII kept as is."""
    return json.dumps(str(text), ensure_ascii=False)


def label_index(state: dict) -> dict:
    """``{label id: label dict}``."""
    out = {}
    for label in state.get("labels", []):
        out[str(label["id"])] = label
    return out


def label_ref(label_id, labels_after: dict, labels_before: dict, new_ids: set) -> str:
    """A label id rendered with its name: ``label_17 "Action Required"``; a brand-new label as ``"X" (the new label)``."""
    key = str(label_id)
    if key in new_ids and key in labels_after:
        return f'{quote(labels_after[key].get("name", ""))} (the new label)'
    label = labels_after.get(key)
    if label is None:
        label = labels_before.get(key)
    if label is None:
        return key
    name = str(label.get("name", ""))
    if name == "" or name == key:
        return key
    return f"{key} {quote(name)}"


def render_scalar(value) -> str:
    if value is None:
        return "none"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value == "":
        return "empty"
    if _is_list(value) or _is_dict(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, str):
        return quote(value)
    return str(value)


def render_color(color) -> str:
    if color is None:
        return "none"
    if _is_dict(color):
        parts = []
        for name in sorted(color):
            parts.append(f"{name} {color[name]}")
        return ", ".join(parts)
    return str(color)


def person(entry) -> str:
    """``Name <email>`` for a ``{name, email}`` dict."""
    if _is_dict(entry):
        name = str(entry.get("name", ""))
        address = str(entry.get("email", ""))
        if name == "":
            return address
        return f"{name} <{address}>"
    return str(entry)


def person_list(entries) -> str:
    parts = []
    for entry in entries:
        parts.append(person(entry))
    return ", ".join(parts)


def email_who(email: dict) -> str:
    """How to find an email: its subject and sender."""
    return f'{quote(_shorten(str(email.get("subject", "")), LONG_SUBJECT))} from {person(email.get("from"))}'


# ---- the diff, per kind --------------------------------------------------------------------------


def _by_key(items: list, key_name: str) -> dict:
    out = {}
    for item in items:
        out[str(item[key_name])] = item
    return out


def _names_of(old: dict, new: dict) -> list:
    names = []
    for name in old:
        names.append(name)
    for name in new:
        if name not in names:
            names.append(name)
    return names


def _list_change(old: list, new: list, render_item) -> str:
    """``add a, b; remove c`` for the set difference of two lists ("" for a pure re-order)."""
    old_keys = {}
    for item in old:
        old_keys[json.dumps(item, sort_keys=True)] = item
    new_keys = {}
    for item in new:
        new_keys[json.dumps(item, sort_keys=True)] = item
    added = []
    for key in new_keys:
        if key not in old_keys:
            added.append(render_item(new_keys[key], True))
    removed = []
    for key in old_keys:
        if key not in new_keys:
            text = render_item(old_keys[key], False)
            if text != "":
                removed.append(text)
    parts = []
    if len(added) > 0:
        parts.append("add " + ", ".join(added))
    if len(removed) > 0:
        parts.append("remove " + ", ".join(removed))
    return "; ".join(parts)


def diff_emails(before: dict, after: dict, labels_after: dict, labels_before: dict, new_label_ids: set,
                deleted_label_ids: set) -> list[dict]:
    changes = []
    old_by_id = _by_key(before.get("emails", []), "id")
    new_by_id = _by_key(after.get("emails", []), "id")

    def render_label_item(item, is_added):
        if not is_added and str(item) in deleted_label_ids:
            return ""  # the label was deleted; stripping it from the mail is a side effect
        return label_ref(item, labels_after, labels_before, new_label_ids)

    for eid in old_by_id:
        old = old_by_id[eid]
        if eid not in new_by_id:
            changes.append({"op": "deleted", "kind": "email", "id": eid, "who": email_who(old), "text": ""})
            continue
        new = new_by_id[eid]
        pieces = []
        for name in _names_of(old, new):
            if name in IGNORED_FIELDS or name.startswith("_"):
                continue
            if old.get(name) == new.get(name):
                continue
            if name == "labels":
                text = _list_change(old.get(name, []), new.get(name, []), render_label_item)
                if text != "":
                    pieces.append("labels: " + text)
            else:
                pieces.append(f"{DISPLAY_NAMES.get(name, name)} {render_scalar(old.get(name))} → "
                              f"{render_scalar(new.get(name))}")
        if len(pieces) > 0:
            changes.append({"op": "changed", "kind": "email", "id": eid, "who": email_who(new),
                            "text": "; ".join(pieces)})
    for eid in new_by_id:
        if eid in old_by_id:
            continue
        email = new_by_id[eid]
        parts = []
        thread_of = None
        for other in before.get("emails", []):
            if other.get("threadId") == email.get("threadId") and email.get("threadId") is not None:
                thread_of = other
        if thread_of is not None:
            parts.append(f'in the thread of email {thread_of["id"]} {email_who(thread_of)}')
        if email.get("from") and email.get("isSent") is not True and email.get("isDraft") is not True:
            parts.append(f"from={person(email.get('from'))}")
        parts.append(f"to=[{person_list(email.get('to', []))}]")
        if len(email.get("cc", [])) > 0:
            parts.append(f"cc=[{person_list(email['cc'])}]")
        if len(email.get("bcc", [])) > 0:
            parts.append(f"bcc=[{person_list(email['bcc'])}]")
        parts.append(f"subject={quote(email.get('subject', ''))}")
        parts.append(f"body={quote(email.get('body', ''))}")
        if len(email.get("attachments", [])) > 0:
            names = []
            for attachment in email["attachments"]:
                names.append(str(attachment.get("name", attachment)))
            parts.append("attachments=[" + ", ".join(names) + "]")
        if email.get("isStarred"):
            parts.append("starred")
        item_labels = []
        for lid in email.get("labels", []):
            if str(lid) in new_label_ids or str(lid) not in ("SENT", "DRAFT", "INBOX"):
                item_labels.append(label_ref(lid, labels_after, labels_before, new_label_ids))
        if len(item_labels) > 0:
            parts.append("labels=[" + ", ".join(item_labels) + "]")
        if email.get("isDraft") is True:
            what = "draft email (save it as a draft, do not send)"
        elif email.get("isSent") is True:
            what = "email (compose and SEND it)"
        else:
            what = "email"
        changes.append({"op": "created", "kind": "email", "id": eid, "who": what, "text": ", ".join(parts)})
    return changes


def diff_labels(before: dict, after: dict, labels_after: dict, labels_before: dict, new_label_ids: set
                ) -> list[dict]:
    changes = []
    for lid in labels_before:
        old = labels_before[lid]
        if lid not in labels_after:
            changes.append({"op": "deleted", "kind": "label", "id": lid, "who": quote(old.get("name", "")),
                            "text": ""})
            continue
        new = labels_after[lid]
        pieces = []
        for name in _names_of(old, new):
            if name in IGNORED_FIELDS or name.startswith("_") or old.get(name) == new.get(name):
                continue
            if name == "color":
                pieces.append(f"color {render_color(old.get(name))} → {render_color(new.get(name))}")
            elif name == "parentId":
                pieces.append(f"parent {label_ref(old.get(name), labels_after, labels_before, new_label_ids)}"
                              f" → {label_ref(new.get(name), labels_after, labels_before, new_label_ids)}")
            else:
                pieces.append(f"{name} {render_scalar(old.get(name))} → {render_scalar(new.get(name))}")
        if len(pieces) > 0:
            changes.append({"op": "changed", "kind": "label", "id": lid, "who": quote(old.get("name", "")),
                            "text": "; ".join(pieces)})
    for lid in labels_after:
        if lid in labels_before:
            continue
        new = labels_after[lid]
        parts = []
        if new.get("parentId") is not None:
            parts.append("nested under " + label_ref(new["parentId"], labels_after, labels_before, new_label_ids))
        else:
            parts.append("top level (no parent)")
        if new.get("color") is not None:
            parts.append("color " + render_color(new["color"]))
        if new.get("visible") is False:
            parts.append("hidden")
        changes.append({"op": "created", "kind": "label", "id": lid, "who": quote(new.get("name", "")),
                        "text": ", ".join(parts)})
    return changes


def filter_summary(flt: dict) -> str:
    """The criteria of a filter as ``from=x, subject=y`` (only the non-blank ones)."""
    parts = []
    criteria = flt.get("criteria", {})
    for name in criteria:
        value = criteria[name]
        if name in FILTER_SKIP_DEFAULTS and criteria.get("size") is None:
            continue
        if value is None or value is False or value == "":
            continue
        parts.append(f"{name}={render_scalar(value)}")
    if len(parts) == 0:
        return "no criteria"
    return ", ".join(parts)


def _filter_actions(flt: dict, labels_after: dict, labels_before: dict, new_label_ids: set) -> str:
    parts = []
    actions = flt.get("actions", {})
    for name in actions:
        value = actions[name]
        if value is None or value is False or value == "":
            continue
        if name == "label":
            parts.append("apply label " + label_ref(value, labels_after, labels_before, new_label_ids))
        elif value is True:
            parts.append(name)
        else:
            parts.append(f"{name}={render_scalar(value)}")
    if len(parts) == 0:
        return "no actions"
    return ", ".join(parts)


def diff_filters(before: dict, after: dict, labels_after: dict, labels_before: dict, new_label_ids: set
                 ) -> list[dict]:
    changes = []
    old_by_id = _by_key(before.get("filters", []), "id")
    new_by_id = _by_key(after.get("filters", []), "id")
    for fid in old_by_id:
        old = old_by_id[fid]
        if fid not in new_by_id:
            changes.append({"op": "deleted", "kind": "filter", "id": fid, "who": f"({filter_summary(old)})",
                            "text": ""})
            continue
        new = new_by_id[fid]
        pieces = []
        for part in ("criteria", "actions"):
            old_part = old.get(part, {})
            new_part = new.get(part, {})
            for name in _names_of(old_part, new_part):
                if old_part.get(name) == new_part.get(name):
                    continue
                if part == "actions" and name == "label":
                    pieces.append("action label "
                                  f"{label_ref(old_part.get(name), labels_after, labels_before, new_label_ids)}"
                                  " → "
                                  f"{label_ref(new_part.get(name), labels_after, labels_before, new_label_ids)}")
                else:
                    word = "criteria" if part == "criteria" else "action"
                    pieces.append(f"{word} {name} {render_scalar(old_part.get(name))} → "
                                  f"{render_scalar(new_part.get(name))}")
        if old.get("enabled") != new.get("enabled"):
            pieces.append(f"enabled {render_scalar(old.get('enabled'))} → {render_scalar(new.get('enabled'))}")
        if len(pieces) > 0:
            changes.append({"op": "changed", "kind": "filter", "id": fid, "who": f"({filter_summary(old)})",
                            "text": "; ".join(pieces)})
    for fid in new_by_id:
        if fid in old_by_id:
            continue
        flt = new_by_id[fid]
        text = (f"criteria [{filter_summary(flt)}], actions [{_filter_actions(flt, labels_after, labels_before, new_label_ids)}]")
        if flt.get("enabled") is False:
            text += ", disabled"
        changes.append({"op": "created", "kind": "filter", "id": fid, "who": "", "text": text})
    return changes


def _flatten(prefix: str, value, out: dict) -> None:
    """Flatten nested dicts / lists of dicts into ``{path: leaf}`` (list items are written ``item N``, 1-based)."""
    if _is_dict(value):
        for name in value:
            if prefix == "":
                _flatten(name, value[name], out)
            else:
                _flatten(f"{prefix} {name}", value[name], out)
    elif _is_list(value) and len(value) > 0 and _is_dict(value[0]):
        position = 1
        for item in value:
            _flatten(f"{prefix} item {position}", item, out)
            position += 1
    else:
        out[prefix] = value


def diff_settings(before: dict, after: dict, key: str, kind: str) -> list[dict]:
    """Leaf-level changes of a dict-valued state key (``settings``, ``currentUser``)."""
    old_flat = {}
    new_flat = {}
    _flatten("", before.get(key, {}), old_flat)
    _flatten("", after.get(key, {}), new_flat)
    paths = []
    for path in old_flat:
        paths.append(path)
    for path in new_flat:
        if path not in paths:
            paths.append(path)
    changes = []
    for path in paths:
        old = old_flat.get(path)
        new = new_flat.get(path)
        if old == new or path.startswith("_"):
            continue
        top = path.split(" ")[0]
        if _is_list(old) and _is_list(new):
            text = _list_change(old, new, lambda item, added: render_scalar(item))
            if text == "":
                continue
        else:
            text = f"{render_scalar(old)} → {render_scalar(new)}"
        where = ""
        if kind == "setting":
            where = f" (Settings > {SETTING_TABS.get(top, 'general')} tab)"
        changes.append({"op": "changed", "kind": kind, "id": path, "who": where, "text": text})
    return changes


def diff_contacts(before: dict, after: dict) -> list[dict]:
    changes = []
    old_by_id = _by_key(before.get("contacts", []), "id")
    new_by_id = _by_key(after.get("contacts", []), "id")
    for cid in old_by_id:
        old = old_by_id[cid]
        who = f'{quote(old.get("name", ""))} <{old.get("email", "")}>'
        if cid not in new_by_id:
            changes.append({"op": "deleted", "kind": "contact", "id": cid, "who": who, "text": ""})
            continue
        new = new_by_id[cid]
        pieces = []
        for name in _names_of(old, new):
            if name in IGNORED_FIELDS or name.startswith("_") or old.get(name) == new.get(name):
                continue
            pieces.append(f"{name} {render_scalar(old.get(name))} → {render_scalar(new.get(name))}")
        if len(pieces) > 0:
            changes.append({"op": "changed", "kind": "contact", "id": cid, "who": who, "text": "; ".join(pieces)})
    for cid in new_by_id:
        if cid in old_by_id:
            continue
        new = new_by_id[cid]
        parts = [f"name={quote(new.get('name', ''))}", f"email={new.get('email', '')}"]
        changes.append({"op": "created", "kind": "contact", "id": cid, "who": "", "text": ", ".join(parts)})
    return changes


def diff_blocked(before: dict, after: dict) -> list[dict]:
    changes = []
    old_by = _by_key(before.get("blockedSenders", []), "email")
    new_by = _by_key(after.get("blockedSenders", []), "email")
    for address in old_by:
        if address not in new_by:
            changes.append({"op": "deleted", "kind": "blocked", "id": address, "who": address, "text": ""})
    for address in new_by:
        if address not in old_by:
            changes.append({"op": "created", "kind": "blocked", "id": address, "who": address, "text": ""})
    return changes


def diff_states(before: dict, after: dict) -> list[dict]:
    """Concrete changes between two states as ``[{op, kind, id, who, text}]`` (emails, labels, filters, settings,
    account, contacts, blocked senders), volatile fields ignored."""
    labels_before = label_index(before)
    labels_after = label_index(after)
    new_label_ids = set()
    deleted_label_ids = set()
    for lid in labels_after:
        if lid not in labels_before:
            new_label_ids.add(lid)
    for lid in labels_before:
        if lid not in labels_after:
            deleted_label_ids.add(lid)
    changes = []
    changes.extend(diff_labels(before, after, labels_after, labels_before, new_label_ids))
    changes.extend(diff_emails(before, after, labels_after, labels_before, new_label_ids, deleted_label_ids))
    changes.extend(diff_filters(before, after, labels_after, labels_before, new_label_ids))
    changes.extend(diff_settings(before, after, "settings", "setting"))
    changes.extend(diff_settings(before, after, "currentUser", "account field"))
    changes.extend(diff_contacts(before, after))
    changes.extend(diff_blocked(before, after))
    return changes


# ---- rendering a task's entry --------------------------------------------------------------------


def _target(change: dict) -> str:
    """``email 15 "Subject" from Name <a@b>`` / ``label_17 "Name"`` / ``filter_3 (from=x)``."""
    kind = change["kind"]
    who = change["who"]
    if kind == "email":
        return f'email {change["id"]} {who}'
    if kind in ("label", "filter", "contact"):
        if who == "":
            return str(change["id"])
        return f'{change["id"]} {who}'
    if kind == "setting":
        return f'setting {change["id"]}{who}'
    if kind == "account field":
        return f'account field {change["id"]}'
    return str(change["id"])


def render_changes(changes: list[dict]) -> str:
    """The right-hand side of an answer-key line for one task.

    Created things are listed first-seen in order; ``changed`` entities of one kind with the identical change
    text (e.g. ``in trash false → true`` on three emails) are grouped into one clause."""
    pieces = []
    done = set()
    for k in range(len(changes)):
        if k in done:
            continue
        change = changes[k]
        done.add(k)
        op = change["op"]
        kind = change["kind"]
        if op == "created":
            if kind == "email":
                pieces.append(f'CREATE {change["who"]} with {change["text"]}.')
            elif kind == "blocked":
                pieces.append(f'BLOCK sender {change["id"]} (Settings > filters tab, blocked addresses).')
            elif kind == "filter":
                pieces.append(f'CREATE filter with {change["text"]}.')
            elif kind == "label":
                pieces.append(f'CREATE label {change["who"]}, {change["text"]}.')
            else:
                pieces.append(f'CREATE {kind} with {change["text"]}.')
        elif op == "deleted":
            group = [change]
            for j in range(k + 1, len(changes)):
                if changes[j]["op"] == "deleted" and changes[j]["kind"] == kind:
                    group.append(changes[j])
                    done.add(j)
            if kind == "blocked":
                addresses = []
                for member in group:
                    addresses.append(member["id"])
                pieces.append("UNBLOCK sender " + ", ".join(addresses) + ".")
            elif kind == "email":
                targets = []
                for member in group:
                    targets.append(f'{member["id"]} {member["who"]}')
                pieces.append("PERMANENTLY DELETE email " + "; email ".join(targets) + ".")
            else:
                targets = []
                for member in group:
                    targets.append(_target(member))
                pieces.append(f"DELETE {kind} " + "; ".join(targets) + ".")
        else:
            group = [change]
            for j in range(k + 1, len(changes)):
                other = changes[j]
                if kind in ("setting", "account field"):
                    break  # each setting is stated on its own line
                if other["op"] == "changed" and other["kind"] == kind and other["text"] == change["text"]:
                    group.append(other)
                    done.add(j)
            targets = []
            for member in group:
                targets.append(_target(member))
            pieces.append("; ".join(targets) + ": " + change["text"] + ".")
    return " ".join(pieces)


def render_entry(instruction: str, changes: list[dict]) -> str:
    return f'- "{instruction}" → ' + render_changes(changes)


# ---- solving every task ----------------------------------------------------------------------------


def solve_all(task_ids: list[str], seed_state: dict, solvers: dict, allow: dict[str, str] | None = None
              ) -> tuple[dict[str, list[dict]], list[str], list[str], list[str]]:
    """``(changes by task, problems, empty-diff task ids, solver-failure task ids)``.

    A problem is an empty diff, a missing solver or a raising solver that is not allow-listed."""
    if allow is None:
        allow = ALLOW_LIST
    out: dict[str, list[dict]] = {}
    problems: list[str] = []
    empty: list[str] = []
    failed: list[str] = []
    for task_id in task_ids:
        solver = solvers.get(task_id)
        if solver is None:
            out[task_id] = []
            failed.append(task_id)
            if task_id not in allow:
                problems.append(f"{task_id}: no solve function")
            continue
        state = copy.deepcopy(seed_state)
        try:
            solver(state)
        except Exception as err:  # noqa: BLE001 -- any solver failure must be reported by task, not abort the scan
            out[task_id] = []
            failed.append(task_id)
            if task_id not in allow:
                problems.append(f"{task_id}: solve function raised {err!r}")
            continue
        changes = diff_states(seed_state, state)
        out[task_id] = changes
        if len(changes) == 0:
            empty.append(task_id)
            if task_id not in allow:
                problems.append(f"{task_id}: EMPTY diff (the solver changed nothing but bookkeeping)")
    return out, problems, empty, failed


# ---- the prompt and the pool -----------------------------------------------------------------------


def build_prompt(tasks: list[dict], changes_by_task: dict[str, list[dict]]) -> tuple[str, list[str]]:
    """``(prompt guidance text, entry lines)`` for the given bank tasks (in the order given)."""
    entries = []
    for task in tasks:
        entries.append(render_entry(str(task["instruction"]), changes_by_task[str(task["id"])]))
    text = HEADER + "\n\n" + KEY_NOTES + "\n\n" + KEY_TITLE + "\n" + "\n".join(entries) + "\n"
    return text, entries


def build_all(seed_state: dict | None = None, solvers: dict | None = None, bank: list[dict] | None = None,
              allow: dict[str, str] | None = None, expected_tasks: int | None = N_TASKS) -> dict:
    """Everything the pool holds, computed without touching the disk. Raises `BuildError` on any problem."""
    if bank is None:
        bank = load_bank()
    if seed_state is None or solvers is None:
        module = load_sanity_module()
        if seed_state is None:
            seed_state = load_seed(module)
        if solvers is None:
            solvers = module.SOLVERS
    if expected_tasks is not None and len(bank) != expected_tasks:
        raise BuildError(f"expected {expected_tasks} Gmail tasks, found {len(bank)}")
    by_id = {}
    for task in bank:
        by_id[str(task["id"])] = task
    all_ids = sorted(by_id)
    changes, problems, empty, failed = solve_all(all_ids, seed_state, solvers, allow)
    if len(problems) > 0:
        raise BuildError("answer key cannot be built; these tasks have no usable diff:\n  " + "\n  ".join(problems))
    tasks = []
    for task_id in all_ids:
        tasks.append(by_id[task_id])
    guidance, entries = build_prompt(tasks, changes)
    arm = make_arm(ARM, guidance, VECTOR)
    return {"pool": POOL, "arm_id": ARM, "task_ids": all_ids, "guidance": guidance, "entries": entries,
            "chars": len(guidance), "approx_tokens": len(guidance) // 4, "arm": arm, "changes": changes,
            "problems": problems, "empty": empty, "failed": failed}


def write_oracle_pool(path: Path, built: dict) -> None:
    meta = {"purpose": "full-coverage Gmail positive-control answer key built from sanity_check_real.py solvers",
            "task_ids": list(built["task_ids"]),
            "generator": "experiments/growing_bandits/retrospective/build_gmail_oracle.py"}
    make_pools.write_pool(path, POOL, [built["arm"]], TEMPLATE, AXES_PATH, meta)


def pool_path(out_dir: Path) -> Path:
    return Path(out_dir) / f"{POOL}.yaml"


def report(built: dict, path: Path | None = None) -> str:
    lines = [f"pool {built['pool']} (arm {built['arm_id']}): {len(built['entries'])} entries, "
             f"{built['chars']} chars, ~{built['approx_tokens']} tokens (chars/4)",
             "empty diffs: " + (", ".join(built["empty"]) if len(built["empty"]) > 0 else "none"),
             "solver failures: " + (", ".join(built["failed"]) if len(built["failed"]) > 0 else "none")]
    if path is not None:
        lines.append(f"wrote {path}")
        lines.append(f"sha256 {file_sha256(path)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="directory for the pool YAML")
    ap.add_argument("--dry-run", action="store_true", help="print the report only; write nothing")
    ap.add_argument("--show", type=str, default=None, help="print the entries of these comma-separated task ids")
    args = ap.parse_args(argv)
    try:
        built = build_all()
    except BuildError as err:
        print(f"BUILD FAILED: {err}", file=sys.stderr)
        return 2
    if args.show is not None:
        wanted = args.show.split(",")
        for task in load_bank():
            if str(task["id"]) in wanted:
                print(render_entry(str(task["instruction"]), built["changes"][str(task["id"])]))
    if args.dry_run:
        print(report(built))
        return 0
    path = pool_path(args.out_dir)
    write_oracle_pool(path, built)
    print(report(built, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
