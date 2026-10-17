"""Build the full-coverage positive-control arm ("answer key" oracle) for the GitLab tasks, mechanically.

    ./.venv/bin/python experiments/growing_bandits/retrospective/build_full_oracle.py            # report only
    ./.venv/bin/python experiments/growing_bandits/retrospective/build_full_oracle.py --write    # also write pool YAMLs

Why.  Leg G's planted positive control, ``GL_anchor_oracle``, hard-codes a cookbook for only 40 bank tasks
(20 of the 110 Leg G tasks), so the control could not have succeeded. This script instead derives an answer key
for EVERY task from ground truth: ``webarena-infinity/apps/gitlab-plan-and-track/sanity_check_real.py`` defines
``solve_task_<id>(state)`` for each real task (it mutates the seed state into the expected end-state). We load
the seed state once (the sanity check's own ``generate_seed_state`` -- it shells out to node), deep-copy it,
run the solver, and DIFF before/after into concrete, human-readable changes (entity, id, title, field old → new,
label/user/milestone/epic ids mapped to their titles). One compact cookbook line per task, keyed by the task's
exact instruction text. Version 1 put it after ``GL_anchor_oracle``'s own routing header with that arm's ``vector``;
its block-A pilot scored 0.633 (gate 0.80), so version 2 (the default; ``HEADER_V2``, ``VECTOR_V2``) states an
explicit look-up-then-verify procedure. Only block-A (non-study) tasks were used to revise it.

Two pools are written (same YAML format and sha256 as every other pool, via ``make_pools.write_pool``):

* ``oraclefull_pilot`` / arm ``GL_oracle_full_pilot``: the 30 "block A" tasks (the bank minus the 110 Leg G tasks);
* ``oraclefull`` / arm ``GL_oracle_full``: exactly the 110 Leg G tasks.

A task whose diff is EMPTY or whose solver raises fails the build loudly unless it is in ``ALLOW_LIST`` with a
reason. No model is called, no server is started.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "empirical"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import make_pools  # noqa: E402

from cold_start.types import Arm, PromptVector  # noqa: E402

APP = "gitlab-plan-and-track"
WINNER_CHECK = ROOT / "data" / "winner_check_gitlab"
ANCHORS_REL = "pools/anchors_gitlab.yaml"
SOURCE_ARM = "GL_anchor_oracle"
TEMPLATE = ROOT / "configs" / "template_gitlab.jinja"
AXES_PATH = ROOT / "configs" / "axes.yaml"

POOL, ARM = "oraclefull", "GL_oracle_full"
N_PILOT_TASKS = 30
N_STUDY_TASKS = 110

#: Header cut: everything in the old oracle's guidance before this marker is the generic routing header.
HEADER_END_MARKER = "Exact issue/epic/milestone/iteration ids:"
#: The old header's sentence that points at the "cookbook"; replaced so there is ONE consistent instruction.
OLD_ROUTING_SENTENCE = ("match the visible task instruction to the cookbook below and execute the\n"
                        "listed route.")
NEW_ROUTING_SENTENCE = "match the visible task instruction to the answer key below and produce the listed final state."
ANSWER_KEY_LINE = ("Match the visible task instruction to the answer key below; it lists the exact final "
                   "state to produce. Make exactly those changes in the UI, save, and finish.")

#: Version 2 header (2026-10-10, after the v1 pilot scored 0.633 < 0.80 on block A: the agent ignored the long key,
#: picked items by searching, and finished without checking -- v1 inherited "Save once, then finish. Do not
#: repeatedly verify" from the old oracle). v2 makes the lookup and a final check explicit. Tuned on block A only.
HEADER_V2 = """This is a synthetic positive-control arm. Below is an answer key: one line per task, giving the
task's exact instruction in quotes, then the exact items (by id, such as issue_17 or epic_11) and the exact
final values to produce.

Procedure, every time:
1. Before any other action, find the ONE answer-key line whose quoted instruction is identical to your task.
   In your first reasoning step, copy that line's target state.
2. Use exactly the ids it lists, and all of them. Do not search for or guess which items are meant: the key
   already names them.
3. Open each item directly by URL on the same host and port as the start page, e.g. <start host>/#/issues/issue_17
   (view) or <start host>/#/edit-issue/issue_17 (edit form); epics are at #/epics/epic_11; milestones, labels,
   iterations and todos have their own pages in the sidebar.
4. Make every listed change and save.
5. Before you finish, reopen each listed item and check every listed value against the key. If any value differs,
   fix it and save again. Finish only when every listed value matches."""
ANSWER_KEY_LINE_V2 = "Answer key lines are matched on the quoted instruction, word for word."
#: v2 vector: minimal actions and an explicit verification step (v1 copied the old oracle's agency 2, "explore
#: multiple paths", which contradicts a looked-up answer).
VECTOR_V2 = {"planning": 0, "verification": 2, "agency": 0, "expertise": 3, "format": 0, "goal": 0}
KEY_TITLE = "Answer key (per-task cookbook):"

#: task id -> reason. A task listed here may have an empty diff / a raising solver without failing the build.
ALLOW_LIST: dict[str, str] = {}

#: Fields that are bookkeeping, never reported (timestamps; the counters start with an underscore).
IGNORED_FIELDS = ("updatedAt", "closedAt", "createdAt", "id", "iid")
#: State keys that hold lists of entities with an ``id``, and what each kind is called.
KINDS = {
    "issues": "issue", "epics": "epic", "milestones": "milestone", "labels": "label", "iterations": "iteration",
    "iterationCadences": "cadence", "boards": "board", "todos": "todo", "timelogs": "timelog", "users": "user",
}
#: Field names as the UI labels them.
DISPLAY_NAMES = {
    "milestoneId": "milestone", "iterationId": "iteration", "epicId": "epic", "parentEpicId": "parent epic",
    "cadenceId": "cadence", "healthStatus": "health status", "dueDate": "due date", "startDate": "start date",
    "timeEstimate": "time estimate", "timeSpent": "time spent", "rollOver": "roll over",
    "durationWeeks": "duration (weeks)", "upcomingIterations": "upcoming iterations",
}
#: Fields of a created entity not worth stating (defaults / set by the app).
CREATED_SKIP = ("status", "textColor", "spentAt", "active")
SECONDS_FIELDS = ("timeEstimate", "timeSpent")
LONG_TEXT = 140


class BuildError(RuntimeError):
    """The answer key cannot be built as specified (empty diff, failed solver, wrong task counts, ...)."""


# ---- ground truth ------------------------------------------------------------------------------


def webarena_root() -> Path:
    from cold_start.tasks.webarena import _webarena_root

    return Path(_webarena_root())


def app_dir() -> Path:
    return webarena_root() / "apps" / APP


def bank_path() -> Path:
    return app_dir() / "real-tasks.json"


def load_bank() -> list[dict]:
    return json.loads(bank_path().read_text(encoding="utf-8"))


def load_sanity_module():
    """``sanity_check_real.py`` imported from its path (it needs ``requests``; its loader needs node)."""
    path = app_dir() / "sanity_check_real.py"
    spec = importlib.util.spec_from_file_location("sanity_check_real_for_oracle", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- the diff ----------------------------------------------------------------------------------


def _display_seconds(value) -> str:
    total = int(value)
    if total == 0:
        return "0"
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    text = ""
    if hours > 0:
        text += f"{hours}h"
    if minutes > 0:
        text += f"{minutes}m"
    if seconds > 0:
        text += f"{seconds}s"
    return text


def entity_title(kind: str, entity: dict, index: dict) -> str:
    """The human name of an entity (issue title, label title, board name, user name, ...)."""
    if kind == "board":
        return str(entity.get("name", ""))
    if kind == "user":
        return str(entity.get("name", ""))
    if kind == "todo":
        target = index.get(entity.get("targetId"))
        target_title = ""
        if target is not None:
            target_title = str(target[1].get("title", ""))
        return f'{entity.get("action", "")} todo for {entity.get("targetId")} ({target_title})'
    if kind == "timelog":
        issue = index.get(entity.get("issueId"))
        issue_title = ""
        if issue is not None:
            issue_title = str(issue[1].get("title", ""))
        return f'time entry on {entity.get("issueId")} ({issue_title})'
    return str(entity.get("title", ""))


def build_index(state: dict) -> dict:
    """``{entity id: (kind, entity dict)}`` over every entity list of a state."""
    index = {}
    for key in KINDS:
        for entity in state.get(key, []):
            index[str(entity["id"])] = (KINDS[key], entity)
    return index


def ref(value, index: dict) -> str:
    """An id rendered with its title/name when it names an entity (``lbl_19 needs-discussion``), else as is."""
    if value is None:
        return "none"
    text = str(value)
    hit = index.get(text)
    if hit is None:
        return text
    kind, entity = hit
    if kind.endswith("*"):
        base = kind[:-1]
        return f'"{entity_title(base, entity, index)}" (the new {base})'
    title = entity_title(kind, entity, index)
    if kind in ("todo", "timelog") or title == "":
        return text
    return f"{text} {title}"


def _shorten(text: str) -> str:
    if len(text) <= LONG_TEXT:
        return text
    return text[:LONG_TEXT] + "..."


def render_value(field: str, value, index: dict) -> str:
    if value is None:
        return "none"
    if field in SECONDS_FIELDS:
        return _display_seconds(value)
    if value is True:
        return "true"
    if value is False:
        return "false"
    if _is_list(value) or _is_dict(value):
        return json.dumps(value, sort_keys=True)
    text = str(value)
    if text in index:
        return ref(text, index)
    if text == "":
        return "empty"
    return _shorten(text)


def _is_list(value) -> bool:
    return hasattr(value, "append")


def _is_dict(value) -> bool:
    return hasattr(value, "keys")


def _item_key(item) -> str:
    return json.dumps(item, sort_keys=True)


def _render_list_item(item, index: dict) -> str:
    if _is_dict(item):
        parts = []
        for name in sorted(item):
            if name in ("id", "position"):
                continue
            parts.append(f"{name}={render_value(name, item[name], index)}")
        return "{" + ", ".join(parts) + "}"
    return ref(item, index)


def display_name(field: str) -> str:
    return DISPLAY_NAMES.get(field, field)


def field_changes(field: str, old, new, index_old: dict, index_new: dict, deleted_ids: set | None = None) -> str:
    """One field's change as text: ``status open → closed`` or ``labels: add lbl_19 needs-discussion``.

    Returns "" for a change that is only a side effect (a reference to a deleted entity being cleared, or a
    pure re-ordering of a list)."""
    if deleted_ids is None:
        deleted_ids = set()
    if _is_list(old) and _is_list(new):
        old_keys = {}
        for item in old:
            old_keys[_item_key(item)] = item
        new_keys = {}
        for item in new:
            new_keys[_item_key(item)] = item
        added = []
        for key in new_keys:
            if key not in old_keys:
                added.append(_render_list_item(new_keys[key], index_new))
        removed = []
        for key in old_keys:
            if key not in new_keys and str(old_keys[key]) not in deleted_ids:
                removed.append(_render_list_item(old_keys[key], index_old))
        parts = []
        if len(added) > 0:
            parts.append("add " + ", ".join(added))
        if len(removed) > 0:
            parts.append("remove " + ", ".join(removed))
        if len(parts) == 0:
            return ""
        return f"{display_name(field)}: " + "; ".join(parts)
    if new is None and old is not None and str(old) in deleted_ids:
        return ""
    name = display_name(field)
    return f"{name} {render_value(field, old, index_old)} → {render_value(field, new, index_new)}"


def _created_fields(kind: str, entity: dict, index: dict) -> str:
    parts = []
    for name in entity:
        if name in IGNORED_FIELDS or name == "authorId" or name.startswith("_"):
            continue
        if name in CREATED_SKIP or (kind == "timelog" and name == "issueId"):
            continue
        value = entity[name]
        if value is None or value is False or value == "" or value == [] or value == {} or value == 0:
            continue
        if name == "title" and kind != "todo":
            continue
        if name == "name" and kind == "board":
            continue
        if _is_list(value):
            rendered = []
            for item in value:
                rendered.append(_render_list_item(item, index))
            parts.append(f"{display_name(name)}=[" + ", ".join(rendered) + "]")
        else:
            parts.append(f"{display_name(name)}={render_value(name, value, index)}")
    return ", ".join(parts)


def diff_states(before: dict, after: dict) -> list[dict]:
    """Concrete changes between two states as ``[{op, kind, id, title, text}]``.

    ``op`` is ``changed`` / ``created`` / ``deleted``. A todo or timelog that disappears only because the
    entity it targets was deleted is a side effect and is not reported.
    """
    index_before = build_index(before)
    index_after = build_index(after)
    for eid in list(index_after):
        if eid not in index_before:
            index_after[eid] = (index_after[eid][0] + "*", index_after[eid][1])
    deleted_ids = set()
    for key in KINDS:
        old_ids = set()
        for entity in before.get(key, []):
            old_ids.add(str(entity["id"]))
        new_ids = set()
        for entity in after.get(key, []):
            new_ids.add(str(entity["id"]))
        for gone in old_ids - new_ids:
            deleted_ids.add(gone)

    changes = []
    for key in KINDS:
        kind = KINDS[key]
        old_by_id = {}
        for entity in before.get(key, []):
            old_by_id[str(entity["id"])] = entity
        new_by_id = {}
        for entity in after.get(key, []):
            new_by_id[str(entity["id"])] = entity
        for eid in old_by_id:
            old = old_by_id[eid]
            if eid not in new_by_id:
                side_effect = False
                if kind == "todo" and str(old.get("targetId")) in deleted_ids:
                    side_effect = True
                if kind == "timelog" and str(old.get("issueId")) in deleted_ids:
                    side_effect = True
                if not side_effect:
                    title = entity_title(kind, old, index_before)
                    changes.append({"op": "deleted", "kind": kind, "id": eid, "title": title,
                                    "text": "DELETE"})
                continue
            new = new_by_id[eid]
            pieces = []
            names = []
            for name in old:
                names.append(name)
            for name in new:
                if name not in names:
                    names.append(name)
            for name in names:
                if name in IGNORED_FIELDS or name.startswith("_"):
                    continue
                if old.get(name) != new.get(name):
                    piece = field_changes(name, old.get(name), new.get(name), index_before, index_after,
                                          deleted_ids)
                    if piece != "":
                        pieces.append(piece)
            if len(pieces) > 0:
                title = entity_title(kind, new, index_after)
                changes.append({"op": "changed", "kind": kind, "id": eid, "title": title,
                                "text": "; ".join(pieces)})
        for eid in new_by_id:
            if eid not in old_by_id:
                new = new_by_id[eid]
                title = entity_title(kind, new, index_after)
                changes.append({"op": "created", "kind": kind, "id": eid, "title": title,
                                "text": _created_fields(kind, new, index_after)})
    return changes


GROUP_TITLE_CHARS = 45


def _short_title(title: str) -> str:
    if len(title) <= GROUP_TITLE_CHARS:
        return title
    return title[:GROUP_TITLE_CHARS].rstrip() + "..."


def render_changes(changes: list[dict]) -> str:
    """The right-hand side of a cookbook line for one task.

    Consecutive ``changed`` entities with the identical change text (e.g. ``status open → closed`` on
    three issues) are grouped into one clause: ``issue_30 "..", issue_41 "..": status open → closed.``"""
    pieces = []
    k = 0
    while k < len(changes):
        change = changes[k]
        if change["op"] == "created":
            head = f'CREATE {change["kind"]} "{change["title"]}"'
            if change["text"] != "":
                head += f' with {change["text"]}'
            pieces.append(head + ".")
            k += 1
        elif change["op"] == "deleted":
            pieces.append(f'DELETE {change["id"]} "{change["title"]}".')
            k += 1
        else:
            group = [change]
            j = k + 1
            while j < len(changes) and changes[j]["op"] == "changed" and changes[j]["text"] == change["text"]:
                group.append(changes[j])
                j += 1
            targets = []
            for member in group:
                if len(group) == 1:
                    name = member["title"]
                else:
                    name = _short_title(member["title"])
                if name == "":
                    targets.append(member["id"])
                else:
                    targets.append(f'{member["id"]} "{name}"')
            pieces.append(", ".join(targets) + ": " + change["text"] + ".")
            k = j
    return " ".join(pieces)


def render_entry(instruction: str, changes: list[dict]) -> str:
    return f'- "{instruction}" → ' + render_changes(changes)


# ---- solving every task ----------------------------------------------------------------------------


def solve_all(task_ids: list[str], seed_state: dict, solvers: dict, allow: dict[str, str] | None = None
              ) -> tuple[dict[str, list[dict]], list[str]]:
    """``({task id: changes}, problems)`` where a problem is an empty diff or a raising solver that is not allow-listed."""
    if allow is None:
        allow = ALLOW_LIST
    out: dict[str, list[dict]] = {}
    problems: list[str] = []
    for task_id in task_ids:
        solver = solvers.get(task_id)
        if solver is None:
            out[task_id] = []
            if task_id not in allow:
                problems.append(f"{task_id}: no solve function")
            continue
        state = copy.deepcopy(seed_state)
        try:
            solver(state)
        except Exception as err:  # noqa: BLE001 -- any solver failure must be reported by task, not abort the scan
            out[task_id] = []
            if task_id not in allow:
                problems.append(f"{task_id}: solve function raised {err!r}")
            continue
        changes = diff_states(seed_state, state)
        out[task_id] = changes
        if len(changes) == 0 and task_id not in allow:
            problems.append(f"{task_id}: EMPTY diff (the solver changed nothing but bookkeeping)")
    return out, problems


# ---- the prompt and the pools ----------------------------------------------------------------------


def oracle_source(anchors_path: Path | None = None) -> dict:
    """The old oracle's pool-file entry (its guidance and vector)."""
    if anchors_path is None:
        anchors_path = WINNER_CHECK / ANCHORS_REL
    with open(anchors_path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    for entry in doc["arms"]:
        if entry["arm_id"] == SOURCE_ARM:
            return entry
    raise BuildError(f"{SOURCE_ARM} not found in {anchors_path}")


def header_from_guidance(guidance: str) -> str:
    """The generic routing / UI instructions: the old guidance up to (not including) its hard-coded id list."""
    cut = guidance.find(HEADER_END_MARKER)
    if cut < 0:
        raise BuildError(f"marker {HEADER_END_MARKER!r} not found in the {SOURCE_ARM} guidance")
    header = guidance[:cut].strip()
    if OLD_ROUTING_SENTENCE not in header:
        raise BuildError("the old routing sentence was not found in the header; cannot make it consistent")
    return header.replace(OLD_ROUTING_SENTENCE, NEW_ROUTING_SENTENCE)


def build_prompt(header: str, tasks: list[dict], changes_by_task: dict[str, list[dict]]) -> tuple[str, list[str]]:
    """``(prompt guidance text, entry lines)`` for the given bank tasks (in bank order)."""
    entries = []
    for task in tasks:
        entries.append(render_entry(str(task["instruction"]), changes_by_task[str(task["id"])]))
    key_line = ANSWER_KEY_LINE
    if header == HEADER_V2:
        key_line = ANSWER_KEY_LINE_V2
    text = header + "\n\n" + key_line + "\n\n" + KEY_TITLE + "\n" + "\n".join(entries) + "\n"
    return text, entries


def make_arm(arm_id: str, guidance: str, vector: dict) -> Arm:
    return Arm(arm_id=arm_id, name=arm_id, vector=PromptVector(**vector), prompt_guidance=guidance)


def write_oracle_pool(path: Path, pool: str, arm: Arm, task_ids: list[str]) -> None:
    meta = {"purpose": "full-coverage positive-control answer key built from sanity_check_real.py solvers",
            "task_ids": list(task_ids), "generator": "experiments/growing_bandits/retrospective/build_full_oracle.py",
            "source_arm": SOURCE_ARM}
    make_pools.write_pool(path, pool, [arm], TEMPLATE, AXES_PATH, meta)


def split_task_ids(bank_ids: list[str], leg_g_ids: list[str]) -> tuple[list[str], list[str]]:
    """``(block A ids, Leg G ids)`` in bank order; asserts 30 + 110 and that Leg G is inside the bank."""
    leg_g = set(leg_g_ids)
    block_a = []
    study = []
    for task_id in bank_ids:
        if task_id in leg_g:
            study.append(task_id)
        else:
            block_a.append(task_id)
    if len(study) != len(leg_g):
        raise BuildError("some Leg G task ids are not in the bank")
    if len(block_a) != N_PILOT_TASKS:
        raise BuildError(f"expected exactly {N_PILOT_TASKS} block-A tasks, found {len(block_a)}")
    if len(study) != N_STUDY_TASKS:
        raise BuildError(f"expected exactly {N_STUDY_TASKS} Leg G tasks, found {len(study)}")
    return block_a, study


def leg_g_task_ids() -> list[str]:
    manifest = json.loads((WINNER_CHECK / "manifest.json").read_text())
    return [str(t) for t in manifest["task_ids"]]


def build_all(seed_state: dict | None = None, solvers: dict | None = None, bank: list[dict] | None = None,
              leg_g_ids: list[str] | None = None, header: str | None = None, vector: dict | None = None,
              allow: dict[str, str] | None = None) -> dict:
    """Everything both pools hold, computed without touching the disk. Raises `BuildError` on any problem."""
    if bank is None:
        bank = load_bank()
    if seed_state is None or solvers is None:
        module = load_sanity_module()
        if seed_state is None:
            seed_state = module.generate_seed_state()
        if solvers is None:
            solvers = module.SOLVERS
    if leg_g_ids is None:
        leg_g_ids = leg_g_task_ids()
    if header is None:
        header = HEADER_V2
    if vector is None:
        vector = VECTOR_V2
    bank_ids = []
    by_id = {}
    for task in bank:
        bank_ids.append(str(task["id"]))
        by_id[str(task["id"])] = task
    block_a, study = split_task_ids(bank_ids, leg_g_ids)
    changes, problems = solve_all(bank_ids, seed_state, solvers, allow)
    # only the 140 = 30 + 110 tasks matter, and the two sets partition the bank
    if len(problems) > 0:
        raise BuildError("answer key cannot be built; these tasks have no usable diff:\n  " + "\n  ".join(problems))
    all_ids = sorted(bank_ids)
    tasks = []
    for task_id in all_ids:
        tasks.append(by_id[task_id])
    guidance, entries = build_prompt(header, tasks, changes)
    pools = {POOL: {"pool": POOL, "arm_id": ARM, "task_ids": all_ids, "guidance": guidance, "entries": entries,
                    "chars": len(guidance), "approx_tokens": len(guidance) // 4,
                    "arm": make_arm(ARM, guidance, vector)}}
    return {"pools": pools, "changes": changes, "problems": problems, "block_a": block_a, "study": study}


def write_pools(built: dict, out_dir: Path) -> dict[str, Path]:
    """Write the two pool YAMLs under `out_dir` (default: scratch); returns {pool: path}."""
    paths = {}
    for pool in built["pools"]:
        info = built["pools"][pool]
        path = Path(out_dir) / f"{pool}.yaml"
        write_oracle_pool(path, pool, info["arm"], info["task_ids"])
        paths[pool] = path
    return paths


def report(built: dict) -> str:
    lines = []
    for pool in built["pools"]:
        info = built["pools"][pool]
        lines.append(f"pool {pool} (arm {info['arm_id']}): {len(info['entries'])} entries, "
                     f"{info['chars']} chars, ~{info['approx_tokens']} tokens (chars/4)")
    lines.append("empty-diff / failed tasks: none" if len(built["problems"]) == 0 else
                 "PROBLEMS: " + "; ".join(built["problems"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="also write the two pool YAMLs to --out")
    ap.add_argument("--out", type=Path, default=None, help="directory for --write (required with it)")
    ap.add_argument("--show", type=str, default=None, help="print the rendered entry of these comma-separated task ids")
    args = ap.parse_args(argv)
    try:
        built = build_all()
    except BuildError as err:
        print(f"BUILD FAILED: {err}", file=sys.stderr)
        return 2
    print(report(built))
    if args.show is not None:
        bank = load_bank()
        for task in bank:
            if str(task["id"]) in args.show.split(","):
                print(render_entry(str(task["instruction"]), built["changes"][str(task["id"])]))
    if args.write:
        if args.out is None:
            ap.error("--write needs --out DIR")
        paths = write_pools(built, args.out)
        for pool in paths:
            print(f"wrote {paths[pool]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
