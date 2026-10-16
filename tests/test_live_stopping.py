"""Synthetic, no-provider correctness gates for genuinely live scheduling."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from experiments.growing_bandits.audit.flatness_eprocess import e1b_log_paths
from experiments.growing_bandits.audit.futility_eprocess import futility_log_paths
from experiments.growing_bandits.empirical import live_stopping as live


def configuration(n=120, **changes):
    prompts = {role: {"text": role, "sha256": hashlib.sha256(role.encode()).hexdigest()}
               for role in live.ROLES}
    config = {"max_pairs": n, "model": "frozen-model-20260101", "prompts": prompts,
              "max_usd": 100.0, "episode_cost_bound_usd": 0.1}
    config.update(changes)
    return config


def tasks(n):
    return [{"task_id": f"sample_{i}", "seed": i, "index": i} for i in range(n)]


def result(config, role, success, **changes):
    value = {"status": "ok", "success": success, "cost_usd": 0.01,
             "input_tokens": 10, "output_tokens": 3, "model": config["model"],
             "prompt_sha256": config["prompts"][role]["sha256"]}
    value.update(changes)
    return value


def evaluator(config, outcomes, calls):
    def evaluate(task, role, episode_id):
        calls.append((task["index"], role, episode_id))
        return result(config, role, int(outcomes[task["index"]][live.ROLES.index(role)]))
    return evaluate


def offline(outcomes, config):
    values = np.array(outcomes, dtype=np.int8).T
    efficacy = e1b_log_paths(values[None, ...], np.array([0]), np.array([[False, True]]))[0]
    futility = futility_log_paths((values[1] - values[0])[None, None, :],
                                 config.get("delta", 0.15))[0, 0]
    hits = np.flatnonzero((efficacy >= math.log(1 / config.get("alpha_e", 0.025)))
                         | (futility >= math.log(1 / config.get("alpha_f", 0.025))))
    return efficacy, futility, int(hits[0] + 1) if len(hits) else len(outcomes)


@pytest.mark.parametrize("kind,expected", [
    ("strong_positive", "efficacy"),
    ("substantially_worse", "futility"),
    ("near_equal", "futility"),
    ("small_positive", "futility"),
    ("no_crossing", "unresolved"),
])
def test_synthetic_outcomes_stop_at_first_real_boundary(tmp_path, kind, expected):
    n = 10 if kind == "no_crossing" else 120
    config = configuration(n)
    if kind == "strong_positive":
        outcomes = [(0, 1)] * n
    elif kind == "substantially_worse":
        outcomes = [(1, 0)] * n
    elif kind == "near_equal":
        outcomes = [(1, 1) if i % 2 else (0, 0) for i in range(n)]
    elif kind == "small_positive":
        outcomes = [(0, 1) if i % 20 == 19 else (1, 1) for i in range(n)]
    else:
        outcomes = [(1, 1)] * n
    calls = []
    state = live.run(config, tmp_path, tasks(n), evaluator(config, outcomes, calls))
    e, f, first = offline(outcomes, config)
    assert state["decision"] == expected
    assert state["completed_pairs"] == first
    assert len(calls) == 2 * first
    assert [role for _, role, _ in calls] == list(live.ROLES) * first
    saved = json.loads((tmp_path / "evidence.json").read_text())
    # Bit-for-bit equality, not merely numerical closeness, with full offline prefix.
    np.testing.assert_array_equal([r["log_efficacy"] for r in saved], e[:first])
    np.testing.assert_array_equal([r["log_futility"] for r in saved], f[:first])
    assert state["known_cost_usd"] == pytest.approx(0.02 * first)
    assert state["input_tokens"] == 20 * first
    assert state["output_tokens"] == 6 * first
    assert state["unscheduled_planned_episodes"] == 2 * (n - first)
    before = list(calls)
    resumed = live.run(config, tmp_path, tasks(n), evaluator(config, outcomes, calls))
    assert calls == before
    assert resumed["terminal"] == state["terminal"]


def test_online_prefix_evidence_exactly_matches_both_existing_offline_functions():
    rng = np.random.default_rng(410)
    config = configuration()
    outcomes = rng.integers(0, 2, size=(120, 2))
    e, f, _ = offline(outcomes, config)
    pairs = []
    for i, (baseline, challenger) in enumerate(outcomes):
        pairs.append({"baseline_success": int(baseline), "challenger_success": int(challenger)})
        path = live.evidence_paths(pairs, {**live.DEFAULTS, **config})
        np.testing.assert_array_equal([p["log_efficacy"] for p in path], e[:i + 1])
        np.testing.assert_array_equal([p["log_futility"] for p in path], f[:i + 1])


def test_zero_discordances_leave_efficacy_at_one(tmp_path):
    config = configuration()
    calls = []
    state = live.run(config, tmp_path, tasks(120), evaluator(config, [(1, 1)] * 120, calls))
    assert state["decision"] == "futility"
    assert state["last_evidence"]["log_efficacy"] == 0.0
    assert all(r["log_efficacy"] == 0.0 for r in json.loads((tmp_path / "evidence.json").read_text()))


def test_actual_boundaries_one_pair_apart_choose_first_without_extra_calls(tmp_path):
    # This deterministic synthetic sequence has FUTILITY at 145 and EFFICACY at 146.
    n = 180
    outcomes = [(1, 1)] * 3 + [(0, 1) if i % 13 == 12 else (1, 1) for i in range(n - 3)]
    config = configuration(n)
    e, f, _ = offline(outcomes, config)
    assert np.flatnonzero(e >= math.log(40))[0] + 1 == 146
    assert np.flatnonzero(f >= math.log(40))[0] + 1 == 145
    calls = []
    state = live.run(config, tmp_path, tasks(n), evaluator(config, outcomes, calls))
    assert state["decision"] == "futility"
    assert state["completed_pairs"] == 145
    assert len(calls) == 290
    assert state["terminal"]["first_crossing"]["pair_index"] == 145


def test_declared_precedence_if_both_flags_set(tmp_path, monkeypatch):
    # Unit test the dispatch rule separately from the statistical implementations.
    def simultaneous(pairs, config):
        return [] if not pairs else [{"pair_index": 1, "log_efficacy": 4.0,
                                      "log_futility": 4.0, "efficacy_crossed": True,
                                      "futility_crossed": True}]
    monkeypatch.setattr(live, "evidence_paths", simultaneous)
    config = configuration(2)
    calls = []
    state = live.run(config, tmp_path, tasks(2), evaluator(config, [(0, 1)] * 2, calls))
    assert state["decision"] == "efficacy"
    assert state["terminal"]["first_crossing"]["futility_crossed"]
    assert len(calls) == 2


@pytest.mark.parametrize("status", ["provider_error", "infrastructure_error", "budget_exhausted"])
def test_error_during_challenger_never_becomes_failure_or_retries(tmp_path, status):
    config = configuration(5)
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(role)
        return result(config, role, 1 if role == "baseline" else None,
                      status="ok" if role == "baseline" else status)

    state = live.run(config, tmp_path, tasks(5), evaluate)
    assert state["decision"] == "unresolved"
    assert state["terminal"]["reason"] == status
    assert state["completed_pairs"] == 0
    assert state["valid_outcome_episodes"] == 1
    assert state["known_cost_usd"] == 0.02
    assert json.loads((tmp_path / "pairs.json").read_text()) == []
    live.run(config, tmp_path, tasks(5), evaluate)
    assert calls == ["baseline", "challenger"]


def test_interruption_after_baseline_receipt_resumes_without_duplicate(tmp_path, monkeypatch):
    config = configuration(4)
    calls = []
    evaluate = evaluator(config, [(1, 1)] * 4, calls)
    real_write = live._atomic_json

    def interrupt_after_receipt(path, value):
        real_write(path, value)
        if path.parent.name == "receipts":
            raise KeyboardInterrupt

    monkeypatch.setattr(live, "_atomic_json", interrupt_after_receipt)
    with pytest.raises(KeyboardInterrupt):
        live.run(config, tmp_path, tasks(4), evaluate)
    assert len(calls) == 1
    monkeypatch.setattr(live, "_atomic_json", real_write)
    state = live.run(config, tmp_path, tasks(4), evaluate)
    assert state["completed_pairs"] == 4
    assert len(calls) == 8
    assert len({episode_id for _, _, episode_id in calls}) == 8


def test_interruption_after_crossing_receipt_recovers_decision_before_scheduling(tmp_path, monkeypatch):
    config = configuration()
    calls = []
    evaluate = evaluator(config, [(0, 1)] * 120, calls)
    real_write = live._atomic_json

    def interrupt_at_crossing(path, value):
        real_write(path, value)
        if path.parent.name == "receipts" and path.stem == "pair_000011_challenger":
            raise KeyboardInterrupt

    monkeypatch.setattr(live, "_atomic_json", interrupt_at_crossing)
    with pytest.raises(KeyboardInterrupt):
        live.run(config, tmp_path, tasks(120), evaluate)
    assert len(calls) == 22
    monkeypatch.setattr(live, "_atomic_json", real_write)
    state = live.run(config, tmp_path, tasks(120), evaluate)
    assert state["decision"] == "efficacy"
    assert state["terminal"]["recovered_before_scheduling"]
    assert len(calls) == 22


def test_ambiguous_paid_request_is_never_repeated(tmp_path):
    config = configuration(3)
    calls = []

    def interrupt(task, role, episode_id):
        calls.append(episode_id)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        live.run(config, tmp_path, tasks(3), interrupt)
    state = live.run(config, tmp_path, tasks(3), interrupt)
    assert len(calls) == 1
    assert state["terminal"]["reason"] == "ambiguous_inflight_episode"
    assert not state["cost_accounting_complete"]
    assert state["ambiguous_episode_ids"] == ["pair_000001_baseline"]


def test_durable_intent_and_registration_precede_callback(tmp_path):
    config = configuration(1)

    def evaluate(task, role, episode_id):
        registration = json.loads((tmp_path / "registration.json").read_text())
        intent = json.loads((tmp_path / "intents" / f"{episode_id}.json").read_text())
        assert intent["frozen_sha256"] == registration["frozen_sha256"]
        assert intent["task_sha256"] == live.canonical_sha256(task)
        assert not (tmp_path / "receipts" / f"{episode_id}.json").exists()
        return result(config, role, 0)

    assert live.run(config, tmp_path, tasks(1), evaluate)["completed_pairs"] == 1


def test_dollar_budget_reserves_whole_pair_before_baseline(tmp_path):
    config = configuration(10, max_usd=0.21)
    calls = []
    state = live.run(config, tmp_path, tasks(10), evaluator(config, [(1, 1)] * 10, calls))
    assert state["completed_pairs"] == 1
    assert state["known_cost_usd"] == 0.02
    assert state["terminal"]["reason"] == "usd_budget"
    assert len(calls) == 2


def test_insufficient_budget_runs_zero_episodes(tmp_path):
    config = configuration(2, max_usd=0.19)
    state = live.run(config, tmp_path, tasks(2), lambda *args: pytest.fail("must not execute"))
    assert state["scheduled_episodes"] == 0
    assert state["terminal"]["reason"] == "usd_budget"


@pytest.mark.parametrize("changes,reason", [
    ({"success": None}, "missing_or_invalid_binary_outcome"),
    ({"cost_usd": None}, "missing_or_invalid_cost_accounting"),
    ({"cost_usd": -1}, "missing_or_invalid_cost_accounting"),
    ({"input_tokens": None}, "missing_or_invalid_token_accounting"),
    ({"model": "moving-alias"}, "model_identifier_mismatch"),
    ({"prompt_sha256": "changed"}, "prompt_hash_mismatch"),
    ({"cost_usd": 0.11}, "episode_cost_bound_exceeded"),
    ({"status": "invented"}, "invalid_episode_status"),
])
def test_invalid_episode_halts_before_challenger(tmp_path, changes, reason):
    config = configuration(3)
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(role)
        value = result(config, role, 1)
        value.update(changes)
        return value

    state = live.run(config, tmp_path, tasks(3), evaluate)
    assert calls == ["baseline"]
    assert state["terminal"]["reason"] == reason
    assert state["completed_pairs"] == 0


def test_unhandled_callback_exception_preserves_unknown_charges(tmp_path):
    config = configuration(3)

    def fail(*args):
        raise RuntimeError("provider transport interrupted")

    state = live.run(config, tmp_path, tasks(3), fail)
    assert state["decision"] == "unresolved"
    assert state["terminal"]["reason"] == "infrastructure_error"
    assert not state["cost_accounting_complete"]
    assert state["recorded_episodes"] == 1


@pytest.mark.parametrize("uncertainty", [
    {"cost_accounting_complete": False},
    {"unknown_billing_requests": ["provider-request-unknown-charge"]},
])
def test_known_subtotal_does_not_misrepresent_complete_accounting(tmp_path, uncertainty):
    config = configuration(3)
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(role)
        return result(config, role, 1, **uncertainty)

    state = live.run(config, tmp_path, tasks(3), evaluate)
    assert calls == ["baseline"]
    assert state["terminal"]["reason"] == "incomplete_cost_accounting"
    assert not state["cost_accounting_complete"]
    assert not state["token_accounting_complete"]
    assert state["known_cost_usd"] == 0.01
    assert state["completed_pairs"] == 0


def test_transport_budget_mode_allows_multiple_requests_per_episode(tmp_path):
    config = configuration(2, budget_mode="transport", minimum_request_reservation_usd=0.01)
    config.pop("episode_cost_bound_usd")
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(role)
        return result(config, role, 1, cost_usd=1.0)

    state = live.run(config, tmp_path, tasks(2), evaluate)
    assert state["completed_pairs"] == 2
    assert state["known_cost_usd"] == 4.0
    assert state["terminal"]["reason"] == "max_pairs"


def test_transport_reports_budget_exhausted_midpair(tmp_path):
    config = configuration(5, budget_mode="transport", minimum_request_reservation_usd=0.01,
                           max_usd=0.2)
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(role)
        return result(config, role, 1 if role == "baseline" else None, cost_usd=0.08,
                      status="ok" if role == "baseline" else "budget_exhausted")

    state = live.run(config, tmp_path, tasks(5), evaluate)
    assert calls == ["baseline", "challenger"]
    assert state["terminal"]["reason"] == "budget_exhausted"
    assert state["completed_pairs"] == 0
    assert state["known_cost_usd"] == 0.16


def test_changed_config_or_task_manifest_is_rejected_before_calls(tmp_path):
    config = configuration(2, max_usd=0.0)
    live.run(config, tmp_path, tasks(2), lambda *args: pytest.fail("must not execute"))
    changed = copy.deepcopy(config)
    changed["delta"] = 0.16
    with pytest.raises(ValueError, match="Frozen configuration"):
        live.run(changed, tmp_path, tasks(2), lambda *args: pytest.fail("must not execute"))
    changed_tasks = tasks(2)
    changed_tasks[0]["seed"] = 999
    with pytest.raises(ValueError, match="Frozen configuration"):
        live.run(config, tmp_path, changed_tasks, lambda *args: pytest.fail("must not execute"))


def test_frozen_file_hash_checked_before_each_episode(tmp_path):
    source = tmp_path / "frozen.py"
    source.write_text("original")
    config = configuration(2, frozen_files={str(source): live.file_sha256(source)})
    calls = []

    def evaluate(task, role, episode_id):
        calls.append(role)
        source.write_text("changed")
        return result(config, role, 1)

    with pytest.raises(ValueError, match="Frozen file changed"):
        live.run(config, tmp_path / "run", tasks(2), evaluate)
    assert calls == ["baseline"]


def test_concurrent_controller_is_rejected(tmp_path):
    config = configuration(2)
    with live._lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another live controller"):
            live.run(config, tmp_path, tasks(2), lambda *args: pytest.fail("must not execute"))
