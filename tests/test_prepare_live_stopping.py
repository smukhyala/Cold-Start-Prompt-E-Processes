"""Outcome-free checks of the frozen task population and sampling procedure."""

from copy import deepcopy

import numpy as np

from experiments.growing_bandits.empirical.prepare_live_stopping import draw_tasks


def _bank(n=120):
    return [{"id": f"source_{i:03d}", "instruction": f"Perform task {i}",
             "verify": f"verifiers/task_{i}.py", "difficulty": "synthetic"}
            for i in range(n)]


def test_draws_are_reproducible_independent_of_source_file_order():
    bank = _bank()
    original = deepcopy(bank)
    seed = 0x1234567890ABCDEF1234567890ABCDEF
    tasks = draw_tasks(bank, seed)
    assert tasks == draw_tasks(list(reversed(bank)), seed)
    assert bank == original
    assert len(tasks) == 120
    assert [task["id"] for task in tasks] == [f"live_{i:03d}" for i in range(1, 121)]
    assert [task["task_id"] for task in tasks] == [task["id"] for task in tasks]
    assert [task["draw_index"] for task in tasks] == list(range(1, 121))
    assert {task["seed_hex"] for task in tasks} == {f"{seed:032x}"}
    assert tasks != draw_tasks(bank, seed + 1)


def test_draws_preserve_full_source_instructions_and_verifiers():
    bank = _bank()
    by_id = {task["id"]: task for task in bank}
    tasks = draw_tasks(bank, 20261016)
    for task in tasks:
        source = by_id[task["source_task_id"]]
        for key, value in source.items():
            if key != "id":
                assert task[key] == value


def test_sampling_is_with_replacement_and_retains_each_duplicate():
    bank = _bank(2)
    tasks = draw_tasks(bank, 17, n=120)
    sources = [task["source_task_id"] for task in tasks]
    assert len(set(sources)) == 2
    assert len(set(task["id"] for task in tasks)) == 120
    assert any(first == second for first, second in zip(sources, sources[1:], strict=False))
    expected = np.random.Generator(np.random.PCG64(17)).integers(0, 2, size=120)
    assert sources == [bank[int(index)]["id"] for index in expected]


def test_sampler_support_includes_every_source_task_without_filtering():
    bank = _bank()
    # A long synthetic stream checks support without selecting the study seed
    # or obtaining any outcome from the real population.
    tasks = draw_tasks(bank, 91, n=12000)
    assert {task["source_task_id"] for task in tasks} == {task["id"] for task in bank}
