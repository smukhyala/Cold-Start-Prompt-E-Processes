from __future__ import annotations

from cold_start.prompts.axes import load_axes
from cold_start.prompts.vector import distance
from cold_start.types import PromptVector


def _v(**kw):
    defaults = {"planning": 0, "verification": 0, "agency": 0, "expertise": 0, "format": 0, "goal": 0}
    defaults.update(kw)
    return PromptVector(**defaults)


def test_vector_id_stable_and_deterministic():
    v1 = _v(planning=3, expertise=2)
    v2 = _v(planning=3, expertise=2)
    assert v1.id() == v2.id()
    assert len(v1.id()) == 8


def test_distance_identity(axes_path):
    axes = load_axes(axes_path)
    v = _v(planning=2, expertise=3)
    assert distance(v, v, axes) == 0.0


def test_distance_symmetry(axes_path):
    axes = load_axes(axes_path)
    a = _v(planning=0, verification=1, expertise=2)
    b = _v(planning=3, verification=3, expertise=0)
    assert abs(distance(a, b, axes) - distance(b, a, axes)) < 1e-12


def test_distance_weighted_manhattan_hand(axes_path):
    axes = load_axes(axes_path)
    # all weights 1, maxes: planning=3, verification=3, agency=2, expertise=3, format=3, goal=2
    a = _v(planning=0, verification=0, agency=0, expertise=0, format=0, goal=0)
    b = _v(planning=3, verification=0, agency=2, expertise=0, format=0, goal=0)
    # delta = 3/3 + 0/3 + 2/2 + 0 + 0 + 0 = 2.0
    assert abs(distance(a, b, axes) - 2.0) < 1e-12
