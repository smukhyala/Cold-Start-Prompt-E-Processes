from __future__ import annotations

from pathlib import Path

import pytest

from cold_start.analysis.summary import distance_matrix
from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def axes():
    return load_axes(_PROJECT_ROOT / "configs" / "axes.yaml")


@pytest.fixture
def arms(axes):
    return load_arms(_PROJECT_ROOT / "configs" / "arms_initial.yaml", axes)


def test_distance_matrix_is_square_and_zero_diagonal(arms, axes):
    m = distance_matrix(arms, axes)
    assert m.shape[0] == m.shape[1] == len(arms)
    for aid in m.columns:
        assert m.loc[aid, aid] == 0.0


def test_distance_matrix_is_symmetric(arms, axes):
    m = distance_matrix(arms, axes)
    for i in m.columns:
        for j in m.columns:
            assert abs(m.loc[i, j] - m.loc[j, i]) < 1e-12


def test_distance_matrix_satisfies_triangle_inequality(arms, axes):
    """Weighted L1 distance is a metric ⇒ d(a,c) ≤ d(a,b) + d(b,c)."""
    m = distance_matrix(arms, axes)
    ids = list(m.columns)
    for a in ids:
        for b in ids:
            for c in ids:
                assert m.loc[a, c] <= m.loc[a, b] + m.loc[b, c] + 1e-9


def test_distance_matrix_is_nonnegative(arms, axes):
    m = distance_matrix(arms, axes)
    assert (m.values >= 0).all()


def test_distance_matrix_distinct_arms_are_distant(arms, axes):
    """No two arms in the default catalog should collapse to the same vector."""
    m = distance_matrix(arms, axes)
    ids = list(m.columns)
    off_diagonal_min = min(
        m.loc[i, j] for i in ids for j in ids if i != j
    )
    assert off_diagonal_min > 0.0, "two arms occupy the same vector — catalog has duplicates"
