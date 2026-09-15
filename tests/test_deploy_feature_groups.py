from __future__ import annotations

import glob
from pathlib import Path

import pytest

from cold_start.growing.deploy import feature_groups as fg

ROOT = Path(__file__).resolve().parents[1]


def test_groups_are_disjoint_and_cover_71():
    union = fg.CLOCK + fg.QUALITY + fg.EVIDENCE + fg.HISTORY
    assert len(union) == 71
    assert len(set(union)) == 71
    assert set(fg.CLOCK_SF) <= set(fg.CLOCK)
    assert set(fg.EVIDENCE_LOGE) <= set(fg.EVIDENCE)


def test_partition_matches_corpus_if_present():
    shards = sorted(glob.glob(str(ROOT / "data" / "oracle_labels" / "part-*.parquet")))
    if not shards:
        pytest.skip("oracle corpus not present")
    import pyarrow.parquet as pq

    cols = pq.read_schema(shards[0]).names
    fg.check_partition(cols)


def test_assert_deployable_rejects_forbidden():
    with pytest.raises(ValueError):
        fg.assert_deployable(["f_t", "oracle_mu_star"])
    with pytest.raises(ValueError):
        fg.assert_deployable(["f_t", "not_a_column"])
    fg.assert_deployable(list(fg.FEATURE_SETS["clock_quality_evidence"]))
