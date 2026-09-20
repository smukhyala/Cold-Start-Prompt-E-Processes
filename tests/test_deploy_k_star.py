"""Behaviour tests for the K* envelope (NEXT-STEPS 2.5).

Real simulator runs on one environment at T=50 with a three-point K grid and M=16, so
the whole test is well under a second. What is asserted is the contract the analysis
will read: one row per (env, T, K) that was actually run, a pooled row per (T, K)
averaging the environment rows, and exactly one `is_argmin` per (level, env, T).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import cells  # noqa: E402
import k_star_envelope as ks  # noqa: E402

ENV = "beta_good_common"


@pytest.fixture(scope="module")
def frame():
    return ks.run_envelope(
        env_ids=(ENV,), horizons=(50,), k_grid=(2, 8, 64), n_replicates=16, split="tune", workers=1
    )


def test_one_row_per_reachable_env_horizon_k(frame):
    env_rows = frame[frame["level"] == "env"]
    # K=64 exceeds T=50 and cannot be reached, so it is not run.
    assert sorted(env_rows["K"].tolist()) == [2, 8]
    assert set(env_rows["env_id"]) == {ENV}
    assert (env_rows["horizon"] == 50).all()
    assert (env_rows["n_episodes"] == 16).all()
    assert (env_rows["cap"] == 50).all(), "the envelope is measured uncapped (cap = T)"
    assert (env_rows["split"] == "tune").all()


def test_env_rows_carry_the_regret_decomposition(frame):
    env_rows = frame[frame["level"] == "env"]
    for col in ("regret", "regret_disc", "regret_sel", "k_final", "search_frac"):
        assert np.isfinite(env_rows[col]).all(), col
    np.testing.assert_allclose(
        env_rows["regret"], env_rows["regret_disc"] + env_rows["regret_sel"], atol=1e-12
    )


def test_fixed_k_reaches_its_k(frame):
    """`fixed_K` is `PowerSchedule(alpha=0, c=K)`: at T=50 uncapped it holds K arms at T."""
    env_rows = frame[frame["level"] == "env"].set_index("K")
    assert env_rows.loc[2, "k_final"] == pytest.approx(2.0)
    assert env_rows.loc[8, "k_final"] == pytest.approx(8.0)


def test_pooled_rows_average_the_env_rows(frame):
    pooled = frame[frame["level"] == "pooled"]
    env_rows = frame[frame["level"] == "env"]
    assert len(pooled) == 2
    assert (pooled["env_id"] == "all").all()
    for _, row in pooled.iterrows():
        sub = env_rows[env_rows["K"] == row["K"]]
        assert row["regret"] == pytest.approx(sub["regret"].mean())
        assert row["n_envs"] == len(sub)


def test_exactly_one_argmin_per_env_and_horizon(frame):
    for (level, env, T), sub in frame.groupby(["level", "env_id", "horizon"]):
        assert int(sub["is_argmin"].sum()) == 1, (level, env, T)
        assert sub.loc[sub["is_argmin"], "regret"].item() == sub["regret"].min()
        assert (sub["k_star"] == sub.loc[sub["is_argmin"], "K"].item()).all()


def test_base_seed_is_the_tune_split_cell_seed(frame):
    """The envelope reuses `cells.make_cell` so its seeds are disjoint from the test split."""
    env_rows = frame[frame["level"] == "env"]
    assert (env_rows["base_seed"] == cells.base_seed("tune", ENV, 50, "T")).all()


def test_cli_writes_the_csv(tmp_path):
    out = tmp_path / "k_star_envelope.csv"
    ks.main(
        [
            "--envs", ENV, "--horizons", "50", "--k-grid", "2,8", "--replicates", "8",
            "--workers", "1", "--out", str(out),
        ]
    )
    text = out.read_text().splitlines()
    assert text[0].startswith("level,split,env_id,family,horizon,cap,base_seed,K,")
    assert len(text) == 1 + 2 + 2  # header + 2 env rows + 2 pooled rows
