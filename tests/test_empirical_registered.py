"""Pre-registration 9's decision rule: the prompt-bootstrap interval behind the flatness guard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "growing_bandits" / "deploy"))

import registered_contrast as rc  # noqa: E402

COL = "regret_posterior_mean_shrunk"


def _cell(root, test, env, T, seed, deltas, n=20):
    base = np.full(n, 0.1)
    for policy, shift in deltas.items():
        path = root / "episodes" / test / f"{env}_T{T}_cap{T}" / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"env_id": env, "horizon": T, "cap": T, "base_seed": seed, "episode": np.arange(n),
                      COL: base + shift}).to_parquet(path, index=False)


def _tree(root, point_delta, boot_deltas, informative, extra_cell_delta=None):
    flat = []
    for pool in ("G", "F"):
        for T in (50, 100, 200):
            env = f"emp_{pool}_npmle"
            inf = (pool, T) in informative
            flat.append({"env_id": env, "pool": pool, "variant": "npmle", "horizon": T,
                         "regret_range": 0.05 if inf else 0.001, "informative": inf})
            d = point_delta if inf or extra_cell_delta is None else extra_cell_delta
            _cell(root, "emp", env, T, 1, {"p3_star": d, "fixed_K_star": 0.0})
            for b, bd in enumerate(boot_deltas):
                _cell(root, "emp_boot", f"{env}_b{b:03d}", T, 100 + b, {"p3_star": bd, "fixed_K_star": 0.0})
    (root / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(root / "tables" / "emp_flatness.csv", index=False)


def _run(root):
    return rc.prompt_bootstrap_contrast("p3_star", "fixed_K_star", horizons=(50, 100, 200), mei=0.002,
                                        rule="noninferiority", out_dir=root).iloc[0]


def test_registrations_are_written_down():
    for name in ("emp_primary", "emp_level", "emp_phi"):
        reg = rc.REGISTRATIONS[name]
        assert reg["test"] == "emp" and reg["horizons"] == (50, 100, 200) and reg["mei"] == 0.002
        assert reg["interval"] == "prompt_bootstrap"
    assert rc.REGISTRATIONS["emp_primary"]["rule"] == "noninferiority"
    assert rc.REGISTRATIONS["emp_level"]["rule"] == "not_better"


def test_uninformative_when_fewer_than_two_cells_move(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 5, informative={("F", 200)})
    row = _run(tmp_path)
    assert row["verdict"] == "uninformative" and row["n_informative"] == 1


def test_supported_when_the_bootstrap_interval_sits_below_mei(tmp_path):
    boots = list(np.linspace(-0.001, 0.001, 40))
    _tree(tmp_path, 0.0, boots, informative={("F", 100), ("F", 200)})
    row = _run(tmp_path)
    assert row["verdict"] == "supported"
    assert row["lo"] == pytest.approx(np.percentile(boots, 2.5))
    assert row["hi"] == pytest.approx(np.percentile(boots, 97.5))
    assert row["n_boot"] == 40


def test_refuted_when_the_interval_sits_above_mei(tmp_path):
    _tree(tmp_path, 0.01, list(np.linspace(0.008, 0.012, 40)), informative={("G", 50), ("F", 50)})
    assert _run(tmp_path)["verdict"] == "refuted"


def test_only_informative_cells_enter_the_point_estimate(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 10, informative={("F", 100), ("F", 200)}, extra_cell_delta=0.5)
    assert _run(tmp_path)["delta"] == pytest.approx(0.0)


def test_a_missing_bootstrap_cell_is_an_error(tmp_path):
    _tree(tmp_path, 0.0, [0.0] * 3, informative={("F", 100), ("F", 200)})
    victim = tmp_path / "episodes" / "emp_boot" / "emp_F_npmle_b001_T200_cap200"
    for f in victim.iterdir():
        f.unlink()
    victim.rmdir()
    with pytest.raises(ValueError, match="b001"):
        _run(tmp_path)
