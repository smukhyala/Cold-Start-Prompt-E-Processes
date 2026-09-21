"""`analyze_capp.py`: the paired cap sweep's tables, and the gate that makes them paired."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "experiments" / "growing_bandits" / "deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import analyze_capp as ac  # noqa: E402

ENVS = [f"e{i}" for i in range(5)]
CAPS = (32, 64, 128)
POLICIES = {"always_search": 0.0, "p3_star": -0.002, "level_star": -0.006, "phi_fake": -0.006}
#: What the fixture's tuning files say each policy deploys at each cap (constructor params).
P3 = {"alpha": 0.5, "c": 3.0}
P3_128 = {"alpha": 0.5, "c": 1.2}
LEVEL = {"alpha": 0.5, "c": 2.0, "b": 2.0}


def _write(root: Path, env: str, T: int, cap: int, seed: int, n: int = 40, break_pairing: bool = False) -> None:
    rng = np.random.default_rng(seed)
    mu_star = rng.normal(0.95, 0.01, size=n) + (0.001 if break_pairing else 0.0)
    base = rng.normal(0.12, 0.02, size=n)
    cell = f"{env}_T{T}_cap{cap}"
    for policy, shift in POLICIES.items():
        rng_p = np.random.default_rng(seed + hash(policy) % 1000)
        regret = base + shift - 0.0001 * (cap - 64) / 64 + rng_p.normal(0, 0.001, size=n)
        path = root / "episodes" / "capp" / cell / f"{policy}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cell": cell, "policy": policy, "env_id": env, "family": "A", "horizon": T, "cap": cap,
            "base_seed": seed, "episode": np.arange(n), "mu_star": mu_star,
            "regret_posterior_mean_shrunk": regret, "k_final": np.full(n, float(min(cap, 40))),
            "search_frac": np.full(n, 0.3),
        }).to_parquet(path, index=False)


def _manifest(root: Path) -> None:
    """Records as the runner writes them: p3_star at cap 32 deployed the cap-64 constants (no cap-32 block
    exists), at cap 128 its own; level_star the same block everywhere; phi_k4 is not a registered policy
    of this fixture's table and carries whatever it carried."""
    lines = []
    for cell_dir in (root / "episodes" / "capp").iterdir():
        cap = int(cell_dir.name.split("_cap")[-1])
        for pq in cell_dir.glob("*.parquet"):
            if pq.stem == "p3_star":
                params = dict(P3_128) if cap == 128 else {**P3, **({"params_tuned": False, "params_cap": 64} if cap == 32 else {})}
            elif pq.stem == "level_star":
                params = dict(LEVEL)
            else:
                params = {}
            lines.append(json.dumps({"kind": "completion", "test": "capp", "cell": cell_dir.name,
                                     "policy": pq.stem, "params": params}))
    (root / "manifest_capp.jsonl").write_text("\n".join(lines) + "\n")
    json.dump({"meta": {"cap": 64}, "p3_star": {"200": P3}, "level_star": LEVEL,
               "by_cap": {"128": {"p3_star": {"200": P3_128}, "level_star": LEVEL}}},
              open(root / "baseline_params.json", "w"))


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "deploy"
    for i, env in enumerate(ENVS):
        for T in (200,):
            for cap in CAPS:
                _write(root, env, T, cap, seed=100 + i)  # same seed for every cap: paired
    _manifest(root)
    return root


def test_pairing_gate_passes_on_identical_mu_star_and_fails_otherwise(run):
    frame = ac.load_episodes(run)
    ac.assert_paired(frame)  # no raise
    _write(run, "e0", 200, 128, seed=100, break_pairing=True)
    with pytest.raises(ValueError, match="mu_star"):
        ac.assert_paired(ac.load_episodes(run))


def test_policy_table_and_within_cap_contrasts(run):
    out = ac.analyze(run, n_boot=200)
    pol = out["policies"]
    assert set(pol["policy"]) == set(POLICIES) and set(pol["cap"]) == set(CAPS)
    row = pol[(pol["policy"] == "p3_star") & (pol["cap"] == 32)].iloc[0]
    assert row["n_envs"] == 5 and not row["params_tuned"], "cap-64 constants deployed at cap 32"
    assert pol[(pol["policy"] == "p3_star") & (pol["cap"] == 64)].iloc[0]["params_tuned"]
    assert pol[(pol["policy"] == "p3_star") & (pol["cap"] == 128)].iloc[0]["params_tuned"]
    assert pol[(pol["policy"] == "level_star") & (pol["cap"] == 32)].iloc[0]["params_tuned"] == False  # noqa: E712
    assert pol[(pol["policy"] == "level_star") & (pol["cap"] == 128)].iloc[0]["params_tuned"]
    con = out["contrasts"]
    assert set(con["policy"]).isdisjoint({"phi_fake"}), "only registered pairs are computed"
    pair = con[(con["policy"] == "level_star") & (con["reference"] == "p3_star") & (con["cap"] == 64)].iloc[0]
    assert pair["delta"] == pytest.approx(-0.004, abs=0.001)
    assert pair["n_envs"] == 5 and pair["cluster_method"] == "env_mean_t"
    pair = con[(con["policy"] == "level_star") & (con["reference"] == "fixed_K_star")]
    assert pair.empty, "a pair whose reference was not deployed is skipped, not fabricated"


def test_cross_cap_rows_are_paired_against_cap_64(run):
    out = ac.analyze(run, n_boot=200)
    x = out["crosscap"]
    assert set(x["cap"]) == {32, 128}, "cap 64 is the reference, not a row"
    row = x[(x["policy"] == "always_search") & (x["cap"] == 128)].iloc[0]
    # regret falls by 0.0001 per 64 arms of cap in the fixture: paired, that is resolvable.
    assert row["delta"] == pytest.approx(-0.0001, abs=3e-4)
    assert row["hi"] - row["lo"] < 0.002 and row["n_envs"] == 5
