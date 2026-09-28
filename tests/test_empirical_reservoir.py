"""`EmpiricalReservoir`: an exact discrete reservoir for real prompt pools."""

from __future__ import annotations

import numpy as np
import pytest

from cold_start.growing.deploy.harness import CellSpec, run_cell
from cold_start.growing.deploy.recommenders import PRIMARY_RECOMMENDER
from cold_start.growing.deploy.rules import make_policy
from cold_start.growing.empirical_reservoir import EmpiricalReservoir
from cold_start.growing.reservoirs import DegenerateReservoirError, build_reservoir
from cold_start.growing.tables import CSTable


def _three() -> EmpiricalReservoir:
    return EmpiricalReservoir([0.2, 0.5, 0.8], [0.25, 0.5, 0.25], label="three")


def test_inverse_cdf_is_the_left_continuous_step_inverse():
    res = _three()
    u = np.array([0.0, 0.25, 0.2500001, 0.75, 0.7500001, 1.0])
    assert res.sample_from_uniforms(u).tolist() == [0.2, 0.2, 0.5, 0.5, 0.8, 0.8]


def test_survival_is_the_mass_strictly_above():
    res = _three()
    x = np.array([-0.1, 0.1, 0.2, 0.49, 0.5, 0.8, 0.9])
    assert np.allclose(res._survival(x), [1.0, 1.0, 0.75, 0.75, 0.25, 0.0, 0.0])
    assert res.tail_prob(0.5) == pytest.approx(0.25)


def test_moments_and_sup():
    res = _three()
    assert res.mean() == pytest.approx(0.5)
    assert res.sd() == pytest.approx(np.sqrt(0.25 * 0.09 + 0.25 * 0.09))
    assert res.essential_sup() == pytest.approx(0.8)


def test_spec_round_trips_through_the_registry():
    res = _three()
    again = build_reservoir(res.to_spec())
    u = np.linspace(0.0, 1.0, 101)
    assert np.array_equal(again.sample_from_uniforms(u), res.sample_from_uniforms(u))
    assert again.label == "three"


def test_duplicate_and_unsorted_atoms_merge():
    res = EmpiricalReservoir([0.5, 0.2, 0.5], [0.25, 0.5, 0.25])
    assert res.atoms.tolist() == [0.2, 0.5]
    assert res.weights.tolist() == [0.5, 0.5]


def test_zero_weight_atoms_are_dropped():
    res = EmpiricalReservoir([0.1, 0.9], [0.0, 1.0])
    assert res.atoms.tolist() == [0.9]


@pytest.mark.parametrize(
    "atoms, weights",
    [([], []), ([0.5], [0.5]), ([0.5, 0.6], [1.2, -0.2]), ([1.5], [1.0]), ([0.5, 0.6], [1.0])],
)
def test_rejects_malformed_input(atoms, weights):
    with pytest.raises(ValueError):
        EmpiricalReservoir(atoms, weights)


def test_a_flat_pool_is_recorded_not_raised():
    res = EmpiricalReservoir([0.6], [1.0])
    assert res.validation_error is not None and "spread" in res.validation_error
    with pytest.raises(DegenerateReservoirError):
        EmpiricalReservoir([0.6], [1.0], validate=True)


def test_runs_through_the_harness():
    res = _three()
    spec = CellSpec(env_id="emp_test", env_spec=res.to_spec(), horizon=20, cap=20, base_seed=1,
                    n_replicates=8)
    table = CSTable.load_or_build(20, 0.05)
    policy = make_policy("fixed_K4", params={"K": 4}, horizon=20, n_replicates=8, table=table)
    out = run_cell(spec, policy, table=table, reservoir=build_reservoir(spec.env_spec))
    regret = out.regret(PRIMARY_RECOMMENDER)
    assert regret.shape == (8,)
    assert np.all(np.isfinite(regret)) and np.all(regret >= -1e-12)


def test_run_deployment_knows_the_family():
    import sys
    from pathlib import Path

    deploy = Path(__file__).resolve().parents[1] / "experiments" / "growing_bandits" / "deploy"
    sys.path.insert(0, str(deploy))
    import run_deployment as rd

    assert rd.family_of(_three().to_spec()) == "E"
