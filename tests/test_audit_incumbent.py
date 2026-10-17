import numpy as np

from experiments.growing_bandits.audit.incumbent_gain import held_out_gains


def test_identical_arms_give_zero_gain():
    incumbent = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    arms = np.vstack([incumbent, incumbent])
    gains = held_out_gains(arms, incumbent, n_splits=50, seed=0)
    assert np.allclose(gains, 0.0)


def test_dominant_arm_gives_its_margin():
    incumbent = np.zeros(10)
    arms = np.vstack([np.ones(10), np.zeros(10)])
    gains = held_out_gains(arms, incumbent, n_splits=50, seed=0)
    assert np.allclose(gains, 1.0)
