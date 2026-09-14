"""Counter-based common random numbers for paired Monte Carlo rollouts.

The oracle label compares two branches (forced SEARCH vs forced REFINE) that
diverge in *which* arm they pull. To couple them we need the reward for a given
(replicate, arm, pull-index) to be the same draw in both branches, addressable in
O(1) with no stored tape. A counter-based hash gives exactly that.

Two rules that are easy to get wrong and silent when you do:

1. Key rewards by the arm's **reservoir uid**, never its slot index. The j-th new
   arm lands in a different slot in the SEARCH branch than in REFINE, so slot-keyed
   streams misalign and the pairing evaporates with no visible symptom.
2. numpy 2.x promotes ``uint64 + np.int64`` to **float64** silently. Every constant
   here is ``np.uint64`` and every index array is cast explicitly; `assert_uint64`
   is called in the hot paths so a promotion fails loudly instead of quietly
   destroying reproducibility.
"""

from __future__ import annotations

import numpy as np

# splitmix64 constants (Steele et al.), as uint64 so numpy never promotes.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)

# Distinct odd multipliers so different coordinates cannot alias.
_P_REPLICATE = np.uint64(0xD6E8FEB86659FD93)
_P_UID = np.uint64(0xA24BAED4963EE407)
_P_DOMAIN = np.uint64(0x9FB21C651E98DF25)

# 2**53: uniforms are compared as integers to avoid an int->float pass and to make
# the comparison exactly reproducible across platforms.
UNIFORM_SCALE = np.uint64(1 << 53)


def assert_uint64(*arrays: np.ndarray) -> None:
    """Raise if any array is not uint64 (guards numpy's silent float64 promotion)."""
    for i, a in enumerate(arrays):
        if np.asarray(a).dtype != np.uint64:
            raise TypeError(
                f"argument {i} has dtype {np.asarray(a).dtype}, expected uint64; "
                "mixing uint64 with np.int64 silently promotes to float64 in numpy 2.x"
            )


def splitmix64(z: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 finalizer. `z` must be uint64; wrapping is intended."""
    assert_uint64(z)
    with np.errstate(over="ignore"):
        z = z + _GOLDEN
        z = (z ^ (z >> _S30)) * _MIX1
        z = (z ^ (z >> _S27)) * _MIX2
        return z ^ (z >> _S31)


def arm_key(base_seed: int, replicate: np.ndarray, uid: np.ndarray, domain: int = 0) -> np.ndarray:
    """Per-(replicate, arm-uid) stream key, computed once when an arm is created.

    Folding the seed in here keeps the per-pull cost to one add plus the finalizer.
    `domain` separates independent uses of the same coordinates (rewards, tiebreaks,
    reservoir draws) so they cannot correlate.
    """
    rep = np.asarray(replicate).astype(np.uint64, copy=False)
    u = np.asarray(uid).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        mixed = (
            np.uint64(base_seed)
            ^ (rep * _P_REPLICATE)
            ^ (u * _P_UID)
            ^ (np.uint64(domain) * _P_DOMAIN)
        )
    return splitmix64(mixed)


def uniform_u53(key: np.ndarray, counter: np.ndarray) -> np.ndarray:
    """Uniform draw in [0, 2**53) for stream `key` at index `counter`."""
    assert_uint64(key, counter)
    with np.errstate(over="ignore"):
        return splitmix64(key + counter * _GOLDEN) >> _S11


def bernoulli(key: np.ndarray, counter: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Bernoulli draw: True iff the u53 uniform is below `threshold` = mu * 2**53.

    Integer comparison, so the result is bit-identical across platforms and does not
    depend on float rounding of `mu`.
    """
    return uniform_u53(key, counter) < threshold


def mu_to_threshold(mu: np.ndarray) -> np.ndarray:
    """Convert true means in [0,1] to uint64 thresholds for `bernoulli`."""
    m = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)
    return (m * float(UNIFORM_SCALE)).astype(np.uint64)


def reservoir_uniforms(base_seed: int, replicate: np.ndarray, draw_index: np.ndarray) -> np.ndarray:
    """Uniforms for the `draw_index`-th reservoir draw in each replicate, in [0,1).

    Both branches consume the same indexed stream, so the j-th new arm a branch
    discovers is the same underlying draw in either branch.
    """
    rep = np.asarray(replicate).astype(np.uint64, copy=False)
    j = np.asarray(draw_index).astype(np.uint64, copy=False)
    key = arm_key(base_seed, rep, j, domain=1)
    return uniform_u53(key, np.zeros_like(key)).astype(np.float64) / float(UNIFORM_SCALE)


def tiebreak_epsilon(
    base_seed: int, replicate: np.ndarray, uid: np.ndarray, scale: float = 1e-6
) -> np.ndarray:
    """Fixed per-(replicate, uid) jitter added to scores before argmax.

    `np.argmax` breaks ties at the lowest index and a freshly searched arm always
    occupies the highest slot, so without this it loses every tie -- a systematic
    downward bias in the oracle advantage, worst early in a rollout when many arms
    share (n, S). Generated from the uid, so it is identical in both branches and
    the CRN coupling survives.
    """
    key = arm_key(base_seed, replicate, uid, domain=2)
    u = uniform_u53(key, np.zeros_like(key)).astype(np.float64) / float(UNIFORM_SCALE)
    return (u * scale).astype(np.float32)
