"""Cached pairwise-evidence lookup for `f_log_e_pair` at deployment.

The corpus computed `f_log_e_pair` with the exact `PairwiseEvidence()` (512 intervals,
uniform prior), one snapshot at a time. Inside a batched rollout that is called at
every step for every replicate, and the exact cover costs ~5 ms per pair -- far too
slow. `PairwiseEvidence.tabulate` precomputes the same cover terms once, in float32,
after which a query is two gathers and a min. This module owns the on-disk cache for
that table and the fallback used when it would not fit.

Two properties are deliberate:

* **`.npy` + `mmap_mode="r"`, written atomically**, for the same reasons as
  `tables.CSTable`: worker processes share one copy through the page cache, and a
  crash mid-write must not leave a truncated file that every later run trips over.
* **Every query is validated, never wrapped.** `PairwiseGridTable` addresses cells as
  `n * stride + S`; an out-of-range `n` or `S > n` either raises a raw IndexError deep
  in a rollout or, worse, lands on a `+inf` cell and silently returns `+inf`. A feature
  that is silently infinite is exactly the kind of parity break this layer exists to
  prevent, so the check is on the hot path and costs a few comparisons.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cold_start.growing.evidence import PairwiseEvidence, PairwiseGridTable

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE_DIR = ROOT / "data" / "pairwise_tables"

# Above this the two float32 tables exceed ~5.9 GB; the exact path is used instead.
MAX_TABULATED_N = 1200
TABULATE_MAX_BYTES = 8 * 2**30
CHUNK_REPLICATES = 64


def _validate_query(
    n_lead: np.ndarray,
    S_lead: np.ndarray,
    n_chal: np.ndarray,
    S_chal: np.ndarray,
    max_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast to int64 and require `0 <= S <= n <= max_n` on both sides."""
    arrays = np.broadcast_arrays(
        np.asarray(n_lead, dtype=np.int64),
        np.asarray(S_lead, dtype=np.int64),
        np.asarray(n_chal, dtype=np.int64),
        np.asarray(S_chal, dtype=np.int64),
    )
    for n, S, side in ((arrays[0], arrays[1], "leader"), (arrays[2], arrays[3], "challenger")):
        if n.size == 0:
            continue
        if int(n.min()) < 0 or int(n.max()) > max_n:
            raise ValueError(
                f"{side} n outside [0, {max_n}]: min={int(n.min())}, max={int(n.max())}"
            )
        if int(S.min()) < 0 or bool((S > n).any()):
            raise ValueError(f"{side} S outside [0, n] for some query")
    return arrays[0], arrays[1], arrays[2], arrays[3]


class CachedPairwiseTable(PairwiseGridTable):
    """`PairwiseGridTable` whose `log_e` validates its arguments before gathering."""

    def log_e(
        self, n_lead: np.ndarray, S_lead: np.ndarray, n_chal: np.ndarray, S_chal: np.ndarray
    ) -> np.ndarray:
        n_lead, S_lead, n_chal, S_chal = _validate_query(n_lead, S_lead, n_chal, S_chal, self.max_n)
        # `np.asarray` drops the memmap subclass a gather from a mapped file carries.
        return np.asarray(super().log_e(n_lead, S_lead, n_chal, S_chal), dtype=np.float64)


class ChunkedPairwise:
    """Exact `PairwiseEvidence.log_e`, evaluated in bounded-memory chunks.

    The fallback for horizons whose tabulation would not fit: the same `.log_e`
    signature as the grid table, the corpus's exact float64 cover, and memory bounded
    by `chunk` replicates at a time (each chunk materialises `(chunk, G)` cover terms).
    """

    def __init__(self, evidence: PairwiseEvidence, max_n: int, chunk: int = CHUNK_REPLICATES):
        self.evidence = evidence
        self.max_n = int(max_n)
        self.chunk = int(chunk)

    def log_e(
        self, n_lead: np.ndarray, S_lead: np.ndarray, n_chal: np.ndarray, S_chal: np.ndarray
    ) -> np.ndarray:
        n_lead, S_lead, n_chal, S_chal = _validate_query(n_lead, S_lead, n_chal, S_chal, self.max_n)
        flat = [a.reshape(-1) for a in (n_lead, S_lead, n_chal, S_chal)]
        out = np.empty(flat[0].size, dtype=np.float64)
        for start in range(0, flat[0].size, self.chunk):
            stop = start + self.chunk
            out[start:stop] = self.evidence.log_e(*(a[start:stop] for a in flat))
        return out.reshape(n_lead.shape)


def _save_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)


def _cache_paths(cache_dir: Path, max_n: int, n_intervals: int) -> tuple[Path, Path]:
    stem = f"pair_T{max_n}_G{n_intervals}"
    return cache_dir / f"{stem}_lead_right.npy", cache_dir / f"{stem}_chal_left.npy"


def get_pairwise_table(
    max_n: int,
    n_intervals: int = 512,
    cache_dir: Path | str | None = None,
) -> CachedPairwiseTable | ChunkedPairwise:
    """Pairwise log-e lookup covering every `(n, S)` with `n <= max_n`.

    Loads the memory-mapped float32 table from `cache_dir` (building and caching it
    on first use); for `max_n > MAX_TABULATED_N` returns the exact chunked evaluator
    instead. Both expose `.log_e(n_lead, S_lead, n_chal, S_chal)`.
    """
    max_n = int(max_n)
    if max_n < 0:
        raise ValueError(f"max_n must be >= 0; got {max_n}")
    evidence = PairwiseEvidence(n_intervals=n_intervals)
    if max_n > MAX_TABULATED_N:
        return ChunkedPairwise(evidence, max_n)

    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    lead_path, chal_path = _cache_paths(cache, max_n, n_intervals)
    stride = max_n + 1
    expected = (stride * stride, n_intervals)

    if lead_path.exists() and chal_path.exists():
        lead = np.load(lead_path, mmap_mode="r")
        chal = np.load(chal_path, mmap_mode="r")
        if lead.shape == expected and chal.shape == expected and lead.dtype == np.float32:
            return CachedPairwiseTable(lead, chal, max_n)
        # A stale or truncated cache entry is rebuilt rather than trusted.

    table = evidence.tabulate(max_n, max_bytes=TABULATE_MAX_BYTES)
    cache.mkdir(parents=True, exist_ok=True)
    _save_atomic(lead_path, table.lead_right)
    _save_atomic(chal_path, table.chal_left)
    return CachedPairwiseTable(
        np.load(lead_path, mmap_mode="r"), np.load(chal_path, mmap_mode="r"), max_n
    )
