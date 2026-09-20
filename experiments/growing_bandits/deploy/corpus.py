"""Oracle-label corpus loading and the row filters every training variant is built from.

The trainer and the offline diagnostics must agree, row for row, on what "decided",
"ambiguous", "truncated" and "demoted" mean; defining the masks once here is what
makes `offline_metrics.csv` and `offline_diagnostics.csv` comparable. Every filter is
an explicit column expression -- nothing here matches on column-name substrings.

Loading returns a pandas DataFrame rather than `fit_models.load_dataset`'s column
dict: the training pipeline groups, filters and joins on metadata constantly, and a
frame keeps those operations readable.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "data" / "oracle_labels"

# Arm cap used when the corpus was generated (`GenerationSpec.max_live_arms`). A forced
# SEARCH at K + k > cap is silently demoted to REFINE after step 0 of the commitment
# window, so those labels are not the labels their k claims (audit B).
MAX_LIVE_ARMS = 64
TRAJECTORIES_PER_SHARD = 8


def load_corpus(data_dir: str | Path = DEFAULT_DATA_DIR, max_shards: int | None = None) -> pd.DataFrame:
    """Concatenate the parquet shards (sorted by file name, so `max_shards` is stable).

    Adds `meta_trajectory`: rows of one trajectory share arms, a reservoir draw sequence
    and a random stream, so it is the finest unit a split may ever separate. Snapshots
    were written trajectory-major within a shard, eight trajectories per shard, so the
    state index modulo eight recovers the trajectory (`DEPLOYMENT_PLAN.md`, audit B).
    """
    files = sorted(glob.glob(str(Path(data_dir) / "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {data_dir}")
    if max_shards is not None:
        files = files[:max_shards]
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f))
    df = pd.concat(frames, ignore_index=True)
    traj = df["meta_state_index"].to_numpy().astype(np.int64) % TRAJECTORIES_PER_SHARD
    df["meta_trajectory"] = df["meta_shard"].astype(str) + ":" + pd.Series(traj).astype(str)
    return df


def label_columns(k: int) -> tuple[str, str]:
    """`(label, se)` column names for commitment horizon `k` (failure-mode #6: SE matched to k)."""
    if k not in (1, 4, 16):
        raise ValueError(f"k must be one of 1, 4, 16; got {k}")
    return f"label_A_k{k}", f"label_se_k{k}"


def decided_mask(df: pd.DataFrame, k: int) -> np.ndarray:
    """Rows whose paired rollouts did not tie exactly; only these carry a sign."""
    label, _ = label_columns(k)
    return df[label].to_numpy(dtype=np.float64) != 0.0


def ambiguous_mask(df: pd.DataFrame, k: int) -> np.ndarray:
    """Rows whose advantage is within one Monte Carlo standard error of zero."""
    label, se = label_columns(k)
    a = df[label].to_numpy(dtype=np.float64)
    s = df[se].to_numpy(dtype=np.float64)
    return np.abs(a) < s


def truncated_or_demoted_mask(df: pd.DataFrame, k: int) -> np.ndarray:
    """Rows whose k-step commitment could not be honoured as labelled.

    Horizon-truncated: fewer than k rounds remain, so the forced action ran for
    `min(k, T - t) < k` rounds. Cap-demoted: `K + k` exceeds the arm cap, so at least
    one forced SEARCH inside the window was silently turned into a REFINE.
    """
    remaining = df["f_remaining_budget"].to_numpy(dtype=np.float64)
    K = df["f_K"].to_numpy(dtype=np.float64)
    return (remaining < k) | (K + k > MAX_LIVE_ARMS)


def lucb_only_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows generated under the LUCB allocation, the one the deployment harness uses."""
    return df["meta_allocation"].to_numpy().astype(str) == "lucb"
