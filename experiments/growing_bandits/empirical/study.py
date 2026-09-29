"""One replay study: its pools (cells), test ids, paths, seeds and noise model.

`replay.py`, `describe.py` and `registered_contrast.py` were written for Pre-registration 9's
one pool pair ("G", "F"); every public function there now takes ``study: Study = PREREG9``
and reads its ids, paths and seeds from it, so a second study reuses the whole pipeline
(estimate -> point / kgrid / boot -> describe -> prompt-bootstrap contrast) without touching
Pre-registration 9's numbers. The module constants those scripts export (`replay.POOLS`,
`replay.RES_DIR`, ...) stay Pre-registration 9's values.

* `PREREG9` -- the real Gmail pools G and F (Pre-registration 9), byte-for-byte as run.
* `HETEROGENEITY` -- prompt heterogeneity across Gmail/GitLab x stylistic/knowledge prompts
  (Pre-registration 10): five cells, GMG re-using Pre-registration 9's frozen G outcomes, GMB
  (the bridge, no replicate pairs) borrowing GMG's within-cell noise.

Seeds: a study's seeds are ``seed_base + 1_000 * index`` (`replay.seed_for`); the index is
below ``(N_BOOT + 1) * 10_000 * len(pools)``, so `HETEROGENEITY`'s base (1e10) lies above
every seed Pre-registration 9 used (4.0e8 .. 4.41e9) and its range (1e10 .. ~2.0e10) is
disjoint from it and from the calibration base (5e10).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.growing import empirical as emp  # noqa: E402

NOISE_MODES: tuple[str, ...] = ("pairwise_decision", "per_pool")


@dataclass(frozen=True)
class Study:
    name: str                      # "prereg9" | "het"
    test: str                      # "emp" | "het"
    boot_test: str                 # "emp_boot" | "het_boot"
    pools: tuple[str, ...]         # ("G", "F") | ("GMG", "GMK", "GMB", "GLG", "GLK")
    data_dir: Path
    res_dir: Path
    log_dirs: tuple[Path, ...]
    seed_base: int
    table_prefix: str              # "emp" | "het"
    noise_mode: str                # "pairwise_decision" (Pre-reg 9) | "per_pool"
    borrowed_noise: Mapping[str, str]  # pool -> pool whose v it uses when it has no replicate pairs
    extra_outcomes: tuple[tuple[Path, str, str], ...]  # (snapshot file, source pool, relabel as)

    def __post_init__(self) -> None:
        if self.noise_mode not in NOISE_MODES:
            raise ValueError(f"{self.name}: unknown noise_mode {self.noise_mode!r}; modes={NOISE_MODES}")
        if len(set(self.pools)) != len(self.pools):
            raise ValueError(f"{self.name}: duplicate pools {self.pools}")
        for pool, donor in self.borrowed_noise.items():
            if pool not in self.pools or donor not in self.pools or donor == pool:
                raise ValueError(f"{self.name}: borrowed_noise {pool!r} -> {donor!r} must name two distinct pools "
                                 f"of {self.pools}")

    def table(self, name: str) -> str:
        """``<table_prefix>_<name>.csv`` -- e.g. ``emp_kgrid.csv`` for Pre-registration 9."""
        return f"{self.table_prefix}_{name}.csv"


_EMP_DATA = ROOT / "data" / "empirical_pool"
_HET_DATA = ROOT / "data" / "heterogeneity"
_HET_LOGS = ROOT / "logs" / "heterogeneity"

PREREG9 = Study(
    name="prereg9", test="emp", boot_test="emp_boot", pools=("G", "F"),
    data_dir=_EMP_DATA, res_dir=_EMP_DATA / "reservoirs", log_dirs=(ROOT / "logs" / "empirical_pool",),
    seed_base=400_000_000, table_prefix="emp", noise_mode="pairwise_decision", borrowed_noise={},
    extra_outcomes=(),
)

HETEROGENEITY = Study(
    name="het", test="het", boot_test="het_boot", pools=("GMG", "GMK", "GMB", "GLG", "GLK"),
    data_dir=_HET_DATA, res_dir=_HET_DATA / "reservoirs",
    log_dirs=(_HET_LOGS / "gitlab", _HET_LOGS / "gmail", _HET_LOGS / "bridge"),
    # Above every Pre-registration 9 seed (4.0e8 .. 4.41e9); see the module docstring.
    seed_base=10_000_000_000, table_prefix="het", noise_mode="per_pool", borrowed_noise={"GMB": "GMG"},
    extra_outcomes=((_EMP_DATA / "reservoirs" / "outcomes_snapshot.jsonl", "G", "GMG"),),
)

STUDIES: dict[str, Study] = {s.name: s for s in (PREREG9, HETEROGENEITY)}


def get_study(name: str) -> Study:
    if name not in STUDIES:
        raise KeyError(f"unknown study {name!r}; studies={sorted(STUDIES)}")
    return STUDIES[name]


def load_study_outcomes(study: Study) -> pd.DataFrame:
    """Terminal outcomes of every log dir of `study`, plus its relabelled extra snapshots.

    Each log dir's ``worker_*.jsonl`` files are read with `emp.load_attempts` and reduced by
    `emp.terminal_outcomes` (duplicate terminal rows raise); a missing log dir is skipped.
    Each ``(snapshot, source, relabel)`` extra contributes its rows with ``pool == source``,
    re-pooled as ``relabel`` with ``arm_id = f"{relabel}_{arm_id}"`` (so a relabelled arm can
    never collide with an arm collected under the new pool's own name).
    """
    parts: list[pd.DataFrame] = []
    for log_dir in study.log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            continue
        parts.append(emp.terminal_outcomes(emp.load_attempts(sorted(log_dir.glob("worker_*.jsonl")))))
    for snapshot, source, relabel in study.extra_outcomes:
        rows = [json.loads(line) for line in Path(snapshot).read_text().splitlines() if line.strip()]
        extra = pd.DataFrame(rows)
        extra = extra[extra["pool"] == source].copy()
        extra["arm_id"] = [f"{relabel}_{a}" for a in extra["arm_id"]]
        extra["pool"] = relabel
        parts.append(extra.reset_index(drop=True))
    if not parts:
        raise FileNotFoundError(f"study {study.name}: no log dir of {[str(d) for d in study.log_dirs]} exists "
                                "and it declares no extra outcomes")
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)
