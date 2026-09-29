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
  (the bridge, no replicate pairs) borrowing GMG's within-cell noise. Its logs are *strict*
  (`Study.strict_logs`): every log dir must exist, every record's ``prompt_sha256`` must equal its
  arm's frozen pool-file sha, and ``replay estimate`` needs each log dir's STATUS in
  `ESTIMABLE_STATUSES`. Its GitLab cells follow the block-A fallback (`BlockFallback`,
  `task_universe`): if any GitLab pool's replicate-0 block B is incomplete, every GitLab cell is
  analysed on block A only.

Every extra snapshot (``extra_outcomes``) is sha-checked against the ``manifest.json`` beside it
(the source study's reservoir manifest) before its rows are used.

Seeds: a study's seeds are ``seed_base + 1_000 * index`` (`replay.seed_for`); the index is
below ``(N_BOOT + 1) * 10_000 * len(pools)``, so `HETEROGENEITY`'s base (1e10) lies above
every seed Pre-registration 9 used (4.0e8 .. 4.41e9) and its range (1e10 .. ~2.0e10) is
disjoint from it and from the calibration base (5e10).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cold_start.growing import empirical as emp  # noqa: E402

NOISE_MODES: tuple[str, ...] = ("pairwise_decision", "per_pool")
#: A log dir's STATUS (collect.py's final status) that ``replay estimate`` accepts for a strict study:
#: the whole queue ran, the budget stopped it, or a ``--through-index`` stage finished cleanly.
ESTIMABLE_STATUSES: tuple[str, ...] = ("done", "budget", "through")
#: `task_universe`'s two answers for a study with a `BlockFallback`.
UNIVERSE_FULL, UNIVERSE_BLOCK_A = "subset_60", "block_a"


@dataclass(frozen=True)
class BlockFallback:
    """Spec 4.3 as amended before registration (I1 ruling): the GitLab design is block A plus block B;
    if ANY of `pool_files`' arms lacks a terminal replicate-0 record on any block-B task (a budget stop
    mid-block-B), EVERY GitLab cell is analysed on block A alone -- mechanically, never by choice.

    ``pool_files`` are relative to the study's ``data_dir``; ``*_key`` name the manifest's task lists."""
    pool_files: tuple[str, ...]
    full_key: str
    block_a_key: str
    block_b_key: str


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
    #: Missing log dir -> error; every record's prompt_sha256 == its arm's ``<data_dir>/pools/*.yaml`` sha;
    #: ``replay estimate`` needs every log dir's STATUS in `ESTIMABLE_STATUSES`.
    strict_logs: bool = False
    block_fallback: BlockFallback | None = None

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
    strict_logs=True,
    block_fallback=BlockFallback(pool_files=("pools/GLG.yaml", "pools/GLK.yaml", "pools/anchors_gitlab.yaml"),
                                 full_key="gitlab_task_subset_60", block_a_key="gitlab_block_a",
                                 block_b_key="gitlab_block_b"),
)

STUDIES: dict[str, Study] = {s.name: s for s in (PREREG9, HETEROGENEITY)}


def get_study(name: str) -> Study:
    if name not in STUDIES:
        raise KeyError(f"unknown study {name!r}; studies={sorted(STUDIES)}")
    return STUDIES[name]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_extra_snapshot(snapshot: Path) -> None:
    """An extra snapshot must be the one its source study froze: ``manifest.json`` beside it names it
    (``outcomes_snapshot.file``) with the same sha256 (``outcomes_snapshot.sha256``)."""
    snapshot = Path(snapshot)
    manifest_path = snapshot.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"extra snapshot {snapshot}: no {manifest_path} to check it against")
    frozen = json.loads(manifest_path.read_text()).get("outcomes_snapshot") or {}
    if frozen.get("file") != snapshot.name:
        raise ValueError(f"extra snapshot {snapshot}: {manifest_path} freezes {frozen.get('file')!r}, not {snapshot.name!r}")
    got = _file_sha256(snapshot)
    if got != frozen.get("sha256"):
        raise ValueError(f"extra snapshot {snapshot} sha256 {got} != its manifest's {frozen.get('sha256')}")


def pool_file_shas(data_dir: Path) -> dict[str, str]:
    """``{arm_id: prompt_sha256}`` over every ``<data_dir>/pools/*.yaml`` (the frozen rendered-prompt sha
    `make_pools.write_pool` records per arm -- the collector stamps the same value on every record)."""
    out: dict[str, str] = {}
    files = sorted((Path(data_dir) / "pools").glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no pool files under {Path(data_dir) / 'pools'}")
    for path in files:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for arm in doc["arms"]:
            arm_id = str(arm["arm_id"])
            if arm_id in out:
                raise ValueError(f"arm {arm_id} appears in more than one pool file under {path.parent}")
            out[arm_id] = str(arm["prompt_sha256"])
    return out


def check_prompt_shas(attempts: pd.DataFrame, shas: Mapping[str, str], *, where: str) -> None:
    """Every record's ``prompt_sha256`` equals its arm's frozen pool-file sha (I5 ruling): a record of a
    prompt that is not the one now in the pool file would silently attach its outcome to another prompt."""
    if attempts.empty:
        return
    if "prompt_sha256" not in attempts.columns:
        raise ValueError(f"{where}: records carry no prompt_sha256")
    want = attempts["arm_id"].astype(str).map(shas)
    unknown = sorted(set(attempts.loc[want.isna(), "arm_id"].astype(str)))
    if unknown:
        raise ValueError(f"{where}: arms {unknown[:5]} are in no pool file")
    bad = attempts["prompt_sha256"].astype(str) != want
    if bad.any():
        eg = attempts.loc[bad, ["arm_id", "task_id", "replicate", "prompt_sha256"]].head(3).to_dict("records")
        raise ValueError(f"{where}: {int(bad.sum())} records' prompt_sha256 differs from the pool file's, e.g. {eg}")


def log_status(log_dir: Path) -> str | None:
    path = Path(log_dir) / "STATUS"
    return path.read_text().strip() if path.exists() else None


def check_log_status(study: Study) -> None:
    """Refuse to estimate from a log dir whose collector has not stopped cleanly: every log dir's STATUS
    must be in `ESTIMABLE_STATUSES` (a running, failed or provider_down collection is not a snapshot)."""
    bad = {str(d): log_status(d) for d in study.log_dirs if log_status(d) not in ESTIMABLE_STATUSES}
    if bad:
        raise RuntimeError(f"study {study.name}: log dir STATUS not in {ESTIMABLE_STATUSES}: {bad} "
                           "(pass --allow-unfinished-logs to estimate anyway)")


def load_study_outcomes(study: Study) -> pd.DataFrame:
    """Terminal outcomes of every log dir of `study`, plus its relabelled extra snapshots.

    Each log dir's ``worker_*.jsonl`` files are read with `emp.load_attempts` and reduced by
    `emp.terminal_outcomes` (duplicate terminal rows raise); a missing log dir is skipped -- or, for a
    `Study.strict_logs` study, raises `FileNotFoundError`, and every record's ``prompt_sha256`` is
    checked against ``<data_dir>/pools/*.yaml`` (`check_prompt_shas`).
    Each ``(snapshot, source, relabel)`` extra is sha-checked against the manifest beside it
    (`check_extra_snapshot`) and contributes its rows with ``pool == source``, re-pooled as ``relabel``
    with ``arm_id = f"{relabel}_{arm_id}"`` (so a relabelled arm can never collide with an arm collected
    under the new pool's own name).
    """
    parts: list[pd.DataFrame] = []
    shas = pool_file_shas(study.data_dir) if study.strict_logs else None
    for log_dir in study.log_dirs:
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            if study.strict_logs:
                raise FileNotFoundError(f"study {study.name}: log dir {log_dir} does not exist")
            continue
        attempts = emp.load_attempts(sorted(log_dir.glob("worker_*.jsonl")))
        if shas is not None:
            check_prompt_shas(attempts, shas, where=str(log_dir))
        parts.append(emp.terminal_outcomes(attempts))
    for snapshot, source, relabel in study.extra_outcomes:
        check_extra_snapshot(snapshot)
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


# ---- the block-A fallback (spec 4.3 as amended, I1 ruling) --------------------------------------


def load_data_manifest(study: Study) -> dict:
    return json.loads((Path(study.data_dir) / "manifest.json").read_text())


def fallback_arms(study: Study) -> list[str]:
    """Every arm id of the study's `BlockFallback` pool files (GitLab: GLG, GLK and the three anchors)."""
    if study.block_fallback is None:
        raise ValueError(f"study {study.name} has no block fallback")
    arms: list[str] = []
    for rel in study.block_fallback.pool_files:
        doc = yaml.safe_load((Path(study.data_dir) / rel).read_text(encoding="utf-8"))
        arms += [str(a["arm_id"]) for a in doc["arms"]]
    if len(set(arms)) != len(arms):
        raise ValueError(f"study {study.name}: duplicate arm ids across {study.block_fallback.pool_files}")
    return arms


def task_universe(outcomes: pd.DataFrame, study: Study, manifest: Mapping | None = None) -> str | None:
    """`UNIVERSE_FULL` (``"subset_60"``) iff every fallback arm has a terminal (``ok`` or ``missing``)
    replicate-0 record on every block-B task; `UNIVERSE_BLOCK_A` (``"block_a"``) otherwise. ``None`` for a
    study without a `BlockFallback`. Read off `outcomes` alone -- no judgment, no threshold."""
    fb = study.block_fallback
    if fb is None:
        return None
    manifest = load_data_manifest(study) if manifest is None else manifest
    block_b = [str(t) for t in manifest[fb.block_b_key]]
    arms = fallback_arms(study)
    term = outcomes[(outcomes["replicate"].astype(int) == 0) & outcomes["status"].isin(emp.TERMINAL_STATUSES)]
    have = set(zip(term["arm_id"].astype(str), term["task_id"].astype(str), strict=True))
    complete = all((a, t) in have for a in arms for t in block_b)
    return UNIVERSE_FULL if complete else UNIVERSE_BLOCK_A


def universe_tasks(study: Study, universe: str, manifest: Mapping | None = None) -> list[str]:
    """The fallback cells' task list under `universe` (the manifest's full list or its block A)."""
    fb = study.block_fallback
    if fb is None:
        raise ValueError(f"study {study.name} has no block fallback")
    manifest = load_data_manifest(study) if manifest is None else manifest
    if universe == UNIVERSE_FULL:
        return [str(t) for t in manifest[fb.full_key]]
    if universe == UNIVERSE_BLOCK_A:
        return [str(t) for t in manifest[fb.block_a_key]]
    raise ValueError(f"unknown task universe {universe!r}; expected {UNIVERSE_FULL!r} or {UNIVERSE_BLOCK_A!r}")


def restrict_to_universe(outcomes: pd.DataFrame, study: Study, universe: str | None,
                         manifest: Mapping | None = None) -> pd.DataFrame:
    """`outcomes` with every fallback arm's rows outside `universe`'s task list dropped (under
    ``block_a``: GitLab's block-B episodes, replicate pairs included); unchanged for ``subset_60`` or a
    study without a fallback."""
    if universe is None or universe == UNIVERSE_FULL:
        return outcomes
    tasks = set(universe_tasks(study, universe, manifest))
    arms = set(fallback_arms(study))
    drop = outcomes["arm_id"].astype(str).isin(arms) & ~outcomes["task_id"].astype(str).isin(tasks)
    return outcomes[~drop].reset_index(drop=True)
