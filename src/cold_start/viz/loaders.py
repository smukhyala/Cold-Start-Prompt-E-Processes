"""Discovery and loading of run JSONLs + webarena artifacts.

A single "run" may span multiple JSONL files (original + one or more
`_resume_from_N` continuations). `discover_runs` groups by base run_id;
`load_run` merges records and produces DataFrame views.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_RUN_ID_RE = re.compile(
    r"^(.+?_trial\d+_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)(_resume_from_\d+)?$"
)


def _base_id(run_id: str) -> str:
    m = _RUN_ID_RE.match(run_id)
    return m.group(1) if m else run_id


def _read_first_record(p: Path) -> dict[str, Any] | None:
    with p.open() as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def _read_last_record(p: Path) -> dict[str, Any] | None:
    last: str | None = None
    with p.open() as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def _count_lines(p: Path) -> int:
    with p.open() as f:
        return sum(1 for line in f if line.strip())


@dataclass(frozen=True)
class RunMeta:
    """Lightweight metadata for the runs list — no records loaded yet."""

    base_id: str
    label: str
    jsonl_paths: tuple[Path, ...]
    n_records: int
    config_hash: str
    latest_ts: str
    task_source_type: str


def discover_runs(logs_dir: Path | str = "logs") -> list[RunMeta]:
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return []

    groups: dict[str, list[Path]] = {}
    for p in logs_dir.glob("*.jsonl"):
        first = _read_first_record(p)
        if first is None:
            continue
        base = _base_id(str(first.get("run_id", p.stem)))
        groups.setdefault(base, []).append(p)

    runs: list[RunMeta] = []
    for base, paths in groups.items():
        n = sum(_count_lines(p) for p in paths)
        if n == 0:
            continue
        last_path = max(paths, key=lambda x: x.stat().st_mtime)
        last = _read_last_record(last_path) or {}
        first_path = min(paths, key=lambda x: x.stat().st_mtime)
        first = _read_first_record(first_path) or {}
        runs.append(
            RunMeta(
                base_id=base,
                label=base,
                jsonl_paths=tuple(sorted(paths)),
                n_records=n,
                config_hash=str(last.get("config_hash", "")),
                latest_ts=str(last.get("timestamp_utc", "")),
                task_source_type=_infer_task_source(first),
            )
        )
    runs.sort(key=lambda r: r.latest_ts, reverse=True)
    return runs


def _infer_task_source(first_rec: dict[str, Any]) -> str:
    """Best-effort: 'webarena' | 'sanity' | 'toy' | '?' from first record's task_id shape."""
    tid = str(first_rec.get("task_id", ""))
    if tid.startswith("sanity-"):
        return "sanity"
    if tid.startswith("toy-"):
        return "toy"
    if tid.startswith("task_"):
        return "webarena"
    return "?"


@dataclass
class Run:
    meta: RunMeta
    records: list[dict[str, Any]]
    df: pd.DataFrame

    @property
    def run_id(self) -> str:
        return self.meta.base_id

    @property
    def config_hash(self) -> str:
        return self.meta.config_hash

    @property
    def n_records(self) -> int:
        return len(self.records)

    @property
    def pass_rate(self) -> float:
        return sum(r["success"] for r in self.records) / self.n_records if self.records else 0.0

    def per_arm_summary(self) -> pd.DataFrame:
        """Final per-arm stats, sourced from the last record's per_arm_state block."""
        if not self.records:
            return pd.DataFrame()
        last_state = self.records[-1]["per_arm_state"]
        rows = []
        for arm, s in last_state.items():
            sub = self.df[self.df.arm_id == arm]
            pulls = int(s["pulls"])
            succ = int(s["successes"])
            rows.append(
                {
                    "arm": arm,
                    "pulls": pulls,
                    "passes": succ,
                    "pass_rate": succ / pulls if pulls else 0.0,
                    "mean_wallclock_s": float(sub["wallclock_s"].mean()) if len(sub) else 0.0,
                    "mean_steps": float(sub["steps"].mean()) if len(sub) else 0.0,
                    "log_e": float(s["log_e"]),
                    "cs_lo": float(s["cs_lo"]),
                    "cs_hi": float(s["cs_hi"]),
                }
            )
        return pd.DataFrame(rows).sort_values("log_e", ascending=False).reset_index(drop=True)

    def state_trajectory(self) -> pd.DataFrame:
        """Long-format: one row per (t, arm) with that arm's state after step t.

        Each record stores per_arm_state for ALL arms (not just the one pulled),
        so arms' trajectories are defined at every t as step functions.
        """
        rows = []
        for r in self.records:
            t = int(r["t"])
            for arm, s in r["per_arm_state"].items():
                rows.append(
                    {
                        "t": t,
                        "arm": arm,
                        "pulls": int(s["pulls"]),
                        "successes": int(s["successes"]),
                        "log_e": float(s["log_e"]),
                        "cs_lo": float(s["cs_lo"]),
                        "cs_hi": float(s["cs_hi"]),
                    }
                )
        return pd.DataFrame(rows).sort_values(["arm", "t"]).reset_index(drop=True)

    def task_artifacts(
        self,
        arm_id: str,
        task_id: str,
        artifacts_dir: Path | str = "logs/webarena",
    ) -> dict[str, Any]:
        """Locate per-task artifacts written by WebArenaInfinityAdapter.

        Note: the adapter overwrites <arm>/<task_id>/ on each call, so if the
        same (arm, task) pair was pulled more than once (cycle mode with T>60),
        only the most recent pull's artifacts survive on disk.
        """
        d = Path(artifacts_dir) / arm_id / task_id
        out: dict[str, Any] = {"dir": str(d), "exists": d.exists()}
        if not d.exists():
            return out
        for sub in ("history.json", "result.json"):
            if (d / sub).exists():
                out[sub] = d / sub
        conv_dir = d / "conversations"
        if conv_dir.exists():
            out["conversations"] = sorted(conv_dir.glob("*.txt"))
        ss_dir = d / "screenshots"
        if ss_dir.exists():
            out["screenshots"] = sorted(ss_dir.glob("*.png"))
        return out


def load_run(meta: RunMeta) -> Run:
    # Read files in mtime order (oldest first) so newer files overwrite older
    # on duplicate t values — handles the case where a manual "plus" file
    # concatenates records already present in a sibling resume JSONL.
    paths = sorted(meta.jsonl_paths, key=lambda p: p.stat().st_mtime)
    by_t: dict[int, dict[str, Any]] = {}
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    by_t[int(rec["t"])] = rec
    records = [by_t[t] for t in sorted(by_t)]
    return _build_run(meta, records)


def merge_runs(metas: list[RunMeta], label: str | None = None) -> Run:
    """Combine multiple RunMetas into one Run. Used when the orchestrator
    didn't group a continuation with its origin (e.g. pre-fix resume runs
    whose inherited run_id didn't match)."""
    if not metas:
        raise ValueError("merge_runs: empty metas")
    all_paths: list[Path] = []
    for m in metas:
        all_paths.extend(m.jsonl_paths)
    merged_meta = RunMeta(
        base_id=label or f"merged({','.join(m.base_id[-20:] for m in metas)})",
        label=label or " + ".join(m.base_id[-20:] for m in metas),
        jsonl_paths=tuple(all_paths),
        n_records=sum(m.n_records for m in metas),
        config_hash=metas[0].config_hash,
        latest_ts=max(m.latest_ts for m in metas),
        task_source_type=metas[0].task_source_type,
    )
    records: list[dict[str, Any]] = []
    for p in all_paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records.sort(key=lambda r: int(r["t"]))
    # Drop duplicate-t records (keep the latest file's — defensive against
    # accidental double-write). Resumes shouldn't produce duplicates but the
    # merge path is used in ad-hoc situations too.
    seen_t: set[int] = set()
    deduped = []
    for r in reversed(records):  # later = precedence
        t = int(r["t"])
        if t in seen_t:
            continue
        seen_t.add(t)
        deduped.append(r)
    deduped.reverse()
    return _build_run(merged_meta, deduped)


def _build_run(meta: RunMeta, records: list[dict[str, Any]]) -> Run:

    flat = []
    for r in records:
        arm = r["arm_id"]
        arm_state = r["per_arm_state"].get(arm, {})
        flat.append(
            {
                "t": int(r["t"]),
                "timestamp_utc": str(r.get("timestamp_utc", "")),
                "arm_id": arm,
                "task_id": str(r["task_id"]),
                "difficulty": str(r.get("task_meta", {}).get("difficulty", "")),
                "success": int(r["success"]),
                "reward": float(r["reward"]),
                "steps": int(r["steps"]),
                "wallclock_s": float(r["wallclock_s"]),
                "log_e": float(arm_state.get("log_e", 0.0)),
                "cs_lo": float(arm_state.get("cs_lo", 0.0)),
                "cs_hi": float(arm_state.get("cs_hi", 1.0)),
                "global_log_e": float(r["global_e"]["log_e"]),
                "global_rejected": bool(r["global_e"]["rejected"]),
            }
        )
    df = pd.DataFrame(flat)
    return Run(meta=meta, records=records, df=df)
