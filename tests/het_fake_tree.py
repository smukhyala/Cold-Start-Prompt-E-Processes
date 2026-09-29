"""A tiny synthetic heterogeneity study on disk (tmp_path only): pool files, the data manifest, the three
profiles' worker logs with STATUS, and a frozen Pre-registration 9 extra snapshot with its manifest.

Shapes mirror make_het_pools / collect.py exactly (pool ids, arm ids, pool files' ``prompt_sha256``,
records' fields), only smaller. Nothing here touches data/, logs/ or results/.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import study as st
import yaml

GL_BLOCK_A = ["task_e0", "task_e1", "task_m0", "task_m1", "task_h0", "task_h1"]
GL_BLOCK_B = ["task_e2", "task_e3", "task_m2", "task_m3", "task_h2", "task_h3"]
GM_TASKS = ["task_e10", "task_e11", "task_m10", "task_m11", "task_h10", "task_h11"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_pool(path: Path, pool: str, arm_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"pool": pool, "arms": [{"arm_id": a, "prompt_sha256": sha(a)} for a in arm_ids]}))


def _p(rng, n, sd, base=0.5):
    return np.clip(base + rng.normal(0.0, sd, n), 0.05, 0.95)


def build(root: Path, *, n_g: int = 8, n_k: int = 6, n_bridge: int = 4, sd_g: float = 0.05, sd_k: float = 0.2,
          drop_block_b: tuple[str, ...] = (), status: str = "done", seed: int = 0) -> tuple[st.Study, Path]:
    """Write the tree under `root`; return ``(study, pool_G.yaml)`` -- the study is `HETEROGENEITY` with
    its paths pointed here (strict logs, block fallback and all). `drop_block_b`: arm ids whose block-B
    replicate-0 episodes are never collected (a budget stop mid-block-B)."""
    rng = np.random.default_rng(seed)
    data, logs = root / "data", root / "logs"
    glg = [f"GLG_{i:02d}" for i in range(n_g)]
    glk = [f"GLK_{i:02d}" for i in range(n_k)]
    gmk = [f"GMK_{i:02d}" for i in range(n_k)]
    gmb = [f"GMB_{i:02d}" for i in range(n_bridge)]
    gl_anchors = ["GL_anchor_baseline", "GL_anchor_explorer", "GL_anchor_oracle"]
    write_pool(data / "pools" / "GLG.yaml", "GLG", glg)
    write_pool(data / "pools" / "GLK.yaml", "GLK", glk)
    write_pool(data / "pools" / "GMK.yaml", "GMK", gmk)
    write_pool(data / "pools" / "GMB.yaml", "GMB", gmb)
    write_pool(data / "pools" / "anchors_gitlab.yaml", "anchor", gl_anchors)
    write_pool(data / "pools" / "anchors_gmail.yaml", "anchor", ["GM_anchor_baseline"])
    manifest = {"gitlab_task_subset_60": GL_BLOCK_A + GL_BLOCK_B, "gitlab_block_a": GL_BLOCK_A,
                "gitlab_block_b": GL_BLOCK_B, "gmail_task_subset_30": GM_TASKS,
                "bridge_source": {b: glg[i] for i, b in enumerate(gmb)}}
    (data / "manifest.json").write_text(json.dumps(manifest))

    gl_tasks = GL_BLOCK_A + GL_BLOCK_B
    task_eff_gl = rng.normal(0.0, 0.15, len(gl_tasks))
    task_eff_gm = rng.normal(0.0, 0.15, len(GM_TASKS))

    def records(pool, arms, tasks, task_eff, p_arm, reps, ended_clock_rate=0.05):
        rows = []
        for i, a in enumerate(arms):
            for j, t in enumerate(tasks):
                if a in drop_block_b and t in GL_BLOCK_B:
                    continue
                p = float(np.clip(p_arm[i] + task_eff[j], 0.02, 0.98))
                for r in (0, 1) if (i, j) in reps else (0,):
                    clock = bool(rng.random() < ended_clock_rate)
                    rows.append({"schema": "empirical_pool/1", "pool": pool, "arm_id": a, "task_id": t,
                                 "replicate": r, "attempt": 1, "status": "ok", "success": int(rng.random() < p),
                                 "cost_usd": 0.01, "prompt_sha256": sha(a), "timed_out": clock,
                                 "ended_by": "clock" if clock else "agent"})
        return rows

    def pairs(n_arms, n_tasks, k, only_tasks=None):
        cells = [(i, j) for i in range(n_arms) for j in range(n_tasks) if only_tasks is None or j in only_tasks]
        idx = rng.choice(len(cells), size=min(k, len(cells)), replace=False)
        return {cells[int(x)] for x in idx}

    a_idx, b_idx = set(range(len(GL_BLOCK_A))), set(range(len(GL_BLOCK_A), len(gl_tasks)))
    # dense replicate pairs (half of every block's cells), so no prompt-bootstrap resample loses its noise
    gl_rows = (records("GLG", glg, gl_tasks, task_eff_gl, _p(rng, n_g, sd_g),
                       pairs(n_g, len(gl_tasks), n_g * 3, a_idx) | pairs(n_g, len(gl_tasks), n_g * 3, b_idx))
               + records("GLK", glk, gl_tasks, task_eff_gl, _p(rng, n_k, sd_k),
                         pairs(n_k, len(gl_tasks), n_k * 3, a_idx) | pairs(n_k, len(gl_tasks), n_k * 3, b_idx))
               + records("anchor", gl_anchors, gl_tasks, task_eff_gl, np.array([0.5, 0.1, 0.9]), set()))
    gm_rows = (records("GMK", gmk, GM_TASKS, task_eff_gm, _p(rng, n_k, sd_k), pairs(n_k, len(GM_TASKS), n_k * 3))
               + records("anchor", ["GM_anchor_baseline"], GM_TASKS, task_eff_gm, np.array([0.6]), set()))
    br_rows = records("GMB", gmb, GM_TASKS, task_eff_gm, _p(rng, n_bridge, sd_g), set(), ended_clock_rate=0.0)
    for name, rows in (("gitlab", gl_rows), ("gmail", gm_rows), ("bridge", br_rows)):
        d = logs / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "worker_0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "STATUS").write_text(status + "\n")

    # Pre-registration 9's frozen snapshot (pool G + its anchor) and the reservoir manifest beside it
    p9 = root / "prereg9" / "reservoirs"
    p9.mkdir(parents=True)
    g_ids = [f"G_{i:02d}" for i in range(n_g)]
    snap_rows = []
    p_g = _p(rng, n_g, sd_g)
    g_pairs = pairs(n_g, len(GM_TASKS), n_g * 3)
    for i, a in enumerate(g_ids):
        for j, t in enumerate(GM_TASKS):
            for r in (0, 1) if (i, j) in g_pairs else (0,):
                p = float(np.clip(p_g[i] + task_eff_gm[j], 0.02, 0.98))
                snap_rows.append({"arm_id": a, "attempt": 1, "pool": "G", "replicate": r, "status": "ok",
                                  "success": int(rng.random() < p), "task_id": t})
    for t in GM_TASKS:
        snap_rows.append({"arm_id": "anchor_baseline", "attempt": 1, "pool": "anchor", "replicate": 0,
                          "status": "ok", "success": int(rng.random() < 0.65), "task_id": t})
    snap = p9 / "outcomes_snapshot.jsonl"
    snap.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in snap_rows))
    (p9 / "manifest.json").write_text(json.dumps({"outcomes_snapshot": {
        "file": snap.name, "sha256": hashlib.sha256(snap.read_bytes()).hexdigest()}}))
    pool_g = root / "prereg9" / "pool_G.yaml"
    write_pool(pool_g, "G", g_ids)

    study = dataclasses.replace(
        st.HETEROGENEITY, data_dir=data, res_dir=data / "reservoirs",
        log_dirs=(logs / "gitlab", logs / "gmail", logs / "bridge"), extra_outcomes=((snap, "G", "GMG"),))
    return study, pool_g
