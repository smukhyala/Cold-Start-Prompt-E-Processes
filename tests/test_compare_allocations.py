from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cold_start.cli import _bootstrap  # noqa: F401  (register components)
from cold_start.cli import compare


def _write_minimal_config(path: Path, num_tasks: int) -> Path:
    """Construct a tiny ToyEnv config inline so the smoke test runs in seconds."""
    cfg = f"""trial:
  name: cmp_smoke
  seed: 7
  num_tasks: {num_tasks}
  num_trials: 1
  paired_streams: true
  max_agent_steps: 4
  output_dir: logs/

prompts:
  axes: configs/axes.yaml
  template: configs/template.jinja
  arms: configs/arms_initial.yaml

model:
  type: mock
  id: mock
  max_tokens: 32
  temperature: 0.0
  prompt_cache: false

task_source:
  type: toy
  params:
    arm_probs:
      baseline: 0.30
      planner: 0.70
      cautious: 0.40
      explorer: 0.45
      balanced: 0.50
      overthinker: 0.35
      rapid: 0.45
      verifier: 0.40
      exploratory: 0.50
      algorithmic: 0.65
      junior_reactive: 0.25
      domain_expert: 0.60

reward:
  type: binary

inference:
  per_arm:
    type: upward_capital
    params: {{ m0: 0.5 }}
  global:
    type: global_null
    params: {{ combine: linear_mixture }}
  confidence_sequence:
    alpha: 0.05
    grid_size: 32

policy:
  type: uniform
  params: {{}}

logging:
  level: WARNING
  jsonl: true
"""
    path.write_text(cfg)
    return path


def test_compare_smoke_writes_expected_artifacts(tmp_path, monkeypatch):
    """End-to-end smoke: two policies, 50 tasks each, must produce comparison.md
    + the overlay/pull-histogram PNGs and a JSONL log per policy."""
    cfg_path = _write_minimal_config(tmp_path / "cmp.yaml", num_tasks=50)
    out_dir = tmp_path / "out"

    # Run from the repo root so relative axes/arms/template paths resolve.
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    monkeypatch.setattr(
        sys, "argv",
        [
            "cold-start-compare",
            "--base", str(cfg_path),
            "--policies", "uniform,ucb",
            "--out", str(out_dir),
        ],
    )
    compare.main()

    # Comparison artifacts.
    assert (out_dir / "comparison.md").exists()
    assert (out_dir / "global_log_e_overlay.png").exists()
    assert (out_dir / "pulls_by_policy.png").exists()

    # Per-policy run logs.
    uniform_dir = out_dir / "uniform"
    ucb_dir = out_dir / "ucb"
    assert any(p.suffix == ".jsonl" for p in uniform_dir.iterdir())
    assert any(p.suffix == ".jsonl" for p in ucb_dir.iterdir())

    md = (out_dir / "comparison.md").read_text()
    assert "Allocation comparison" in md
    assert "τ_α" in md or "tau_alpha" in md  # stopping-time section heading
    assert "uniform" in md and "ucb" in md


@pytest.mark.parametrize("bad_arg", ["", "  ,  "])
def test_compare_rejects_empty_policies(tmp_path, monkeypatch, bad_arg):
    cfg_path = _write_minimal_config(tmp_path / "cmp.yaml", num_tasks=5)
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(
        sys, "argv",
        [
            "cold-start-compare",
            "--base", str(cfg_path),
            "--policies", bad_arg,
            "--out", str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit):
        compare.main()
