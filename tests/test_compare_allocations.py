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


def test_compare_continues_when_one_policy_crashes(tmp_path, monkeypatch):
    """If one policy's run_trial raises, the comparison should still produce
    a report for the policies that completed and mark the failed one."""
    cfg_path = _write_minimal_config(tmp_path / "cmp.yaml", num_tasks=20)
    out_dir = tmp_path / "out"
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    # Inject a failure: stub run_trial to raise on the second call only.
    real_run_trial = compare.run_trial
    call_count = {"n": 0}

    def flaky_run_trial(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated browser-use hang")
        return real_run_trial(*a, **kw)

    monkeypatch.setattr(compare, "run_trial", flaky_run_trial)

    monkeypatch.setattr(
        sys, "argv",
        [
            "cold-start-compare",
            "--base", str(cfg_path),
            "--policies", "uniform,ucb,thompson",
            "--out", str(out_dir),
        ],
    )
    # main() should not raise — the comparison report is still produced.
    compare.main()

    md = (out_dir / "comparison.md").read_text()
    # Failed policy surfaced in its own section
    assert "Failed policies" in md
    assert "ucb" in md and "simulated browser-use hang" in md
    # Completed policies still in the τ_α table
    assert "uniform" in md and "thompson" in md


def test_compare_aborts_cleanly_if_every_policy_fails(tmp_path, monkeypatch):
    cfg_path = _write_minimal_config(tmp_path / "cmp.yaml", num_tasks=5)
    out_dir = tmp_path / "out"
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(project_root)

    def always_fail(*a, **kw):
        raise RuntimeError("every policy is doomed")

    monkeypatch.setattr(compare, "run_trial", always_fail)
    monkeypatch.setattr(
        sys, "argv",
        [
            "cold-start-compare",
            "--base", str(cfg_path),
            "--policies", "uniform,ucb",
            "--out", str(out_dir),
        ],
    )
    with pytest.raises(SystemExit):
        compare.main()


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
