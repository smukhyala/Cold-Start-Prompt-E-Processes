from __future__ import annotations

import socket
from pathlib import Path

import pytest

from cold_start.config import load_config
from cold_start.runner.pre_flight import PreflightError, run_preflight, _port_in_use


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return load_config(_PROJECT_ROOT / "configs" / name)


def test_preflight_passes_for_toy_pilot(monkeypatch, tmp_path):
    """Toy pilot with mock model + no anthropic key should pass cleanly."""
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _load("pilot_toy.yaml")
    run_preflight(cfg, output_dir=tmp_path)


def test_preflight_passes_for_paper_faithful_pilot(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _load("pilot_toy_upward.yaml")
    run_preflight(cfg, output_dir=tmp_path)


def test_preflight_requires_anthropic_key_for_anthropic_model(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _load("experiment_500.yaml")  # model.type = anthropic, task = toy
    with pytest.raises(PreflightError, match="ANTHROPIC_API_KEY"):
        run_preflight(cfg, output_dir=tmp_path)


def test_preflight_accepts_anthropic_key_when_present(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    cfg = _load("experiment_500.yaml")
    run_preflight(cfg, output_dir=tmp_path)  # should not raise


def test_preflight_detects_unwritable_output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _load("pilot_toy.yaml")
    # Make the output dir read-only.
    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o500)
    try:
        with pytest.raises(PreflightError, match="not writable"):
            run_preflight(cfg, output_dir=out)
    finally:
        out.chmod(0o700)  # restore so tmp_path cleanup can remove it


def test_preflight_detects_busy_port_for_webarena(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Bind a real socket on a random free port, then point the config at it
    # so the preflight sees it as "in use".
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        busy_port = sock.getsockname()[1]
        cfg = _load("webarena_gmail.yaml")
        cfg.task_source.params["port"] = busy_port
        with pytest.raises(PreflightError, match="port .* already in use"):
            run_preflight(cfg, output_dir=tmp_path)


def test_preflight_detects_missing_webarena_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("WEBARENA_INFINITY_ROOT", str(tmp_path / "missing"))
    cfg = _load("webarena_gmail.yaml")
    with pytest.raises(PreflightError, match="webarena-infinity not found"):
        run_preflight(cfg, output_dir=tmp_path)


def _free_port() -> int:
    """Bind a socket transiently to pick an OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_preflight_detects_missing_app_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = _load("webarena_gmail.yaml")
    # Use a free port so we exercise the app-dir check, not the port check
    # (8001 may be busy if a real webarena run is in progress).
    cfg.task_source.params["port"] = _free_port()
    cfg.task_source.params["web_app"] = "apps/this-app-does-not-exist"
    with pytest.raises(PreflightError, match="app dir not found"):
        run_preflight(cfg, output_dir=tmp_path)


def test_preflight_detects_missing_task_suite(monkeypatch, tmp_path):
    monkeypatch.chdir(_PROJECT_ROOT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = _load("webarena_gmail.yaml")
    cfg.task_source.params["port"] = _free_port()
    cfg.task_source.params["task_suite"] = "no-such-suite"
    with pytest.raises(PreflightError, match="task suite file not found"):
        run_preflight(cfg, output_dir=tmp_path)


def test_port_in_use_helper_detects_listening_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        assert _port_in_use(sock.getsockname()[1])


def test_port_in_use_helper_returns_false_for_free_port():
    # Bind, get port, close, then ask the helper — likely free now.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert not _port_in_use(port)
