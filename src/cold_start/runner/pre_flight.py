"""Pre-flight checks for `cold-start-run`.

Fail fast on the conditions that are easy to detect before any task runs and
expensive to discover halfway through one:

- API key set (when the configured model or WebArena LLM provider needs one)
- output directory writable
- (webarena) port free
- (webarena) sibling repo present, importable
- (webarena) configured app subdirectory exists, task suite file exists

The goal is "if these pass, the run will reach t=1." Per-task failures
(server crashes, browser timeouts, model 5xx) are not in scope here — those
have to be handled inline.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

from cold_start.config import RunConfig

logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Raised when a pre-flight condition fails. Message is user-facing."""


def run_preflight(cfg: RunConfig, *, output_dir: str | Path | None = None) -> None:
    """Validate everything we can validate before the orchestrator starts.

    Raises PreflightError on the first failure with a message that names the
    file/key/port involved.
    """
    out_dir = Path(output_dir) if output_dir else Path(cfg.trial.output_dir)
    _check_output_dir_writable(out_dir)
    _check_anthropic_key(cfg)

    if cfg.task_source.type == "webarena":
        _check_webarena_prereqs(cfg)


def _check_output_dir_writable(out_dir: Path) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreflightError(f"output dir {out_dir} not creatable: {exc}") from exc
    probe = out_dir / ".preflight_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise PreflightError(f"output dir {out_dir} not writable: {exc}") from exc


def _check_anthropic_key(cfg: RunConfig) -> None:
    """Only required when a path through the run actually calls Anthropic.

    Toy / mock model + non-webarena task source can run offline. The WebArena
    adapter needs the provider-specific key for browser-use's LLM client; the
    same is true for the orchestrator's text-only runner when
    `model.type == 'anthropic'`.
    """
    if cfg.model.type == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise PreflightError(
            "ANTHROPIC_API_KEY not set. Put it in .env or export it before running. "
            "(Anthropic is reached via the model: block.)"
        )
    if cfg.task_source.type != "webarena":
        return

    params = cfg.task_source.params or {}
    provider = str(params.get("llm_provider", "anthropic")).lower()
    if provider == "anthropic":
        env_key = "ANTHROPIC_API_KEY"
    elif provider == "openai":
        env_key = "OPENAI_API_KEY"
    else:
        raise PreflightError(
            f"unsupported webarena llm_provider={provider!r}; use 'anthropic' or 'openai'."
        )
    if not os.environ.get(env_key):
        raise PreflightError(
            f"{env_key} not set. Put it in .env or export it before running. "
            f"(WebArena browser-use provider is {provider!r}.)"
        )


def _check_webarena_prereqs(cfg: RunConfig) -> None:
    from cold_start.tasks.webarena import _webarena_root  # local import: circular-safe

    try:
        root = _webarena_root()
    except RuntimeError as exc:
        raise PreflightError(str(exc)) from exc

    params = cfg.task_source.params or {}
    port = int(params.get("port", 8001))
    if _port_in_use(port):
        raise PreflightError(
            f"port {port} is already in use. Free it with "
            f"`lsof -ti :{port} | xargs kill` and rerun. A leftover webarena "
            "server from a prior aborted run is the usual culprit."
        )

    web_app_rel = params.get("web_app", "apps/gmail")
    app_dir = root / web_app_rel
    if not app_dir.exists():
        raise PreflightError(
            f"webarena app dir not found: {app_dir}. Check task_source.params.web_app."
        )
    suite_name = params.get("task_suite", "real-tasks")
    suite_file = app_dir / f"{suite_name}.json"
    if not suite_file.exists():
        raise PreflightError(
            f"webarena task suite file not found: {suite_file}. "
            f"Check task_source.params.task_suite (current: {suite_name!r})."
        )

    try:
        import browser_use  # noqa: F401
    except ImportError as exc:
        raise PreflightError(
            "browser-use is not installed. Run `pip install -e .[webarena]` "
            "to install the WebArena dependency group."
        ) from exc


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
            return True
        except (ConnectionRefusedError, OSError):
            return False
