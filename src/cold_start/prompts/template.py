"""Deterministic prompt rendering from a PromptVector.

Byte-identical output for identical vectors is a hard invariant — it's what
lets Anthropic prompt caching hit on the system block across every call that
uses the same arm.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from cold_start.prompts.axes import AxesSpec
from cold_start.types import Arm, PromptVector


def _env(template_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(
    vec: PromptVector,
    axes: AxesSpec,
    template_path: str | Path,
    *,
    prompt_guidance: str = "",
    arm_id: str = "",
    arm_name: str = "",
) -> str:
    template_path = Path(template_path)
    env = _env(template_path.parent)
    template = env.get_template(template_path.name)
    context = {
        "v": vec.as_dict(),
        "labels": {name: axes[name].label(val) for name, val in vec.as_dict().items()},
        "arm": {
            "id": arm_id,
            "name": arm_name,
            "prompt_guidance": prompt_guidance,
        },
    }
    return template.render(**context)


def render_arm_prompt(arm: Arm, axes: AxesSpec, template_path: str | Path) -> str:
    return render_prompt(
        arm.vector,
        axes,
        template_path,
        prompt_guidance=arm.prompt_guidance,
        arm_id=arm.arm_id,
        arm_name=arm.name,
    )
