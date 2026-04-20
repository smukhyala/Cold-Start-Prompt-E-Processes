from __future__ import annotations

from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms
from cold_start.prompts.template import render_prompt


def test_render_is_deterministic(axes_path, arms_path, template_path):
    axes = load_axes(axes_path)
    arms = load_arms(arms_path, axes)
    for arm in arms:
        a = render_prompt(arm.vector, axes, template_path)
        b = render_prompt(arm.vector, axes, template_path)
        assert a == b, f"non-deterministic render for {arm.arm_id}"


def test_all_initial_arms_render(axes_path, arms_path, template_path):
    axes = load_axes(axes_path)
    arms = load_arms(arms_path, axes)
    assert len(arms) == 6
    for arm in arms:
        text = render_prompt(arm.vector, axes, template_path)
        assert arm.vector.as_dict(), "sanity"
        # non-trivial content
        assert len(text) > 80
