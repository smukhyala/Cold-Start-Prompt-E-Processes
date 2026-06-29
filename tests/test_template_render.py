from __future__ import annotations

from cold_start.prompts.axes import load_axes
from cold_start.prompts.catalog import load_arms
from cold_start.prompts.template import render_arm_prompt, render_prompt


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
    # The catalog has been extended over time (paper § 3.7 motivates a broader
    # spread). Pin only a lower bound so adding more arms doesn't break the test.
    assert len(arms) >= 6
    for arm in arms:
        text = render_prompt(arm.vector, axes, template_path)
        assert arm.vector.as_dict(), "sanity"
        # non-trivial content
        assert len(text) > 80


def test_all_initial_arms_have_unique_ids(axes_path, arms_path):
    axes = load_axes(axes_path)
    arms = load_arms(arms_path, axes)
    assert len({a.arm_id for a in arms}) == len(arms)


def test_all_initial_arms_have_unique_vectors(axes_path, arms_path):
    """The arm catalog should not contain duplicate prompt vectors —
    a duplicate would render to byte-identical text and add no exploration
    value while consuming compute."""
    axes = load_axes(axes_path)
    arms = load_arms(arms_path, axes)
    seen: dict[tuple[int, ...], str] = {}
    for arm in arms:
        tup = arm.vector.as_tuple()
        assert tup not in seen, (
            f"arm {arm.arm_id!r} duplicates the vector of {seen[tup]!r}: {tup}"
        )
        seen[tup] = arm.arm_id


def test_gitlab_arm_guidance_renders_through_template():
    axes = load_axes("configs/axes.yaml")
    arms = load_arms("configs/arms_gitlab_strong.yaml", axes)
    domain = {a.arm_id: a for a in arms}["gitlab_domain_expert"]
    text = render_arm_prompt(domain, axes, "configs/template_gitlab.jinja")
    assert "Domain-specific operating guidance:" in text
    assert "Search before creating." in text
    assert "Never delete unless explicitly requested." in text


def test_generic_arm_render_unchanged_with_empty_guidance():
    axes = load_axes("configs/axes.yaml")
    arms = load_arms("configs/arms_gitlab_strong.yaml", axes)
    baseline = {a.arm_id: a for a in arms}["baseline"]
    old_style = render_prompt(baseline.vector, axes, "configs/template.jinja")
    new_style = render_arm_prompt(baseline, axes, "configs/template_gitlab.jinja")
    assert old_style == new_style
