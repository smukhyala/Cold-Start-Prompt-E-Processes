"""Sanity-test environment: easy, unambiguously graded tasks against a real model.

Purpose is to exercise every wire in the stack end-to-end — AnthropicClient →
agent loop → arm system prompt → reward → hedged-capital e-process → global
e-process — on a single short run, before investing in WebArena-Infinity.

Tasks are a small hardcoded bank of arithmetic / logic / trivia items with
exact expected answers. The model is asked to reply with an `ANSWER: <x>` line;
the grader extracts that line and does a lenient string/numeric match.
"""

from __future__ import annotations

import re

from cold_start.prompts.axes import load_axes
from cold_start.prompts.template import render_arm_prompt
from cold_start.registry import register
from cold_start.tasks.base import AgentRunner, EnvironmentAdapter
from cold_start.types import Arm, RunResult, Task

_ANSWER_DIRECTIVE = (
    "\n\nRespond with your final answer on the last line, prefixed by "
    '"ANSWER: " and nothing else on that line — no units, no explanation.'
)

_ANSWER_RE = re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

_TASK_BANK: list[tuple[str, str]] = [
    ("What is 12 + 7?", "19"),
    ("What is 8 times 6?", "48"),
    ("What is 100 - 37?", "63"),
    ("What is 144 divided by 12?", "12"),
    ("What is 2 to the power of 5?", "32"),
    ("Which is larger, 0.7 or 0.65? Answer with the number.", "0.7"),
    ("What is the next number in the sequence 3, 6, 9, 12?", "15"),
    ("What is the next number in the sequence 1, 4, 9, 16?", "25"),
    ("Is 91 a prime number? Answer yes or no.", "no"),
    ("Is 29 a prime number? Answer yes or no.", "yes"),
    ("How many letters are in the word 'sequence'?", "8"),
    ("What is the capital of France?", "paris"),
    (
        "If all cats are mammals and Felix is a cat, is Felix a mammal? "
        "Answer yes or no.",
        "yes",
    ),
    (
        "True or false: the sum of two odd numbers is always even. "
        "Answer true or false.",
        "true",
    ),
    (
        "Which of these does not belong: dog, cat, fish, car? Answer with one word.",
        "car",
    ),
    ("What is 15% of 200?", "30"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the smallest prime number greater than 7?", "11"),
    ("What is (3 + 4) * 2?", "14"),
    ("How many vowels are in the word 'education'?", "5"),
]


@register("sanity", kind="task_source")
class SanityEnv(EnvironmentAdapter):
    def __init__(self, axes_path: str, template_path: str):
        self._axes = load_axes(axes_path)
        self._template_path = template_path
        self._prompt_cache: dict[str, str] = {}

    def reset(self, seed: int) -> None:
        return None

    def sample_task(self, t: int) -> Task:
        idx = (t - 1) % len(_TASK_BANK)
        question, expected = _TASK_BANK[idx]
        return Task(
            task_id=f"sanity-{idx:03d}",
            payload={"instruction": question + _ANSWER_DIRECTIVE},
            metadata={"expected": expected, "bank_index": idx},
        )

    def run_arm(
        self,
        arm: Arm,
        task: Task,
        runner: AgentRunner,
        max_steps: int,
    ) -> RunResult:
        system_prompt = self._prompt_cache.get(arm.arm_id)
        if system_prompt is None:
            system_prompt = render_arm_prompt(arm, self._axes, self._template_path)
            self._prompt_cache[arm.arm_id] = system_prompt

        raw = runner(system_prompt=system_prompt, task=task, max_steps=max_steps)
        final_text = str(raw.trace.get("final_text", ""))
        expected = str(task.metadata.get("expected", "")).strip().lower()
        success = _grade(final_text, expected)

        return RunResult(
            success=success,
            reward=float(success),
            steps=raw.steps,
            wallclock_s=raw.wallclock_s,
            trace={
                "env": "sanity",
                "final_text": final_text[:500],
                "expected": expected,
                "extracted": _extract_answer(final_text),
            },
            tokens=raw.tokens,
        )


def _extract_answer(text: str) -> str:
    if not text:
        return ""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: last non-empty line.
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return s
    return ""


def _grade(text: str, expected: str) -> bool:
    got = _extract_answer(text).lower().strip().rstrip(".")
    if not got:
        return False
    if got == expected:
        return True
    # Numeric tolerance: try to parse both sides as floats.
    try:
        return abs(float(got) - float(expected)) < 1e-6
    except ValueError:
        pass
    # Word-match leniency: expected is a substring of a short response.
    if len(got) <= 40 and expected in got:
        return True
    return False
