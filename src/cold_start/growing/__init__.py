"""Growing-arm bandits: SEARCH (draw a new arm) vs REFINE (re-evaluate an existing one).

This subpackage is deliberately decoupled from the LLM/WebArena side of `cold_start`.
It imports only `cold_start.registry` (and, in tests, the scalar `cold_start.inference`
reference implementations), so it can be lifted into its own repository unchanged.
"""

from __future__ import annotations
