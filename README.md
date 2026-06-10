# coldStartPrompts

Sequential, framework-style evaluation of LLM **cold-start prompts** as arms in a multi-armed hypothesis test, using **e-processes** for anytime-valid inference.

Collaboration between Sanjay Mukhyala and Ian Waudby-Smith (Summer–Winter 2026).

## Why

Classical LLM evaluation tests a fixed prompt on a fixed benchmark with a fixed sample size. Real systems are queried interactively and continuously, and the choice of *cold-start* (role framing, expertise, planning / verification style, etc.) meaningfully shapes behavior. We treat each cold-start configuration as an **arm** and ask: which cold-starts work, and how fast can we tell?

The inference backbone is betting-based e-processes (Waudby-Smith & Ramdas 2023) with a SPRUCE-style UCB policy over log-wealth growth rate — so we get anytime-valid type-I error control and a principled adaptive sampler in the same object.

## Paper

The formal framework — six-axis prompt vectors `v_a`, deterministic rendering `g(v_a)`, per-arm upward-capital e-processes `E_{a,t} = ∏(1 + λ_{a,i}(Y_i − μ_0))`, linear convex mixture global null `E_t = Σ_a w_a E_{a,t}`, hypotheses `H_0: μ_a ≤ μ_0` vs. `H_1: ∃a, μ_a > μ_0`, stopping time `τ_α = inf{t : E_t ≥ 1/α}`, and the four allocation policies (uniform, ε-greedy, Thompson, UCB, SPRUCE) — is laid out in:

> Mukhyala & Waudby-Smith. *Evaluating Cold-Start Prompt Strategies with E-Processes.* UC Berkeley, June 2026.

Code in this repo maps directly onto § 3 of the paper:

| Paper concept | Code |
|---|---|
| 6-axis prompt vector `v_a` | `src/cold_start/types.py:PromptVector` |
| Rendering function `g(v_a)` | `src/cold_start/prompts/template.py:render_prompt` |
| Distance `d(a,b)` (§ 3.2) | `src/cold_start/prompts/vector.py:distance` |
| One-sided e-process `E_{a,t}` (§ 3.4) | `src/cold_start/inference/upward_capital.py` |
| Linear mixture global null (§ 3.4) | `src/cold_start/inference/global_null.py` (`combine="linear_mixture"`) |
| UCB allocation (§ 3.5, ref [2]) | `src/cold_start/policies/ucb.py` |
| Stopping time `τ_α` (§ 3.6, § 3.8) | `src/cold_start/analysis/summary.py` |
| Pipeline (§ 3.7) | `src/cold_start/runner/orchestrator.py` |

`configs/pilot_toy_upward.yaml` is a runnable instantiation of the paper's exact construction on the synthetic ToyEnv (no network/model required).

## Install

```bash
make setup                 # creates .venv and pip-installs -e ".[dev]"
cp .env.example .env       # add ANTHROPIC_API_KEY
```

## Run a toy pilot (no network)

```bash
make pilot                                                  # configs/pilot_toy.yaml (SPRUCE + hedged-capital)
cold-start-run --config configs/pilot_toy_upward.yaml       # paper-faithful: upward_capital + linear_mixture + UCB
cold-start-report --log logs/<run>.jsonl                    # writes summary.md (μ_a, Δ_a, τ_α, distance matrix) + plots
```

## Compare allocations on the same task stream

```bash
cold-start-compare --base configs/pilot_toy_upward.yaml \
                   --policies uniform,epsilon_greedy,thompson,ucb,spruce
# → reports/pilot_toy_upward_comparison/comparison.md
```

## Project structure

See `docs/architecture.md` for the component map. Everything pluggable sits behind a registry; experiments are YAML-configured.

- `src/cold_start/prompts/` — prompt-vector → system-prompt rendering
- `src/cold_start/inference/` — e-processes (`hedged_capital`, `upward_capital`), confidence sequences, global null
- `src/cold_start/policies/` — `uniform` / `epsilon_greedy` / `thompson` / `ucb` / `spruce`
- `src/cold_start/rewards/` — `binary`, `continuous` (uses `RunResult.partial_score`)
- `src/cold_start/tasks/` — environment adapters (ToyEnv, WebArena-Infinity)
- `src/cold_start/models/` — Anthropic client with prompt caching; pluggable
- `src/cold_start/runner/` — orchestrator and tool-using agent loop
- `src/cold_start/analysis/` — JSONL loaders, plots, summary reports
- `configs/` — axes, arms (12 in the catalog), templates, experiment specs

## References

- Mukhyala, Waudby-Smith. *Evaluating Cold-Start Prompt Strategies with E-Processes.* UC Berkeley, 2026.
- Waudby-Smith, Ramdas. *Estimating means of bounded random variables by betting.* JRSS-B 2023.
- Sandoval, Waudby-Smith, Jordan. *Multi-Armed Sequential Hypothesis Testing by Betting* (SPRUCE). 2026.
- Ramdas, Grünwald, Vovk, Shafer. *Hypothesis Testing with E-Values.*
- Auer, Cesa-Bianchi, Fischer. *Finite-time analysis of the multiarmed bandit problem.* Machine Learning 47, 2002.
