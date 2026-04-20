# coldStartPrompts

Sequential, framework-style evaluation of LLM **cold-start prompts** as arms in a multi-armed hypothesis test, using **e-processes** for anytime-valid inference.

Collaboration between Sanjay Mukhyala and Ian Waudby-Smith (Summer–Winter 2026).

## Why

Classical LLM evaluation tests a fixed prompt on a fixed benchmark with a fixed sample size. Real systems are queried interactively and continuously, and the choice of *cold-start* (role framing, expertise, planning / verification style, etc.) meaningfully shapes behavior. We treat each cold-start configuration as an **arm** and ask: which cold-starts work, and how fast can we tell?

The inference backbone is betting-based e-processes (Waudby-Smith & Ramdas 2023) with a SPRUCE-style UCB policy over log-wealth growth rate — so we get anytime-valid type-I error control and a principled adaptive sampler in the same object.

## Install

```bash
make setup                 # creates .venv and pip-installs -e ".[dev]"
cp .env.example .env       # add ANTHROPIC_API_KEY
```

## Run a toy pilot (no network)

```bash
make pilot                 # configs/pilot_toy.yaml, synthetic Bernoulli arms
```

## Project structure

See `docs/architecture.md` for the component map. Everything pluggable sits behind a registry; experiments are YAML-configured.

- `src/cold_start/prompts/` — prompt-vector → system-prompt rendering
- `src/cold_start/inference/` — e-processes, confidence sequences
- `src/cold_start/policies/` — uniform / ε-greedy / Thompson / SPRUCE
- `src/cold_start/tasks/` — environment adapters (ToyEnv, WebArena-Infinity stub)
- `src/cold_start/models/` — Anthropic client with prompt caching; pluggable
- `src/cold_start/runner/` — orchestrator and tool-using agent loop
- `src/cold_start/analysis/` — JSONL loaders, plots, summary reports
- `configs/` — axes, arms, templates, experiment specs

## References

- Waudby-Smith, Ramdas. *Estimating means of bounded random variables by betting.* JRSS-B 2023.
- Waudby-Smith, Wu, Ramdas. *Multi-Armed Sequential Hypothesis Testing by Betting* (SPRUCE).
- Ramdas, Grünwald, Vovk, Shafer. *Hypothesis Testing with E-Values.*
