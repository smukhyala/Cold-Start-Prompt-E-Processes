# Repository layout and local artifacts

## Where files belong

| Location | Purpose | Git policy |
| --- | --- | --- |
| `src/cold_start/` | Reusable implementation | Tracked |
| `experiments/` | Study-specific collection and analysis code | Tracked; preserve unfinished local code |
| `scripts/` | Operational utilities and reproducible figure builders | Tracked; preserve unfinished local code |
| `tests/`, `configs/` | Tests and experiment configuration | Tracked |
| `docs/paper/`, `docs/growing_bandits/` | Protocols, amendments, evidence explanations, and research plans | Tracked research record; not manuscript drafts |
| `paper/` | Local manuscript drafts, bibliography, compiled PDFs, publication figures and tables | Ignored; not supplied by a fresh clone |
| `results/`, `data/` | Experimental evidence and study inputs | Ignored by default; selected frozen artifacts are intentionally tracked |
| `logs/`, `reports/` | Local runtime logs and generated reports | Ignored |
| `analysis/adaptive_framework_analysis/` | Older local analysis bundle and figures | Ignored; preserved in place because its scripts and links depend on this path |
| `scratchpad/` | Local exploratory scripts and cached data | Ignored; preserved |
| `tmp/` | Disposable rendering and build intermediates | Ignored |
| `.superpowers/`, local editor state, Python caches | Local tooling state | Ignored |

## Paper workspace

- `paper/main.tex`: current manuscript; `paper/main.pdf`: compiled paper.
- `paper/prompt_search_legacy.tex`: previous manuscript draft.
- `paper/figures/`: current publication figures, generated LaTeX tables, plotted
  data, and `RESULTS_VISUALS.md` with exact artifact provenance.
- `paper/framework_figures/`: existing SWE-bench leaderboard figure.
- `paper/figures/legacy_global_log_e_growth/`: older standalone plots previously
  scattered in the repository-root `figures/` directory.
- `scripts/build_paper_visuals.py`: versionable builder for the current results
  visuals; it reads saved experimental artifacts only.
- `experiments/growing_bandits/empirical/plot_live_stopping_continuation.py`:
  original live-trajectory plotting source.

From the repository root:

```sh
.venv/bin/python scripts/build_paper_visuals.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/main.tex
```

The generator recreates publication artifacts without collecting outcomes or
rerunning statistical analyses. Manuscript sources must already exist locally
to compile the paper. Ignoring `paper/` removes it from future ordinary commits;
it does not erase copies in existing Git history.

## Organization review, 2026-10-17

The implementation, experiment, configuration, and test directories have distinct
roles and do not need a broad relocation. The actionable clutter was local paper
files still tracked, standalone root-level plots, temporary PDF renders, and
incomplete ignore coverage for old exploratory work.

Cleanup keeps `paper/` at its established path so links, plotting defaults, and
historical protocol references remain valid. Its five previously tracked files
are removed from the Git index while remaining on disk. Root-level legacy plots
are moved beneath `paper/figures/`; their local generator's output path follows
the move. The current results-visual builder moves into `scripts/` so reproducible
source is separate from ignored artifacts. Disposable preview PNGs and TeX build
intermediates are removed; the compiled manuscript is retained.

The audit found 319 tracked files under `results/` and 219 under `data/`. These are
not automatically garbage: they include preregistration inputs, outcome tables,
receipts, amendment histories, and manifests. Repeated live-study exports are
referenced by audit hashes and are retained unchanged. No duplicate nonempty
tracked source files were found outside those evidence directories. Four empty
tracked files are Python package markers (`__init__.py`), so they are retained.

Unfinished linguistic-analysis code and tests, the local research/literature
reviews, and the existing deployment-results edit remain untouched. A clean
working tree is not obtained by deleting or hiding unfinished source work.
