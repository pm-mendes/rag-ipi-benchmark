# Changelog

## 1.0.0 — Camera-ready release (2026-09)

Public release accompanying the accepted WI-IAT 2026 paper. Reorganized from
the original flat research-script layout into a packaged, tested,
documented repository:

- **Package**: extracted the shared pipeline/defense/metric/attack/dataset
  code into an installable `ragipi` package under `src/`
  (`pip install -e .`). No experiment logic or numeric result was changed —
  see `docs/ARCHITECTURE.md` for what moved where, and
  `git log`-equivalent notes in `docs/EXPERIMENTS.md` for provenance.
- **Terminology**: the utility metric shipped during development as
  "Answer Relevance (AR)" (`evaluate_ar.py`) was renamed to **SAU**
  (Security-Oriented Answer Utility, `ragipi.metrics.sau`) to match the
  paper's final terminology. `compute_ar` → `compute_sau`. The formula and
  every reported number are unchanged.
- **Experiments**: the ~45 root-level `ablation_*.py` / `attack_*.py` /
  `generate_*.py` scripts were organized into `experiments/<rq>/`,
  grouped by the research question or paper section they support, with a
  full table → script → result-file mapping in `docs/EXPERIMENTS.md`.
  Scripts that were superseded, incomplete, or not cited in the accepted
  paper were kept, unmodified in logic, under
  `experiments/exploratory_and_legacy/` rather than deleted — see that
  directory's `README.md` for the specific reason each one is there.
  A single shared attack registry (`ragipi.attacks.catalog`) was extracted
  from the heuristic-ablation script to remove the previous
  experiment-imports-experiment coupling.
- **Tests & CI**: added a `pytest` suite (`tests/`) covering the pure
  logic of the metrics, defense layers, attack catalog, and statistics
  helpers, plus a GitHub Actions workflow (lint + tests + byte-compile) —
  none of it requires API keys, network access, or Ollama.
- **Docs**: added `docs/ARCHITECTURE.md`, `docs/METRICS.md`,
  `docs/THREAT_MODEL.md`, `docs/REPRODUCIBILITY.md`, and
  `docs/EXPERIMENTS.md`; rewrote the root `README.md` against the accepted
  paper (previous drafts referenced an earlier version of the metric set
  and table numbering).
- **Housekeeping**: dropped committed vector-store caches (`chroma_*/`),
  virtualenvs, and `__pycache__` from version control (already git-ignored
  in the source repository; carried the same `.gitignore` policy forward).
  Removed the ad hoc `test_env.py` smoke script (explicitly marked
  "to delete after validation" in its own docstring).

## Pre-1.0.0

Development history of the underlying research code (not separately
versioned) — see the paper's Sections IV–VII for the experimental design
this repository implements.
