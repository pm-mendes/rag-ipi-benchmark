# Changelog

## 1.1.0 — Camera-ready revision (2026-09)

Full numeric and methodological audit of the accepted paper ahead of the
6-page WI-IAT 2026 short-paper limit, plus real independent human
validation of the ASR judge (superseding the earlier synthetic-text
validation):

- **Human annotation**: two annotators independently labeled 188 real
  pipeline outputs blind to configuration, per an O1–O4 rubric
  (`docs/ANNOTATION_PROTOCOL.md`); disagreements were jointly adjudicated
  (`results/annotation_disagreements_for_adjudication.csv`). Scored
  against this ground truth (`results/annotation_scored.json`), the
  marker-based judge used for every ASR figure in the paper reaches
  precision/recall/F1 = 92.9%/92.9%/0.929 — an independent semantic
  O1–O4 judge built to check circularity concerns scores far worse
  (recall=0%), reported as-is.
- **Statistical corrections**: the seed-reproducibility experiment was
  running on a single hardcoded seed instead of the reported 3
  (`experiments/rq1_defense_effectiveness/seed_reproducibility.py`,
  `results/statistical_results_RECOVERED_3seeds.json`,
  `results/table3_pooled_3seed_exact_ci.json`); a significance test
  reported as McNemar's had a fabricated per-query pairing, replaced by
  Fisher's exact test on the correct unpaired design
  (`results/unpaired_significance_corrected.json`); Clopper-Pearson exact
  intervals added for rare-event configurations
  (`results/rare_event_and_effect_sizes.json`).
- **Retrieval-optimized attacks**: a mislabeled experiment config had
  the reported "20% ASR under All heuristic" actually computed with no
  defense applied; corrected with a genuine paired re-run
  (`results/retrieval_optimized_ahr_casr.json`), plus a new fair
  Dense/BM25/Hybrid retriever-targeted camouflage comparison
  (`results/retriever_optimized_attacks.json`).
- **New pilot experiments**: L2 causal-isolation ablation
  (`results/l2_isolation_ablation.json`), L1c anomaly-threshold
  calibration on a held-out split (`results/l1c_threshold_calibration.json`),
  L0 Unicode-normalization fidelity on legitimate non-English/code/math
  content (`results/l0_fidelity_results.json`), a benign-hard false-positive
  probe (`results/benign_hard_fpr_results.json`), an extended
  multi-format canary exfiltration probe
  (`results/canary_extended_results.json`), a prompt-assembly-format
  ablation (`results/prompt_assembly_ablation.json`), and NQ EM/F1 against
  reference answers (`results/nq_em_f1_results.json`).
- **Consolidated values**: `results/MASTER_VALUES.json` maps every number
  reported in the paper to its source result file and field.
- **Paper**: `paper/main.tex` rewritten against all of the above and cut
  from the original draft to the required 6 pages; content moved out for
  space (full per-attack tables, latency table, fairness/accountability
  discussion, agentic and web-like extensions) is preserved in full in
  the new `docs/EXTENDED_RESULTS.md`, which the paper points to.

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
