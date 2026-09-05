# Experiments

Every script in this tree is a runnable reproduction of one part of the
paper. See **[docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md)** for the full
table → script → result-file mapping — start there.

Layout:

- `rq1_defense_effectiveness/` — heuristic defense ablation, KB-density
  effect across datasets, cross-model evaluation, extended 350-query
  validation.
- `rq2_semantic_and_adaptive/` — semantic defense layers (L0/L1b/L1c),
  adaptive (DeepSeek-V3-generated) attacks, LLM-judge validation.
- `rq3_retrieval_design/` — top-k sensitivity, retriever strategy
  (Dense/BM25/Hybrid), AHR@k / CASR decomposition.
- `rq4_composition_agentic_web/` — non-monotonic defense composition in an
  agentic tool-calling setting, and the Web-like content-format evaluation.
- `validation_and_statistics/` — bootstrap CIs, McNemar significance tests,
  judge κ agreement, SAU correlation with human judgment, canary leakage,
  retrieval-optimized attack, latency measurement.
- `qualitative_and_figures/` — qualitative examples, human annotation of
  judge disagreements, the full attack matrix, and Figure 2.
- `data_preparation/` — one-off scripts that generated the adversarial
  payloads and auxiliary datasets consumed by the ablations above.
- `exploratory_and_legacy/` — superseded, partial, or non-canonical runs,
  kept for provenance. See its own `README.md`. **Not** part of the
  reproduction path.

## Running a script

All scripts assume the repository root as the working directory (they read
relative paths like `data/...` and write to `results/...` / `./chroma_*`):

```bash
pip install -e .
cp .env.example .env   # fill in your API keys
python experiments/rq1_defense_effectiveness/heuristic_ablation.py
```

A handful of scripts (mostly in `exploratory_and_legacy/`, plus a few in
`rq1_defense_effectiveness/`) import from another experiment script rather
than from the `ragipi` package — this mirrors how the original research
code was written (e.g. the MITRE and TriviaQA ablations reuse the NQ
adversarial-document builder). Those must be run as modules from the
repository root instead of as plain scripts:

```bash
python -m experiments.rq1_defense_effectiveness.kb_density_mitre
```

Plain `python path/to/script.py` also works for these as long as the
repository root is on `PYTHONPATH` (`PYTHONPATH=. python experiments/...`).
