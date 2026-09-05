# Architecture

## Package layout

```
src/ragipi/               Installable library (pip install -e .)
├── pipeline.py            Core RAG pipeline: load docs, chunk, embed,
│                          retrieve (Chroma), assemble prompt, generate.
├── llm_factory.py          Multi-provider LLM factory (OpenAI / Groq /
│                          DeepSeek / NVIDIA-hosted Llama 3.1).
├── attacks/
│   ├── catalog.py          CANONICAL_ATTACKS — the 6-attack registry,
│   │                       single source of truth (see below).
│   └── runner.py           SUCCESS_INDICATORS + is_attack_successful
│                           (lexical ASR detector) + run_single_attack.
├── defenses/                One module per layer in Figure 1 of the paper:
│   ├── content_filter.py    L1 — 14 regex patterns, pre-generation.
│   ├── context_isolator.py  L2 — isolation-aware system prompt.
│   ├── output_verifier.py   L3 — 13 compromise-marker post-check.
│   └── semantic_defense.py  L0 (Unicode NFKC) + L1b (LLaMA 3 8B judge)
│                            + L1c (embedding anomaly score).
├── metrics/
│   ├── core.py              ASR, RD, CI (`evaluate.py` in the original
│   │                        research code).
│   └── sau.py                SAU, the novel Security-Oriented Answer
│                             Utility metric (`evaluate_ar.py` / "AR" in
│                             the original research code — renamed to
│                             match the paper).
├── judge/
│   └── cascade.py           LLM-judge cascade for advanced/adaptive ASR.
├── agents/
│   ├── pipeline.py           Minimal tool-calling RAG agent.
│   └── tools.py               search_kb / summarize_doc (authorized),
│                              send_email / export_context (restricted,
│                              simulated — see module docstring).
├── datasets/                 One loader per corpus: nq, mitre, hotpotqa,
│                             triviaqa, weblike.
└── stats/                    (reserved — the statistics helpers currently
                              live in experiments/validation_and_statistics/,
                              see note below)

experiments/                 Runnable reproductions, one script per paper
                              table/section. See docs/EXPERIMENTS.md.

data/, results/              Inputs and (committed) outputs of every
                              experiment above.

paper/                       LaTeX source and camera-ready PDF.
```

**Why `stats/` is empty**: the bootstrap-CI, McNemar, and Cohen's-κ
computations are implemented as standalone analysis scripts
(`experiments/validation_and_statistics/*.py`) rather than as a reusable
library module, because — unlike the defenses or metrics — nothing else in
the codebase imports them; they are read-once-write-once statistical
analyses over the JSON files in `results/`. They are still unit-tested (see
`tests/test_stats.py`), just not exposed as a package API. The directory is
kept as a placeholder in case that changes.

## Data flow (Figure 1 in the paper)

```
Query → Retriever (BM25/Dense/Hybrid)
           │
      [L0] Unicode normalization  (ragipi.defenses.semantic_defense)
           │
      [L1] Content filter          (ragipi.defenses.content_filter)
       ├── [L1b] LLM-judge (LLaMA 3 8B)   (ragipi.defenses.semantic_defense)
       └── [L1c] Embedding anomaly δ>0.05 (ragipi.defenses.semantic_defense)
           │
      [L2] Context isolator        (ragipi.defenses.context_isolator)
           │
      Prompt assembler → LLM generator (ragipi.pipeline / ragipi.llm_factory)
           │
      [L3] Output verifier          (ragipi.defenses.output_verifier)
           │
      Final answer
```

Heuristic layers (L1, L2, L3) are the ones ablated first (RQ1); semantic
layers (L0, L1b, L1c) are layered on top for RQ2. Each experiment script
composes these modules directly rather than through a single configurable
"defense pipeline" object — this mirrors how the ablations were actually
run (one script per configuration matrix) and keeps each experiment
self-contained and independently auditable.

## Why the attack catalog was extracted

Several experiment scripts (`retriever_strategy.py`, `topk_sensitivity.py`,
`ahr_casr_decomposition.py`, `judge_validation_200samples.py`,
`retrieval_optimized_attack.py`) originally imported the canonical
6-attack list directly from the heuristic-ablation driver script
(`ablation_v2.py`, i.e. `experiments/rq1_defense_effectiveness/heuristic_ablation.py`).
That created a fragile dependency of one experiment on another. The list
was extracted verbatim into `ragipi.attacks.catalog.CANONICAL_ATTACKS`, and
every consumer — including `heuristic_ablation.py` itself — now imports
from there. No attack definition, file path, or downstream numeric result
was changed by this extraction.

## Import layout note

A handful of scripts (mostly `kb_density_mitre.py`, `kb_density_hotpotqa.py`,
`kb_density_triviaqa.py`, `cross_model_nq.py`, and two files in
`exploratory_and_legacy/`) import shared NQ attack-construction helpers
(`ATTACKS_NQ`, `create_nq_adversarial_docs`, `is_nq_attack_successful`)
directly from `experiments/rq1_defense_effectiveness/kb_density_nq.py`
rather than from the `ragipi` package. This mirrors the original research
code's structure (`ablation_nq.py` was reused as a mini-library by several
other ablations) and was left as-is rather than further refactored, to
avoid touching logic beyond import-path fixes. See `experiments/README.md`
for how to run these.
