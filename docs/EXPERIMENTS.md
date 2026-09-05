# Experiment map: paper tables → scripts → result files

This document is the single source of truth linking every claim in the
accepted paper (*"Evaluating Indirect Prompt Injection in RAG Systems:
Defense-in-Depth, Retrieval Effects, and Security–Utility Trade-offs"*,
WI-IAT 2026) to the exact script that produced it and the exact result file
that stores its output. Every number below was cross-checked against the
committed JSON in `results/` before this repository was assembled.

All commands are run from the repository root, after `pip install -e .`
(see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for environment setup, API
keys, and dataset downloads).

## RQ1 — Defense effectiveness against canonical IPI

| Paper artifact | Script | Result file |
|---|---|---|
| Table "Heuristic ablation" (`tab:heuristic`) — Custom 10q, GPT-3.5-turbo | `python experiments/rq1_defense_effectiveness/heuristic_ablation.py` | `results/ablation_v2_results.json` |
| Seed reproducibility (seeds {42,123,456}, mean±std in `tab:heuristic`) | `python experiments/rq1_defense_effectiveness/seed_reproducibility.py` | `results/statistical_results.json` |
| Table "KB density effect" (`tab:kbdensity`) — SQuAD/NQ row | `python experiments/rq1_defense_effectiveness/kb_density_nq.py` | `results/ablation_nq_results.json` |
| `tab:kbdensity` — MITRE ATT&CK row | `python experiments/rq1_defense_effectiveness/kb_density_mitre.py` | `results/ablation_mitre_results.json` |
| `tab:kbdensity` — HotpotQA row | `python experiments/rq1_defense_effectiveness/kb_density_hotpotqa.py` | `results/ablation_hotpotqa_results.json` |
| `tab:kbdensity` — TriviaQA row | `python experiments/rq1_defense_effectiveness/kb_density_triviaqa.py` | `results/ablation_triviaqa_results.json` |
| `tab:kbdensity` — Custom (8 docs) row | `python experiments/rq1_defense_effectiveness/kb_density_custom_8docs.py` | `results/ablation_custom_full_results.json` |
| Table "Cross-dataset/model results" (`tab:kbdensity2`) — Custom (10q) columns | `python experiments/rq1_defense_effectiveness/cross_model_custom.py` | `results/ablation_multimodel_results.json` (+ `results/ablation_multimodel_final.txt` for the OpenAI/DeepSeek runs of the same sweep) |
| `tab:kbdensity2` — NQ (50q) columns | `python experiments/rq1_defense_effectiveness/cross_model_nq.py` | `results/ablation_multimodel_nq_results.json` |
| "Extended 350-Query Validation" (100q + 250q = 350 unique queries, NQ+HotpotQA+TriviaQA) | `python experiments/rq1_defense_effectiveness/extended_validation_100_250q.py --output results/ablation_100q_results.json` (run once at 100q, once at 250q; it is the same parametrized script) | `results/ablation_100q_results.json`, `results/ablation_100q_summary.json`, `results/ablation_250q_results.json`, `results/ablation_250q_summary.json` |

## RQ2 — Semantic obfuscation, adaptive attacks, and semantic defenses

| Paper artifact | Script | Result file |
|---|---|---|
| Table "Semantic defense ablation" (`tab:semantic`) | `python experiments/rq2_semantic_and_adaptive/semantic_defense_ablation.py` | `results/semantic_defense_results.json` |
| Table "Adaptive attack evaluation" (`tab:adaptive`) — Custom 30q columns | `python experiments/rq2_semantic_and_adaptive/adaptive_attacks_custom.py` | `results/adaptive_semantic_results.json` |
| `tab:adaptive` — MITRE 30q columns | `python experiments/rq2_semantic_and_adaptive/adaptive_attacks_mitre.py` | `results/adaptive_mitre_semantic_results.json` |
| LLM-judge validation (κ=0.550 LLaMA / κ=0.200 DeepSeek, 200 stratified samples, Section "Results") | `python experiments/rq2_semantic_and_adaptive/judge_validation_200samples.py` | `results/judge_validation_extended.json` |
| LLM-judge pilot validation (30 samples, superseded by the 200-sample run above; kept for the development history of the judge protocol) | `python experiments/rq2_semantic_and_adaptive/judge_validation_pilot_30samples.py` | `results/llm_judge_results.json` |

## RQ3 — Retrieval design as a security parameter

| Paper artifact | Script | Result file |
|---|---|---|
| Table "Impact of top-k" (`tab:topk`) | `python experiments/rq3_retrieval_design/topk_sensitivity.py` | `results/ablation_topk_results.json` |
| Table "Retriever strategy ablation" (`tab:retriever`) | `python experiments/rq3_retrieval_design/retriever_strategy.py` | `results/ablation_retriever_results.json` |
| Table "AHR@k and CASR by attack family" (`tab:ahr`) | `python experiments/rq3_retrieval_design/ahr_casr_decomposition.py` | `results/ahr_casr_results.json` |

## RQ4 — Composition, agentic tool-use, and Web-like formats

| Paper artifact | Script | Result file |
|---|---|---|
| Table "Agentic IPI evaluation" (`tab:agentic`) — TMR_vol (`tool_choice=auto`) | `python experiments/rq4_composition_agentic_web/agentic_voluntary_tmr.py` | `results/ablation_agent_results.json` |
| `tab:agentic` — PCR (forced tool call) | `python experiments/rq4_composition_agentic_web/agentic_forced_pcr.py` | `results/ablation_agent_forced_results.json`, `results/ablation_agent_forced_deepseek_results.json` |
| "Web-like IPI Evaluation" section (HTML/Markdown/email/forum/meta-tag injection, 300 evaluations) | `python experiments/rq4_composition_agentic_web/weblike_evaluation.py` | `results/ablation_weblike_results.json` |

## Validation, statistics, and supplementary attacks (Discussion / Limitations)

| Paper artifact | Script | Result file |
|---|---|---|
| Bootstrap 95% CIs (reported throughout `tab:heuristic`, RQ1 extended finding, etc.) | `python experiments/validation_and_statistics/bootstrap_confidence_intervals.py` | `results/bootstrap_ci_results.json` |
| McNemar significance tests (Dense vs. BM25 χ²=7.11, p=0.008; L1b vs. Semantic full χ²=0.500, p=0.480) | `python experiments/validation_and_statistics/significance_tests.py` | `results/statistical_tests_results.json` |
| Judge κ agreement (Table in Section "Results", per-category breakdown) | `python experiments/validation_and_statistics/judge_kappa_agreement.py` | printed to stdout, derived from `results/judge_validation_extended.json` |
| SAU correlation with human relevance (r=0.921) and BERTScore (r=0.639) — Limitations | `python experiments/validation_and_statistics/sau_human_correlation.py` | `results/ar_correlation_results.json` |
| Canary leakage probe (CLR=0%, token `SYS-CANARY-7F3A`) — Limitations, "Exfiltration" | `python experiments/validation_and_statistics/canary_leakage_probe.py` | `results/canary_leakage_results.json` |
| Retrieval-optimized attack (sim=0.689 vs. 0.162 generic, Δ=+0.528, ASR→20%) — Limitations, "Retrieval-optimized attacks" | `python experiments/validation_and_statistics/retrieval_optimized_attack.py` | `results/attack_retrieval_optimized.json` |
| Table "Defense layer latency" (`tab:latency`) | `python experiments/validation_and_statistics/latency_measurement.py` | `results/latency_results.txt` |

## Qualitative analysis and figures

| Purpose | Script | Output |
|---|---|---|
| Qualitative response examples (custom KB) | `experiments/qualitative_and_figures/generate_qualitative_examples.py` | `results/qualitative_examples.json` |
| Qualitative response examples (MITRE KB) | `experiments/qualitative_and_figures/generate_qualitative_mitre.py` | `results/qualitative_examples_mitre.json` |
| Human annotation of the 45 hardest judge disagreements | `experiments/qualitative_and_figures/annotate_disagreements.py` | `results/human_annotation_disagreements.json` |
| Retrieval-optimized attack similarity + false-positive extraction | `experiments/qualitative_and_figures/extract_retrieval_opt_and_fp.py` | `results/retrieval_opt_and_fp_results.json` |
| Figure 2 (heuristic ablation bar charts) | `experiments/qualitative_and_figures/generate_figure2.py` | `results/figure2a_ablation.{png,pdf}`, `results/figure2b_ablation.{png,pdf}` |
| Full 12-attack × 5-dataset × 3-defense ASR matrix (referenced as "available at the project repository") | `experiments/qualitative_and_figures/build_attack_matrix.py` | `results/attack_matrix_results.json`, `results/attack_matrix.tex` |

## Data preparation (one-off, run before the ablations above)

These scripts generate adversarial payloads or auxiliary datasets consumed
by the experiments above. They are not themselves cited by a table, but
their *output* (files under `data/adversarial/`, `data/weblike/`,
`data_agent/`) is a prerequisite for reproducing several of the tables.

| Script | Produces |
|---|---|
| `experiments/data_preparation/generate_advanced_attacks.py` | The 2 advanced attack payloads (Unicode override, implicit injection) |
| `experiments/data_preparation/generate_weblike_dataset.py` | The 40-document Web-like KB and its 5 format-specific attacks |
| `experiments/data_preparation/generate_sprint1_themed.py`, `gen_sprint1_custom.py` | The 4 adaptive (DeepSeek-V3-generated) attack payloads, themed per dataset |
| `experiments/data_preparation/patch_nq_advanced.py` | **Historical, already applied** — do not re-run (see file docstring) |
| `experiments/data_preparation/integrate_100q_results.py` | Merges the 10q and 100q/250q result files into the combined summary used for the extended-validation finding |

## `experiments/exploratory_and_legacy/` — what's in there and why

That directory holds scripts that are **not** part of the reproduction path
above: earlier iterations superseded by a later script, partial/incomplete
runs (their result file is missing or was never finalized), or attacks that
were evaluated during development but are not reported in the accepted
paper. They are kept, unmodified in logic, for development provenance and
transparency — not deleted, and not presented as canonical. See
`experiments/exploratory_and_legacy/README.md` for the specific reason each
one is there.

## A note on multi-defense ablations across the RQ1 table set

The "8-attack subset" referenced in several `tab:kbdensity` captions is 6
canonical + 2 advanced attacks; the "12 attack families" referenced in the
attack matrix additionally includes the 4 adaptive (DeepSeek-V3-generated)
attacks. Per-script attack counts are documented in each script's own
module docstring.
