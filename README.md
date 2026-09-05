# RAG-IPI Benchmark

**Evaluating Indirect Prompt Injection in RAG Systems: Defense-in-Depth, Retrieval Effects, and Security–Utility Trade-offs**

IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology — WI-IAT 2026

*Yulliwas Ameur · Miqeas Pedro Mendes · Samia Bouzefrane · Lyes Khoukhi*
Efrei Research Lab, Université Paris-Panthéon-Assas · Cédric, Cnam Paris

[Paper (PDF)](paper/WI_IAT2026_RAG_IPI_Benchmark.pdf) · [LaTeX source](paper/main.tex) · [Experiment map](docs/EXPERIMENTS.md) · [Reproducibility guide](docs/REPRODUCIBILITY.md)

---

## Overview

Indirect prompt injection (IPI) exploits the fact that a RAG system's
prompt is assembled from externally retrieved documents that an adversary
may control. This repository is the full experimental framework behind the
paper's central claim: **IPI robustness in RAG is not a model-only
property** — it emerges from the interaction between retrieval exposure
(which documents enter the context), prompt assembly (how they're
presented to the generator), and defense-layer composition (how individual
defenses interact, sometimes non-monotonically).

- **12 IPI attack families** — 6 canonical, 2 advanced (Unicode
  obfuscation, implicit/narrative injection), 4 adaptive
  (DeepSeek-V3-generated, built to evade known filters)
- **4 metrics** — ASR, RD, CI, and the novel **SAU** (Security-Oriented
  Answer Utility), which distinguishes secure-but-useless responses from
  secure-and-informative ones
- **6-layer defense-in-depth pipeline** — heuristic (L1 content filter, L2
  context isolator, L3 output verifier) and semantic (L0 Unicode
  normalization, L1b LLM-judge, L1c embedding anomaly detection)
- **5 datasets** spanning KB densities 5→498 documents (Custom, SQuAD/NQ,
  MITRE ATT&CK, TriviaQA, HotpotQA)
- **3 generator models** — GPT-3.5-turbo, DeepSeek-V3, Llama 3.1 70B
- **14,580 evaluations** total, plus a minimal agentic tool-calling
  extension and a Web-like content-format evaluation
- Every raw result reported in the paper is committed under `results/` as
  JSON, with an exact [script → table mapping](docs/EXPERIMENTS.md)

## Headline findings

| Finding | Evidence |
|---|---|
| Layered heuristic defenses reach **0% ASR** on evaluated canonical attacks, at less utility cost than output verification alone | `tab:heuristic`: All-heuristic ASR=0.0%, SAU=0.724 vs. Verify-only SAU=0.553 |
| Defense-in-depth can be **non-monotonic**: adding a semantic layer can *increase* attack success | Semantic-full advanced-attack ASR=20% > L1b-alone ASR=10%; MITRE adaptive-attack ASR rises from 3.3% (all-heuristic) to 6% (semantic-full) |
| Retrieval design is a **security parameter**, not just a relevance knob | ASR grows from 3.3% (top-k=1) to 45.0% (top-k=10); BM25 cuts ASR by 15pp vs. dense retrieval (McNemar χ²=7.11, p=0.008) |
| Retrieval-optimized and semantically adaptive attacks remain an **open threat** | A retrieval-optimized document (sim=0.689 vs. 0.162 generic) doubles ASR to 20% under all heuristic defenses |
| SAU is a validated proxy **for controlled ablations only** | r=0.921 vs. human relevance judgments on our sample; not a substitute for EM/F1/BERTScore, and not independently validated at scale |

See the [paper](paper/WI_IAT2026_RAG_IPI_Benchmark.pdf) for the full
discussion (RQ1–RQ4) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for
every number's exact source script and result file.

## Repository layout

```
rag-ipi-benchmark/
├── src/ragipi/            Installable library — pipeline, defenses,
│                          metrics, attacks, LLM-judge, agents, dataset
│                          loaders. See docs/ARCHITECTURE.md.
├── experiments/           One runnable script per paper table/section,
│                          grouped by research question. See
│                          docs/EXPERIMENTS.md for the full map and
│                          experiments/README.md for how to run them.
├── data/                  Adversarial payloads, the Custom KB, and the
│                          processed per-dataset documents/questions.
│                          Large raw downloads are git-ignored — see
│                          docs/REPRODUCIBILITY.md.
├── results/               Every reported result, committed as JSON/PDF.
├── paper/                 LaTeX source (main.tex, refs.bib) and the
│                          camera-ready PDF.
├── docs/                  Architecture, threat model, metric definitions,
│                          reproducibility guide, experiment map.
└── tests/                 Pytest suite for the pure-function core (no
                           API keys or network required).
```

## Attack strategies (12 total)

### Canonical (6)

| ID | Objective | Description |
|----|-----------|-------------|
| OVERRIDE | O1 | Explicit instruction replacement (`IGNORE ALL PREVIOUS INSTRUCTIONS`) |
| EXFIL | O2 | HTML comment embedding an audit directive that leaks the system prompt |
| ROLE | O3 | DAN persona assignment overriding safety constraints |
| DENIAL | O4 | Administrative notice blocking all responses |
| TECHNICAL_OVERRIDE | O1 | Injection framed as a security patch in technical content |
| DATA_EXFIL | O2 | Compliance-audit notice in domain-specific content |

### Advanced (2)

| ID | Description |
|----|-------------|
| UNICODE_OVERRIDE | Zero-width Unicode interleaving (U+200B/C/D) — invisible to lexical filters |
| IMPLICIT_INJECTION | Narrative/audit-session framing, no explicit instruction keywords |

### Adaptive (4) — generated by DeepSeek-V3 to evade known filters

| ID | Description |
|----|-------------|
| LLM_GENERATED | Payload embedded in domain-accurate generated content |
| BASE64_ENCODING | Payload Base64-encoded, invisible to string filters |
| MANY_SHOT | Payload repeated 5× with syntactic variation |
| AUTO_PARAPHRASE | Keyword-free reformulation avoiding all known compromise markers |

Domain-aligned variants for SQuAD/NQ, MITRE ATT&CK, and HotpotQA live under
`data/adversarial/{nq,mitre,hotpotqa}/` — same payload, dataset-specific
camouflage prefix. See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for the
formal threat model.

## Defense pipeline

```
Retrieved document
      │
  [L0] Unicode normalization        NFKC + zero-width removal        ~0.01 ms
      │
  [L1] Content filter               14 regex patterns                ~0.02 ms
      │
  [L1b] LLM-judge (LLaMA 3 8B)      Adversarial-intent classification ~100 s CPU
      │
  [L1c] Embedding anomaly           δ = sim_imperative − sim_descriptive > 0.05   ~109 ms
      │
  [L2] Context isolator             Isolation-aware system prompt     ~0.08 ms
      │
  LLM generator
      │
  [L3] Output verifier              13 compromise-marker scan         ~0.00 ms
      │
  Final answer
```

> **Deployment note** (Table "Defense layer latency"): L1b costs ~100s on
> CPU and should be applied at *indexing* time, not query time, to amortize
> its cost across all future queries against a given document; a GPU or a
> fine-tuned classifier is needed for real-time use. L2 showed
> non-monotonic interactions with upstream semantic filters in our
> experiments — validate empirically before enabling it in combination
> with L0/L1c.

## Metrics

| Metric | Direction | Formula |
|--------|:---:|---------|
| ASR — Attack Success Rate | ↓ | mean of `1[success(rᵃ)]`, via lexical heuristics + LLM-judge cascade |
| RD — Robustness Degradation | ↓ | mean cosine distance between baseline and attacked response embeddings |
| CI — Context Integrity | ↑ | `max(0, 1 − 0.25 × |compromise markers|)` |
| **SAU** — Security-Oriented Answer Utility (novel) | ↑ | `|expected keywords ∩ response| / |expected keywords|` |

Full definitions, formulas, and validation status: [docs/METRICS.md](docs/METRICS.md).

## Installation

```bash
git clone <this-repo-url>
cd rag-ipi-benchmark
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

For the L1b LLM-judge layer, install [Ollama](https://ollama.ai) and pull
the model: `ollama pull llama3:8b`.

Full setup, dataset downloads, and determinism notes: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Reproducing the paper's results

Every table, figure, and statistical test has a dedicated script and a
committed result file — see the complete map in
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**. Quick examples:

```bash
# RQ1 — heuristic ablation (Table "Heuristic ablation")
python experiments/rq1_defense_effectiveness/heuristic_ablation.py

# RQ2 — semantic defenses vs. advanced attacks (Table "Semantic defense ablation")
python experiments/rq2_semantic_and_adaptive/semantic_defense_ablation.py

# RQ3 — top-k sensitivity (Table "Impact of top-k")
python experiments/rq3_retrieval_design/topk_sensitivity.py

# RQ4 — agentic tool-misuse evaluation (Table "Agentic IPI evaluation")
python experiments/rq4_composition_agentic_web/agentic_voluntary_tmr.py

# Statistics — bootstrap 95% CIs and McNemar significance tests
python experiments/validation_and_statistics/bootstrap_confidence_intervals.py
python experiments/validation_and_statistics/significance_tests.py
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite covers the pure-function core (ASR/CI/SAU formulas, the L1
content filter and L3 output verifier, the attack catalog, bootstrap CI and
McNemar significance) — no API keys, network access, or Ollama required.
CI (`.github/workflows/ci.yml`) runs this suite plus a lint pass and a
full byte-compile of every script on every push.

## Limitations (see paper, Section VIII, for the full discussion)

- Canonical/generic payloads only; a retrieval-optimized adversary (à la
  PoisonedRAG) is a harder, only partially addressed, open threat.
- The LLM-judge (LLaMA 3 8B) has 55% recall on advanced/adaptive attacks —
  reported advanced/adaptive ASR are conservative lower bounds.
- Lexical filters are English-centric; multilingual/code-switching attacks
  and non-English content fairness were not evaluated.
- SAU is validated for controlled defense ablations only; it is not a
  substitute for reference-based utility metrics.
- The non-monotonic defense-composition finding holds across two attack
  families and two datasets, and across a minimal single-turn agentic
  extension — broader statistical validation at scale is future work.

## Citation

```bibtex
@inproceedings{ameur2026ragipi,
  title     = {Evaluating Indirect Prompt Injection in {RAG} Systems:
               Defense-in-Depth, Retrieval Effects, and
               Security--Utility Trade-offs},
  author    = {Ameur, Yulliwas and Mendes, Miqeas Pedro
               and Bouzefrane, Samia and Khoukhi, Lyes},
  booktitle = {IEEE/WIC/ACM International Joint Conference on
               Web Intelligence and Intelligent Agent Technology
               (WI-IAT)},
  year      = {2026}
}
```

See also [`CITATION.cff`](CITATION.cff).

## Acknowledgment

Generative AI tools, including Microsoft 365 Copilot, were used to assist
with language polishing in parts of the manuscript. The authors reviewed,
edited, and take full responsibility for all scientific content,
experiments, results, and conclusions.

## License

Code is released under the [MIT License](LICENSE). Data and results
(adversarial payloads, knowledge-base documents, and JSON result files
under `data/` and `results/`) are released under the same terms unless
noted otherwise in a specific file.
