# Reproducibility

## 1. Environment

```bash
git clone <this-repo-url>
cd rag-ipi-benchmark
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

For the L1b LLM-judge layer (LLaMA 3 8B) and the semantic defense pipeline,
install [Ollama](https://ollama.ai) and pull the model:

```bash
ollama pull llama3:8b
```

Development/CI extras (pytest, ruff):

```bash
pip install -r requirements-dev.txt
```

## 2. API keys (`.env`)

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_MODEL` | Primary generator (GPT-3.5-turbo) |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | Cross-model evaluation, adaptive-attack generation |
| `GROQ_API_KEY` | Optional alternate open-weights provider |
| `NVIDIA_API_KEY` | Llama 3.1 70B (cross-model evaluation), accessed via the NVIDIA API |
| `EMBEDDING_MODEL` | Defaults to `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| `TOP_K` | Retrieval depth, defaults to 3 |

None of these are committed. `.env` is git-ignored.

## 3. Datasets

The **Custom** KB (`data/clean_docs/`, `data/adversarial/`) and its 30
queries (`data/questions.json`) are committed in full — no download needed.

The four public benchmark datasets require a one-time download (their raw
files are too large to commit and are git-ignored):

| Dataset | Expected file | Source |
|---|---|---|
| SQuAD/NQ | `data/nq/squad_dev.json` | `curl -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o data/nq/squad_dev.json` |
| MITRE ATT&CK Enterprise | `data/mitre/enterprise_attack.json` | [github.com/mitre/cti](https://github.com/mitre/cti) — `enterprise-attack/enterprise-attack.json` |
| TriviaQA | `data/triviaqa/wikipedia-dev.json` | [nlp.cs.washington.edu/triviaqa](http://nlp.cs.washington.edu/triviaqa/) — Wikipedia dev split |
| HotpotQA | `data/hotpotqa/hotpot_dev.json` | `curl -L http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -o data/hotpotqa/hotpot_dev.json` |

Each loader (`src/ragipi/datasets/{nq,mitre,triviaqa,hotpotqa}.py`) asserts
the expected file exists and prints the exact download command if it does
not. After downloading, each loader's own subset-generation function
(`load_*_documents` / `load_*_questions`, or the module's `__main__` block)
converts the raw file into the per-document `.txt` files and question JSON
consumed by the ablation scripts, using `RANDOM_SEED = 42` for the sampled
subset.

The Web-like KB (`data/weblike/`) is generated synthetically — see
`experiments/data_preparation/generate_weblike_dataset.py`.

## 4. Determinism and seeds

All generation calls use `temperature=0`. Combined with a fixed corpus and
a deterministic dense retriever, this makes the pipeline fully
deterministic for a fixed model *version* — API-side model updates by the
provider are the only source of drift.

`experiments/rq1_defense_effectiveness/seed_reproducibility.py` reruns the
heuristic ablation under seeds `{42, 123, 456}` and reports mean ± std
(`results/statistical_results.json`). The near-zero variance observed
confirms **setup reproducibility** (same corpus, same deterministic
pipeline) — it is not a claim of statistical robustness to dataset
sampling variability, which is instead addressed by the extended
100q/250q validation (see `docs/EXPERIMENTS.md`).

## 5. What CI does and does not check

`.github/workflows/ci.yml` runs `pytest tests/` (pure-function unit tests
for the metrics, defenses, attack catalog, and statistics helpers — no API
calls, no network, no Ollama) plus a byte-compile pass over every script.
It does **not** run the experiment scripts themselves: those require paid
LLM API access, a running Ollama daemon, and the downloaded datasets above.
Running them is a manual, local step (see `docs/EXPERIMENTS.md`).

## 6. Cost estimates

Several scripts document their own expected token cost in a header comment
(e.g. `extended_validation_100_250q.py`: ≈900K tokens ≈ $0.45 at
GPT-3.5-turbo pricing for a 100-query run). The full reproduction of every
table in the paper, across three generator models, is estimated at a few
tens of dollars of API spend; the dominant cost is the cross-model and
extended-validation sweeps, not the small controlled ablations.
