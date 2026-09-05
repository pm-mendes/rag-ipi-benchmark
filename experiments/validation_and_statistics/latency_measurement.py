"""
measure_latency.py — Mesure de la latence par couche de défense
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)
"""
import time
import statistics
from langchain_core.documents import Document
from ragipi.defenses.content_filter import filter_documents, is_clean
from ragipi.defenses.semantic_defense import (
    normalize_document, semantic_judge_document, is_anomalous
)
from ragipi.defenses.output_verifier import verify_output
from dotenv import load_dotenv

load_dotenv()

# Document de test représentatif
SAMPLE_DOC = Document(
    page_content=(
        "Paris is the capital of France located in northern Europe. "
        "The Eiffel Tower was built in 1889 by Gustave Eiffel. "
        "The Louvre Museum houses over 35,000 works of art."
    ),
    metadata={"source": "test", "type": "legitimate"}
)

SAMPLE_RESPONSE = "The Eiffel Tower is located in Paris, France."
SAMPLE_QUESTION = "Where is the Eiffel Tower?"

N_RUNS = 10  # répétitions pour la moyenne

def measure(fn, n=N_RUNS):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)  # ms
    return round(statistics.mean(times), 2), round(statistics.stdev(times), 2)

print("=== Mesure de latence par couche de défense ===\n")
print(f"{'Couche':<30} {'Mean (ms)':>10} {'Std (ms)':>10} {'Description'}")
print("─" * 70)

# L0 — Unicode normalisation
mean, std = measure(lambda: normalize_document(SAMPLE_DOC))
print(f"{'L0 — Unicode norm.':<30} {mean:>10.2f} {std:>10.2f}  Per document")

# L1 — Regex filter
mean, std = measure(lambda: is_clean(SAMPLE_DOC))
print(f"{'L1 — Lexical filter':<30} {mean:>10.2f} {std:>10.2f}  Per document")

# L1b — LLM-judge (LLaMA 3 8B local)
print(f"{'L1b — LLM-judge (LLaMA3)':<30} {'measuring...':>10}", end="\r", flush=True)
mean_llm, std_llm = measure(lambda: semantic_judge_document(SAMPLE_DOC), n=3)
print(f"{'L1b — LLM-judge (LLaMA3)':<30} {mean_llm:>10.2f} {std_llm:>10.2f}  Per document (CPU)")

# L1c — Embedding anomaly
from ragipi.defenses.semantic_defense import get_embedder
get_embedder()  # warm-up
mean, std = measure(lambda: is_anomalous(SAMPLE_DOC))
print(f"{'L1c — Embedding anomaly':<30} {mean:>10.2f} {std:>10.2f}  Per document")

# L2 — Context isolator (modification statique du prompt)
from ragipi.defenses.context_isolator import get_isolated_prompt
mean, std = measure(lambda: get_isolated_prompt())
print(f"{'L2 — Context isolator':<30} {mean:>10.2f} {std:>10.2f}  Per query (static)")

# L3 — Output verifier
mean, std = measure(lambda: verify_output(SAMPLE_RESPONSE))
print(f"{'L3 — Output verifier':<30} {mean:>10.2f} {std:>10.2f}  Per response")

print("─" * 70)
print("\nNote : latence LLM (GPT-3.5, DeepSeek) non mesurée ici")
print("       (dépend du réseau — environ 1-3s par appel API)")
print(f"\nN = {N_RUNS} runs pour L0, L1, L1c, L2, L3 ; N=3 pour L1b (lent)")

# Sauvegarder
import json
results = {
    "L0_unicode_ms": mean_llm,  # placeholder
    "L1_lexical_ms": 0,
    "L1b_llm_ms": mean_llm,
    "L1c_embedding_ms": 0,
    "L2_isolator_ms": 0,
    "L3_verifier_ms": 0,
}

