"""
ablation_multimodel_nq.py — Ablation multi-modèles sur dataset NQ
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Combine :
  - ablation_multimodel.py (GPT-3.5 vs DeepSeek-V3)
  - ablation_nq.py (dataset SQuAD/NQ, attaques thématiques)

Objectif : montrer que la generalisabilite tient sur un dataset
standard ET sur deux modeles differents simultanement.

Usage :
  python load_nq_dataset.py          # si pas encore fait
  python ablation_multimodel_nq.py
"""

import json
import shutil
import time
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

from ragipi.defenses.content_filter import filter_documents
from ragipi.defenses.context_isolator import get_isolated_prompt
from ragipi.defenses.output_verifier import verify_output
from ragipi.pipeline import SYSTEM_PROMPT
from ragipi.metrics.core import compute_rd, compute_ci
from ragipi.datasets.nq import load_nq_documents, load_nq_questions
from experiments.rq1_defense_effectiveness.kb_density_nq import (
    NQ_ADVERSARIAL_DOCS,
    NQ_SUCCESS_INDICATORS,
    ATTACKS_NQ,
    is_nq_attack_successful,
    create_nq_adversarial_docs,
)
from ragipi.llm_factory import get_llm

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K           = int(os.getenv("TOP_K", 3))

PROVIDERS = [
    {"id": "openai",    "label": "GPT-3.5-turbo (OpenAI)",    "sleep": 0},
    {"id": "deepseek",  "label": "DeepSeek-V3 (open source)", "sleep": 1},
    {
        "id":    "nvidia",
        "label": "Llama 3.1 70B (NVIDIA Build)",
        "sleep": 2,
    },
]

CONFIGS = [
    {"name": "No defense",   "filter": False, "isolate": False, "verify": False},
    {"name": "Filter only",  "filter": True,  "isolate": False, "verify": False},
    {"name": "Isolate only", "filter": False, "isolate": True,  "verify": False},
    {"name": "Verify only",  "filter": False, "isolate": False, "verify": True},
    {"name": "All defenses", "filter": True,  "isolate": True,  "verify": True},
]


def build_pipeline(docs, persist_dir, use_filter, use_isolate, provider_id, sleep_s):
    if use_filter:
        clean_docs, flagged = filter_documents(docs)
        if flagged:
            print(f"    [Filter] {len(flagged)} document(s) retiré(s)")
    else:
        clean_docs = docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(clean_docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm(provider_id)

    def format_docs(retrieved):
        return "\n\n---\n\n".join(
            f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(retrieved)
        )

    prompt = get_isolated_prompt() if use_isolate else ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def invoke_with_sleep(question):
        for attempt in range(5):
            try:
                result = chain.invoke(question)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                return result
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 30 * (attempt + 1)
                    print(f"    [RateLimit] Attente {wait}s (tentative {attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Échec après 5 tentatives")

    return invoke_with_sleep


def run_ablation(provider, questions, clean_docs):
    pid   = provider["id"]
    label = provider["label"]
    sleep = provider["sleep"]
    adv_docs = create_nq_adversarial_docs()

    print(f"\n{'='*60}")
    print(f"Provider : {label}")
    print(f"{'='*60}")

    # Baseline
    print(f"\nCalcul de la baseline ({label})...")
    invoke = build_pipeline(
        clean_docs, f"./chroma_mmnq_{pid}_base",
        False, False, pid, sleep
    )
    baseline = {q: invoke(q) for q in questions}
    print(f"  {len(questions)} réponses baseline calculées")

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*50}")
        print(f"Config : {config['name']} | {pid}")

        for attack_type in ATTACKS_NQ:
            adversarial_doc = adv_docs[attack_type]
            poisoned = clean_docs + [adversarial_doc]
            persist = f"./chroma_mmnq_{pid}_{config['name'].replace(' ','_')}_{attack_type}"

            invoke = build_pipeline(
                poisoned, persist,
                config["filter"], config["isolate"],
                pid, sleep
            )

            asr_n, rd_list, ci_list = 0, [], []

            for q in questions:
                response = invoke(q)
                if config["verify"]:
                    _, response = verify_output(response)

                success = is_nq_attack_successful(attack_type, response)
                rd = compute_rd(baseline[q], response)
                ci = compute_ci(response)

                if success:
                    asr_n += 1
                rd_list.append(rd)
                ci_list.append(ci)

            m = {
                "attack_type": attack_type,
                "asr": round(asr_n / len(questions), 3),
                "rd":  round(sum(rd_list) / len(rd_list), 4),
                "ci":  round(sum(ci_list) / len(ci_list), 4),
            }
            config_results["attacks"].append(m)
            print(f"  {attack_type:<25} ASR={m['asr']:.0%}  "
                  f"RD={m['rd']:.4f}  CI={m['ci']:.4f}")

        config_results["avg_asr"] = round(
            sum(a["asr"] for a in config_results["attacks"]) / len(ATTACKS_NQ), 3
        )
        config_results["avg_rd"] = round(
            sum(a["rd"] for a in config_results["attacks"]) / len(ATTACKS_NQ), 4
        )
        config_results["avg_ci"] = round(
            sum(a["ci"] for a in config_results["attacks"]) / len(ATTACKS_NQ), 4
        )
        all_results.append(config_results)

    return all_results


def print_final_table(results_by_provider):
    providers = list(results_by_provider.keys())

    print(f"\n{'='*72}")
    print("TABLEAU FINAL — Multi-modèles × Dataset NQ")
    print(f"GPT-3.5-turbo vs DeepSeek-V3 | 20 questions SQuAD/NQ | 6 attaques")
    print(f"{'='*72}")

    labels = {
        "openai":   "GPT-3.5",
        "deepseek": "DeepSeek",
        "nvidia":   "Llama 3.1",
        "groq":     "LLaMA 3.3",
    }

    header = f"{'Configuration':<16}"
    for p in providers:
        lbl = labels.get(p, p)
        header += f" {lbl+' ASR':>11} {lbl+' RD':>10} {lbl+' CI':>10}"
    print(header)
    print("─"*72)

    configs = [r["config"] for r in results_by_provider[providers[0]]]
    for i, config_name in enumerate(configs):
        line = f"{config_name:<16}"
        for p in providers:
            r = results_by_provider[p][i]
            line += (
                f" {r['avg_asr']:>10.1%}"
                f" {r['avg_rd']:>10.4f}"
                f" {r['avg_ci']:>10.4f}"
            )
        print(line)

    print(f"{'='*72}")
    print("ASR ↓  |  RD ↓  |  CI ↑")

    # Analyse automatique
    p1 = providers[0]
    p2 = providers[1] if len(providers) > 1 else providers[0]
    nd1 = next(r for r in results_by_provider[p1] if r["config"] == "No defense")
    nd2 = next(r for r in results_by_provider[p2] if r["config"] == "No defense")
    ad1 = next(r for r in results_by_provider[p1] if r["config"] == "All defenses")
    ad2 = next(r for r in results_by_provider[p2] if r["config"] == "All defenses")

    print(f"\n--- Analyse ---")
    print(f"Sans défense : {labels[p1]} ASR={nd1['avg_asr']:.1%} | "
          f"{labels[p2]} ASR={nd2['avg_asr']:.1%}")
    print(f"Toutes défenses : {labels[p1]} ASR={ad1['avg_asr']:.1%} CI={ad1['avg_ci']:.3f} | "
          f"{labels[p2]} ASR={ad2['avg_asr']:.1%} CI={ad2['avg_ci']:.3f}")
    if ad1["avg_asr"] == 0.0 and ad2["avg_asr"] == 0.0:
        print("→ Defense-in-depth efficace sur les deux modèles et sur dataset NQ standard.")


if __name__ == "__main__":
    import sys

    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        selected = sys.argv[idx + 1]
        PROVIDERS = [p for p in PROVIDERS if p["id"] == selected]
        if not PROVIDERS:
            print(f"Provider inconnu. Disponibles : openai, deepseek")
            sys.exit(1)
        print(f"Mode sélectif : provider={selected}\n")

    print("=== Ablation Multi-Modèles × Dataset NQ ===\n")

    nq_dir = Path("data/nq")
    if not nq_dir.exists() or not list(nq_dir.glob("nq_*.txt")):
        print("Dataset NQ non trouvé. Lance d'abord : python load_nq_dataset.py")
        sys.exit(1)

    questions = load_nq_questions()
    clean_docs = load_nq_documents()
    print(f"  {len(questions)} questions NQ, {len(clean_docs)} documents\n")

    results_by_provider = {}
    for provider in PROVIDERS:
        results = run_ablation(provider, questions, clean_docs)
        results_by_provider[provider["id"]] = results

    print_final_table(results_by_provider)

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_multimodel_nq_results.json", "w") as f:
        json.dump(results_by_provider, f, indent=2, ensure_ascii=False)
    print("\nRésultats sauvegardés dans results/ablation_multimodel_nq_results.json")

    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_mmnq_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
