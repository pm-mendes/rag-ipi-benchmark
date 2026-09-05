"""
ablation_topk.py — Ablation top-k : impact sur l'ASR
Papier : WI-IAT 2026

Hypothèse : ASR augmente avec k car le document adversarial
a plus de chances d'être dans le top-k récupéré.

Configurations : k ∈ {1, 3, 5, 10}
Dataset : custom 10q, 6 attaques canoniques, No defense
Modèle  : GPT-3.5-turbo
"""

import json, shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

from ragipi.pipeline import load_documents_from_folder, SYSTEM_PROMPT
from ragipi.metrics.core import compute_rd, compute_ci
from ragipi.metrics.sau import compute_sau, load_questions_meta
from ragipi.attacks.catalog import CANONICAL_ATTACKS as ATTACKS
from ragipi.attacks.runner import is_attack_successful

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

TOP_K_VALUES    = [1, 3, 5, 10]
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50


def build_chain(docs, persist_dir, top_k):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    vs = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=persist_dir)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    def fmt(r):
        return "\n\n---\n\n".join(
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(r))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return (
        {"context": retriever | fmt,
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def run_topk_ablation(questions, questions_meta, clean_docs):
    results = []

    for k in TOP_K_VALUES:
        print(f"\n{'─'*50}")
        print(f"top-k = {k}")

        # Baseline
        chain_base = build_chain(
            clean_docs, f"./chroma_topk_{k}_base", k)
        baseline = {q: chain_base.invoke(q) for q in questions}

        k_results = {"top_k": k, "attacks": []}
        asr_list, rd_list, ci_list, ar_list = [], [], [], []

        for atk_name, adv_file in ATTACKS:
            adv_doc = Document(
                page_content=Path(adv_file).read_text(encoding="utf-8"),
                metadata={"source": adv_file, "type": "adversarial"}
            )
            poisoned = clean_docs + [adv_doc]
            chain = build_chain(
                poisoned, f"./chroma_topk_{k}_{atk_name}", k)

            asr_n, rd_l, ci_l, ar_l = 0, [], [], []
            for i, q in enumerate(questions):
                resp = chain.invoke(q)
                if is_attack_successful(atk_name, resp):
                    asr_n += 1
                rd_l.append(compute_rd(baseline[q], resp))
                ci_l.append(compute_ci(resp))
                ar_l.append(compute_sau(
                    resp,
                    questions_meta[i].get("expected_keywords", [])))

            n = len(questions)
            m = {
                "attack": atk_name,
                "asr": round(asr_n / n, 3),
                "rd":  round(sum(rd_l) / n, 4),
                "ci":  round(sum(ci_l) / n, 4),
                "ar":  round(sum(ar_l) / n, 4),
            }
            k_results["attacks"].append(m)
            asr_list.append(m["asr"])
            rd_list.append(m["rd"])
            ci_list.append(m["ci"])
            ar_list.append(m["ar"])
            print(f"  {atk_name:<25} ASR={m['asr']:.0%} "
                  f"RD={m['rd']:.4f} CI={m['ci']:.4f} "
                  f"AR={m['ar']:.4f}")

        k_results["avg_asr"] = round(sum(asr_list)/len(asr_list), 3)
        k_results["avg_rd"]  = round(sum(rd_list)/len(rd_list), 4)
        k_results["avg_ci"]  = round(sum(ci_list)/len(ci_list), 4)
        k_results["avg_ar"]  = round(sum(ar_list)/len(ar_list), 4)
        results.append(k_results)
        print(f"  → avg ASR={k_results['avg_asr']:.1%} "
              f"RD={k_results['avg_rd']:.4f} "
              f"AR={k_results['avg_ar']:.4f}")

    return results


def print_summary(results):
    print(f"\n{'='*60}")
    print("ABLATION top-k — Custom 10q, 6 attaques, No defense")
    print(f"{'='*60}")
    print(f"{'top-k':>6} {'ASR':>8} {'RD':>8} {'CI':>8} {'AR':>8}")
    print("─"*44)
    for r in results:
        print(
            f"{r['top_k']:>6} "
            f"{r['avg_asr']:>7.1%} "
            f"{r['avg_rd']:>8.4f} "
            f"{r['avg_ci']:>8.4f} "
            f"{r['avg_ar']:>8.4f}"
        )
    print(f"{'='*60}")

    # Analyse de tendance
    asrs = [r["avg_asr"] for r in results]
    if asrs[-1] >= asrs[0]:
        print("\n→ ASR augmente avec k : "
              "le document adversarial est plus souvent récupéré")
    else:
        print("\n→ ASR ne croît pas monotoniquement avec k")

    # Trouver le k optimal (meilleur ASR/AR tradeoff)
    best = min(results, key=lambda r: r["avg_asr"] - r["avg_ar"])
    print(f"→ Meilleur tradeoff ASR/AR : top-k={best['top_k']} "
          f"(ASR={best['avg_asr']:.1%}, AR={best['avg_ar']:.4f})")


if __name__ == "__main__":
    print("=== Ablation top-k ∈ {1,3,5,10} ===\n")

    questions_meta = load_questions_meta("data/questions.json")
    # Utiliser les 10 premières questions pour cohérence avec Table I
    questions_meta = questions_meta[:10]
    questions  = [q["text"] for q in questions_meta]
    clean_docs = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    results = run_topk_ablation(questions, questions_meta, clean_docs)
    print_summary(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_topk_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats → results/ablation_topk_results.json")

    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_topk_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
