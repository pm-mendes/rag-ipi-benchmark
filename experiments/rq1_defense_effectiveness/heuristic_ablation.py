"""
ablation_v2.py — Ablation study enrichie (10 questions / 5 docs / 6 attaques)
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)
"""

import json
import shutil
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

from ragipi.defenses.content_filter import filter_documents
from ragipi.defenses.context_isolator import get_isolated_prompt
from ragipi.defenses.output_verifier import verify_output, SAFE_FALLBACK
from ragipi.pipeline import load_documents_from_folder, SYSTEM_PROMPT
from ragipi.metrics.core import compute_rd, compute_ci, load_questions
from ragipi.attacks.runner import is_attack_successful
from ragipi.attacks.catalog import CANONICAL_ATTACKS

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TOP_K           = int(os.getenv("TOP_K", 3))

CONFIGS = [
    {"name": "No defense",   "filter": False, "isolate": False, "verify": False},
    {"name": "Filter only",  "filter": True,  "isolate": False, "verify": False},
    {"name": "Isolate only", "filter": False, "isolate": True,  "verify": False},
    {"name": "Verify only",  "filter": False, "isolate": False, "verify": True},
    {"name": "All defenses", "filter": True,  "isolate": True,  "verify": True},
]

# Canonical 6-attack registry — see ragipi.attacks.catalog for the single
# source of truth (also reused, unchanged, by several downstream scripts).
ATTACKS = CANONICAL_ATTACKS


def build_defended_pipeline(docs, persist_dir, use_filter, use_isolate):
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
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

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
    return chain


def run_ablation_v2(questions, clean_docs, attacks):
    print("Calcul des réponses baseline...")
    base_chain = build_defended_pipeline(
        clean_docs, "./chroma_v2_base", False, False
    )
    baseline = {q: base_chain.invoke(q) for q in questions}
    print(f"  {len(questions)} réponses baseline calculées\n")

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*55}")
        print(f"Configuration : {config['name']}")

        for attack_type, adv_file in attacks:
            adv_content = Path(adv_file).read_text(encoding="utf-8")
            adversarial_doc = Document(
                page_content=adv_content,
                metadata={"source": adv_file, "type": "adversarial",
                          "attack_type": attack_type}
            )
            poisoned = clean_docs + [adversarial_doc]
            persist = f"./chroma_v2_{config['name'].replace(' ', '_')}_{attack_type}"

            chain = build_defended_pipeline(
                poisoned, persist, config["filter"], config["isolate"]
            )

            asr_n, rd_list, ci_list = 0, [], []

            for q in questions:
                response = chain.invoke(q)
                if config["verify"]:
                    _, response = verify_output(response)

                success = is_attack_successful(attack_type, response)
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
            print(f"  {attack_type:<22} ASR={m['asr']:.0%}  "
                  f"RD={m['rd']:.4f}  CI={m['ci']:.4f}")

        config_results["avg_asr"] = round(
            sum(a["asr"] for a in config_results["attacks"]) / len(attacks), 3
        )
        config_results["avg_rd"] = round(
            sum(a["rd"] for a in config_results["attacks"]) / len(attacks), 4
        )
        config_results["avg_ci"] = round(
            sum(a["ci"] for a in config_results["attacks"]) / len(attacks), 4
        )
        all_results.append(config_results)

    return all_results


def print_ablation_table_v2(results):
    print(f"\n{'='*62}")
    print("ABLATION STUDY v2 — 10 questions / 5 docs / 6 attaques")
    print(f"{'='*62}")
    print(f"{'Configuration':<16} {'ASR moy':>9} {'RD moy':>9} {'CI moy':>9}")
    print("─"*46)
    for r in results:
        print(
            f"{r['config']:<16} "
            f"{r['avg_asr']:>8.1%} "
            f"{r['avg_rd']:>9.4f} "
            f"{r['avg_ci']:>9.4f}"
        )
    print(f"{'='*62}")
    print("Objectif : ASR ↓  |  RD ↓  |  CI ↑")

    print(f"\n{'='*62}")
    print("DÉTAIL PAR ATTAQUE — Configuration 'All defenses'")
    print(f"{'='*62}")
    all_def = next(r for r in results if r["config"] == "All defenses")
    print(f"{'Attaque':<24} {'ASR':>7} {'RD':>9} {'CI':>9}")
    print("─"*52)
    for a in all_def["attacks"]:
        print(
            f"{a['attack_type']:<24} "
            f"{a['asr']:>6.0%} "
            f"{a['rd']:>9.4f} "
            f"{a['ci']:>9.4f}"
        )


if __name__ == "__main__":
    print("=== Ablation Study v2 — Benchmark enrichi ===\n")

    questions = load_questions("data/questions.json")
    clean_docs = load_documents_from_folder("data/clean_docs")

    results = run_ablation_v2(questions, clean_docs, ATTACKS)
    print_ablation_table_v2(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRésultats sauvegardés dans results/ablation_v2_results.json")