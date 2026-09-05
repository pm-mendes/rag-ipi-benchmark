"""
ablation_semantic.py — Ablation avec défenses sémantiques (L0+L1b+L1c)
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Compare 3 configurations de défense sémantique contre les
attaques canoniques ET avancées :
  - Semantic L0 only  : Unicode normalisation
  - Semantic L1b only : LLM-judge sur documents
  - Semantic full     : L0 + L1b + L1c (pipeline complet)
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
from ragipi.defenses.output_verifier import verify_output
from ragipi.defenses.semantic_defense import (
    semantic_filter_pipeline,
    normalize_documents,
)
from ragipi.pipeline import load_documents_from_folder, SYSTEM_PROMPT
from ragipi.metrics.core import compute_rd, compute_ci, load_questions
from ragipi.attacks.runner import is_attack_successful
from ragipi.judge.cascade import cascade_detect, ADVANCED_ATTACK_TYPES

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TOP_K           = int(os.getenv("TOP_K", 3))

# ── Configurations testées ────────────────────────────────────────────────────

CONFIGS = [
    # Référence
    {
        "name":         "No defense",
        "lex_filter":   False,
        "sem_unicode":  False,
        "sem_llm":      False,
        "sem_embed":    False,
        "isolate":      False,
        "verify":       False,
    },
    # Pipeline heuristique complet (baseline de comparaison)
    {
        "name":         "All heuristic",
        "lex_filter":   True,
        "sem_unicode":  False,
        "sem_llm":      False,
        "sem_embed":    False,
        "isolate":      True,
        "verify":       True,
    },
    # L0 seul — Unicode uniquement
    {
        "name":         "L0 Unicode only",
        "lex_filter":   True,
        "sem_unicode":  True,
        "sem_llm":      False,
        "sem_embed":    False,
        "isolate":      False,
        "verify":       False,
    },
    # L1b seul — LLM-judge sur docs
    {
        "name":         "L1b LLM-judge",
        "lex_filter":   True,
        "sem_unicode":  False,
        "sem_llm":      True,
        "sem_embed":    False,
        "isolate":      False,
        "verify":       False,
    },
    # L1c seul — Embedding anomaly
    {
        "name":         "L1c Embedding",
        "lex_filter":   True,
        "sem_unicode":  False,
        "sem_llm":      False,
        "sem_embed":    True,
        "isolate":      False,
        "verify":       False,
    },
    # Pipeline sémantique complet
    {
        "name":         "Semantic full",
        "lex_filter":   True,
        "sem_unicode":  True,
        "sem_llm":      True,
        "sem_embed":    True,
        "isolate":      True,
        "verify":       True,
    },
]

ALL_ATTACKS = [
    ("OVERRIDE",           "data/adversarial/override.txt"),
    ("EXFIL",              "data/adversarial/exfil.txt"),
    ("ROLE",               "data/adversarial/role.txt"),
    ("DENIAL",             "data/adversarial/denial.txt"),
    ("TECHNICAL_OVERRIDE", "data/adversarial/technical_override.txt"),
    ("DATA_EXFIL",         "data/adversarial/data_exfil.txt"),
    ("UNICODE_OVERRIDE",   "data/adversarial/unicode_override.txt"),
    ("IMPLICIT_INJECTION", "data/adversarial/implicit_injection.txt"),
]


def build_pipeline_semantic(docs, persist_dir, config):
    """Construit le pipeline avec les défenses sémantiques configurées."""

    working_docs = list(docs)

    # L0 — Unicode normalisation
    if config["sem_unicode"]:
        working_docs, n_norm = normalize_documents(working_docs)
        if n_norm > 0:
            print(f"    [L0] {n_norm} doc(s) normalisé(s) Unicode")

    # L1 — Filtre lexical
    if config["lex_filter"]:
        working_docs, flagged_lex = filter_documents(working_docs)
        if flagged_lex:
            print(f"    [L1-Lex] {len(flagged_lex)} doc(s) retiré(s)")

    # L1b — LLM-judge sémantique sur documents
    if config["sem_llm"]:
        working_docs, report = semantic_filter_pipeline(
            working_docs,
            use_unicode_norm=False,
            use_llm_judge=True,
            use_embedding_anom=False,
            verbose=True,
        )

    # L1c — Embedding anomaly
    if config["sem_embed"]:
        working_docs, report = semantic_filter_pipeline(
            working_docs,
            use_unicode_norm=False,
            use_llm_judge=False,
            use_embedding_anom=True,
            verbose=True,
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(working_docs)
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
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(retrieved)
        )

    prompt = get_isolated_prompt() if config["isolate"] \
        else ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_semantic_ablation(questions, clean_docs):
    print("\nCalcul baseline...")
    chain_base = build_pipeline_semantic(
        clean_docs, "./chroma_sem_base",
        {"lex_filter": False, "sem_unicode": False, "sem_llm": False,
         "sem_embed": False, "isolate": False, "verify": False}
    )
    baseline = {q: chain_base.invoke(q) for q in questions}
    print(f"  {len(questions)} réponses baseline\n")

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*55}")
        print(f"Configuration : {config['name']}")

        canon_asr, adv_asr = [], []

        for attack_type, adv_file in ALL_ATTACKS:
            adv_content = Path(adv_file).read_text(encoding="utf-8")
            adversarial_doc = Document(
                page_content=adv_content,
                metadata={"source": adv_file, "type": "adversarial",
                          "attack_type": attack_type}
            )
            poisoned = clean_docs + [adversarial_doc]
            persist = (f"./chroma_sem_"
                       f"{config['name'].replace(' ','_')}_{attack_type}")

            chain = build_pipeline_semantic(poisoned, persist, config)

            asr_n, rd_list, ci_list = 0, [], []

            for q in questions:
                response = chain.invoke(q)
                if config["verify"]:
                    _, response = verify_output(response)

                # Détection cascade pour les attaques avancées
                if attack_type in ADVANCED_ATTACK_TYPES:
                    success, _ = cascade_detect(attack_type, response, q)
                else:
                    success = is_attack_successful(attack_type, response)

                rd = compute_rd(baseline[q], response)
                ci = compute_ci(response)

                if success:
                    asr_n += 1
                rd_list.append(rd)
                ci_list.append(ci)

            m = {
                "attack_type": attack_type,
                "is_advanced": attack_type in ADVANCED_ATTACK_TYPES,
                "asr": round(asr_n / len(questions), 3),
                "rd":  round(sum(rd_list) / len(rd_list), 4),
                "ci":  round(sum(ci_list) / len(ci_list), 4),
            }
            config_results["attacks"].append(m)

            tag = "[ADV]" if attack_type in ADVANCED_ATTACK_TYPES else "     "
            print(f"  {tag} {attack_type:<22} ASR={m['asr']:.0%} "
                  f"RD={m['rd']:.4f} CI={m['ci']:.4f}")

            if attack_type in ADVANCED_ATTACK_TYPES:
                adv_asr.append(m["asr"])
            else:
                canon_asr.append(m["asr"])

        config_results["avg_asr_canon"] = round(
            sum(canon_asr) / len(canon_asr), 3)
        config_results["avg_asr_adv"]   = round(
            sum(adv_asr) / len(adv_asr), 3)
        all_results.append(config_results)

    return all_results


def print_semantic_table(results):
    print(f"\n{'='*68}")
    print("ABLATION SÉMANTIQUE — Heuristique vs Défenses Sémantiques")
    print(f"{'='*68}")
    print(f"{'Configuration':<20} {'Canon ASR':>10} {'Adv ASR':>9}")
    print("─"*42)
    for r in results:
        marker = " ←" if r["config"] == "Semantic full" else ""
        print(
            f"{r['config']:<20} "
            f"{r['avg_asr_canon']:>9.1%} "
            f"{r['avg_asr_adv']:>8.1%}"
            f"{marker}"
        )
    print(f"{'='*68}")
    print("Canon = 6 attaques canoniques | Adv = 2 attaques avancées")

    sem_full = next(r for r in results if r["config"] == "Semantic full")
    heur     = next(r for r in results if r["config"] == "All heuristic")
    print(f"\nGain Semantic full vs All heuristic :")
    print(f"  Canon : {heur['avg_asr_canon']:.1%} → "
          f"{sem_full['avg_asr_canon']:.1%}")
    print(f"  Adv   : {heur['avg_asr_adv']:.1%} → "
          f"{sem_full['avg_asr_adv']:.1%}")


if __name__ == "__main__":
    print("=== Ablation Défenses Sémantiques ===\n")

    questions = load_questions("data/questions.json")
    clean_docs = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} documents")
    print(f"  {len(ALL_ATTACKS)} attaques (6 canoniques + 2 avancées)")

    results = run_semantic_ablation(questions, clean_docs)
    print_semantic_table(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/semantic_defense_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRésultats → results/semantic_defense_results.json")

    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_sem_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
