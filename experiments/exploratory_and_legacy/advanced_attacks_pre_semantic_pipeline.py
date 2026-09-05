"""
ablation_advanced.py — Évaluation des 2 attaques avancées
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Attaques avancées :
  UNICODE_OVERRIDE   : injection via caractères Unicode zero-width
                       contourne les regex du Content Filter
  IMPLICIT_INJECTION : injection sémantique sans mots-clés explicites
                       contourne la détection lexicale

Objectif scientifique :
  Montrer les limites des défenses heuristiques face à des
  attaques sophistiquées — et valider que l'isolation de
  contexte (L2) offre une protection résiduelle même contre
  ces attaques.
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
from ragipi.pipeline import load_documents_from_folder, SYSTEM_PROMPT
from ragipi.metrics.core import compute_rd, compute_ci, load_questions
from ragipi.attacks.runner import is_attack_successful

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TOP_K           = int(os.getenv("TOP_K", 3))

CONFIGS = [
    {"name": "No defense",   "filter": False, "isolate": False, "verify": False},
    {"name": "Filter only",  "filter": True,  "isolate": False, "verify": False},
    {"name": "Isolate only", "filter": False, "isolate": True,  "verify": False},
    {"name": "Verify only",  "filter": False, "isolate": False, "verify": True},
    {"name": "All defenses", "filter": True,  "isolate": True,  "verify": True},
]

ADVANCED_ATTACKS = [
    ("UNICODE_OVERRIDE",   "data/adversarial/unicode_override.txt"),
    ("IMPLICIT_INJECTION", "data/adversarial/implicit_injection.txt"),
]

# Inclure aussi les attaques canoniques pour comparaison
CANONICAL_ATTACKS = [
    ("OVERRIDE", "data/adversarial/override.txt"),
    ("DENIAL",   "data/adversarial/denial.txt"),
]


def build_pipeline(docs, persist_dir, use_filter, use_isolate):
    if use_filter:
        clean_docs, flagged = filter_documents(docs)
        if flagged:
            print(f"    [Filter] {len(flagged)} doc(s) retiré(s)")
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
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(retrieved)
        )

    prompt = get_isolated_prompt() if use_isolate \
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


def run_advanced_ablation(questions, clean_docs, attacks, label=""):
    """Ablation sur les attaques avancées."""

    print(f"\nCalcul baseline ({label})...")
    chain_base = build_pipeline(
        clean_docs, f"./chroma_adv_base_{label}", False, False
    )
    baseline = {q: chain_base.invoke(q) for q in questions}

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*50}")
        print(f"Config : {config['name']}")

        for attack_type, adv_file in attacks:
            adv_content = Path(adv_file).read_text(encoding="utf-8")

            # Vérifier si le filtre détecte l'attaque
            test_doc = Document(
                page_content=adv_content,
                metadata={"source": adv_file, "type": "adversarial",
                          "attack_type": attack_type}
            )
            _, flagged = filter_documents([test_doc])
            filter_detects = len(flagged) > 0

            poisoned = clean_docs + [test_doc]
            persist = (f"./chroma_adv_{label}_"
                       f"{config['name'].replace(' ','_')}_{attack_type}")

            chain = build_pipeline(
                poisoned, persist,
                config["filter"], config["isolate"]
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
                "attack_type":    attack_type,
                "filter_detects": filter_detects,
                "asr": round(asr_n / len(questions), 3),
                "rd":  round(sum(rd_list) / len(rd_list), 4),
                "ci":  round(sum(ci_list) / len(ci_list), 4),
            }
            config_results["attacks"].append(m)

            detect_str = "✓ détecté" if filter_detects else "✗ non détecté"
            print(f"  {attack_type:<22} [{detect_str}] "
                  f"ASR={m['asr']:.0%} RD={m['rd']:.4f} CI={m['ci']:.4f}")

        config_results["avg_asr"] = round(
            sum(a["asr"] for a in config_results["attacks"])
            / len(attacks), 3
        )
        all_results.append(config_results)

    return all_results


def print_advanced_table(results_adv, results_canon):
    """Tableau comparatif attaques naïves vs avancées."""
    print(f"\n{'='*65}")
    print("COMPARAISON — Attaques canoniques vs Avancées")
    print(f"{'='*65}")
    print(f"{'Configuration':<16} {'Canon. ASR':>12} {'Avancées ASR':>14}")
    print("─"*44)
    for rc, ra in zip(results_canon, results_adv):
        print(
            f"{rc['config']:<16} "
            f"{rc['avg_asr']:>11.1%} "
            f"{ra['avg_asr']:>13.1%}"
        )
    print(f"{'='*65}")
    print("\nNote : les attaques avancées (Unicode, implicite) montrent")
    print("les limites des défenses heuristiques et motivent des")
    print("approches sémantiques (LLM-judge) comme travaux futurs.")


if __name__ == "__main__":
    print("=== Évaluation Attaques Avancées ===\n")

    questions = load_questions("data/questions.json")
    clean_docs = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    print("--- Attaques AVANCÉES (Unicode + Implicite) ---")
    results_adv = run_advanced_ablation(
        questions, clean_docs, ADVANCED_ATTACKS, "adv"
    )

    print("\n--- Attaques CANONIQUES (référence) ---")
    results_canon = run_advanced_ablation(
        questions, clean_docs, CANONICAL_ATTACKS, "canon"
    )

    print_advanced_table(results_adv, results_canon)

    Path("results").mkdir(exist_ok=True)
    with open("results/advanced_attacks_results.json", "w") as f:
        json.dump({
            "advanced": results_adv,
            "canonical": results_canon,
        }, f, indent=2, ensure_ascii=False)
    print("\nRésultats sauvegardés dans results/advanced_attacks_results.json")

    for d in Path(".").glob("chroma_adv_*/"):
        shutil.rmtree(d)
    print("ChromaDB nettoyé.")
