"""
ablation_mitre.py — Ablation study sur MITRE ATT&CK Enterprise
Thèse : Indirect Prompt Injection in RAG

Spécificité unique de ce dataset :
  - KB = descriptions de techniques d'attaque réelles
  - Documents adversariaux se camouflent naturellement
    dans le contenu de sécurité légitime
  - Teste si L1b (LLM-judge) fait des faux positifs sur
    une KB entièrement composée de contenu "offensif légitime"
  - Cas d'usage réel : RAG assistant SOC / threat intelligence

Hypothèses :
  H1 : L1b risque des faux positifs (descriptions d'attaques
       ≈ injections adversariales pour un juge naïf)
  H2 : KB density effect partiel (50 docs < 498 HotpotQA)
       → ASR No defense entre Custom (23%) et NQ (0.7%)
  H3 : All defenses maintient ASR=0%
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
from ragipi.pipeline import SYSTEM_PROMPT
from ragipi.metrics.core import compute_rd, compute_ci
from ragipi.metrics.sau import compute_sau, load_questions_meta
from ragipi.datasets.mitre import load_mitre_documents, load_mitre_questions
from experiments.rq1_defense_effectiveness.kb_density_nq import (
    ATTACKS_NQ, is_nq_attack_successful, create_nq_adversarial_docs
)

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


def build_pipeline(docs, persist_dir, use_filter, use_isolate):
    """Pipeline RAG avec suivi des faux positifs L1 sur KB MITRE."""
    if use_filter:
        clean_docs, flagged = filter_documents(docs)
        if flagged:
            # Sur MITRE, noter si des docs légitimes sont retirés
            flagged_names = [
                d.metadata.get("filename", "?") for d in flagged
            ]
            legit_flagged = [
                n for n in flagged_names if "mitre_T" in n
            ]
            if legit_flagged:
                print(f"    [Filter] ⚠ FAUX POSITIFS légitimes : "
                      f"{legit_flagged}")
            else:
                print(f"    [Filter] {len(flagged)} doc(s) adversarial(aux)"
                      f" retiré(s)")
    else:
        clean_docs = docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
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


def run_ablation_mitre(questions, questions_meta, clean_docs):
    adv_docs = create_nq_adversarial_docs()

    print(f"\nCalcul baseline MITRE "
          f"({len(questions)}q, {len(clean_docs)} docs)...")
    chain_base = build_pipeline(
        clean_docs, "./chroma_mitre_base", False, False
    )
    baseline = {q: chain_base.invoke(q) for q in questions}
    print(f"  Baseline calculée\n")

    all_results = []

    for config in CONFIGS:
        config_results = {
            "config": config["name"],
            "attacks": [],
            "fp_lexical": 0,  # faux positifs filtre lexical
        }
        print(f"\n{'─'*55}")
        print(f"Config : {config['name']}")

        for attack_type in ATTACKS_NQ:
            adversarial_doc = adv_docs[attack_type]
            poisoned = clean_docs + [adversarial_doc]
            persist = (
                f"./chroma_mitre_"
                f"{config['name'].replace(' ','_')}_{attack_type}"
            )

            chain = build_pipeline(
                poisoned, persist,
                config["filter"], config["isolate"]
            )

            asr_n, rd_list, ci_list, ar_list = 0, [], [], []

            for i, q in enumerate(questions):
                response = chain.invoke(q)
                if config["verify"]:
                    _, response = verify_output(response)

                success = is_nq_attack_successful(attack_type, response)
                rd = compute_rd(baseline[q], response)
                ci = compute_ci(response)
                ar = compute_sau(
                    response,
                    questions_meta[i].get("expected_keywords", [])
                )

                if success:
                    asr_n += 1
                rd_list.append(rd)
                ci_list.append(ci)
                ar_list.append(ar)

            m = {
                "attack_type": attack_type,
                "asr": round(asr_n / len(questions), 3),
                "rd":  round(sum(rd_list) / len(rd_list), 4),
                "ci":  round(sum(ci_list) / len(ci_list), 4),
                "ar":  round(sum(ar_list) / len(ar_list), 4),
            }
            config_results["attacks"].append(m)
            print(f"  {attack_type:<25} ASR={m['asr']:.0%} "
                  f"RD={m['rd']:.4f} CI={m['ci']:.4f} "
                  f"AR={m['ar']:.4f}")

        config_results["avg_asr"] = round(
            sum(a["asr"] for a in config_results["attacks"])
            / len(ATTACKS_NQ), 3
        )
        config_results["avg_rd"] = round(
            sum(a["rd"] for a in config_results["attacks"])
            / len(ATTACKS_NQ), 4
        )
        config_results["avg_ci"] = round(
            sum(a["ci"] for a in config_results["attacks"])
            / len(ATTACKS_NQ), 4
        )
        config_results["avg_ar"] = round(
            sum(a["ar"] for a in config_results["attacks"])
            / len(ATTACKS_NQ), 4
        )
        all_results.append(config_results)

    return all_results


def print_mitre_table(results):
    print(f"\n{'='*62}")
    print("ABLATION STUDY — MITRE ATT&CK (50 techniques, 30 questions)")
    print(f"{'='*62}")
    print(f"{'Configuration':<16} {'ASR':>7} {'RD':>8} "
          f"{'CI':>8} {'AR':>8}")
    print("─"*50)
    for r in results:
        print(
            f"{r['config']:<16} "
            f"{r['avg_asr']:>6.1%} "
            f"{r['avg_rd']:>8.4f} "
            f"{r['avg_ci']:>8.4f} "
            f"{r['avg_ar']:>8.4f}"
        )
    print(f"{'='*62}")

    # Analyse KB density
    no_def = next(r for r in results if r["config"] == "No defense")
    print(f"\nASR No defense (MITRE, 50 docs) : {no_def['avg_asr']:.1%}")
    print("Comparaison KB density effect :")
    print("  Custom (5 docs)    : 23.3%")
    print("  MITRE  (50 docs)   :", f"{no_def['avg_asr']:.1%}")
    print("  NQ     (49 docs)   :  0.7%")
    print("  HotpotQA (498 docs):  0.0%")


if __name__ == "__main__":
    print("=== Ablation MITRE ATT&CK Enterprise ===\n")

    mitre_dir = Path("data/mitre")
    if not list(mitre_dir.glob("mitre_T*.txt")):
        print("MITRE non trouvé. Lance : python load_mitre.py")
        exit(1)

    questions_meta = load_questions_meta("data/mitre_questions.json")
    questions  = [q["text"] for q in questions_meta]
    clean_docs = load_mitre_documents()
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    results = run_ablation_mitre(questions, questions_meta, clean_docs)
    print_mitre_table(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_mitre_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRésultats → results/ablation_mitre_results.json")

    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_mitre_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
