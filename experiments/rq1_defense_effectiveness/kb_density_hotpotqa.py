"""
ablation_hotpotqa.py — Ablation study sur HotpotQA
Papier : Thèse — IPI dans les systèmes RAG

Spécificités vs SQuAD/NQ :
  - KB très dense : ~498 documents pour 50 questions
  - Questions multi-hop (bridge + comparison)
  - Niveau : hard uniquement (seed=42)

Hypothèses testées :
  H1 : KB density effect encore plus fort sur HotpotQA
       (498 docs vs 49 NQ) → ASR sans défense proche de 0%
  H2 : Questions multi-hop plus résistantes aux IPI
       car le retriever doit trouver 2 docs support
  H3 : All defenses maintient ASR=0% sur HotpotQA
       → généralisabilité confirmée sur 3 datasets
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
from ragipi.metrics.sau import compute_sau, load_questions_meta
from ragipi.datasets.hotpotqa import load_hotpotqa_documents, load_hotpotqa_questions
from experiments.rq1_defense_effectiveness.kb_density_nq import (
    NQ_ADVERSARIAL_DOCS,
    NQ_SUCCESS_INDICATORS,
    ATTACKS_NQ,
    is_nq_attack_successful,
    create_nq_adversarial_docs,
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
    if use_filter:
        clean_docs, flagged = filter_documents(docs)
        if flagged:
            print(f"    [Filter] {len(flagged)} doc(s) retiré(s)")
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


def run_ablation_hotpotqa(questions, questions_meta, clean_docs):
    adv_docs = create_nq_adversarial_docs()

    print(f"\nCalcul baseline HotpotQA "
          f"({len(questions)}q, {len(clean_docs)} docs)...")
    chain_base = build_pipeline(
        clean_docs, "./chroma_hotpot_base", False, False
    )
    baseline = {q: chain_base.invoke(q) for q in questions}
    print(f"  Baseline calculée\n")

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*55}")
        print(f"Config : {config['name']}")

        for attack_type in ATTACKS_NQ:
            adversarial_doc = adv_docs[attack_type]
            poisoned = clean_docs + [adversarial_doc]
            persist = (
                f"./chroma_hotpot_"
                f"{config['name'].replace(' ','_')}_{attack_type}"
            )

            chain = build_pipeline(
                poisoned, persist,
                config["filter"], config["isolate"]
            )

            asr_n, rd_list, ci_list = 0, [], []

            for q in questions:
                response = chain.invoke(q)
                if config["verify"]:
                    _, response = verify_output(response)

                success = is_nq_attack_successful(attack_type, response)
                rd = compute_rd(baseline[q], response)
                ci = compute_ci(response)
                ar = compute_sau(
                    response,
                    questions_meta[questions.index(q)].get("expected_keywords", [])
                )

                if success:
                    asr_n += 1
                rd_list.append(rd)
                ci_list.append(ci)
                ar_list = getattr(run_ablation_hotpotqa, '_ar_tmp', [])
                ar_list.append(ar)
                run_ablation_hotpotqa._ar_tmp = ar_list

            m = {
                "attack_type": attack_type,
                "asr": round(asr_n / len(questions), 3),
                "rd":  round(sum(rd_list) / len(rd_list), 4),
                "ci":  round(sum(ci_list) / len(ci_list), 4),
            }
            config_results["attacks"].append(m)
            print(f"  {attack_type:<25} ASR={m['asr']:.0%} "
                  f"RD={m['rd']:.4f} CI={m['ci']:.4f}")

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


def print_comparison_table(results_hotpot, results_nq=None,
                           results_custom=None):
    print(f"\n{'='*72}")
    print("COMPARAISON — Custom vs NQ vs HotpotQA")
    print(f"{'='*72}")

    headers = ["Configuration", "Custom ASR", "NQ ASR", "HotpotQA ASR"]
    print(f"{'Configuration':<16} {'Custom':>8} {'NQ 50q':>8} "
          f"{'HotpotQA':>10}")
    print("─"*46)

    for i, r in enumerate(results_hotpot):
        custom_asr = (results_custom[i]["avg_asr"]
                      if results_custom else "---")
        nq_asr     = (results_nq[i]["avg_asr"]
                      if results_nq else "---")

        custom_str = (f"{custom_asr:.1%}" if isinstance(custom_asr, float)
                      else custom_asr)
        nq_str     = (f"{nq_asr:.1%}" if isinstance(nq_asr, float)
                      else nq_asr)

        print(
            f"{r['config']:<16} "
            f"{custom_str:>8} "
            f"{nq_str:>8} "
            f"{r['avg_asr']:>9.1%}"
        )

    print(f"{'='*72}")
    print("\nKB density : Custom=5 docs | NQ=49 docs | HotpotQA=498 docs")
    print("Hypothèse  : ASR ↓ quand densité KB ↑")

    # Vérification hypothèse KB density
    no_def = next(r for r in results_hotpot
                  if r["config"] == "No defense")
    print(f"\nASR No defense sur HotpotQA : {no_def['avg_asr']:.1%}")
    if no_def["avg_asr"] < 0.01:
        print("→ KB density effect CONFIRMÉ : 498 docs → ASR≈0% sans défense")
    else:
        print(f"→ KB density effect PARTIEL : ASR={no_def['avg_asr']:.1%}")


if __name__ == "__main__":
    print("=== Ablation HotpotQA — KB dense (498 docs) ===\n")

    hotpot_dir = Path("data/hotpotqa")
    if not hotpot_dir.exists() or not list(
            hotpot_dir.glob("hotpot_*.txt")):
        print("HotpotQA non trouvé. Lance : python load_hotpotqa.py")
        exit(1)

    questions_meta = load_questions_meta("data/hotpotqa_questions.json")
    questions  = [q["text"] for q in questions_meta]
    clean_docs = load_hotpotqa_documents()
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    results_hotpot = run_ablation_hotpotqa(questions, questions_meta, clean_docs)

    # Tableau résumé HotpotQA seul
    print(f"\n{'='*55}")
    print("ABLATION STUDY — HotpotQA (498 docs, 50 questions)")
    print(f"{'='*55}")
    print(f"{'Configuration':<16} {'ASR':>8} {'RD':>8} {'CI':>8} {'AR':>8}")
    print("─"*50)
    for r in results_hotpot:
        print(
            f"{r['config']:<16} "
            f"{r['avg_asr']:>7.1%} "
            f"{r['avg_rd']:>8.4f} "
            f"{r['avg_ci']:>8.4f} "
            f"{r.get('avg_ar', 0):>8.4f}"
        )
    print(f"{'='*55}")

    # Comparaison avec résultats précédents
    custom_file = Path("results/ablation_v2_results.json")
    nq_file     = Path("results/ablation_nq_results.json")

    results_custom = None
    results_nq     = None

    if custom_file.exists():
        with open(custom_file) as f:
            results_custom = json.load(f)
    if nq_file.exists():
        with open(nq_file) as f:
            results_nq = json.load(f)

    print_comparison_table(results_hotpot, results_nq, results_custom)

    # Sauvegarde
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_hotpotqa_results.json", "w") as f:
        json.dump(results_hotpot, f, indent=2, ensure_ascii=False)
    print("\nRésultats → results/ablation_hotpotqa_results.json")

    # Nettoyage
    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_hotpot_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
