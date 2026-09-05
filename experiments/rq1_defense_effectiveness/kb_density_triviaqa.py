"""
ablation_triviaqa.py — Ablation study sur TriviaQA Wikipedia
Thèse : Indirect Prompt Injection in RAG

Spécificité : KB légère (76 docs, ratio 1.5/question)
→ KB density effect intermédiaire attendu entre Custom (5) et NQ (49)
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
from ragipi.datasets.triviaqa import load_triviaqa_documents, load_triviaqa_questions
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
    if use_filter:
        clean_docs, flagged = filter_documents(docs)
        if flagged:
            print(f"    [Filter] {len(flagged)} doc(s) retiré(s)")
    else:
        clean_docs = docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(clean_docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    def fmt(r):
        return "\n\n---\n\n".join(
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(r)
        )

    prompt = get_isolated_prompt() if use_isolate \
        else ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT), ("human", "{question}"),
        ])

    return (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def run_ablation_triviaqa(questions, questions_meta, clean_docs):
    adv_docs = create_nq_adversarial_docs()

    print(f"\nCalcul baseline TriviaQA "
          f"({len(questions)}q, {len(clean_docs)} docs)...")
    chain_base = build_pipeline(
        clean_docs, "./chroma_trivia_base", False, False
    )
    baseline = {q: chain_base.invoke(q) for q in questions}
    print(f"  Baseline calculée\n")

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*55}")
        print(f"Config : {config['name']}")

        for attack_type in ATTACKS_NQ:
            adv_doc = adv_docs[attack_type]
            poisoned = clean_docs + [adv_doc]
            persist = (
                f"./chroma_trivia_"
                f"{config['name'].replace(' ','_')}_{attack_type}"
            )

            chain = build_pipeline(
                poisoned, persist,
                config["filter"], config["isolate"]
            )

            asr_n, rd_l, ci_l, ar_l = 0, [], [], []

            for i, q in enumerate(questions):
                resp = chain.invoke(q)
                if config["verify"]:
                    _, resp = verify_output(resp)

                if is_nq_attack_successful(attack_type, resp):
                    asr_n += 1
                rd_l.append(compute_rd(baseline[q], resp))
                ci_l.append(compute_ci(resp))
                ar_l.append(compute_sau(
                    resp,
                    questions_meta[i].get("expected_keywords", [])
                ))

            n = len(questions)
            m = {
                "attack_type": attack_type,
                "asr": round(asr_n / n, 3),
                "rd":  round(sum(rd_l) / n, 4),
                "ci":  round(sum(ci_l) / n, 4),
                "ar":  round(sum(ar_l) / n, 4),
            }
            config_results["attacks"].append(m)
            print(f"  {attack_type:<25} ASR={m['asr']:.0%} "
                  f"RD={m['rd']:.4f} CI={m['ci']:.4f} "
                  f"AR={m['ar']:.4f}")

        config_results["avg_asr"] = round(
            sum(a["asr"] for a in config_results["attacks"])
            / len(ATTACKS_NQ), 3)
        config_results["avg_rd"]  = round(
            sum(a["rd"]  for a in config_results["attacks"])
            / len(ATTACKS_NQ), 4)
        config_results["avg_ci"]  = round(
            sum(a["ci"]  for a in config_results["attacks"])
            / len(ATTACKS_NQ), 4)
        config_results["avg_ar"]  = round(
            sum(a["ar"]  for a in config_results["attacks"])
            / len(ATTACKS_NQ), 4)
        all_results.append(config_results)

    return all_results


if __name__ == "__main__":
    print("=== Ablation TriviaQA — KB légère (76 docs) ===\n")

    questions_meta = load_questions_meta("data/triviaqa_questions.json")
    questions  = [q["text"] for q in questions_meta]
    clean_docs = load_triviaqa_documents()
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    results = run_ablation_triviaqa(questions, questions_meta, clean_docs)

    print(f"\n{'='*62}")
    print("ABLATION STUDY — TriviaQA (76 docs, 50 questions, 8 att.)")
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

    # KB density comparison
    print("\nKB density effect — 5 datasets :")
    print(f"  Custom    (  5 docs) : 23.3% No defense")
    nd = next(r for r in results if r["config"] == "No defense")
    print(f"  TriviaQA  ( 76 docs) : {nd['avg_asr']:.1%} No defense")
    print(f"  MITRE     ( 50 docs) :  0.0% No defense")
    print(f"  NQ        ( 49 docs) :  0.7% No defense")
    print(f"  HotpotQA  (498 docs) :  0.0% No defense")

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_triviaqa_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nRésultats → results/ablation_triviaqa_results.json")

    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_trivia_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
