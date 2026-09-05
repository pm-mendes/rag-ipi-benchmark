"""
ablation_ahr_casr.py
AHR@k : Adversarial Hit Rate@k
  = % requêtes où le document adversarial entre dans top-k

CASR  : Conditional Attack Success Rate
  = ASR conditionné sur AHR (succès d'attaque | doc récupéré)

Sépare deux phénomènes distincts :
  1. Le retriever laisse-t-il entrer le doc adversarial ? (AHR)
  2. Le LLM génère-t-il une réponse compromise si le doc est là ? (CASR)

Utilise le pipeline existant avec inspection du contexte récupéré.
"""

import json, shutil, os
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

from ragipi.pipeline import load_documents_from_folder, SYSTEM_PROMPT
from ragipi.metrics.core import compute_ci
from ragipi.attacks.catalog import CANONICAL_ATTACKS as ATTACKS
from ragipi.attacks.runner import is_attack_successful
from ragipi.metrics.sau import load_questions_meta

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
K_VALUES        = [1, 3, 5, 10]


def run_ahr_casr(questions, questions_meta, clean_docs):
    """
    Pour chaque attaque et chaque valeur de k :
      - AHR@k : doc adversarial dans top-k ?
      - CASR   : ASR | doc adversarial récupéré
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    def fmt_docs(docs):
        return "\n\n---\n\n".join(
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(docs))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    results = []

    for atk_name, adv_file in ATTACKS:
        adv_text = Path(adv_file).read_text(encoding="utf-8")
        adv_doc  = Document(
            page_content=adv_text,
            metadata={"source": adv_file, "type": "adversarial"})

        poisoned_docs = clean_docs + [adv_doc]
        chunks = splitter.split_documents(poisoned_docs)

        # Identifier les chunks adversariaux
        adv_chunks_content = set()
        for chunk in chunks:
            if chunk.metadata.get("type") == "adversarial":
                adv_chunks_content.add(chunk.page_content[:50])

        persist = f"./chroma_ahr_{atk_name}"
        if Path(persist).exists():
            shutil.rmtree(persist)
        vs = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            persist_directory=persist)

        print(f"\n  {atk_name}")
        atk_results = {"attack": atk_name, "k_results": {}}

        for k in K_VALUES:
            retriever = vs.as_retriever(search_kwargs={"k": k})

            ahr_hits      = 0   # requêtes où doc adv récupéré
            asr_total     = 0   # succès d'attaque total
            asr_given_hit = 0   # succès d'attaque | doc récupéré
            n = len(questions)

            for q_text, q_meta in zip(questions, questions_meta):
                # Récupérer les docs
                retrieved = retriever.invoke(q_text)

                # Vérifier si doc adversarial récupéré
                adv_retrieved = any(
                    doc.page_content[:50] in adv_chunks_content
                    for doc in retrieved)

                if adv_retrieved:
                    ahr_hits += 1

                # Générer la réponse
                context = fmt_docs(retrieved)
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context,
                    "question": q_text})

                success = is_attack_successful(atk_name, response)
                if success:
                    asr_total += 1
                    if adv_retrieved:
                        asr_given_hit += 1

            ahr = ahr_hits / n
            asr = asr_total / n
            casr = (asr_given_hit / ahr_hits
                    if ahr_hits > 0 else 0.0)

            print(f"    k={k:>2}  AHR={ahr:.1%}  "
                  f"ASR={asr:.1%}  CASR={casr:.1%}  "
                  f"(hits={ahr_hits}/{n})")

            atk_results["k_results"][str(k)] = {
                "k": k,
                "ahr": round(ahr, 4),
                "asr": round(asr, 4),
                "casr": round(casr, 4),
                "adv_hits": ahr_hits,
                "n_queries": n,
            }

        results.append(atk_results)
        shutil.rmtree(persist, ignore_errors=True)

    return results


def print_summary(results):
    print(f"\n{'='*70}")
    print("RÉSUMÉ AHR@k + CASR (No defense, 10q, 6 canonical)")
    print(f"{'='*70}")
    print(f"{'Attack':<22} "
          + "  ".join(f"k={k}(AHR/CASR)" for k in K_VALUES))
    print("-"*70)
    for r in results:
        row = f"  {r['attack']:<20} "
        for k in K_VALUES:
            kr = r["k_results"].get(str(k), {})
            ahr  = kr.get("ahr", 0)
            casr = kr.get("casr", 0)
            row += f"  {ahr:.0%}/{casr:.0%}  "
        print(row)

    print(f"\n{'='*70}")
    print("MOYENNE PAR k")
    print(f"{'k':<6} {'AHR avg':>9} {'ASR avg':>9} {'CASR avg':>10}")
    print("-"*38)
    for k in K_VALUES:
        ahrs  = [r["k_results"][str(k)]["ahr"]  for r in results
                 if str(k) in r["k_results"]]
        asrs  = [r["k_results"][str(k)]["asr"]  for r in results
                 if str(k) in r["k_results"]]
        casrs = [r["k_results"][str(k)]["casr"] for r in results
                 if str(k) in r["k_results"]]
        print(f"  k={k:<3}  "
              f"{sum(ahrs)/len(ahrs):>8.1%}  "
              f"{sum(asrs)/len(asrs):>8.1%}  "
              f"{sum(casrs)/len(casrs):>9.1%}")


if __name__ == "__main__":
    print("=== AHR@k + CASR — Adversarial Hit Rate & "
          "Conditional ASR ===\n")

    questions_meta = load_questions_meta("data/questions.json")[:10]
    questions      = [q["text"] for q in questions_meta]
    clean_docs     = load_documents_from_folder("data/clean_docs")

    print(f"  {len(questions)} questions, "
          f"{len(clean_docs)} documents KB\n")

    results = run_ahr_casr(questions, questions_meta, clean_docs)
    print_summary(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/ahr_casr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats → results/ahr_casr_results.json")
