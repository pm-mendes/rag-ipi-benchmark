"""
ablation_retriever.py — BM25 vs Dense vs Hybrid retriever
Hypothèse : BM25 plus résistant aux attaques IPI car les
préfixes adversariaux maximisent la similarité sémantique
(dense) mais pas la correspondance lexicale exacte (BM25).

Configurations :
  Dense  : all-MiniLM-L6-v2 (config actuelle)
  BM25   : retrieval lexical exact
  Hybrid : RRF (Reciprocal Rank Fusion) BM25 + Dense
"""

import json, shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
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
TOP_K           = 3


class HybridRetriever(BaseRetriever):
    """
    Reciprocal Rank Fusion (RRF) de BM25 + Dense.
    Score RRF = sum(1 / (k + rank_i)) pour chaque retriever.
    k=60 est le paramètre standard de RRF.
    """
    bm25: BM25Retriever
    dense_retriever: object
    top_k: int = TOP_K
    rrf_k: int = 60

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query, *, run_manager=None):
        # Récupérer depuis les deux retrievers
        bm25_docs  = self.bm25.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)

        # RRF scoring
        scores = {}
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1/(self.rrf_k + rank + 1)
            scores[key+"__doc__"] = doc

        for rank, doc in enumerate(dense_docs):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1/(self.rrf_k + rank + 1)
            scores[key+"__doc__"] = doc

        # Trier par score RRF décroissant
        doc_scores = [
            (scores[k+"__doc__"], scores[k])
            for k in scores if not k.endswith("__doc__")
            and k+"__doc__" in scores
        ]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:self.top_k]]


def build_chain_dense(docs, persist_dir):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    vs = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=persist_dir)
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    return retriever, chunks


def build_chain_bm25(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    retriever = BM25Retriever.from_documents(chunks, k=TOP_K)
    return retriever, chunks


def make_rag_chain(retriever):
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


def run_retriever_ablation(questions, questions_meta, clean_docs):
    results = []

    for atk_name, adv_file in ATTACKS:
        adv_doc = Document(
            page_content=Path(adv_file).read_text(encoding="utf-8"),
            metadata={"source": adv_file, "type": "adversarial"}
        )
        poisoned = clean_docs + [adv_doc]

        # Splitter commun
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(poisoned)

        # Baseline Dense (clean)
        clean_chunks = splitter.split_documents(clean_docs)
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        persist = f"./chroma_ret_base_{atk_name}"
        if Path(persist).exists():
            shutil.rmtree(persist)
        vs_base = Chroma.from_documents(
            documents=clean_chunks, embedding=emb,
            persist_directory=persist)
        ret_base = vs_base.as_retriever(search_kwargs={"k": TOP_K})
        chain_base = make_rag_chain(ret_base)
        baseline = {q: chain_base.invoke(q) for q in questions}

        # Dense retriever
        persist_d = f"./chroma_ret_dense_{atk_name}"
        if Path(persist_d).exists():
            shutil.rmtree(persist_d)
        vs_d = Chroma.from_documents(
            documents=chunks, embedding=emb,
            persist_directory=persist_d)
        ret_dense = vs_d.as_retriever(search_kwargs={"k": TOP_K})

        # BM25 retriever
        ret_bm25 = BM25Retriever.from_documents(chunks, k=TOP_K)

        # Hybrid retriever
        ret_hybrid = HybridRetriever(
            bm25=ret_bm25,
            dense_retriever=ret_dense,
            top_k=TOP_K
        )

        retrievers = {
            "Dense":  ret_dense,
            "BM25":   ret_bm25,
            "Hybrid": ret_hybrid,
        }

        atk_results = {"attack": atk_name}
        print(f"\n  {atk_name}")

        for ret_name, retriever in retrievers.items():
            chain = make_rag_chain(retriever)
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
                "asr": round(asr_n / n, 3),
                "rd":  round(sum(rd_l) / n, 4),
                "ci":  round(sum(ci_l) / n, 4),
                "ar":  round(sum(ar_l) / n, 4),
            }
            atk_results[ret_name] = m
            print(f"    {ret_name:<8} ASR={m['asr']:.0%} RD={m['rd']:.4f} CI={m['ci']:.4f} AR={m['ar']:.4f}")

        results.append(atk_results)

    return results


def print_summary(results):
    print(f"\n{'='*62}")
    print("RÉSUMÉ — BM25 vs Dense vs Hybrid (No defense, 10q, 6 att.)")
    print(f"{'='*62}")
    print(f"{'Retriever':<10} {'ASR avg':>9} {'RD avg':>8} {'AR avg':>8}")
    print("─"*38)

    for ret in ["Dense", "BM25", "Hybrid"]:
        asr_avg = sum(r[ret]["asr"] for r in results) / len(results)
        rd_avg  = sum(r[ret]["rd"]  for r in results) / len(results)
        ar_avg  = sum(r[ret]["ar"]  for r in results) / len(results)
        print(f"{ret:<10} {asr_avg:>8.1%} {rd_avg:>8.4f} {ar_avg:>8.4f}")

    print(f"{'='*62}")

    # Analyse
    dense_asr  = sum(r["Dense"]["asr"]  for r in results) / len(results)
    bm25_asr   = sum(r["BM25"]["asr"]   for r in results) / len(results)
    hybrid_asr = sum(r["Hybrid"]["asr"] for r in results) / len(results)

    if bm25_asr < dense_asr:
        diff = dense_asr - bm25_asr
        print(f"\n→ BM25 plus résistant que Dense : "
              f"ASR {bm25_asr:.1%} vs {dense_asr:.1%} "
              f"(-{diff:.1%})")
        print("  Hypothèse confirmée : les préfixes adversariaux "
              "optimisent la similarité sémantique, pas lexicale.")
    else:
        print(f"\n→ Dense aussi résistant que BM25 sur ce dataset.")

    if hybrid_asr <= min(dense_asr, bm25_asr):
        print("→ Hybrid meilleur ou égal aux deux : "
              "RRF bénéficie des deux signaux.")


if __name__ == "__main__":
    print("=== BM25 vs Dense vs Hybrid Retriever — IPI Ablation ===\n")

    questions_meta = load_questions_meta("data/questions.json")
    questions_meta = questions_meta[:10]
    questions  = [q["text"] for q in questions_meta]
    clean_docs = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    results = run_retriever_ablation(
        questions, questions_meta, clean_docs)
    print_summary(results)

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_retriever_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats → results/ablation_retriever_results.json")

    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_ret_*/"):
        shutil.rmtree(d)
    print("  Terminé.")
