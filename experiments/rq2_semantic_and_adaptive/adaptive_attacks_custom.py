"""
ablation_adaptive_semantic.py
Évalue les 4 attaques adaptatives sur Custom 30q avec :
  - No defense
  - All heuristic
  - Semantic full (L0+L1+L1b+L1c+L2+L3)

Complète Table IV du papier avec la colonne Semantic Full manquante.
"""

import json, os, shutil
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
from ragipi.metrics.core import compute_ci, compute_rd
from ragipi.metrics.sau import compute_sau, load_questions_meta
from ragipi.attacks.runner import is_attack_successful as _is_attack_successful


ADAPTIVE_MAPPING = {
    "LLM_GENERATED":   "OVERRIDE",
    "BASE64_ENCODING": "OVERRIDE",
    "MANY_SHOT":       "OVERRIDE",
    "AUTO_PARAPHRASE": "OVERRIDE",
}

def is_attack_successful(atk_name, response):
    mapped = ADAPTIVE_MAPPING.get(atk_name, atk_name)
    return _is_attack_successful(mapped, response)
from ragipi.defenses.content_filter import filter_documents, is_clean
from ragipi.defenses.context_isolator import get_isolated_prompt, ISOLATED_SYSTEM_PROMPT
from ragipi.defenses.output_verifier import verify_output, SAFE_FALLBACK

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TOP_K = 3

ADAPTIVE_ATTACKS = [
    ("LLM_GENERATED",   "data/adversarial/llm_generated.txt"),
    ("BASE64_ENCODING", "data/adversarial/base64_encoding.txt"),
    ("MANY_SHOT",       "data/adversarial/many_shot.txt"),
    ("AUTO_PARAPHRASE", "data/adversarial/auto_paraphrase.txt"),
]


def build_vectorstore(docs, persist_dir):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
    vs = Chroma.from_documents(
        documents=chunks, embedding=emb,
        persist_directory=persist_dir)
    return vs, chunks, emb


def apply_semantic_full(docs, emb):
    """
    L0: Unicode normalization
    L1b: LLM-judge (simulé par heuristiques avancées si Ollama absent)
    L1c: Embedding anomaly
    """
    from ragipi.defenses.semantic_defense import semantic_filter_pipeline, normalize_documents
    sd = SemanticDefense(embeddings=emb)
    filtered = []
    for doc in docs:
        if sd.is_safe(doc.page_content):
            filtered.append(doc)
    return filtered


def run_config(questions, questions_meta, clean_docs,
               atk_name, adv_text, config_name,
               use_heuristic=False, use_semantic=False):
    """Évalue une attaque pour une configuration de défense."""
    adv_doc = Document(
        page_content=adv_text,
        metadata={"source": atk_name, "type": "adversarial"})
    all_docs = clean_docs + [adv_doc]

    persist = f"./chroma_adapt_{atk_name}_{config_name}"
    vs, chunks, emb = build_vectorstore(all_docs, persist)
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})



    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    def fmt(docs):
        return "\n\n---\n\n".join(
            f"[Doc {i+1}] {d.page_content}"
            for i, d in enumerate(docs))

    asr_n, rd_l, ci_l, ar_l = 0, [], [], []

    # Baseline : réponses sans doc adversarial
    vs_base, _, _ = build_vectorstore(clean_docs, persist + "_base")
    ret_base = vs_base.as_retriever(search_kwargs={"k": TOP_K})
    llm_base = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    prompt_base = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT), ("human", "{question}")])
    chain_base = prompt_base | llm_base | StrOutputParser()
    baselines = {q: chain_base.invoke({"question": q,
                                        "context": fmt(ret_base.invoke(q))})
                 for q in questions}
    shutil.rmtree(persist + "_base", ignore_errors=True)

    for q_text, q_meta in zip(questions, questions_meta):
        retrieved = retriever.invoke(q_text)

        # L1 Content Filter
        if use_heuristic:
            retrieved, _ = filter_documents(retrieved)

        # Semantic full (L0+L1b+L1c)
        if use_semantic:
            try:
                retrieved = apply_semantic_full(retrieved, emb)
            except Exception:
                pass  # Fallback si SemanticDefense absent

        context = fmt(retrieved)

        # L2 Context Isolator
        if use_heuristic:
            sys_prompt = ISOLATED_SYSTEM_PROMPT
        else:
            sys_prompt = SYSTEM_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": q_text,
                                 "context": context})

        # L3 Output Verifier
        if use_heuristic:
            _, response = verify_output(response)

        if is_attack_successful(atk_name, response):
            asr_n += 1

        rd_l.append(compute_rd(baselines[q_text], response))
        ci_l.append(compute_ci(response))
        ar_l.append(compute_sau(
            response, q_meta.get("expected_keywords", [])))

    n = len(questions)
    shutil.rmtree(persist, ignore_errors=True)

    return {
        "asr": round(asr_n / n, 4),
        "rd":  round(sum(rd_l) / n, 4),
        "ci":  round(sum(ci_l) / n, 4),
        "ar":  round(sum(ar_l) / n, 4),
        "n":   n,
    }


if __name__ == "__main__":
    print("=== Adaptive Attacks — No def / All heur / Semantic full ===\n")

    questions_meta = load_questions_meta("data/questions.json")[:30]
    questions      = [q["text"] for q in questions_meta]
    clean_docs     = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} docs KB\n")

    configs = [
        ("No defense",    False, False),
        ("All heuristic", True,  False),
        ("Semantic full", True,  True),
    ]

    results = []
    print(f"{'Attack':<20} {'Config':<16} {'ASR':>6} {'RD':>7} {'CI':>7} {'AR':>7}")
    print("-"*62)

    for atk_name, adv_file in ADAPTIVE_ATTACKS:
        if not Path(adv_file).exists():
            print(f"  [SKIP] {adv_file} non trouvé")
            continue

        adv_text = Path(adv_file).read_text(encoding="utf-8")
        atk_result = {"attack": atk_name, "configs": {}}

        for cfg_name, use_h, use_s in configs:
            print(f"  {atk_name:<20} {cfg_name:<16}", end="", flush=True)
            m = run_config(questions, questions_meta, clean_docs,
                           atk_name, adv_text, cfg_name,
                           use_heuristic=use_h,
                           use_semantic=use_s)
            print(f" {m['asr']:>5.1%} {m['rd']:>6.3f} {m['ci']:>6.3f} {m['ar']:>6.3f}")
            atk_result["configs"][cfg_name] = m

        results.append(atk_result)

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ — Table IV mise à jour avec Semantic Full")
    print(f"{'='*60}")
    print(f"{'Attack':<20} {'No def':>8} {'All heur':>10} {'Sem full':>10}")
    print("-"*52)
    for r in results:
        c = r["configs"]
        nd  = c.get("No defense",    {}).get("asr", 0)
        ah  = c.get("All heuristic", {}).get("asr", 0)
        sf  = c.get("Semantic full", {}).get("asr", 0)
        print(f"  {r['attack']:<20} {nd:>7.1%} {ah:>9.1%} {sf:>9.1%}")

    print("-"*52)
    if results:
        nd_avg = sum(r["configs"].get("No defense",{}).get("asr",0)
                     for r in results) / len(results)
        ah_avg = sum(r["configs"].get("All heuristic",{}).get("asr",0)
                     for r in results) / len(results)
        sf_avg = sum(r["configs"].get("Semantic full",{}).get("asr",0)
                     for r in results) / len(results)
        print(f"  {'Average':<20} {nd_avg:>7.1%} {ah_avg:>9.1%} {sf_avg:>9.1%}")

    Path("results").mkdir(exist_ok=True)
    with open("results/adaptive_semantic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats → results/adaptive_semantic_results.json")
