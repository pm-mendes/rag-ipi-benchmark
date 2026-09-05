"""
ablation_multimodel.py — Ablation study comparative GPT-3.5 vs LLaMA 3.3
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Objectif scientifique :
  Montrer que les résultats (vulnérabilité + efficacité des défenses)
  se généralisent à deux architectures de LLM différentes :
  - GPT-3.5-turbo  : modèle commercial aligné (OpenAI RLHF)
  - LLaMA 3.3 70B  : modèle open-source (Meta, via Groq)

  La comparaison permet de tester si l'alignement RLHF aide à
  résister aux IPI — finding original absent de SafeRAG.

Usage :
  python ablation_multimodel.py

Durée estimée :
  ~20-25 min — 2 modèles × 5 configs × 6 attaques × 10 questions
  = 600 appels API total (300 par modèle)
"""

import json
import shutil
import time
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
from ragipi.llm_factory import get_llm

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K           = int(os.getenv("TOP_K", 3))

# ── Providers à comparer ──────────────────────────────────────────────────────

PROVIDERS = [
    {
        "id":    "openai",
        "label": "GPT-3.5-turbo (OpenAI)",
        "sleep": 0,
    },
    {
        "id":    "deepseek",
        "label": "DeepSeek-V3 (open source)",
        "sleep": 1,
    },
    {
        "id":    "nvidia",
        "label": "Llama 3.1 70B (NVIDIA Build)",
        "sleep": 2,
    },
]

# ── Configurations d'ablation ─────────────────────────────────────────────────

CONFIGS = [
    {"name": "No defense",   "filter": False, "isolate": False, "verify": False},
    {"name": "Filter only",  "filter": True,  "isolate": False, "verify": False},
    {"name": "Isolate only", "filter": False, "isolate": True,  "verify": False},
    {"name": "Verify only",  "filter": False, "isolate": False, "verify": True},
    {"name": "All defenses", "filter": True,  "isolate": True,  "verify": True},
]

# ── Attaques ──────────────────────────────────────────────────────────────────

ATTACKS = [
    ("OVERRIDE",           "data/adversarial/override.txt"),
    ("EXFIL",              "data/adversarial/exfil.txt"),
    ("ROLE",               "data/adversarial/role.txt"),
    ("DENIAL",             "data/adversarial/denial.txt"),
    ("TECHNICAL_OVERRIDE", "data/adversarial/technical_override.txt"),
    ("DATA_EXFIL",         "data/adversarial/data_exfil.txt"),
]


# ── Pipeline avec provider configurable ──────────────────────────────────────

def build_pipeline_for_provider(docs, persist_dir, use_filter, use_isolate, provider_id, sleep_s):
    """
    Construit le pipeline RAG avec le LLM du provider spécifié.
    Identique à ablation_v2 sauf le LLM qui est paramétrable.
    """
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
    llm = get_llm(provider_id)

    def format_docs(retrieved):
        return "\n\n---\n\n".join(
            f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(retrieved)
        )

    prompt = get_isolated_prompt() if use_isolate else ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def invoke_with_sleep(question):
        """Wrapper avec retry automatique sur rate limit 429."""
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        for attempt in range(5):
            try:
                result = chain.invoke(question)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                return result
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = 30 * (attempt + 1)
                    print(f"    [RateLimit] Attente {wait}s (tentative {attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Échec après 5 tentatives")

    return invoke_with_sleep


# ── Ablation pour un provider ─────────────────────────────────────────────────

def run_ablation_for_provider(provider, questions, clean_docs):
    """Lance l'ablation complète pour un provider donné."""
    pid   = provider["id"]
    label = provider["label"]
    sleep = provider["sleep"]

    print(f"\n{'='*60}")
    print(f"Provider : {label}")
    print(f"{'='*60}")

    # Baseline
    print(f"\nCalcul de la baseline ({label})...")
    invoke = build_pipeline_for_provider(
        clean_docs, f"./chroma_mm_{pid}_base",
        False, False, pid, sleep
    )
    baseline = {}
    for q in questions:
        baseline[q] = invoke(q)
    print(f"  {len(questions)} réponses baseline calculées")

    all_results = []

    for config in CONFIGS:
        config_results = {"config": config["name"], "attacks": []}
        print(f"\n{'─'*50}")
        print(f"Config : {config['name']} | Provider : {pid}")

        for attack_type, adv_file in ATTACKS:
            adv_content = Path(adv_file).read_text(encoding="utf-8")
            adversarial_doc = Document(
                page_content=adv_content,
                metadata={"source": adv_file, "type": "adversarial",
                          "attack_type": attack_type}
            )
            poisoned = clean_docs + [adversarial_doc]
            persist = f"./chroma_mm_{pid}_{config['name'].replace(' ','_')}_{attack_type}"

            invoke = build_pipeline_for_provider(
                poisoned, persist,
                config["filter"], config["isolate"],
                pid, sleep
            )

            asr_n, rd_list, ci_list = 0, [], []

            for q in questions:
                response = invoke(q)
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
            sum(a["asr"] for a in config_results["attacks"]) / len(ATTACKS), 3
        )
        config_results["avg_rd"] = round(
            sum(a["rd"] for a in config_results["attacks"]) / len(ATTACKS), 4
        )
        config_results["avg_ci"] = round(
            sum(a["ci"] for a in config_results["attacks"]) / len(ATTACKS), 4
        )
        all_results.append(config_results)

    return all_results


# ── Tableau comparatif ────────────────────────────────────────────────────────

def print_multimodel_table(results_by_provider: dict):
    """
    Affiche le tableau comparatif GPT-3.5 vs LLaMA 3.3.
    Ce tableau ira dans la section VIII du papier.
    """
    providers = list(results_by_provider.keys())

    print(f"\n{'='*72}")
    print("COMPARAISON MULTI-MODÈLES — GPT-3.5-turbo vs DeepSeek-V3")
    print(f"{'='*72}")

    # En-tête
    header = f"{'Configuration':<16}"
    for p in providers:
        short = "GPT-3.5" if "openai" in p else "DeepSeek"
        header += f" {short+' ASR':>12} {short+' RD':>10} {short+' CI':>10}"
    print(header)
    print("─"*72)

    # Lignes
    configs = [r["config"] for r in results_by_provider[providers[0]]]
    for i, config_name in enumerate(configs):
        line = f"{config_name:<16}"
        for p in providers:
            r = results_by_provider[p][i]
            line += (
                f" {r['avg_asr']:>11.1%}"
                f" {r['avg_rd']:>10.4f}"
                f" {r['avg_ci']:>10.4f}"
            )
        print(line)

    print(f"{'='*72}")
    print("ASR ↓ = moins vulnérable  |  RD ↓ = moins dégradé  |  CI ↑ = plus intègre")

    # Analyse automatique
    providers = list(results_by_provider.keys())
    p1, p2 = providers[0], providers[1] if len(providers) > 1 else providers[0]
    gpt_nodef = next(r for r in results_by_provider[p1]
                     if r["config"] == "No defense")
    llm_nodef = next(r for r in results_by_provider[p2]
                     if r["config"] == "No defense")
    gpt_all   = next(r for r in results_by_provider[p1]
                     if r["config"] == "All defenses")
    llm_all   = next(r for r in results_by_provider[p2]
                     if r["config"] == "All defenses")

    print(f"\n--- Analyse ---")
    print(f"Vulnérabilité sans défense :")
    print(f"  GPT-3.5  : ASR={gpt_nodef['avg_asr']:.1%}")
    print(f"  DeepSeek: ASR={llm_nodef['avg_asr']:.1%}")
    diff = llm_nodef['avg_asr'] - gpt_nodef['avg_asr']
    if diff > 0.05:
        print(f"  → LLaMA 3.3 est plus vulnérable (+{diff:.1%}) — l'alignement RLHF aide partiellement")
    elif diff < -0.05:
        print(f"  → GPT-3.5 est plus vulnérable ({diff:.1%}) — l'alignement RLHF ne suffit pas")
    else:
        print(f"  → Vulnérabilité similaire — indépendante du modèle")

    print(f"\nAvec toutes les défenses :")
    print(f"  GPT-3.5  : ASR={gpt_all['avg_asr']:.1%}, CI={gpt_all['avg_ci']:.4f}")
    print(f"  DeepSeek: ASR={llm_all['avg_asr']:.1%}, CI={llm_all['avg_ci']:.4f}")
    if gpt_all['avg_asr'] == 0.0 and llm_all['avg_asr'] == 0.0:
        print(f"  → Defense-in-depth efficace sur les deux modèles — généralisabilité confirmée")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Sélection du provider via --provider deepseek / groq / openai
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        selected = sys.argv[idx + 1]
        PROVIDERS = [p for p in PROVIDERS if p["id"] == selected]
        if not PROVIDERS:
            print(f"Provider inconnu. Disponibles : openai, groq, deepseek")
            sys.exit(1)
        print(f"Mode sélectif : provider={selected}\n")

    print("=== Ablation Multi-Modèles — GPT-3.5 vs DeepSeek-V3 ===\n")

    # Charger les données (dataset custom)
    questions = load_questions("data/questions.json")
    clean_docs = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} documents\n")

    results_by_provider = {}

    for provider in PROVIDERS:
        results = run_ablation_for_provider(provider, questions, clean_docs)
        results_by_provider[provider["id"]] = results

    # Afficher le tableau comparatif
    print_multimodel_table(results_by_provider)

    # Sauvegarder
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_multimodel_results.json", "w") as f:
        json.dump(results_by_provider, f, indent=2, ensure_ascii=False)
    print("\nRésultats sauvegardés dans results/ablation_multimodel_results.json")

    # Nettoyage des bases vectorielles temporaires
    print("\nNettoyage des bases ChromaDB temporaires...")
    for d in Path(".").glob("chroma_mm_*/"):
        shutil.rmtree(d)
    print("  Nettoyage terminé.")


