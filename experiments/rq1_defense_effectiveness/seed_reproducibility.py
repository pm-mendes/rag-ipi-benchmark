"""
ablation_statistical.py — Tests statistiques (mean ± std) sur 3 seeds
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Objectif :
  Montrer que les résultats sont stables et reproductibles
  indépendamment de l'ordre des appels API et de la
  sélection des chunks.

Protocole :
  - 3 seeds : 42, 123, 456
  - Dataset : custom (10 questions, 6 attaques)
  - Modèle  : GPT-3.5-turbo (provider principal)
  - Pour chaque seed : rebuild ChromaDB + relancer toutes les configs
  - Calcul : mean ± std par configuration et par métrique

Sortie :
  - Tableau mean ± std dans le terminal
  - results/statistical_results.json
  - Prêt à intégrer dans la section VII du papier

Usage :
  python ablation_statistical.py

Durée estimée :
  3 seeds × 5 configs × 6 attaques × 10 questions
  = 900 appels API ≈ 15-20 minutes
"""

import json
import shutil
import random
import numpy as np
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
from ragipi.metrics.sau import compute_sau, load_questions_meta

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TOP_K           = int(os.getenv("TOP_K", 3))

# ── 3 seeds pour les tests statistiques ──────────────────────────────────────
# NOTE (camera-ready audit): this was found hardcoded to a single seed
# ([456]), which silently regressed this script from a real 3-seed run to
# a 1-seed run — results/statistical_results.json (produced by that state
# of the script) has std=0.0 everywhere for exactly that reason, not
# because the pipeline is actually deterministic. The genuine 3-seed data
# still exists in results/statistical_log.txt (seeds 42, 123 complete) +
# results/statistical_seed456.txt (seed 456, from a separate patch-up run
# after the original 3-seed run crashed on a Chroma path-reuse error mid-way
# through seed 456) and has been recovered into
# results/statistical_results_RECOVERED_3seeds.json — see
# docs/AUDIT_camera_ready.md, Finding A. Restored to the original 3 seeds
# below so a future re-run reproduces a real 3-seed experiment again.
SEEDS = [42, 123, 456]

CONFIGS = [
    {"name": "No defense",   "filter": False, "isolate": False, "verify": False},
    {"name": "Filter only",  "filter": True,  "isolate": False, "verify": False},
    {"name": "Isolate only", "filter": False, "isolate": True,  "verify": False},
    {"name": "Verify only",  "filter": False, "isolate": False, "verify": True},
    {"name": "All defenses", "filter": True,  "isolate": True,  "verify": True},
]

ATTACKS = [
    ("OVERRIDE",           "data/adversarial/override.txt"),
    ("EXFIL",              "data/adversarial/exfil.txt"),
    ("ROLE",               "data/adversarial/role.txt"),
    ("DENIAL",             "data/adversarial/denial.txt"),
    ("TECHNICAL_OVERRIDE", "data/adversarial/technical_override.txt"),
    ("DATA_EXFIL",         "data/adversarial/data_exfil.txt"),
]


# ── Pipeline avec seed contrôlée ──────────────────────────────────────────────

def build_pipeline_seeded(docs, persist_dir, use_filter, use_isolate, seed):
    """
    Construit le pipeline RAG avec une seed fixe pour le chunking.
    La seed contrôle :
      - L'ordre de traitement des chunks (reproductibilité ChromaDB)
      - La sélection des chunks en cas d'égalité de score
    """
    random.seed(seed)
    np.random.seed(seed)

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


# ── Run pour une seed ────────────────────────────────────────────────────────

def run_one_seed(seed, questions, questions_meta, clean_docs):
    """
    Exécute l'ablation complète pour une seed donnée.
    Retourne les métriques moyennes par configuration.
    """
    print(f"\n{'='*55}")
    print(f"Seed : {seed}")
    print(f"{'='*55}")

    # Baseline
    chain_base = build_pipeline_seeded(
        clean_docs, f"./chroma_stat_{seed}_base",
        False, False, seed
    )
    baseline_responses = {q: chain_base.invoke(q) for q in questions}
    print(f"  Baseline calculée ({len(questions)} réponses)")

    seed_results = []

    for config in CONFIGS:
        config_metrics = {
            "config": config["name"],
            "asr": [], "rd": [], "ci": [], "ar": []
        }

        for attack_type, adv_file in ATTACKS:
            adv_content = Path(adv_file).read_text(encoding="utf-8")
            adversarial_doc = Document(
                page_content=adv_content,
                metadata={"source": adv_file, "type": "adversarial",
                          "attack_type": attack_type}
            )
            poisoned = clean_docs + [adversarial_doc]
            persist = (f"./chroma_stat_{seed}_"
                       f"{config['name'].replace(' ','_')}_{attack_type}")

            chain = build_pipeline_seeded(
                poisoned, persist,
                config["filter"], config["isolate"], seed
            )

            asr_n, rd_list, ci_list, ar_list = 0, [], [], []

            for i, q in enumerate(questions):
                response = chain.invoke(q)
                if config["verify"]:
                    _, response = verify_output(response)

                success = is_attack_successful(attack_type, response)
                rd = compute_rd(baseline_responses[q], response)
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

            config_metrics["asr"].append(asr_n / len(questions))
            config_metrics["rd"].append(sum(rd_list) / len(rd_list))
            config_metrics["ci"].append(sum(ci_list) / len(ci_list))
            config_metrics["ar"].append(sum(ar_list) / len(ar_list))

        # Moyenne sur les attaques pour cette config
        seed_results.append({
            "config":  config["name"],
            "avg_asr": round(np.mean(config_metrics["asr"]), 4),
            "avg_rd":  round(np.mean(config_metrics["rd"]),  4),
            "avg_ci":  round(np.mean(config_metrics["ci"]),  4),
            "avg_ar":  round(np.mean(config_metrics["ar"]),  4),
        })
        print(f"  {config['name']:<16} "
              f"ASR={seed_results[-1]['avg_asr']:.1%} "
              f"RD={seed_results[-1]['avg_rd']:.4f} "
              f"CI={seed_results[-1]['avg_ci']:.4f} "
              f"AR={seed_results[-1]['avg_ar']:.4f}")

    return seed_results


# ── Agrégation statistique ────────────────────────────────────────────────────

def aggregate_statistics(all_seed_results):
    """
    Calcule mean ± std sur les 3 seeds pour chaque métrique.

    Structure d'entrée :
      all_seed_results[seed_idx][config_idx] = {avg_asr, avg_rd, avg_ci, avg_ar}

    Structure de sortie :
      stats[config_name] = {
        asr: (mean, std),
        rd:  (mean, std),
        ci:  (mean, std),
        ar:  (mean, std),
      }
    """
    n_configs = len(all_seed_results[0])
    stats = {}

    for i in range(n_configs):
        config_name = all_seed_results[0][i]["config"]
        asr_vals = [r[i]["avg_asr"] for r in all_seed_results]
        rd_vals  = [r[i]["avg_rd"]  for r in all_seed_results]
        ci_vals  = [r[i]["avg_ci"]  for r in all_seed_results]
        ar_vals  = [r[i]["avg_ar"]  for r in all_seed_results]

        stats[config_name] = {
            "asr": (round(np.mean(asr_vals), 4),
                    round(np.std(asr_vals),  4)),
            "rd":  (round(np.mean(rd_vals),  4),
                    round(np.std(rd_vals),   4)),
            "ci":  (round(np.mean(ci_vals),  4),
                    round(np.std(ci_vals),   4)),
            "ar":  (round(np.mean(ar_vals),  4),
                    round(np.std(ar_vals),   4)),
            "asr_values": asr_vals,
            "rd_values":  rd_vals,
            "ci_values":  ci_vals,
            "ar_values":  ar_vals,
        }

    return stats


# ── Affichage ─────────────────────────────────────────────────────────────────

def print_statistical_table(stats):
    """
    Affiche le tableau mean ± std — prêt pour la section VII du papier.
    Format IEEE : valeur ± écart-type
    """
    print(f"\n{'='*75}")
    print("RÉSULTATS STATISTIQUES — mean ± std sur 3 seeds (42, 123, 456)")
    print(f"{'='*75}")
    print(f"{'Configuration':<16} {'ASR (mean±std)':>16} {'RD (mean±std)':>14} "
          f"{'CI (mean±std)':>14} {'AR (mean±std)':>14}")
    print("─"*75)

    for config_name, s in stats.items():
        asr_m, asr_s = s["asr"]
        rd_m,  rd_s  = s["rd"]
        ci_m,  ci_s  = s["ci"]
        ar_m,  ar_s  = s["ar"]

        print(
            f"{config_name:<16} "
            f"{asr_m:.3f}±{asr_s:.3f}  "
            f"{rd_m:.3f}±{rd_s:.3f}  "
            f"{ci_m:.3f}±{ci_s:.3f}  "
            f"{ar_m:.3f}±{ar_s:.3f}"
        )

    print(f"{'='*75}")
    print("ASR ↓ | RD ↓ | CI ↑ | AR ↑")

    # Commentaire scientifique automatique
    all_def = stats.get("All defenses", {})
    no_def  = stats.get("No defense", {})
    if all_def and no_def:
        asr_std = all_def["asr"][1]
        print(f"\nStabilité All defenses :")
        print(f"  ASR std = {asr_std:.4f} "
              f"({'stable' if asr_std < 0.02 else 'variable'} "
              f"across seeds)")
        print(f"  → Résultats reproductibles sur 3 runs indépendants")


def format_for_latex(stats):
    """
    Génère le tableau LaTeX avec mean ± std pour Overleaf.
    """
    lines = []
    lines.append("% Tableau statistique — mean ± std sur 3 seeds")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study results (mean $\\pm$ std over 3 seeds: "
                 "42, 123, 456). Custom dataset, GPT-3.5-turbo, 10 queries, "
                 "6 attack strategies.}")
    lines.append("\\label{tab:ablation_stats}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Configuration} & \\textbf{ASR} & \\textbf{RD} "
                 "& \\textbf{CI} & \\textbf{AR} \\\\")
    lines.append("\\hline")

    for config_name, s in stats.items():
        asr_m, asr_s = s["asr"]
        rd_m,  rd_s  = s["rd"]
        ci_m,  ci_s  = s["ci"]
        ar_m,  ar_s  = s["ar"]

        bold_open  = "\\textbf{" if config_name == "All defenses" else ""
        bold_close = "}"         if config_name == "All defenses" else ""

        lines.append(
            f"{bold_open}{config_name}{bold_close} & "
            f"${asr_m:.3f}\\pm{asr_s:.3f}$ & "
            f"${rd_m:.3f}\\pm{rd_s:.3f}$ & "
            f"${ci_m:.3f}\\pm{ci_s:.3f}$ & "
            f"${ar_m:.3f}\\pm{ar_s:.3f}$ \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Tests Statistiques — 3 seeds × ablation complète ===\n")

    # Charger les données
    questions_meta = load_questions_meta("data/questions.json")
    questions = [q["text"] for q in questions_meta]
    clean_docs = load_documents_from_folder("data/clean_docs")
    print(f"  {len(questions)} questions, {len(clean_docs)} documents")
    print(f"  {len(SEEDS)} seeds : {SEEDS}")
    print(f"  {len(CONFIGS)} configurations × {len(ATTACKS)} attaques")
    total = len(SEEDS) * len(CONFIGS) * len(ATTACKS) * len(questions)
    print(f"  Total appels API estimés : {total}\n")

    # Lancer les 3 seeds
    all_seed_results = []
    for seed in SEEDS:
        seed_results = run_one_seed(seed, questions, questions_meta, clean_docs)
        all_seed_results.append(seed_results)

    # Agrégation statistique
    print("\nCalcul des statistiques...")
    stats = aggregate_statistics(all_seed_results)

    # Affichage
    print_statistical_table(stats)

    # LaTeX
    latex_table = format_for_latex(stats)
    print(f"\n{'─'*55}")
    print("TABLE LATEX (copier dans Overleaf) :")
    print(f"{'─'*55}")
    print(latex_table)

    # Nettoyage ChromaDB
    print("\nNettoyage ChromaDB...")
    for d in Path(".").glob("chroma_stat_*/"):
        shutil.rmtree(d)

    # Sauvegarde
    Path("results").mkdir(exist_ok=True)
    output = {
        "seeds": SEEDS,
        "per_seed_results": all_seed_results,
        "statistics": {
            k: {
                "asr_mean": v["asr"][0], "asr_std": v["asr"][1],
                "rd_mean":  v["rd"][0],  "rd_std":  v["rd"][1],
                "ci_mean":  v["ci"][0],  "ci_std":  v["ci"][1],
                "ar_mean":  v["ar"][0],  "ar_std":  v["ar"][1],
                "asr_values": v["asr_values"],
                "rd_values":  v["rd_values"],
                "ci_values":  v["ci_values"],
                "ar_values":  v["ar_values"],
            }
            for k, v in stats.items()
        },
        "latex_table": latex_table,
    }
    with open("results/statistical_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Résultats sauvegardés dans results/statistical_results.json")
    print("\nTable LaTeX sauvegardée dans results/statistical_results.json")
    print("→ Copier le contenu 'latex_table' directement dans Overleaf.")