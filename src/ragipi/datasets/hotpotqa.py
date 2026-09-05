"""
load_hotpotqa.py — Chargement du dataset HotpotQA
Papier : Indirect Prompt Injection in RAG (thèse)

Spécificités HotpotQA vs SQuAD/NQ :
  - 10 documents par question (2 support + 8 distracteurs)
  - 2 types : comparison (comparer deux entités) et bridge
    (raisonnement en deux étapes)
  - 2 niveaux : easy, medium, hard
  - Réponses courtes : yes/no ou entités nommées

Pourquoi c'est intéressant pour l'IPI :
  - KB dense (10 docs/question) → teste le KB density effect
  - Documents distracteurs → camouflage adversarial naturel
  - Multi-hop → attaque sur un seul maillon suffit
"""

import json
import random
from pathlib import Path

from langchain_core.documents import Document

N_QUESTIONS   = 50
RANDOM_SEED   = 42
OUTPUT_DIR    = Path("data/hotpotqa")
HOTPOT_FILE   = OUTPUT_DIR / "hotpot_dev.json"


def load_hotpotqa_subset(n: int = N_QUESTIONS,
                          level: str = None,
                          q_type: str = None) -> list[dict]:
    """
    Charge n exemples depuis HotpotQA dev distractor.

    Args:
        n      : nombre d'exemples
        level  : 'easy', 'medium', 'hard' ou None (tous)
        q_type : 'comparison', 'bridge' ou None (tous)
    """
    assert HOTPOT_FILE.exists(), (
        f"Fichier manquant : {HOTPOT_FILE}\n"
        "Lance : curl -L http://curtis.ml.cmu.edu/datasets/hotpot/"
        "hotpot_dev_distractor_v1.json -o data/hotpotqa/hotpot_dev.json"
    )

    print(f"Lecture de {HOTPOT_FILE}...")
    with open(HOTPOT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  {len(data)} exemples disponibles")

    # Filtrage
    filtered = [
        ex for ex in data
        if ex.get("answer")
        and (level is None or ex.get("level") == level)
        and (q_type is None or ex.get("type") == q_type)
    ]
    print(f"  {len(filtered)} après filtrage "
          f"(level={level}, type={q_type})")

    # Échantillonnage reproductible
    random.seed(RANDOM_SEED)
    selected = random.sample(filtered, min(n, len(filtered)))
    print(f"  {len(selected)} sélectionnés (seed={RANDOM_SEED})")

    # Stats
    types  = {ex["type"]  for ex in selected}
    levels = {ex["level"] for ex in selected}
    print(f"  Types  : {types}")
    print(f"  Levels : {levels}")

    return selected


def convert_to_benchmark_format(
        examples: list[dict],
        max_docs_per_question: int = 10
) -> tuple[list[Document], list[dict]]:
    """
    Convertit les exemples HotpotQA au format benchmark.

    Chaque exemple a 10 documents (title + sentences).
    On les convertit tous en Documents LangChain pour
    constituer une KB dense — c'est la spécificité HotpotQA.

    Returns:
        docs      : tous les documents (KB dense, ~500 docs pour 50q)
        questions : liste de dicts avec text + expected_keywords
    """
    docs      = []
    questions = []
    seen_titles = set()

    for i, ex in enumerate(examples):
        question = ex["question"]
        answer   = ex["answer"]

        # Ajouter tous les passages comme documents KB
        for j, (title, sentences) in enumerate(
                ex["context"][:max_docs_per_question]):
            if title not in seen_titles:
                seen_titles.add(title)
                content = f"{title}. " + " ".join(sentences)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source":    f"hotpot_{i:03d}_{j:02d}",
                        "filename":  f"hotpot_{i:03d}_{j:02d}.txt",
                        "title":     title,
                        "type":      "legitimate",
                        "dataset":   "HotpotQA",
                        "q_type":    ex.get("type", "unknown"),
                        "q_level":   ex.get("level", "unknown"),
                        "is_support": j < 2,  # les 2 premiers sont support
                    }
                ))

        # Mots-clés depuis la réponse
        if answer.lower() in ("yes", "no"):
            keywords = [answer.lower()]
        else:
            keywords = list(set(
                w.lower() for w in answer.split() if len(w) > 2
            ))[:5]

        questions.append({
            "id":                f"HP{i+1:03d}",
            "text":              question,
            "domain":            "hotpotqa",
            "q_type":            ex.get("type", "unknown"),
            "q_level":           ex.get("level", "unknown"),
            "answer":            answer,
            "expected_keywords": keywords,
        })

    return docs, questions


def save_benchmark_data(docs: list[Document],
                        questions: list[dict]):
    """Sauvegarde les données au format attendu par le pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder les documents
    for doc in docs:
        filepath = OUTPUT_DIR / doc.metadata["filename"]
        filepath.write_text(doc.page_content, encoding="utf-8")
    print(f"  {len(docs)} documents sauvegardés dans {OUTPUT_DIR}/")

    # Sauvegarder les questions
    qfile = Path("data/hotpotqa_questions.json")
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f,
                  indent=2, ensure_ascii=False)
    print(f"  {len(questions)} questions sauvegardées dans {qfile}")


def load_hotpotqa_documents() -> list[Document]:
    docs = []
    for txt_file in sorted(OUTPUT_DIR.glob("hotpot_*.txt")):
        docs.append(Document(
            page_content=txt_file.read_text(encoding="utf-8"),
            metadata={
                "source":   str(txt_file),
                "filename": txt_file.name,
                "type":     "legitimate",
                "dataset":  "HotpotQA",
            }
        ))
    print(f"  {len(docs)} documents HotpotQA chargés")
    return docs


def load_hotpotqa_questions() -> list[str]:
    with open("data/hotpotqa_questions.json", encoding="utf-8") as f:
        data = json.load(f)
    questions = [q["text"] for q in data["questions"]]
    print(f"  {len(questions)} questions HotpotQA chargées")
    return questions


if __name__ == "__main__":
    print("=== Chargement HotpotQA pour benchmark IPI ===\n")

    examples = load_hotpotqa_subset(N_QUESTIONS)

    print("\nConversion au format benchmark...")
    docs, questions = convert_to_benchmark_format(examples)
    print(f"  {len(docs)} documents uniques")
    print(f"  {len(questions)} questions")
    print(f"  Ratio moyen : {len(docs)/len(questions):.1f} docs/question")

    print("\nSauvegarde...")
    save_benchmark_data(docs, questions)

    print("\n--- Aperçu des 3 premières questions ---")
    for q in questions[:3]:
        print(f"\n  [{q['id']}] ({q['q_type']}/{q['q_level']})")
        print(f"  Q : {q['text']}")
        print(f"  A : {q['answer']}")
        print(f"  K : {q['expected_keywords']}")

    print("\n=== Prêt. Lance : python ablation_hotpotqa.py ===")
