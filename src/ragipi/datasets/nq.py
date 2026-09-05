"""
load_nq_dataset.py — Chargement du dataset SQuAD/NQ depuis JSON local
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Téléchargement préalable :
  curl -L https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json \
       -o data/nq/squad_dev.json
"""

import json
import random
from pathlib import Path

from langchain_core.documents import Document

N_QUESTIONS = 50
RANDOM_SEED = 42
OUTPUT_DIR  = Path("data/nq")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SQUAD_FILE  = OUTPUT_DIR / "squad_dev.json"


def load_nq_subset(n: int = N_QUESTIONS) -> list[dict]:
    """
    Charge n exemples depuis le fichier SQuAD JSON local.
    Pas de HuggingFace, pas de RAM excessive — lecture directe.
    """
    assert SQUAD_FILE.exists(), (
        f"Fichier manquant : {SQUAD_FILE}\n"
        "Lance : curl -L https://rajpurkar.github.io/SQuAD-explorer/"
        "dataset/dev-v1.1.json -o data/nq/squad_dev.json"
    )

    print(f"Lecture de {SQUAD_FILE}...")
    with open(SQUAD_FILE, encoding="utf-8") as f:
        squad = json.load(f)

    # Aplatir la structure SQuAD : dataset > articles > paragraphes > QA
    examples = []
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa.get("answers"):
                    examples.append({
                        "question": qa["question"],
                        "context":  context,
                        "answers":  {"text": [a["text"] for a in qa["answers"]]},
                        "id":       qa["id"],
                    })

    print(f"  {len(examples)} exemples disponibles dans SQuAD dev")

    # Échantillonnage reproductible
    random.seed(RANDOM_SEED)
    selected = random.sample(examples, min(n, len(examples)))
    print(f"  {len(selected)} exemples sélectionnés (seed={RANDOM_SEED})")
    return selected


def convert_to_benchmark_format(examples):
    docs, questions = [], []
    seen = set()

    for i, ex in enumerate(examples):
        context  = ex["context"].strip()
        question = ex["question"].strip()
        answers  = ex["answers"]["text"]

        # Un document par contexte unique
        key = context[:100]
        if key not in seen:
            seen.add(key)
            docs.append(Document(
                page_content=context,
                metadata={
                    "source":   f"squad_nq_{i:03d}",
                    "filename": f"nq_{i:03d}.txt",
                    "type":     "legitimate",
                    "dataset":  "SQuAD-NQ",
                }
            ))

        # Mots-clés depuis les réponses
        keywords = list(set(
            w.lower() for ans in answers
            for w in ans.split() if len(w) > 2
        ))[:5]

        questions.append({
            "id":                f"NQ{i+1:03d}",
            "text":              question,
            "domain":            "nq_open",
            "expected_keywords": keywords,
            "answers":           answers,
        })

    return docs, questions


def save_benchmark_data(docs, questions):
    for doc in docs:
        (OUTPUT_DIR / doc.metadata["filename"]).write_text(
            doc.page_content, encoding="utf-8"
        )
    print(f"  {len(docs)} documents sauvegardés dans {OUTPUT_DIR}/")

    qfile = Path("data/nq_questions.json")
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)
    print(f"  {len(questions)} questions sauvegardées dans {qfile}")


def load_nq_documents() -> list[Document]:
    docs = []
    for txt_file in sorted(OUTPUT_DIR.glob("nq_*.txt")):
        docs.append(Document(
            page_content=txt_file.read_text(encoding="utf-8"),
            metadata={
                "source":   str(txt_file),
                "filename": txt_file.name,
                "type":     "legitimate",
                "dataset":  "NQ",
            }
        ))
    print(f"  {len(docs)} documents NQ chargés depuis {OUTPUT_DIR}/")
    return docs


def load_nq_questions() -> list[str]:
    with open("data/nq_questions.json", encoding="utf-8") as f:
        data = json.load(f)
    questions = [q["text"] for q in data["questions"]]
    print(f"  {len(questions)} questions NQ chargées")
    return questions


if __name__ == "__main__":
    print("=== Chargement du dataset SQuAD/NQ (JSON local) ===\n")

    examples = load_nq_subset(N_QUESTIONS)

    print("\nConversion au format benchmark...")
    docs, questions = convert_to_benchmark_format(examples)
    print(f"  {len(docs)} documents uniques, {len(questions)} questions")

    print("\nSauvegarde...")
    save_benchmark_data(docs, questions)

    print("\n--- Aperçu des 3 premières questions ---")
    for q in questions[:3]:
        print(f"\n  [{q['id']}] {q['text']}")
        print(f"  Réponses  : {q['answers'][:2]}")
        print(f"  Mots-clés : {q['expected_keywords']}")

    print("\n=== Dataset prêt. Lance : python ablation_nq.py ===")
