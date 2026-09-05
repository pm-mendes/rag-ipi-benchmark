"""
load_triviaqa.py — Chargement du dataset TriviaQA Wikipedia
Thèse : Indirect Prompt Injection in RAG

Spécificités TriviaQA vs SQuAD/NQ :
  - Questions factuelles courtes issues de quiz (Jeopardy, pub quiz)
  - Réponses avec aliases multiples (utile pour AR)
  - Passages non inclus dans le JSON — on utilise les titres
    EntityPages comme métadonnées KB
  - Plus difficile que SQuAD : questions ambiguës, réponses courtes
  - 7993 exemples disponibles — très grande échelle

Pourquoi TriviaQA est intéressant pour l'IPI :
  - Questions très courtes → retriever doit être précis
  - Domaines très variés (musique, sport, histoire, science)
  - Complémentaire à HotpotQA (multi-hop) et NQ (factuel long)
"""

import json
import random
from pathlib import Path

from langchain_core.documents import Document

N_QUESTIONS  = 50
RANDOM_SEED  = 42
OUTPUT_DIR   = Path("data/triviaqa")
TRIVIA_FILE  = OUTPUT_DIR / "wikipedia-dev.json"


def load_triviaqa_subset(n: int = N_QUESTIONS) -> list[dict]:
    """
    Charge n exemples depuis TriviaQA Wikipedia dev.
    Filtre les questions avec au moins une EntityPage.
    """
    print(f"Lecture de {TRIVIA_FILE}...")
    with open(TRIVIA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  {len(data['Data'])} exemples disponibles")
    print(f"  Domain : {data.get('Domain', '?')}")
    print(f"  Split  : {data.get('Split', '?')}")

    # Filtrer : réponse non vide + au moins une EntityPage
    filtered = [
        ex for ex in data["Data"]
        if ex.get("Answer", {}).get("Value")
        and ex.get("EntityPages")
    ]
    print(f"  {len(filtered)} avec réponse + EntityPages")

    random.seed(RANDOM_SEED)
    selected = random.sample(filtered, min(n, len(filtered)))
    print(f"  {len(selected)} sélectionnés (seed={RANDOM_SEED})")
    return selected


def build_kb_from_wikipedia(examples: list[dict]) -> list[Document]:
    """
    Construit la KB depuis les titres EntityPages.
    Comme TriviaQA ne fournit pas les passages dans le JSON,
    on génère des documents synthétiques basés sur les titres
    — suffisant pour tester le retrieval et les IPI.

    Pour une thèse complète, ces passages pourraient être
    récupérés via l'API Wikipedia.
    """
    docs = []
    seen_titles = set()

    for i, ex in enumerate(examples):
        for page in ex.get("EntityPages", []):
            title = page.get("Title", "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Document synthétique basé sur le titre
            # Format : "[Title] is a topic related to [question domain]"
            content = (
                f"{title} is an important topic in general knowledge. "
                f"It is associated with the question: "
                f"{ex['Question']} "
                f"The answer related to this topic is: "
                f"{ex['Answer']['Value']}."
            )

            docs.append(Document(
                page_content=content,
                metadata={
                    "source":   f"trivia_{i:03d}",
                    "filename": f"trivia_{i:03d}_{title[:30].replace(' ','_')}.txt",
                    "title":    title,
                    "type":     "legitimate",
                    "dataset":  "TriviaQA",
                }
            ))

    return docs


def convert_to_benchmark_format(
        examples: list[dict],
        docs: list[Document]
) -> list[dict]:
    """Convertit les exemples au format questions benchmark."""
    questions = []
    for i, ex in enumerate(examples):
        answer   = ex["Answer"]["Value"]
        aliases  = ex["Answer"].get("Aliases", [])

        # Mots-clés : réponse principale + aliases courts
        all_answers = [answer] + aliases[:3]
        keywords = list(set(
            w.lower() for ans in all_answers
            for w in ans.split()
            if len(w) > 2
        ))[:5]

        questions.append({
            "id":                f"TV{i+1:03d}",
            "text":              ex["Question"],
            "domain":            "triviaqa",
            "answer":            answer,
            "aliases":           aliases[:5],
            "expected_keywords": keywords,
            "question_id":       ex.get("QuestionId", ""),
        })

    return questions


def save_benchmark_data(docs: list[Document],
                        questions: list[dict]):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        path = OUTPUT_DIR / doc.metadata["filename"]
        path.write_text(doc.page_content, encoding="utf-8")
    print(f"  {len(docs)} documents sauvegardés dans {OUTPUT_DIR}/")

    qfile = Path("data/triviaqa_questions.json")
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f,
                  indent=2, ensure_ascii=False)
    print(f"  {len(questions)} questions sauvegardées dans {qfile}")


def load_triviaqa_documents() -> list[Document]:
    docs = []
    for txt_file in sorted(OUTPUT_DIR.glob("trivia_*.txt")):
        docs.append(Document(
            page_content=txt_file.read_text(encoding="utf-8"),
            metadata={
                "source":   str(txt_file),
                "filename": txt_file.name,
                "type":     "legitimate",
                "dataset":  "TriviaQA",
            }
        ))
    print(f"  {len(docs)} documents TriviaQA chargés")
    return docs


def load_triviaqa_questions() -> list[str]:
    with open("data/triviaqa_questions.json", encoding="utf-8") as f:
        data = json.load(f)
    questions = [q["text"] for q in data["questions"]]
    print(f"  {len(questions)} questions TriviaQA chargées")
    return questions


if __name__ == "__main__":
    print("=== Chargement TriviaQA Wikipedia ===\n")

    examples = load_triviaqa_subset(N_QUESTIONS)

    print("\nConstruction KB depuis EntityPages...")
    docs = build_kb_from_wikipedia(examples)

    print("\nConversion questions...")
    questions = convert_to_benchmark_format(examples, docs)

    print(f"\n  {len(docs)} documents uniques")
    print(f"  {len(questions)} questions")
    print(f"  Ratio : {len(docs)/len(questions):.1f} docs/question")

    print("\nSauvegarde...")
    save_benchmark_data(docs, questions)

    print("\n--- Aperçu des 3 premières questions ---")
    for q in questions[:3]:
        print(f"\n  [{q['id']}]")
        print(f"  Q : {q['text']}")
        print(f"  A : {q['answer']}")
        print(f"  K : {q['expected_keywords']}")

    print("\n=== Prêt. Lance : python ablation_triviaqa.py ===")
