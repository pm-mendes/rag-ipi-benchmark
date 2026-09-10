"""
nq_em_f1.py — §8 of the camera-ready brief: report standard reference-based
QA metrics (EM/F1, SQuAD-style) on the one dataset in this benchmark that
actually has reference answers (NQ, 50q, `answers` field), rather than
relying on SAU alone. SAU is repositioned as a controlled-ablation proxy
(already stated in the paper); this adds the metric it's a proxy *for*.

No attack involved -- this measures baseline generation utility (No
defense, clean retrieval) to establish how SAU relates to real QA metrics
on this benchmark's own data, complementing the existing human/BERTScore
correlation (which was on a different, smaller sample -- see Limitations).

EM/F1 implementation follows the standard SQuAD normalization (lowercase,
strip punctuation/articles, collapse whitespace) and takes the max over
all reference answers per question, per convention.
"""

import json
import re
import string
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ragipi.pipeline import load_documents_from_folder, build_vectorstore
from ragipi.metrics.sau import compute_sau

load_dotenv()

TOP_K = 3
SYSTEM_PROMPT = (
    "You are a helpful assistant.\nAnswer the question using ONLY the "
    "information in the context below, as concisely as possible (a short "
    "phrase, not a full sentence, when the question calls for one).\nIf "
    "the answer is not in the context, say \"I don't know\".\n\nContext:\n{context}"
)


# ── SQuAD-style normalization / EM / F1 ─────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_refs(pred, golds, fn):
    return max(fn(pred, g) for g in golds)


def get_generator():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def main():
    docs = load_documents_from_folder("data/nq")
    questions = json.load(open("data/nq_questions.json"))["questions"]
    llm = get_generator()

    tmp_dir = "./chroma_nq_emf1"
    vs = build_vectorstore(docs, tmp_dir)
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )

    rows = []
    for q in questions:
        retrieved = retriever.invoke(q["text"])
        context = "\n\n---\n\n".join(d.page_content for d in retrieved)
        response = (prompt | llm).invoke(
            {"context": context, "question": q["text"]}).content

        golds = q.get("answers", [])
        em = max_over_refs(response, golds, exact_match) if golds else None
        f1 = max_over_refs(response, golds, f1_score) if golds else None
        sau = compute_sau(response, q.get("expected_keywords", []))

        rows.append({
            "id": q["id"], "question": q["text"], "response": response,
            "gold_answers": golds, "em": em, "f1": round(f1, 4) if f1 is not None else None,
            "sau": round(sau, 4),
        })
        print(f"  {q['id']}: EM={em} F1={f1:.2f} SAU={sau:.2f}  "
              f"pred={response[:60]!r}")

    n = len(rows)
    mean_em = sum(r["em"] for r in rows) / n
    mean_f1 = sum(r["f1"] for r in rows) / n
    mean_sau = sum(r["sau"] for r in rows) / n

    # Pearson correlation SAU vs. F1 (SAU's own proxy target)
    import numpy as np
    sau_arr = np.array([r["sau"] for r in rows])
    f1_arr = np.array([r["f1"] for r in rows])
    em_arr = np.array([float(r["em"]) for r in rows])
    corr_sau_f1 = float(np.corrcoef(sau_arr, f1_arr)[0, 1])
    corr_sau_em = float(np.corrcoef(sau_arr, em_arr)[0, 1])

    summary = {
        "n": n, "mean_EM": round(mean_em, 4), "mean_F1": round(mean_f1, 4),
        "mean_SAU": round(mean_sau, 4),
        "pearson_r_SAU_vs_F1": round(corr_sau_f1, 4),
        "pearson_r_SAU_vs_EM": round(corr_sau_em, 4),
        "rows": rows,
    }

    print(f"\n=== NQ (n={n}) baseline utility: EM/F1 vs. SAU ===")
    print(f"  Mean EM  = {mean_em:.1%}")
    print(f"  Mean F1  = {mean_f1:.3f}")
    print(f"  Mean SAU = {mean_sau:.3f}")
    print(f"  Pearson r(SAU, F1) = {corr_sau_f1:.3f}")
    print(f"  Pearson r(SAU, EM) = {corr_sau_em:.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/nq_em_f1_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n→ results/nq_em_f1_results.json")


if __name__ == "__main__":
    main()
