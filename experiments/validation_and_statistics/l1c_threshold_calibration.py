"""
l1c_threshold_calibration.py — §10 of the camera-ready brief.

Audit finding first: no calibration script or held-out dataset for L1c's
delta > 0.05 threshold exists anywhere in this repository (see
docs/AUDIT_camera_ready.md) -- searched `src/`, `experiments/`, `docs/`,
`paper/main.tex`. The paper's "(calibrated on held-out validation)" footnote
is not backed by an artifact. This script does not reconstruct that
missing calibration after the fact (per the brief's own rule); it performs
a *real* one now, with a proper calibration/test split, and reports the
result as what it is: today's calibration, on the samples available in
this repository, not a universal threshold.

Method:
  - Positive (adversarial) class: all 17 files in data/adversarial/*.txt
    (the full existing payload set — canonical, advanced, adaptive,
    dataset-themed variants).
  - Negative (legitimate) class: the 8 files in data/clean_docs/ plus 10
    MITRE ATT&CK technique descriptions (data/mitre/) as a *harder*
    negative set — legitimate but technical/directive-sounding content,
    the exact kind of document the brief's benign-hard false-positive
    concern (§9) is about.
  - A fixed stratified 50/50 split into calibration and test sets (seed
    below), computed once and never touched again after inspecting test
    performance — the calibration/test separation the brief asks for.
  - On the calibration half: sweep thresholds, report precision/recall/F1
    per threshold, pick the F1-maximizing threshold.
  - On the test half (held out from threshold selection): report
    precision/recall/F1/PR-AUC at both the chosen threshold and the
    paper's original 0.05, on legitimate/adversarial docs separately.

Caveat stated once, not repeated per number: total N=35 documents (17 pos,
18 neg) is small. This is a real calibration/test protocol, not a
large-scale validation — report it as such.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from langchain_core.documents import Document
from ragipi.defenses.semantic_defense import embedding_anomaly_score

SEED = 42
CANDIDATE_THRESHOLDS = np.arange(-0.10, 0.16, 0.01)


def load_positive():
    docs = []
    for p in sorted(Path("data/adversarial").glob("*.txt")):
        docs.append(Document(page_content=p.read_text(encoding="utf-8"),
                              metadata={"source": str(p)}))
    return docs


def load_negative():
    docs = []
    for p in sorted(Path("data/clean_docs").glob("*.txt")):
        docs.append(Document(page_content=p.read_text(encoding="utf-8"),
                              metadata={"source": str(p), "class": "clean_generic"}))
    mitre_files = sorted(Path("data/mitre").glob("*.txt"))[:10]
    for p in mitre_files:
        docs.append(Document(page_content=p.read_text(encoding="utf-8"),
                              metadata={"source": str(p), "class": "clean_technical_hard"}))
    return docs


def precision_recall_f1(scores, labels, threshold):
    preds = [s > threshold for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)
    tn = sum(1 for p, l in zip(preds, labels) if not p and not l)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"threshold": round(float(threshold), 3), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3), "fpr": round(fpr, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    pos_docs, neg_docs = load_positive(), load_negative()
    print(f"Positive (adversarial): {len(pos_docs)} | Negative (legitimate): {len(neg_docs)}")

    all_docs = [(d, 1) for d in pos_docs] + [(d, 0) for d in neg_docs]
    scored = []
    for doc, label in all_docs:
        score, sim_imp, sim_desc = embedding_anomaly_score(doc)
        scored.append({"source": doc.metadata.get("source"), "label": label,
                        "score": score, "sim_imp": sim_imp, "sim_desc": sim_desc,
                        "class": doc.metadata.get("class", "adversarial" if label else "clean")})

    # Stratified 50/50 calibration/test split, fixed seed.
    rng = np.random.default_rng(SEED)
    pos_idx = [i for i, s in enumerate(scored) if s["label"] == 1]
    neg_idx = [i for i, s in enumerate(scored) if s["label"] == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    calib_idx = set(pos_idx[:len(pos_idx) // 2] + neg_idx[:len(neg_idx) // 2])
    calib = [s for i, s in enumerate(scored) if i in calib_idx]
    test = [s for i, s in enumerate(scored) if i not in calib_idx]
    print(f"Calibration set: {len(calib)} ({sum(s['label'] for s in calib)} pos) | "
          f"Test set: {len(test)} ({sum(s['label'] for s in test)} pos)")

    # Sweep on calibration only.
    calib_scores = [s["score"] for s in calib]
    calib_labels = [s["label"] for s in calib]
    sweep = [precision_recall_f1(calib_scores, calib_labels, t) for t in CANDIDATE_THRESHOLDS]
    best = max(sweep, key=lambda r: r["f1"])
    print(f"\nBest calibration threshold (max F1 on calibration set): "
          f"{best['threshold']} (F1={best['f1']})")

    # Evaluate chosen threshold AND the paper's original 0.05 on the held-out test set.
    test_scores = [s["score"] for s in test]
    test_labels = [s["label"] for s in test]
    result_chosen = precision_recall_f1(test_scores, test_labels, best["threshold"])
    result_paper = precision_recall_f1(test_scores, test_labels, 0.05)

    precisions, recalls, _ = precision_recall_curve(test_labels, test_scores)
    pr_auc = auc(recalls, precisions)

    print(f"\n=== Held-out test set (never used for threshold selection) ===")
    print(f"  At calibrated threshold {best['threshold']}: {result_chosen}")
    print(f"  At paper's original 0.05: {result_paper}")
    print(f"  PR-AUC (test set): {pr_auc:.3f}")

    # Per-class (hard vs generic negatives) breakdown at 0.05.
    hard_neg = [s for s in test if s.get("class") == "clean_technical_hard"]
    if hard_neg:
        hard_fpr = sum(1 for s in hard_neg if s["score"] > 0.05) / len(hard_neg)
        print(f"\n  FPR on HARD (MITRE technical) negatives only, threshold=0.05: "
              f"{hard_fpr:.1%} (n={len(hard_neg)})")

    out = {
        "n_positive": len(pos_docs), "n_negative": len(neg_docs),
        "calibration_set_size": len(calib), "test_set_size": len(test),
        "calibration_sweep": sweep,
        "best_calibration_threshold": best,
        "test_at_calibrated_threshold": result_chosen,
        "test_at_paper_threshold_0.05": result_paper,
        "test_pr_auc": round(float(pr_auc), 4),
        "hard_negative_fpr_at_0.05": round(hard_fpr, 4) if hard_neg else None,
        "per_document_scores": scored,
        "caveat": ("Small-N (35 documents) real calibration/test protocol on this "
                   "repository's existing payload files, run because no prior "
                   "calibration artifact exists (see docs/AUDIT_camera_ready.md). "
                   "Not a claim of a universal threshold, model-specific and "
                   "corpus-specific per the brief's own instruction."),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/l1c_threshold_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n→ results/l1c_threshold_calibration.json")


if __name__ == "__main__":
    main()
