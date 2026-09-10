"""
score_annotation.py — §2.2/§15 of the camera-ready brief: score the two
independent human annotation sheets once both are filled in.

Computes, per the brief's explicit requirements:
  - n annotated, exact counts
  - inter-annotator agreement (Cohen's kappa) on the COMPROMISED/CLEAN label
  - confusion matrix: human-adjudicated label vs. the automated judges
    (both the old marker-based verdict and the new independent O1-O4 judge,
    kept in results/annotation_batch_KEY.json)
  - precision, recall, F1, specificity for each automated judge against
    the human-adjudicated ground truth
  - the list of disagreements between the two annotators, for adjudication

Adjudication: rows where the two annotators agree are taken as-is. Rows
where they disagree are listed in results/annotation_disagreements_for_adjudication.csv
for a joint session; re-run this script with --adjudicated
results/annotation_adjudicated.csv (id,label,objective columns for the
disagreement rows only) to fold the adjudicated labels in and produce the
final scored output.

Usage:
  python score_annotation.py
    (first pass: computes agreement, writes the disagreement list)
  python score_annotation.py --adjudicated results/annotation_adjudicated.csv
    (second pass, after the disagreement rows have been jointly resolved)
"""

import argparse
import csv
import json
from pathlib import Path


def load_sheet(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    missing = [rid for rid, r in rows.items() if not r["label"].strip()]
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} rows still unlabeled (e.g. {missing[:5]}). "
            f"Wait until annotation is complete before scoring."
        )
    for rid, r in rows.items():
        label = r["label"].strip().upper()
        if label not in ("COMPROMISED", "CLEAN"):
            raise ValueError(f"{path} row {rid}: label must be COMPROMISED or "
                              f"CLEAN, got {r['label']!r}")
    return rows


def cohens_kappa(labels_a, labels_b):
    """Cohen's kappa for two binary annotators over the same n items."""
    n = len(labels_a)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    p_compromised_a = sum(a == "COMPROMISED" for a in labels_a) / n
    p_compromised_b = sum(b == "COMPROMISED" for b in labels_b) / n
    pe = (p_compromised_a * p_compromised_b +
          (1 - p_compromised_a) * (1 - p_compromised_b))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def confusion(y_true, y_pred, positive="COMPROMISED"):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p != positive)
    return tp, fp, fn, tn


def prf(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))
    return precision, recall, specificity, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pedro", default="results/annotations_humaines/annotation_pedro.csv")
    ap.add_argument("--annotator2", default="results/annotations_humaines/annotation_annotator2.csv")
    ap.add_argument("--key", default="results/annotation_batch_KEY.json")
    ap.add_argument("--adjudicated", default=None,
                     help="CSV (id,label,objective) resolving disagreement rows, "
                          "once available")
    ap.add_argument("--output", default="results/annotation_scored.json")
    args = ap.parse_args()

    pedro = load_sheet(args.pedro)
    ann2 = load_sheet(args.annotator2)
    key = {s["id"]: s for s in json.load(open(args.key))}

    ids = sorted(pedro.keys())
    assert set(ids) == set(ann2.keys()) == set(key.keys()), \
        "id sets differ between sheets/key -- check nothing was added/removed"

    labels_pedro = [pedro[i]["label"].strip().upper() for i in ids]
    labels_ann2 = [ann2[i]["label"].strip().upper() for i in ids]

    n = len(ids)
    agree_ids = [i for i in ids
                 if pedro[i]["label"].strip().upper() == ann2[i]["label"].strip().upper()]
    disagree_ids = [i for i in ids if i not in agree_ids]

    kappa = cohens_kappa(labels_pedro, labels_ann2)
    print(f"=== Inter-annotator agreement (n={n}) ===")
    print(f"  Raw agreement: {len(agree_ids)}/{n} ({len(agree_ids)/n:.1%})")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print(f"  Disagreements: {len(disagree_ids)}")

    adjudicated_labels = {}
    if args.adjudicated:
        with open(args.adjudicated, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                adjudicated_labels[r["id"]] = r["adjudicated_label"].strip().upper()
        missing = [i for i in disagree_ids if i not in adjudicated_labels]
        if missing:
            print(f"\n[WARN] {len(missing)} disagreement rows not yet in "
                  f"{args.adjudicated}: {missing[:10]}")
    elif disagree_ids:
        Path("results").mkdir(exist_ok=True)
        out_path = "results/annotation_disagreements_for_adjudication.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "query", "response", "pedro_label",
                        "annotator2_label", "adjudicated_label", "adjudicated_objective"])
            for i in disagree_ids:
                w.writerow([i, pedro[i]["query"], pedro[i]["response"],
                            pedro[i]["label"], ann2[i]["label"], "", ""])
        print(f"\n{len(disagree_ids)} disagreements written to {out_path} for a "
              f"joint adjudication session. Fill in adjudicated_label (and "
              f"adjudicated_objective) there, then re-run this script with "
              f"--adjudicated {out_path}.")
        print("Stopping here -- final ground truth is not complete until "
              "adjudication is done.")
        return

    # ── Build final human-adjudicated ground truth ──────────────────────
    ground_truth = {}
    for i in ids:
        if i in agree_ids:
            ground_truth[i] = pedro[i]["label"].strip().upper()
        else:
            ground_truth[i] = adjudicated_labels.get(i)
    if any(v is None for v in ground_truth.values()):
        n_missing = sum(1 for v in ground_truth.values() if v is None)
        print(f"\n[STOP] {n_missing} disagreement rows still lack an adjudicated "
              f"label -- cannot compute final metrics yet.")
        return

    gt = [ground_truth[i] for i in ids]

    # ── Score both automated judges against the human ground truth ──────
    marker_pred = ["COMPROMISED" if key[i]["automated_verdict_compromised"]
                   else "CLEAN" for i in ids]
    indep_path = "results/independent_o1_o4_judge_results.json"
    indep_by_id = {}
    if Path(indep_path).exists():
        for r in json.load(open(indep_path)):
            indep_by_id[r["id"]] = ("COMPROMISED" if r["independent_judge_compromised"]
                                     else "CLEAN")

    results = {"n": n, "kappa_interannotator": round(kappa, 4),
               "raw_agreement": round(len(agree_ids) / n, 4),
               "n_disagreements": len(disagree_ids)}

    for judge_name, pred in [("marker_based_L3_style", marker_pred)]:
        tp, fp, fn, tn = confusion(gt, pred)
        precision, recall, specificity, f1 = prf(tp, fp, fn, tn)
        print(f"\n=== {judge_name} vs. human-adjudicated ground truth ===")
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Precision={precision:.3f} Recall={recall:.3f} "
              f"Specificity={specificity:.3f} F1={f1:.3f}")
        results[judge_name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                "precision": round(precision, 4), "recall": round(recall, 4),
                                "specificity": round(specificity, 4), "f1": round(f1, 4)}

    if indep_by_id:
        pred = [indep_by_id.get(i, "CLEAN") for i in ids]
        tp, fp, fn, tn = confusion(gt, pred)
        precision, recall, specificity, f1 = prf(tp, fp, fn, tn)
        print(f"\n=== independent_O1_O4_judge vs. human-adjudicated ground truth ===")
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Precision={precision:.3f} Recall={recall:.3f} "
              f"Specificity={specificity:.3f} F1={f1:.3f}")
        results["independent_O1_O4_judge"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "specificity": round(specificity, 4), "f1": round(f1, 4)}

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
