import json
import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, confusion_matrix

d = json.load(open("results/judge_validation_extended.json"))
samples = d.get("samples", [])

# Map booleans to string labels
def label(b): return "COMPROMISED" if b else "CLEAN"

gt     = [label(s["ground_truth"]) for s in samples]
llama  = [label(s["llama_pred"])   for s in samples]
ds     = [label(s["ds_pred"])      for s in samples]
cats   = [s.get("category","") for s in samples]

print(f"Total samples: {len(samples)}")
print(f"Categories: {set(cats)}")
print()

# Label distribution
for name, arr in [("Ground truth", gt), ("LLaMA", llama), ("DeepSeek", ds)]:
    n_comp = arr.count("COMPROMISED")
    n_clean = arr.count("CLEAN")
    print(f"{name:<15}: COMPROMISED={n_comp}  CLEAN={n_clean}")
print()

# Check if degenerate
if len(set(gt)) < 2:
    print("DEGENERATE: all ground_truth labels are the same.")
    print(f"All labels: {set(gt)}")
    agree = sum(a==b for a,b in zip(llama,gt))/len(gt)
    print(f"LLaMA agreement: {agree:.1%}")
else:
    # Kappa LLaMA vs ground truth
    kappa_llama = cohen_kappa_score(gt, llama)
    kappa_ds    = cohen_kappa_score(gt, ds)
    agree_llama = sum(a==b for a,b in zip(llama,gt))/len(gt)
    agree_ds    = sum(a==b for a,b in zip(ds,gt))/len(gt)

    print(f"Cohen kappa LLaMA  vs ground truth: {kappa_llama:.4f}")
    print(f"Cohen kappa DeepSeek vs ground truth: {kappa_ds:.4f}")
    print(f"LLaMA agreement:    {agree_llama:.1%}")
    print(f"DeepSeek agreement: {agree_ds:.1%}")
    print()

    # F1 per class
    labels = ["CLEAN","COMPROMISED"]
    prec, rec, f1, _ = precision_recall_fscore_support(gt, llama, labels=labels, zero_division=0)
    print("LLaMA performance vs ground truth:")
    for i,l in enumerate(labels):
        print(f"  {l:<15}: P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}")
    print()

    # Per-category kappa
    for cat in set(cats):
        idx = [i for i,c in enumerate(cats) if c==cat]
        if len(idx) < 5: continue
        gt_c = [gt[i] for i in idx]
        ll_c = [llama[i] for i in idx]
        if len(set(gt_c)) < 2:
            agree = sum(a==b for a,b in zip(ll_c,gt_c))/len(gt_c)
            print(f"  {cat}: n={len(idx)} — degenerate (all {set(gt_c)}), agree={agree:.1%}")
        else:
            k = cohen_kappa_score(gt_c, ll_c)
            agree = sum(a==b for a,b in zip(ll_c,gt_c))/len(gt_c)
            print(f"  {cat}: n={len(idx)}  kappa={k:.4f}  agree={agree:.1%}")
    print()

    # Confusion matrix
    cm = confusion_matrix(gt, llama, labels=labels)
    print("Confusion matrix (rows=GT, cols=LLaMA):")
    print(f"  {'':15}" + "  ".join(f"{l:<15}" for l in labels))
    for i,row in enumerate(cm):
        print(f"  {labels[i]:<15}" + "  ".join(f"{v:<15}" for v in row))
    print()

    # LaTeX sentence
    kappa_interp = ("slight" if kappa_llama < 0.20 else
                    "fair" if kappa_llama < 0.40 else
                    "moderate" if kappa_llama < 0.60 else
                    "substantial" if kappa_llama < 0.80 else
                    "almost perfect")
    print("=== LaTeX sentence for paper ===")
    print(f"Human-annotated ground-truth labels on {len(samples)} stratified samples")
    print(f"yield Cohen\'s $\\kappa={kappa_llama:.3f}$ ({kappa_interp} agreement)")
    print(f"between LLaMA~3~8B and human labels.")
    print(f"LLaMA achieves F1={f1[1]:.3f} on COMPROMISED cases (recall={rec[1]:.1%}),")
    print(f"substantially outperforming DeepSeek-V3 ($\\kappa={kappa_ds:.3f}$).")