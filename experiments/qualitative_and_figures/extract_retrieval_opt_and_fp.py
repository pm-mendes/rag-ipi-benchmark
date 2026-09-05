"""
extract_retrieval_opt_and_fp.py
================================
Two analyses on existing JSON results — no new LLM calls needed.

#15 — Retrieval-optimized attack results
  Reads results/attack_retrieval_optimized.json and reports:
  - ASR per attack family
  - AHR@k (similarity gain vs generic attack)
  - Comparison with generic attack ASR from ablation_v2_results.json

#18 — False positive rate by defense layer
  Reads ablation_v2_results.json (or ablation_custom_full_results.json)
  and computes how often each layer blocks a legitimate query
  (i.e., SAU=0 on a run that had no adversarial document).
  Falls back to estimating FPR from SAU degradation if raw
  per-query data is unavailable.

Usage:
  python extract_retrieval_opt_and_fp.py \
    --retrieval-opt-file  results/attack_retrieval_optimized.json \
    --ablation-file       results/ablation_v2_results.json \
    --semantic-file       results/ablation_custom_full_results.json \
    --output              results/retrieval_opt_and_fp_results.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# ── #15: Retrieval-optimized attack analysis ──────────────────

def analyze_retrieval_optimized(data) -> Dict:
    """
    Handles multiple possible structures for retrieval-optimized results:
      A) List of {attack_type, asr, ahr, sim_gain, ...}
      B) Dict with keys like 'results', 'attacks', etc.
      C) List of {config, attacks: [{attack_type, asr, ...}]}
    """
    if not data:
        return {"error": "empty data"}

    results = []

    # Structure A or C: list
    if isinstance(data, list):
        for entry in data:
            if "attack_type" in entry:
                # Structure A: flat list of attack results
                results.append({
                    "attack":    entry.get("attack_type", entry.get("attack", "?")),
                    "asr":       entry.get("asr", entry.get("ASR", None)),
                    "ahr":       entry.get("ahr", entry.get("AHR", entry.get("ahr_k3", None))),
                    "sim_gain":  entry.get("sim_gain", entry.get("similarity_gain", None)),
                    "asr_generic": entry.get("asr_generic", entry.get("baseline_asr", None)),
                })
            elif "config" in entry:
                # Structure C: standard ablation format
                for atk in entry.get("attacks", []):
                    results.append({
                        "attack":  atk.get("attack_type", "?"),
                        "config":  entry.get("config"),
                        "asr":     atk.get("asr", None),
                        "ahr":     atk.get("ahr", None),
                        "sim_gain": atk.get("sim_gain", None),
                    })
            elif "attack" in entry:
                # Structure with 'attack' key (like retriever format)
                results.append({
                    "attack":   entry.get("attack", "?"),
                    "asr":      entry.get("asr", None),
                    "ahr":      entry.get("ahr", None),
                    "sim_gain": entry.get("sim_gain", entry.get("similarity_gain", None)),
                })

    # Structure B: dict
    elif isinstance(data, dict):
        for key in ["results", "attacks", "data"]:
            if key in data and isinstance(data[key], list):
                return analyze_retrieval_optimized(data[key])
        # Maybe the dict IS the result
        results = [{"attack": k, "asr": v if isinstance(v, float) else
                    v.get("asr") if isinstance(v, dict) else None}
                   for k, v in data.items()]

    if not results:
        return {"error": "could not parse structure", "raw_keys":
                list(data[0].keys()) if isinstance(data, list) and data else str(type(data))}

    # Aggregate
    valid = [r for r in results if r.get("asr") is not None]
    mean_asr = float(np.mean([r["asr"] for r in valid])) if valid else None
    sim_gains = [r["sim_gain"] for r in results if r.get("sim_gain") is not None]
    mean_sim_gain = float(np.mean(sim_gains)) if sim_gains else None

    return {
        "n_attacks":     len(results),
        "mean_asr":      round(mean_asr, 4) if mean_asr is not None else None,
        "mean_sim_gain": round(mean_sim_gain, 4) if mean_sim_gain is not None else None,
        "per_attack":    results,
        "summary":       (f"Retrieval-optimized attacks: mean ASR={mean_asr:.1%}"
                          if mean_asr is not None else "ASR not found in data"),
    }


def compare_with_generic(opt_results: Dict,
                          ablation_data: List[Dict],
                          config_name: str = "No defense") -> Dict:
    """
    Compares retrieval-optimized ASR with generic (no-prefix) ASR
    from the ablation file for the same attack families.
    """
    generic_asr = {}
    for entry in ablation_data:
        if entry.get("config", "").lower() == config_name.lower():
            for atk in entry.get("attacks", []):
                generic_asr[atk["attack_type"]] = atk.get("asr", 0.0)

    comparisons = []
    for r in opt_results.get("per_attack", []):
        atk = r.get("attack", "")
        gen = generic_asr.get(atk, None)
        if gen is not None and r.get("asr") is not None:
            comparisons.append({
                "attack":        atk,
                "asr_generic":   gen,
                "asr_optimized": r["asr"],
                "uplift":        round(r["asr"] - gen, 4),
                "uplift_pct":    f"+{(r['asr']-gen)*100:.1f}pp",
            })

    mean_uplift = (float(np.mean([c["uplift"] for c in comparisons]))
                   if comparisons else None)
    return {
        "comparisons":  comparisons,
        "mean_uplift":  round(mean_uplift, 4) if mean_uplift is not None else None,
        "n_compared":   len(comparisons),
    }


# ── #18: False positive rate by layer ────────────────────────

def compute_false_positive_rates(ablation_data: List[Dict]) -> Dict:
    """
    Estimates false positive rate (legitimate documents/responses
    incorrectly blocked) per defense layer.

    Approach:
      - For each config, the SAU degradation on clean queries
        (approximated from the difference between baseline SAU
        and defended SAU on the full dataset) estimates the rate
        at which legitimate responses are degraded.
      - If 'clean_queries' or 'clean_asr' fields exist, use directly.
      - Otherwise: FPR_layer ≈ (SAU_baseline - SAU_layer) / SAU_baseline
        (this is a conservative upper bound, not an exact FPR).

    Returns per-layer FPR estimates with interpretation.
    """
    # Find baseline (No defense) SAU
    baseline_sau = None
    layer_results = {}

    for entry in ablation_data:
        config = entry.get("config", "")
        attacks = entry.get("attacks", [])

        # Aggregate SAU across attacks
        sau_values = [a.get("ar", a.get("sau", a.get("SAU", None)))
                      for a in attacks if a.get("ar") or a.get("sau")]
        mean_sau = float(np.mean(sau_values)) if sau_values else None

        # Check for explicit FPR/clean data
        explicit_fpr = entry.get("fpr", entry.get("false_positive_rate", None))
        clean_block = entry.get("clean_blocked", entry.get("legitimate_blocked", None))

        layer_results[config] = {
            "mean_sau":     round(mean_sau, 4) if mean_sau is not None else None,
            "explicit_fpr": explicit_fpr,
            "clean_blocked": clean_block,
        }

        if config.lower() in ["no defense", "no_defense", "baseline"]:
            baseline_sau = mean_sau

    # Compute SAU-based FPR estimate
    fpr_estimates = {}
    for config, vals in layer_results.items():
        if baseline_sau and vals["mean_sau"] is not None:
            sau_drop = baseline_sau - vals["mean_sau"]
            fpr_est = max(0.0, sau_drop / baseline_sau) if baseline_sau > 0 else 0.0
            fpr_estimates[config] = {
                "sau_baseline":    round(baseline_sau, 4),
                "sau_defended":    vals["mean_sau"],
                "sau_drop":        round(sau_drop, 4),
                "fpr_upper_bound": round(fpr_est, 4),
                "explicit_fpr":    vals["explicit_fpr"],
                "interpretation": (
                    f"Up to {fpr_est:.1%} of legitimate responses "
                    f"degraded vs baseline (SAU drop = {sau_drop:.3f})"
                ),
            }
        else:
            fpr_estimates[config] = {
                "sau_baseline":    baseline_sau,
                "sau_defended":    vals["mean_sau"],
                "fpr_upper_bound": None,
                "explicit_fpr":    vals["explicit_fpr"],
                "interpretation":  "Insufficient data for FPR estimation",
            }

    return {
        "method":       "SAU-degradation upper bound (not exact FPR)",
        "note":         ("FPR is estimated as the fraction of legitimate "
                         "response utility lost vs no-defense baseline. "
                         "True FPR requires per-query clean-run evaluation."),
        "per_config":   fpr_estimates,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-opt-file",
        default="results/attack_retrieval_optimized.json")
    parser.add_argument("--ablation-file",
        default="results/ablation_v2_results.json")
    parser.add_argument("--semantic-file",
        default="results/ablation_custom_full_results.json")
    parser.add_argument("--output",
        default="results/retrieval_opt_and_fp_results.json")
    args = parser.parse_args()

    def load(path):
        p = Path(path)
        if not p.exists():
            print(f"[WARN] Not found: {path}")
            return None
        with open(p) as f:
            return json.load(f)

    ret_opt_data  = load(args.retrieval_opt_file)
    ablation_data = load(args.ablation_file)
    semantic_data = load(args.semantic_file)

    output = {}

    # ── #15 ──
    print("=" * 60)
    print("#15 — Retrieval-Optimized Attack Analysis")
    print("=" * 60)
    if ret_opt_data is not None:
        opt_results = analyze_retrieval_optimized(ret_opt_data)
        print(f"Structure detected: {len(opt_results.get('per_attack', []))} attack entries")
        print(f"Summary: {opt_results.get('summary', 'N/A')}")
        if opt_results.get("mean_sim_gain"):
            print(f"Mean similarity gain: +{opt_results['mean_sim_gain']:.3f}")

        if ablation_data:
            comparison = compare_with_generic(opt_results, ablation_data)
            print(f"\nComparison with generic attacks (No defense):")
            for c in comparison.get("comparisons", []):
                print(f"  {c['attack']:<25} generic={c['asr_generic']:.1%} "
                      f"optimized={c['asr_optimized']:.1%} "
                      f"uplift={c['uplift_pct']}")
            if comparison.get("mean_uplift") is not None:
                print(f"  Mean uplift: +{comparison['mean_uplift']*100:.1f}pp")
            opt_results["comparison_with_generic"] = comparison

        output["retrieval_optimized"] = opt_results
    else:
        print("[SKIP] File not found.")
        output["retrieval_optimized"] = {"error": "file not found"}

    # ── #18 ──
    print()
    print("=" * 60)
    print("#18 — False Positive Rate by Defense Layer")
    print("=" * 60)

    for label, data in [("heuristic", ablation_data),
                        ("semantic",  semantic_data)]:
        if data:
            fp_results = compute_false_positive_rates(data)
            print(f"\n[{label} ablation]")
            print(f"Method: {fp_results['method']}")
            for config, vals in fp_results["per_config"].items():
                fpr = vals.get("fpr_upper_bound")
                sau_drop = vals.get("sau_drop", 0)
                if fpr is not None:
                    print(f"  {config:<25} SAU_drop={sau_drop:.3f} "
                          f"FPR_upper={fpr:.1%}")
                else:
                    print(f"  {config:<25} insufficient data")
            output[f"false_positive_rates_{label}"] = fp_results

    # ── Save ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DONE] Saved to {out_path}")


if __name__ == "__main__":
    main()