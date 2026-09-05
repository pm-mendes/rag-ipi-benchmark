"""
integrate_100q_results.py
=========================
Reads results/ablation_100q_results.json and produces:
  1. Bootstrap 95% CI on ASR per config
  2. LaTeX snippet ready to paste into the paper (v32)
  3. Comparison table vs 10q ablation

Usage (after run completes):
  python integrate_100q_results.py \
    --results-100q  results/ablation_100q_results.json \
    --results-10q   results/ablation_v2_results.json \
    --output        results/ablation_100q_summary.json
"""

import json
import numpy as np
from pathlib import Path
import argparse


def bootstrap_ci(successes: int, n: int,
                 n_boot: int = 10000, seed: int = 42) -> tuple:
    """Bootstrap 95% CI on a proportion."""
    rng = np.random.default_rng(seed)
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    boots = rng.binomial(n, p, size=n_boot) / n
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_results(path: str) -> list:
    if not Path(path).exists():
        return []
    return json.load(open(path))


def summarize_100q(data: list) -> dict:
    summary = {}
    for entry in data:
        cfg = entry.get("config", "")
        attacks = entry.get("attacks", [])
        n_total = sum(a.get("n_queries", 0) for a in attacks)
        n_success = sum(
            round(a.get("asr", 0) * a.get("n_queries", 0))
            for a in attacks
        )
        asr = n_success / n_total if n_total > 0 else 0
        lo, hi = bootstrap_ci(n_success, n_total)

        canon = [a for a in attacks if a.get("attack_type") in
                 ["OVERRIDE","EXFIL","ROLE","DENIAL",
                  "TECHNICAL_OVERRIDE","DATA_EXFIL"]]
        n_c = sum(a.get("n_queries", 0) for a in canon)
        n_c_s = sum(round(a.get("asr",0)*a.get("n_queries",0)) for a in canon)
        asr_c = n_c_s / n_c if n_c > 0 else 0
        lo_c, hi_c = bootstrap_ci(n_c_s, n_c)

        mean_ar = np.mean([a.get("ar", 0) for a in attacks]) if attacks else 0
        mean_ci_val = np.mean([a.get("ci", 1) for a in attacks]) if attacks else 1

        summary[cfg] = {
            "n_queries":     n_total,
            "asr":           round(asr, 4),
            "asr_ci_lo":     round(lo, 4),
            "asr_ci_hi":     round(hi, 4),
            "asr_canonical": round(asr_c, 4),
            "asr_canon_lo":  round(lo_c, 4),
            "asr_canon_hi":  round(hi_c, 4),
            "mean_ar":       round(float(mean_ar), 4),
            "mean_ci":       round(float(mean_ci_val), 4),
        }
    return summary


def print_latex(summary: dict, summary_10q: dict) -> str:
    lines = []
    lines.append("% ── 100-query ablation results ──────────────────────────────")
    lines.append("% Add to Section VIII or Table I caption:")
    lines.append("%")
    lines.append("% Bootstrap 95\\% CI over 100 queries (Custom 30q + MITRE 30q + TriviaQA 40q),")
    lines.append("% 6 canonical attacks, GPT-3.5-turbo.")
    lines.append("%")

    for cfg, v in summary.items():
        lo  = v['asr_canon_lo']
        hi  = v['asr_canon_hi']
        asr = v['asr_canonical']
        n   = v['n_queries']
        ar  = v['mean_ar']
        ci_val = v['mean_ci']
        lines.append(f"% {cfg:<20}: ASR={asr:.1%} [{lo:.1%},{hi:.1%}]  "
                     f"AR={ar:.3f}  CI={ci_val:.3f}  n={n}")

    lines.append("%")
    lines.append("% Sentence for paper body:")
    nd  = summary.get("no_defense",    {})
    ah  = summary.get("all_heuristic", {})
    sf  = summary.get("semantic_full", {})

    if nd and ah:
        lines.append(f"% Extended evaluation on 100 queries confirms controlled results:")
        lines.append(f"% No defense ASR={nd['asr_canonical']:.1%} "
                     f"[{nd['asr_canon_lo']:.1%},{nd['asr_canon_hi']:.1%}],")
        lines.append(f"% All heuristic ASR={ah['asr_canonical']:.1%} "
                     f"[{ah['asr_canon_lo']:.1%},{ah['asr_canon_hi']:.1%}].")
        if sf:
            lines.append(f"% Semantic full ASR={sf['asr_canonical']:.1%} "
                         f"[{sf['asr_canon_lo']:.1%},{sf['asr_canon_hi']:.1%}].")

    lines.append("% ────────────────────────────────────────────────────────────")
    result = "\n".join(lines)
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-100q",
        default="results/ablation_100q_results.json")
    parser.add_argument("--results-10q",
        default="results/ablation_v2_results.json")
    parser.add_argument("--output",
        default="results/ablation_100q_summary.json")
    args = parser.parse_args()

    data_100q = load_results(args.results_100q)
    data_10q  = load_results(args.results_10q)

    if not data_100q:
        print(f"[ERROR] {args.results_100q} not found or empty.")
        return

    print(f"Loaded {len(data_100q)} config entries from 100q results.")
    summary_100q = summarize_100q(data_100q)
    summary_10q  = summarize_100q(data_10q) if data_10q else {}

    print()
    print("=== 100-QUERY ABLATION SUMMARY ===")
    print(f"{'Config':<20} {'ASR (canon)':>12} {'95% CI':>20} {'AR':>8} {'CI_val':>8}")
    print("-" * 72)
    for cfg, v in summary_100q.items():
        print(f"{cfg:<20} {v['asr_canonical']:>11.1%} "
              f"  [{v['asr_canon_lo']:.1%},{v['asr_canon_hi']:.1%}]"
              f"  {v['mean_ar']:>7.3f}  {v['mean_ci']:>7.3f}")

    if summary_10q:
        print()
        print("=== COMPARISON 10q vs 100q ===")
        print(f"{'Config':<20} {'ASR 10q':>10} {'ASR 100q':>10} {'Delta':>8}")
        print("-" * 52)
        for cfg in summary_100q:
            a10 = summary_10q.get(cfg, {}).get("asr_canonical", None)
            a100 = summary_100q[cfg]["asr_canonical"]
            if a10 is not None:
                delta = a100 - a10
                print(f"{cfg:<20} {a10:>9.1%}  {a100:>9.1%}  {delta:>+7.1%}")

    print()
    latex = print_latex(summary_100q, summary_10q)

    out = {
        "summary_100q": summary_100q,
        "summary_10q":  summary_10q,
        "latex_comment": latex,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\n[DONE] Saved to {args.output}")
    print()
    print("Once run completes, integrate into paper with:")
    print("  python integrate_100q_results.py")
    print("  Then share output here for v32 integration.")


if __name__ == "__main__":
    main()