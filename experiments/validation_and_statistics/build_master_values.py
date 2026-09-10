"""
build_master_values.py — §4/§15 of the camera-ready brief: the single
script that reads every raw result file produced during this revision and
emits ONE canonical table of the values actually cited in paper/main.tex,
each tagged with its exact source file/field. This is the anti-pattern to
"reconcile a number by picking whichever looks more convenient" that the
brief explicitly forbids: every number below is read programmatically from
a committed JSON, not retyped by hand.

Run this after any result file changes, before touching paper/main.tex, so
the LaTeX source and results/*.json cannot silently drift apart. Diff
results/MASTER_VALUES.json across commits to see exactly which reported
numbers changed and why.

Covers the corrections/new results from this revision pass specifically
(pre-existing tables audited in docs/AUDIT_camera_ready.md but not
re-derived by a script are listed as "audited, not recomputed" -- see that
document for their provenance instead of duplicating it here).
"""

import json
from pathlib import Path

R = Path("results")


def load(name):
    p = R / name
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    master = {}

    # ── Table III (tab:heuristic) — corrected Isolate-only ASR + exact CIs ──
    recovered = load("statistical_results_RECOVERED_3seeds.json")
    pooled_ci = load("table3_pooled_3seed_exact_ci.json")
    if recovered and pooled_ci:
        master["table3_heuristic_ablation"] = {
            "source": "statistical_results_RECOVERED_3seeds.json + table3_pooled_3seed_exact_ci.json",
            "rows": {
                cfg: {
                    "asr_mean": recovered[cfg]["asr_mean"],
                    "asr_std": recovered[cfg]["asr_std"],
                    "rd_mean": recovered[cfg]["rd_mean"], "rd_std": recovered[cfg]["rd_std"],
                    "ci_mean": recovered[cfg]["ci_mean"], "ci_std": recovered[cfg]["ci_std"],
                    "sau_mean": recovered[cfg]["ar_mean"], "sau_std": recovered[cfg]["ar_std"],
                    "asr_95_interval": pooled_ci.get(cfg, {}).get("ci95"),
                }
                for cfg in recovered
            },
        }

    # ── §3 rare-event / effect-size corrections ──
    ree = load("rare_event_and_effect_sizes.json")
    if ree:
        master["rare_event_effect_sizes"] = {
            "source": "rare_event_and_effect_sizes.json",
            "effect_no_defense_vs_all": ree.get("effect_no_defense_vs_all"),
        }

    # ── §3 corrected (unpaired) significance tests, replacing fabricated McNemar ──
    unpaired = load("unpaired_significance_corrected.json")
    if unpaired:
        master["significance_tests_corrected"] = {
            "source": "unpaired_significance_corrected.json",
            "note": "replaces the fabricated-pairing McNemar tests -- see docs/AUDIT_camera_ready.md Finding F",
            **unpaired,
        }

    # ── §5 retrieval-optimized attack, corrected "All heuristic" bug ──
    ro = load("retrieval_optimized_ahr_casr.json")
    if ro:
        master["retrieval_optimized_attack"] = {
            "source": "retrieval_optimized_ahr_casr.json",
            "note": "corrects the mislabeled-defense bug -- see docs/AUDIT_camera_ready.md Finding G",
            "no_defense": {k: ro["no_defense"][k] for k in
                           ("AHR_at_k", "CASR", "ASR", "mean_rank", "mean_legit_docs_in_context")},
            "all_heuristic": {k: ro["all_heuristic"][k] for k in
                              ("AHR_at_k", "CASR", "ASR", "mean_rank", "mean_legit_docs_in_context")},
        }

    # ── §6 retriever-optimized (Dense/BM25/Hybrid fair comparison) ──
    retr = load("retriever_optimized_attacks.json")
    if retr:
        master["retriever_fair_comparison"] = {
            "source": "retriever_optimized_attacks.json", "raw": retr,
        }

    # ── §7 L2 isolation ablation ──
    l2 = load("l2_isolation_ablation.json")
    if l2:
        master["l2_isolation_ablation"] = {"source": "l2_isolation_ablation.json", "raw": l2}

    # ── §9 benign-hard FPR ──
    bh = load("benign_hard_fpr_results.json")
    if bh:
        master["benign_hard_fpr"] = {
            "source": "benign_hard_fpr_results.json",
            "l1_fpr": bh["l1_fpr"], "l1c_fpr_at_0.05": bh["l1c_fpr_at_0.05"],
            "mean_utility_loss_when_l1_flags": bh["mean_utility_loss_when_l1_flags"],
        }

    # ── §10 L1c threshold calibration ──
    l1c = load("l1c_threshold_calibration.json")
    if l1c:
        master["l1c_calibration"] = {"source": "l1c_threshold_calibration.json", "raw": l1c}

    # ── §11 prompt assembly ablation ──
    pa = load("prompt_assembly_ablation.json")
    if pa:
        by_format = {}
        for row in pa:
            by_format.setdefault(row["format"], []).append(row)
        master["prompt_assembly"] = {
            "source": "prompt_assembly_ablation.json",
            "mean_asr_by_format": {
                fmt: round(sum(r["asr"] for r in rows) / len(rows), 4)
                for fmt, rows in by_format.items()
            },
        }

    # ── Finding D: extended canary probe ──
    canary = load("canary_extended_results.json")
    if canary:
        n = len(canary)
        n_leaked = sum(r["leaked"] for r in canary)
        master["canary_extended"] = {
            "source": "canary_extended_results.json",
            "n_total": n, "n_leaked": n_leaked, "clr": round(n_leaked / n, 4),
        }

    # ── §12 L0 fidelity ──
    l0 = load("l0_fidelity_results.json")
    if l0:
        legit = [k for k in l0 if k != "attack_payload"]
        master["l0_fidelity"] = {
            "source": "l0_fidelity_results.json",
            "n_legit_categories": len(legit),
            "n_visible_changed_conservative": sum(
                1 for k in legit if l0[k]["conservative"]["visible_content_changed"]),
        }

    # ── Finding E: independent O1-O4 judge vs. marker-based verdict ──
    judge = load("independent_o1_o4_judge_results.json")
    if judge:
        n = len(judge)
        agree = sum(1 for r in judge
                    if r["independent_judge_compromised"] == r["l3_marker_verdict_compromised"])
        master["independent_judge_agreement"] = {
            "source": "independent_o1_o4_judge_results.json",
            "n": n, "agree": agree, "agreement_rate": round(agree / n, 4),
        }

    # ── §8 NQ EM/F1 ──
    nq = load("nq_em_f1_results.json")
    if nq:
        master["nq_em_f1"] = {
            "source": "nq_em_f1_results.json",
            "n": nq["n"], "mean_EM": nq["mean_EM"], "mean_F1": nq["mean_F1"],
            "mean_SAU": nq["mean_SAU"],
            "pearson_r_SAU_vs_F1": nq["pearson_r_SAU_vs_F1"],
            "pearson_r_SAU_vs_EM": nq["pearson_r_SAU_vs_EM"],
        }

    with open(R / "MASTER_VALUES.json", "w") as f:
        json.dump(master, f, indent=2)

    print(f"=== Master values consolidated from {len(master)} result groups ===")
    for k in master:
        print(f"  - {k}  (source: {master[k].get('source', '?')})")
    print(f"\n→ results/MASTER_VALUES.json")
    print("Cross-check every corresponding number in paper/main.tex against this "
          "file before any future edit -- see docs/AUDIT_camera_ready.md for the "
          "pre-existing (audited-but-not-rescripted) tables this doesn't cover.")


if __name__ == "__main__":
    main()
