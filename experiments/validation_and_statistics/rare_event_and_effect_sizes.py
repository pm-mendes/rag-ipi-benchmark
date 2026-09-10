"""
rare_event_and_effect_sizes.py — Statistical redo requested in §3 of the
camera-ready brief:

  1. Replace the degenerate bootstrap [0%, 0%] CIs (see
     bootstrap_confidence_intervals.py) with an exact one-sided upper bound
     for zero-success configurations. The existing bootstrap resamples the
     6 per-attack ASR *point estimates* (n=6), which collapses to a point
     mass at 0 whenever every attack's ASR is exactly 0 — that is not a
     meaningful confidence interval, it is an artifact of bootstrapping too
     few, already-aggregated units.
  2. Report an effect size + CI (risk difference) for the headline
     comparisons, not just a p-value.
  3. Make explicit that the McNemar test at n=20 (L1b vs. Semantic full) is
     underpowered — report post-hoc what risk-difference magnitude that
     sample size could actually detect, instead of only citing p=0.480.

Data caveat, stated once here rather than re-derived per number: no result
file in this repository stores raw per-query binary outcomes — every script
saves only the rounded per-attack ASR. Where the attack was run on exactly
N=10 queries (the Custom-10 ablations; confirmed in
docs/AUDIT_camera_ready.md Finding A by exhaustive search over candidate
N), the exact success count is recoverable as round(asr * 10) without
ambiguity. This script only applies the exact-binomial treatment where that
recovery is unambiguous, and says so per config; it does not invent counts
for files where N is not confirmed.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

N_QUERIES_CUSTOM10 = 10  # confirmed unambiguous in docs/AUDIT_camera_ready.md


def recover_counts(attacks, n_queries=N_QUERIES_CUSTOM10, tol=1e-2):
    """Recover exact (successes, n) per attack from a rounded ASR ratio.
    Returns None if any value is not consistent with n_queries (so we never
    silently fabricate a count)."""
    counts = []
    for a in attacks:
        k = a["asr"] * n_queries
        if abs(k - round(k)) > tol:
            return None
        counts.append(int(round(k)))
    return counts


def exact_binomial_ci(successes, n, alpha=0.05, one_sided_upper_if_zero=True):
    """Clopper-Pearson exact CI. If successes == 0 and
    one_sided_upper_if_zero, report a one-sided 95% upper bound (the
    standard treatment for a rare/zero-event rate) instead of a symmetric
    two-sided interval whose lower bound is trivially 0."""
    if successes == 0 and one_sided_upper_if_zero:
        upper = stats.beta.ppf(1 - alpha, successes + 1, n - successes)
        return {"type": "one-sided upper (rule for zero events)",
                "point": 0.0, "lower": 0.0, "upper": float(upper)}
    lower = stats.beta.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return {"type": "two-sided (Clopper-Pearson)",
            "point": successes / n, "lower": float(lower), "upper": float(upper)}


def risk_difference_ci(k1, n1, k2, n2, alpha=0.05):
    """Wald CI on the risk difference p1 - p2 (Newcombe/Wald hybrid would be
    more accurate at small n; Wald is reported here with the small-n caveat
    made explicit in the output rather than hidden)."""
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    return {"risk_difference": diff, "se": float(se),
            "ci95_low": diff - z * se, "ci95_high": diff + z * se,
            "n1": n1, "n2": n2, "small_n_warning": (n1 < 30 or n2 < 30)}


def minimum_detectable_effect(n_pairs, power=0.80, alpha=0.05):
    """For a paired McNemar-type design with n discordant-eligible pairs,
    what is the smallest true discordance rate difference that a test at
    this n could reliably detect? Rough normal-approximation sample-size
    inversion — reported to make the 'underpowered' claim concrete rather
    than asserted."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # Solve n = (z_a + z_b)^2 / d^2 for d (assuming small discordant
    # proportion, conservative approximation).
    d = (z_alpha + z_beta) / np.sqrt(n_pairs)
    return float(d)


def process_ablation_file(path, label):
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data:
        cfg = entry["config"]
        attacks = entry.get("attacks", [])
        if not attacks:
            continue
        counts = recover_counts(attacks)
        if counts is None:
            out[cfg] = {"status": "N not confirmed unambiguous — skipped, "
                                   "see docstring; do not report a bootstrap "
                                   "CI here without confirming N first"}
            continue
        total_k, total_n = sum(counts), len(counts) * N_QUERIES_CUSTOM10
        ci = exact_binomial_ci(total_k, total_n)
        out[cfg] = {
            "successes": total_k, "n_trials": total_n,
            "n_attacks": len(counts), "n_queries_per_attack": N_QUERIES_CUSTOM10,
            "exact_ci": ci,
        }
    print(f"\n=== {label} ({path.name}) — exact binomial CIs "
          f"(replaces degenerate [0%,0%] bootstrap) ===")
    for cfg, r in out.items():
        if "exact_ci" in r:
            ci = r["exact_ci"]
            print(f"  {cfg:<16} {r['successes']:>3}/{r['n_trials']:<3} "
                  f"ASR={ci['point']:.1%}  {ci['type']}: "
                  f"[{ci['lower']:.1%}, {ci['upper']:.1%}]")
        else:
            print(f"  {cfg:<16} {r['status']}")
    return out


def main():
    results_dir = Path("results")
    all_out = {}

    all_out["ablation_v2"] = process_ablation_file(
        results_dir / "ablation_v2_results.json", "Table III (Custom-10)")

    # ── Effect size: No defense vs. All defenses (ablation_v2) ──────────
    with open(results_dir / "ablation_v2_results.json") as f:
        v2 = json.load(f)
    nodef = next(c for c in v2 if c["config"] == "No defense")
    alldef = next(c for c in v2 if c["config"] == "All defenses")
    k1, n1 = sum(recover_counts(nodef["attacks"])), len(nodef["attacks"]) * N_QUERIES_CUSTOM10
    k2, n2 = sum(recover_counts(alldef["attacks"])), len(alldef["attacks"]) * N_QUERIES_CUSTOM10
    rd = risk_difference_ci(k1, n1, k2, n2)
    all_out["effect_no_defense_vs_all"] = rd
    print(f"\n=== Effect size: No-defense vs. All-defenses (Custom-10) ===")
    print(f"  Risk difference = {rd['risk_difference']:.1%} "
          f"[{rd['ci95_low']:.1%}, {rd['ci95_high']:.1%}]  "
          f"(n1={n1}, n2={n2}{'  -- SMALL-N, Wald approx.' if rd['small_n_warning'] else ''})")

    # ── McNemar n=20 power check (L1b vs. Semantic full) ────────────────
    mde = minimum_detectable_effect(n_pairs=20)
    all_out["mcnemar_l1b_vs_semantic_power_check"] = {
        "n_pairs": 20,
        "minimum_detectable_discordance_diff_at_80pct_power": mde,
        "interpretation": (
            f"At n=20 paired queries, a test at alpha=0.05 with 80% power "
            f"could only reliably detect a discordance-rate difference of "
            f"roughly {mde:.1%} or larger. The reported non-significant "
            f"p=0.480 (chi2=0.500) is consistent with a real effect of "
            f"this or smaller magnitude going undetected -- 'not "
            f"significant' at this n means 'underpowered to confirm or "
            f"rule out the effect size actually observed', not 'no "
            f"effect'. Report both facts together, per the brief's "
            f"required phrasing: 'we observe a recurring non-monotonic "
            f"pattern' rather than 'non-monotonicity is proven' or "
            f"'disproven'."
        ),
    }
    print(f"\n=== McNemar (L1b vs. Semantic full, n=20) power check ===")
    print(f"  Minimum detectable discordance-rate difference at 80% power: "
          f"{mde:.1%}")
    print(f"  {all_out['mcnemar_l1b_vs_semantic_power_check']['interpretation']}")

    out_path = results_dir / "rare_event_and_effect_sizes.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2, default=str)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
