"""
compute_bootstrap_ci.py — Bootstrap 95% CI sur ASR, RD, CI, AR
Bootstrap sur les attaques (unités d'échantillonnage disponibles).
"""

import json
import numpy as np
from pathlib import Path

SEED = 42
N_BOOTSTRAP = 10000
rng = np.random.default_rng(SEED)

def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05):
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    boot_means = [rng.choice(values, size=n, replace=True).mean()
                  for _ in range(n_boot)]
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lower), float(upper)

def process_file(json_path, asr_key="avg_asr"):
    print(f"\n{'='*62}")
    print(f"Fichier : {json_path.name}")
    print(f"{'='*62}")
    print(f"{'Config':<22} {'ASR mean':>9} {'95% CI':>18} {'n_attacks':>10}")
    print("-"*62)

    with open(json_path) as f:
        data = json.load(f)

    ci_results = {}
    for entry in data:
        cfg = entry["config"]
        attacks = entry.get("attacks", [])
        if not attacks:
            continue

        asr_vals = [a["asr"] for a in attacks]
        rd_vals  = [a.get("rd", 0) for a in attacks]
        ci_vals  = [a.get("ci", 1) for a in attacks]
        ar_vals  = [a.get("ar", float('nan')) for a in attacks
                    if not np.isnan(a.get("ar", float('nan')))]

        asr_m, asr_l, asr_u = bootstrap_ci(asr_vals)
        rd_m,  rd_l,  rd_u  = bootstrap_ci(rd_vals)
        ci_m,  ci_l,  ci_u  = bootstrap_ci(ci_vals)

        print(f"  {cfg:<20} {asr_m:>8.1%}  "
              f"[{asr_l:.1%}, {asr_u:.1%}]  "
              f"{len(asr_vals):>6} attacks")

        ci_results[cfg] = {
            "n_attacks": len(asr_vals),
            "asr": {"mean": round(asr_m,4),
                    "ci95_low": round(asr_l,4),
                    "ci95_high": round(asr_u,4)},
            "rd":  {"mean": round(rd_m,4),
                    "ci95_low": round(rd_l,4),
                    "ci95_high": round(rd_u,4)},
            "ci":  {"mean": round(ci_m,4),
                    "ci95_low": round(ci_l,4),
                    "ci95_high": round(ci_u,4)},
        }
        if ar_vals:
            ar_m, ar_l, ar_u = bootstrap_ci(ar_vals)
            ci_results[cfg]["ar"] = {
                "mean": round(ar_m,4),
                "ci95_low": round(ar_l,4),
                "ci95_high": round(ar_u,4)
            }
            print(f"    AR={ar_m:.3f} [{ar_l:.3f}, {ar_u:.3f}]")

    return ci_results

if __name__ == "__main__":
    results_dir = Path("results")
    files = {
        "ablation_v2": "ablation_v2_results.json",
        "custom_full": "ablation_custom_full_results.json",
        "nq":          "ablation_nq_results.json",
        "hotpotqa":    "ablation_hotpotqa_results.json",
    }

    all_ci = {}
    for label, fname in files.items():
        fpath = results_dir / fname
        if fpath.exists():
            all_ci[label] = process_file(fpath)
        else:
            print(f"\n[SKIP] {fname}")

    out = results_dir / "bootstrap_ci_results.json"
    with open(out, "w") as f:
        json.dump(all_ci, f, indent=2)

    # Résumé LaTeX pour Table I
    print("\n\n" + "="*62)
    print("LATEX — Table I enrichie avec bootstrap CI")
    print("="*62)
    if "ablation_v2" in all_ci:
        for cfg, m in all_ci["ablation_v2"].items():
            asr = m["asr"]
            rd  = m["rd"]
            ci  = m["ci"]
            ar  = m.get("ar", {"mean":0,"ci95_low":0,"ci95_high":0})
            print(f"  {cfg:<20} "
                  f"ASR={asr['mean']:.1%}$_{{[{asr['ci95_low']:.1%},{asr['ci95_high']:.1%}]}}$ "
                  f"RD={rd['mean']:.3f} CI={ci['mean']:.3f}")

    print(f"\nRésultats → {out}")
