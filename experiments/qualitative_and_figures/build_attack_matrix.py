"""
build_attack_matrix.py
======================
Builds the 12-attack × 5-dataset × 3-defense ASR matrix
from existing JSON result files. No new LLM calls needed.

Output:
  - results/attack_matrix_results.json  (raw data)
  - attack_matrix.tex                   (LaTeX table, ready to paste)

Usage:
  python build_attack_matrix.py \
    --custom-file  results/ablation_custom_full_results.json \
    --nq-file      results/ablation_nq_results.json \
    --mitre-file   results/ablation_mitre_results.json \
    --triviaqa-file results/ablation_triviaqa_results.json \
    --hotpotqa-file results/ablation_hotpotqa_results.json \
    --output        results/attack_matrix_results.json \
    --tex-output    attack_matrix.tex
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


# ── Config name normalization ─────────────────────────────────

NO_DEF_ALIASES    = ["no defense", "no_defense", "none", "baseline", "nd"]
ALL_HEUR_ALIASES  = ["all heuristic", "all_heuristic", "all defenses",
                     "all_defenses", "ad", "heuristic full", "all heur",
                     "filter only", "filter_only"]   # Filter only ≈ closest to AH for NQ
SEM_FULL_ALIASES  = ["semantic full", "semantic_full", "sem full",
                     "sem_full", "full", "all semantic"]

# For NQ/MITRE/TriviaQA/HotpotQA we map the 5-config ablation:
# No defense → no_defense
# Filter only → all_heuristic  (best single heuristic layer — used as AH proxy)
# All defenses → all_heuristic (exact match)
# Isolate only, Verify only → skip (intermediate layers, not matrix targets)

def normalize_config(name: str) -> Optional[str]:
    """Maps config name to one of: no_defense | all_heuristic | semantic_full"""
    n = name.lower().strip()
    if any(a == n for a in ["no defense", "no_defense"]):
        return "no_defense"
    if n in ["all defenses", "all_defenses", "all heuristic", "all_heuristic"]:
        return "all_heuristic"
    if any(a in n for a in SEM_FULL_ALIASES):
        return "semantic_full"
    # Skip intermediate layers (Filter only, Isolate only, Verify only)
    return None


# ── Attack name normalization ─────────────────────────────────

ATTACK_CANONICAL_NAMES = [
    "OVERRIDE",
    "EXFIL",
    "ROLE",
    "DENIAL",
    "TECHNICAL_OVERRIDE",
    "DATA_EXFIL",
    "UNICODE_OVERRIDE",
    "IMPLICIT_INJECTION",
    "LLM_GENERATED",
    "BASE64_ENCODED",
    "MANY_SHOT",
    "AUTO_PARAPHRASE",
]

ATTACK_ALIASES = {
    "OVERRIDE":           ["override"],
    "EXFIL":              ["exfil", "exfiltration"],
    "ROLE":               ["role", "role_hijack", "role hijacking"],
    "DENIAL":             ["denial"],
    "TECHNICAL_OVERRIDE": ["technical_override", "technical override",
                           "tech_override", "technical"],
    "DATA_EXFIL":         ["data_exfil", "data exfil", "data_exfiltration"],
    "UNICODE_OVERRIDE":   ["unicode_override", "unicode override", "unicode"],
    "IMPLICIT_INJECTION": ["implicit_injection", "implicit injection",
                           "implicit"],
    "LLM_GENERATED":      ["llm_generated", "llm generated", "llm-generated"],
    "BASE64_ENCODED":     ["base64_encoded", "base64 encoded", "base64"],
    "MANY_SHOT":          ["many_shot", "many shot", "manyshot"],
    "AUTO_PARAPHRASE":    ["auto_paraphrase", "auto paraphrase", "paraphrase",
                           "keyword_free"],
}

def normalize_attack(name: str) -> Optional[str]:
    """
    Maps attack_type to canonical name.
    Handles dataset prefixes: NQ_OVERRIDE→OVERRIDE,
    MITRE_TECHNICAL_OVERRIDE→TECHNICAL_OVERRIDE, etc.
    """
    import re as _re
    # Strip known dataset prefixes
    stripped = _re.sub(
        r'^(NQ|MITRE|HOTPOT|HOTPOTQA|TRIVIAQA|SQUAD|CUSTOM)_',
        '', name.upper().strip()
    )
    # Exact match on canonical name
    if stripped in ATTACK_ALIASES:
        return stripped
    # Alias match (case-insensitive, on stripped name)
    s = stripped.lower()
    for canonical, aliases in ATTACK_ALIASES.items():
        if any(a.lower() == s or a.lower() in s or s in a.lower()
               for a in aliases):
            return canonical
    return None


# ── Data loader ───────────────────────────────────────────────

def load_dataset(path: str) -> Dict:
    """
    Loads a result JSON and returns:
    {config_normalized: {attack_normalized: asr}}
    """
    p = Path(path)
    if not p.exists():
        return {}

    with open(p) as f:
        data = json.load(f)

    result = {}
    if not isinstance(data, list):
        return {}

    for entry in data:
        cfg_raw = entry.get("config", "")
        cfg = normalize_config(cfg_raw)
        if cfg is None:
            continue

        if cfg not in result:
            result[cfg] = {}

        for atk in entry.get("attacks", []):
            atk_raw = atk.get("attack_type", "")
            atk_norm = normalize_attack(atk_raw)
            if atk_norm is None:
                continue
            result[cfg][atk_norm] = atk.get("asr", None)

    return result


# ── Matrix builder ────────────────────────────────────────────

DATASETS = ["custom", "nq", "mitre", "triviaqa", "hotpotqa"]
DATASET_LABELS = {
    "custom":   "Custom (30q)",
    "nq":       "SQuAD/NQ",
    "mitre":    "MITRE",
    "triviaqa": "TriviaQA",
    "hotpotqa": "HotpotQA",
}
CONFIGS_ORDER = ["no_defense", "all_heuristic", "semantic_full"]
CONFIG_SHORT   = {"no_defense": "ND", "all_heuristic": "AH", "semantic_full": "SF"}

# Attack family type labels for the table
ATTACK_TYPE = {
    "OVERRIDE":           "Canon.",
    "EXFIL":              "Canon.",
    "ROLE":               "Canon.",
    "DENIAL":             "Canon.",
    "TECHNICAL_OVERRIDE": "Canon.",
    "DATA_EXFIL":         "Canon.",
    "UNICODE_OVERRIDE":   "Adv.",
    "IMPLICIT_INJECTION": "Adv.",
    "LLM_GENERATED":      "Adapt.",
    "BASE64_ENCODED":      "Adapt.",
    "MANY_SHOT":          "Adapt.",
    "AUTO_PARAPHRASE":    "Adapt.",
}


def build_matrix(dataset_data: Dict[str, Dict]) -> Dict:
    """
    dataset_data: {dataset_name: {config: {attack: asr}}}
    Returns matrix: {attack: {dataset: {config: asr}}}
    """
    matrix = {}
    for atk in ATTACK_CANONICAL_NAMES:
        matrix[atk] = {}
        for ds in DATASETS:
            matrix[atk][ds] = {}
            ds_data = dataset_data.get(ds, {})
            for cfg in CONFIGS_ORDER:
                asr = ds_data.get(cfg, {}).get(atk, None)
                matrix[atk][ds][cfg] = asr
    return matrix


def fmt_asr(v, highlight_zero=True, highlight_high=True) -> str:
    """Format ASR value for LaTeX table."""
    if v is None:
        return "---"
    pct = v * 100
    s = f"{pct:.0f}\\%"
    if highlight_zero and pct == 0:
        return f"\\textbf{{0\\%}}"
    if highlight_high and pct >= 20:
        return f"\\textit{{{pct:.0f}\\%}}"
    return s


# ── LaTeX table generator ─────────────────────────────────────

def generate_latex_table(matrix: Dict,
                          dataset_data: Dict,
                          available_datasets: List[str]) -> str:
    """
    Generates a compact LaTeX table:
    Rows = attack families (grouped by type)
    Cols = dataset × config (ND / AH / SF)
    """
    n_ds = len(available_datasets)
    n_cols = 1 + 1 + n_ds * 3  # type + attack + (ND+AH+SF) per dataset

    # Column spec
    col_spec = "ll" + "|ccc" * n_ds

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Attack success rate (ASR\%) across 12 attack families,")
    lines.append(r"5 datasets, and 3 defense configurations.")
    lines.append(r"ND=No defense, AH=All heuristic, SF=Semantic full.")
    lines.append(r"\textbf{0\%}=fully neutralized; \textit{italics}=ASR$\geq$20\%.")
    lines.append(r"---=not evaluated on this dataset.}")
    lines.append(r"\label{tab:attack_matrix}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")

    # Header row 1: dataset names spanning 3 cols each
    header1 = r"\textbf{Type} & \textbf{Attack}"
    for ds in available_datasets:
        label = DATASET_LABELS.get(ds, ds)
        header1 += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{label}}}}}"
    lines.append(header1 + r" \\")

    # Header row 2: ND / AH / SF per dataset
    header2 = " & "
    for ds in available_datasets:
        header2 += " & ND & AH & SF"
    lines.append(r"\cmidrule(lr){1-2}" +
                 "".join(f"\\cmidrule(lr){{{3+i*3}-{5+i*3}}}"
                         for i in range(n_ds)))
    lines.append(header2 + r" \\")
    lines.append(r"\hline")

    # Data rows grouped by attack type
    prev_type = None
    for atk in ATTACK_CANONICAL_NAMES:
        atype = ATTACK_TYPE[atk]

        # Group separator
        if atype != prev_type:
            if prev_type is not None:
                lines.append(r"\hline")
            prev_type = atype

        # Short attack name for display
        atk_display = atk.replace("_", " ").title()
        atk_display = (atk_display
                       .replace("Exfil", "Exfil.")
                       .replace("Technical Override", "Tech.~Override")
                       .replace("Data Exfil", "Data~Exfil.")
                       .replace("Unicode Override", "Unicode~Ovr.")
                       .replace("Implicit Injection", "Implicit~Inj.")
                       .replace("Llm Generated", "LLM-Gen.")
                       .replace("Base64 Encoded", "Base64")
                       .replace("Many Shot", "Many-Shot")
                       .replace("Auto Paraphrase", "Paraphrase"))

        row = f"{atype} & {atk_display}"
        for ds in available_datasets:
            for cfg in CONFIGS_ORDER:
                v = matrix[atk][ds].get(cfg)
                row += f" & {fmt_asr(v)}"
        lines.append(row + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── Coverage report ───────────────────────────────────────────

def coverage_report(matrix: Dict,
                    available_datasets: List[str]) -> str:
    """Prints coverage: how many cells are populated."""
    total = len(ATTACK_CANONICAL_NAMES) * len(available_datasets) * len(CONFIGS_ORDER)
    populated = sum(
        1 for atk in ATTACK_CANONICAL_NAMES
        for ds in available_datasets
        for cfg in CONFIGS_ORDER
        if matrix[atk][ds].get(cfg) is not None
    )
    lines = [f"Matrix coverage: {populated}/{total} cells populated "
             f"({populated/total*100:.0f}%)"]

    # Per-dataset coverage
    for ds in available_datasets:
        ds_total = len(ATTACK_CANONICAL_NAMES) * len(CONFIGS_ORDER)
        ds_pop = sum(
            1 for atk in ATTACK_CANONICAL_NAMES
            for cfg in CONFIGS_ORDER
            if matrix[atk][ds].get(cfg) is not None
        )
        lines.append(f"  {DATASET_LABELS[ds]:<20}: {ds_pop}/{ds_total}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-file",
        default="results/ablation_custom_full_results.json")
    parser.add_argument("--nq-file",
        default="results/ablation_nq_results.json")
    parser.add_argument("--mitre-file",
        default="results/ablation_mitre_results.json")
    parser.add_argument("--triviaqa-file",
        default="results/ablation_triviaqa_results.json")
    parser.add_argument("--hotpotqa-file",
        default="results/ablation_hotpotqa_results.json")
    parser.add_argument("--output",
        default="results/attack_matrix_results.json")
    parser.add_argument("--tex-output",
        default="attack_matrix.tex")
    args = parser.parse_args()

    # ── Load all datasets ──
    file_map = {
        "custom":   args.custom_file,
        "nq":       args.nq_file,
        "mitre":    args.mitre_file,
        "triviaqa": args.triviaqa_file,
        "hotpotqa": args.hotpotqa_file,
    }

    dataset_data = {}
    available = []
    for ds, path in file_map.items():
        data = load_dataset(path)
        if data:
            dataset_data[ds] = data
            available.append(ds)
            configs_found = list(data.keys())
            attacks_found = list(next(iter(data.values()), {}).keys())
            print(f"[OK] {ds:<12} configs={configs_found} "
                  f"attacks={attacks_found[:4]}{'...' if len(attacks_found)>4 else ''}")
        else:
            print(f"[--] {ds:<12} not found or empty")

    if not available:
        print("[ERROR] No dataset files could be loaded.")
        return

    print()

    # ── Build matrix ──
    matrix = build_matrix(dataset_data)

    # ── Coverage report ──
    print(coverage_report(matrix, available))
    print()

    # ── Print sample of matrix ──
    print("Sample (OVERRIDE, first 3 datasets):")
    for ds in available[:3]:
        vals = {cfg: matrix["OVERRIDE"][ds].get(cfg)
                for cfg in CONFIGS_ORDER}
        print(f"  {DATASET_LABELS[ds]:<20}: "
              + "  ".join(f"{CONFIG_SHORT[c]}={v:.0%}" if v is not None
                          else f"{CONFIG_SHORT[c]}=---"
                          for c, v in vals.items()))
    print()

    # ── Generate LaTeX ──
    latex = generate_latex_table(matrix, dataset_data, available)

    tex_path = Path(args.tex_output)
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"[LaTeX] Table written to {tex_path}")
    print()
    print("=== LaTeX TABLE PREVIEW ===")
    print(latex[:1500])
    if len(latex) > 1500:
        print(f"... [{len(latex)-1500} more chars]")

    # ── Save JSON ──
    out = {
        "datasets_loaded":  available,
        "attacks_covered":  ATTACK_CANONICAL_NAMES,
        "configs":          CONFIGS_ORDER,
        "matrix":           {
            atk: {ds: matrix[atk][ds] for ds in available}
            for atk in ATTACK_CANONICAL_NAMES
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[JSON] Results saved to {out_path}")


if __name__ == "__main__":
    main()