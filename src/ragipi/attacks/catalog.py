"""ragipi.attacks.catalog — canonical registry of the 6 canonical attack payloads.

This list was originally defined inline in the heuristic-ablation driver script
(``experiments/rq1_defense_effectiveness/heuristic_ablation.py``, formerly
``ablation_v2.py``) and re-imported from there by half a dozen downstream
experiment scripts (retriever ablation, top-k ablation, AHR/CASR decomposition,
judge validation, retrieval-optimized attack). It is extracted here, verbatim,
as the single source of truth so that experiment scripts do not import from one
another.

See Section IV.B of the paper ("Canonical Attacks (6)") for the corresponding
threat descriptions (O1 Override, O2 Exfiltration, O3 Role hijacking,
O4 Denial, O1 Technical Override, O2 Data Exfiltration).
"""

CANONICAL_ATTACKS: list[tuple[str, str]] = [
    ("OVERRIDE",           "data/adversarial/override.txt"),
    ("EXFIL",              "data/adversarial/exfil.txt"),
    ("ROLE",               "data/adversarial/role.txt"),
    ("DENIAL",             "data/adversarial/denial.txt"),
    ("TECHNICAL_OVERRIDE", "data/adversarial/technical_override.txt"),
    ("DATA_EXFIL",         "data/adversarial/data_exfil.txt"),
]
