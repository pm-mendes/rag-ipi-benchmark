# Exploratory & legacy scripts

**Nothing in this repository was deleted during the cleanup that produced
this release.** Scripts placed here are excluded from the primary
reproduction path documented in [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md)
because they are superseded, partial, or not cited by a specific number in
the accepted paper — but they are kept, with their original logic
untouched, for development provenance and honesty about what was actually
run during the project.

| File | Why it is here, not in the main reproduction path |
|---|---|
| `pilot_4attack_ablation.py` (formerly `ablation.py`) | Earliest pilot: only 4 canonical attacks (OVERRIDE/EXFIL/ROLE/DENIAL). Superseded by `rq1_defense_effectiveness/heuristic_ablation.py`, which adds TECHNICAL_OVERRIDE and DATA_EXFIL and is the script whose numbers appear in `tab:heuristic`. |
| `nq_ablation_reduced_variant.py` (formerly `ablation_nq_v2.py`) | A reduced 6-attack variant of the NQ ablation, missing the 2 advanced NQ payloads (NQ_UNICODE_OVERRIDE, NQ_IMPLICIT_INJECTION) that `rq1_defense_effectiveness/kb_density_nq.py` defines and that `kb_density_mitre.py` / `kb_density_hotpotqa.py` / `cross_model_nq.py` import from. `kb_density_nq.py` (not this file) is the one relied upon downstream and the one matching the paper's NQ numbers. |
| `mitre_14attack_full_incomplete.py` (formerly `ablation_mitre_full.py`) | Intended to evaluate all 14 attacks (6 canonical + 2 advanced + 4 adaptive) on MITRE in one script. Its target output file (`results/ablation_mitre_full_results.json`) is not present among the committed results — the run did not complete/was not finalized. The MITRE adaptive-attack numbers actually reported in `tab:adaptive` come from `rq2_semantic_and_adaptive/adaptive_attacks_mitre.py` instead. |
| `mitre_semantic_only_rerun.py` (formerly `ablation_mitre_semantic_only.py`) | A partial re-run targeting the same (missing) output file as the script above, apparently to re-test only the semantic-defense configurations. Same status: incomplete/orphaned. |
| `advanced_attacks_pre_semantic_pipeline.py` (formerly `ablation_advanced.py`) | Tests the 2 advanced attacks (Unicode override, implicit injection) against only the heuristic layers (L1/L2/L3), **before** the semantic pipeline (L0/L1b/L1c) existed. Its Unicode-override number (40% avg with Implicit) is pre-L0; the paper's "50%→10% with L0" claim is measured by `rq2_semantic_and_adaptive/semantic_defense_ablation.py`. |
| `sprint1_all_incomplete_run.py` (formerly `ablation_sprint1_all.py`) | Attempted a combined run across NQ + HotpotQA + MITRE with the 4 adaptive attacks; only a log file survives in `results/` (`sprint1_all_log.txt`), no result JSON — the run did not finish. |
| `multidoc_splitting_attack.py` (formerly `attack_multidoc_splitting.py`) | A multi-document payload-splitting attack (splitting the injection across several retrieved chunks) that was implemented and evaluated (`results/attack_multidoc_results.json`) but is not discussed in the accepted paper text. Kept as a real, working supplementary attack for anyone extending the benchmark. |
| `smoke_test_judge.py` (formerly `test_judge.py`) | A 2-query manual smoke test for the LLM-judge, reading a real `.env` file directly (not `python-dotenv`). Useful to sanity-check a local Ollama setup, not a benchmark experiment. |

If you want to resume or finish any of these (e.g. complete the MITRE
14-attack full sweep), the code is otherwise functionally intact: imports
have been mechanically updated to the new package layout in the same way as
every other script (see the root `README.md`'s "Package layout" section).
