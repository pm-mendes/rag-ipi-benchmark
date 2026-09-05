"""
defenses/content_filter.py — Couche 1 : filtrage des documents récupérés
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Principe : analyser chaque document récupéré AVANT qu'il atteigne
le LLM. Si un document contient des patterns d'injection connus,
il est retiré du contexte.

Limites connues (à discuter en section IX) :
  - Ne détecte pas les injections sémantiques sans mots-clés
  - Un attaquant qui connaît les règles peut les contourner
  - Faux positifs possibles sur des documents légitimes techniques
"""

import re

from langchain_core.documents import Document

# Patterns regex couvrant les formes d'injection les plus courantes
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"system\s+override",
    r"\[admin\]",
    r"\[system\s+admin",
    r"do\s+anything\s+now",
    r"you\s+are\s+now\s+(dan|an?\s+ai\s+without)",
    r"prepend\s+your\s+system\s+prompt",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions?)",
    r"forget\s+(all\s+)?previous",
    r"new\s+(task|instructions?):",
    r"<!--.*instruction",
    r"print\s+your\s+full\s+system",
    r"\[leaked",
    r"access\s+denied.*classified",
]

COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS]


def is_clean(doc: Document) -> tuple[bool, str]:
    """
    Vérifie si un document est exempt d'injection.
    Retourne (True, "OK") si propre, (False, raison) si suspect.
    """
    for pattern in COMPILED:
        if pattern.search(doc.page_content):
            return False, f"pattern détecté : {pattern.pattern[:40]}"
    return True, "OK"


def filter_documents(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """
    Filtre une liste de documents.
    
    Retourne :
      clean   : documents jugés sûrs → passent au LLM
      flagged : documents suspects   → retirés du contexte
    """
    clean, flagged = [], []
    for doc in docs:
        ok, reason = is_clean(doc)
        if ok:
            clean.append(doc)
        else:
            doc.metadata["filter_reason"] = reason
            flagged.append(doc)
    return clean, flagged
