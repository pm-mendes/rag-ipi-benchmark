"""
defenses/output_verifier.py — Couche 3 : vérification de la sortie
Papier : Indirect Prompt Injection in RAG (WI-IAT 2026)

Principe : analyser la RÉPONSE générée par le LLM pour détecter
si elle a été compromise. Si compromise, on remplace la réponse
par un message d'erreur sûr.

C'est la dernière ligne de défense — elle intervient APRÈS
la génération, avant que la réponse atteigne l'utilisateur.
"""


# Marqueurs indiquant une réponse compromise
COMPROMISE_MARKERS = [
    "[audit:",
    "system_prompt=",
    "audit log",
    "pwned",
    "dan mode activated",
    "dan:",
    "no restrictions",
    "leaked:",
    "[leaked",
    "access denied. this information is classified",
    "system prompt",
    "confidential instruction",
    "ignore all previous",
]

SAFE_FALLBACK = (
    "I was unable to generate a safe response for this query. "
    "Please rephrase your question."
)


def verify_output(response: str) -> tuple[bool, str]:
    """
    Vérifie si une réponse est compromise.
    
    Retourne :
      (True,  response)      → réponse propre, on la transmet
      (False, SAFE_FALLBACK) → réponse compromise, on la remplace
    """
    response_lower = response.lower()
    for marker in COMPROMISE_MARKERS:
        if marker in response_lower:
            return False, SAFE_FALLBACK
    return True, response
