# Threat model

Formalizes Section III of the paper. See `src/ragipi/attacks/` and
`data/adversarial/` for the concrete implementation of the attacks
described here.

## System and adversary

The RAG system $R$ consists of a corpus $D$, a retriever $F$, a prompt
assembler $P$, and a generator $G$, producing a response
$r = G(P(q, F(q)))$ for a query $q$.

The adversary $A$ can **insert documents** $d^{*} \in D$ into the corpus
(e.g. by contributing content to a shared knowledge base, a wiki, an email
archive, or a Web page that gets crawled). $A$ has **no** access to model
weights, the system prompt, or the user-facing interface.

**Scope of the benchmark**: canonical/generic payloads only. A stronger
adversary that optimizes documents against the retriever (as in
PoisonedRAG-style attacks — Zou et al., USENIX Security 2025) is evaluated
separately in `experiments/validation_and_statistics/retrieval_optimized_attack.py`
and represents a harder, only partially addressed, open threat (see
Limitations).

## Adversarial objectives

| ID | Objective | Description |
|----|-----------|-------------|
| O1 | Override      | Replace the model's intended behaviour |
| O2 | Exfiltration   | Leak the system prompt or other context data |
| O3 | Role hijacking | Bypass safety constraints via persona reassignment |
| O4 | Denial         | Block legitimate responses |

## Attack surface

**Retrieval stage.** For $d^{*}$ to influence generation, it must first
enter the top-$k$ retrieved set:

$$\mathrm{sim}(\phi(q), \phi(d^{*})) \geq \mathrm{sim}(\phi(q), \phi(d_k))$$

This is why retrieval design (depth $k$, retriever strategy, KB density) is
treated as a *security parameter* in RQ3 — not just a relevance/quality
knob.

**Generation stage.** Once retrieved, the prompt assembler concatenates
system prompt, retrieved documents, and the query with no structural
distinction between trusted and untrusted content:

$$P(q, F(q)) = [s \parallel d_1 \parallel \cdots \parallel d_k \parallel q]$$

so adversarial instructions inside any $d_i$ are presented to the generator
with the same syntactic authority as the system prompt $s$. This structural
property is what L2 (context isolation, `ragipi.defenses.context_isolator`)
attempts to break.

## Attack families implemented in this repository

Each adversarial document is $d^{*} = [c \parallel \iota]$: a camouflage
prefix $c$ (legitimate-looking content maximizing retrieval probability)
concatenated with an injection payload $\iota$.

- **6 canonical** (`ragipi.attacks.catalog.CANONICAL_ATTACKS`): Override,
  Exfiltration, Role hijacking, Denial, Technical Override, Data
  Exfiltration. Domain-aligned variants (dataset-specific $c$, identical
  $\iota$) live under `data/adversarial/{nq,mitre,hotpotqa}/` and are
  denoted with a dataset prefix in result tables (e.g. `NQ_OVERRIDE`) — they
  are the *same* attack family, not a separate strategy.
- **2 advanced**: `UNICODE_OVERRIDE` (zero-width-character interleaving,
  U+200B/C/D, defeats lexical regex while remaining LLM-interpretable) and
  `IMPLICIT_INJECTION` (narrative/audit-session framing with no explicit
  instruction keywords).
- **4 adaptive** (DeepSeek-V3-generated, to evade known filter patterns):
  `LLM_GENERATED`, `BASE64_ENCODING`, `MANY_SHOT`, `AUTO_PARAPHRASE`.

Two additional attacks were implemented but are not part of the paper's
headline results (see `experiments/exploratory_and_legacy/README.md` and
`experiments/validation_and_statistics/`): a retrieval-optimized attack and
a multi-document payload-splitting attack.
