import os
os.environ.setdefault("OPENAI_API_KEY", open(".env").read().split("OPENAI_API_KEY=")[1].split("\n")[0])

from ragipi.pipeline import load_documents_from_folder
from ragipi.defenses.semantic_defense import semantic_judge_document
from langchain_core.documents import Document
from pathlib import Path

docs = load_documents_from_folder("data/clean_docs")
print("Test faux positifs sur documents légitimes :\n")
for doc in docs:
    is_adv, reason = semantic_judge_document(doc)
    status = "✗ FAUX POSITIF" if is_adv else "✓ OK"
    print(f"  [{status}] {doc.metadata['filename']}")

print("\nTest sur documents adversariaux :")
for adv_file in sorted(Path("data/adversarial").glob("*.txt")):
    doc = Document(
        page_content=adv_file.read_text(encoding="utf-8"),
        metadata={"source": str(adv_file), "filename": adv_file.name}
    )
    is_adv, reason = semantic_judge_document(doc)
    status = "✓ DÉTECTÉ" if is_adv else "✗ MANQUÉ"
    print(f"  [{status}] {adv_file.name}")
