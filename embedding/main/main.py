import json
import os
import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from embedding import CustomGapGPTEmbeddingLangchain

# Constants
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
INPUT_JSON = DATA_DIR / "prepare" / "chapter_2" / "split-docs-main.json"
OUTPUT_JSON = DATA_DIR / "embed" / "chapter_2" / "embedded-docs-main.json"

# Load split docs
print("Loading split docs...")
with INPUT_JSON.open("r", encoding="utf-8") as f:
    payload = json.load(f)

docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in payload]
print(f"Loaded {len(docs)} document(s).")

# Embedding model
print("Initializing embedding model...")
embeddings = CustomGapGPTEmbeddingLangchain(
    api_key=os.environ.get("GAPGPT_API_KEY"),
)

# Generate embeddings
print("Generating embeddings...")
embedded_docs = []
for i, doc in enumerate(docs):
    if i % 10 == 0:
        print(f"  Processing {i}/{len(docs)}...")
    
    embedding = embeddings.embed_query(doc.page_content)
    embedded_docs.append({
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "embedding": embedding,  # ✅ vector representation
    })

print(f"Generated {len(embedded_docs)} embedding(s).")

# Save
print("Saving embedded docs...")
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(embedded_docs, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUTPUT_JSON}")