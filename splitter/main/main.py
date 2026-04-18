import json
import os
import sys
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from embedding import CustomGapGPTEmbeddingLangchain

# Constants
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chapter_2"
INPUT_JSON = DATA_DIR / "doc-loader-main.json"
OUTPUT_JSON = DATA_DIR / "split-docs-main.json"

# Load docs from previous step
print("Loading normalized docs...")
with INPUT_JSON.open("r", encoding="utf-8") as f:
    payload = json.load(f)

docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in payload]
print(f"Loaded {len(docs)} document(s).")

# Embedding model
embeddings = CustomGapGPTEmbeddingLangchain(
    api_key=os.environ.get("GAPGPT_API_KEY"),
)

# Semantic Splitter
print("Splitting docs semantically...")
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # یا "standard_deviation" / "interquartile"
    breakpoint_threshold_amount=50,   
)

split_docs = splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunk(s).")

# Save
print("Saving split docs...")
output = [{"page_content": d.page_content, "metadata": d.metadata} for d in split_docs]
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUTPUT_JSON}")