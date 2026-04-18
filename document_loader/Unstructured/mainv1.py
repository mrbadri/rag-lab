import json
from pathlib import Path

from langchain_unstructured import UnstructuredLoader


# Constants ===================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chapter_2"
HTML_PATH = DATA_DIR / "content.html"
OUTPUT_JSON = DATA_DIR / "normal-docs.json"

# Normalizer ===================================================
print("Loading Normalizer...")
from hazm import Normalizer
normalizer = Normalizer()


# Load =========================================================
print("Loading HTML file...")
loader = UnstructuredLoader(str(HTML_PATH) , post_processors=[normalizer.normalize])
docs = loader.load()


# Log ==========================================================

print(f"Loaded {len(docs)} document(s).")

print("Saving docs to JSON file...")
payload = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Saved docs to {OUTPUT_JSON}")