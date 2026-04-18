import json
from pathlib import Path

from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title


# Constants ===================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chapter_2"
HTML_PATH = DATA_DIR / "content.html"
OUTPUT_JSON = DATA_DIR / "normal-docs-unstructured.json"

# Normalizer ===================================================
print("Loading Normalizer...")
from hazm import Normalizer
normalizer = Normalizer()


# مرحله 1: partition =========================================
print("Partitioning HTML file...")
elements = partition_html(
    filename=str(HTML_PATH),
    languages=["fas"],
)

print(f"Extracted {len(elements)} element(s).")


# مرحله 2: chunking هوشمند ⭐ ================================
print("Chunking elements...")
chunks = chunk_by_title(
    elements,
    combine_text_under_n_chars=300,   # عناوین و پاراگراف‌های کوچیک رو ادغام کن
    max_characters=2000,              # حداکثر سایز
    new_after_n_chars=1500,           # بعد از 1500 کاراکتر chunk جدید بساز
    overlap=200,                      # 200 کاراکتر overlap
)

print(f"Created {len(chunks)} chunk(s).")


# Normalize text ==============================================
print("Normalizing text...")
normalized_chunks = []
for chunk in chunks:
    normalized_text = normalizer.normalize(chunk.text)
    normalized_chunks.append({
        "page_content": normalized_text,
        "metadata": chunk.metadata.to_dict()
    })


# Save ========================================================
print("Saving chunks to JSON file...")
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(normalized_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved {len(normalized_chunks)} chunk(s) to {OUTPUT_JSON}")