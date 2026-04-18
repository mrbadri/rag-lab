import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Constants
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chapter_2"
INPUT_JSON = DATA_DIR / "normal-docs.json"
OUTPUT_JSON = DATA_DIR / "split-docs.json"

# Load docs from previous step
print("Loading normalized docs...")
with INPUT_JSON.open("r", encoding="utf-8") as f:
    payload = json.load(f)

docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in payload]
print(f"Loaded {len(docs)} document(s).")

# Split
print("Splitting docs...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # تعداد کاراکتر هر chunk
    chunk_overlap=50,     # overlap بین chunk‌ها برای حفظ context
    separators=["\n\n", "\n", ".", "،", "؟", "!", " ", ""],  # جداکننده‌های فارسی هم اضافه شد
)

split_docs = splitter.split_documents(docs)
print(f"Split into {len(split_docs)} chunk(s).")

# Save
print("Saving split docs...")
output = [{"page_content": d.page_content, "metadata": d.metadata} for d in split_docs]
with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved to {OUTPUT_JSON}")