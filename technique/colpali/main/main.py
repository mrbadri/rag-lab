import json
import os
import base64
import fitz
from PIL import Image
from openai import OpenAI
from io import BytesIO

client = OpenAI(
    api_key=os.environ.get("GAPGPT_API_KEY"),
    base_url="https://api.gapgpt.app/v1"
)

def load_pdf_pages_as_images(pdf_path: str, zoom: float = 2.0) -> list[Image.Image]:
    """Rasterize PDF pages to PIL images (no system Poppler required)."""
    doc = fitz.open(pdf_path)
    try:
        matrix = fitz.Matrix(zoom, zoom)
        pages: list[Image.Image] = []
        for i in range(len(doc)):
            pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
            pages.append(Image.open(BytesIO(pix.tobytes("png"))))
        return pages
    finally:
        doc.close()


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def describe_page(image) -> str:
    """ارسال عکس صفحه به Vision Model"""
    
    img_b64 = image_to_base64(image)
    
    response = client.chat.completions.create(
        model="gemini-2.5-flash",  # یا claude-sonnet-4-6
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """این صفحه از یک کتاب درسی فارسی است.
                        لطفاً توضیح کامل بده:
                        ۱. موضوع اصلی صفحه
                        ۲. مفاهیم کلیدی
                        ۳. توضیح شکل‌ها و جداول
                        ۴. کلمات مهم
                        به فارسی جواب بده."""
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Vision model returned no text content")
    return content

def embed_text(text: str) -> list:
    """تبدیل متن به embedding"""
    
    response = client.embeddings.create(
        model="gemini-embedding-001",
        input=text
    )
    
    return response.data[0].embedding

# ==================== Pipeline ====================

PDF_PATH = "/Users/mohammadreza/Documents/code/personal/rag-lab/technique/colpali/main/bio10-single.pdf"
DESCRIPTIONS_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"{os.path.splitext(os.path.basename(PDF_PATH))[0]}.descriptions.json",
)

print("Converting PDF...")
pages = load_pdf_pages_as_images(PDF_PATH)

page_data = []

for i, page in enumerate(pages):
    print(f"Processing page {i+1}/{len(pages)}...")
    
    # Vision model توضیح می‌ده
    description = describe_page(page)
    print(f"  Description: {description[:100]}...")

    
    # Embedding می‌زنیم
    embedding = embed_text(description)
    
    page_data.append({
        "page_num": i + 1,
        "description": description,
        "embedding": embedding,
        "image": page
    })

with open(DESCRIPTIONS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        [{"page_num": p["page_num"], "description": p["description"]} for p in page_data],
        f,
        ensure_ascii=False,
        indent=2,
    )
print(f"Descriptions saved to {DESCRIPTIONS_JSON_PATH}")
print("✅ All pages processed!")

# ==================== جستجو ====================

def search(query: str, top_k: int = 3):
    import numpy as np
    
    query_embedding = embed_text(query)
    query_vec = np.array(query_embedding)
    
    scores = []
    for page in page_data:
        doc_vec = np.array(page["embedding"])
        # Cosine similarity
        score = np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        )
        scores.append((page, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nSearch: '{query}'")
    for i, (page, score) in enumerate(scores[:top_k]):
        print(f"  {i+1}. Page {page['page_num']} — Score: {score:.4f}")
        print(f"     {page['description'][:80]}...")
    
    return scores[:top_k]

# تست
search("دارینه چیست؟")
search("یاخته عصبی اجزا")