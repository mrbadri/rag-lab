import os
import sys
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from embedding import CustomGapGPTEmbeddingLangchain

embeddings = CustomGapGPTEmbeddingLangchain(
    api_key=os.environ.get("GAPGPT_API_KEY"),
)

client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://45.90.74.242:6333"))

vector_size = len(embeddings.embed_query("sample text"))

collection_name = "kb_textbook_semantic_langchain_chapter_2"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)



# print("Search")

# results = vector_store.similarity_search_with_score("مقدمه", k=4)

# for doc, score in results:
#     print("Score:", score)
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)
#     print("------")

# print(results)
