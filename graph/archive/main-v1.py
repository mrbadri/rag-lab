from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType
from langchain_neo4j.graphs.graph_document import GraphDocument as Neo4jGraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, SecretStr
from typing import List, cast
from dotenv import load_dotenv
import os
import re

from embedding.custom import CustomGapGPTEmbeddingLangchain, GAPGPT_BASE_URL

load_dotenv()

GAPGPT_API_KEY = os.environ["GAPGPT_API_KEY"]

llm = ChatOpenAI(  # pyright: ignore[reportCallIssue]
    model="gpt-4o",
    temperature=0,
    api_key=SecretStr(GAPGPT_API_KEY),
    base_url=GAPGPT_BASE_URL,
)

# http://45.90.74.242:7474/browser/
NEO4J_URL      = "bolt://45.90.74.242:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password123"
DOCUMENT_NODE_LABEL = "Document"
DOCUMENT_TEXT_PROPERTIES = ["text"]
EMBEDDING_NODE_PROPERTY = "embedding"
RELATIONSHIP_SEARCH_LIMIT = 50
FULLTEXT_MATCH_LIMIT = 2
ALLOWED_RELATIONSHIP_TYPES = {
    "ASSOCIATED_WITH",
    "CAUSES",
    "DESCRIBES",
    "DEVELOPED_BY",
    "DISCOVERED_BY",
    "FORMULATED_BY",
    "IMPLIES",
    "INCLUDES",
    "PART_OF",
    "RELATED_TO",
}
ENTITY_ALIASES = {
    "نیوتن": "اسحاق نیوتن",
    "لایب نیتس": "گاتفرید لایب‌نیتس",
    "لایبنیتس": "گاتفرید لایب‌نیتس",
    "لایب‌نیتس": "گاتفرید لایب‌نیتس",
}

# ── ۱. ساخت گراف ────────────────────────────────────────────────────────
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# سناریو: RAG ساده (فقط شباهت برداری) معمولاً تکهٔ «فریب» را بالا می‌آورد؛
# Graph RAG از یال‌ها و موجودیت‌ها همان جواب درست را پیدا می‌کند.
#
# سند الف — فریب: طولانی و پر از کلمات «مشتق/انتگرال/قضیه اساسی» اما نتیجهٔ غلط
# (بسیاری از امبدینگ‌ها این را به سوال نزدیک‌تر می‌گیرند).
_DISTRACTOR = """
در آموزش مرسوم حساب دیفرانسیل و انتگرال، پیوند میان مشتق و انتگرال همواره مورد بحث است.
دانشجویان می‌آموزند که انتگرال معکوس عملگر مشتق است و قضیه اساسی حساب در این زمینه جایگاه محوری دارد.
برای کنکور و کلاس‌های حل تمرین، خلاصه‌های زیادی منتشر شده که بارها این نکته را تکرار می‌کنند.
در بسیاری از اسلایدهای آموزشی و ویکی‌های غیررسمی آمده است که «نقش مقدم کشف این پیوند بنیادین»
اغلب به اویلر و گاوس نسبت داده می‌شود؛ این جمله در منابع عمومی زیاد تکرار شده هرچند با منابع تاریخی متناقض است.
"""
# حقیقت پراکنده: هر سند به‌تنهایی برای «کی کشف کرد؟» کافی نیست؛ گراف موجودیت‌ها را به هم وصل می‌کند.
_TRUTH_A = (
    "قضیه اساسی حساب بیانگر ارتباط میان مشتق و انتگرال است؛ "
    "اما نام کسانی که این پیوند را صریح و بنیادی صورت‌بندی کردند در این متن نیامده است."
)
_TRUTH_B = (
    "کتاب درسی می‌گوید دو فرد مستقل به‌نام «اسحاق نیوتن» و «گاتفرید لایب‌نیتس» در سدهٔ ۱۷ "
    "پایه‌های نوین حساب دیفرانسیل و انتگرال را توسعه دادند؛ "
    "جملهٔ بعدی فقط تأکید می‌کند این دو در منازعهٔ اولویت کشف هم شرکت داشتند."
)
_TRUTH_C = (
    "در فهرست کشفیات مرتبط با قضیه اساسی حساب، پژوهشگرانِ مورخ رایجاً "
    "«قضیه اساسی حساب» را به‌عنوان پل بین مشتق و انتگرال ذکر می‌کنند "
    "و پیوند رسمی کشف آن را به همان دو نویسندهٔ سدهٔ ۱۷ می‌آورند."
)

docs = [
    Document(page_content=_DISTRACTOR.strip(), metadata={"role": "distractor"}),
    Document(page_content=_TRUTH_A.strip(), metadata={"role": "truth_a"}),
    Document(page_content=_TRUTH_B.strip(), metadata={"role": "truth_b"}),
    Document(page_content=_TRUTH_C.strip(), metadata={"role": "truth_c"}),
]

_raw_graph_docs = LLMGraphTransformer(llm=llm).convert_to_graph_documents(docs)
graph_docs = [Neo4jGraphDocument.model_validate(d.model_dump()) for d in _raw_graph_docs]

graph.add_graph_documents(
    graph_docs,
    baseEntityLabel=False,
    include_source=True,
)
graph.query("MATCH (n) WHERE NOT n:Document SET n:__Entity__")

# ── ۲. Fulltext index روی entities (یک‌بار اجرا میشه) ────────────────────
graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# ── ۳. Unstructured retriever (vector + keyword) ─────────────────────────
embedding_model = CustomGapGPTEmbeddingLangchain()


def _backfill_document_embeddings(batch_size: int = 64) -> None:
    while True:
        rows = graph.query(
            """
            MATCH (d:Document)
            WHERE d.text IS NOT NULL AND d.embedding IS NULL
            RETURN elementId(d) AS element_id, d.text AS text
            LIMIT $batch_size
            """,
            {"batch_size": batch_size},
        )
        if not rows:
            break
        embeddings = embedding_model.embed_documents([row["text"] for row in rows])
        payload = [
            {"element_id": row["element_id"], "embedding": embedding}
            for row, embedding in zip(rows, embeddings)
        ]
        graph.query(
            """
            UNWIND $rows AS row
            MATCH (d) WHERE elementId(d) = row.element_id
            SET d.embedding = row.embedding
            """,
            {"rows": payload},
        )


_backfill_document_embeddings()

vector_index = Neo4jVector.from_existing_graph(
    embedding=embedding_model,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    search_type=SearchType.HYBRID,
    node_label=DOCUMENT_NODE_LABEL,
    text_node_properties=DOCUMENT_TEXT_PROPERTIES,
    embedding_node_property=EMBEDDING_NODE_PROPERTY,
)

# ── ۴. Graph retriever (entity extraction + traversal) ───────────────────
class Entities(BaseModel):
    names: List[str] = Field(description="مفاهیم، افراد یا موضوعات موجود در متن")

entity_chain = ChatPromptTemplate.from_messages([
    ("system", "مفاهیم و موجودیت‌های کلیدی را از متن استخراج کن."),
    ("human", "{question}"),
]) | llm.with_structured_output(Entities)


def _fuzzy_query(text: str) -> str:
    words = [w for w in re.sub(r'[^\w\s]', '', text).split() if w]
    return " AND ".join(f"{w}~2" for w in words)


def _normalize_entity_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name).strip()
    return ENTITY_ALIASES.get(normalized, normalized)


def structured_retriever(question: str) -> str:
    entities: Entities = cast(Entities, entity_chain.invoke({"question": question}))
    results = []
    normalized_entities = {
        _normalize_entity_name(name) for name in entities.names if name.strip()
    }
    for entity in normalized_entities:
        rows = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit: $fulltext_limit})
            YIELD node, score
            CALL (node) {
                MATCH (node)-[r]->(neighbor)
                WHERE type(r) IN $allowed_relationship_types
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION
                MATCH (node)<-[r]-(neighbor)
                WHERE type(r) IN $allowed_relationship_types
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT $relationship_limit
            """,
            {
                "query": _fuzzy_query(entity),
                "fulltext_limit": FULLTEXT_MATCH_LIMIT,
                "relationship_limit": RELATIONSHIP_SEARCH_LIMIT,
                "allowed_relationship_types": sorted(ALLOWED_RELATIONSHIP_TYPES),
            },
        )
        results.extend(row["output"] for row in rows)
    return "\n".join(results)


# ── ۵. ترکیب هر دو retriever ─────────────────────────────────────────────
def retriever(question: str) -> str:
    graph_data  = structured_retriever(question)
    vector_data = "\n".join(d.page_content for d in vector_index.similarity_search(question))
    return f"اطلاعات گراف:\n{graph_data}\n\nاطلاعات متنی:\n{vector_data}"


# ── ۶. Chain نهایی ────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "فقط بر اساس context زیر به فارسی پاسخ بده."),
    ("human", "Context:\n{context}\n\nسوال: {question}"),
])

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

# سوال عمداً «چه کسی قضیهٔ پل مشتق–انتگرال را صریح کشف کرد؟» است؛
# بردار فقط با سند فریب می‌تواند «اویلر و گاوس» را توجیه کند؛
# گراف موجودیت‌ها (نیوتن، لایب‌نیتس، قضیه اساسی حساب) مسیر درست را نگه می‌دارد.
print(
    chain.invoke(
        "در قضیه اساسی حساب که مشتق و انتگرال را به‌هم پیوند می‌دهد، "
        "کشف صریح این پیوند بنیادین در منابع تاریخیِ پذیرفته‌شده به چه کسانی نسبت داده می‌شود؟"
    )
)