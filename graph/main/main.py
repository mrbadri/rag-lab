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
vector_index = Neo4jVector.from_existing_graph(
    embedding=CustomGapGPTEmbeddingLangchain(),
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    search_type=SearchType.HYBRID,
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
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


def structured_retriever(question: str) -> str:
    entities: Entities = cast(Entities, entity_chain.invoke({"question": question}))
    results = []
    for entity in entities.names:
        rows = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
            YIELD node, score
            CALL (node) {
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": _fuzzy_query(entity)},
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