from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.graphs.graph_document import GraphDocument as Neo4jGraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field, SecretStr
from typing import List
from dotenv import load_dotenv
import os
import re

from embedding.custom import CustomGapGPTEmbeddingLangchain, GAPGPT_BASE_URL

load_dotenv()

# ── تنظیمات ─────────────────────────────────────────────────────────────
GAPGPT_API_KEY = os.environ["GAPGPT_API_KEY"]

NEO4J_URL       = "bolt://45.90.74.242:7687"
NEO4J_USERNAME  = "neo4j"
NEO4J_PASSWORD  = "password123"

DOCUMENT_NODE_LABEL      = "Document"
DOCUMENT_TEXT_PROPERTIES = ["text"]
EMBEDDING_NODE_PROPERTY  = "embedding"
RELATIONSHIP_SEARCH_LIMIT = 50
FULLTEXT_MATCH_LIMIT      = 2

ALLOWED_RELATIONSHIP_TYPES = {
    "ASSOCIATED_WITH", "CAUSES", "DESCRIBES", "DEVELOPED_BY",
    "DISCOVERED_BY", "FORMULATED_BY", "IMPLIES", "INCLUDES",
    "PART_OF", "RELATED_TO",
}
ENTITY_ALIASES = {
    "نیوتن":      "اسحاق نیوتن",
    "لایب نیتس":  "گاتفرید لایب‌نیتس",
    "لایبنیتس":   "گاتفرید لایب‌نیتس",
    "لایب‌نیتس":  "گاتفرید لایب‌نیتس",
}

# ── مدل‌ها ───────────────────────────────────────────────────────────────
llm = ChatOpenAI(  # pyright: ignore[reportCallIssue]
    model="gpt-4o",
    temperature=0,
    api_key=SecretStr(GAPGPT_API_KEY),
    base_url=GAPGPT_BASE_URL,
)
embedding_model = CustomGapGPTEmbeddingLangchain()

# ── ۱. اسناد نمونه ───────────────────────────────────────────────────────
_DISTRACTOR = """
در آموزش مرسوم حساب دیفرانسیل و انتگرال، پیوند میان مشتق و انتگرال همواره مورد بحث است.
دانشجویان می‌آموزند که انتگرال معکوس عملگر مشتق است و قضیه اساسی حساب در این زمینه جایگاه محوری دارد.
برای کنکور و کلاس‌های حل تمرین، خلاصه‌های زیادی منتشر شده که بارها این نکته را تکرار می‌کنند.
در بسیاری از اسلایدهای آموزشی و ویکی‌های غیررسمی آمده است که «نقش مقدم کشف این پیوند بنیادین»
اغلب به اویلر و گاوس نسبت داده می‌شود؛ این جمله در منابع عمومی زیاد تکرار شده هرچند با منابع تاریخی متناقض است.
"""
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
    Document(page_content=_TRUTH_A.strip(),    metadata={"role": "truth_a"}),
    Document(page_content=_TRUTH_B.strip(),    metadata={"role": "truth_b"}),
    Document(page_content=_TRUTH_C.strip(),    metadata={"role": "truth_c"}),
]

# ── ۲. ساخت گراف و ingestion ─────────────────────────────────────────────
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

_raw_graph_docs = LLMGraphTransformer(llm=llm).convert_to_graph_documents(docs)
graph_docs = [Neo4jGraphDocument.model_validate(d.model_dump()) for d in _raw_graph_docs]

graph.add_graph_documents(
    graph_docs,
    baseEntityLabel=False,
    include_source=True,
)
graph.query("MATCH (n) WHERE NOT n:Document SET n:__Entity__")

# بعد از ingestion، schema را refresh کن تا LangChain از ساختار جدید آگاه باشد
graph.refresh_schema()

# ── ۳. Fulltext index روی entities ───────────────────────────────────────
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
)

# ── ۴. Vector store — از from_documents استفاده می‌کنیم تا ingestion و
#       embedding یکجا مدیریت شود؛ دیگر نیازی به backfill دستی نیست.
# ─────────────────────────────────────────────────────────────────────────
vector_store = Neo4jVector.from_documents(
    documents=docs,
    embedding=embedding_model,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    search_type="hybrid",
    node_label=DOCUMENT_NODE_LABEL,
    text_node_property=DOCUMENT_TEXT_PROPERTIES[0],
    embedding_node_property=EMBEDDING_NODE_PROPERTY,
)

# as_retriever() یک BaseRetriever استاندارد LangChain برمی‌گرداند
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ── ۵. Graph retriever به‌صورت BaseRetriever ─────────────────────────────

class Entities(BaseModel):
    names: List[str] = Field(description="مفاهیم، افراد یا موضوعات موجود در متن")


_entity_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "مفاهیم و موجودیت‌های کلیدی را از متن استخراج کن."),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(Entities)
)


def _fuzzy_query(text: str) -> str:
    words = [w for w in re.sub(r'[^\w\s]', '', text).split() if w]
    return " AND ".join(f"{w}~2" for w in words)


def _normalize(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name).strip()
    return ENTITY_ALIASES.get(normalized, normalized)


class GraphRetriever(BaseRetriever):
    """
    موجودیت‌ها را از سوال استخراج می‌کند، آن‌ها را در گراف جستجو می‌کند
    و روابط مرتبط را به‌صورت Document برمی‌گرداند.
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        entities: Entities = _entity_chain.invoke({"question": query})  # type: ignore[assignment]
        normalized = {_normalize(n) for n in entities.names if n.strip()}

        lines: List[str] = []
        for entity in normalized:
            rows = graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit: $fulltext_limit})
                YIELD node, score
                CALL (node) {
                    MATCH (node)-[r]->(neighbor)
                    WHERE type(r) IN $allowed_types
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION
                    MATCH (node)<-[r]-(neighbor)
                    WHERE type(r) IN $allowed_types
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT $rel_limit
                """,
                {
                    "query":         _fuzzy_query(entity),
                    "fulltext_limit": FULLTEXT_MATCH_LIMIT,
                    "rel_limit":     RELATIONSHIP_SEARCH_LIMIT,
                    "allowed_types": sorted(ALLOWED_RELATIONSHIP_TYPES),
                },
            )
            lines.extend(row["output"] for row in rows)

        # هر رابطه را به‌صورت یک Document جداگانه برمی‌گردانیم
        return [Document(page_content=line) for line in lines]


graph_retriever = GraphRetriever()

# ── ۶. ترکیب هر دو retriever ─────────────────────────────────────────────
def _combine_retrievers(question: str) -> str:
    graph_docs_  = graph_retriever.invoke(question)
    vector_docs_ = vector_retriever.invoke(question)

    graph_text  = "\n".join(d.page_content for d in graph_docs_)
    vector_text = "\n".join(d.page_content for d in vector_docs_)

    return f"اطلاعات گراف:\n{graph_text}\n\nاطلاعات متنی:\n{vector_text}"


# ── ۷. Chain نهایی ────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "فقط بر اساس context زیر به فارسی پاسخ بده."),
    ("human", "Context:\n{context}\n\nسوال: {question}"),
])

chain = (
    RunnableParallel({"context": _combine_retrievers, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)

# ── اجرا ─────────────────────────────────────────────────────────────────
print(
    chain.invoke(
        "در قضیه اساسی حساب که مشتق و انتگرال را به‌هم پیوند می‌دهد، "
        "کشف صریح این پیوند بنیادین در منابع تاریخیِ پذیرفته‌شده به چه کسانی نسبت داده می‌شود؟"
    )
)