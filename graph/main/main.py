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

docs = [Document(page_content="انتگرال معکوس مشتق است. قضیه اساسی حساب رابطه بین آن‌ها را بیان می‌کند. نیوتن و لایب‌نیتس آن را کشف کردند.")]

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

print(chain.invoke("رابطه انتگرال و مشتق چیست؟"))