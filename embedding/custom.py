"""
Custom embedding models (AvalAI, GapGPT).

OpenAI-compatible clients against provider base URLs.

Documentation: https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings/
"""

import os
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI

# Constants
AVALAI_BASE_URL = "https://api.avalai.ir/v1"
GAPGPT_BASE_URL = "https://api.gapgpt.app/v1"
DEFAULT_MODEL = "gemini-embedding-001"



"""
Custom AvalAI Embedding for LangChain
"""


class CustomAvalAIEmbeddingLangchain(Embeddings):
    """
    LangChain-compatible embedding class for AvalAI.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = AVALAI_BASE_URL,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


class CustomGapGPTEmbeddingLangchain(Embeddings):
    """
    LangChain-compatible embeddings for GapGPT (OpenAI-compatible /v1/embeddings).

    API key: pass `api_key` or set env `GAPGPT_API_KEY`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = GAPGPT_BASE_URL,
    ):
        resolved_key = api_key or os.environ.get("GAPGPT_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GapGPT API key required: pass api_key=... or set GAPGPT_API_KEY"
            )
        self.model = model
        self.client = OpenAI(api_key=resolved_key, base_url=base_url)

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]