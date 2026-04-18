# test gapgpt embedding model

import os
from custom import CustomGapGPTEmbeddingLangchain

# Constants
API_KEY =  os.environ.get("GAPGPT_API_KEY")
MODEL = "gemini-embedding-001"

# Initialize the embedding model
embedding = CustomGapGPTEmbeddingLangchain(api_key=API_KEY, model=MODEL)

# Test the embedding model
print(embedding.embed_query("Hello, world!"))