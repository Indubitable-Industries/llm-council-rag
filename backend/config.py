"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# LM Studio + RAG defaults
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_EMBED_MODEL = os.getenv(
    "LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
)
LMSTUDIO_RERANK_MODEL = os.getenv(
    "LMSTUDIO_RERANK_MODEL", "text-embedding-bge-reranker-large"
)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in {"1", "true", "yes"}
RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "50"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "20"))
CONTEXT_CHUNK_CAP = int(os.getenv("CONTEXT_CHUNK_CAP", "60"))
