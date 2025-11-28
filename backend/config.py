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
CHAIRMAN_MODEL = "openai/gpt-5.1"

# Model context limits (input tokens) for budgeting; tweak via env overrides
MODEL_CONTEXT_LIMITS = {
    "openai/gpt-5.1": int(os.getenv("CTX_LIMIT_GPT_5_1", "400000")),
    "google/gemini-3-pro-preview": int(os.getenv("CTX_LIMIT_GEMINI_3_PRO_PREVIEW", "1048576")),
    "anthropic/claude-sonnet-4.5": int(os.getenv("CTX_LIMIT_CLAUDE_SONNET_4_5", "1000000")),
    "x-ai/grok-4": int(os.getenv("CTX_LIMIT_GROK_4", "256000")),
}

# Safety margin + output allowance (tokens) for budgeting
CONTEXT_SAFETY_MARGIN = float(os.getenv("CONTEXT_SAFETY_MARGIN", "0.85"))
OUTPUT_TOKEN_ALLOWANCE = int(os.getenv("OUTPUT_TOKEN_ALLOWANCE", "4000"))

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

# Git-based indexing settings
INDEX_INCLUDE_GLOBS = (
    os.getenv("INDEX_INCLUDE_GLOBS", "").split(",") if os.getenv("INDEX_INCLUDE_GLOBS") else []
)
INDEX_EXCLUDE_GLOBS = (
    os.getenv("INDEX_EXCLUDE_GLOBS", "").split(",") if os.getenv("INDEX_EXCLUDE_GLOBS") else []
)
INDEX_INCLUDE_UNTRACKED = os.getenv("INDEX_INCLUDE_UNTRACKED", "true").lower() in {"1", "true", "yes"}
INDEX_MANIFEST_PATH = os.getenv("INDEX_MANIFEST_PATH", "data/index_manifest.json")
