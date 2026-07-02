from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# LLM provider selector: "ollama" (default), "groq", "gemini", or "deepseek".
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# Groq-only settings (read when LLM_PROVIDER="groq").
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_REQUEST_TIMEOUT: float = float(os.getenv("GROQ_REQUEST_TIMEOUT", "60"))
# Groq rate-limit budgets (free-tier defaults; override via env vars for paid plans).
GROQ_RPM: int = int(os.getenv("GROQ_RPM", "30"))       # requests per minute
GROQ_TPM: int = int(os.getenv("GROQ_TPM", "12000"))    # tokens per minute

# DeepSeek settings (read when LLM_PROVIDER="deepseek").
# deepseek-chat (fast/cheap), deepseek-reasoner (R1, slower, higher accuracy)
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_REQUEST_TIMEOUT: float = float(os.getenv("DEEPSEEK_REQUEST_TIMEOUT", "120"))

# Google Gemini settings (read when LLM_PROVIDER="gemini").
# Uses the official google-genai SDK (https://ai.google.dev/gemini-api/docs/libraries).
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_REQUEST_TIMEOUT: float = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "120"))
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "earnings_db")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "earnings")

# Redis queue settings (used by the 8-K worker).
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_QUEUE_NAME: str = os.getenv("REDIS_QUEUE_NAME", "sec:filings")

# Seconds before HTTP requests time out
HTTP_TIMEOUT: int = 30

EXTRACTION_MAX_CHARS: int = int(os.getenv("EXTRACTION_MAX_CHARS", "400000"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "300"))

# Maximum concurrent Ollama requests across all parallel company workers.
# Local Ollama is single-threaded, so >1 here only helps when using a
# remote / multi-GPU Ollama instance. Default: 1 (serialize LLM calls).
OLLAMA_CONCURRENCY: int = int(os.getenv("OLLAMA_CONCURRENCY", "1"))

# When True (default), refuse to upsert a document whose accounting identity
# checks failed (e.g. Gross margin ≠ Revenue − COGS). When False, the document
# is saved with an "identity_warnings" field listing the failures.
STRICT_ACCURACY: bool = os.getenv("STRICT_ACCURACY", "1").strip().lower() not in {
    "0", "false", "no", "off", ""
}

# Dev LLM response cache — opt-in, never enabled in production.
# Set LLM_CACHE=1 (or true/yes) in .env to cache LLM responses to disk.
# Responses are keyed by sha256(provider:model + prompt) and persist across
# runs until the cache directory is deleted manually.
LLM_CACHE_ENABLED: bool = os.getenv("LLM_CACHE", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
LLM_CACHE_DIR: str = os.getenv("LLM_CACHE_DIR", ".llm_cache")

# When True (default), run an additional LLM cleanup pass over the extracted
# metrics before saving. The cleanup is constrained: it can ONLY drop keys
# (duplicates, obvious scale errors). It cannot invent or mutate values —
# any such attempt is rejected by deterministic guardrails.
CLEANUP_METRICS: bool = os.getenv("CLEANUP_METRICS", "1").strip().lower() not in {
    "0", "false", "no", "off", ""
}

# Maximum extraction passes in the agentic loop (initial pass + retries).
# Override with the MAX_EXTRACTION_ATTEMPTS environment variable.
MAX_EXTRACTION_ATTEMPTS: int = int(os.getenv("MAX_EXTRACTION_ATTEMPTS", "3"))
