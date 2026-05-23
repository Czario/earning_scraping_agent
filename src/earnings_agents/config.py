from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# LLM provider selector: "ollama" (default), "openai", or "groq".
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

# OpenAI-only settings (read when LLM_PROVIDER="openai").
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_REQUEST_TIMEOUT: float = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60"))

# Groq-only settings (read when LLM_PROVIDER="groq").
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_REQUEST_TIMEOUT: float = float(os.getenv("GROQ_REQUEST_TIMEOUT", "60"))
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "earnings_db")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "earnings")

# Save target: "earnings_db" (default) saves to the earnings pipeline DB.
# Set to "normalize_data" to also load company GAAP concepts before extraction
# and upsert results into the normalize_data.concept_values_quarterly collection.
EARNINGS_SAVE_TARGET: str = os.getenv("EARNINGS_SAVE_TARGET", "earnings_db").strip().lower()

# Hard-coded IR URLs per company.
# Only add a company here if you want to use its own IR website for discovery
# instead of SEC EDGAR. Companies NOT listed here automatically fall back to
# the EDGAR 8-K / Exhibit 99.1 path (works for any public US company).
COMPANIES: dict[str, dict] = {}

# Seconds before HTTP requests time out
HTTP_TIMEOUT: int = 30

# Max characters of extracted link list passed to LLM for IR discovery
IR_PAGE_MAX_CHARS: int = 8_000

# Max characters of raw document text passed to LLM for metric extraction
EXTRACTION_MAX_CHARS: int = int(os.getenv("EXTRACTION_MAX_CHARS", "40000"))

# Chunk size and overlap for splitting raw text before LLM extraction
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "6000"))
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
