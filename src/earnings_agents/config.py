from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB", "earnings_db")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "earnings")

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
