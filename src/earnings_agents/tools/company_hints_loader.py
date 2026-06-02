"""Company-specific extraction hints loader.

Reads human-curated hint files from ``data/company_hints/{TICKER}.md`` and
returns their contents as a string to be injected into LLM prompts.

Centralising this I/O here makes it easy to add caching or a remote-fetch
fallback in the future without touching extraction logic.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve relative to package root (src/earnings_agents/tools/ -> repo root)
_HINTS_DIR = Path(__file__).parents[3] / "data" / "company_hints"


def load_company_hints(ticker: str) -> str:
    """Return contents of ``data/company_hints/{TICKER}.md``, or empty string.

    Parameters
    ----------
    ticker:
        Uppercase ticker symbol (e.g. ``"AAPL"``).  Case-normalised internally.
    """
    hint_file = _HINTS_DIR / f"{ticker.upper()}.md"
    if hint_file.is_file():
        content = hint_file.read_text(encoding="utf-8").strip()
        if content:
            logger.info("Loaded company hints for %s (%d chars)", ticker, len(content))
            return content
    return ""
