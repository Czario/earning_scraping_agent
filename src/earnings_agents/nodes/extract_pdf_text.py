from __future__ import annotations

import io
import logging

import pdfplumber
import requests

from earnings_agents.config import HTTP_TIMEOUT
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def extract_pdf_text_node(state: EarningsAgentState) -> EarningsAgentState:
    """Download a PDF earnings document and extract all text with pdfplumber."""
    url = state.get("discovered_file_url", "")
    ticker = state["ticker"]

    try:
        response = requests.get(url, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()

        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        raw_text = "\n\n".join(pages_text).strip()
        logger.info("PDF extracted %d chars for %s", len(raw_text), ticker)
        return {**state, "raw_text": raw_text, "status": "text_extracted"}
    except Exception as exc:  # noqa: BLE001
        return {**state, "status": "failed", "error": f"PDF extraction failed: {exc}"}
