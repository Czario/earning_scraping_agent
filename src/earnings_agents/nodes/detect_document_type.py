from __future__ import annotations

import logging

import requests

from earnings_agents.config import HTTP_TIMEOUT
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

_PDF_CONTENT_TYPES = frozenset({"application/pdf", "application/x-pdf"})

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def detect_document_type_node(state: EarningsAgentState) -> EarningsAgentState:
    """Detect the file type of the discovered earnings document.

    Uses the URL extension first; if ambiguous, sends a HEAD request to inspect
    the Content-Type header.
    """
    url = state.get("discovered_file_url", "")
    if not url:
        return {**state, "status": "failed", "error": "No file URL to fetch"}

    # Fast path: extension-based detection
    path = url.split("?")[0].lower()
    if path.endswith(".pdf"):
        logger.info("File type (extension): pdf — %s", url)
        return {**state, "file_type": "pdf", "status": "fetched"}
    if path.endswith((".htm", ".html")):
        logger.info("File type (extension): html — %s", url)
        return {**state, "file_type": "html", "status": "fetched"}

    # Fallback: HEAD request
    try:
        response = requests.head(
            url, headers=_HEADERS, timeout=HTTP_TIMEOUT, allow_redirects=True
        )
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        file_type = "pdf" if content_type in _PDF_CONTENT_TYPES else "html"
        logger.info("File type (Content-Type %s): %s — %s", content_type, file_type, url)
        return {**state, "file_type": file_type, "status": "fetched"}
    except requests.RequestException as exc:
        return {**state, "status": "failed", "error": f"File type detection failed: {exc}"}
