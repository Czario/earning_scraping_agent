from __future__ import annotations

import logging

from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


def detect_document_type_node(state: EarningsAgentState) -> EarningsAgentState:
    """Confirm the discovered earnings document is an HTML file.

    SEC 8-K press releases are always served as HTML. This node sets
    ``file_type='html'`` and advances ``status`` to ``'fetched'``.
    """
    url = state.get("discovered_file_url", "")
    if not url:
        return {**state, "status": "failed", "error": "No file URL to fetch"}

    logger.info("File type: html — %s", url)
    return {**state, "file_type": "html", "status": "fetched"}
