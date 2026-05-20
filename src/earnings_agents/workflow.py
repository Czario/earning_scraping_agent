from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import END, StateGraph

from earnings_agents.nodes.extract_financial_metrics import extract_financial_metrics_node
from earnings_agents.nodes.detect_document_type import detect_document_type_node
from earnings_agents.nodes.extract_html_text import extract_html_text_node
from earnings_agents.nodes.discover_earnings_release import discover_earnings_release_node
from earnings_agents.nodes.extract_pdf_text import extract_pdf_text_node
from earnings_agents.nodes.reflect_metrics import reflect_metrics_node, MAX_EXTRACTION_ATTEMPTS
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.tools.mongodb_client import upsert_earnings

logger = logging.getLogger(__name__)


# ── Routing helpers ──────────────────────────────────────────────────────────

def _route_after_discovery(
    state: EarningsAgentState,
) -> Literal["detect_document_type", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "detect_document_type"


def _route_by_file_type(
    state: EarningsAgentState,
) -> Literal["extract_pdf_text", "extract_html_text", "__end__"]:
    if state.get("status") == "failed":
        return "__end__"
    return "extract_pdf_text" if state.get("file_type") == "pdf" else "extract_html_text"


def _route_after_extraction(
    state: EarningsAgentState,
) -> Literal["reflect_metrics", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "reflect_metrics"


def _route_after_reflection(
    state: EarningsAgentState,
) -> Literal["extract_financial_metrics", "mongodb_save"]:
    """Loop back to re-extract if the reflection node flagged missing metrics."""
    if (
        state.get("status") == "text_extracted"
        and state.get("extraction_attempts", 0) < MAX_EXTRACTION_ATTEMPTS
    ):
        return "extract_financial_metrics"
    return "mongodb_save"


# ── MongoDB save node ────────────────────────────────────────────────────────

def mongodb_save_node(state: EarningsAgentState) -> EarningsAgentState:
    """Persist extracted earnings metrics to MongoDB."""
    ticker = state["ticker"]
    now = datetime.now(timezone.utc)

    # Use a stable _id so re-runs upsert rather than duplicate
    doc_id = f"{ticker}_{now.year}_latest"

    doc = {
        "_id": doc_id,
        "ticker": ticker,
        "company_name": state["company_name"],
        "source_url": state.get("discovered_file_url"),
        "file_type": state.get("file_type"),
        "metrics": state.get("metrics"),
        "scraped_at": now,
        "status": "success",
    }

    try:
        upsert_earnings(doc)
        logger.info("Saved earnings for %s as %s", ticker, doc_id)
        return {**state, "status": "saved"}
    except Exception as exc:  # noqa: BLE001
        return {**state, "status": "failed", "error": f"MongoDB save failed: {exc}"}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    """Compile and return the LangGraph earnings scraping workflow."""
    graph = StateGraph(EarningsAgentState)

    graph.add_node("discover_earnings_release", discover_earnings_release_node)
    graph.add_node("detect_document_type", detect_document_type_node)
    graph.add_node("extract_pdf_text", extract_pdf_text_node)
    graph.add_node("extract_html_text", extract_html_text_node)
    graph.add_node("extract_financial_metrics", extract_financial_metrics_node)
    graph.add_node("reflect_metrics", reflect_metrics_node)
    graph.add_node("mongodb_save", mongodb_save_node)

    graph.set_entry_point("discover_earnings_release")

    graph.add_conditional_edges(
        "discover_earnings_release",
        _route_after_discovery,
        {"detect_document_type": "detect_document_type", "__end__": END},
    )
    graph.add_conditional_edges(
        "detect_document_type",
        _route_by_file_type,
        {
            "extract_pdf_text": "extract_pdf_text",
            "extract_html_text": "extract_html_text",
            "__end__": END,
        },
    )
    graph.add_edge("extract_pdf_text", "extract_financial_metrics")
    graph.add_edge("extract_html_text", "extract_financial_metrics")
    graph.add_conditional_edges(
        "extract_financial_metrics",
        _route_after_extraction,
        {"reflect_metrics": "reflect_metrics", "__end__": END},
    )
    graph.add_conditional_edges(
        "reflect_metrics",
        _route_after_reflection,
        {
            "extract_financial_metrics": "extract_financial_metrics",
            "mongodb_save": "mongodb_save",
        },
    )
    graph.add_edge("mongodb_save", END)

    return graph.compile()
