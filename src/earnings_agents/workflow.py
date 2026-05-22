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
from earnings_agents.nodes.reflect_metrics import MAX_EXTRACTION_ATTEMPTS
from earnings_agents.nodes.analyze_metrics import analyze_metrics_node
from earnings_agents.nodes.cleanup_metrics import cleanup_metrics_node
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.tools.mongodb_client import upsert_earnings
from earnings_agents.hooks import with_hooks
from earnings_agents.config import STRICT_ACCURACY

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
) -> Literal["analyze_metrics", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "analyze_metrics"


def _route_after_analysis(
    state: EarningsAgentState,
) -> Literal["extract_financial_metrics", "cleanup_metrics"]:
    """Loop back to re-extract when analyze_metrics flagged a critical gap."""
    if (
        state.get("status") == "text_extracted"
        and state.get("extraction_attempts", 0) < MAX_EXTRACTION_ATTEMPTS
    ):
        return "extract_financial_metrics"
    return "cleanup_metrics"


# ── MongoDB save node ────────────────────────────────────────────────────────

def mongodb_save_node(state: EarningsAgentState) -> EarningsAgentState:
    """Persist extracted earnings metrics to MongoDB.

    Refuses to save when accounting identity checks failed and
    ``STRICT_ACCURACY`` is enabled (default). When STRICT_ACCURACY is off,
    saves the document with an ``identity_warnings`` field so the bad rows
    are queryable.
    """
    ticker = state["ticker"]
    now = datetime.now(timezone.utc)
    identity_warnings = state.get("identity_warnings") or []

    if identity_warnings and STRICT_ACCURACY:
        msg = (
            f"Refusing to save {ticker}: {len(identity_warnings)} accounting "
            f"identity check(s) failed — "
            + "; ".join(identity_warnings)
        )
        logger.error(msg)
        return {**state, "status": "failed", "error": msg}

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
    if identity_warnings:
        doc["identity_warnings"] = identity_warnings

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

    graph.add_node("discover_earnings_release", with_hooks(discover_earnings_release_node))
    graph.add_node("detect_document_type", with_hooks(detect_document_type_node))
    graph.add_node("extract_pdf_text", with_hooks(extract_pdf_text_node))
    graph.add_node("extract_html_text", with_hooks(extract_html_text_node))
    graph.add_node("extract_financial_metrics", with_hooks(extract_financial_metrics_node))
    graph.add_node("analyze_metrics", with_hooks(analyze_metrics_node))
    graph.add_node("cleanup_metrics", with_hooks(cleanup_metrics_node))
    graph.add_node("mongodb_save", with_hooks(mongodb_save_node))

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
        {"analyze_metrics": "analyze_metrics", "__end__": END},
    )
    graph.add_conditional_edges(
        "analyze_metrics",
        _route_after_analysis,
        {
            "extract_financial_metrics": "extract_financial_metrics",
            "cleanup_metrics": "cleanup_metrics",
        },
    )
    graph.add_edge("cleanup_metrics", "mongodb_save")
    graph.add_edge("mongodb_save", END)

    return graph.compile()
