from __future__ import annotations

import logging
import re as _re
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import END, StateGraph

from earnings_agents.nodes.extract_financial_metrics import extract_financial_metrics_node
from earnings_agents.nodes.detect_document_type import detect_document_type_node
from earnings_agents.nodes.extract_html_text import extract_html_text_node
from earnings_agents.nodes.discover_earnings_release import discover_earnings_release_node
from earnings_agents.nodes.extract_pdf_text import extract_pdf_text_node
from earnings_agents.nodes.analyze_metrics import analyze_metrics_node
from earnings_agents.nodes.cleanup_metrics import cleanup_metrics_node
from earnings_agents.nodes.load_company_concepts_node import load_company_concepts_node
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.tools.mongodb_client import upsert_earnings
from earnings_agents.hooks import with_hooks
from earnings_agents.config import EARNINGS_SAVE_TARGET, STRICT_ACCURACY

logger = logging.getLogger(__name__)


# ── Routing helpers ──────────────────────────────────────────────────────────

def _route_after_discovery(
    state: EarningsAgentState,
) -> Literal["load_company_concepts", "__end__"]:
    return "__end__" if state.get("status") == "failed" else "load_company_concepts"


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
) -> Literal["extract_financial_metrics", "cleanup_metrics", "__end__"]:
    """Route after analyze_metrics: loop back, proceed, or abort on failure."""
    if state.get("status") == "failed":
        return "__end__"
    if state.get("needs_reextract"):
        return "extract_financial_metrics"
    return "cleanup_metrics"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fiscal_year_from_period(metrics: dict, fallback_year: int) -> int:
    """Extract the 4-digit fiscal year from the ``__period__`` metric key.

    Falls back to *fallback_year* (UTC current year) when the period label
    is absent or contains no parseable year.
    """
    period = (metrics or {}).get("__period__")
    if period:
        m = _re.search(r"\b(20\d{2}|19\d{2})\b", str(period))
        if m:
            return int(m.group(1))
    return fallback_year


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

    # Use a stable _id so re-runs upsert rather than duplicate.
    # Year is taken from the __period__ label in the extracted metrics (fiscal
    # year from the source document), falling back to UTC current year.
    metrics = state.get("metrics") or {}
    fiscal_year = _fiscal_year_from_period(metrics, now.year)
    doc_id = f"{ticker}_{fiscal_year}_latest"

    # Mark as degraded when high-severity findings remain after all loop passes.
    findings = state.get("findings") or []
    high_unresolved = [
        f for f in findings
        if isinstance(f, dict) and f.get("severity") == "high"
    ]
    doc_status = "degraded" if high_unresolved else "success"
    if high_unresolved:
        logger.warning(
            "Saving %s as 'degraded' — %d unresolved critical finding(s): %s",
            ticker,
            len(high_unresolved),
            [f.get("message") for f in high_unresolved],
        )

    doc = {
        "_id": doc_id,
        "ticker": ticker,
        "company_name": state["company_name"],
        "source_url": state.get("discovered_file_url"),
        "file_type": state.get("file_type"),
        "metrics": metrics,
        "scraped_at": now,
        "status": doc_status,
    }
    if findings:
        doc["findings"] = findings
    if identity_warnings:
        doc["identity_warnings"] = identity_warnings
    if high_unresolved:
        doc["unresolved_findings"] = high_unresolved

    try:
        upsert_earnings(doc)
        logger.info("Saved earnings for %s as %s", ticker, doc_id)
    except Exception as exc:  # noqa: BLE001
        return {**state, "status": "failed", "error": f"MongoDB save failed: {exc}"}

    # When EARNINGS_SAVE_TARGET=normalize_data, also upsert into the
    # normalize_data DB using the concept_id-keyed metrics.
    if EARNINGS_SAVE_TARGET == "normalize_data":
        concept_metrics: dict = state.get("concept_metrics") or {}
        cik: str | None = state.get("company_cik")  # type: ignore[assignment]
        fy_end_month: int | None = state.get("fiscal_year_end_month")  # type: ignore[assignment]
        fy_end_code: str = str(state.get("fiscal_year_end_code") or "1231")
        period_str: str = str(metrics.get("__period__") or "")
        if concept_metrics and cik and fy_end_month and period_str:
            from earnings_agents.tools.normalize_data_client import upsert_concept_values
            try:
                n = upsert_concept_values(
                    cik=cik,
                    company_name=state["company_name"],
                    concept_metrics=concept_metrics,
                    period_str=period_str,
                    fiscal_year_end_month=fy_end_month,
                    fiscal_year_end_code=fy_end_code,
                )
                logger.info(
                    "normalize_data: upserted %d concept value(s) for %s", n, ticker
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "normalize_data upsert failed for %s: %s", ticker, exc
                )
        else:
            logger.warning(
                "normalize_data mode: skipping concept upsert for %s — "
                "missing concept_metrics=%s cik=%s fy_end_month=%s period=%r",
                ticker,
                bool(concept_metrics),
                cik,
                fy_end_month,
                period_str,
            )

    return {**state, "status": "saved"}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    """Compile and return the LangGraph earnings scraping workflow."""
    graph = StateGraph(EarningsAgentState)

    graph.add_node("discover_earnings_release", with_hooks(discover_earnings_release_node))
    graph.add_node("load_company_concepts", with_hooks(load_company_concepts_node))
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
        {"load_company_concepts": "load_company_concepts", "__end__": END},
    )
    graph.add_edge("load_company_concepts", "detect_document_type")
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
            "__end__": END,
        },
    )
    graph.add_edge("cleanup_metrics", "mongodb_save")
    graph.add_edge("mongodb_save", END)

    return graph.compile()
