from __future__ import annotations

import logging
from datetime import date as _date
from typing import Literal

from langgraph.graph import END, StateGraph

from earnings_agents.nodes.extract_financial_metrics import extract_financial_metrics_node
from earnings_agents.nodes.detect_document_type import detect_document_type_node
from earnings_agents.nodes.extract_html_text import extract_html_text_node
from earnings_agents.nodes.analyze_metrics import analyze_metrics_node
from earnings_agents.nodes.cleanup_metrics import cleanup_metrics_node
from earnings_agents.nodes.load_company_concepts_node import load_company_concepts_node
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.hooks import with_hooks
from earnings_agents.config import STRICT_ACCURACY

logger = logging.getLogger(__name__)


# ── Routing helpers ──────────────────────────────────────────────────────────

def _fail_or(state: EarningsAgentState, next_node: str) -> str:
    """Return ``next_node`` unless the state signals a failure, in which case END.

    Used to collapse guard-only routing helpers (those that only check
    ``status == "failed"``) into a single reusable primitive.
    """
    return "__end__" if state.get("status") == "failed" else next_node


def _route_after_concepts(
    state: EarningsAgentState,
) -> Literal["detect_document_type", "__end__"]:
    """End the run when concept loading failed or was skipped.

    Targeted extraction requires stored concepts; ``load_company_concepts``
    sets ``status="skipped"`` when none are available, in which case there is
    nothing further to do.
    """
    if state.get("status") in ("failed", "skipped"):
        return "__end__"
    return "detect_document_type"


def _route_after_extraction(
    state: EarningsAgentState,
) -> Literal["analyze_metrics", "__end__"]:
    return _fail_or(state, "analyze_metrics")  # type: ignore[return-value]


def _route_after_analysis(
    state: EarningsAgentState,
) -> Literal["extract_financial_metrics", "cleanup_metrics", "__end__"]:
    """Route after analyze_metrics: loop back, proceed, or abort on failure."""
    if state.get("status") == "failed":
        return "__end__"
    if state.get("needs_reextract"):
        return "extract_financial_metrics"
    return "cleanup_metrics"


# ── MongoDB save node ────────────────────────────────────────────────────────

def mongodb_save_node(state: EarningsAgentState) -> EarningsAgentState:
    """Upsert extracted concept metrics into normalize_data.

    Routes to ``concept_values_quarterly`` or ``concept_values_annual``
    based on the ``__period__`` string extracted from the filing (via
    ``upsert_concept_values`` → ``detect_period_type``).

    Refuses to save when accounting identity checks failed and
    ``STRICT_ACCURACY`` is enabled (default).
    """
    ticker = state["ticker"]
    identity_warnings = state.get("identity_warnings") or []

    if identity_warnings and STRICT_ACCURACY:
        msg = (
            f"Refusing to save {ticker}: {len(identity_warnings)} accounting "
            f"identity check(s) failed — "
            + "; ".join(identity_warnings)
        )
        logger.error(msg)
        return {**state, "status": "failed", "error": msg}

    metrics = state.get("metrics") or {}
    sec_report_date_str: str | None = state.get("sec_report_date")  # type: ignore[assignment]
    sec_rd: _date | None = None
    if sec_report_date_str:
        try:
            sec_rd = _date.fromisoformat(sec_report_date_str)
        except ValueError:
            pass

    findings = state.get("findings") or []
    high_unresolved = [
        f for f in findings
        if isinstance(f, dict) and f.get("severity") == "high"
    ]
    if high_unresolved:
        logger.warning(
            "Saving %s with %d unresolved high-severity finding(s): %s",
            ticker,
            len(high_unresolved),
            [f.get("message") for f in high_unresolved],
        )

    concept_metrics: dict = state.get("concept_metrics") or {}
    derived_ids: set[str] = set(state.get("derived_concept_ids") or [])
    cik: str | None = state.get("company_cik")  # type: ignore[assignment]
    fy_end_month: int | None = state.get("fiscal_year_end_month")  # type: ignore[assignment]
    fy_end_code: str = str(state.get("fiscal_year_end_code") or "1231")
    period_str: str = str(metrics.get("__period__") or "")
    detected_period_type: str | None = state.get("detected_period_type")  # type: ignore[assignment]

    if concept_metrics and cik and fy_end_month and (period_str or sec_rd):
        from earnings_agents.tools.normalize_data_client import upsert_concept_values
        try:
            n = upsert_concept_values(
                cik=cik,
                company_name=state["company_name"],
                concept_metrics=concept_metrics,
                period_str=period_str,
                fiscal_year_end_month=fy_end_month,
                fiscal_year_end_code=fy_end_code,
                report_date=sec_rd,
                period_type_override=detected_period_type,
                derived_concept_ids=derived_ids,
            )
            logger.info(
                "normalize_data: upserted %d concept value(s) for %s", n, ticker
            )
        except Exception as exc:  # noqa: BLE001
            return {**state, "status": "failed", "error": f"normalize_data upsert failed: {exc}"}
    else:
        logger.warning(
            "Skipping normalize_data upsert for %s — "
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

    graph.add_node("load_company_concepts", with_hooks(load_company_concepts_node))
    graph.add_node("detect_document_type", with_hooks(detect_document_type_node))
    graph.add_node("extract_html_text", with_hooks(extract_html_text_node))
    graph.add_node("extract_financial_metrics", with_hooks(extract_financial_metrics_node))
    graph.add_node("analyze_metrics", with_hooks(analyze_metrics_node))
    graph.add_node("cleanup_metrics", with_hooks(cleanup_metrics_node))
    graph.add_node("mongodb_save", with_hooks(mongodb_save_node))

    graph.set_entry_point("load_company_concepts")

    graph.add_conditional_edges(
        "load_company_concepts",
        _route_after_concepts,
        {"detect_document_type": "detect_document_type", "__end__": END},
    )
    graph.add_conditional_edges(
        "detect_document_type",
        lambda state: _fail_or(state, "extract_html_text"),
        {"extract_html_text": "extract_html_text", "__end__": END},
    )
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
