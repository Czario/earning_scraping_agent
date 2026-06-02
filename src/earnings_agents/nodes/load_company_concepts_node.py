"""Load company GAAP concepts from normalize_data before extraction.

Looks up the company by ticker in normalize_data.companies, then fetches
income-statement concepts from normalize_data.normalized_concepts_quarterly.
Populates ``company_cik``, ``target_concepts``, and ``fiscal_year_end_month``
in state so ``extract_financial_metrics_node`` can build a targeted prompt.

Failure is always graceful: if the company is not found or the DB is
unreachable the node falls back to ``target_concepts=[]`` and lets the
generic income-statement extraction proceed.  It never sets ``status=failed``.
"""
from __future__ import annotations

import logging

from earnings_agents.tools.normalize_data_client import (
    get_company_by_ticker,
    get_statement_concepts,
)
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


def load_company_concepts_node(state: EarningsAgentState) -> EarningsAgentState:
    """Load GAAP concepts for targeted extraction from normalize_data.

    Falls back to ``target_concepts=[]`` (generic extraction) when the company
    is not found in the DB or the DB is unreachable.
    """
    ticker = state["ticker"]

    try:
        company = get_company_by_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_company_concepts: DB lookup failed for %s (%s) — "
            "falling back to generic extraction",
            ticker,
            exc,
        )
        return {
            **state,
            "target_concepts": [],
            "company_cik": None,
            "fiscal_year_end_month": None,
            "fiscal_year_end_code": None,
        }

    if company is None:
        logger.info(
            "load_company_concepts: %s not found in normalize_data.companies — "
            "falling back to generic extraction",
            ticker,
        )
        return {
            **state,
            "target_concepts": [],
            "company_cik": None,
            "fiscal_year_end_month": None,
            "fiscal_year_end_code": None,
        }

    cik: str = company["cik"]

    try:
        concepts = get_statement_concepts(cik, statement_types=["income_statement"])
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_company_concepts: concept query failed for %s CIK=%s (%s) — "
            "falling back to generic extraction",
            ticker,
            cik,
            exc,
        )
        return {
            **state,
            "target_concepts": [],
            "company_cik": cik,
            "fiscal_year_end_month": company["fiscal_year_end_month"],
            "fiscal_year_end_code": company.get("fiscal_year_end_code"),
        }

    logger.info(
        "load_company_concepts: loaded %d income-statement concept(s) for %s (CIK %s)",
        len(concepts),
        ticker,
        cik,
    )

    return {
        **state,
        "company_cik": cik,
        "target_concepts": concepts,
        "fiscal_year_end_month": company["fiscal_year_end_month"],
        "fiscal_year_end_code": company.get("fiscal_year_end_code"),
    }
