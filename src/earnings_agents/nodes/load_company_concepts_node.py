"""Load company GAAP concepts from normalize_data before extraction.

Looks up the company by ticker in normalize_data.companies, then fetches
income-statement concepts from the appropriate normalized_concepts collection:
  - ``normalized_concepts_quarterly`` for quarterly filings (most 8-Ks)
  - ``normalized_concepts_annual``    for annual filings (Q4 / year-end 8-Ks)

Period type is inferred from ``sec_report_date`` and the company's
``fiscal_year_end_month``: if the report date falls in the fiscal year-end
month the filing is treated as annual; otherwise quarterly.  When
``sec_report_date`` is absent (IR path) the node defaults to quarterly.

Populates ``company_cik``, ``target_concepts``, ``fiscal_year_end_month``,
and ``detected_period_type`` in state so ``extract_financial_metrics_node``
can build a targeted prompt.

Failure is always graceful: if the company is not found or the DB is
unreachable the node falls back to ``target_concepts=[]`` and lets the
generic income-statement extraction proceed.  It never sets ``status=failed``.
"""
from __future__ import annotations

import logging
from datetime import date

from earnings_agents.tools.normalize_data_client import (
    get_company_by_ticker,
    get_statement_concepts,
)
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)


def _nominal_fye_date(year: int, fy_month: int, fy_day: int) -> date | None:
    """Build the nominal fiscal-year-end ``date`` for *year*, clamping bad days.

    Handles ``02/29``-style codes on non-leap years by stepping the day back to
    the last valid day of the month.  Returns ``None`` only when *fy_month* is
    out of range.
    """
    for day in range(fy_day, 0, -1):
        try:
            return date(year, fy_month, day)
        except ValueError:
            continue
    return None


def _infer_period_type(
    sec_report_date_str: str | None,
    fiscal_year_end_month: int | None,
    fiscal_year_end_code: str | None = None,
) -> str:
    """Return ``'annual'`` when the report date marks the fiscal year-end.

    Uses ``sec_report_date`` (set on the SEC path) and the company's
    ``fiscal_year_end_month`` from normalize_data.  Defaults to
    ``'quarterly'`` when either value is absent.

    Two acceptance rules, both pointing at the same fiscal year-end:

    * **Calendar-month match** — the report month equals the fiscal year-end
      month (covers the common case of a fixed-date fiscal year-end).
    * **52/53-week proximity** — 52/53-week filers anchor their year-end to a
      weekday "nearest" a fixed date, so the period-end can drift a few days
      across the calendar-month boundary between years.  When
      ``fiscal_year_end_code`` ("MMDD") is available, the report date is
      treated as annual if it lands within 7 days of the nominal fiscal
      year-end.  Seven days is far below the ~91-day gap to the nearest
      quarter-end, so this never misclassifies a quarterly filing.
    """
    if not sec_report_date_str or not fiscal_year_end_month:
        return "quarterly"
    try:
        report_date = date.fromisoformat(sec_report_date_str)
    except ValueError:
        return "quarterly"
    if report_date.month == fiscal_year_end_month:
        return "annual"

    # 52/53-week boundary drift: compare against the nominal year-end day.
    fy_day: int | None = None
    if fiscal_year_end_code and len(str(fiscal_year_end_code)) >= 4:
        try:
            fy_day = int(str(fiscal_year_end_code)[2:4])
        except ValueError:
            fy_day = None
    if fy_day:
        for year in (report_date.year - 1, report_date.year, report_date.year + 1):
            fye = _nominal_fye_date(year, fiscal_year_end_month, fy_day)
            if fye is not None and abs((report_date - fye).days) <= 7:
                return "annual"
    return "quarterly"


def load_company_concepts_node(state: EarningsAgentState) -> EarningsAgentState:
    """Load GAAP concepts for targeted extraction from normalize_data.

    Falls back to ``target_concepts=[]`` (generic extraction) when the company
    is not found in the DB or the DB is unreachable.
    """
    ticker = state["ticker"]

    _fallback = {
        **state,
        "target_concepts": [],
        "company_cik": None,
        "fiscal_year_end_month": None,
        "fiscal_year_end_code": None,
        "detected_period_type": "quarterly",
    }

    try:
        company = get_company_by_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_company_concepts: DB lookup failed for %s (%s) — "
            "falling back to generic extraction",
            ticker,
            exc,
        )
        return _fallback

    if company is None:
        logger.info(
            "load_company_concepts: %s not found in normalize_data.companies — "
            "falling back to generic extraction",
            ticker,
        )
        return _fallback

    cik: str = company["cik"]
    fy_end_month: int = company["fiscal_year_end_month"]
    fy_end_code: str | None = company.get("fiscal_year_end_code")
    sec_report_date_str: str | None = state.get("sec_report_date")  # type: ignore[assignment]
    period_type = _infer_period_type(sec_report_date_str, fy_end_month, fy_end_code)

    logger.info(
        "load_company_concepts: %s (CIK %s) — detected period_type=%s "
        "(sec_report_date=%s, fy_end_month=%s)",
        ticker, cik, period_type, sec_report_date_str, fy_end_month,
    )

    try:
        concepts = get_statement_concepts(
            cik,
            statement_types=["income_statement"],
            period_type=period_type,
        )
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
            "fiscal_year_end_month": fy_end_month,
            "fiscal_year_end_code": company.get("fiscal_year_end_code"),
            "detected_period_type": period_type,
        }

    logger.info(
        "load_company_concepts: loaded %d income-statement concept(s) for %s (CIK %s, %s)",
        len(concepts),
        ticker,
        cik,
        period_type,
    )

    return {
        **state,
        "company_cik": cik,
        "target_concepts": concepts,
        "fiscal_year_end_month": fy_end_month,
        "fiscal_year_end_code": company.get("fiscal_year_end_code"),
        "detected_period_type": period_type,
    }
