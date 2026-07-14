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

from earnings_agents.config import PROMPT_HISTORY_PERIODS
from earnings_agents.tools.normalize_data_client import (
    get_company_by_ticker,
    get_next_period_type,
    get_recently_valued_concept_ids,
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

    Targeted extraction requires stored historical concepts for the ticker.
    When the company is absent from normalize_data, the DB is unreachable, or
    no income-statement concepts are stored, the run is *skipped*
    (``status="skipped"``) with a clear error message — we do not fall back to
    generic extraction.
    """
    ticker = state["ticker"]

    def _skip(message: str, **extra: object) -> EarningsAgentState:
        from earnings_agents.hooks import report_call
        report_call(f"  [load concepts]  ✗ skipped — {message[:80]}")
        logger.info("load_company_concepts: %s", message)
        skipped = {
            **state,
            "status": "skipped",
            "error": message,
            "target_concepts": [],
            "calculated_concepts": [],
            "company_cik": None,
            "fiscal_year_end_month": None,
            "fiscal_year_end_code": None,
            "detected_period_type": "quarterly",
        }
        skipped.update(extra)
        return skipped  # type: ignore[return-value]

    from earnings_agents.hooks import report_call
    report_call(f"  [load concepts]  looking up company in normalize_data…")
    try:
        company = get_company_by_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"No historical data for {ticker}: normalize_data lookup failed "
            f"({exc}); we don't have historical data for the company so we "
            f"can't proceed."
        )

    if company is None:
        return _skip(
            f"No historical data for {ticker} in normalize_data — we don't have "
            f"historical data for the company so we can't proceed."
        )

    cik: str = company["cik"]
    fy_end_month: int = company["fiscal_year_end_month"]
    fy_end_code: str | None = company.get("fiscal_year_end_code")
    sec_report_date_str: str | None = state.get("sec_report_date")  # type: ignore[assignment]
    report_call(f"  [load concepts]  found CIK {cik}  (FY end month {fy_end_month})")

    # Period-type decision (Option B — cadence drives, date is the safety net):
    #   * cadence_based — the deterministic filing-cycle state machine derived
    #                     from the last stored document (Q1→Q2→Q3→annual→Q1…).
    #                     This is the primary signal because the stored doc
    #                     already records exactly where the company sits in the
    #                     cycle — no fragile date arithmetic.
    #   * date_based    — report-date month vs fiscal year-end month. Used as:
    #                       (a) BOOTSTRAP when nothing is stored yet (cadence is
    #                           None), and
    #                       (b) a SAFETY OVERRIDE: if the period-end lands on the
    #                           fiscal year-end it is annual even when cadence
    #                           says quarterly — this covers a DB gap where one
    #                           or more quarters were skipped.
    date_based = _infer_period_type(sec_report_date_str, fy_end_month, fy_end_code)
    try:
        current_period_end: date | None = None
        if sec_report_date_str:
            try:
                current_period_end = date.fromisoformat(sec_report_date_str)
            except ValueError:
                current_period_end = None
        cadence_based = get_next_period_type(cik, current_period_end)
    except Exception as exc:  # noqa: BLE001 — cadence is best-effort
        logger.debug("load_company_concepts: cadence lookup failed for %s (%s)", ticker, exc)
        cadence_based = None

    if cadence_based is None:
        # Bootstrap: no stored history → trust the date signal alone.
        period_type = date_based
    elif date_based == "annual":
        # Date safety override: a period-end on the fiscal year-end is annual
        # even if cadence (working from a gappy DB) says otherwise.
        period_type = "annual"
    else:
        period_type = cadence_based

    if cadence_based is not None and cadence_based != date_based:
        logger.warning(
            "load_company_concepts: period-type signals diverge for %s — "
            "cadence=%s date=%s → using %s",
            ticker, cadence_based, date_based, period_type,
        )

    logger.info(
        "load_company_concepts: %s (CIK %s) — detected period_type=%s "
        "(sec_report_date=%s, fy_end_month=%s, cadence=%s)",
        ticker, cik, period_type, sec_report_date_str, fy_end_month, cadence_based,
    )

    try:
        concepts = get_statement_concepts(
            cik,
            statement_types=["income_statement"],
            period_type=period_type,
        )
    except Exception as exc:  # noqa: BLE001
        return _skip(
            f"No historical data for {ticker}: concept query failed ({exc}); "
            f"we don't have historical data for the company so we can't proceed.",
            company_cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period_type=period_type,
        )

    report_call(
        f"  [load concepts]  loaded {len(concepts)} income-statement concept(s) "
        f"({period_type})"
    )
    logger.info(
        "load_company_concepts: loaded %d income-statement concept(s) for %s (CIK %s, %s)",
        len(concepts),
        ticker,
        cik,
        period_type,
    )

    if not concepts:
        return _skip(
            f"No income-statement concepts stored for {ticker} in normalize_data — "
            f"we don't have historical data for the company so we can't proceed.",
            company_cik=cik,
            fiscal_year_end_month=fy_end_month,
            fiscal_year_end_code=company.get("fiscal_year_end_code"),
            detected_period_type=period_type,
        )

    # Prompt-pruning signal: which concepts has the company actually reported in
    # its last N periods?  Concepts with no recent history (dimensional [Member]
    # rows, retired line items) are dropped from the extraction prompt by
    # extract_financial_metrics_node.  Best-effort — an empty set disables pruning.
    recent_concept_ids: list[str] = []
    if PROMPT_HISTORY_PERIODS > 0:
        try:
            recent = get_recently_valued_concept_ids(
                cik, period_type=period_type, n_periods=PROMPT_HISTORY_PERIODS
            )
            recent_concept_ids = sorted(recent)
            logger.info(
                "load_company_concepts: %s has %d concept(s) with values in the "
                "last %d %s period(s) (of %d total) — prompt will be pruned",
                ticker, len(recent_concept_ids), PROMPT_HISTORY_PERIODS,
                period_type, len(concepts),
            )
        except Exception as exc:  # noqa: BLE001 — pruning is best-effort
            logger.debug(
                "load_company_concepts: recent-concept lookup failed for %s (%s) "
                "— extraction prompt will use the full concept list",
                ticker, exc,
            )

    return {
        **state,
        "company_cik": cik,
        "target_concepts": concepts,
        "recent_concept_ids": recent_concept_ids,
        "calculated_concepts": [],
        "fiscal_year_end_month": fy_end_month,
        "fiscal_year_end_code": company.get("fiscal_year_end_code"),
        "detected_period_type": period_type,
    }
