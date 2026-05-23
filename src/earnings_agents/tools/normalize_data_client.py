"""Client for reading from and writing to the normalize_data MongoDB database.

Used when EARNINGS_SAVE_TARGET=normalize_data.  The module keeps a single
shared MongoClient instance (same pattern as mongodb_client.py) and registers
an atexit handler for clean shutdown.
"""
from __future__ import annotations

import atexit
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId
from pymongo import MongoClient, UpdateOne

from earnings_agents.config import MONGODB_URI

logger = logging.getLogger(__name__)

_NORMALIZE_DB = "normalize_data"
_client: Optional[MongoClient] = None  # type: ignore[type-arg]


def _get_client() -> MongoClient:  # type: ignore[type-arg]
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
        atexit.register(lambda: _client.close() if _client else None)  # type: ignore[union-attr]
    return _client


# ── Company lookup ───────────────────────────────────────────────────────────

def get_company_by_ticker(ticker: str) -> dict[str, Any] | None:
    """Return ``{cik, name, fiscal_year_end_month}`` for *ticker*, or ``None``.

    Queries normalize_data.companies by ``ticker_symbol`` (case-insensitive).
    ``fiscal_year_end_month`` is derived from the ``corporate_info.fiscal_year_end``
    field, which is stored as an "MMDD" string (e.g. "0630" for June 30).
    """
    db = _get_client()[_NORMALIZE_DB]
    doc = db["companies"].find_one(
        {"ticker_symbol": ticker.upper()},
        {"cik": 1, "name": 1, "corporate_info.fiscal_year_end": 1},
    )
    if doc is None:
        return None
    fy_code: str = (doc.get("corporate_info") or {}).get("fiscal_year_end", "1231") or "1231"
    try:
        fy_end_month = int(fy_code[:2])
        if not (1 <= fy_end_month <= 12):
            fy_end_month = 12
    except (ValueError, TypeError):
        fy_end_month = 12
    return {
        "cik": str(doc["cik"]),
        "name": doc.get("name", ticker),
        "fiscal_year_end_month": fy_end_month,
        "fiscal_year_end_code": fy_code,
    }


# ── Concept lookup ───────────────────────────────────────────────────────────

def get_statement_concepts(
    cik: str,
    statement_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted concept dicts for *cik* and *statement_types*.

    Filters out abstract and dimension-only concepts.  Results are sorted by
    ``path`` so the prompt lists concepts in the order they appear in the
    financial statement.

    Each returned dict has keys: ``_id`` (str), ``concept`` (GAAP name),
    ``label``, ``path``, ``statement_type``.
    """
    if statement_types is None:
        statement_types = ["income_statement"]
    db = _get_client()[_NORMALIZE_DB]
    cursor = db["normalized_concepts_quarterly"].find(
        {
            "company_cik": cik,
            "statement_type": {"$in": statement_types},
            "dimension_concept": {"$ne": True},
            "abstract": {"$ne": True},
        },
        {
            "_id": 1,
            "concept": 1,
            "label": 1,
            "path": 1,
            "statement_type": 1,
        },
    ).sort("path", 1)
    return [
        {
            "_id": str(d["_id"]),
            "concept": d.get("concept", ""),
            "label": d.get("label", ""),
            "path": d.get("path", ""),
            "statement_type": d.get("statement_type", ""),
        }
        for d in cursor
    ]


# ── Period helpers ───────────────────────────────────────────────────────────

_MONTH_NAME_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})",
    re.IGNORECASE,
)

# Keywords that indicate a full-year / annual period.
_ANNUAL_PERIOD_RE = re.compile(
    r"\b(year|twelve\s+months?|52\s+weeks?|53\s+weeks?|annual|full[- ]year)\b",
    re.IGNORECASE,
)

# Number words that appear in US earnings period strings.
_PERIOD_WORD_NUMS: dict[str, int] = {
    "three": 3,
    "six": 6,
    "nine": 9,
    "twelve": 12,
    "thirteen": 13,
    "twenty-six": 26,
    "thirty-nine": 39,
    "fifty-two": 52,
    "fifty-three": 53,
}

# Matches "Thirteen Weeks", "26 Weeks", "Six Months", "9 Months", etc.
# Longer word forms must precede their sub-strings in the alternation.
_DURATION_RE = re.compile(
    r"\b(thirteen|twenty-six|thirty-nine|fifty-(?:two|three)"
    r"|three|six|nine|twelve|\d+)"
    r"\s+(weeks?|months?)\b",
    re.IGNORECASE,
)

# Matches "First Quarter", "Second Quarter", etc. or bare "Q1"–"Q4".
_ORDINAL_QUARTER_RE = re.compile(
    r"\b(first|second|third|fourth)\s+quarter\b|\bQ([1-4])\b",
    re.IGNORECASE,
)
_ORDINAL_TO_NUM: dict[str, int] = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _extract_duration(period_str: str) -> tuple[int, str] | None:
    """Return ``(count, unit)`` parsed from *period_str*, or ``None``.

    *unit* is either ``"weeks"`` or ``"months"``.
    """
    m = _DURATION_RE.search(period_str)
    if not m:
        return None
    raw = m.group(1).lower()
    unit = "months" if m.group(2).lower().startswith("m") else "weeks"
    try:
        count = int(raw)
    except ValueError:
        count = _PERIOD_WORD_NUMS.get(raw)
        if count is None:
            return None
    return count, unit


def _quarter_from_period_str(period_str: str) -> int | None:
    """Return the fiscal quarter (1–3) inferred from *period_str*, or ``None``.

    Uses cumulative duration to determine the quarter:

    * Week-based (52-week fiscal-year companies always report cumulative):
      13 w → Q1, 26 w → Q2, 39 w → Q3.
    * Month-based unambiguous values:
      6 m → Q2, 9 m → Q3.  (3 months is ambiguous — falls back to date math.)
    * Ordinal words: "First Quarter" → Q1, etc.

    Annual periods (52/53 weeks, 12 months) are handled by
    ``detect_period_type`` upstream and never reach here.
    """
    if not period_str:
        return None

    # Explicit ordinal label takes priority.
    m = _ORDINAL_QUARTER_RE.search(period_str)
    if m:
        if m.group(1):
            return _ORDINAL_TO_NUM.get(m.group(1).lower())
        return int(m.group(2))

    duration = _extract_duration(period_str)
    if duration is None:
        return None
    count, unit = duration

    if unit == "weeks":
        # 52-week fiscal-year companies report cumulative weeks from FY start.
        if 12 <= count <= 14:
            return 1
        if 25 <= count <= 27:
            return 2
        if 38 <= count <= 40:
            return 3
    elif unit == "months":
        # 6- and 9-month figures are unambiguously cumulative from FY start.
        if count == 6:
            return 2
        if count == 9:
            return 3
    return None


def detect_period_type(period_str: str) -> str:
    """Return ``'annual'`` or ``'quarterly'`` based on *period_str*.

    Annual indicators: "Year Ended", "Twelve Months Ended", "52/53 Weeks Ended".
    Everything else (including "Three Months", "Thirteen Weeks") → quarterly.
    """
    return "annual" if _ANNUAL_PERIOD_RE.search(period_str) else "quarterly"


def parse_period_end_date(period_str: str) -> date | None:
    """Extract a ``date`` from a period string like "Three Months Ended March 31, 2026".

    Returns ``None`` when no recognisable date pattern is found.
    """
    m = _MONTH_NAME_RE.search(period_str)
    if m:
        try:
            return datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
            ).date()
        except ValueError:
            pass
    return None


def compute_fiscal_period(
    period_end_date: date,
    fiscal_year_end_month: int,
    period_str: str = "",
) -> tuple[int, int]:
    """Return ``(fiscal_year, quarter)`` for a period end date.

    ``fiscal_year_end_month``: 1-12 (e.g. 6 for June, 12 for December).

    When *period_str* is provided the cumulative duration it contains
    (e.g. "Twenty-Six Weeks" → Q2) is used to identify the quarter.
    This is essential for 52-week fiscal-year companies (e.g. BJ Wholesale)
    whose quarter end dates fall one calendar month later than strict
    3-month boundaries would suggest.

    When *period_str* gives no unambiguous quarter (e.g. "Three Months"),
    the function falls back to a calendar-month calculation that is exact
    for companies whose quarter ends align with calendar month boundaries.

    Examples (MSFT, fy_end_month=6):
      - 2026-03-31 → FY2026 Q3
      - 2026-06-30 → FY2026 Q4
      - 2025-09-30 → FY2026 Q1

    Examples (BJ, fy_end_month=1, with period_str):
      - "Thirteen Weeks Ended May 2, 2026"  → FY2027 Q1
      - "Twenty-Six Weeks Ended Aug 1, 2026" → FY2027 Q2
    """
    m = period_end_date.month
    y = period_end_date.year

    # Fiscal year: if period month falls within the FY ending in fy_end_month
    # of year y (i.e. m <= fy_end_month) → fiscal_year = y; otherwise y + 1.
    fiscal_year = y if m <= fiscal_year_end_month else y + 1

    # Quarter: prefer period-string-derived value; fall back to calendar math.
    quarter = _quarter_from_period_str(period_str)
    if quarter is None:
        # Calendar-month fallback — exact for standard (non-52-week) companies.
        fy_start_month = fiscal_year_end_month % 12 + 1
        fy_month_offset = (m - fy_start_month) % 12
        quarter = fy_month_offset // 3 + 1

    return fiscal_year, quarter


def parse_period_start_date(period_str: str, end_date: date) -> date | None:
    """Return the first calendar day of the reporting period, or ``None``.

    Uses the cumulative duration encoded in *period_str* to count backwards
    from *end_date*:

    * Week-based: ``end_date - (weeks * 7) + 1 day``
      e.g. "Thirteen Weeks Ended May 2, 2026" → Feb 1, 2026
    * Month-based: first day of the month that is *n* months before the
      end month (inclusive of the period).
      e.g. "Six Months Ended June 30, 2026" → Jan 1, 2026

    Returns ``None`` when no duration can be parsed from *period_str*.
    """
    duration = _extract_duration(period_str)
    if duration is None:
        return None
    count, unit = duration
    if unit == "weeks":
        return end_date - timedelta(weeks=count) + timedelta(days=1)
    # months
    start_m = end_date.month - count + 1
    start_y = end_date.year
    while start_m <= 0:
        start_m += 12
        start_y -= 1
    return date(start_y, start_m, 1)


# ── Upsert ───────────────────────────────────────────────────────────────────

def upsert_concept_values(
    cik: str,
    company_name: str,
    concept_metrics: dict[str, float],  # concept_id → value
    period_str: str,
    fiscal_year_end_month: int,
    fiscal_year_end_code: str = "1231",
    statement_type: str = "income_statement",
) -> int:
    """Bulk-upsert concept values into the appropriate collection.

    Routes to ``concept_values_quarterly`` or ``concept_values_annual``
    depending on *period_str*:
      - "Three Months Ended …" / "Thirteen Weeks Ended …" → quarterly
      - "Year Ended …" / "Twelve Months Ended …" / "52/53 Weeks Ended …" → annual

    Documents are written to match the existing schema used by the SEC-based
    pipeline, with ``concept_id`` stored as ``ObjectId`` and ``end_date`` as
    a native ``datetime`` so the upsert filter correctly de-duplicates
    re-runs of the same earnings release.

    Returns the number of operations submitted (0 on early-exit failures).
    """
    if not concept_metrics:
        logger.debug("upsert_concept_values: empty concept_metrics — nothing to do")
        return 0

    end_date = parse_period_end_date(period_str)
    if end_date is None:
        logger.warning(
            "upsert_concept_values: cannot parse period end date from %r — skipping",
            period_str,
        )
        return 0

    period_type = detect_period_type(period_str)  # "annual" | "quarterly"
    collection_name = f"concept_values_{period_type}"
    form_type = "10-K" if period_type == "annual" else "10-Q"

    fiscal_year, quarter = compute_fiscal_period(end_date, fiscal_year_end_month, period_str)
    start_date = parse_period_start_date(period_str, end_date)
    # Store end_date as a native UTC datetime to match the existing collection schema.
    end_datetime = datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0,
                            tzinfo=timezone.utc)
    period_date_str = end_date.strftime("%Y-%m-%d")
    now = datetime.now(tz=timezone.utc)

    db = _get_client()[_NORMALIZE_DB]
    collection = db[collection_name]

    ops: list[UpdateOne] = []
    for concept_id_str, value in concept_metrics.items():
        try:
            concept_oid = ObjectId(concept_id_str)
        except Exception:  # noqa: BLE001 — invalid ObjectId string, skip
            logger.warning(
                "upsert_concept_values: invalid ObjectId %r — skipping", concept_id_str
            )
            continue

        period_doc: dict[str, Any] = {
            "end_date": end_datetime,
            "period_date": period_date_str,
            "form_type": form_type,
            "fiscal_year_end_code": fiscal_year_end_code,
            "fiscal_year": fiscal_year,
            "data_source": "earnings_press_release",
            "company_cik": cik,
            "company_name": company_name,
            "unit": "USD",
        }
        if start_date is not None:
            period_doc["start_date"] = datetime(
                start_date.year, start_date.month, start_date.day,
                tzinfo=timezone.utc,
            )
        # Annual periods have no quarter dimension; quarterly periods do.
        if period_type == "quarterly":
            period_doc["quarter"] = quarter

        doc: dict[str, Any] = {
            "concept_id": concept_oid,
            "company_cik": cik,
            "statement_type": statement_type,
            "form_type": form_type,
            "reporting_period": period_doc,
            "value": value,
            "created_at": now,
            "dimension_value": False,
            "calculated": False,
        }
        ops.append(
            UpdateOne(
                {
                    "concept_id": concept_oid,
                    "reporting_period.end_date": end_datetime,
                },
                {"$set": doc},
                upsert=True,
            )
        )

    if not ops:
        return 0

    collection.bulk_write(ops, ordered=False)
    if period_type == "quarterly":
        logger.info(
            "upsert_concept_values: %d concept(s) → %s  CIK %s FY%d Q%d",
            len(ops), collection_name, cik, fiscal_year, quarter,
        )
    else:
        logger.info(
            "upsert_concept_values: %d concept(s) → %s  CIK %s FY%d",
            len(ops), collection_name, cik, fiscal_year,
        )
    return len(ops)
