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

_CONCEPT_PREFIX_RX = re.compile(r"(?:us-gaap|system|ifrs-full|dei|srt):", re.IGNORECASE)
_MEMBER_RX = re.compile(
    r"(?:us-gaap|system|ifrs-full|dei|srt):([A-Za-z0-9_]+)", re.IGNORECASE
)
_FULL_MEMBER_RX = re.compile(
    r"((?:us-gaap|ifrs-full|dei|srt):[A-Za-z0-9_]+Member)", re.IGNORECASE
)
_CAMEL_SPLIT_RX = re.compile(r"(?<!^)(?=[A-Z])")


def _extract_member_tag(raw: str) -> str:
    """Return the full XBRL member concept tag from a raw label string.

    Raw dimensional labels look like::

        "Net sales\\n\\n\\nus-gaap:ProductMember"

    Returns the full tag (e.g. ``"us-gaap:ProductMember"``) or ``""``.
    """
    m = _FULL_MEMBER_RX.search(raw)
    return m.group(1) if m else ""


def _clean_label(raw: str) -> tuple[str, str]:
    """Split a raw concept label into ``(base_label, member_qualifier)``.

    Some upstream rows store ``label`` as a multi-line string with an XBRL
    axis member appended after blank lines, e.g.::

        "Net sales\\n\\n\\n\\nus-gaap:ProductMember"

    Splitting it lets us:
      * use ``base_label`` (``"Net sales"``) so the LLM can match the document
        text directly when the breakdown labels are already unique;
      * fall back to ``"Net sales (Product)"`` only when another row in the
        same statement collapses to the same ``base_label``.

    Returns ``("", "")`` when the raw string is empty or contains nothing but
    a concept reference.
    """
    if not raw:
        return "", ""
    parts = _CONCEPT_PREFIX_RX.split(raw, maxsplit=1)
    head = re.sub(r"\s+", " ", parts[0]).strip()
    if not head:
        return "", ""
    member = ""
    m = _MEMBER_RX.search(raw)
    if m:
        token = re.sub(r"Member$", "", m.group(1))
        member = _CAMEL_SPLIT_RX.sub(" ", token).strip()
    return head, member


def get_statement_concepts(
    cik: str,
    statement_types: list[str] | None = None,
    period_type: str = "quarterly",
) -> list[dict[str, Any]]:
    """Return sorted concept dicts for *cik* and *statement_types*.

    *period_type* selects the source collection:
      - ``"quarterly"`` (default) → ``normalized_concepts_quarterly``
      - ``"annual"``              → ``normalized_concepts_annual``

    Filters out only abstract (``abstract: true``) and hidden (``hide: true``)
    rows.  All other rows — including calculated/system concepts, dimensional
    breakdown rows, and XBRL structural labels — are included.

    Results are sorted by ``path`` so the prompt lists concepts in statement
    order.

    Each returned dict has keys: ``_id`` (str), ``concept`` (GAAP name),
    ``label`` (cleaned, disambiguated only when needed), ``path``,
    ``statement_type``, ``taxonomy_key`` (stable XBRL identity used as the
    JSON key in the extraction prompt and as the mapping key back to
    ``concept_id``).

    Rows whose ``label`` is empty after cleanup are dropped with a debug log.
    When two rows in the same statement collapse to the same base label, the
    axis member qualifier (e.g. ``"(Product)"``) is appended to keep keys
    unique; otherwise the bare base label is used so it matches the
    document text exactly.
    """
    if statement_types is None:
        statement_types = ["income_statement"]
    collection_name = (
        "normalized_concepts_annual"
        if period_type == "annual"
        else "normalized_concepts_quarterly"
    )
    db = _get_client()[_NORMALIZE_DB]
    from earnings_agents.hooks import report_call as _report_call
    _report_call(f"  [db]  query {collection_name}  concepts for CIK {cik}")
    cursor = db[collection_name].find(
        {
            "company_cik": cik,
            "statement_type": {"$in": statement_types},
            "active": {"$ne": False},
            "$or": [
                # Regular concepts: not abstract, not hidden
                {"abstract": {"$ne": True}, "hide": {"$ne": True}},
                # Calculated/system: include regardless of abstract/hide flag
                {"concept": {"$regex": "^system:", "$options": "i"}},
                {"calculated": {"$in": [True, "True", "true"]}},
            ],
        },
        {
            "_id": 1,
            "concept": 1,
            "label": 1,
            "path": 1,
            "statement_type": 1,
        },
    ).sort("path", 1)

    # First pass: collect rows with cleaned labels.
    parsed: list[tuple[dict[str, Any], str, str, str]] = []  # (doc, head, member, member_tag)
    base_counts: dict[tuple[str, str], int] = {}
    for d in cursor:
        concept = d.get("concept", "") or ""
        raw_label = d.get("label", "")
        head, member = _clean_label(raw_label)
        if not head:
            logger.debug(
                "get_statement_concepts: dropping concept with empty label "
                "(cik=%s concept=%s raw_label=%r)",
                cik, concept, raw_label,
            )
            continue
        member_tag = _extract_member_tag(raw_label)
        parsed.append((d, head, member, member_tag))
        key = (d.get("statement_type", ""), head.lower())
        base_counts[key] = base_counts.get(key, 0) + 1

    # Second pass: disambiguate only when a base label collides with another
    # row in the same statement; emit dedup_key to drop exact duplicates.
    out: list[dict[str, Any]] = []
    seen_final: set[tuple[str, str]] = set()
    for d, head, member, member_tag in parsed:
        concept = d.get("concept", "") or ""
        base_key = (d.get("statement_type", ""), head.lower())
        if base_counts[base_key] > 1 and member:
            final_label = f"{head} ({member})"
        else:
            final_label = head
        final_key = (d.get("statement_type", ""), final_label.lower())
        if final_key in seen_final:
            logger.debug(
                "get_statement_concepts: dropping duplicate label %r "
                "(cik=%s concept=%s)",
                final_label, cik, concept,
            )
            continue
        seen_final.add(final_key)
        # Build stable XBRL taxonomy key: base concept alone, or
        # base|member for dimensional rows sharing the same GAAP tag.
        taxonomy_key = f"{concept}|{member_tag}" if member_tag else concept
        out.append(
            {
                "_id": str(d["_id"]),
                "concept": concept,
                "label": final_label,
                "path": d.get("path", ""),
                "statement_type": d.get("statement_type", ""),
                "taxonomy_key": taxonomy_key,
            }
        )
    return out


def get_calculated_concepts(
    cik: str,
    statement_types: list[str] | None = None,
    period_type: str = "quarterly",
) -> list[dict[str, Any]]:
    """Return calculated/system concept dicts for *cik*.

    Mirrors ``get_statement_concepts`` but returns ONLY the rows that function
    excludes — i.e. rows with a ``system:``-prefixed concept name or
    ``calculated: True``.  These represent metrics that the downstream
    normaliser derives; they are not present verbatim in earnings press releases
    but can be computed from extracted values by the derivation engine in
    ``analysis/calculators.py``.

    Each returned dict has the same shape as ``get_statement_concepts`` output
    (``_id``, ``concept``, ``label``, ``path``, ``statement_type``) so it can
    be passed alongside ``target_concepts`` to ``derive_missing_concept_metrics``.
    """
    if statement_types is None:
        statement_types = ["income_statement"]
    collection_name = (
        "normalized_concepts_annual"
        if period_type == "annual"
        else "normalized_concepts_quarterly"
    )
    db = _get_client()[_NORMALIZE_DB]
    from earnings_agents.hooks import report_call as _report_call
    _report_call(f"  [db]  query {collection_name}  calculated concepts for CIK {cik}")
    cursor = db[collection_name].find(
        {
            "company_cik": cik,
            "statement_type": {"$in": statement_types},
            "abstract": {"$ne": True},
            "hide": {"$ne": True},
            "active": {"$ne": False},
            "$or": [
                {"concept": {"$regex": "^system:", "$options": "i"}},
                {"calculated": {"$in": [True, "True", "true"]}},
            ],
        },
        {
            "_id": 1,
            "concept": 1,
            "label": 1,
            "path": 1,
            "statement_type": 1,
        },
    ).sort("path", 1)

    out: list[dict[str, Any]] = []
    for d in cursor:
        raw_label = d.get("label", "")
        head, _ = _clean_label(raw_label)
        if not head:
            logger.debug(
                "get_calculated_concepts: dropping concept with empty label "
                "(cik=%s concept=%s raw_label=%r)",
                cik, d.get("concept", ""), raw_label,
            )
            continue
        out.append(
            {
                "_id": str(d["_id"]),
                "concept": d.get("concept", ""),
                "label": head,
                "path": d.get("path", ""),
                "statement_type": d.get("statement_type", ""),
            }
        )

    logger.debug(
        "get_calculated_concepts: found %d calculated concept(s) for cik=%s (%s)",
        len(out), cik, period_type,
    )
    return out


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

    * Week-based unambiguous values:
      26 w → Q2, 39 w → Q3.  (13 weeks is ambiguous — falls back to date math.)
    * Month-based unambiguous values:
      6 m → Q2, 9 m → Q3.  (3 months is ambiguous — falls back to date math.)
    * Ordinal words: "First Quarter" → Q1, etc.

    ``"Thirteen Weeks"`` / ``"3 Months"`` is **never** inferred as Q1 because
    many companies report non-cumulatively — they print the same duration
    label for every quarter.  The caller must fall back to calendar-month
    math using the company's ``fiscal_year_end_month`` and the period-end date.

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
        # 13 w is NOT inferred as Q1 — just like "3 months", it is
        # ambiguous (non-cumulative reporters print "Thirteen Weeks"
        # for every quarter).  Only cumulative durations are unambiguous.
        if 25 <= count <= 27:
            return 2
        if 38 <= count <= 40:
            return 3
    elif unit == "months":
        # 3 months is ambiguous — not inferred.  6/9 months are cumulative.
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

    *Fiscal year* is determined by comparing the period-end month against
    ``fiscal_year_end_month``.  *Quarter* uses the company's
    ``fiscal_year_end_month`` to divide the fiscal year into 3-month
    blocks — no prediction from duration text (see below).

    When *period_str* carries an unambiguous cumulative duration
    (e.g. "Twenty-Six Weeks" → Q2, "Six Months" → Q2, "Nine Months" → Q3,
    or an ordinal like "Second Quarter"), that value is used.
    "Thirteen Weeks" and "Three Months" are **never** treated as Q1
    because many companies report non-cumulatively — the same label
    appears for every quarter.  The function falls back to calendar-month
    math using *period_end_date* (or the date in *period_str*, which is
    preferred when available).

    The period_str **date** (e.g. "May 25, 2026" from "Three Months Ended
    May 25, 2026") overrides *period_end_date* when both are present and
    differ — the LLM reads the actual column header from the document,
    which is more trustworthy than SEC EDGAR's company-set reportDate on 8-Ks.

    Examples (MSFT, fy_end_month=6):
      - 2026-03-31 → FY2026 Q3
      - 2026-06-30 → FY2026 Q4
      - 2025-09-30 → FY2026 Q1

    Examples (SMPL, fy_end_month=8):
      - "Three Months Ended May 30, 2026"   → FY2026 Q3
      - "Thirteen Weeks Ended May 30, 2026"  → FY2026 Q3
      - "Thirteen Weeks Ended Nov 30, 2025"  → FY2026 Q1
    """
    m = period_end_date.month
    y = period_end_date.year

    # When period_str carries a parseable date (e.g. "Three Months Ended
    # May 25, 2026"), use it for the calendar-month fallback.  The
    # period_end_date argument may come from SEC EDGAR's reportDate which
    # on 8-Ks is company-set and sometimes the announcement/filing date
    # rather than the true fiscal period end — the LLM reads the actual
    # column header from the document, which is more trustworthy.
    str_date = parse_period_end_date(period_str)
    if str_date is not None and str_date != period_end_date:
        logger.debug(
            "compute_fiscal_period: period_str date %s overrides "
            "period_end_date %s for quarter calculation",
            str_date, period_end_date,
        )
        m = str_date.month
        y = str_date.year

    # Fiscal year: if period month falls within the FY ending in fy_end_month
    # of year y (i.e. m <= fy_end_month) → fiscal_year = y; otherwise y + 1.
    fiscal_year = y if m <= fiscal_year_end_month else y + 1

    # Quarter: prefer unambiguous period-string-derived value; fall back to
    # calendar-month math (no guessing — "13 weeks"/"3 months" are ambiguous).
    quarter = _quarter_from_period_str(period_str)
    if quarter is None:
        # Calendar-month fallback — exact for companies whose quarter ends
        # align with calendar month boundaries (the vast majority).
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


# ── Latest period lookup ─────────────────────────────────────────────────────

def get_latest_period(cik: str) -> dict[str, Any] | None:
    """Return the most recently stored period for *cik* across both collections.

    Queries ``concept_values_annual`` and ``concept_values_quarterly`` and
    returns the record with the latest ``reporting_period.end_date``.

    Returned dict has keys:
      ``period_type``  — ``"annual"`` or ``"quarterly"``
      ``fiscal_year``  — int
      ``quarter``      — int | None  (None for annual)
      ``end_date``     — ``datetime`` (UTC)

    Returns ``None`` when no data exists for *cik* in either collection.
    """
    db = _get_client()[_NORMALIZE_DB]
    best: dict[str, Any] | None = None
    best_end: datetime | None = None

    for period_type in ("quarterly", "annual"):
        col = db[f"concept_values_{period_type}"]
        doc = col.find_one(
            {"company_cik": cik},
            {"reporting_period.end_date": 1, "reporting_period.fiscal_year": 1,
             "reporting_period.quarter": 1},
            sort=[("reporting_period.end_date", -1)],
        )
        if doc is None:
            continue
        rp = doc.get("reporting_period", {})
        end_dt: datetime | None = rp.get("end_date")
        if end_dt is None:
            continue
        if best_end is None or end_dt > best_end:
            best_end = end_dt
            best = {
                "period_type": period_type,
                "fiscal_year": rp.get("fiscal_year"),
                "quarter": rp.get("quarter"),  # None for annual
                "end_date": end_dt,
            }

    return best


def fiscal_period_exists(
    cik: str,
    fiscal_year: int,
    quarter: int | None = None,
) -> bool:
    """Return True when concept values already exist for *cik* + *fiscal_year* (+ *quarter*).

    When *quarter* is None the check is against ``concept_values_annual``
    (annual / full-year filings); otherwise ``concept_values_quarterly``.
    """
    db = _get_client()[_NORMALIZE_DB]
    collection_name = (
        "concept_values_annual" if quarter is None else "concept_values_quarterly"
    )
    filt: dict[str, Any] = {
        "company_cik": cik,
        "reporting_period.fiscal_year": fiscal_year,
    }
    if quarter is not None:
        filt["reporting_period.quarter"] = quarter
    return db[collection_name].count_documents(filt, limit=1) > 0


def get_recently_valued_concept_ids(
    cik: str,
    period_type: str = "quarterly",
    n_periods: int = 3,
) -> set[str]:
    """Return concept_id strings that had a value in the last *n_periods* periods.

    Queries the ``concept_values_{annual|quarterly}`` collection for *cik*,
    finds the *n_periods* most recent distinct ``reporting_period.end_date``
    values, and returns the set of ``concept_id`` values (as strings) that had
    at least one stored value in any of those periods.

    Purpose: prune the extraction prompt.  A concept that has not been reported
    in any of the recent periods is very unlikely to appear in the current
    filing, so it is dropped from the LLM prompt (keeping the prompt small and
    the LLM focused).  The concept remains in ``target_concepts`` for mapping
    and derivation, so nothing downstream is affected.

    Returns an **empty set** when no history exists (a brand-new company), which
    the caller must treat as "no filter — use the full concept list" (bootstrap).
    """
    col_name = (
        "concept_values_annual" if period_type == "annual"
        else "concept_values_quarterly"
    )
    db = _get_client()[_NORMALIZE_DB]
    col = db[col_name]
    periods = col.distinct("reporting_period.end_date", {"company_cik": cik})
    periods = sorted([p for p in periods if p is not None], reverse=True)[:n_periods]
    if not periods:
        return set()
    ids = col.distinct(
        "concept_id",
        {"company_cik": cik, "reporting_period.end_date": {"$in": periods}},
    )
    return {str(i) for i in ids if i is not None}


def get_next_period_type(
    cik: str, current_period_end: date | None = None
) -> str | None:
    """Return the period type implied by the company's filing **cadence**.

    A company's filings follow a fixed, deterministic cycle:

        Q1 → Q2 → Q3 → annual (10-K, covers Q4 + full year) → Q1 (next FY) → …

    There is no standalone Q4 release — the **annual 10-K *is* the Q4 report**.
    The most recently stored document already records exactly where the company
    sits in that cycle (its ``period_type`` and ``quarter``), so the *next*
    release type is a pure state-machine transition off that stored state — no
    date arithmetic needed:

      * last was quarter **Q3** → next is **annual** (the year is not yet closed)
      * last was quarter **Q1/Q2** → next is **quarterly**
      * last was **annual** *or* a legacy **Q4** record → the year is already
        closed → next is **quarterly** (Q1 of the next fiscal year)

    The Q4-vs-annual equivalence matters because some companies carry a legacy
    computed ``Q4`` quarterly record (from an older ingestion).  Treating Q4 as
    "annual still pending" would mislabel the next filing (the new fiscal year's
    Q1) as annual.  Only an **exact Q3** maps forward to the annual.

    ``current_period_end`` is the period-end date of the filing being processed
    (the EDGAR-inferred ``sec_report_date``).  It positions the current filing
    against the stored one:

      * **strictly newer** → advance the cycle (return the *next* type above).
      * **same or older** (a re-run or stale 8-K) → do *not* advance; return the
        stored period's own type so re-processing the latest Q3 release stays
        quarterly instead of being mislabelled annual (which would write a
        phantom 10-K alongside the real 10-Q — a cross-collection duplicate).

    Returns ``None`` only when no prior period is stored (first-ever filing or
    the IR path); the caller then falls back to the date-based signal.
    """
    latest = get_latest_period(cik)
    if latest is None:
        return None

    stored_type = latest.get("period_type")

    # Position the current filing against the stored one.  Same-or-older means
    # we are not advancing the cycle — report the stored period's own type.
    if current_period_end is not None:
        latest_end = latest.get("end_date")
        if latest_end is not None:
            stored_end = (
                latest_end.date() if hasattr(latest_end, "date") else latest_end
            )
            if current_period_end <= stored_end:
                return stored_type

    # Strictly newer (or position unknown): advance the cadence state machine.
    # Only an exact Q3 maps to the annual — Q4 is itself the annual (year
    # closed), so a Q4 (or annual) stored period advances to the next FY's Q1.
    if stored_type == "quarterly" and (latest.get("quarter") or 0) == 3:
        return "annual"
    return "quarterly"



# ── Upsert ───────────────────────────────────────────────────────────────────

def upsert_concept_values(
    cik: str,
    company_name: str,
    concept_metrics: dict[str, float],  # concept_id → value
    period_str: str,
    fiscal_year_end_month: int,
    fiscal_year_end_code: str = "1231",
    statement_type: str = "income_statement",
    report_date: date | None = None,
    period_type_override: str | None = None,
    derived_concept_ids: set[str] | None = None,
) -> int:
    """Bulk-upsert concept values into the appropriate collection.

    Routes to ``concept_values_quarterly`` or ``concept_values_annual``.
    Routing precedence (highest first):

    1. *period_type_override* — when the caller already resolved the period
       type upstream (``state["detected_period_type"]`` from
       ``load_company_concepts_node``), it is authoritative.  This keeps the
       prompt's period selection and the save collection in lock-step from a
       single source of truth.
    2. **Fiscal year-end month** — when the resolved period-end month equals
       the company's *fiscal_year_end_month*, the filing is annual.
    3. **Duration keywords** in *period_str* (``detect_period_type``):
      - "Three Months Ended …" / "Thirteen Weeks Ended …" → quarterly
      - "Year Ended …" / "Twelve Months Ended …" / "52/53 Weeks Ended …" → annual

    *report_date*: when provided (sourced from the SEC submissions API
    ``reportDate`` field), it overrides the end date that would otherwise be
    parsed from *period_str* via ``parse_period_end_date``.  This ensures the
    stored ``end_date`` is the exact period-end date declared to the SEC rather
    than an LLM-extracted approximation.  *period_str* is still used for
    duration detection (quarterly vs annual, quarter number, start date).

    Documents are written to match the existing schema used by the SEC-based
    pipeline, with ``concept_id`` stored as ``ObjectId`` and ``end_date`` as
    a native ``datetime`` so the upsert filter correctly de-duplicates
    re-runs of the same earnings release.

    Returns the number of operations submitted (0 on early-exit failures).
    """
    if not concept_metrics:
        logger.debug("upsert_concept_values: empty concept_metrics — nothing to do")
        return 0

    # Resolve the period end date.
    #
    # The LLM's __period__ label (e.g. "Three Months Ended May 31, 2026") is
    # read directly from the document and is consistent across re-runs.  The
    # SEC reportDate on 8-Ks is company-set and may differ between runs
    # (e.g. June 1 vs May 31), causing duplicate documents when the upsert
    # filter uses exact end_date matching.
    #
    # Prefer the LLM's date when both are available — it is the stable source
    # of truth for dedup.  Fall back to SEC reportDate only when the LLM
    # provides no parseable date.
    parsed = parse_period_end_date(period_str)
    if parsed is not None:
        end_date = parsed
    elif report_date is not None:
        end_date = report_date
        logger.debug(
            "upsert_concept_values: using SEC reportDate %s "
            "(period_str %r had no parseable date)",
            report_date, period_str,
        )
    else:
        logger.warning(
            "upsert_concept_values: cannot parse period end date from %r — skipping",
            period_str,
        )
        return 0

    # Annual vs quarterly routing.  An explicit *period_type_override* from the
    # upstream node (``detected_period_type``) is authoritative — it keeps the
    # extraction prompt's column selection and the save collection consistent.
    # Otherwise the company's fiscal year-end month (from the normalize_data
    # ``companies`` collection, e.g. "0430" → April) decides: when the resolved
    # period-end month equals the fiscal year-end month, this is the full-year
    # (annual) filing — regardless of how the LLM labelled ``__period__``.
    # Failing both, fall back to the duration keywords in *period_str*.
    if period_type_override in ("annual", "quarterly"):
        period_type = period_type_override
    elif end_date.month == fiscal_year_end_month:
        period_type = "annual"
    else:
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
        # Annual periods have no quarter dimension and no start_date; quarterly do.
        if period_type == "quarterly":
            period_doc["quarter"] = quarter
            if start_date is not None:
                period_doc["start_date"] = datetime(
                    start_date.year, start_date.month, start_date.day,
                    tzinfo=timezone.utc,
                )

        doc: dict[str, Any] = {
            "concept_id": concept_oid,
            "company_cik": cik,
            "statement_type": statement_type,
            "form_type": form_type,
            "reporting_period": period_doc,
            "value": value,
            "earning_data": True,
            "created_at": now,
            "dimension_value": False,
            "calculated": concept_id_str in (derived_concept_ids or set()),
        }
        filter_doc: dict[str, Any] = {
            "concept_id": concept_oid,
            "reporting_period.end_date": end_datetime,
            "reporting_period.form_type": form_type,
        }
        if period_type == "quarterly":
            filter_doc["reporting_period.quarter"] = quarter

        ops.append(
            UpdateOne(
                filter_doc,
                {"$set": doc},
                upsert=True,
            )
        )

    if not ops:
        return 0

    from earnings_agents.hooks import report_call
    period_label = f"FY{fiscal_year} Q{quarter}" if period_type == "quarterly" else f"FY{fiscal_year}"
    report_call(f"  [db]  upsert {len(ops)} concept(s) → {collection_name}  {period_label}")
    collection.bulk_write(ops, ordered=False)
    report_call(f"  [db]  ✓ upserted {len(ops)} concept(s)")
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
