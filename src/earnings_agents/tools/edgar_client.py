"""SEC EDGAR client — finds the most recent earnings press release for any public company.

Uses the EDGAR Submissions API (no API key required) to locate the latest 8-K
with Item 2.02 (Results of Operations), then retrieves the filing index to find
the Exhibit 99.1 press release document URL.

EDGAR rate-limit guideline: ≤10 requests/second.
"""
from __future__ import annotations

import logging
import os
import threading as _th
import time as _time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"
_EDGAR_INDEX_HTML = _EDGAR_ARCHIVES_BASE + "/{acc}-index.htm"

# SEC requires a descriptive User-Agent with contact info for automated access
_HEADERS = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}

_EX99_TYPES = frozenset({
    "EX-99.1", "EX-99", "EX99.1", "EX-99.01",
    "EX-99.2", "EX99.2", "EX-99.02",
    "EX-99.3", "EX99.3", "EX-99.03",
})


class _TokenBucket:
    """Thread-safe token-bucket rate limiter."""

    __slots__ = ("_rate", "_tokens", "_last", "_lock")

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate
        self._last = _time.monotonic()
        self._lock = _th.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = _time.monotonic()
                self._tokens = min(
                    self._rate,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            _time.sleep(1.0 / self._rate)


# ≤ 8 req/s against sec.gov (SEC guideline: ≤ 10 req/s per user-agent)
_EDGAR_RATE_LIMITER = _TokenBucket(rate=float(os.getenv("EDGAR_RATE_LIMIT", "8")))

# HTTP status codes that warrant a retry (transient server-side errors).
_EDGAR_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_EDGAR_MAX_RETRIES: int = 3
_EDGAR_RETRY_BASE_DELAY: float = 1.0  # seconds; doubles on each retry


def _edgar_get(url: str, **kwargs) -> requests.Response:
    """Rate-limited GET with exponential-backoff retry for transient EDGAR errors.

    Retries up to ``_EDGAR_MAX_RETRIES`` times on HTTP 429/5xx or connection
    errors, re-acquiring the rate-limit token before each attempt.
    """
    last_exc: Exception | None = None
    for attempt in range(_EDGAR_MAX_RETRIES + 1):
        _EDGAR_RATE_LIMITER.acquire()
        try:
            resp = requests.get(url, headers=_HEADERS, **kwargs)
            if resp.status_code not in _EDGAR_RETRY_STATUSES or attempt == _EDGAR_MAX_RETRIES:
                return resp
            delay = _EDGAR_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "EDGAR %s returned HTTP %d (attempt %d/%d); retrying in %.1f s",
                url, resp.status_code, attempt + 1, _EDGAR_MAX_RETRIES + 1, delay,
            )
            _time.sleep(delay)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == _EDGAR_MAX_RETRIES:
                raise
            delay = _EDGAR_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "EDGAR %s connection error (attempt %d/%d): %s; retrying in %.1f s",
                url, attempt + 1, _EDGAR_MAX_RETRIES + 1, exc, delay,
            )
            _time.sleep(delay)
    # Unreachable in normal operation; satisfies the type checker.
    raise requests.RequestException(f"_edgar_get exhausted retries for {url}") from last_exc


def _find_all_ex_99_urls(cik_int: str, acc: str, acc_nodash: str) -> list[str]:
    """Parse the EDGAR HTML filing index to find ALL EX-99 exhibit document URLs.

    Returns a list of URLs in filing-index order. The first entry is typically
    EX-99.1 (the main earnings press release), and subsequent entries are
    supplemental financial exhibits (EX-99.2, EX-99.3, etc.).

    Returns an empty list when no exhibits are found.
    """
    index_url = _EDGAR_INDEX_HTML.format(cik_int=cik_int, acc_nodash=acc_nodash, acc=acc)
    try:
        resp = _edgar_get(index_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("EDGAR HTML index fetch failed for %s: %s", index_url, exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    urls: list[str] = []

    # Filing index table: columns are Seq | Description | Document | Type | Size
    for row in soup.select("table.tableFile tr, table tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        # Type is in the 4th column (index 3); link is in the 3rd column (index 2)
        doc_type = cells[3].get_text(strip=True).upper()
        if doc_type in _EX99_TYPES:
            link_tag = cells[2].find("a", href=True)
            if link_tag:
                href: str = link_tag["href"]
                if href.startswith("/"):
                    href = f"https://www.sec.gov{href}"
                urls.append(href)
                logger.info("Found exhibit %s for %s/%s: %s", doc_type, cik_int, acc, href)

    logger.info(
        "Found %d EX-99 exhibit(s) in index for %s/%s",
        len(urls), cik_int, acc,
    )
    return urls


def normalize_cik(cik: str) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return str(int(cik)).zfill(10)


def _infer_period_end(
    recent: dict,
    filing_date_str: str,
    raw_report_date: str | None,
) -> str | None:
    """Return the fiscal period-end date for an earnings 8-K.

    EDGAR ``reportDate`` on 8-K filings is company-set and may be the
    announcement date rather than the actual fiscal period end.

    **Step 1 — prior-year projection (most reliable):** find the 10-Q or 10-K
    whose ``filingDate`` is closest to ``filing_date - 1 year`` and project its
    ``reportDate`` forward one year.  10-Q/10-K reportDates are always the
    exact fiscal period end (SEC-mandated), so this gives the correct date
    even when the 8-K's raw reportDate is the announcement date.

    **Step 2 — validate raw date (fallback):** when no matching prior-year
    filing is found, check whether *raw_report_date* falls between the
    most-recent 10-Q/10-K ``reportDate`` and the 8-K ``filingDate``.  A true
    period end must be after the prior quarter end and before the filing that
    announces it.  When it passes we accept it; when it fails the date is
    almost certainly the announcement date and we return ``None`` so the
    caller can fall back to the LLM-extracted ``__period__`` label.

    Returns ``"YYYY-MM-DD"`` or ``None``.
    """
    from datetime import date as _d, timedelta

    if not filing_date_str:
        return None
    try:
        filing_date = _d.fromisoformat(filing_date_str)
    except ValueError:
        return None

    forms: list[str] = recent.get("form", [])
    filing_dates: list[str] = recent.get("filingDate", [])
    report_dates: list[str] = recent.get("reportDate", [])

    # ── Step 1: prior-year projection ──────────────────────────────────────
    prior_year_center = filing_date.replace(year=filing_date.year - 1)
    window = timedelta(days=90)  # wide enough for late filers

    best_rd: str | None = None
    best_delta = timedelta.max

    for i, form in enumerate(forms):
        if form not in ("10-Q", "10-K"):
            continue
        fd_str = filing_dates[i] if i < len(filing_dates) else ""
        rd_str = report_dates[i] if i < len(report_dates) else ""
        if not fd_str or not rd_str:
            continue
        try:
            fd = _d.fromisoformat(fd_str)
        except ValueError:
            continue
        delta = abs(fd - prior_year_center)
        if delta <= window and delta < best_delta:
            best_delta = delta
            best_rd = rd_str

    if best_rd is not None:
        try:
            prior_end = _d.fromisoformat(best_rd)
            inferred = prior_end.replace(year=prior_end.year + 1)
        except (ValueError, OverflowError):
            pass
        else:
            days_before = (filing_date - inferred).days
            if 1 <= days_before <= 100:
                logger.debug(
                    "_infer_period_end: prior-year projection %s "
                    "(from %s reportDate %s, %d days before filing)",
                    inferred.isoformat(), best_rd[:4], prior_end.isoformat(),
                    days_before,
                )
                return inferred.isoformat()

    # ── Step 2: validate raw reportDate against most-recent 10-Q/10-K ──────
    if not raw_report_date:
        return None
    try:
        raw_rd = _d.fromisoformat(raw_report_date)
    except ValueError:
        return None

    # Find the most-recent 10-Q or 10-K whose reportDate is before this 8-K filing.
    best_prior_rd: _d | None = None
    for i, form in enumerate(forms):
        if form not in ("10-Q", "10-K"):
            continue
        rd_str = report_dates[i] if i < len(report_dates) else ""
        if not rd_str:
            continue
        try:
            rd = _d.fromisoformat(rd_str)
        except ValueError:
            continue
        if rd < filing_date and (best_prior_rd is None or rd > best_prior_rd):
            best_prior_rd = rd

    # A true period end must be after the prior quarter end AND strictly
    # before the 8-K filing date.  ``<`` (not ``<=``) because when
    # reportDate == filingDate the value is the announcement/event date.
    if best_prior_rd is not None and best_prior_rd < raw_rd < filing_date:
        logger.debug(
            "_infer_period_end: raw reportDate %s accepted "
            "(between prior period end %s and filing %s)",
            raw_report_date, best_prior_rd.isoformat(),
            filing_date.isoformat(),
        )
        return raw_report_date

    if best_prior_rd is None:
        # No prior periodic filing to compare against (new IPO, etc.).
        # Return the raw date as-is — the skip guard needs a date to work
        # with, and the upsert now prefers the LLM's __period__ date anyway.
        logger.debug(
            "_infer_period_end: no prior 10-Q/10-K — "
            "keeping raw reportDate %s (unvalidated)",
            raw_report_date,
        )
        return raw_report_date

    # Validation failed: raw date falls outside the expected window between
    # the prior quarter end and the 8-K filing date.  It is almost certainly
    # the announcement date rather than the period end.
    #
    # Two heuristics to estimate the true period end:
    #
    # (a) When the most-recent 10-Q/10-K was filed within 60 days of the 8-K
    #     (common — companies often file both on the same day), its reportDate
    #     is the correct period end for the 8-K as well.
    # (b) Otherwise, estimate as filing_date − 30 days — the median delay
    #     between quarter end and earnings announcement.
    #
    # These estimates feed the skip guard's FY+quarter comparison only; the
    # upsert prefers the LLM's exact __period__ date, so an approximate date
    # here never corrupts stored data.
    if best_prior_rd is not None:
        days_since_period_end = (filing_date - best_prior_rd).days
        if days_since_period_end <= 60:
            # 10-Q/10-K filed same day or within 60 days of 8-K — its
            # reportDate is the same period the 8-K reports on.
            logger.debug(
                "_infer_period_end: using most-recent 10-Q/10-K reportDate %s "
                "(%d days before 8-K filing) as estimated period end",
                best_prior_rd.isoformat(), days_since_period_end,
            )
            return best_prior_rd.isoformat()

    estimated = filing_date - timedelta(days=30)
    logger.debug(
        "_infer_period_end: raw reportDate %s outside expected range "
        "(prior period end=%s, filing=%s) — using estimated period end %s",
        raw_report_date, best_prior_rd.isoformat() if best_prior_rd else "none",
        filing_date.isoformat(), estimated.isoformat(),
    )
    return estimated.isoformat()


def _infer_8k_fiscal_period(
    recent: dict,
    filing_date_str: str | None,
    fiscal_year_end_month: int | None,
) -> tuple[int, int, str] | None:
    """Return ``(fiscal_year, quarter, period_type)`` for the 8-K filing.

    Uses the most-recent 10-Q/10-K in *recent* to determine which fiscal
    period the 8-K reports on.  This is the **same data the DB stores**
    (``reporting_period.fiscal_year``, ``reporting_period.quarter``), so the
    caller can directly check ``concept_values_quarterly`` or
    ``concept_values_annual``.

    Heuristic:
      1. Find the most-recent 10-Q or 10-K whose ``reportDate`` is before
         the 8-K's ``filingDate``.
      2. If that 10-Q/10-K was filed within 60 days of the 8-K, the 8-K
         reports on the **same** fiscal period (company filed both together).
      3. Otherwise, the 8-K reports on the **next** fiscal period after
         that 10-Q/10-K's period.

    Returns ``None`` when the period cannot be determined (missing data).
    """
    from datetime import date as _d, timedelta
    from earnings_agents.tools.normalize_data_client import compute_fiscal_period

    if not filing_date_str or not fiscal_year_end_month:
        return None
    try:
        filing_date = _d.fromisoformat(filing_date_str)
    except ValueError:
        return None

    forms: list[str] = recent.get("form", [])
    report_dates: list[str] = recent.get("reportDate", [])
    filing_dates: list[str] = recent.get("filingDate", [])

    # Find the most-recent 10-Q/10-K whose reportDate is before the 8-K filing.
    best_rd: _d | None = None
    best_fd: _d | None = None
    best_form: str = ""
    for i, form in enumerate(forms):
        if form not in ("10-Q", "10-K"):
            continue
        rd_str = report_dates[i] if i < len(report_dates) else ""
        fd_str = filing_dates[i] if i < len(filing_dates) else ""
        if not rd_str:
            continue
        try:
            rd = _d.fromisoformat(rd_str)
        except ValueError:
            continue
        if rd < filing_date and (best_rd is None or rd > best_rd):
            best_rd = rd
            best_form = form
            try:
                best_fd = _d.fromisoformat(fd_str) if fd_str else None
            except ValueError:
                best_fd = None

    if best_rd is None:
        return None

    # Compute the fiscal period of the most-recent 10-Q/10-K.
    ref_fy, ref_q = compute_fiscal_period(best_rd, fiscal_year_end_month)
    ref_period_type = "annual" if best_form == "10-K" else "quarterly"

    # If the 10-Q/10-K was filed within 60 days of the 8-K, the 8-K is for
    # the SAME period (company filed both together — common).  Otherwise it's
    # for the NEXT period after the 10-Q/10-K.
    days_apart = abs((filing_date - best_fd).days) if best_fd else 999
    same_period = best_fd is not None and days_apart <= 60

    if same_period:
        logger.debug(
            "_infer_8k_fiscal_period: %s filed %d days from 8-K "
            "→ same period FY%d %s",
            best_form, days_apart, ref_fy,
            f"Q{ref_q}" if ref_period_type == "quarterly" else "(annual)",
        )
        return (ref_fy, ref_q, ref_period_type)
    else:
        # 8-K is for the NEXT period after the 10-Q/10-K.
        if ref_q is not None and ref_q < 4:
            next_q = ref_q + 1
            next_fy = ref_fy
        else:
            # Q4 or annual → next period is Q1 of next FY
            next_q = 1
            next_fy = ref_fy + 1
        period_type = "quarterly"  # earnings 8-Ks are always quarterly
        logger.debug(
            "_infer_8k_fiscal_period: %s filed %d days from 8-K "
            "→ next period FY%d Q%d",
            best_form, days_apart, next_fy, next_q,
        )
        return (next_fy, next_q, period_type)


def get_latest_earnings_url(cik: str) -> tuple[Optional[str], list[str], Optional[str]]:
    """Return ``(filing_url, supplemental_urls, report_date)`` for the most recent
    earnings press release.

    ``filing_url`` is the URL to the primary earnings release (Exhibit 99.1).
    ``supplemental_urls`` is a list of additional exhibit URLs (EX-99.2, EX-99.3,
    etc.) that contain supplemental financial data.
    ``report_date`` is the EDGAR ``reportDate`` field (``"YYYY-MM-DD"`` string)
    — the period-end date for the filing as declared to the SEC.  It is the
    authoritative source for the reporting period end date and should be
    preferred over the LLM-extracted ``__period__`` label.

    Falls back to the most recent 8-K primary document if no Exhibit 99.1 is found.
    Returns ``(None, [], None)`` if no 8-K filing is available.
    """
    cik_padded = normalize_cik(cik)
    cik_int = str(int(cik_padded))  # no leading zeros for archive paths

    # ── 1. Fetch submissions ─────────────────────────────────────────────────
    sub_url = _EDGAR_SUBMISSIONS.format(cik=cik_padded)
    try:
        resp = _edgar_get(sub_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("EDGAR submissions fetch failed for CIK %s: %s", cik_padded, exc)
        return None, [], None

    recent = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    items_list: list[str] = recent.get("items", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])
    report_dates: list[str] = recent.get("reportDate", [])
    filing_dates: list[str] = recent.get("filingDate", [])

    # ── 2. Find latest 8-K with Item 2.02 (earnings results) ─────────────────
    target_idx: Optional[int] = None
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        item_str = items_list[i] if i < len(items_list) else ""
        if "2.02" in item_str:
            target_idx = i
            break

    # Fallback: use first available 8-K of any item type
    if target_idx is None:
        for i, form in enumerate(forms):
            if form == "8-K":
                target_idx = i
                logger.info(
                    "No Item 2.02 8-K found for CIK %s — using first available 8-K",
                    cik_padded,
                )
                break

    if target_idx is None:
        logger.warning("No 8-K filings found for CIK %s", cik_padded)
        return None, [], None

    acc = accessions[target_idx]       # e.g. "0000320193-26-000011"
    acc_nodash = acc.replace("-", "")  # e.g. "000032019326000011"
    report_date: Optional[str] = (
        report_dates[target_idx] if target_idx < len(report_dates) else None
    )
    filing_date_str: Optional[str] = (
        filing_dates[target_idx] if target_idx < len(filing_dates) else None
    )

    # 8-K ``reportDate`` may be the announcement date rather than the fiscal
    # quarter end (company-specific behaviour, e.g. NVIDIA, LEVI).  Correct it
    # using the same submissions API response we already fetched — zero extra
    # HTTP calls.  Falls back to the LLM-extracted __period__ label when
    # correction is impossible.
    report_date = _infer_period_end(recent, filing_date_str, report_date)

    # ── 3. Parse HTML filing index to find ALL EX-99 exhibits ────────────────
    ex99_urls = _find_all_ex_99_urls(cik_int, acc, acc_nodash)

    if ex99_urls:
        primary_url = ex99_urls[0]
        supplemental_urls = ex99_urls[1:]
        if supplemental_urls:
            logger.info(
                "Found %d supplemental exhibit(s) for CIK %s: %s",
                len(supplemental_urls), cik_padded, supplemental_urls,
            )
        return primary_url, supplemental_urls, report_date

    # ── 4. Last resort: primary document from submissions metadata ────────────
    primary_doc = primary_docs[target_idx] if target_idx < len(primary_docs) else ""
    if primary_doc:
        url = f"{_EDGAR_ARCHIVES_BASE.format(cik_int=cik_int, acc_nodash=acc_nodash)}/{primary_doc}"
        logger.info("EDGAR primary doc fallback for CIK %s: %s", cik_padded, url)
        return url, [], report_date

    logger.warning("Could not resolve document URL for CIK %s accession %s", cik_padded, acc)
    return None, [], None


def get_next_8k_status(
    cik: str,
    last_stored_report_date_str: str,
) -> dict:
    """Check SEC EDGAR for an 8-K filed *after* *last_stored_report_date_str*.

    Fetches the EDGAR submissions JSON (one rate-limited request) and scans
    8-K Item\u202002.02 filings.

    Returns a dict with:
      ``available``       \u2014 True if a newer earnings 8-K is already on SEC
      ``sec_report_date`` \u2014 period-end date of that filing (``"YYYY-MM-DD"`` or None)
      ``filing_url``      \u2014 Exhibit\u202099.1 URL (or primary-doc fallback, or None)
      ``estimated_date``  \u2014 projected filing date if not yet available (or None)
    """
    from datetime import date as _d, timedelta
    import calendar

    _empty: dict = {"available": False, "sec_report_date": None,
                    "filing_url": None, "estimated_date": None}

    cik_padded = normalize_cik(cik)
    cik_int = str(int(cik_padded))

    try:
        resp = _edgar_get(_EDGAR_SUBMISSIONS.format(cik=cik_padded), timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("get_next_8k_status: submissions fetch failed CIK %s: %s", cik, exc)
        return _empty

    recent = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    items_list: list[str] = recent.get("items", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])
    report_dates: list[str] = recent.get("reportDate", [])
    filing_dates: list[str] = recent.get("filingDate", [])

    try:
        last_stored = _d.fromisoformat(last_stored_report_date_str)
    except (ValueError, TypeError):
        return _empty

    # Capture the most recent 8-K report_date from any 8-K — used as a
    # fallback to detect the "already stored" case even when the company
    # doesn't tag its earnings 8-K as Item 2.02 in EDGAR submissions.
    latest_any_8k_report_date: Optional[str] = None
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        rd_str = report_dates[i] if i < len(report_dates) else ""
        if rd_str:
            latest_any_8k_report_date = rd_str
            break

    # Collect 8-K Item 2.02 filings in chronological-desc order (as EDGAR returns them)
    earnings_8ks: list[dict] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        item_str = items_list[i] if i < len(items_list) else ""
        if "2.02" not in item_str:
            continue
        rd_str = report_dates[i] if i < len(report_dates) else ""
        fd_str = filing_dates[i] if i < len(filing_dates) else ""
        earnings_8ks.append({
            "report_date": rd_str,
            "filing_date": fd_str,
            "accession": accessions[i] if i < len(accessions) else "",
            "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
        })

    # ── Next expected period end ≈ last_stored + 3 months ────────────────────
    # EDGAR report_date on an 8-K Item 2.02 is the *earnings announcement* date,
    # not the period end date. The announcement for the already-stored period
    # arrives ~14-45 days after that period ends — well before the next period
    # ends. Requiring rd >= expected_next_period_end - 30 days filters out the
    # current-period announcement and only matches the next quarter's filing.
    try:
        nm = last_stored.month + 3
        ny = last_stored.year + (nm - 1) // 12
        nm = (nm - 1) % 12 + 1
        last_day = calendar.monthrange(ny, nm)[1]
        expected_next_period_end = _d(ny, nm, min(last_stored.day, last_day))
    except Exception:
        return _empty

    min_next_report_date = expected_next_period_end - timedelta(days=30)

    latest_edgar_report_date: Optional[str] = None
    for filing in earnings_8ks:  # most-recent first
        rd_str = filing["report_date"]
        if not rd_str:
            continue
        try:
            rd = _d.fromisoformat(rd_str)
        except ValueError:
            continue
        # Capture the most recent 8-K report date regardless of whether it
        # clears the threshold — used to detect the "already stored" case.
        latest_edgar_report_date = rd_str
        if rd >= min_next_report_date:
            acc = filing["accession"]
            acc_nodash = acc.replace("-", "")
            ex99_urls = _find_all_ex_99_urls(cik_int, acc, acc_nodash)
            url = ex99_urls[0] if ex99_urls else None
            if not url:
                pdoc = filing["primary_doc"]
                if pdoc:
                    url = (
                        f"{_EDGAR_ARCHIVES_BASE.format(cik_int=cik_int, acc_nodash=acc_nodash)}"
                        f"/{pdoc}"
                    )
            return {
                "available": True,
                "sec_report_date": rd_str,
                "filing_url": url,
                "latest_edgar_report_date": rd_str,
            }
        break  # first entry is most recent; once we've seen it, stop

    return {
        "available": False,
        "sec_report_date": None,
        "filing_url": None,
        # Prefer the Item 2.02 date; fall back to latest any-8-K date so
        # companies that don't tag Item 2.02 still trigger the "already stored"
        # detection in the coverage display.
        "latest_edgar_report_date": latest_edgar_report_date or latest_any_8k_report_date,
    }

