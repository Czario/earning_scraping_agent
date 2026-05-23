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

_EX99_TYPES = frozenset({"EX-99.1", "EX-99", "EX99.1", "EX-99.01"})


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


def _find_exhibit_99_in_index(cik_int: str, acc: str, acc_nodash: str) -> Optional[str]:
    """Parse the EDGAR HTML filing index to find the Exhibit 99.1 document URL."""
    index_url = _EDGAR_INDEX_HTML.format(cik_int=cik_int, acc_nodash=acc_nodash, acc=acc)
    try:
        resp = _edgar_get(index_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("EDGAR HTML index fetch failed for %s: %s", index_url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

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
                logger.info("Found Exhibit 99.1 for %s/%s: %s", cik_int, acc, href)
                return href

    logger.info("No Exhibit 99.1 found in index for %s/%s", cik_int, acc)
    return None


def normalize_cik(cik: str) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return str(int(cik)).zfill(10)


def get_latest_earnings_url(cik: str) -> Optional[str]:
    """Return the URL of the most recent earnings press release (Exhibit 99.1)
    from an 8-K Item 2.02 filing for the given CIK.

    Falls back to the most recent 8-K primary document if no Exhibit 99.1 is found.
    Returns ``None`` if no 8-K filing is available.
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
        return None

    recent = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    items_list: list[str] = recent.get("items", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])

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
        return None

    acc = accessions[target_idx]       # e.g. "0000320193-26-000011"
    acc_nodash = acc.replace("-", "")  # e.g. "000032019326000011"

    # ── 3. Parse HTML filing index to find Exhibit 99.1 ──────────────────────
    ex99_url = _find_exhibit_99_in_index(cik_int, acc, acc_nodash)
    if ex99_url:
        return ex99_url

    # ── 4. Last resort: primary document from submissions metadata ────────────
    primary_doc = primary_docs[target_idx] if target_idx < len(primary_docs) else ""
    if primary_doc:
        url = f"{_EDGAR_ARCHIVES_BASE.format(cik_int=cik_int, acc_nodash=acc_nodash)}/{primary_doc}"
        logger.info("EDGAR primary doc fallback for CIK %s: %s", cik_padded, url)
        return url

    logger.warning("Could not resolve document URL for CIK %s accession %s", cik_padded, acc)
    return None

