"""Company lookup by ticker or CIK using the SEC's company_tickers.json.

Fetches and caches the full SEC company registry (~10 000+ entries) on first
use.  The cache is a disk-backed JSON file at ``data/reference/sec_company_tickers.json``
with a 24-hour TTL so the pipeline never stalls on a cold cache.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from earnings_agents.config import HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_HEADERS = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}
_CACHE_TTL_SECONDS = 86_400  # 24 hours


def _cache_path() -> Path:
    """Return the path to the cached SEC company registry JSON file."""
    root = Path(__file__).parent.parent.parent
    ref_dir = root / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    return ref_dir / "sec_company_tickers.json"


def _is_cache_fresh(path: Path) -> bool:
    """Return True when the cache file exists and is younger than TTL."""
    try:
        return (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS
    except FileNotFoundError:
        return False


def _fetch_from_sec() -> dict:
    """Download the SEC company tickers JSON and return the parsed dict.

    Raises ``requests.RequestException`` on failure.
    """
    logger.info("Fetching SEC company tickers from %s", _SEC_COMPANY_TICKERS_URL)
    resp = requests.get(
        _SEC_COMPANY_TICKERS_URL,
        headers=_SEC_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ── In-memory lookup structures built once from the raw SEC data ──────────────
# Protected by _lock so concurrent threads building the index don't race.
_lock = threading.Lock()
_index: dict[str, dict] | None = None  # built on first use


def _build_index() -> dict[str, dict]:
    """Return {ticker→info, cik→info} built from SEC company_tickers.json.

    The returned dict has three top-level keys:

    ``"ticker_to_cik"``
        Maps uppercase ticker → zero-padded 10-digit CIK string.
    ``"cik_to_ticker"``
        Maps zero-padded CIK → uppercase ticker.
    ``"cik_to_company_name"``
        Maps zero-padded CIK → company title from SEC.
    """
    global _index
    if _index is not None:
        return _index

    with _lock:
        if _index is not None:  # double-check inside lock
            return _index

        cache_path = _cache_path()

        # ── Disk cache ──────────────────────────────────────────────────────
        if _is_cache_fresh(cache_path):
            logger.debug("Loading SEC company registry from disk cache")
            with open(cache_path) as fh:
                raw_entries: dict = json.load(fh)
        else:
            raw_entries = _fetch_from_sec()
            # Persist to disk so a cold-start / network failure doesn't
            # block the next run.
            try:
                with open(cache_path, "w") as fh:
                    json.dump(raw_entries, fh)
                logger.debug(
                    "Cached %d SEC company entries to %s",
                    len(raw_entries), cache_path,
                )
            except OSError:
                pass  # best-effort; in-memory index still works

        # ── Build lookup maps ────────────────────────────────────────────────
        ticker_to_cik: dict[str, str] = {}
        cik_to_ticker: dict[str, str] = {}
        cik_to_company_name: dict[str, str] = {}

        for entry in raw_entries.values():
            cik_int: int = entry.get("cik_str", 0)
            ticker: str = (entry.get("ticker") or "").strip().upper()
            title: str = (entry.get("title") or "").strip()
            if not cik_int or not ticker:
                continue
            cik_padded = str(cik_int).zfill(10)

            # When two entries share the same ticker (e.g. old vs new CIK),
            # the last one in iteration wins — SEC data is sorted with newest
            # entries last, so the most recent CIK prevails.
            ticker_to_cik[ticker] = cik_padded
            cik_to_ticker[cik_padded] = ticker
            cik_to_company_name[cik_padded] = title

        _index = {
            "ticker_to_cik": ticker_to_cik,
            "cik_to_ticker": cik_to_ticker,
            "cik_to_company_name": cik_to_company_name,
        }
        logger.info(
            "SEC company registry loaded: %d tickers, %d CIKs",
            len(ticker_to_cik), len(cik_to_ticker),
        )
        return _index


def normalize_cik(cik: str) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return str(int(cik)).zfill(10)


def lookup_by_cik(cik: str) -> Optional[dict]:
    """Look up a company by CIK number using the SEC company registry.

    Returns::

        {"cik": "0000320193", "ticker": "AAPL", "company_name": "Apple Inc."}

    or ``None`` if the CIK is not in the SEC registry.
    """
    index = _build_index()
    cik_norm = normalize_cik(cik)
    company_name = index["cik_to_company_name"].get(cik_norm)
    if not company_name:
        logger.warning("CIK %s not found in SEC company registry", cik_norm)
        return None
    ticker = index["cik_to_ticker"].get(cik_norm)
    return {"cik": cik_norm, "ticker": ticker, "company_name": company_name}


def lookup_by_ticker(ticker: str) -> Optional[dict]:
    """Look up a company by ticker symbol using the SEC company registry.

    Returns the same dict shape as :func:`lookup_by_cik`, or ``None``.
    """
    index = _build_index()
    cik = index["ticker_to_cik"].get(ticker.upper())
    if not cik:
        logger.warning("Ticker %s not found in SEC company registry", ticker.upper())
        return None
    company_name = index["cik_to_company_name"].get(cik, ticker.upper())
    return {"cik": cik, "ticker": ticker.upper(), "company_name": company_name}
