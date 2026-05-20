from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _tickers_path() -> Path:
    # tickers.py lives at src/earnings_agents/tickers.py
    # three .parent steps -> project root (earning_scrapping_agents/)
    root = Path(__file__).parent.parent.parent
    preferred = root / "data" / "reference" / "tickers.json"
    legacy = root / "tickers.json"
    if preferred.exists():
        return preferred
    return legacy


@lru_cache(maxsize=1)
def _load() -> dict:
    path = _tickers_path()
    if not path.exists():
        raise FileNotFoundError(f"tickers.json not found at {path}")
    with open(path) as f:
        return json.load(f)


def normalize_cik(cik: str) -> str:
    """Return a zero-padded 10-digit CIK string."""
    return str(int(cik)).zfill(10)


def lookup_by_cik(cik: str) -> Optional[dict]:
    """Look up a company by CIK number.

    Returns::

        {"cik": "0000320193", "ticker": "AAPL", "company_name": "Apple Inc."}

    or ``None`` if the CIK is not in the file.
    """
    data = _load()
    cik_norm = normalize_cik(cik)
    company_name = data["cik_to_company_name"].get(cik_norm)
    if not company_name:
        logger.warning("CIK %s not found in tickers.json", cik_norm)
        return None
    ticker = data["cik_to_ticker"].get(cik_norm)
    return {"cik": cik_norm, "ticker": ticker, "company_name": company_name}


def lookup_by_ticker(ticker: str) -> Optional[dict]:
    """Look up a company by ticker symbol.

    Returns the same dict shape as :func:`lookup_by_cik`, or ``None``.
    """
    data = _load()
    cik = data["ticker_to_cik"].get(ticker.upper())
    if not cik:
        logger.warning("Ticker %s not found in tickers.json", ticker.upper())
        return None
    company_name = data["cik_to_company_name"].get(cik, ticker.upper())
    return {"cik": cik, "ticker": ticker.upper(), "company_name": company_name}
