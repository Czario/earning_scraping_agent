"""Tests for tickers.py CIK/ticker lookup."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from earnings_agents.company_registry import lookup_by_cik, lookup_by_ticker, normalize_cik

_MOCK_DATA = {
    "ticker_to_cik": {"AAPL": "0000320193", "MSFT": "0000789019"},
    "cik_to_ticker": {"0000320193": "AAPL", "0000789019": "MSFT"},
    "cik_to_company_name": {
        "0000320193": "Apple Inc.",
        "0000789019": "MICROSOFT CORP",
    },
    "last_updated": "2025-06-18T09:44:39",
}


@pytest.fixture(autouse=True)
def mock_tickers_data():
    with patch("earnings_agents.company_registry._load", return_value=_MOCK_DATA):
        yield


def test_normalize_cik_pads_to_10_digits():
    assert normalize_cik("320193") == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"
    assert normalize_cik("789019") == "0000789019"


def test_lookup_by_cik_found():
    result = lookup_by_cik("0000320193")
    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["company_name"] == "Apple Inc."
    assert result["cik"] == "0000320193"


def test_lookup_by_cik_accepts_unpadded():
    result = lookup_by_cik("320193")
    assert result is not None
    assert result["ticker"] == "AAPL"


def test_lookup_by_cik_not_found():
    result = lookup_by_cik("9999999999")
    assert result is None


def test_lookup_by_ticker_found():
    result = lookup_by_ticker("MSFT")
    assert result is not None
    assert result["cik"] == "0000789019"
    assert result["company_name"] == "MICROSOFT CORP"


def test_lookup_by_ticker_case_insensitive():
    result = lookup_by_ticker("aapl")
    assert result is not None
    assert result["ticker"] == "AAPL"


def test_lookup_by_ticker_not_found():
    result = lookup_by_ticker("ZZZZ")
    assert result is None
