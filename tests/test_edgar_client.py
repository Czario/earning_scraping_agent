"""Tests for edgar_client.py — SEC EDGAR 8-K filing URL resolution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.tools.edgar_client import get_latest_earnings_url, normalize_cik


def test_normalize_cik():
    assert normalize_cik("320193") == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"


def _mock_submissions(forms, items, accessions, primary_docs):
    return {
        "filings": {
            "recent": {
                "form": forms,
                "items": items,
                "accessionNumber": accessions,
                "primaryDocument": primary_docs,
            }
        }
    }


# Minimal EDGAR filing index HTML with an Exhibit 99.1 row
_INDEX_HTML_WITH_EX99 = """
<html><body>
<table class="tableFile">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr>
    <td>1</td><td>FORM 8-K</td>
    <td><a href="/Archives/edgar/data/320193/000032019326000011/aapl-20260430.htm">aapl-20260430.htm</a></td>
    <td>8-K</td><td>12KB</td>
  </tr>
  <tr>
    <td>2</td><td>PRESS RELEASE</td>
    <td><a href="/Archives/edgar/data/320193/000032019326000011/ex991pressrelease.htm">ex991pressrelease.htm</a></td>
    <td>EX-99.1</td><td>45KB</td>
  </tr>
</table>
</body></html>
"""

# Index HTML with no Exhibit 99.1
_INDEX_HTML_NO_EX99 = """
<html><body>
<table class="tableFile">
  <tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
  <tr>
    <td>1</td><td>FORM 8-K</td>
    <td><a href="/Archives/edgar/data/789019/000078901926000002/msft-8k.htm">msft-8k.htm</a></td>
    <td>8-K</td><td>8KB</td>
  </tr>
</table>
</body></html>
"""


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_finds_exhibit_99_from_8k_item_202(mock_get):
    """Returns Exhibit 99.1 URL from the latest 8-K with Item 2.02."""
    submissions_resp = MagicMock()
    submissions_resp.raise_for_status = MagicMock()
    submissions_resp.json.return_value = _mock_submissions(
        forms=["8-K", "10-Q"],
        items=["2.02,9.01", ""],
        accessions=["0000320193-26-000011", "0000320193-25-000010"],
        primary_docs=["aapl-20260430.htm", "aapl-10q.htm"],
    )

    index_resp = MagicMock()
    index_resp.raise_for_status = MagicMock()
    index_resp.text = _INDEX_HTML_WITH_EX99

    mock_get.side_effect = [submissions_resp, index_resp]

    url = get_latest_earnings_url("0000320193")

    assert url is not None
    assert "ex991pressrelease.htm" in url
    assert url.startswith("https://www.sec.gov")


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_falls_back_to_first_8k_when_no_item_202(mock_get):
    """When no Item 2.02 exists, uses the first available 8-K."""
    submissions_resp = MagicMock()
    submissions_resp.raise_for_status = MagicMock()
    submissions_resp.json.return_value = _mock_submissions(
        forms=["10-Q", "8-K"],
        items=["", "8.01,9.01"],
        accessions=["0000789019-25-000001", "0000789019-25-000002"],
        primary_docs=["msft-10q.htm", "msft-8k.htm"],
    )

    index_resp = MagicMock()
    index_resp.raise_for_status = MagicMock()
    index_resp.text = _INDEX_HTML_NO_EX99

    mock_get.side_effect = [submissions_resp, index_resp]

    url = get_latest_earnings_url("0000789019")
    # Falls back to primary doc since no EX-99.1 in index
    assert url is not None
    assert "msft-8k.htm" in url


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_returns_none_when_no_8k_filings(mock_get):
    """Returns None when a company has no 8-K filings at all."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = _mock_submissions(
        forms=["10-K", "10-Q"],
        items=["", ""],
        accessions=["0001234567-25-000001", "0001234567-25-000002"],
        primary_docs=["doc1.htm", "doc2.htm"],
    )
    mock_get.return_value = resp

    url = get_latest_earnings_url("0001234567")
    assert url is None


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_returns_none_on_submissions_api_error(mock_get):
    """Returns None when the EDGAR API call fails."""
    import requests as req
    mock_get.side_effect = req.RequestException("timeout")

    url = get_latest_earnings_url("0000320193")
    assert url is None


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_falls_back_to_primary_doc_when_index_fails(mock_get):
    """When the HTML index fetch fails, uses primaryDocument from submissions."""
    import requests as req

    submissions_resp = MagicMock()
    submissions_resp.raise_for_status = MagicMock()
    submissions_resp.json.return_value = _mock_submissions(
        forms=["8-K"],
        items=["2.02,9.01"],
        accessions=["0000320193-26-000011"],
        primary_docs=["aapl-20260430.htm"],
    )

    mock_get.side_effect = [
        submissions_resp,
        req.RequestException("index not found"),
    ]

    url = get_latest_earnings_url("0000320193")
    assert url is not None
    assert "aapl-20260430.htm" in url

