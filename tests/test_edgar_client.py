"""Tests for edgar_client.py — SEC EDGAR 8-K filing URL resolution."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.tools.edgar_client import get_latest_earnings_url, normalize_cik


def test_normalize_cik():
    assert normalize_cik("320193") == "0000320193"
    assert normalize_cik("0000320193") == "0000320193"


def _mock_submissions(
    forms, items, accessions, primary_docs, report_dates=None, filing_dates=None
):
    return {
        "filings": {
            "recent": {
                "form": forms,
                "items": items,
                "accessionNumber": accessions,
                "primaryDocument": primary_docs,
                "reportDate": report_dates or [""] * len(forms),
                "filingDate": filing_dates or [""] * len(forms),
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
        report_dates=["2026-03-29", ""],
    )

    index_resp = MagicMock()
    index_resp.raise_for_status = MagicMock()
    index_resp.text = _INDEX_HTML_WITH_EX99

    mock_get.side_effect = [submissions_resp, index_resp]

    url, report_date = get_latest_earnings_url("0000320193")

    assert url is not None
    assert "ex991pressrelease.htm" in url
    assert url.startswith("https://www.sec.gov")
    assert report_date == "2026-03-29"


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

    url, report_date = get_latest_earnings_url("0000789019")
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

    url, report_date = get_latest_earnings_url("0001234567")
    assert url is None
    assert report_date is None


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_returns_none_on_submissions_api_error(mock_get):
    """Returns None when the EDGAR API call fails."""
    import requests as req
    mock_get.side_effect = req.RequestException("timeout")

    url, report_date = get_latest_earnings_url("0000320193")
    assert url is None
    assert report_date is None


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_falls_back_to_primary_doc_when_index_fails(mock_get):
    """When the HTML index fetch fails on every retry, uses primaryDocument from submissions."""
    import requests as req
    from earnings_agents.tools.edgar_client import _EDGAR_MAX_RETRIES

    submissions_resp = MagicMock()
    submissions_resp.raise_for_status = MagicMock()
    submissions_resp.json.return_value = _mock_submissions(
        forms=["8-K"],
        items=["2.02,9.01"],
        accessions=["0000320193-26-000011"],
        primary_docs=["aapl-20260430.htm"],
        report_dates=["2026-03-29"],
    )

    # Provide enough RequestExceptions to exhaust all retry attempts, then the
    # caller (_find_exhibit_99_in_index) catches the final raised exception and
    # falls back to the primary document.
    index_errors = [req.RequestException("index not found")] * (_EDGAR_MAX_RETRIES + 1)

    with patch("earnings_agents.tools.edgar_client._time.sleep", return_value=None):
        mock_get.side_effect = [submissions_resp, *index_errors]
        url, report_date = get_latest_earnings_url("0000320193")

    assert url is not None
    assert "aapl-20260430.htm" in url
    assert report_date == "2026-03-29"


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_uses_prior_year_10q_for_period_end(mock_get):
    """When the 8-K reportDate is the earnings announcement date (not the fiscal
    quarter end), the prior-year same-quarter 10-Q is used to project the
    correct period end date one year forward.

    Scenario mirrors NVIDIA Q1 FY2027:
      8-K  filingDate=2026-05-28  reportDate=2026-05-20  (announcement date — wrong)
      10-Q filingDate=2025-05-29  reportDate=2025-04-27  (prior-year Q1 — correct end)
    Expected period end: 2025-04-27 + 1 year = 2026-04-27
    """
    submissions_resp = MagicMock()
    submissions_resp.raise_for_status = MagicMock()
    submissions_resp.json.return_value = _mock_submissions(
        forms=["8-K", "10-Q"],
        items=["2.02,9.01", ""],
        accessions=["0001045810-26-000051", "0001045810-25-000030"],
        primary_docs=["q1fy27pr.htm", "q1fy26.htm"],
        report_dates=["2026-05-20", "2025-04-27"],
        filing_dates=["2026-05-28", "2025-05-29"],
    )

    index_resp = MagicMock()
    index_resp.raise_for_status = MagicMock()
    index_resp.text = _INDEX_HTML_WITH_EX99

    mock_get.side_effect = [submissions_resp, index_resp]

    url, report_date = get_latest_earnings_url("0001045810")

    assert url is not None
    # Prior-year 10-Q reportDate 2025-04-27 + 1 year → 2026-04-27 (actual quarter end)
    assert report_date == "2026-04-27"


@patch("earnings_agents.tools.edgar_client.requests.get")
def test_falls_back_to_raw_report_date_when_no_prior_year_10q(mock_get):
    """When no matching prior-year 10-Q exists, the raw EDGAR reportDate is used."""
    submissions_resp = MagicMock()
    submissions_resp.raise_for_status = MagicMock()
    submissions_resp.json.return_value = _mock_submissions(
        forms=["8-K"],
        items=["2.02,9.01"],
        accessions=["0000320193-26-000011"],
        primary_docs=["aapl-20260430.htm"],
        report_dates=["2026-03-29"],
        filing_dates=["2026-05-01"],
    )

    index_resp = MagicMock()
    index_resp.raise_for_status = MagicMock()
    index_resp.text = _INDEX_HTML_WITH_EX99

    mock_get.side_effect = [submissions_resp, index_resp]

    url, report_date = get_latest_earnings_url("0000320193")

    assert url is not None
    # No prior-year 10-Q in mock — raw reportDate returned unchanged
    assert report_date == "2026-03-29"


# ---------------------------------------------------------------------------
# _edgar_get — retry behaviour
# ---------------------------------------------------------------------------
# The rate-limiter token bucket uses _time.sleep and _time.monotonic internally.
# Patching _EDGAR_RATE_LIMITER.acquire to a no-op keeps these tests fast and
# deterministic — rate-limiting is already covered by the TokenBucket unit.

@patch("earnings_agents.tools.edgar_client._EDGAR_RATE_LIMITER")
@patch("earnings_agents.tools.edgar_client.requests.get")
@patch("earnings_agents.tools.edgar_client._time.sleep", return_value=None)
def test_edgar_get_retries_on_503(mock_sleep, mock_get, mock_limiter):
    """_edgar_get retries on HTTP 503 and eventually returns the successful response."""
    from earnings_agents.tools.edgar_client import _edgar_get

    fail_resp = MagicMock()
    fail_resp.status_code = 503

    ok_resp = MagicMock()
    ok_resp.status_code = 200

    mock_get.side_effect = [fail_resp, ok_resp]

    resp = _edgar_get("https://data.sec.gov/submissions/CIK0000320193.json", timeout=10)

    assert resp.status_code == 200
    assert mock_get.call_count == 2
    assert mock_sleep.call_count == 1   # one delay between attempts


@patch("earnings_agents.tools.edgar_client._EDGAR_RATE_LIMITER")
@patch("earnings_agents.tools.edgar_client.requests.get")
@patch("earnings_agents.tools.edgar_client._time.sleep", return_value=None)
def test_edgar_get_exhausts_retries_and_returns_last_error_response(mock_sleep, mock_get, mock_limiter):
    """After MAX_RETRIES all fail with 503, _edgar_get returns the final response."""
    from earnings_agents.tools.edgar_client import _edgar_get, _EDGAR_MAX_RETRIES

    fail_resp = MagicMock()
    fail_resp.status_code = 503
    mock_get.return_value = fail_resp

    resp = _edgar_get("https://data.sec.gov/test", timeout=10)

    assert resp.status_code == 503
    assert mock_get.call_count == _EDGAR_MAX_RETRIES + 1


@patch("earnings_agents.tools.edgar_client._EDGAR_RATE_LIMITER")
@patch("earnings_agents.tools.edgar_client.requests.get")
@patch("earnings_agents.tools.edgar_client._time.sleep", return_value=None)
def test_edgar_get_retries_on_connection_error_then_succeeds(mock_sleep, mock_get, mock_limiter):
    """A transient RequestException is retried; success on the second attempt."""
    import requests as req
    from earnings_agents.tools.edgar_client import _edgar_get

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    mock_get.side_effect = [req.RequestException("timeout"), ok_resp]

    resp = _edgar_get("https://data.sec.gov/test", timeout=10)

    assert resp.status_code == 200
    assert mock_sleep.call_count == 1

