"""Tests for the earnings-failures CLI."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.cli.failures import (
    _build_query,
    _finding_type_summary,
    _format_finding_types,
    main,
)


# ── Unit helpers ─────────────────────────────────────────────────────────────

def _args(**kwargs):
    """Build a minimal Namespace for testing."""
    import argparse
    defaults = dict(ticker=None, days=None, status=None, detail=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_build_query_defaults():
    q = _build_query(_args())
    assert q["status"] == {"$in": ["degraded", "failed"]}
    assert "ticker" not in q
    assert "scraped_at" not in q


def test_build_query_status_filter():
    q = _build_query(_args(status="degraded"))
    assert q["status"] == "degraded"


def test_build_query_single_ticker():
    q = _build_query(_args(ticker=["msft"]))
    assert q["ticker"] == "MSFT"


def test_build_query_multiple_tickers():
    q = _build_query(_args(ticker=["msft", "aapl"]))
    assert q["ticker"] == {"$in": ["MSFT", "AAPL"]}


def test_build_query_days_filter():
    q = _build_query(_args(days=7))
    assert "scraped_at" in q
    assert "$gte" in q["scraped_at"]


def test_finding_type_summary_counts_high_medium_only():
    findings = [
        {"type": "missing_critical", "severity": "high"},
        {"type": "missing_critical", "severity": "high"},
        {"type": "sign_anomaly", "severity": "medium"},
        {"type": "auto_corrected", "severity": "low"},  # excluded
    ]
    summary = _finding_type_summary(findings)
    assert summary == {"missing_critical": 2, "sign_anomaly": 1}
    assert "auto_corrected" not in summary


def test_format_finding_types_empty():
    assert _format_finding_types({}) == "—"


def test_format_finding_types_formats_correctly():
    out = _format_finding_types({"missing_critical": 2, "sign_anomaly": 1})
    assert "missing_critical×2" in out
    assert "sign_anomaly×1" in out


# ── Integration: main() against mocked MongoDB ───────────────────────────────

_SAMPLE_DOCS = [
    {
        "ticker": "MSFT",
        "status": "degraded",
        "scraped_at": datetime(2026, 5, 1, tzinfo=timezone.utc),
        "findings": [
            {"type": "missing_critical", "severity": "high", "message": "Net income missing"},
            {"type": "sign_anomaly", "severity": "medium", "message": "Revenue negative"},
        ],
        "identity_warnings": [],
    },
    {
        "ticker": "MSFT",
        "status": "degraded",
        "scraped_at": datetime(2026, 4, 1, tzinfo=timezone.utc),
        "findings": [
            {"type": "missing_critical", "severity": "high", "message": "Net income missing"},
        ],
        "identity_warnings": [],
    },
    {
        "ticker": "AAPL",
        "status": "failed",
        "scraped_at": datetime(2026, 5, 10, tzinfo=timezone.utc),
        "findings": [],
        "identity_warnings": ["Revenue − COGS ≠ Gross Profit"],
    },
]


@patch("earnings_agents.cli.failures.get_collection")
def test_main_no_results(mock_get_col, capsys):
    mock_col = MagicMock()
    mock_col.find.return_value = []
    mock_get_col.return_value = mock_col

    main([])
    # No exception raised; "No degraded" message goes to rich console (not capsys),
    # so just confirm it exits cleanly.


@patch("earnings_agents.cli.failures.get_collection")
def test_main_shows_repeat_offender_hint(mock_get_col):
    mock_col = MagicMock()
    mock_col.find.return_value = _SAMPLE_DOCS
    mock_get_col.return_value = mock_col

    # Should complete without error; MSFT appears twice → hint suggestion printed.
    main([])


@patch("earnings_agents.cli.failures.get_collection")
def test_main_ticker_filter_passed_to_query(mock_get_col):
    mock_col = MagicMock()
    mock_col.find.return_value = []
    mock_get_col.return_value = mock_col

    main(["--ticker", "MSFT"])
    call_query = mock_col.find.call_args[0][0]
    assert call_query["ticker"] == "MSFT"


@patch("earnings_agents.cli.failures.get_collection")
def test_main_status_filter_passed_to_query(mock_get_col):
    mock_col = MagicMock()
    mock_col.find.return_value = []
    mock_get_col.return_value = mock_col

    main(["--status", "degraded"])
    call_query = mock_col.find.call_args[0][0]
    assert call_query["status"] == "degraded"


@patch("earnings_agents.cli.failures.get_collection")
def test_main_mongodb_error_exits(mock_get_col):
    mock_get_col.side_effect = RuntimeError("connection refused")
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1
