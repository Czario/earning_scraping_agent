"""Tests for the IR discovery node."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.discover_earnings_release import discover_earnings_release_node


def _base_state(**overrides):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "ir_url": "https://investor.apple.com/press-releases/default.aspx",
        "discovered_file_url": None,
        "file_type": None,
        "raw_text": None,
        "metrics": None,
        "error": None,
        "status": "pending",
        **overrides,
    }


@patch("earnings_agents.nodes.discover_earnings_release.fetch_page_js")
@patch("earnings_agents.nodes.discover_earnings_release.fetch_page")
@patch("earnings_agents.nodes.discover_earnings_release.OllamaLLM")
def test_discovery_succeeds_on_static_fetch(mock_llm_cls, mock_fetch, mock_fetch_js):
    """Happy path: static fetch returns HTML with links; LLM returns a valid URL."""
    # HTML is padded to exceed the 500-char threshold so Playwright is not triggered.
    padding = " " * 500
    mock_fetch.return_value = (
        f'<html><body>{padding}'
        '<a href="https://investor.apple.com/press-releases/detail/fy2025q1.pdf">'
        'Q1 FY2025 Earnings Press Release</a>'
        '</body></html>',
        True,
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"url": "https://investor.apple.com/press-releases/detail/fy2025q1.pdf",'
        ' "reason": "Most recent quarterly press release"}'
    )
    mock_llm_cls.return_value = mock_llm

    result = discover_earnings_release_node(_base_state())

    assert result["status"] == "discovered"
    assert result["discovered_file_url"] == (
        "https://investor.apple.com/press-releases/detail/fy2025q1.pdf"
    )
    assert result["error"] is None


@patch("earnings_agents.nodes.discover_earnings_release.fetch_page_js")
@patch("earnings_agents.nodes.discover_earnings_release.fetch_page")
@patch("earnings_agents.nodes.discover_earnings_release.OllamaLLM")
def test_discovery_falls_back_to_playwright(mock_llm_cls, mock_fetch, mock_fetch_js):
    """When static fetch returns too little content, Playwright is used."""
    mock_fetch.return_value = ("<html></html>", True)  # too short → Playwright
    mock_fetch_js.return_value = (
        '<html><body>'
        '<a href="https://abc.xyz/earnings/q1-2025.html">Q1 2025 Earnings</a>'
        '</body></html>'
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"url": "https://abc.xyz/earnings/q1-2025.html", "reason": "Earnings page"}'
    )
    mock_llm_cls.return_value = mock_llm

    state = _base_state(
        ticker="GOOGL",
        company_name="Alphabet Inc.",
        ir_url="https://abc.xyz/investor/",
    )
    result = discover_earnings_release_node(state)

    mock_fetch_js.assert_called_once()
    assert result["status"] == "discovered"


@patch("earnings_agents.nodes.discover_earnings_release.fetch_page")
def test_discovery_fails_when_page_unreachable(mock_fetch):
    """Node transitions to failed when neither fetch method returns content."""
    mock_fetch.return_value = ("", False)

    with patch("earnings_agents.nodes.discover_earnings_release.fetch_page_js", return_value=""):
        result = discover_earnings_release_node(_base_state())

    assert result["status"] == "failed"
    assert "Could not fetch IR page" in result["error"]


@patch("earnings_agents.nodes.discover_earnings_release.fetch_page_js")
@patch("earnings_agents.nodes.discover_earnings_release.fetch_page")
@patch("earnings_agents.nodes.discover_earnings_release.OllamaLLM")
def test_discovery_fails_on_bad_llm_json(mock_llm_cls, mock_fetch, mock_fetch_js):
    """Node transitions to failed when the LLM returns malformed JSON."""
    padding = " " * 500
    mock_fetch.return_value = (
        f'<html><body>{padding}<a href="/q1.pdf">Q1 Earnings</a></body></html>',
        True,
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot determine which link is the earnings release."
    mock_llm_cls.return_value = mock_llm

    result = discover_earnings_release_node(_base_state())

    assert result["status"] == "failed"
    assert "LLM IR discovery parsing failed" in result["error"]
