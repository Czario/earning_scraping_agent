"""Tests for the PDF extractor node."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.extract_pdf_text import extract_pdf_text_node


def _base_state(**overrides):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "discovered_file_url": "https://example.com/q1-2025.pdf",
        "file_type": "pdf",
        "raw_text": None,
        "metrics": None,
        "error": None,
        "status": "fetched",
        **overrides,
    }


def _make_mock_pdfplumber_page(text: str):
    page = MagicMock()
    page.extract_text.return_value = text
    return page


@patch("earnings_agents.nodes.extract_pdf_text.pdfplumber.open")
@patch("earnings_agents.nodes.extract_pdf_text.requests.get")
def test_pdf_extraction_happy_path(mock_get, mock_pdf_open):
    """Extracted text from all pages is joined and stored in state."""
    mock_response = MagicMock()
    mock_response.content = b"%PDF fake bytes"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [
        _make_mock_pdfplumber_page("Revenue: $124.3 billion"),
        _make_mock_pdfplumber_page("EPS: $2.40 diluted"),
    ]
    mock_pdf_open.return_value = mock_pdf

    result = extract_pdf_text_node(_base_state())

    assert result["status"] == "text_extracted"
    assert "Revenue: $124.3 billion" in result["raw_text"]
    assert "EPS: $2.40 diluted" in result["raw_text"]
    assert result["error"] is None


@patch("earnings_agents.nodes.extract_pdf_text.requests.get")
def test_pdf_extraction_fails_on_request_error(mock_get):
    """Node transitions to failed when the PDF download raises an exception."""
    mock_get.side_effect = Exception("Connection refused")

    result = extract_pdf_text_node(_base_state())

    assert result["status"] == "failed"
    assert "PDF extraction failed" in result["error"]


@patch("earnings_agents.nodes.extract_pdf_text.pdfplumber.open")
@patch("earnings_agents.nodes.extract_pdf_text.requests.get")
def test_pdf_extraction_skips_empty_pages(mock_get, mock_pdf_open):
    """Pages with no extractable text are silently skipped."""
    mock_response = MagicMock()
    mock_response.content = b"%PDF fake"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [
        _make_mock_pdfplumber_page(None),  # scanned / image-only page
        _make_mock_pdfplumber_page("Net income: $33.9 billion"),
    ]
    mock_pdf_open.return_value = mock_pdf

    result = extract_pdf_text_node(_base_state())

    assert result["status"] == "text_extracted"
    assert "Net income" in result["raw_text"]
