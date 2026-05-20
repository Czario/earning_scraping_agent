"""Tests for the data extraction node (LLM → flexible metrics dict)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.extract_financial_metrics import _merge_metrics, extract_financial_metrics_node


def _base_state(**overrides):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "ir_url": "https://investor.apple.com/",
        "discovered_file_url": "https://example.com/q1-2025.pdf",
        "file_type": "pdf",
        "raw_text": (
            "Apple reports quarterly earnings.\n"
            "Revenue: $124.3 billion. Net income: $33.9 billion. "
            "Diluted EPS: $2.40.\n"
            "Guidance: Revenue of $89-93 billion expected for Q2 FY2025."
        ),
        "metrics": None,
        "error": None,
        "status": "text_extracted",
        **overrides,
    }


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_returns_company_labels(mock_llm_cls):
    """LLM response using company-specific labels is stored as-is."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"Diluted EPS": 2.40, "Total Net Revenue": 124300000000, '
        '"Net Income": 33900000000, "Gross Margin": 57900000000, '
        '"Revenue Growth YoY": 4.0, '
        '"Guidance Revenue Low": 89000000000, "Guidance Revenue High": 93000000000}'
    )
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "extracted"
    assert result["metrics"]["Diluted EPS"] == pytest.approx(2.40)
    assert result["metrics"]["Total Net Revenue"] == pytest.approx(124_300_000_000)
    assert result["metrics"]["Net Income"] == pytest.approx(33_900_000_000)
    assert result["metrics"]["Revenue Growth YoY"] == pytest.approx(4.0)
    assert result["error"] is None


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_drops_null_fields(mock_llm_cls):
    """Null values are stripped from the final metrics dict."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"Diluted EPS": 1.85, "Total Revenue": null, "Net Income": null}'
    )
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "extracted"
    assert result["metrics"]["Diluted EPS"] == pytest.approx(1.85)
    assert "Total Revenue" not in result["metrics"]
    assert "Net Income" not in result["metrics"]


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_fails_on_bad_json(mock_llm_cls):
    """Node transitions to failed when the LLM returns non-JSON output."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot extract the data from this document."
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "failed"
    assert "chunk" in result["error"].lower()


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_strips_markdown_fences(mock_llm_cls):
    """Markdown code fences around the JSON response are handled gracefully."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        "```json\n"
        '{"Diluted EPS": 3.10, "Total Revenue": 211900000000}\n'
        "```"
    )
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "extracted"
    assert result["metrics"]["Diluted EPS"] == pytest.approx(3.10)


def test_extraction_fails_when_no_raw_text():
    """Node transitions to failed immediately when raw_text is absent."""
    result = extract_financial_metrics_node(_base_state(raw_text=None))

    assert result["status"] == "failed"
    assert "No raw text" in result["error"]


def test_merge_first_nonnull_wins():
    """Later non-null values fill in fields left null by earlier chunks."""
    chunks = [
        {"Revenue": None, "Net Income": 5000000000},
        {"Revenue": 20000000000, "Net Income": None},
    ]
    merged = _merge_metrics(chunks)
    assert merged["Revenue"] == pytest.approx(20_000_000_000)
    assert merged["Net Income"] == pytest.approx(5_000_000_000)


def test_merge_discards_implausible_dollar_values():
    """Values implausibly small compared to revenue are dropped."""
    chunks = [
        {"Total Revenue": 80000000000, "Net Income": 500},   # 500 = unscaled
    ]
    merged = _merge_metrics(chunks)
    assert merged.get("Net Income") is None
    assert merged["Total Revenue"] == pytest.approx(80_000_000_000)


def test_scale_millions_applied_by_python():
    """__scale__:millions causes Python to multiply raw table values × 1_000_000."""
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    raw = '{"__scale__": "millions", "Total Revenue": 82886, "Net Income": 31778, "Diluted EPS": 4.27}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["Total Revenue"] == pytest.approx(82_886_000_000)
    assert result["Net Income"] == pytest.approx(31_778_000_000)
    assert result["Diluted EPS"] == pytest.approx(4.27)   # EPS not multiplied


def test_scale_as_is_no_multiplication():
    """__scale__:as-is leaves full-USD narrative values untouched."""
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    raw = '{"__scale__": "as-is", "Total Revenue": 82886000000, "Diluted EPS": 4.27}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["Total Revenue"] == pytest.approx(82_886_000_000)
    assert result["Diluted EPS"] == pytest.approx(4.27)



def _base_state(**overrides):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "ir_url": "https://investor.apple.com/",
        "discovered_file_url": "https://example.com/q1-2025.pdf",
        "file_type": "pdf",
        "raw_text": (
            "Apple reports quarterly earnings.\n"
            "Revenue: $124.3 billion. Net income: $33.9 billion. "
            "Diluted EPS: $2.40.\n"
            "Guidance: Revenue of $89-93 billion expected for Q2 FY2025."
        ),
        "metrics": None,
        "error": None,
        "status": "text_extracted",
        **overrides,
    }


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_parses_all_fields(mock_llm_cls):
    """All metrics fields are correctly parsed; __scale__:as-is passes full-USD values through."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"__scale__": "as-is", "Diluted EPS": 2.40, "Total Net Revenue": 124300000000, '
        '"Net Income": 33900000000, "Operating Income": 46580000000, '
        '"Gross Margin %": 46.9, "Operating Margin %": 34.5, '
        '"Revenue Growth YoY": 4.0, '
        '"Guidance Revenue Low": 89000000000, "Guidance Revenue High": 93000000000, '
        '"Guidance EPS Low": 2.20, "Guidance EPS High": 2.35, '
        '"Guidance": "Revenue of $89-93B for Q2 FY2025"}'
    )
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "extracted"
    assert result["metrics"]["Diluted EPS"] == pytest.approx(2.40)
    assert result["metrics"]["Total Net Revenue"] == pytest.approx(124_300_000_000)
    assert result["metrics"]["Net Income"] == pytest.approx(33_900_000_000)
    assert result["metrics"]["Operating Income"] == pytest.approx(46_580_000_000)
    assert result["metrics"]["Gross Margin %"] == pytest.approx(46.9)
    assert result["metrics"]["Operating Margin %"] == pytest.approx(34.5)
    assert result["metrics"]["Revenue Growth YoY"] == pytest.approx(4.0)
    assert result["metrics"]["Guidance Revenue Low"] == pytest.approx(89_000_000_000)
    assert result["metrics"]["Guidance Revenue High"] == pytest.approx(93_000_000_000)
    assert result["metrics"]["Guidance EPS Low"] == pytest.approx(2.20)
    assert result["metrics"]["Guidance EPS High"] == pytest.approx(2.35)
    assert "Q2 FY2025" in result["metrics"]["Guidance"]
    assert result["error"] is None


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_handles_null_fields(mock_llm_cls):
    """Null-valued fields are stripped; only the non-null value is kept."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"__scale__": "as-is", "Diluted EPS": 1.85, "Total Revenue": null, "Net Income": null}'
    )
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "extracted"
    assert result["metrics"]["Diluted EPS"] == pytest.approx(1.85)
    # Null-valued keys must be absent from the returned dict
    assert "Total Revenue" not in result["metrics"]
    assert "Net Income" not in result["metrics"]


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_fails_on_bad_json(mock_llm_cls):
    """Node transitions to failed when the LLM returns non-JSON output."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot extract the data from this document."
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "failed"
    assert "chunk" in result["error"].lower()


@patch("earnings_agents.nodes.extract_financial_metrics.OllamaLLM")
def test_extraction_strips_markdown_fences(mock_llm_cls):
    """Markdown code fences around the JSON response are handled gracefully."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        "```json\n"
        '{"__scale__": "as-is", "Diluted EPS": 3.10, "Total Revenue": 211900000000}\n'
        "```"
    )
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "extracted"
    assert result["metrics"]["Diluted EPS"] == pytest.approx(3.10)


def test_extraction_fails_when_no_raw_text():
    """Node transitions to failed immediately when raw_text is absent."""
    result = extract_financial_metrics_node(_base_state(raw_text=None))

    assert result["status"] == "failed"
    assert "No raw text" in result["error"]
