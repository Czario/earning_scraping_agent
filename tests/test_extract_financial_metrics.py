"""Tests for the data extraction node (LLM → flexible metrics dict)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.extract_financial_metrics import (
    _chunk_text,
    _merge_metrics,
    extract_financial_metrics_node,
)


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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
def test_extraction_fails_on_bad_json(mock_llm_cls):
    """Node transitions to failed when the LLM returns non-JSON output."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot extract the data from this document."
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "failed"
    assert "chunk" in result["error"].lower()


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
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


def test_merge_null_chunks_do_not_block_real_values():
    """Null in one chunk does not prevent a real value in another chunk from being kept."""
    chunks = [
        {"Revenue": None, "Net Income": 5000000000},
        {"Revenue": 20000000000, "Net Income": None},
    ]
    merged = _merge_metrics(chunks)
    assert merged["Revenue"] == pytest.approx(20_000_000_000)
    assert merged["Net Income"] == pytest.approx(5_000_000_000)


def test_merge_median_resists_outlier_chunk():
    """When one chunk returns an unscaled outlier, the median picks the correct value."""
    # Two chunks agree on 80 B; one rogue chunk returns 80 000 (unscaled millions).
    chunks = [
        {"Total Revenue": 80_000_000_000},
        {"Total Revenue": 80_000_000_000},
        {"Total Revenue": 80_000},          # rogue: forgot to scale
    ]
    merged = _merge_metrics(chunks)
    # Median of [80_000, 80_000_000_000, 80_000_000_000] = 80_000_000_000
    assert merged["Total Revenue"] == pytest.approx(80_000_000_000)


def test_merge_median_even_number_of_chunks():
    """Median with an even number of chunks averages the two middle values."""
    chunks = [
        {"Net Income": 30_000_000_000},
        {"Net Income": 32_000_000_000},
    ]
    merged = _merge_metrics(chunks)
    # median of [30B, 32B] = 31B
    assert merged["Net Income"] == pytest.approx(31_000_000_000)


def test_merge_discards_implausible_dollar_values():
    """Values implausibly small compared to revenue are dropped."""
    chunks = [
        {"Total Revenue": 80000000000, "Net Income": 500},   # 500 = unscaled
    ]
    merged = _merge_metrics(chunks)
    # Key preserved as-is; tiny Net Income discarded by plausibility check.
    assert merged.get("Net Income") is None
    assert merged["Total Revenue"] == pytest.approx(80_000_000_000)


def test_merge_preserves_synonym_variants_for_llm_cleanup():
    """Synonym/case duplicates are NOT folded by the merge step.

    Duplicate folding now happens in cleanup_metrics_node (the constrained
    LLM pass), which can apply context-aware judgement instead of relying
    on a hard-coded synonym table.
    """
    chunks = [
        {
            "Revenue": 82_886_000_000,
            "Total revenue": 82_886_000_000,
            "Operating Income": 38_398_000_000,
            "Operating income": 38_398_000_000,
        },
    ]
    merged = _merge_metrics(chunks)
    # All four keys survive the merge — cleanup_metrics_node will drop the
    # duplicates afterwards.
    assert merged["Revenue"] == pytest.approx(82_886_000_000)
    assert merged["Total revenue"] == pytest.approx(82_886_000_000)
    assert merged["Operating Income"] == pytest.approx(38_398_000_000)
    assert merged["Operating income"] == pytest.approx(38_398_000_000)


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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
def test_extraction_fails_on_bad_json(mock_llm_cls):
    """Node transitions to failed when the LLM returns non-JSON output."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "I cannot extract the data from this document."
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(_base_state())

    assert result["status"] == "failed"
    assert "chunk" in result["error"].lower()


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
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


def test_validate_metrics_passes_consistent_income_statement():
    """A self-consistent income statement produces no identity warnings."""
    from earnings_agents.nodes.extract_financial_metrics import _validate_metrics

    metrics = {
        "Revenue": 82_886_000_000,
        "Cost of revenue": 26_828_000_000,
        "Gross margin": 56_058_000_000,
        "Research and development": 8_915_000_000,
        "Sales and marketing": 6_814_000_000,
        "General and administrative": 1_931_000_000,
        "Operating income": 38_398_000_000,
        "Other income (expense), net": 942_000_000,
        "Income before income taxes": 39_340_000_000,
        "Provision for income taxes": 7_562_000_000,
        "Net income": 31_778_000_000,
        "Diluted Earnings per Share": 4.27,
        "Weighted average shares outstanding: Diluted": 7_445_000_000,
    }
    cleaned, warnings = _validate_metrics(metrics)
    assert warnings == []
    assert cleaned["Net income"] == 31_778_000_000


def test_validate_metrics_flags_broken_identity():
    """A bad Net income that breaks the EPS × shares sanity check is blocking."""
    from earnings_agents.nodes.extract_financial_metrics import _validate_metrics

    # EPS 4.27 × Diluted shares 7.445B ≈ $31.8B Net income.
    # Asserting a Net income of $12.3B trips the universal EPS sanity check.
    metrics = {
        "Net income": 12_345_000_000,
        "Diluted Earnings per Share": 4.27,
        "Weighted average shares outstanding: Diluted": 7_445_000_000,
    }
    _, warnings = _validate_metrics(metrics)
    assert warnings, "expected EPS sanity warning for broken Net income"
    assert any("EPS" in w for w in warnings)


def test_validate_metrics_advisory_only_for_structural_drift():
    """Structural decomposition drift (e.g. Net income = Pre-tax − Tax) is advisory only."""
    from earnings_agents.nodes.extract_financial_metrics import _validate_metrics

    # Pre-tax 39.34B − Tax 7.56B = 31.78B, but reported Net income is 12.3B.
    # Without EPS / shares to cross-check, this is logged as advisory only.
    metrics = {
        "Income before income taxes": 39_340_000_000,
        "Provision for income taxes": 7_562_000_000,
        "Net income": 12_345_000_000,
    }
    _, warnings = _validate_metrics(metrics)
    assert warnings == [], "structural decomposition mismatches must not block"


def test_save_gate_blocks_when_identity_warnings_and_strict():
    """mongodb_save_node refuses to upsert when identity warnings exist and STRICT is on."""
    import earnings_agents.workflow as wf

    state = {
        "ticker": "TEST",
        "company_name": "Test Co",
        "discovered_file_url": "https://example.com",
        "file_type": "html",
        "metrics": {"Revenue": 1},
        "identity_warnings": ["Net income drift 50.00%"],
        "status": "extracted",
    }
    original = wf.STRICT_ACCURACY
    wf.STRICT_ACCURACY = True
    try:
        result = wf.mongodb_save_node(state)
    finally:
        wf.STRICT_ACCURACY = original
    assert result["status"] == "failed"
    assert "identity" in result["error"].lower()


def test_save_gate_saves_with_warnings_when_lenient(monkeypatch):
    """When STRICT_ACCURACY is False the document is upserted with warnings attached."""
    import earnings_agents.workflow as wf

    captured: dict = {}

    def _fake_upsert(doc):
        captured["doc"] = doc

    monkeypatch.setattr(wf, "upsert_earnings", _fake_upsert)
    state = {
        "ticker": "TEST",
        "company_name": "Test Co",
        "discovered_file_url": "https://example.com",
        "file_type": "html",
        "metrics": {"Revenue": 1},
        "identity_warnings": ["Gross margin drift 5%"],
        "status": "extracted",
    }
    original = wf.STRICT_ACCURACY
    wf.STRICT_ACCURACY = False
    try:
        result = wf.mongodb_save_node(state)
    finally:
        wf.STRICT_ACCURACY = original
    assert result["status"] == "saved"
    assert captured["doc"]["identity_warnings"] == ["Gross margin drift 5%"]


# ---------------------------------------------------------------------------
# _chunk_text — line-boundary snapping
# ---------------------------------------------------------------------------

def test_chunk_text_short_text_returned_as_single_chunk():
    """Text shorter than chunk_size is returned as-is."""
    text = "Revenue: $1B\nNet income: $500M\n"
    result = _chunk_text(text, chunk_size=500, overlap=50)
    assert result == [text]


def test_chunk_text_does_not_split_mid_line():
    """Every chunk boundary coincides with a newline, not mid-row."""
    # Build a text where each line is 40 chars long, total > chunk_size.
    line = "Revenue:  $82,886,000,000 | Q1 FY2026  \n"  # 40 chars incl \n
    text = line * 30  # 1200 chars total
    chunks = _chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:  # last chunk may not end with \n if at EOF
        assert chunk.endswith("\n"), f"Chunk does not end at a newline: {chunk[-20:]!r}"


def test_chunk_text_covers_all_content():
    """Union of all chunk content covers the original text without gaps."""
    line = "Operating income: $38,398,000,000 Q1 FY2026\n"
    text = line * 25
    chunks = _chunk_text(text, chunk_size=300, overlap=60)
    # Reconstruct by ensuring each char appears in at least one chunk.
    # Simplest check: first chunk starts at text[0], last chunk ends at text[-1].
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])


def test_chunk_text_overlap_start_is_clean_line():
    """Each chunk (except the first) starts at a line boundary."""
    line = "Net income: $31,778,000,000 per quarter end\n"  # 45 chars
    text = line * 20  # 900 chars
    chunks = _chunk_text(text, chunk_size=250, overlap=90)
    assert len(chunks) > 2
    for chunk in chunks[1:]:  # first chunk always starts at 0 (clean)
        # The chunk must start at the beginning of a line — either at text[0]
        # or immediately after a newline in the original text.
        start_pos = text.index(chunk[:20])  # locate by prefix
        assert start_pos == 0 or text[start_pos - 1] == "\n", (
            f"Chunk does not start at a line boundary; preceding char: {text[start_pos-1]!r}"
        )
