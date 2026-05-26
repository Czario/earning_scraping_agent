"""Tests for the data extraction node (LLM → flexible metrics dict)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.extract_financial_metrics import (
    _HINTS_DIR,
    _chunk_text,
    _llm_map_concepts,
    _load_company_hints,
    _merge_metrics,
    _prescan_document,
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


# ---------------------------------------------------------------------------
# _prescan_document — scale detection
# ---------------------------------------------------------------------------

class TestPrescanDocument:
    """_prescan_document detects the document scale from table headers."""

    def test_classic_in_thousands(self):
        """Standard '(In thousands)' header is detected."""
        text = "Condensed Consolidated Statements of Operations\n(In thousands, except per share data)\nRevenue 5661524\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_amounts_in_thousands(self):
        """BJ Wholesale-style '(Amounts in thousands, except per share amounts)' is detected."""
        text = "CONDENSED CONSOLIDATED STATEMENTS OF INCOME\n(Amounts in thousands, except per share amounts)\nNet revenues 5661524\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_in_millions(self):
        """Standard '(In millions)' header is detected."""
        text = "Condensed Statements of Operations\n(In millions, except per share data)\nRevenue 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    def test_amounts_in_millions(self):
        """'(Amounts in millions, except per share data)' variant is detected."""
        text = "Consolidated Statements of Operations\n(Amounts in millions, except per share data)\nRevenue 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    def test_in_billions(self):
        """Standard '(In billions)' header is detected."""
        text = "Statements of Income\n(In billions, except EPS)\nRevenue 1.24\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "billions"

    def test_no_scale_header_returns_none(self):
        """Text with no scale header returns None."""
        text = "Revenue was $132.4 million in the quarter. Gross profit of $1.03 billion.\n"
        scale, _, _ = _prescan_document(text)
        assert scale is None


# ---------------------------------------------------------------------------
# _load_company_hints
# ---------------------------------------------------------------------------

class TestLoadCompanyHints:
    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        """Missing hint file returns empty string."""
        monkeypatch.setattr(
            "earnings_agents.nodes.extract_financial_metrics._HINTS_DIR", tmp_path
        )
        assert _load_company_hints("AAPL") == ""

    def test_returns_content_when_file_exists(self, tmp_path, monkeypatch):
        """Existing hint file returns its stripped content."""
        (tmp_path / "AAPL.md").write_text("  Report shares in thousands.\n", encoding="utf-8")
        monkeypatch.setattr(
            "earnings_agents.nodes.extract_financial_metrics._HINTS_DIR", tmp_path
        )
        assert _load_company_hints("AAPL") == "Report shares in thousands."

    def test_ticker_uppercased_for_lookup(self, tmp_path, monkeypatch):
        """Ticker is upper-cased so 'aapl' finds AAPL.md."""
        (tmp_path / "AAPL.md").write_text("hint content", encoding="utf-8")
        monkeypatch.setattr(
            "earnings_agents.nodes.extract_financial_metrics._HINTS_DIR", tmp_path
        )
        assert _load_company_hints("aapl") == "hint content"

    def test_empty_file_returns_empty_string(self, tmp_path, monkeypatch):
        """An all-whitespace hint file is treated as absent."""
        (tmp_path / "MSFT.md").write_text("   \n\n  ", encoding="utf-8")
        monkeypatch.setattr(
            "earnings_agents.nodes.extract_financial_metrics._HINTS_DIR", tmp_path
        )
        assert _load_company_hints("MSFT") == ""

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_hint_injected_into_prompt(self, mock_build_llm, tmp_path, monkeypatch):
        """Company hint text appears in the LLM prompt when a hint file is present."""
        (tmp_path / "AAPL.md").write_text("Always look for segment revenue.", encoding="utf-8")
        monkeypatch.setattr(
            "earnings_agents.nodes.extract_financial_metrics._HINTS_DIR", tmp_path
        )
        captured_prompts: list[str] = []

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = lambda prompt: (
            captured_prompts.append(prompt)
            or '{"Net income": 31000000000}'
        )
        mock_build_llm.return_value = mock_llm

        state = _base_state(ticker="AAPL", company_name="Apple Inc.")
        extract_financial_metrics_node(state)

        assert captured_prompts, "LLM was never invoked"
        assert "Always look for segment revenue." in captured_prompts[0]

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_no_hint_file_does_not_alter_prompt(self, mock_build_llm, tmp_path, monkeypatch):
        """When no hint file exists the prompt is unchanged (no hint section)."""
        monkeypatch.setattr(
            "earnings_agents.nodes.extract_financial_metrics._HINTS_DIR", tmp_path
        )
        captured_prompts: list[str] = []

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = lambda prompt: (
            captured_prompts.append(prompt)
            or '{"Net income": 31000000000}'
        )
        mock_build_llm.return_value = mock_llm

        state = _base_state(ticker="AAPL", company_name="Apple Inc.")
        extract_financial_metrics_node(state)

        assert captured_prompts, "LLM was never invoked"
        assert "Company-specific extraction hints" not in captured_prompts[0]


# ---------------------------------------------------------------------------
# Provider escalation
# ---------------------------------------------------------------------------

class TestProviderEscalation:
    """Verify that extraction attempt 2+ escalates to Groq when configured."""

    @patch("earnings_agents.nodes.extract_financial_metrics.GROQ_API_KEY", "test-key")
    @patch("earnings_agents.nodes.extract_financial_metrics.LLM_PROVIDER", "ollama")
    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_attempt_2_escalates_to_groq(self, mock_build_llm):
        """Second extraction pass passes provider='groq' to build_llm."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"Net income": 5000000000}'
        mock_build_llm.return_value = mock_llm

        state = _base_state(extraction_attempts=1)  # already did attempt 1
        extract_financial_metrics_node(state)

        call_kwargs = mock_build_llm.call_args.kwargs
        assert call_kwargs.get("provider") == "groq"

    @patch("earnings_agents.nodes.extract_financial_metrics.GROQ_API_KEY", "test-key")
    @patch("earnings_agents.nodes.extract_financial_metrics.LLM_PROVIDER", "ollama")
    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_attempt_1_uses_default_provider(self, mock_build_llm):
        """First extraction pass leaves provider=None (uses configured default)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"Net income": 5000000000}'
        mock_build_llm.return_value = mock_llm

        state = _base_state(extraction_attempts=0)
        extract_financial_metrics_node(state)

        call_kwargs = mock_build_llm.call_args.kwargs
        assert call_kwargs.get("provider") is None

    @patch("earnings_agents.nodes.extract_financial_metrics.GROQ_API_KEY", "")
    @patch("earnings_agents.nodes.extract_financial_metrics.LLM_PROVIDER", "ollama")
    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_no_groq_key_stays_on_ollama(self, mock_build_llm):
        """Without GROQ_API_KEY, attempt 2 stays on default provider (no escalation)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"Net income": 5000000000}'
        mock_build_llm.return_value = mock_llm

        state = _base_state(extraction_attempts=1)
        extract_financial_metrics_node(state)

        call_kwargs = mock_build_llm.call_args.kwargs
        assert call_kwargs.get("provider") is None

    @patch("earnings_agents.nodes.extract_financial_metrics.GROQ_API_KEY", "test-key")
    @patch("earnings_agents.nodes.extract_financial_metrics.LLM_PROVIDER", "groq")
    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_groq_primary_does_not_double_escalate(self, mock_build_llm):
        """When LLM_PROVIDER is already 'groq', no escalation override is applied."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"Net income": 5000000000}'
        mock_build_llm.return_value = mock_llm

        state = _base_state(extraction_attempts=1)
        extract_financial_metrics_node(state)

        call_kwargs = mock_build_llm.call_args.kwargs
        assert call_kwargs.get("provider") is None


# ── Taxonomy-key targeted extraction ─────────────────────────────────────────

class TestTaxonomyKeyMapping:
    """Verify two-tier concept mapping in targeted (normalize_data) mode."""

    _TARGET_CONCEPTS = [
        {
            "_id": "aaa111",
            "concept": "us-gaap:Revenues",
            "label": "Net sales",
            "taxonomy_key": "us-gaap:Revenues",
            "path": "001",
            "statement_type": "income_statement",
        },
        {
            "_id": "bbb222",
            "concept": "us-gaap:CostOfRevenue",
            "label": "Cost of sales",
            "taxonomy_key": "us-gaap:CostOfRevenue",
            "path": "002",
            "statement_type": "income_statement",
        },
        {
            "_id": "ccc333",
            "concept": "us-gaap:OperatingIncomeLoss",
            "label": "Operating income",
            "taxonomy_key": "us-gaap:OperatingIncomeLoss",
            "path": "003",
            "statement_type": "income_statement",
        },
    ]

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_exact_label_match_maps_concept_ids(self, mock_build_llm):
        """Tier 1: exact label match populates concept_metrics."""
        mock_llm = MagicMock()
        # LLM echoes exact labels
        mock_llm.invoke.return_value = (
            '{"__scale__": "millions", "__period__": "Three Months Ended Mar 31, 2026",'
            ' "Net sales": 5234, "Cost of sales": 3900}'
        )
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=self._TARGET_CONCEPTS)
        result = extract_financial_metrics_node(state)

        assert result["status"] == "extracted"
        cm = result.get("concept_metrics", {})
        assert cm["aaa111"] == pytest.approx(5_234_000_000)
        assert cm["bbb222"] == pytest.approx(3_900_000_000)
        assert "ccc333" not in cm  # not returned by LLM

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_normalised_label_match_handles_casing_drift(self, mock_build_llm):
        """Tier 1: normalised match handles casing/whitespace drift."""
        mock_llm = MagicMock()
        # LLM returns slightly different casing
        mock_llm.invoke.return_value = (
            '{"__scale__": "as-is", "__period__": null,'
            ' "net sales": 5234000000, "COST OF SALES": 3900000000}'
        )
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=self._TARGET_CONCEPTS)
        result = extract_financial_metrics_node(state)

        cm = result.get("concept_metrics", {})
        assert cm.get("aaa111") == pytest.approx(5_234_000_000)
        assert cm.get("bbb222") == pytest.approx(3_900_000_000)

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_llm_semantic_mapping_used_for_residual(self, mock_build_llm):
        """Tier 2: LLM mapping resolves semantically similar but unmatched keys."""
        # First call (extraction): LLM returns a synonym key not in target labels
        extraction_response = (
            '{"__scale__": "as-is", "__period__": null,'
            ' "Revenue from operations": 5234000000}'
        )
        # Second call (mapping): LLM semantic mapper returns the match
        mapping_response = (
            '{"aaa111": "Revenue from operations", "bbb222": null, "ccc333": null}'
        )
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [extraction_response, mapping_response]
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=self._TARGET_CONCEPTS)
        result = extract_financial_metrics_node(state)

        cm = result.get("concept_metrics", {})
        assert cm.get("aaa111") == pytest.approx(5_234_000_000)

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_llm_mapping_ignores_hallucinated_keys(self, mock_build_llm):
        """Tier 2: LLM mapper cannot hallucinate keys not in extracted metrics."""
        extraction_response = (
            '{"__scale__": "as-is", "__period__": null, "Revenue": 5234000000}'
        )
        # LLM mapping hallucinates a key that wasn't extracted
        mapping_response = '{"aaa111": "Net sales"}'  # "Net sales" not in extracted
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [extraction_response, mapping_response]
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=self._TARGET_CONCEPTS)
        result = extract_financial_metrics_node(state)

        # "Net sales" was hallucinated → must NOT be mapped
        cm = result.get("concept_metrics", {})
        assert "aaa111" not in cm


# ── _llm_map_concepts unit tests ──────────────────────────────────────────────

class TestLlmMapConcepts:
    """Unit tests for the _llm_map_concepts helper (no full pipeline)."""

    _CONCEPTS = [
        {"_id": "aaa", "concept": "us-gaap:Revenues", "label": "Net sales"},
        {"_id": "bbb", "concept": "us-gaap:CostOfRevenue", "label": "Cost of sales"},
    ]

    def test_valid_mapping_returned(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"aaa": "Revenue from operations", "bbb": null}'
        result = _llm_map_concepts(["Revenue from operations", "Other"], self._CONCEPTS, mock_llm)
        assert result == {"aaa": "Revenue from operations"}

    def test_hallucinated_key_rejected(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"aaa": "Invented key"}'
        result = _llm_map_concepts(["Revenue from operations"], self._CONCEPTS, mock_llm)
        assert result == {}  # "Invented key" not in extracted_keys

    def test_duplicate_key_assignment_rejected(self):
        """Same extracted key cannot be assigned to two concepts."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"aaa": "Revenue", "bbb": "Revenue"}'
        result = _llm_map_concepts(["Revenue"], self._CONCEPTS, mock_llm)
        # First assignment wins; second is dropped
        assert len(result) == 1
        assert result.get("aaa") == "Revenue"

    def test_llm_failure_returns_empty(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "not json at all"
        result = _llm_map_concepts(["Revenue"], self._CONCEPTS, mock_llm)
        assert result == {}

    def test_empty_inputs_return_empty(self):
        mock_llm = MagicMock()
        assert _llm_map_concepts([], self._CONCEPTS, mock_llm) == {}
        assert _llm_map_concepts(["Revenue"], [], mock_llm) == {}
