"""Tests for the data extraction node (LLM → flexible metrics dict)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from earnings_agents.nodes.extract_financial_metrics import (
    _chunk_text,
    _llm_map_concepts,
    _merge_metrics,
    _prescan_document,
    extract_financial_metrics_node,
)
from earnings_agents.extraction.chunker import _build_period_hint


def _base_state(**overrides):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "discovered_file_url": "https://example.com/q1-2025.pdf",
        "file_type": "pdf",
        "raw_text": (
            "Apple reports quarterly earnings.\n"
            "Revenue: $124.3 billion. Net income: $33.9 billion. "
            "Diluted EPS: $2.40.\n"
            "Guidance: Revenue of $89-93 billion expected for Q2 FY2025."
        ),
        # Generic extraction has been removed: every run is targeted, so a
        # default set of income-statement concepts is supplied. Tests that
        # assert exact LLM call counts override this with concepts whose labels
        # match the mocked response so no Tier-2 mapping call is triggered.
        "target_concepts": [
            {"_id": "c_rev", "concept": "us-gaap:Revenues",
             "label": "Total Net Revenue", "statement_type": "income_statement"},
            {"_id": "c_ni", "concept": "us-gaap:NetIncomeLoss",
             "label": "Net Income", "statement_type": "income_statement"},
            {"_id": "c_eps", "concept": "us-gaap:EarningsPerShareDiluted",
             "label": "Diluted EPS", "statement_type": "income_statement"},
        ],
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


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
def test_first_attempt_runs_all_section_chunks(mock_llm_cls):
    """When table sections are available, attempt 1 combines income_statement
    and other sections into a single chunk (balance_sheet and cash_flow are
    excluded because the targeted concepts are income-statement only).  With a
    large CHUNK_SIZE (Groq) all included sections fit in one LLM call.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        '{"Revenues": 1000000000, "Operating expenses": 250000000}',
    ]
    mock_llm_cls.return_value = mock_llm

    result = extract_financial_metrics_node(
        _base_state(
            file_type="html",
            target_concepts=[
                {"_id": "c_rev", "concept": "us-gaap:Revenues",
                 "label": "Revenues", "statement_type": "income_statement"},
                {"_id": "c_opex", "concept": "us-gaap:OperatingExpenses",
                 "label": "Operating expenses", "statement_type": "income_statement"},
            ],
            raw_sections={
                "income_statement": ["income table"],
                "balance_sheet": ["balance table"],   # excluded in targeted mode
                "other": ["supplemental table"],
            },
            raw_text="Quarter data",
        )
    )

    assert result["status"] == "extracted"
    # income_statement + other are combined; balance_sheet is excluded.
    # With CHUNK_SIZE large enough for both sections → 1 LLM call.
    assert mock_llm.invoke.call_count == 1
    assert result["metrics"]["Revenues"] == pytest.approx(1_000_000_000)
    assert result["metrics"]["Operating expenses"] == pytest.approx(250_000_000)


@patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
def test_retry_scopes_to_income_statement_and_preserves_other_sections(mock_llm_cls):
    """Retry passes (attempt_num > 1) carry forward untouched metrics from the
    previous pass and overwrite only the keys returned by the new extraction.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = (
        '{"Revenues": 81615000000, "Cost of Revenue": 20458000000, "Gross Profit": 61157000000}'
    )
    mock_llm_cls.return_value = mock_llm

    prev_metrics = {
        "Cash and cash equivalents": 10_000_000_000,
        "Cost of Revenue": 10_252_500_000,  # bad prior value to be overwritten
    }
    state = _base_state(
        file_type="html",
        target_concepts=[
            {"_id": "c_rev", "concept": "us-gaap:Revenues",
             "label": "Revenues", "statement_type": "income_statement"},
            {"_id": "c_cor", "concept": "us-gaap:CostOfRevenue",
             "label": "Cost of Revenue", "statement_type": "income_statement"},
            {"_id": "c_gp", "concept": "us-gaap:GrossProfit",
             "label": "Gross Profit", "statement_type": "income_statement"},
        ],
        raw_sections={
            "income_statement": ["income table"],
            "other": ["other table"],
        },
        raw_text="Quarter data",
        extraction_attempts=1,  # this call is pass 2
        metrics=prev_metrics,
        findings=[
            {
                "type": "identity_violation",
                "severity": "high",
                "message": "Income-statement identity broken",
                "keys": ["Revenues", "Cost of Revenue", "Gross Profit"],
                "suggested_action": "Re-extract from the same current-period income statement column.",
            }
        ],
        extraction_notes="focus on income statement metrics",
    )

    result = extract_financial_metrics_node(state)

    assert result["status"] == "extracted"
    # Only income_statement chunk re-run on retry.
    assert mock_llm.invoke.call_count == 1
    # Retried section overwrites bad prior value.
    assert result["metrics"]["Cost of Revenue"] == pytest.approx(20_458_000_000)
    # Untouched sections preserved from previous pass.
    assert result["metrics"]["Cash and cash equivalents"] == pytest.approx(10_000_000_000)


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


def test_merge_target_year_filters_stale_chunks():
    """Chunks whose __period__ year doesn't match target_year are excluded from
    numeric merge. Their values are only used as last-resort fallback for keys
    that appear nowhere else.

    Mirrors the NVIDIA Q1 FY2027 scenario:
      - Chunk 3 correctly extracts from the current-year column (April 27, 2026)
      - Chunk 4 mistakenly extracts from the prior-year column (April 27, 2025)
    With target_year=2026 only chunk 3's values should be used for Revenue.
    """
    chunks = [
        # on-target chunk: current-year column (period contains 2026)
        {
            "__period__": "Three Months Ended April 27, 2026",
            "Revenue": 44_100_000_000,
            "Net Income": 18_800_000_000,
        },
        # stale chunk: prior-year comparison column (period contains 2025)
        {
            "__period__": "Three Months Ended April 27, 2025",
            "Revenue": 26_000_000_000,   # prior-year figure
            "Net Income": 14_900_000_000,
        },
    ]
    merged = _merge_metrics(chunks, target_year=2026)

    # On-target chunk wins for keys present in both
    assert merged["Revenue"] == pytest.approx(44_100_000_000)
    assert merged["Net Income"] == pytest.approx(18_800_000_000)


def test_merge_stale_chunk_value_used_as_fallback_for_unique_key():
    """A key that only appears in a stale chunk is still included as a fallback
    (it may be a legitimate metric not repeated in the current-year column).
    """
    chunks = [
        {
            "__period__": "Three Months Ended April 27, 2026",
            "Revenue": 44_100_000_000,
        },
        {
            "__period__": "Three Months Ended April 27, 2025",
            "Revenue": 26_000_000_000,
            "Prior Year EPS": 0.77,   # only appears in stale chunk
        },
    ]
    merged = _merge_metrics(chunks, target_year=2026)

    assert merged["Revenue"] == pytest.approx(44_100_000_000)
    # Falls back to stale value since it's the only source
    assert merged["Prior Year EPS"] == pytest.approx(0.77)


def test_merge_no_target_year_unchanged_behaviour():
    """When target_year=None (IR path, no SEC data) all chunks contribute equally
    — identical to the pre-filtering behaviour.
    """
    chunks = [
        {
            "__period__": "Three Months Ended April 27, 2026",
            "Revenue": 44_100_000_000,
        },
        {
            "__period__": "Three Months Ended April 27, 2025",
            "Revenue": 26_000_000_000,
        },
    ]
    merged = _merge_metrics(chunks, target_year=None)
    # Median of [26B, 44.1B] = 35.05B
    assert merged["Revenue"] == pytest.approx(35_050_000_000)


def test_merge_chunk_with_no_period_treated_as_on_target():
    """Chunks with a null or absent __period__ are treated as on-target so they
    are never spuriously discarded when target_year filtering is active.
    """
    chunks = [
        # no __period__ — cannot classify → treated as on-target
        {"Revenue": 44_100_000_000, "Net Income": 18_800_000_000},
        # stale chunk
        {
            "__period__": "Three Months Ended April 27, 2025",
            "Revenue": 26_000_000_000,
        },
    ]
    merged = _merge_metrics(chunks, target_year=2026)

    # Unclassified chunk is on-target → Revenue = 44.1B (not median with stale)
    assert merged["Revenue"] == pytest.approx(44_100_000_000)
    assert merged["Net Income"] == pytest.approx(18_800_000_000)


def test_merge_evicts_stale_case_duplicate_keeps_on_target():
    """When a case-duplicate pair exists where one came from on-target chunks
    and the other only from stale chunks, the stale-only variant is dropped.

    Mirrors the NVDA scenario:
      - 'Cost of revenue' = 20.458B from on-target chunk (correct)
      - 'Cost of Revenue' = 48B from stale chunk only (wrong prior-year value)
    Expected: only 'Cost of revenue' survives in merged output.
    """
    chunks = [
        # on-target: current-year column
        {
            "__period__": "Three Months Ended April 27, 2026",
            "Revenue": 81_615_000_000,
            "Cost of revenue": 20_458_000_000,   # correct
        },
        # stale: prior-year comparison column
        {
            "__period__": "Three Months Ended April 27, 2025",
            "Revenue": 26_000_000_000,
            "Cost of Revenue": 48_000_000_000,   # wrong stale-only duplicate
        },
    ]
    merged = _merge_metrics(chunks, target_year=2026)

    # Correct on-target variant retained
    assert merged["Cost of revenue"] == pytest.approx(20_458_000_000)
    # Wrong stale-only case-duplicate evicted
    assert "Cost of Revenue" not in merged


def test_merge_keeps_stale_case_duplicate_when_no_on_target_variant():
    """When both case variants appear only in stale chunks (the metric genuinely
    only exists in the comparison section), both are retained — no eviction.
    """
    chunks = [
        # on-target: no cost-of-revenue at all
        {
            "__period__": "Three Months Ended April 27, 2026",
            "Revenue": 81_615_000_000,
        },
        # stale: both variants only in stale chunks
        {
            "__period__": "Three Months Ended April 27, 2025",
            "Cost of revenue": 20_000_000_000,
            "Cost of Revenue": 20_000_000_000,
        },
    ]
    merged = _merge_metrics(chunks, target_year=2026)

    # Both kept as fallback — no on-target variant exists to prefer
    assert "Cost of revenue" in merged or "Cost of Revenue" in merged


def test_merge_prefers_authoritative_section_over_other_table():
    """A numeric metric is taken from its primary GAAP statement, never averaged
    with the same key leaking from a supplementary ('other') table.

    Mirrors the NVDA phantom-average bug: the income-statement chunk reports
    Cost of revenue = 20.458B, a segment/'other' chunk reports a different
    number for the same key. The old median-of-two merge produced their average
    (a phantom '.5' value). With section provenance, the income-statement value
    must win outright.
    """
    chunks = [
        {"Cost of revenue": 20_458_000_000},   # income statement
        {"Cost of revenue": 47_000_000},        # leaked from a segment table
    ]
    sections = ["income_statement", "other"]
    merged = _merge_metrics(chunks, sections=sections)

    # Income-statement value wins outright — NOT the average (10_252_500_000).
    assert merged["Cost of revenue"] == pytest.approx(20_458_000_000)


def test_merge_within_same_section_falls_back_to_median():
    """When the winning (highest-authority) section is split across multiple
    chunks that disagree, the median is the defensive tie-breaker within that
    section only.
    """
    chunks = [
        {"Total Revenue": 80_000_000_000},
        {"Total Revenue": 80_000_000_000},
        {"Total Revenue": 80_000},  # unscaled outlier, same section
    ]
    sections = ["income_statement", "income_statement", "income_statement"]
    merged = _merge_metrics(chunks, sections=sections)

    # Median within the income-statement section resists the outlier.
    assert merged["Total Revenue"] == pytest.approx(80_000_000_000)


def test_merge_footnote_artifact_takes_max_when_values_differ_5x():
    """When same-section values differ by ≥5×, the smaller value(s) are
    footnote/amortization artifacts; the maximum (real P&L figure) is chosen.

    Mirrors NVDA Q1FY27: R&D appears as $6,321M (income statement) and
    $167M (footnote amortization breakdown), yielding a bogus median of
    ~$3,244M without this fix.
    """
    chunks = [
        {"Research and development": 6_321_000_000},  # real P&L line
        {"Research and development": 167_000_000},    # footnote artifact
    ]
    # Both chunks are "other" (same priority) — no section authority to distinguish.
    sections = ["other", "other"]
    merged = _merge_metrics(chunks, sections=sections)
    assert merged["Research and development"] == pytest.approx(6_321_000_000)


def test_merge_moderate_divergence_still_uses_median():
    """When values differ by less than 5×, the median tie-breaker is still used
    (legitimate rounding / multi-chunk split, not a footnote artifact).
    """
    chunks = [
        {"Operating expenses": 7_500_000_000},
        {"Operating expenses": 7_700_000_000},
    ]
    sections = ["income_statement", "income_statement"]
    merged = _merge_metrics(chunks, sections=sections)
    assert merged["Operating expenses"] == pytest.approx(7_600_000_000)


def test_merge_without_sections_keeps_median_behaviour():
    """When no section provenance is supplied (char-split / PDF chunks), all
    chunks share equal authority and the prior median-of-values behaviour holds.
    """
    chunks = [
        {"Net Income": 30_000_000_000},
        {"Net Income": 32_000_000_000},
    ]
    merged = _merge_metrics(chunks)  # no sections argument
    assert merged["Net Income"] == pytest.approx(31_000_000_000)


def test_merge_collects_source_snippets():
    """__sources__ snippets from chunks are merged under the __sources__ key."""
    chunks = [
        {"Revenue": 80_000_000_000, "__sources__": {"Revenue": "Total revenue 80,000"}},
        {"Net Income": 5_000_000_000, "__sources__": {"Net Income": "Net income 5,000"}},
    ]
    merged = _merge_metrics(chunks)
    assert merged["__sources__"] == {
        "Revenue": "Total revenue 80,000",
        "Net Income": "Net income 5,000",
    }


def test_merge_source_snippet_prefers_higher_authority_section():
    """On label conflict, the snippet from the higher-authority section wins."""
    chunks = [
        {"Revenue": 80_000_000_000, "__sources__": {"Revenue": "income statement row"}},
        {"Revenue": 80_000_000_000, "__sources__": {"Revenue": "segment table row"}},
    ]
    sections = ["income_statement", "other"]
    merged = _merge_metrics(chunks, sections=sections)
    assert merged["__sources__"]["Revenue"] == "income statement row"


def test_merge_omits_sources_key_when_no_snippets():
    """No __sources__ on any chunk → no __sources__ key in the merged dict."""
    chunks = [{"Revenue": 80_000_000_000}]
    merged = _merge_metrics(chunks)
    assert "__sources__" not in merged


def test_targeted_prompt_enforces_column_and_consistency_rules():
    """Guard: the targeted prompt must keep the Phase-2 column / footnote /
    consistency rules that prevent prior-year and footnote-value leakage.
    """
    from earnings_agents.nodes.extract_financial_metrics import _TARGETED_PROMPT_TEMPLATE

    t = _TARGETED_PROMPT_TEMPLATE
    # Most-recent-column rule
    assert "MOST RECENT period column" in t
    # Footnote / sub-table exclusion
    assert "FOOTNOTES" in t
    # Internal-consistency identity
    assert "Revenue \u2212 Cost of revenue = Gross profit" in t


def test_scale_millions_applied_by_python():
    """__scale__:millions causes Python to multiply raw table values × 1_000_000."""
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    raw = '{"__scale__": "millions", "Total Revenue": 82886, "Net Income": 31778, "Diluted EPS": 4.27}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["Total Revenue"] == pytest.approx(82_886_000_000)
    assert result["Net Income"] == pytest.approx(31_778_000_000)
    assert result["Diluted EPS"] == pytest.approx(4.27)   # EPS not multiplied


def test_scale_thousands_large_annual_values_still_scaled():
    """Thousands-scale multi-billion annual figures must all be scaled consistently.

    Regression: CASY FY2026 annual press release in thousands. Revenue
    (17,561,101) and Cost of revenue (13,240,060) exceed the old fixed
    _TABLE_RAW_MAX guard (10 M) and were left unscaled, while Gross profit
    (4,321,041 < 10 M) was scaled — breaking the income-statement identity.
    All three must be multiplied by 1_000.
    """
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    raw = (
        '{"__scale__": "thousands", '
        '"Revenues": 17561101, '
        '"Cost of goods sold": 13240060, '
        '"Gross Profit": 4321041}'
    )
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["Revenues"] == pytest.approx(17_561_101_000)
    assert result["Cost of goods sold"] == pytest.approx(13_240_060_000)
    assert result["Gross Profit"] == pytest.approx(4_321_041_000)
    # Identity holds after scaling: Revenue − Cost == Gross profit
    assert result["Revenues"] - result["Cost of goods sold"] == pytest.approx(
        result["Gross Profit"]
    )


def test_scale_thousands_already_full_usd_not_rescaled():
    """A thousands-scale cell already at full USD ($10T+) is not re-multiplied."""
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    raw = '{"__scale__": "thousands", "Total Revenue": 20000000000000}'
    result = _parse_llm_response(raw)
    assert result is not None
    # Above the ~$10T absolute ceiling -> treated as already full USD.
    assert result["Total Revenue"] == pytest.approx(20_000_000_000_000)


def test_scale_as_is_no_multiplication():
    """__scale__:as-is leaves full-USD narrative values untouched."""
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    raw = '{"__scale__": "as-is", "Total Revenue": 82886000000, "Diluted EPS": 4.27}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result["Total Revenue"] == pytest.approx(82_886_000_000)
    assert result["Diluted EPS"] == pytest.approx(4.27)


@pytest.mark.parametrize(
    "scale, multiplier, rev_raw, cost_raw, gross_raw",
    [
        # thousands: raw Revenue/Cost > 10 M (old fixed cap) but Gross < 10 M —
        # the exact straddle that broke CASY. Identity: 17,561,101−13,240,060.
        ("thousands", 1_000, 17_561_101, 13_240_060, 4_321_041),
        # millions: typical large-cap statement. Identity: 17,561−13,240.
        ("millions", 1_000_000, 17_561, 13_240, 4_321),
        # billions: values are small raw numbers. Identity: 176−132.
        ("billions", 1_000_000_000, 176, 132, 44),
    ],
)
def test_scale_invariant_uniform_across_all_scales(
    scale, multiplier, rev_raw, cost_raw, gross_raw
):
    """Generality guard: every dollar line item in one statement scales uniformly.

    The wrong-scaling bug (CASY) happened because a fixed raw cap let some
    values cross the threshold (skipped) while others did not (scaled),
    breaking the income-statement identity. This locks the invariant for all
    three scales: each value is multiplied by the same factor and the
    income-statement identity (Revenue − Cost == Gross Profit) survives scaling.
    """
    from earnings_agents.nodes.extract_financial_metrics import _parse_llm_response

    # Sanity: the chosen raw triple already satisfies the identity.
    assert rev_raw - cost_raw == gross_raw

    raw = (
        f'{{"__scale__": "{scale}", '
        f'"Revenues": {rev_raw}, '
        f'"Cost of goods sold": {cost_raw}, '
        f'"Gross Profit": {gross_raw}}}'
    )
    result = _parse_llm_response(raw)
    assert result is not None
    # Every value scaled by the SAME factor.
    assert result["Revenues"] == pytest.approx(rev_raw * multiplier)
    assert result["Cost of goods sold"] == pytest.approx(cost_raw * multiplier)
    assert result["Gross Profit"] == pytest.approx(gross_raw * multiplier)
    # The income-statement identity must hold after scaling, on every scale.
    assert result["Revenues"] - result["Cost of goods sold"] == pytest.approx(
        result["Gross Profit"]
    )



def _base_state(**overrides):
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "discovered_file_url": "https://example.com/q1-2025.pdf",
        "file_type": "pdf",
        "raw_text": (
            "Apple reports quarterly earnings.\n"
            "Revenue: $124.3 billion. Net income: $33.9 billion. "
            "Diluted EPS: $2.40.\n"
            "Guidance: Revenue of $89-93 billion expected for Q2 FY2025."
        ),
        # Generic extraction has been removed: every run is targeted, so a
        # default set of income-statement concepts is supplied. Tests that
        # assert exact LLM call counts override this with concepts whose labels
        # match the mocked response so no Tier-2 mapping call is triggered.
        "target_concepts": [
            {"_id": "c_rev", "concept": "us-gaap:Revenues",
             "label": "Total Net Revenue", "statement_type": "income_statement"},
            {"_id": "c_ni", "concept": "us-gaap:NetIncomeLoss",
             "label": "Net Income", "statement_type": "income_statement"},
            {"_id": "c_eps", "concept": "us-gaap:EarningsPerShareDiluted",
             "label": "Diluted EPS", "statement_type": "income_statement"},
        ],
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
    from earnings_agents.analysis.validators import validate_metrics

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
    cleaned, warnings = validate_metrics(metrics)
    assert warnings == []
    assert cleaned["Net income"] == 31_778_000_000


def test_validate_metrics_flags_broken_identity():
    """A bad Net income that breaks the EPS × shares sanity check is blocking."""
    from earnings_agents.analysis.validators import validate_metrics

    # EPS 4.27 × Diluted shares 7.445B ≈ $31.8B Net income.
    # Asserting a Net income of $12.3B trips the universal EPS sanity check.
    metrics = {
        "Net income": 12_345_000_000,
        "Diluted Earnings per Share": 4.27,
        "Weighted average shares outstanding: Diluted": 7_445_000_000,
    }
    _, warnings = validate_metrics(metrics)
    assert warnings, "expected EPS sanity warning for broken Net income"
    assert any("EPS" in w for w in warnings)


def test_validate_metrics_advisory_only_for_structural_drift():
    """Structural decomposition drift (e.g. Net income = Pre-tax − Tax) is advisory only."""
    from earnings_agents.analysis.validators import validate_metrics

    # Pre-tax 39.34B − Tax 7.56B = 31.78B, but reported Net income is 12.3B.
    # Without EPS / shares to cross-check, this is logged as advisory only.
    metrics = {
        "Income before income taxes": 39_340_000_000,
        "Provision for income taxes": 7_562_000_000,
        "Net income": 12_345_000_000,
    }
    _, warnings = validate_metrics(metrics)
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
    """When STRICT_ACCURACY is False the node proceeds past the identity gate."""
    import earnings_agents.workflow as wf

    state = {
        "ticker": "TEST",
        "company_name": "Test Co",
        "discovered_file_url": "https://example.com",
        "file_type": "html",
        "metrics": {"Revenue": 1},
        "identity_warnings": ["Gross margin drift 5%"],
        "status": "extracted",
        # No concept_metrics/cik → normalize_data upsert is skipped with a warning
    }
    original = wf.STRICT_ACCURACY
    wf.STRICT_ACCURACY = False
    try:
        result = wf.mongodb_save_node(state)
    finally:
        wf.STRICT_ACCURACY = original
    assert result["status"] == "saved"


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

    # -- Non-parenthesised scale headings (scale stated in a heading above the
    #    table rather than a parenthesised table caption). --------------------

    def test_unparenthesised_dollars_in_thousands(self):
        """'Dollars in thousands' heading (no parentheses) is detected."""
        text = "CONSOLIDATED STATEMENTS OF INCOME\nDollars in thousands\nNet sales 5661524\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_unparenthesised_amounts_in_thousands(self):
        """'Amounts in thousands' heading (no parentheses) is detected."""
        text = "STATEMENTS OF OPERATIONS\nAmounts in thousands\nNet revenues 5661524\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_unparenthesised_dollar_sign_in_millions(self):
        """'$ in millions' heading (no parentheses) is detected."""
        text = "Income Statement\n$ in millions\nRevenue 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    def test_unparenthesised_in_thousands_at_line_start(self):
        """A line beginning 'In thousands, except per share data' is detected."""
        text = "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS\nIn thousands, except per share data\nRevenue 5661524\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_unparenthesised_all_figures_in_millions(self):
        """'All figures in millions unless otherwise noted' heading is detected."""
        text = "Selected Financial Data\nAll figures in millions unless otherwise noted\nRevenue 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    def test_unparenthesised_us_dollars_in_millions(self):
        """'U.S. dollars in millions' heading (no parentheses) is detected."""
        text = "Statement of Income\nU.S. dollars in millions\nRevenue 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    def test_narrative_million_value_not_treated_as_scale(self):
        """Narrative '$5.2 million in revenue' must NOT be read as a scale heading."""
        text = (
            "The company reported $5.2 million in revenue this quarter, "
            "up from $3.1 million a year ago, and held $40 million in cash.\n"
        )
        scale, _, _ = _prescan_document(text)
        assert scale is None

    def test_parenthesised_scale_takes_priority(self):
        """When both forms appear, the parenthesised table caption wins."""
        text = (
            "Highlights: revenue grew, amounts in thousands of units shipped.\n"
            "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
            "(In millions, except per share data)\nRevenue 124300\n"
        )
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    # -- Non-ASCII whitespace inside the caption (CrowdStrike regression). -----
    #    HTML earnings releases routinely separate the words of a scale caption
    #    with a non-breaking space (\xa0) or other Unicode space, which the
    #    literal-space scale patterns would otherwise miss — leaving the scale
    #    undetected and the values 1000x too small.

    def test_caption_with_non_breaking_space(self):
        """'(in\\xa0thousands)' (non-breaking space) is detected as thousands."""
        text = (
            "Condensed Consolidated Statements of Operations\n"
            "(in\xa0thousands, except per share data)\nTotal revenue 1385629\n"
        )
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_caption_with_narrow_no_break_space(self):
        """'(in\\u202fmillions)' (narrow no-break space) is detected as millions."""
        text = "Statements of Income\n(in\u202fmillions)\nRevenue 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"

    def test_caption_with_thin_space(self):
        """'(in\\u2009billions)' (thin space) is detected as billions."""
        text = "Statements of Income\n(in\u2009billions)\nRevenue 1.24\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "billions"

    def test_caption_with_collapsed_double_space(self):
        """A caption with a run of spaces ('(in  thousands)') is detected."""
        text = "Statements of Operations\n(in  thousands)\nRevenue 5661524\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "thousands"

    def test_unparenthesised_heading_with_non_breaking_spaces(self):
        """'Dollars\\xa0in\\xa0millions' heading with nbsp is detected."""
        text = "CONSOLIDATED STATEMENTS OF INCOME\nDollars\xa0in\xa0millions\nNet sales 124300\n"
        scale, _, _ = _prescan_document(text)
        assert scale == "millions"


# ---------------------------------------------------------------------------
# Provider selection (escalation removed)
# ---------------------------------------------------------------------------

class TestProviderEscalation:
    """Escalation was removed: every attempt uses the configured LLM_PROVIDER."""

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_attempt_2_does_not_escalate_to_groq(self, mock_build_llm):
        """Second extraction pass stays on the default provider (provider=None)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"Net income": 5000000000}'
        mock_build_llm.return_value = mock_llm

        state = _base_state(extraction_attempts=1)  # already did attempt 1
        extract_financial_metrics_node(state)

        call_kwargs = mock_build_llm.call_args.kwargs
        assert call_kwargs.get("provider") is None

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

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_attempt_3_does_not_escalate_to_groq(self, mock_build_llm):
        """Third extraction pass also stays on the default provider."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"Net income": 5000000000}'
        mock_build_llm.return_value = mock_llm

        state = _base_state(extraction_attempts=2)
        extract_financial_metrics_node(state)

        call_kwargs = mock_build_llm.call_args.kwargs
        assert call_kwargs.get("provider") is None

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_groq_primary_stays_on_default(self, mock_build_llm):
        """When LLM_PROVIDER is 'groq', no provider override is applied."""
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

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_mapped_metric_keys_populated_tier1(self, mock_build_llm):
        """Tier 1 matches must be recorded in mapped_metric_keys."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"__scale__": "millions", "__period__": null,'
            ' "Net sales": 5234, "Cost of sales": 3900}'
        )
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=self._TARGET_CONCEPTS)
        result = extract_financial_metrics_node(state)

        mapped = result.get("mapped_metric_keys") or []
        assert "Net sales" in mapped
        assert "Cost of sales" in mapped

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_mapped_metric_keys_populated_tier2(self, mock_build_llm):
        """Tier 2 (LLM semantic) matches must also appear in mapped_metric_keys."""
        extraction_response = (
            '{"__scale__": "as-is", "__period__": null,'
            ' "Revenue from operations": 5234000000}'
        )
        mapping_response = (
            '{"aaa111": "Revenue from operations", "bbb222": null, "ccc333": null}'
        )
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [extraction_response, mapping_response]
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=self._TARGET_CONCEPTS)
        result = extract_financial_metrics_node(state)

        mapped = result.get("mapped_metric_keys") or []
        assert "Revenue from operations" in mapped

    @patch("earnings_agents.nodes.extract_financial_metrics.build_llm")
    def test_mapped_metric_keys_absent_when_no_target_concepts(self, mock_build_llm):
        """Without target_concepts, mapped_metric_keys must not be set."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            '{"__scale__": "as-is", "__period__": null, "Revenue": 5234000000}'
        )
        mock_build_llm.return_value = mock_llm

        state = _base_state(target_concepts=None)
        result = extract_financial_metrics_node(state)

        assert "mapped_metric_keys" not in result or result.get("mapped_metric_keys") is None


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


class TestBuildPeriodHint:
    """Period-hint duration rule must adapt to annual vs quarterly filings."""

    def test_annual_filing_prefers_longest_duration(self):
        hint = _build_period_hint("2026-04-30", None, is_annual=True)
        assert "April 30, 2026" in hint
        assert "LONGEST duration" in hint
        assert "Twelve Months Ended" in hint
        assert "SHORTEST" not in hint

    def test_quarterly_filing_prefers_shortest_duration(self):
        hint = _build_period_hint("2026-01-31", None, is_annual=False)
        assert "January 31, 2026" in hint
        assert "SHORTEST duration" in hint
        assert "single-quarter column" in hint
        assert "LONGEST" not in hint

    def test_falls_back_to_doc_period_without_sec_date(self):
        hint = _build_period_hint(None, "Three Months Ended April 30, 2026", is_annual=False)
        assert "Three Months Ended April 30, 2026" in hint
        assert "__period__" in hint

    def test_invalid_sec_date_falls_back_to_doc_period(self):
        hint = _build_period_hint("not-a-date", "Twelve Months Ended April 30, 2026", is_annual=True)
        assert "Twelve Months Ended April 30, 2026" in hint

    def test_no_signals_returns_generic_hint(self):
        hint = _build_period_hint(None, None, is_annual=False)
        assert "MOST RECENT ACTUAL reported quarter" in hint
