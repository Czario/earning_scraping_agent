"""Tests for HTML table classification helpers in extract_html_text."""
from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from earnings_agents.nodes.extract_html_text import (
    _classify_table,
    _get_table_context,
    _llm_classify_other_batch,
)


def _make_table(rows: list[str]) -> BeautifulSoup:
    """Return a BeautifulSoup object whose first element is a <table>."""
    rows_html = "".join(f"<tr><td>{r}</td></tr>" for r in rows)
    return BeautifulSoup(f"<table>{rows_html}</table>", "lxml")


# ── _classify_table ─────────────────────────────────────────────────────────


class TestClassifyTable:
    # --- income statement ---

    def test_net_sales_income_statement(self):
        table_text = "Net sales $ 5,529,145 $ 5,033,094 Membership fee income 132,355"
        assert _classify_table(table_text, "") == "income_statement"

    def test_total_revenues_income_statement(self):
        table_text = "Total revenues 12,456,789 Operating income 1,234 Net income 890"
        assert _classify_table(table_text, "") == "income_statement"

    def test_net_revenues_income_statement(self):
        table_text = "Net revenues 8,000 Cost of goods sold 6,000 Gross profit 2,000"
        assert _classify_table(table_text, "") == "income_statement"

    def test_statement_of_operations_in_context(self):
        ctx = "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS"
        table_text = "Revenue 100 Expenses 80 Net income 20"
        assert _classify_table(table_text, ctx) == "income_statement"

    # --- balance sheet ---

    def test_total_assets_balance_sheet(self):
        table_text = "ASSETS Current assets Cash 27,826 Total assets 4,321,000"
        assert _classify_table(table_text, "") == "balance_sheet"

    def test_balance_sheet_in_context(self):
        ctx = "CONDENSED CONSOLIDATED BALANCE SHEETS"
        table_text = "Cash 50,000 Inventory 100,000 Total liabilities 200,000"
        assert _classify_table(table_text, ctx) == "balance_sheet"

    # --- cash flow ---

    def test_cash_flows_from_operating(self):
        table_text = "CASH FLOWS FROM OPERATING ACTIVITIES Net income 142,726"
        assert _classify_table(table_text, "") == "cash_flow"

    def test_net_cash_provided_by_operating(self):
        table_text = "Net cash provided by operating activities 139,958"
        # This matches cash_flow UNLESS the full probe also contains non-GAAP keywords
        assert _classify_table(table_text, "") == "cash_flow"

    def test_statement_of_cash_flows_context(self):
        ctx = "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS"
        table_text = (
            "CASH FLOWS FROM OPERATING ACTIVITIES Net income 142,726 "
            "Depreciation 90,156 Changes in working capital (10,000) "
            "Net cash provided by operating activities 139,958"
        )
        assert _classify_table(table_text, ctx) == "cash_flow"

    # --- non_gaap ---

    def test_adjusted_net_income_non_gaap(self):
        table_text = (
            "Net income as reported 142,726 "
            "Adjustments Restructuring 0 "
            "Adjusted net income 142,726"
        )
        assert _classify_table(table_text, "") == "non_gaap"

    def test_ebitda_table_non_gaap(self):
        table_text = "Net income 142,726 Interest 12,367 Taxes 52,820 D&A 90,156 EBITDA 298,070"
        assert _classify_table(table_text, "") == "non_gaap"

    def test_free_cash_flow_table_non_gaap(self):
        table_text = (
            "Net cash provided by operating activities 139,958 "
            "Less Additions to property (71,942) "
            "Free cash flow 68,016"
        )
        assert _classify_table(table_text, "") == "non_gaap"

    def test_non_gaap_keyword_in_context_wins(self):
        # Context says "Non-GAAP reconciliation" → should win even if table looks like income stmt
        ctx = "Reconciliation of GAAP to Non-GAAP Financial Measures"
        table_text = "Net income 100 Adjustments 10 Adjusted net income 110"
        assert _classify_table(table_text, ctx) == "non_gaap"

    def test_non_gaap_overrides_income_statement_keywords(self):
        # Table has "net sales" AND "Adjusted EBITDA" — non_gaap must win
        table_text = "Net sales 5,000 Net income 200 Adjusted EBITDA 300"
        assert _classify_table(table_text, "") == "non_gaap"

    def test_reconciliation_context_non_gaap(self):
        ctx = "Reconciliation to adjusted EBITDA (Amounts in thousands)"
        table_text = "Net income 142,726 Interest 12,367 D&A 90,156"
        assert _classify_table(table_text, ctx) == "non_gaap"

    # --- other ---

    def test_no_keywords_returns_other(self):
        table_text = "Quarter Q1 Q2 Q3 Store count 243 248 252"
        assert _classify_table(table_text, "") == "other"

    def test_empty_table_returns_other(self):
        assert _classify_table("", "") == "other"


# ── segment table classification ────────────────────────────────────────────


class TestSegmentTableClassification:
    """Segment/geographic/product revenue tables must classify as 'segment'
    (bypassing the LLM filter) so they always reach the extraction LLM."""

    # --- context-heading driven (real-world patterns) ---

    def test_divisional_revenues_nike(self):
        # Nike's exact heading: "DIVISIONAL REVENUES"
        ctx = "NIKE, Inc. DIVISIONAL REVENUES (Unaudited)"
        table_text = "North America Footwear 3230 Apparel 1310 Equipment 292 Total 4832"
        assert _classify_table(table_text, ctx) == "segment"

    def test_divisional_revenues_with_nongaap_footnote_in_body(self):
        # Critical: Nike's DIVISIONAL REVENUES table embeds a footnote cell
        # "currency-neutral basis is considered a non-GAAP financial measure."
        # The full table text triggers _NON_GAAP_TABLE_RX — but the context
        # check fires FIRST, so the table is still classified as segment.
        ctx = "NIKE, Inc. DIVISIONAL REVENUES (Unaudited)"
        table_text = (
            "North America Footwear 3230 Total 4832 "
            "1 The percent change calculated using actual exchange rates "
            "is considered a non-GAAP financial measure."
        )
        assert _classify_table(table_text, ctx) == "segment"

    def test_net_sales_by_reportable_segment_apple(self):
        # Apple-style heading
        ctx = "Net Sales by Reportable Segment"
        table_text = "Americas 40,000 Europe 25,000 Greater China 18,000 Total 83,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_net_sales_by_category(self):
        ctx = "Net Sales by Category"
        table_text = "Products 57,000 Services 26,000 Total net sales 83,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_revenue_by_geography_context(self):
        ctx = "Revenue by Geography"
        table_text = "Americas 45,000 Europe 22,000 Asia Pacific 18,000 Total 85,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_revenue_by_segment_context(self):
        ctx = "Revenue by Segment"
        table_text = "Cloud 35,000 Devices 15,000 Gaming 5,000 Total 55,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_geographic_revenue_adjective_form(self):
        ctx = "Geographic Revenue"
        table_text = "North America 80,000 International 40,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_segment_results_heading(self):
        ctx = "Segment Results"
        table_text = "Intelligent Cloud 28,500 Productivity 25,100 Personal Computing 13,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_segment_information_heading(self):
        ctx = "Segment Information"
        table_text = "Segment A 100 Segment B 200 Total 300"
        assert _classify_table(table_text, ctx) == "segment"

    def test_supplemental_revenue_data(self):
        ctx = "Supplemental Revenue Data"
        table_text = "Product revenue 60,000 Service revenue 25,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_revenue_breakdown_by_in_context(self):
        ctx = "Revenue breakdown by product line"
        table_text = "Hardware 30,000 Software 50,000 Services 20,000"
        assert _classify_table(table_text, ctx) == "segment"

    def test_operating_segments_heading(self):
        ctx = "Operating Segments"
        table_text = "Segment A Revenue 100,000 Segment B Revenue 50,000"
        assert _classify_table(table_text, ctx) == "segment"

    # --- priority / guard tests ---

    def test_non_gaap_context_wins_over_segment(self):
        # Context explicitly says "Non-GAAP" → guard blocks segment, falls
        # through to full-probe non_gaap check.
        ctx = "Non-GAAP Adjusted Segment Revenue"
        table_text = "Cloud (GAAP) 35,000 Adjustments (2,000) Cloud (Non-GAAP) 33,000"
        assert _classify_table(table_text, ctx) == "non_gaap"

    def test_reconciliation_context_wins_over_segment(self):
        ctx = "Reconciliation of GAAP to Adjusted Segment Revenue"
        table_text = "Segment A GAAP 100,000 Adjustments (5,000) Adjusted 95,000"
        assert _classify_table(table_text, ctx) == "non_gaap"

    def test_segment_context_wins_over_nongaap_footnote_in_table_body(self):
        # Segment context is authoritative; non-GAAP only in footnote inside
        # the table body must NOT reclassify a GAAP segment table.
        ctx = "Revenue by Segment"
        table_text = "Cloud 35,000 Services 20,000 1 non-GAAP measure for reporting purposes"
        assert _classify_table(table_text, ctx) == "segment"

    def test_income_statement_keywords_in_body_do_not_override_segment_context(self):
        # Context matches segment → returns "segment" before GAAP body checks.
        ctx = "Revenue by Segment (Three Months Ended)"
        table_text = "Cloud Net Revenue 35,000 Services Net Revenue 25,000"
        assert _classify_table(table_text, ctx) == "segment"




class TestGetTableContext:
    def test_captures_direct_previous_sibling(self):
        soup = BeautifulSoup(
            "<div>"
            "<p>(Amounts in thousands, except per share amounts)</p>"
            "<table><tr><td>Net sales 100</td></tr></table>"
            "</div>",
            "lxml",
        )
        table = soup.find("table")
        ctx = _get_table_context(table)
        assert "(Amounts in thousands" in ctx

    def test_captures_multiple_preceding_siblings(self):
        soup = BeautifulSoup(
            "<div>"
            "<h4>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS</h4>"
            "<p>(In thousands)</p>"
            "<table><tr><td>Revenue 100</td></tr></table>"
            "</div>",
            "lxml",
        )
        table = soup.find("table")
        ctx = _get_table_context(table)
        assert "STATEMENTS OF OPERATIONS" in ctx
        assert "(In thousands)" in ctx

    def test_no_siblings_returns_empty(self):
        soup = BeautifulSoup(
            "<div><table><tr><td>Revenue 100</td></tr></table></div>",
            "lxml",
        )
        table = soup.find("table")
        ctx = _get_table_context(table)
        assert ctx == ""

    def test_does_not_include_parent_siblings(self):
        """Parent-level text must NOT bleed into context (prevents distant
        non-GAAP disclaimers from contaminating context for GAAP tables)."""
        soup = BeautifulSoup(
            "<body>"
            "<div><p>Non-GAAP Financial Measures We refer to non-GAAP metrics</p></div>"
            "<div><table><tr><td>Net sales 5,000</td></tr></table></div>"
            "</body>",
            "lxml",
        )
        table = soup.find("table")
        ctx = _get_table_context(table)
        # The non-GAAP disclaimer is in a parent-sibling div, not a direct sibling
        assert "Non-GAAP" not in ctx

    def test_decompose_previous_table_isolates_context(self):
        """After decomposing table 0, table 1's context should only include
        text between the two tables, not table 0's rows."""
        soup = BeautifulSoup(
            "<div>"
            "<table id='t0'><tr><td>Adjusted net income 142,726</td></tr></table>"
            "<p>Scale: (Amounts in thousands)</p>"
            "<table id='t1'><tr><td>Net sales 5,529,145</td></tr></table>"
            "</div>",
            "lxml",
        )
        tables = list(soup.find_all("table"))
        tables[0].decompose()  # Simulate processing table 0 first
        ctx = _get_table_context(tables[1])
        assert "Adjusted net income" not in ctx
        assert "(Amounts in thousands)" in ctx


# ── End-to-end classification via HTML snippet ───────────────────────────────


class TestClassifyFromHTML:
    """Verify that the full classify_table + get_table_context pipeline
    correctly identifies GAAP vs non-GAAP tables from realistic HTML."""

    _INCOME_STMT_HTML = """
    <div>
      <p>(Amounts in thousands, except per share amounts)</p>
      <table>
        <tr><th>Thirteen Weeks Ended</th><th>May 2, 2026</th><th>May 3, 2025</th></tr>
        <tr><td>Net sales</td><td>5,529,145</td><td>5,033,094</td></tr>
        <tr><td>Membership fee income</td><td>132,355</td><td>120,401</td></tr>
        <tr><td>Total revenues</td><td>5,661,500</td><td>5,153,495</td></tr>
        <tr><td>Net income</td><td>142,726</td><td>149,768</td></tr>
      </table>
    </div>
    """

    _NON_GAAP_HTML = """
    <div>
      <h4>Reconciliation of net income to Adjusted net income</h4>
      <table>
        <tr><td>Net income as reported</td><td>142,726</td></tr>
        <tr><td>Adjustments: Restructuring</td><td>0</td></tr>
        <tr><td>Adjusted net income</td><td>142,726</td></tr>
      </table>
    </div>
    """

    def test_income_statement_html(self):
        soup = BeautifulSoup(self._INCOME_STMT_HTML, "lxml")
        table = soup.find("table")
        ctx = _get_table_context(table)
        ttype = _classify_table(table.get_text(" ", strip=True), ctx)
        assert ttype == "income_statement"

    def test_non_gaap_html(self):
        soup = BeautifulSoup(self._NON_GAAP_HTML, "lxml")
        table = soup.find("table")
        ctx = _get_table_context(table)
        ttype = _classify_table(table.get_text(" ", strip=True), ctx)
        assert ttype == "non_gaap"


# ── _llm_classify_other_batch ────────────────────────────────────────────────


class _FakeLLM:
    """Minimal LLM stub: always returns the given JSON string."""

    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, prompt: str) -> str:  # noqa: ARG002
        return self._response


class TestLlmClassifyOtherBatch:
    def test_all_useful_returns_all_true(self):
        llm = _FakeLLM('{"0": true, "1": true}')
        result = _llm_classify_other_batch(
            [("Revenue 81,615 Cost 20,458", ""), ("Gross profit 61,157", "")],
            llm,
        )
        assert result == [True, True]

    def test_mixed_useful_and_skip(self):
        llm = _FakeLLM('{"0": true, "1": false}')
        result = _llm_classify_other_batch(
            [("Revenue 81,615", ""), ("Toshiya Hari toshiyah@nvidia.com", "")],
            llm,
        )
        assert result == [True, False]

    def test_footnote_table_returns_false(self):
        llm = _FakeLLM('{"0": false}')
        result = _llm_classify_other_batch(
            [("(A) Acquisition-related costs Cost of revenue: 47 R&D: 167", "")],
            llm,
        )
        assert result == [False]

    def test_parse_error_falls_back_to_all_true(self):
        llm = _FakeLLM("not json at all")
        result = _llm_classify_other_batch(
            [("table A", ""), ("table B", "")],
            llm,
        )
        assert result == [True, True]

    def test_missing_index_falls_back_to_true(self):
        # JSON only has "0", missing "1" → default True for "1"
        llm = _FakeLLM('{"0": false}')
        result = _llm_classify_other_batch(
            [("table A", ""), ("table B", "")],
            llm,
        )
        assert result == [False, True]

    def test_empty_candidates_returns_empty(self):
        llm = _FakeLLM("{}")
        assert _llm_classify_other_batch([], llm) == []

    def test_context_and_table_appear_in_prompt(self):
        received: list[str] = []

        class _CaptureLLM:
            def invoke(self, prompt: str) -> str:
                received.append(prompt)
                return '{"0": true}'

        _llm_classify_other_batch([("TABLE CONTENT", "HEADING TEXT")], _CaptureLLM())
        assert "TABLE CONTENT" in received[0]
        assert "HEADING TEXT" in received[0]


# ── Document-level scale re-attachment (CrowdStrike regression) ──────────────


class TestDocumentScaleInjection:
    """extract_html_text_node re-attaches the document scale caption to GAAP
    statements when the caption is dropped by boilerplate stripping.

    CrowdStrike prints "(in thousands, except per share amounts)" in a <div>
    that is neither a table cell nor a direct sibling of the income statement
    table, and places a "Forward-Looking Statements" boilerplate marker BEFORE
    the financial statements. The boilerplate stripper then removes the caption
    from the prose, leaving table values 1000x too small. The node captures the
    scale from the full document before stripping and re-attaches it.
    """

    # HTML mirroring CRWD's structure: the scale caption sits in a standalone
    # <div> that is NOT a direct previous sibling of the income statement table
    # (the table lives inside its own wrapper <div>), so `_get_table_context`
    # cannot see it. A "Forward-Looking Statements" marker appears in the second
    # half before the statements, so `_strip_boilerplate` removes the caption
    # from the prose. The injection path is the only thing that can re-attach it.
    _CRWD_LIKE_HTML = (
        "<html><body>"
        "<p>CrowdStrike Reports First Quarter Fiscal Year 2027 Results</p>"
        "<p>Total revenue was $1.39 billion, up 26%.</p>"
        "<p>Forward-Looking Statements This press release contains "
        "forward-looking statements within the meaning of the Securities Act.</p>"
        "<div><font>(in thousands, except per share amounts)</font></div>"
        "<div>"
        "<p>Condensed Consolidated Statements of Operations</p>"
        "<table>"
        "<tr><td>Total revenue</td><td>1,385,629</td></tr>"
        "<tr><td>Net income (loss)</td><td>45,966</td></tr>"
        "</table>"
        "</div>"
        "</body></html>"
    )

    def _run(self, monkeypatch, html: str):
        import earnings_agents.nodes.extract_html_text as mod

        class _Resp:
            text = html

        monkeypatch.setattr(mod, "_http_get", lambda url, sec=False: _Resp())
        return mod.extract_html_text_node(
            {
                "discovered_file_url": "https://www.sec.gov/x.htm",
                "file_type": "html",
                "status": "fetched",
                "ticker": "CRWD",
            }
        )

    def test_scale_reattached_to_income_statement(self, monkeypatch):
        out = self._run(monkeypatch, self._CRWD_LIKE_HTML)
        inc = out["raw_sections"]["income_statement"]
        assert inc, "income statement section should be populated"
        assert "(in thousands)" in inc[0]
        # Caption must survive into raw_text so the downstream prescan fires.
        assert "thousand" in out["raw_text"].lower()

    def test_downstream_prescan_detects_scale(self, monkeypatch):
        from earnings_agents.extraction.chunker import _prescan_document

        out = self._run(monkeypatch, self._CRWD_LIKE_HTML)
        scale, _, _ = _prescan_document(out["raw_text"])
        assert scale == "thousands"

    def test_existing_table_scale_not_overridden(self, monkeypatch):
        """When the table already states its own scale, the document scale is
        NOT injected on top of it (no double caption)."""
        html = (
            "<html><body>"
            "<p>Forward-Looking Statements blah blah blah blah blah blah.</p>"
            "<p>Statements of Operations</p>"
            "<div><font>(in thousands)</font></div>"
            "<p>(In millions)</p>"
            "<table><tr><td>Total revenue</td><td>1,385</td></tr></table>"
            "</body></html>"
        )
        out = self._run(monkeypatch, html)
        inc = out["raw_sections"]["income_statement"][0]
        # The table's own "(In millions)" caption is preserved and we do NOT
        # prepend a conflicting "(in thousands)" document caption.
        assert "(In millions)" in inc
        assert not inc.startswith("(in thousands)")

    def test_scale_caption_nested_several_layers_up(self, monkeypatch):
        """Caption wrapped in nested spans/divs (not a direct sibling) is still
        found — structure-agnostic backward search, not direct-sibling only."""
        html = (
            "<html><body>"
            "<p>Forward-Looking Statements aaaa bbbb cccc dddd eeee ffff.</p>"
            "<div><section><span><font>(in thousands)</font></span></section></div>"
            "<article><div><p>Statements of Operations</p>"
            "<table><tr><td>Total revenue</td><td>1,385,629</td></tr></table>"
            "</div></article>"
            "</body></html>"
        )
        out = self._run(monkeypatch, html)
        inc = out["raw_sections"]["income_statement"][0]
        assert "(in thousands)" in inc

    def test_mixed_scale_per_statement(self, monkeypatch):
        """A filing whose statements use DIFFERENT scales keeps the correct
        scale on each — the nearest preceding caption wins, not a single
        document-wide value."""
        html = (
            "<html><body>"
            "<p>Forward-Looking Statements aaaa bbbb cccc dddd eeee ffff gggg.</p>"
            "<div><font>(in thousands)</font></div>"
            "<div><p>Condensed Statements of Operations</p>"
            "<table><tr><td>Total revenue</td><td>1,385,629</td></tr></table>"
            "</div>"
            "<div><font>(in millions)</font></div>"
            "<div><p>Condensed Balance Sheets</p>"
            "<table><tr><td>Total assets</td><td>7,123</td></tr></table>"
            "</div>"
            "</body></html>"
        )
        out = self._run(monkeypatch, html)
        inc = out["raw_sections"]["income_statement"][0]
        bal = out["raw_sections"]["balance_sheet"][0]
        assert "(in thousands)" in inc
        assert "(in millions)" in bal

    def test_no_scale_anywhere_injects_nothing(self, monkeypatch):
        """A filing with no scale caption anywhere gets no injected caption
        (values are full dollars / handled by the LLM as-is)."""
        html = (
            "<html><body>"
            "<p>Condensed Statements of Operations</p>"
            "<table><tr><td>Total revenue</td><td>1385629000</td></tr></table>"
            "</body></html>"
        )
        out = self._run(monkeypatch, html)
        inc = out["raw_sections"]["income_statement"][0]
        assert "(in " not in inc.lower()

    def test_narrative_million_not_injected_as_scale(self, monkeypatch):
        """Narrative '$256 million' before the table must NOT be treated as a
        scale caption (no spurious '(in millions)' injection)."""
        html = (
            "<html><body>"
            "<p>Achieves record net new ARR of $256 million, up 32%.</p>"
            "<p>Condensed Statements of Operations</p>"
            "<table><tr><td>Total revenue</td><td>1385629000</td></tr></table>"
            "</body></html>"
        )
        out = self._run(monkeypatch, html)
        inc = out["raw_sections"]["income_statement"][0]
        assert "(in millions)" not in inc.lower()

