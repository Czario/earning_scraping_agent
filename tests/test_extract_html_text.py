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


# ── _get_table_context ───────────────────────────────────────────────────────


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

