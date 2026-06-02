from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

from earnings_agents.config import HTTP_TIMEOUT
from earnings_agents.llm_factory import build_llm
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.tools.playwright_scraper import fetch_page_js

logger = logging.getLogger(__name__)

# Tags that carry no earnings content
_NOISE_TAGS = frozenset(
    {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}
)

# SEC EDGAR programmatic access requires a non-browser User-Agent with contact info
_SEC_HEADERS = {
    "User-Agent": "earning-agents data-pipeline@truegrids.com",
    "Accept-Encoding": "gzip, deflate",
}

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Minimum meaningful content length; below this we assume JS rendering is needed
_MIN_CONTENT_CHARS = 300

# ── HTML table classification ───────────────────────────────────────────────
# Each 8-K press release typically contains several HTML tables:
#   • 1-2 GAAP income statement tables (sometimes with a % growth variant)
#   • 1 GAAP balance sheet table
#   • 1 GAAP cash flow table
#   • N non-GAAP reconciliation tables (Adjusted income, EBITDA, FCF, Net debt…)
#
# Mixing these into one flat text blob causes the LLM to:
#   – Read non-GAAP "Adjusted net income" instead of GAAP "Net income"
#   – Get confused by narrative dollar amounts ("$132.4 million") that use a
#     different scale than the table values ("132,355" in thousands)
#
# These patterns classify tables by their content + preceding heading text.
# non_gaap is checked FIRST because non-GAAP tables often contain metric names
# ("Net income") that would otherwise trigger positive GAAP classification.

_NON_GAAP_TABLE_RX = re.compile(
    r"non.?gaap"
    r"|reconciliation\s+of\s+(?:gaap|net\s+income|adjusted)"
    r"|\badjusted\s+(?:ebitda|net\s+income|operating\s+income|earnings)\b"
    r"|\bebitda\b"
    r"|\bfree\s+cash\s+flow\b",
    re.I,
)

_INCOME_STMT_TABLE_RX = re.compile(
    r"statement[s]?\s+of\s+(?:operations|income|earnings|loss)"
    r"|total\s+revenue[s]?\b"
    r"|net\s+revenue[s]?\b"
    r"|net\s+sales\b",
    re.I,
)
_BALANCE_SHEET_TABLE_RX = re.compile(
    r"balance\s+sheet[s]?"
    r"|financial\s+position"
    r"|\btotal\s+assets\b",
    re.I,
)
_CASH_FLOW_TABLE_RX = re.compile(
    r"cash\s+flow[s]?"
    r"|statement[s]?\s+of\s+cash"
    r"|net\s+cash\s+(?:provided|used)\s+by\s+operating",
    re.I,
)


def _get_table_context(table, max_chars: int = 400) -> str:
    """Return up to *max_chars* of text from DOM elements immediately preceding *table*.

    Only looks at **direct previous siblings** of the table element — not at
    ancestor siblings.  This keeps the context tightly scoped to the heading
    and scale indicator that appear between the prior table and this one (e.g.
    "Reconciliation to adjusted EBITDA" or "(In thousands)"), and prevents
    distant sections of the press release (non-GAAP disclaimers, earlier
    narrative) from contaminating the classification probe.

    Because tables are decomposed one at a time before this is called for the
    next table, the previous siblings only cover text *between* consecutive
    tables — not the rows of the preceding table itself.
    """
    parts: list[str] = []
    total = 0

    for sib in table.previous_siblings:
        if total >= max_chars:
            break
        t = sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib).strip()
        if t:
            parts.append(t)
            total += len(t)

    return "\n".join(reversed(parts))


def _classify_table(table_text: str, context_text: str) -> str:
    """Classify a financial table using fast keyword regex.

    Returns one of: ``'income_statement'``, ``'balance_sheet'``, ``'cash_flow'``,
    ``'non_gaap'``, or ``'other'`` (catch-all for tables that don't match any
    keyword pattern).

    ``non_gaap`` is checked against the *full* table text because reconciliation
    tables often have the key phrase ('Adjusted net income') near the bottom.
    Positive GAAP classifications use a shorter probe to avoid false matches on
    non-GAAP tables that share metric names (e.g. 'Net income as reported').

    Tables that fall through to ``'other'`` are subsequently screened by
    :func:`_llm_classify_other` inside the extraction node.
    """
    full_probe = context_text + " " + table_text
    if _NON_GAAP_TABLE_RX.search(full_probe):
        return "non_gaap"
    short_probe = context_text[-600:] + " " + table_text[:600]
    if _INCOME_STMT_TABLE_RX.search(short_probe):
        return "income_statement"
    if _CASH_FLOW_TABLE_RX.search(short_probe):
        return "cash_flow"
    if _BALANCE_SHEET_TABLE_RX.search(short_probe):
        return "balance_sheet"
    return "other"


_OTHER_FILTER_PROMPT = """\
You are classifying tables from an earnings press release.

For each table below, decide whether it contains PRIMARY GAAP financial statement
data worth extracting — rows with numeric values for revenue, cost, expenses,
income, EPS, or share counts.

Mark useful=true for: primary income-statement rows, condensed GAAP summaries.
Mark useful=false for:
- Contact blocks (names, email addresses, phone numbers, job titles)
- Footnote annotation tables (rows beginning with (A), (B), * etc.)
- Supplementary stock-based compensation breakdowns
- Guidance / outlook tables
- Any other non-primary-statement content

Tables to classify:
{tables_block}

Return ONLY a JSON object mapping each table index (as a string) to true or false:
{{"0": true, "1": false, ...}}"""


def _llm_classify_other_batch(
    candidates: list[tuple[str, str]],
    llm: Any,
) -> list[bool]:
    """Classify a batch of 'other' tables in a single LLM call.

    ``candidates`` is a list of ``(table_text, context_text)`` pairs.
    Returns a parallel list of booleans: ``True`` = keep, ``False`` = skip.
    Falls back to all-True on any error so no table is silently dropped.
    """
    if not candidates:
        return []
    blocks: list[str] = []
    for i, (table_text, context_text) in enumerate(candidates):
        ctx = context_text[-300:].strip() or "(none)"
        blocks.append(
            f"Table {i}\nContext: {ctx}\nContent: {table_text[:600]}"
        )
    prompt = _OTHER_FILTER_PROMPT.format(tables_block="\n\n".join(blocks))
    try:
        raw = llm.invoke(prompt)
        data = json.loads(raw)
        return [bool(data.get(str(i), True)) for i in range(len(candidates))]
    except Exception:
        return [True] * len(candidates)  # safe fallback — keep all


# Boilerplate section markers that contain no financial data.
# Only searched in the second half of the document to avoid accidentally cutting
# the beginning of a document that opens with a disclaimer.
_BOILERPLATE_RX = re.compile(
    r"\n+(?:About |ABOUT )[A-Za-z]"
    r"|\n+Forward[- ]Looking Statements?"
    r"|\n+FORWARD[- ]LOOKING STATEMENTS?"
    r"|\n+Cautionary (?:Note|Statement)"
    r"|\n+Safe Harbor Statement",
    re.IGNORECASE,
)


def _table_to_markdown(table) -> str:
    """Convert an HTML <table> to pipe-delimited markdown rows.

    Preserves column alignment so the LLM can correctly identify which values
    belong to which period column (e.g. Q1 2026 vs Q1 2025).
    Flattening tables with get_text() destroys this structure entirely.
    """
    lines: list[str] = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
        if any(cells):  # skip completely empty rows
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _strip_boilerplate(text: str) -> str:
    """Remove trailing boilerplate sections (About company, Safe Harbor, etc.).

    Only strips content in the second half of the document to avoid removing
    financial data that appears near the top of the press release.
    """
    mid = len(text) // 2
    m = _BOILERPLATE_RX.search(text, mid)
    if m:
        stripped = text[: m.start()].rstrip()
        logger.debug("Boilerplate stripped: %d → %d chars", len(text), len(stripped))
        return stripped
    return text


def _pick_headers(url: str) -> dict:
    """Return SEC-specific headers for EDGAR URLs, browser headers otherwise."""
    if "sec.gov" in url:
        return _SEC_HEADERS
    return _BROWSER_HEADERS


def _strip_sgml_wrapper(html: str) -> str:
    """Extract the HTML payload from an EDGAR SGML wrapper if present.

    EDGAR archive files are often wrapped in SGML::

        <DOCUMENT>
        <TYPE>EX-99.1
        ...
        <TEXT>
        <html>...</html>
        </TEXT>
        </DOCUMENT>

    This function returns the content after the ``<TEXT>`` tag so that
    BeautifulSoup only sees valid HTML.
    """
    if "<DOCUMENT>" not in html.upper():
        return html
    match = re.search(r"<TEXT>(.*)", html, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else html


def extract_html_text_node(state: EarningsAgentState) -> EarningsAgentState:
    """Fetch an HTML earnings page and extract clean article text.

    Handles:
    - SEC EDGAR programmatic User-Agent requirement
    - EDGAR SGML document wrappers
    - JS-rendered pages (Playwright fallback for non-SEC URLs)
    """
    # TypedDict.get may still infer Optional[str]; normalize to plain str
    # so substring checks ("sec.gov" in/not in url) are type-safe.
    url = state.get("discovered_file_url") or ""
    ticker = state["ticker"]

    try:
        headers = _pick_headers(url)
        response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        html = response.text

        # Unwrap EDGAR SGML envelope before parsing
        html = _strip_sgml_wrapper(html)

        # Detect JS-gated pages (non-SEC only — SEC archives are static)
        if "sec.gov" not in url:
            quick_text = BeautifulSoup(html, "lxml").get_text(strip=True)
            if len(quick_text) < _MIN_CONTENT_CHARS:
                logger.info("Static HTML appears JS-rendered for %s — trying Playwright", url)
                html = fetch_page_js(url)

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # ── Table-aware structured extraction ────────────────────────────────
        # Classify each HTML table by financial statement type, then assemble
        # raw_text so GAAP tables appear first and non-GAAP reconciliation
        # tables are isolated at the end with a clear label.
        #
        # Processing order matters: each table is decomposed from the soup
        # before the next table's context is collected, so `_get_table_context`
        # only sees text *between* consecutive tables (not the prior table's
        # rows), keeping the scale/period header attached to the right table.
        sections: dict[str, list[str]] = {
            "income_statement": [],
            "balance_sheet": [],
            "cash_flow": [],
            "other": [],
            "non_gaap": [],
        }
        has_gaap_tables = False

        # Pending 'other' tables — classified in one batched LLM call after the loop.
        # Each entry: (table_markdown_entry, table_text_for_llm)
        _other_pending: list[tuple[str, str]] = []

        for table in list(soup.find_all("table")):
            context = _get_table_context(table)
            table_text = table.get_text(" ", strip=True)
            ttype = _classify_table(table_text, context)
            md = _table_to_markdown(table)
            entry = (f"{context}\n" if context.strip() else "") + md
            if ttype == "other":
                _other_pending.append((entry, table_text, context))
            else:
                sections[ttype].append(entry)
                if ttype in ("income_statement", "balance_sheet", "cash_flow"):
                    has_gaap_tables = True
            table.decompose()  # Remove before processing next table

        # Batch-classify all 'other' tables in a single LLM call.
        if _other_pending:
            llm = build_llm(format_json=True)
            candidates = [(t, c) for _, t, c in _other_pending]
            keep_flags = _llm_classify_other_batch(candidates, llm)
            skipped = 0
            for (entry, _, _ctx), keep in zip(_other_pending, keep_flags):
                if keep:
                    sections["other"].append(entry)
                else:
                    skipped += 1
            if skipped:
                logger.debug(
                    "LLM batch filter: skipped %d/%d 'other' table(s) for %s",
                    skipped, len(_other_pending), ticker,
                )

        # Prose text with all tables removed
        prose = soup.get_text(separator="\n", strip=True)
        prose = _strip_boilerplate(prose)

        if has_gaap_tables:
            # Structured output: GAAP tables first → narrative → non-GAAP
            parts: list[str] = []
            for stmt_type, label in [
                ("income_statement", "GAAP INCOME STATEMENT"),
                ("balance_sheet",    "GAAP BALANCE SHEET"),
                ("cash_flow",        "GAAP CASH FLOWS"),
                ("other",            "FINANCIAL DATA"),
            ]:
                for entry in sections[stmt_type]:
                    parts.append(f"=== {label} ===\n{entry}")
            if prose.strip():
                parts.append(f"=== NARRATIVE ===\n{prose}")
            for entry in sections["non_gaap"]:
                parts.append(
                    "=== NON-GAAP (FOR REFERENCE ONLY — DO NOT USE FOR GAAP METRICS) ===\n"
                    + entry
                )
            raw_text = "\n\n".join(parts)
            logger.info(
                "HTML extracted %d chars for %s — %d income, %d balance, "
                "%d cashflow, %d other, %d non-gaap table(s)",
                len(raw_text), ticker,
                len(sections["income_statement"]), len(sections["balance_sheet"]),
                len(sections["cash_flow"]), len(sections["other"]),
                len(sections["non_gaap"]),
            )
            return {
                **state,
                "raw_text": raw_text,
                "raw_sections": sections,
                "status": "text_extracted",
            }
        else:
            # Fallback: no GAAP tables classified — convert all tables to
            # markdown inline (original behaviour) so nothing is lost.
            soup2 = BeautifulSoup(html, "lxml")
            for tag in soup2(_NOISE_TAGS):
                tag.decompose()
            for tbl in soup2.find_all("table"):
                tbl.replace_with(_table_to_markdown(tbl) + "\n")
            raw_text = soup2.get_text(separator="\n", strip=True)
            raw_text = _strip_boilerplate(raw_text)
            logger.info(
                "HTML extracted %d chars for %s (fallback — no GAAP tables classified)",
                len(raw_text), ticker,
            )
            return {**state, "raw_text": raw_text, "status": "text_extracted"}
    except Exception as exc:  # noqa: BLE001
        return {**state, "status": "failed", "error": f"HTML extraction failed: {exc}"}

