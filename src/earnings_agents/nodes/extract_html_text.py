from __future__ import annotations

import logging
import re
from typing import Any
from bs4 import BeautifulSoup

from earnings_agents.extraction.chunker import _prescan_document
from earnings_agents.llm_factory import build_llm
from earnings_agents.tools.http_client import SEC_HEADERS as _SEC_HEADERS
from earnings_agents.tools.http_client import BROWSER_HEADERS as _BROWSER_HEADERS
from earnings_agents.tools.http_client import get as _http_get
from earnings_agents.tools.llm_table_classifier import classify_other_tables_batch as _classify_other_tables_batch
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.tools.playwright_scraper import fetch_page_js

logger = logging.getLogger(__name__)

# Tags that carry no earnings content
_NOISE_TAGS = frozenset(
    {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form"}
)

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

# ── Deterministic "drop" patterns for 'other' tables ────────────────────────
# These catch the common junk categories the LLM batch filter decides on, so
# tables matching any of these are dropped without an LLM call.
# Only the clearly junk categories are listed — anything ambiguous is left to
# the LLM fallback (accuracy is never sacrificed for speed).

# Contact / IR info blocks: email addresses, US phone formats, IR keywords.
_CONTACT_BLOCK_RX = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"   # email
    r"|\(\d{3}\)\s*\d{3}[\-.\s]\d{4}"                           # (xxx) xxx-xxxx
    r"|\b\d{3}[\-.\s]\d{3}[\-.\s]\d{4}\b"                       # xxx-xxx-xxxx
    r"|investor\s+relations"
    r"|media\s+contact"
    r"|press\s+(?:contact|release\s+contact)",
    re.I,
)

# Footnote annotation tables: >30 % of non-empty rows begin with a lettered /
# numbered / symbol footnote marker.  Checked in Python (count-based) rather
# than a single regex so short tables with one real row aren't misclassified.
_FOOTNOTE_MARKER_RX = re.compile(
    r"^\s*(?:\([A-Za-z]\)|\([0-9]+\)|\*{1,3}|†|‡|\d+\))\s",
    re.MULTILINE,
)

# Stock-based / share-based compensation disclosure tables.
_STOCK_COMP_RX = re.compile(
    r"stock[- ]based\s+compensation\s+(?:included\s+in|expense\s+by|by\s+)"
    r"|share[- ]based\s+(?:compensation|payment)\s+(?:included|expense\s+by)",
    re.I,
)

# Guidance / outlook / forecast tables.
# Uses anchored phrases to avoid false matches on period headers like
# "Fiscal Year Ended" or narrative like "we expect revenue to grow".
_GUIDANCE_RX = re.compile(
    r"\bguidance\b"
    r"|\boutlook\b"
    r"|\bfull[- ]year\s+(?:guidance|outlook|target|expectation|range)\b"
    r"|\bfiscal\s+20\d\d\s+(?:guidance|outlook|target|expectation)\b"
    r"|\bq[1-4]\s+20\d\d\s+(?:guidance|outlook|target)\b",
    re.I,
)


def _is_footnote_table(table_text: str) -> bool:
    """Return True when the majority of non-empty rows are footnote annotations."""
    lines = [l for l in table_text.splitlines() if l.strip()]
    if not lines:
        return False
    marker_count = sum(1 for l in lines if _FOOTNOTE_MARKER_RX.match(l))
    return marker_count / len(lines) >= 0.30


# Positively identifies supplementary GAAP revenue-detail tables:
# revenue by segment, geography, product line, business unit, etc.
#
# These patterns are matched against the table CONTEXT (heading) only, BEFORE
# the full-probe non_gaap check runs. Many real-world segment tables carry a
# footnote cell like "currency-neutral basis is considered a non-GAAP measure"
# — that footnote refers only to a percentage column, not to the dollar values.
# Matching on context (never the table body) prevents that footnote text from
# triggering the non_gaap classification on the entire table.
#
# Safety: the caller also checks that the context itself does NOT match
# _NON_GAAP_TABLE_RX, so "Non-GAAP Revenue by Segment" is never misclassified.
_SEGMENT_TABLE_RX = re.compile(
    # "Revenue/Sales by X" — e.g. "Revenue by Geography",
    # "Net Sales by Reportable Segment" (Apple), "Sales by Region"
    r"(?:revenues?|net\s+(?:revenues?|sales)|sales)\s+by\s+"
    r"(?:segment|reportable\s+segment|geography|region|product|category|type|channel|business|division)\b"
    # Adjective-noun — e.g. "Geographic Revenue", "Divisional Revenues" (Nike),
    # "Segment Results", "Product Breakdown"
    r"|(?:segment|geographic|product|category|regional|divisional)\s+"
    r"(?:revenues?|net\s+(?:revenues?|sales)|results|breakdown)\b"
    # Bare "Divisional Revenues" (Nike) not already caught above
    r"|\bdivisional\s+revenues?\b"
    # "Revenue/Sales breakdown/detail/mix by …"
    r"|(?:revenues?|sales)\s+(?:breakdown|detail|mix)\s+by\b"
    # Supplemental financial data
    r"|\bsupplemental\s+(?:revenue|financial)\s+(?:data|information|detail)\b"
    # "Segment information/data/results/summary"
    r"|\bsegment(?:al)?\s+(?:information|data|details?|summary|results)\b"
    # "Operating Segments" section heading
    r"|\boperating\s+segments?\b",
    re.I,
)

# Detects whether a table entry already carries its own scale caption, so a
# document/preceding scale is only injected when the table itself says nothing
# — never overriding a table that declares its own scale. ``[^\S\n]`` matches
# horizontal whitespace only, so a caption split by a non-breaking space
# ("in\xa0thousands") still matches.
_ENTRY_HAS_SCALE_RX = re.compile(r"\bin[^\S\n]+(?:thousands|millions|billions)\b", re.I)


def _find_preceding_scale(table) -> str | None:
    """Return the scale ("thousands"/"millions"/"billions") of the NEAREST scale
    caption that appears before *table* anywhere in the document, or ``None``.

    Issuers place the "(in thousands)" caption in wildly different DOM
    structures: inside a table cell, in a direct sibling ``<div>``, in a
    ``<font>`` wrapped several layers up, or in a heading block separated from
    the statement by other elements. ``_get_table_context`` only inspects
    *direct previous siblings*, so it misses every caption that isn't a direct
    sibling. This helper is structure-agnostic: it walks backward through the
    document in reading order (``find_all_previous``) and returns the first
    (i.e. nearest-preceding) string that matches the chunker's tuned scale
    patterns. Because tables are decomposed as they are processed, prior table
    bodies are already gone, so only captions/headings/prose are considered —
    and the parenthesised/line-start patterns reject narrative phrasing such as
    "net new ARR of $256 million". Returning the *nearest* caption also keeps
    the right scale on each statement when a filing mixes scales across tables.
    """
    for s in table.find_all_previous(string=True):
        scale, _, _ = _prescan_document(str(s))
        if scale:
            return scale
    return None


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
    ``'segment'``, ``'non_gaap'``, or ``'other'``.

    Classification order (earlier wins):

    1. **Segment context** — if the heading (context) matches *_SEGMENT_TABLE_RX*
       AND does NOT itself contain non-GAAP keywords, return ``'segment'``
       immediately.  This must fire before the full-probe non-GAAP check because
       many GAAP segment tables carry a footnote cell such as "currency-neutral
       figures are a non-GAAP measure" — that footnote describes only a
       percentage column and should never cause the entire table (which contains
       GAAP dollar revenues) to be suppressed.  Checking context-only is safe
       because genuine non-GAAP tables are excluded by the guard on the context.

    2. **Full-probe non-GAAP** — ``non_gaap`` is checked against the full table
       text because reconciliation tables often have the key phrase ('Adjusted
       net income') near the bottom, not the top.

    3. **Short-probe GAAP statements** — income statement, cash flow, balance
       sheet checked on context + first 600 chars to avoid false matches on
       non-GAAP tables that share metric names ('Net income as reported').

    4. **Short-probe segment fallback** — catches segment tables where the
       context heading is ambiguous but the opening rows contain segment keywords.

    5. **other** — catch-all; subsequently screened by the LLM batch filter.
    """
    # ── 1. Segment context check (BEFORE full-probe non_gaap) ────────────────
    # Match only on context (heading text), never the table body, to avoid
    # footnote-text contamination. Reject if the context itself advertises
    # non-GAAP content — e.g. "Non-GAAP Revenue by Segment".
    if _SEGMENT_TABLE_RX.search(context_text) and not _NON_GAAP_TABLE_RX.search(context_text):
        return "segment"

    # ── 2. Full-probe non-GAAP ────────────────────────────────────────────────
    full_probe = context_text + " " + table_text
    if _NON_GAAP_TABLE_RX.search(full_probe):
        # Guard: many bank filing tables contain non-GAAP footnotes or a handful
        # of "Adjusted" metric rows inside an otherwise GAAP statement table
        # (e.g. "Adjusted net income" in a Consolidated Statements of Income
        # table).  The heading / opening context determines the primary type.
        #
        # A table is treated as GAAP (not non-GAAP) when:
        #   (a) The context (preceding heading) identifies it as IS/BS/CF AND
        #       the context itself does NOT contain non-GAAP keywords, OR
        #   (b) The TABLE'S OWN opening text (first 600 chars) identifies it
        #       as IS/BS/CF AND that opening section is non-GAAP-free — this
        #       catches tables whose headings are inside the table body (common
        #       in press releases where "Statement of Income Highlights" is a
        #       data row rather than a preceding DOM element), OR
        #   (c) Segment-table guard (original logic): full body matches a
        #       segment pattern AND context is non-GAAP-free.
        context_ngaap = _NON_GAAP_TABLE_RX.search(context_text)
        opening = table_text[:600]
        is_gaap_by_context = (
            not context_ngaap
            and (_INCOME_STMT_TABLE_RX.search(context_text)
                 or _BALANCE_SHEET_TABLE_RX.search(context_text)
                 or _CASH_FLOW_TABLE_RX.search(context_text))
        )
        is_gaap_by_opening = (
            not context_text.strip()
            and not _NON_GAAP_TABLE_RX.search(opening)
            and (_INCOME_STMT_TABLE_RX.search(opening)
                 or _BALANCE_SHEET_TABLE_RX.search(opening)
                 or _CASH_FLOW_TABLE_RX.search(opening))
        )
        is_segment = _SEGMENT_TABLE_RX.search(full_probe) and not context_ngaap
        if not (is_gaap_by_context or is_gaap_by_opening or is_segment):
            return "non_gaap"
        logger.debug(
            "Non-GAAP keywords found but table identified as GAAP via "
            "context/opening — falling through to GAAP checks "
            "(context=%r..., opening=%r...)",
            context_text[:80], opening[:80],
        )

    # ── 3. Short-probe GAAP statement checks ─────────────────────────────────
    short_probe = context_text[-600:] + " " + table_text[:600]
    if _INCOME_STMT_TABLE_RX.search(short_probe):
        return "income_statement"
    if _CASH_FLOW_TABLE_RX.search(short_probe):
        return "cash_flow"
    if _BALANCE_SHEET_TABLE_RX.search(short_probe):
        return "balance_sheet"

    # ── 4. Short-probe segment fallback ──────────────────────────────────────
    if _SEGMENT_TABLE_RX.search(short_probe):
        return "segment"

    # ── 5. Full-probe segment fallback ───────────────────────────────────────
    # Catches tables where the segment heading appears deeper in the body
    # (e.g. after a "(Dollars in millions)" caption row).
    if _SEGMENT_TABLE_RX.search(full_probe):
        return "segment"

    # ── 6. Deterministic drop patterns ───────────────────────────────────────
    # Catch well-known junk categories without an LLM call.  Only patterns
    # that are unambiguous are listed here — anything uncertain stays as
    # "other" and is handled by the LLM batch filter.
    if _CONTACT_BLOCK_RX.search(full_probe):
        return "drop"
    if _STOCK_COMP_RX.search(full_probe):
        return "drop"
    if _GUIDANCE_RX.search(context_text):
        return "drop"
    if _is_footnote_table(table_text):
        return "drop"

    return "other"


_OTHER_FILTER_PROMPT = """(moved to tools/llm_table_classifier)"""


def _llm_classify_other_batch(
    candidates: list[tuple[str, str]],
    llm: Any,
) -> list[bool]:
    """Backward-compatible wrapper — delegates to tools.llm_table_classifier."""
    return _classify_other_tables_batch(candidates, llm)


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


def _classify_document_tables(
    soup: BeautifulSoup,
    doc_scale: str | None,
    skip_llm_filter: bool = False,
) -> tuple[dict[str, list[str]], bool, str]:
    """Classify all HTML tables in *soup* into financial statement sections.

    Applies the same classification pipeline as the main EX-99.1 processing:
    table-type detection (income_statement, balance_sheet, cash_flow, segment,
    non_gaap, other), scale caption injection for GAAP tables.

    When ``skip_llm_filter=True``, the LLM batch filter for 'other' tables is
    deferred — pending entries are returned so the caller can batch them across
    multiple documents into a single LLM call.

    **Modifies *soup* in place** — all ``<table>`` elements are decomposed
    during classification so prose extraction is table-free.

    Returns:
        sections:   ``{income_statement: [], balance_sheet: [], cash_flow: [],
                    other: [], non_gaap: []}`` — each value is a list of
                    markdown-formatted table entries with leading context.
                    When ``skip_llm_filter=True``, "other" entries that passed
                    deterministic checks are already included; the unfiltered
                    pending batch is returned separately.
        has_gaap:   ``True`` if at least one table was classified as a GAAP
                    statement type.
        prose:      All non-table text (noise tags removed, boilerplate stripped).
        pending:    ``None`` when ``skip_llm_filter=False``.  Otherwise, a list
                    of ``(markdown_entry, table_text, context)`` tuples for
                    tables classified as ``'other'`` that still need LLM batch
                    filtering.
    """
    sections: dict[str, list[str]] = {
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": [],
        "other": [],
        "non_gaap": [],
    }
    has_gaap = False

    # Pending 'other' tables — classified in one batched LLM call after the loop.
    _other_pending: list[tuple[str, str, str]] = []

    for table in list(soup.find_all("table")):
        context = _get_table_context(table)
        table_text = table.get_text(" ", strip=True)
        ttype = _classify_table(table_text, context)
        logger.debug(
            "Table classified as %r — context=%r... body_start=%r...",
            ttype,
            context[:100],
            table_text[:100],
        )
        md = _table_to_markdown(table)
        entry = (f"{context}\n" if context.strip() else "") + md
        # Re-attach a scale caption to GAAP statements whose own text says
        # nothing about scale.
        if ttype in ("income_statement", "balance_sheet", "cash_flow") and not _ENTRY_HAS_SCALE_RX.search(entry):
            table_scale = _find_preceding_scale(table) or doc_scale
            if table_scale:
                entry = f"(in {table_scale})\n{entry}"
        if ttype == "other":
            _other_pending.append((entry, table_text, context))
        elif ttype == "drop":
            logger.debug(
                "Table deterministically dropped (regex) — context=%r...",
                context[:80],
            )
        elif ttype == "segment":
            if not _ENTRY_HAS_SCALE_RX.search(entry):
                table_scale = _find_preceding_scale(table) or doc_scale
                if table_scale:
                    entry = f"(in {table_scale})\n{entry}"
            sections["other"].append(entry)
            has_gaap = True
        else:
            sections[ttype].append(entry)
            if ttype in ("income_statement", "balance_sheet", "cash_flow"):
                has_gaap = True
        table.decompose()

    # Batch-classify all 'other' tables.
    pending: list | None = None
    if _other_pending and not skip_llm_filter:
        from earnings_agents.config import LLM_PROVIDER as _LLM_PROVIDER
        from earnings_agents.hooks import report_call as _report_call
        _report_call(f"  [llm]  classify {len(_other_pending)} 'other' table(s)  → calling llm  ({_LLM_PROVIDER or 'llm'})")
        llm = build_llm(format_json=True)
        candidates = [(t, c) for _, t, c in _other_pending]
        keep_flags = _llm_classify_other_batch(candidates, llm)
        kept = 0
        for (entry, _, _ctx), keep in zip(_other_pending, keep_flags):
            if keep:
                sections["other"].append(entry)
                kept += 1
        if kept < len(_other_pending):
            from earnings_agents.hooks import report_call as _report_call2
            _report_call2(
                f"  [filter]  kept {kept}/{len(_other_pending)} 'other' table(s)"
                f"  (dropped {len(_other_pending) - kept})"
            )
    elif _other_pending:
        # Defer LLM filter: return pending for batched processing upstream.
        pending = _other_pending

    # Prose text with all tables removed
    prose = soup.get_text(separator="\n", strip=True)
    prose = _strip_boilerplate(prose)

    return sections, has_gaap, prose, pending


def _fetch_and_classify_supplement(
    url: str, ticker: str, index: int,
) -> tuple[dict[str, list[str]], str, list] | None:
    """Fetch a supplemental exhibit and classify its tables using the same
    pipeline as the main document.

    Returns ``(sections, prose, pending)`` where *sections* is a classified
    table dict and *pending* is a list of ``(entry, table_text, context)``
    for 'other' tables that still need LLM batch filtering, or ``None`` if
    already filtered.  Returns ``None`` on fetch/parse failure.
    """
    from earnings_agents.hooks import report_call
    try:
        ex_label = f"supplement {index + 2}"
        report_call(f"  [http]  GET supplemental {url[:80]}")
        is_sec = "sec.gov" in url
        response = _http_get(url, sec=is_sec)
        supp_html = response.text

        supp_html = _strip_sgml_wrapper(supp_html)

        supp_soup = BeautifulSoup(supp_html, "lxml")
        for tag in supp_soup(_NOISE_TAGS):
            tag.decompose()

        supp_scale, _, _ = _prescan_document(supp_soup.get_text(" ", strip=True))
        supp_sections, _, supp_prose, supp_pending = _classify_document_tables(
            supp_soup, supp_scale, skip_llm_filter=True,
        )

        # Count total tables across all sections
        total_tables = sum(len(v) for v in supp_sections.values())

        # Skip exhibits that are investor presentations (slide decks, charts)
        # rather than financial data supplements.  Presentations have few or no
        # extractable HTML tables (they are image/chart-based), so they add
        # prompt size without useful extraction targets.
        if total_tables == 0:
            report_call(
                f"  [{ex_label}]  no financial tables found — "
                f"skipping (likely investor presentation)"
            )
            logger.info(
                "Supplemental exhibit %d for %s: no financial tables found — "
                "skipping (likely investor presentation)",
                index + 2, ticker,
            )
            return None

        is_c = len(supp_sections["income_statement"])
        bs_c = len(supp_sections["balance_sheet"])
        cf_c = len(supp_sections["cash_flow"])
        oth_c = len(supp_sections["other"])
        ngaap_c = len(supp_sections["non_gaap"])
        logger.info(
            "Supplemental exhibit %d for %s: %d tables (%d IS, %d BS, %d CF, "
            "%d other, %d non-gaap)",
            index + 2, ticker, total_tables,
            is_c, bs_c, cf_c, oth_c, ngaap_c,
        )

        return supp_sections, supp_prose, supp_pending
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to fetch supplemental exhibit %s for %s: %s",
            url, ticker, exc,
        )
        return None


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
        from earnings_agents.hooks import report_call
        is_sec = "sec.gov" in url
        report_call(f"  [http]  GET {url[:80]}")
        response = _http_get(url, sec=is_sec)
        html = response.text

        # Unwrap EDGAR SGML envelope before parsing
        html = _strip_sgml_wrapper(html)

        # Detect JS-gated pages (non-SEC only — SEC archives are static)
        if "sec.gov" not in url:
            quick_text = BeautifulSoup(html, "lxml").get_text(strip=True)
            if len(quick_text) < _MIN_CONTENT_CHARS:
                logger.info("Static HTML appears JS-rendered for %s — trying Playwright", url)
                report_call(f"  [playwright]  JS render  {url[:80]}")
                html = fetch_page_js(url)

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # ── Document-level scale fallback ────────────────────────────────────
        # Scan the FULL document text for a scale caption (e.g. "(in
        # thousands)") *before* tables are decomposed and *before* boilerplate
        # stripping runs. This is the fallback used when a table has no scale
        # caption anywhere before it (e.g. the caption sits after the table, or
        # the only caption is far away). Capturing it here matters because some
        # issuers place a "Forward-Looking Statements" boilerplate marker ahead
        # of the financial statements, so `_strip_boilerplate` would otherwise
        # delete the caption before it ever reaches the LLM — leaving values
        # 1000x too small. Per-table detection (`_find_preceding_scale`) is
        # preferred over this document-wide value so mixed-scale filings keep
        # the correct scale on each statement.
        doc_scale, _, _ = _prescan_document(soup.get_text(" ", strip=True))

        # ── Table-aware structured extraction ────────────────────────────────
        # Classify each HTML table by financial statement type, then assemble
        # raw_text so GAAP tables appear first and non-GAAP reconciliation
        # tables are isolated at the end with a clear label.
        # Collect 'other' table pending entries from the MAIN document for
        # batched LLM filtering.  Supplement 'other' tables are always excluded
        # (IS-only merge policy).
        all_pending: list[tuple[str, str, str]] = []

        sections, has_gaap_tables, prose, pending = _classify_document_tables(
            soup, doc_scale, skip_llm_filter=True,
        )
        if pending:
            all_pending.extend(pending)

        # Capture main document table counts for CLI display (before supplement merge)
        main_is = len(sections["income_statement"])
        main_bs = len(sections["balance_sheet"])
        main_cf = len(sections["cash_flow"])
        main_other = len(sections["other"])
        main_ngaap = len(sections["non_gaap"])
        main_total = main_is + main_bs + main_cf + main_other + main_ngaap
        report_call(f"  [main doc]      {main_total} tables ({main_is} IS, {main_bs} BS, {main_cf} CF, {main_other} other)")

        # ── Process supplemental exhibits (EX-99.2, EX-99.3, ...) ──────────────
        # Each supplemental document is parsed and classified independently using
        # the same pipeline as the main EX-99.1 document.  Table results are
        # merged into the main sections with a provenance prefix so the LLM
        # extractor sees which exhibit each table originated from.
        supplemental_urls = state.get("supplemental_file_urls") or []
        supp_blocks: list[str] = []  # raw_text blocks, assembled below
        # Only IS tables go to extraction now (BS/CF/other/non-GAAP excluded).
        supplemental_log: list[str] = [
            f"  [main doc]      {main_is} IS → extraction"
            + (f"  (classified {main_total} total: {main_bs} BS, {main_other} other, {main_ngaap} non-gaap — not sent)" if main_total > main_is else "")
        ]
        for supp_idx, supp_url in enumerate(supplemental_urls):
            report_call(f"  [supplement {supp_idx+2}]  processing  {supp_url.rsplit('/', 1)[-1]}")
            result = _fetch_and_classify_supplement(supp_url, ticker, supp_idx)
            if result is None:
                supplemental_log.append(
                    f"  [supplement {supp_idx+2}]   skipped"
                )
                continue
            supp_sections, supp_prose, supp_pending = result
            # Supplement 'other' tables are never included in extraction
            # (IS-only merge).  Don't add them to the batched filter either
            # since their results are discarded — saves an LLM call.
            # (Pending from main doc *is* added above and still needs filtering.)
            # Only merge income_statement tables from supplements.
            # Balance sheet, cash flow, "other" (non-IS financial data like loan
            # portfolios, capital ratios, deposit data), and non-GAAP tables are
            # excluded — targeted extraction focuses on income-statement concepts,
            # so only explicitly IS-classified tables contribute relevant data.
            is_count = len(supp_sections["income_statement"])
            bs_count = len(supp_sections["balance_sheet"])
            cf_count = len(supp_sections["cash_flow"])
            other_count = len(supp_sections["other"])
            ngaap_count = len(supp_sections["non_gaap"])
            prefix = f"[SUPPLEMENTAL EXHIBIT ({supp_idx + 2})] "
            for entry in supp_sections["income_statement"]:
                sections["income_statement"].append(prefix + entry)
                has_gaap_tables = True
            if supp_prose.strip() and is_count:
                supp_blocks.append(f"=== SUPPLEMENTAL EXHIBIT ({supp_idx + 2}) NARRATIVE ===\n{prefix}{supp_prose}")
            supplemental_log.append(
                f"  [supplement {supp_idx+2}]   {is_count} IS → extraction"
            )
            logger.info(
                "Supplemental exhibit %d merged for %s (%d IS tables)",
                supp_idx + 2, ticker, is_count,
            )

        # ── LLM filter for main doc 'other' tables ──────────────────────────────
        # Supplement 'other' tables are excluded (IS-only merge) so only main doc
        # pending entries need the LLM batch filter.
        if all_pending:
            from earnings_agents.config import LLM_PROVIDER as _LLM_PROVIDER
            report_call(
                f"  [llm]  classify {len(all_pending)} 'other' table(s)"
                f"  (across all exhibits)  → calling llm  ({_LLM_PROVIDER or 'llm'})"
            )
            llm = build_llm(format_json=True)
            candidates = [(t, c) for _, t, c in all_pending]
            keep_flags = _llm_classify_other_batch(candidates, llm)
            kept = 0
            for (entry, _, _ctx), keep in zip(all_pending, keep_flags):
                if keep:
                    sections["other"].append(entry)
                    kept += 1
            report_call(
                f"  [filter]  kept {kept}/{len(all_pending)} 'other' table(s)"
                f"  (dropped {len(all_pending) - kept})"
            )
            if kept:
                has_gaap_tables = True

        if has_gaap_tables:
            # Structured output: only income_statement tables are included.
            # Balance sheet, cash flow, "other", and non-GAAP sections are
            # excluded — targeted extraction focuses on IS concepts only.
            parts: list[str] = []
            for entry in sections["income_statement"]:
                parts.append(f"=== GAAP INCOME STATEMENT ===\n{entry}")
            if prose.strip() and sections["income_statement"]:
                parts.append(f"=== NARRATIVE ===\n{prose}")
            # Append supplemental narrative blocks at the end of raw_text
            raw_text = "\n\n".join(parts)
            if supp_blocks:
                raw_text += "\n\n" + "\n\n".join(supp_blocks)

            logger.info(
                "HTML extracted %d chars for %s — %d IS table(s) to extraction%s",
                len(raw_text), ticker,
                len(sections["income_statement"]),
                f"  (+{len(supp_blocks)} supplemental narrative)" if supp_blocks else "",
            )

            # ── Table summary ─────────────────────────────────────────────────
            # Show IS tables going to extraction by source.
            _is_entries = sections.get("income_statement", [])
            if _is_entries:
                _main = sum(1 for e in _is_entries if not e.startswith("[SUPPLEMENTAL"))
                _supp = len(_is_entries) - _main
                _parts = []
                if _main:
                    _parts.append(f"{_main} main")
                if _supp:
                    _parts.append(f"{_supp} supplement")
                report_call(
                    f"  [IS → LLM]  {len(_is_entries)} table(s)"
                    + (f"  ({', '.join(_parts)})" if _parts else "")
                )

            return {
                **state,
                "raw_text": raw_text,
                "raw_sections": sections,
                "_supplemental_log": supplemental_log,
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

            # Append supplemental narrative blocks (tables already merged above)
            if supp_blocks:
                raw_text += "\n\n" + "\n\n".join(supp_blocks)

            return {**state, "raw_text": raw_text, "_supplemental_log": supplemental_log, "status": "text_extracted"}
    except Exception as exc:  # noqa: BLE001
        return {**state, "status": "failed", "error": f"HTML extraction failed: {exc}"}

