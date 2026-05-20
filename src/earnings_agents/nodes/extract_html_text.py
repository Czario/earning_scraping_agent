from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

from earnings_agents.config import HTTP_TIMEOUT
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
    url = state.get("discovered_file_url", "")
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

        # Convert financial tables to markdown before text extraction.
        # This preserves column structure (period headers, side-by-side values)
        # that get_text() would otherwise destroy, making multi-period
        # column selection reliable for the LLM.
        for table in soup.find_all("table"):
            table.replace_with(_table_to_markdown(table) + "\n")

        raw_text = soup.get_text(separator="\n", strip=True)
        raw_text = _strip_boilerplate(raw_text)
        logger.info("HTML extracted %d chars for %s", len(raw_text), ticker)
        return {**state, "raw_text": raw_text, "status": "text_extracted"}
    except Exception as exc:  # noqa: BLE001
        return {**state, "status": "failed", "error": f"HTML extraction failed: {exc}"}

