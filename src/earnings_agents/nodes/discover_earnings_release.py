from __future__ import annotations

import logging

from earnings_agents.config import IR_PAGE_MAX_CHARS
from earnings_agents.llm_factory import build_llm
from earnings_agents.workflow_state import EarningsAgentState
from earnings_agents.tools.llm_link_picker import DISCOVERY_SCHEMA, pick_earnings_url
from earnings_agents.tools.playwright_scraper import fetch_page_js
from earnings_agents.tools.static_scraper import extract_links, fetch_page

logger = logging.getLogger(__name__)


def _is_valid_url(url: str) -> bool:
    """Return True only for absolute http/https URLs."""
    return bool(url and (url.startswith("http://") or url.startswith("https://")))


def discover_earnings_release_node(state: EarningsAgentState) -> EarningsAgentState:
    """Fetch the company IR page and ask the LLM to identify the earnings release URL.

    If ``discovered_file_url`` is already set in the state (e.g. pre-resolved via
    SEC EDGAR by the CLI), the node is skipped and the state is passed through.

    Strategy for IR-page discovery:
    1. Try a plain HTTP fetch first (fast, no JS overhead).
    2. If the response is empty or too short, fall back to Playwright.
    3. Extract all anchor links and pass a compact list to Ollama.
    4. Parse the LLM JSON response to get the target URL.
    """
    # Short-circuit: URL already resolved upstream (e.g. via EDGAR)
    if state.get("discovered_file_url"):
        logger.info(
            "IR discovery skipped for %s — URL already provided: %s",
            state["ticker"],
            state["discovered_file_url"],
        )
        return {**state, "status": "discovered"}

    ticker = state["ticker"]
    ir_url = state["ir_url"]


    # ── 1. Fetch IR page ──────────────────────────────────────────────────────
    html, ok = fetch_page(ir_url)
    if not ok or len(html.strip()) < 500:
        logger.info("Static fetch insufficient for %s — falling back to Playwright", ticker)
        html = fetch_page_js(ir_url)

    if not html:
        return {
            **state,
            "status": "failed",
            "error": f"Could not fetch IR page for {ticker}",
        }

    # ── 2. Extract links ──────────────────────────────────────────────────────
    links = extract_links(html, ir_url)
    if not links:
        return {
            **state,
            "status": "failed",
            "error": f"No links found on IR page for {ticker}",
        }

    # ── 3. Build compact link list for LLM ───────────────────────────────────
    link_lines = [
        f"- [{lk['text'][:120]}]({lk['url']})"
        for lk in links
        if lk.get("text")
    ]
    links_text = "\n".join(link_lines)[:IR_PAGE_MAX_CHARS]

    # ── 4. Ask LLM ───────────────────────────────────────────────────────────
    llm = build_llm(json_schema=DISCOVERY_SCHEMA)
    try:
        file_url = pick_earnings_url(
            links_text=links_text,
            company_name=state["company_name"],
            ticker=ticker,
            llm=llm,
        )
        return {**state, "discovered_file_url": file_url, "status": "discovered"}
    except (ValueError, KeyError) as exc:
        return {
            **state,
            "status": "failed",
            "error": f"LLM IR discovery parsing failed for {ticker}: {exc}",
        }
