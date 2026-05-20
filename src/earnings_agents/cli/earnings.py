"""Command-line interface for the earnings scraping pipeline.

Pass one or more company identifiers and the graph runs for each.

Usage examples:
    # SEC EDGAR path (default) — looks up latest 8-K Exhibit 99.1 automatically
    uv run earnings --cik 0000320193
    uv run earnings --ticker AAPL MSFT GOOGL
    uv run earnings --source sec --cik 0000320193 0000789019

    # IR website path — LLM scans the IR page to find the earnings release URL
    uv run earnings --source ir --ticker AAPL
    uv run earnings --source ir --cik 0000320193 --ir-url https://investor.apple.com/news/press-releases/default.aspx

Source flag behaviour:
  --source sec (default)
      Queries SEC EDGAR submissions API for the latest 8-K Item 2.02 filing,
      extracts Exhibit 99.1 URL, and injects it directly into the graph
    (the discover_earnings_release node is skipped).

  --source ir
      Uses the company's Investor Relations website. The discover_earnings_release node
      fetches the IR page, extracts all links, and asks Ollama to identify
      the earnings release URL.
      IR URL resolution order:
        1. --ir-url argument (applies to every company in this run)
        2. COMPANIES dict in config.py (per-ticker hard-coded URL)
        3. Error — cannot proceed without an IR URL
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from earnings_agents.config import COMPANIES  # noqa: E402
from earnings_agents.workflow import build_graph  # noqa: E402
from earnings_agents.company_registry import lookup_by_cik, lookup_by_ticker  # noqa: E402
from earnings_agents.tools.edgar_client import get_latest_earnings_url  # noqa: E402

SEP = "=" * 64


def _resolve_companies(ciks: list[str], tickers: list[str]) -> list[dict]:
    """Return a list of company dicts ready to feed into the graph."""
    companies: list[dict] = []

    for cik in ciks:
        info = lookup_by_cik(cik)
        if not info:
            print(f"[WARN] CIK {cik} not found in tickers.json — skipping")
            continue
        companies.append(info)

    for t in tickers:
        info = lookup_by_ticker(t)
        if not info:
            print(f"[WARN] Ticker {t.upper()} not found in tickers.json — skipping")
            continue
        companies.append(info)

    return companies


def _build_initial_state(info: dict, source: str = "sec", ir_url_override: str = "") -> dict:
    """Build the LangGraph initial state for one company.

    source="sec"  → query SEC EDGAR for the latest 8-K Exhibit 99.1 and
                     inject the URL directly (discover_earnings_release node is skipped).
    source="ir"   → use the company's IR website; the discover_earnings_release node
                     fetches the page and asks Ollama to find the earnings URL.
                     IR URL is taken from ir_url_override, then COMPANIES config.
    """
    ticker = info.get("ticker") or ""
    company_name = info["company_name"]
    cik = info["cik"]

    _base = {
        "ticker": ticker or cik,
        "company_name": company_name,
        "file_type": None,
        "raw_text": None,
        "metrics": None,
        "error": None,
        "extraction_attempts": 0,
        "extraction_notes": None,
    }

    if source == "ir":
        # Resolve IR URL: CLI override > COMPANIES config > error
        ir_url = ir_url_override or COMPANIES.get(ticker.upper(), {}).get("ir_url", "")
        if not ir_url:
            return {
                **_base,
                "ir_url": "",
                "discovered_file_url": None,
                "status": "failed",
                "error": (
                    f"No IR URL for {ticker or cik}. "
                    "Provide --ir-url or add the ticker to COMPANIES in config.py."
                ),
            }
        print(f"  [IR]     {company_name} ({ticker or cik}) → {ir_url}")
        return {
            **_base,
            "ir_url": ir_url,
            "discovered_file_url": None,
            "status": "pending",
        }

    # source == "sec" (default)
    print(f"  [EDGAR]  {company_name} ({ticker or cik}) querying SEC EDGAR...")
    filing_url = get_latest_earnings_url(cik)
    if not filing_url:
        return {
            **_base,
            "ir_url": "",
            "discovered_file_url": None,
            "status": "failed",
            "error": f"No 8-K earnings filing found on SEC EDGAR for CIK {cik}",
        }
    return {
        **_base,
        "ir_url": "",
        "discovered_file_url": filing_url,
        # Skip IR discovery — jump straight to file-type detection
        "status": "discovered",
    }


def _run_company(graph, info: dict, source: str = "sec", ir_url_override: str = "") -> dict:
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    print(f"\n{SEP}")
    print(f"  Company : {label}")
    print(f"  CIK     : {info['cik']}")
    print(f"  Source  : {source.upper()}")

    state = _build_initial_state(info, source=source, ir_url_override=ir_url_override)

    if state["status"] == "failed":
        print(f"  [SKIP]  {state['error']}")
        print(SEP)
        return state

    if state["status"] != "failed":
        if state.get("discovered_file_url"):
            print(f"  Filing  : {state['discovered_file_url']}")
        else:
            print(f"  IR URL  : {state['ir_url']}")
    print(SEP)

    final = graph.invoke(state)

    print(f"\n  Status  : {final.get('status')}")
    print(f"  File URL: {final.get('discovered_file_url')}")
    print(f"  Type    : {final.get('file_type')}")
    if final.get("metrics"):
        m = final["metrics"]
        print(f"  Metrics ({len(m)} fields):")
        for label, value in m.items():
            print(f"    {label:<40} {value}")
    if final.get("error"):
        print(f"  Error   : {final.get('error')}")
    print(SEP)

    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the earnings scraping pipeline for one or more companies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run earnings --cik 0000320193\n"
            "  uv run earnings --cik 0000320193 0000789019\n"
            "  uv run earnings --ticker AAPL MSFT\n"
        ),
    )
    parser.add_argument(
        "--cik",
        nargs="+",
        metavar="CIK",
        default=[],
        help="One or more CIK numbers (with or without leading zeros)",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="TICKER",
        default=[],
        help="One or more ticker symbols (e.g. AAPL MSFT)",
    )
    parser.add_argument(
        "--source",
        choices=["ir", "sec"],
        default="sec",
        help="URL discovery method: 'sec' (default) queries SEC EDGAR; 'ir' scrapes the company's IR website via LLM.",
    )
    parser.add_argument(
        "--ir-url",
        metavar="URL",
        default="",
        help="IR website URL to use when --source ir is set (overrides COMPANIES config for all companies in this run).",
    )
    args = parser.parse_args()

    if not args.cik and not args.ticker:
        parser.error("Provide at least one --cik or --ticker argument.")

    if args.ir_url and args.source != "ir":
        parser.error("--ir-url is only meaningful with --source ir.")

    companies = _resolve_companies(args.cik, args.ticker)
    if not companies:
        print("No valid companies resolved. Exiting.")
        sys.exit(1)

    graph = build_graph()
    results = [_run_company(graph, c, source=args.source, ir_url_override=args.ir_url) for c in companies]

    failed = [r for r in results if r.get("status") != "saved"]
    print(f"\nDone: {len(results) - len(failed)}/{len(results)} succeeded.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

