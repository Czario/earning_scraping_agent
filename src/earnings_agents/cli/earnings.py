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
import concurrent.futures
import logging
import sys

import requests
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from earnings_agents.config import (  # noqa: E402
    COMPANIES,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from earnings_agents.nodes.detect_document_type import detect_document_type_node  # noqa: E402
from earnings_agents.workflow import build_graph  # noqa: E402
from earnings_agents.company_registry import lookup_by_cik, lookup_by_ticker  # noqa: E402
from earnings_agents.tools.edgar_client import get_latest_earnings_url  # noqa: E402
from earnings_agents.hooks import set_detail_callback, set_node_callback  # noqa: E402

SEP = "=" * 64

# Human-readable stage names for the rich progress description
_NODE_LABELS: dict[str, str] = {
    "discover_earnings_release_node": "discover",
    "detect_document_type_node": "detect type",
    "extract_html_text_node": "fetch text",
    "extract_pdf_text_node": "fetch pdf",
    "extract_financial_metrics_node": "extract metrics",
    "reflect_metrics_node": "reflect",
    "mongodb_save_node": "save",
}


def _check_ollama() -> tuple[bool, str]:
    """Return (ok, detail) for Ollama reachability and model availability."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, f"Ollama responded {resp.status_code}"
        available = [m["name"] for m in resp.json().get("models", [])]
        base_name = OLLAMA_MODEL.split(":")[0]
        if not any(m.startswith(base_name) for m in available):
            return False, f"Model '{OLLAMA_MODEL}' not pulled (available: {available or 'none'})"
        return True, f"Ollama OK — model '{OLLAMA_MODEL}' available"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama unreachable: {exc}"


def _check_llm() -> tuple[bool, str]:
    """Return (ok, detail) for the selected LLM provider configuration."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            return False, "OpenAI API key missing (set OPENAI_API_KEY in .env)"
        return True, f"OpenAI configured — model '{OPENAI_MODEL}'"
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            return False, "Groq API key missing (set GROQ_API_KEY in .env)"
        return True, f"Groq configured — model '{GROQ_MODEL}'"
    return _check_ollama()


def _check_mongodb() -> tuple[bool, str]:
    """Return (ok, detail) for MongoDB reachability."""
    try:
        from pymongo import MongoClient  # local import — only needed in dry-run path

        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return True, f"MongoDB OK — {MONGODB_DB}.{MONGODB_COLLECTION}"
    except Exception as exc:  # noqa: BLE001
        return False, f"MongoDB unreachable: {exc}"


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


def _build_initial_state(info: dict, source: str = "sec", ir_url_override: str = "", printer=print) -> dict:
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
        printer(f"  [IR]     {company_name} ({ticker or cik}) → {ir_url}")
        return {
            **_base,
            "ir_url": ir_url,
            "discovered_file_url": None,
            "status": "pending",
        }

    # source == "sec" (default)
    printer(f"  [EDGAR]  {company_name} ({ticker or cik}) querying SEC EDGAR...")
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


def _run_company(graph, info: dict, source: str = "sec", ir_url_override: str = "", printer=print) -> dict:
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    printer(f"\n{SEP}")
    printer(f"  Company : {label}")
    printer(f"  CIK     : {info['cik']}")
    printer(f"  Source  : {source.upper()}")

    state = _build_initial_state(info, source=source, ir_url_override=ir_url_override, printer=printer)

    if state["status"] == "failed":
        printer(f"  [SKIP]  {state['error']}")
        printer(SEP)
        return state

    if state["status"] != "failed":
        if state.get("discovered_file_url"):
            printer(f"  Filing  : {state['discovered_file_url']}")
        else:
            printer(f"  IR URL  : {state['ir_url']}")
    printer(SEP)

    final = graph.invoke(state)

    printer(f"\n  Status  : {final.get('status')}")
    printer(f"  File URL: {final.get('discovered_file_url')}")
    printer(f"  Type    : {final.get('file_type')}")
    if final.get("metrics"):
        m = final["metrics"]
        printer(f"  Metrics ({len(m)} fields):")
        for lbl, value in m.items():
            printer(f"    {lbl:<40} {value}")
    if final.get("error"):
        printer(f"  Error   : {final.get('error')}")
    printer(SEP)

    return final


def _dry_run_company(
    info: dict,
    source: str = "sec",
    ir_url_override: str = "",
    printer=print,
) -> dict:
    """Resolve URLs and check service connectivity without running the LLM or saving.

    Returns the initial state augmented with a ``_dry_run_verdict`` key:
    ``"ready"`` | ``"warning"`` | ``"blocked"``.
    """
    label = f"{info['company_name']} ({info.get('ticker') or info['cik']})"
    printer(f"\n{SEP}")
    printer(f"  DRY-RUN : {label}")
    printer(f"  CIK     : {info['cik']}")
    printer(f"  Source  : {source.upper()}")

    state = _build_initial_state(info, source=source, ir_url_override=ir_url_override, printer=printer)

    llm_ok, llm_detail = _check_llm()
    mongo_ok, mongo_detail = _check_mongodb()

    file_type: str | None = None
    url_blocked = state.get("status") == "failed"
    if not url_blocked and state.get("discovered_file_url"):
        dt = detect_document_type_node(state)
        file_type = dt.get("file_type")

    if url_blocked:
        verdict = "blocked"
    elif not llm_ok or not mongo_ok:
        verdict = "warning"
    else:
        verdict = "ready"

    url_display = state.get("discovered_file_url") or state.get("ir_url") or "(none)"
    printer(SEP)
    printer(f"  URL     : {url_display}")
    if file_type:
        printer(f"  DocType : {file_type}")
    printer(f"  LLM     : {'OK  ' if llm_ok else 'FAIL'} — {llm_detail}")
    printer(f"  MongoDB : {'OK  ' if mongo_ok else 'FAIL'} — {mongo_detail}")
    printer(f"  Verdict : {verdict.upper()}")
    if verdict == "blocked":
        printer(f"  Reason  : {state.get('error', 'URL resolution failed')}")
    elif verdict == "warning":
        if not llm_ok:
            if LLM_PROVIDER == "ollama":
                printer(
                    "  Action  : Start Ollama and pull the required model (see OLLAMA_MODEL in .env)"
                )
            elif LLM_PROVIDER == "openai":
                printer("  Action  : Set OPENAI_API_KEY (and optionally OPENAI_MODEL) in .env")
            else:
                printer("  Action  : Set GROQ_API_KEY (and optionally GROQ_MODEL) in .env")
        if not mongo_ok:
            printer("  Action  : Start MongoDB or check MONGODB_URI in .env")
    else:
        printer("  Action  : Run without --dry-run to execute the full pipeline")
    printer(SEP)

    return {**state, "_dry_run_verdict": verdict}


# ── Thread-pool workers ──────────────────────────────────────────────────────

def _is_already_saved(ticker: str) -> bool:
    """Return True if an earnings document for *ticker* already exists in MongoDB for the current year."""
    from datetime import datetime, timezone
    from earnings_agents.tools.mongodb_client import get_collection
    year = datetime.now(timezone.utc).year
    doc_id = f"{ticker}_{year}_latest"
    try:
        return get_collection().count_documents({"_id": doc_id}, limit=1) > 0
    except Exception:  # noqa: BLE001
        return False


def _run_company_parallel(args: tuple) -> dict:
    """Thread worker: run one company and update the shared rich Progress."""
    graph, info, source, ir_url_override, skip_existing, progress, overall_task = args
    ticker = info.get("ticker", "")
    name = ticker or info.get("company_name", "?")
    label = f"[cyan]{name}[/]"

    if skip_existing and ticker and _is_already_saved(ticker):
        progress.update(overall_task, advance=1)
        return {"status": "skipped", "ticker": ticker}

    company_task = progress.add_task(f"{label}  resolving\u2026", total=None)

    _completed: list[str] = []
    _current_stage: list[str] = ["resolving\u2026"]
    _current_detail: list[str] = [""]

    def _render() -> str:
        parts = [f"[green]{s} \u2713[/]" for s in _completed]
        active = f"[bold]{_current_stage[0]}[/]"
        if _current_detail[0]:
            active = f"{active} [dim]{_current_detail[0]}[/]"
        parts.append(active)
        return f"{label}  " + "  \u2192  ".join(parts)

    def _node_cb(node_name: str, event: str, _ticker: str) -> None:
        stage = _NODE_LABELS.get(node_name, node_name.replace("_node", ""))
        if event == "start":
            _current_stage[0] = stage
            _current_detail[0] = ""
            progress.update(company_task, description=_render())
        elif event == "end":
            if stage not in _completed:
                _completed.append(stage)
            progress.update(company_task, description=_render())

    def _detail_cb(detail: str) -> None:
        _current_detail[0] = detail
        progress.update(company_task, description=_render())

    set_node_callback(_node_cb)
    set_detail_callback(_detail_cb)
    result = _run_company(graph, info, source=source, ir_url_override=ir_url_override,
                          printer=lambda _: None)
    set_node_callback(None)
    set_detail_callback(None)

    status = result.get("status", "?")
    if status == "saved":
        desc = f"[green]{name}[/]  saved \u2713"
    elif status == "failed":
        desc = f"[red]{name}[/]  failed \u2717"
    else:
        desc = f"[yellow]{name}[/]  {status}"
    progress.update(company_task, total=1, completed=1, description=desc)
    progress.update(overall_task, advance=1)
    return result


def _dry_run_company_parallel(args: tuple) -> dict:
    """Thread worker: dry-run one company and update the shared rich Progress."""
    info, source, ir_url_override, skip_existing, progress, overall_task = args
    ticker = info.get("ticker", "")
    name = ticker or info.get("company_name", "?")
    label = f"[cyan]{name}[/]"

    if skip_existing and ticker and _is_already_saved(ticker):
        progress.update(overall_task, advance=1)
        return {"_dry_run_verdict": "ready", "status": "skipped", "ticker": ticker}

    company_task = progress.add_task(f"{label}  checking\u2026", total=None)
    result = _dry_run_company(info, source=source, ir_url_override=ir_url_override,
                              printer=lambda _: None)
    verdict = result.get("_dry_run_verdict", "?")
    color = {"ready": "green", "warning": "yellow", "blocked": "red"}.get(verdict, "white")
    progress.update(company_task, total=1, completed=1,
                    description=f"[{color}]{name}[/]  {verdict}")
    progress.update(overall_task, advance=1)
    return result


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Preview mode: resolve URLs and check service connectivity without running LLM "
            "extraction or saving to MongoDB. Prints a ready/warning/blocked verdict per company."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        metavar="N",
        help="Maximum parallel workers when processing multiple companies (default: 8).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip companies that already have a saved document in MongoDB for the current year.",
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

    _progress = Progress(
        SpinnerColumn(finished_text="[green]\u2713[/]"),
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=36),
        TimeElapsedColumn(),
    )

    # Route all logging through rich's console so warnings print above the
    # live progress display instead of writing raw bytes to stderr and
    # breaking the ANSI cursor control.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=_progress.console, show_path=False, rich_tracebacks=False)],
        force=True,
    )

    if LLM_PROVIDER == "openai":
        llm_label = f"openai:{OPENAI_MODEL}"
    elif LLM_PROVIDER == "groq":
        llm_label = f"groq:{GROQ_MODEL}"
    else:
        llm_label = f"ollama:{OLLAMA_MODEL}"
    _progress.console.print(f"[bold cyan]LLM[/]      : {llm_label}")

    if args.dry_run:
        total = len(companies)
        with _progress as progress:
            overall_task = progress.add_task("[bold]Dry-run[/]", total=total)
            worker_args = [
                (c, args.source, args.ir_url, args.skip_existing, progress, overall_task)
                for c in companies
            ]
            results = []
            if total == 1:
                results.append(_dry_run_company_parallel(worker_args[0]))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                    for future in concurrent.futures.as_completed(
                        pool.submit(_dry_run_company_parallel, wa) for wa in worker_args
                    ):
                        results.append(future.result())

        blocked = sum(1 for r in results if r.get("_dry_run_verdict") == "blocked")
        warnings = sum(1 for r in results if r.get("_dry_run_verdict") == "warning")
        ready = total - blocked - warnings
        print(f"Dry-run: {total} companies \u2014 {ready} ready, {warnings} warning, {blocked} blocked.")
        sys.exit(1 if blocked else 0)

    # Live run
    graph = build_graph()
    total = len(companies)
    with _progress as progress:
        overall_task = progress.add_task("[bold]Companies[/]", total=total)
        worker_args = [
            (graph, c, args.source, args.ir_url, args.skip_existing, progress, overall_task)
            for c in companies
        ]
        results = []
        if total == 1:
            results.append(_run_company_parallel(worker_args[0]))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                for future in concurrent.futures.as_completed(
                    pool.submit(_run_company_parallel, wa) for wa in worker_args
                ):
                    results.append(future.result())

    saved = [r for r in results if r.get("status") == "saved"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") not in ("saved", "skipped")]
    summary = f"Done: {len(saved)}/{total} saved"
    if skipped:
        summary += f", {len(skipped)} skipped"
    if failed:
        summary += f", {len(failed)} failed"
    print(summary + ".")
    if failed:
        print("Failed: " + ", ".join(r.get("ticker", "?") for r in failed))
        for r in failed:
            if r.get("error"):
                print(f"  {r.get('ticker', '?')}: {r['error']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

