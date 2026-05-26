"""``uv run earnings-failures`` — query MongoDB for degraded/failed runs.

Aggregates findings by ticker and finding type so you can spot recurring
extraction problems and decide which tickers need a company hint file.

Usage examples::

    uv run earnings-failures                    # all degraded/failed docs
    uv run earnings-failures --ticker MSFT AAPL # filter to specific tickers
    uv run earnings-failures --days 30          # only last 30 days
    uv run earnings-failures --status degraded  # degraded only (excludes failed)
    uv run earnings-failures --status failed    # pipeline errors only
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from rich.console import Console
from rich.table import Table

from earnings_agents.tools.mongodb_client import get_collection

console = Console()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="earnings-failures",
        description="Report degraded/failed earnings runs from MongoDB.",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="TICKER",
        help="Filter to one or more tickers (e.g. MSFT AAPL).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="Only include docs scraped in the last N days.",
    )
    parser.add_argument(
        "--status",
        choices=["degraded", "failed"],
        default=None,
        help="Filter by document status. Omit to show both.",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show individual finding messages, not just counts.",
    )
    return parser.parse_args(argv)


def _build_query(args: argparse.Namespace) -> dict[str, Any]:
    query: dict[str, Any] = {}

    # Status filter
    if args.status:
        query["status"] = args.status
    else:
        query["status"] = {"$in": ["degraded", "failed"]}

    # Ticker filter
    if args.ticker:
        tickers = [t.upper() for t in args.ticker]
        query["ticker"] = {"$in": tickers} if len(tickers) > 1 else tickers[0]

    # Date filter
    if args.days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        query["scraped_at"] = {"$gte": cutoff}

    return query


def _finding_type_summary(findings: list[dict]) -> dict[str, int]:
    """Count findings by type, considering only high/medium severity."""
    counts: dict[str, int] = defaultdict(int)
    for f in findings:
        if isinstance(f, dict) and f.get("severity") in ("high", "medium"):
            counts[f.get("type", "unknown")] += 1
    return dict(counts)


def _format_finding_types(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    return "  ".join(f"{t}×{n}" for t, n in sorted(counts.items()))


def _format_messages(findings: list[dict]) -> str:
    msgs = [
        f.get("message", "")
        for f in findings
        if isinstance(f, dict) and f.get("severity") in ("high", "medium")
    ]
    return "\n".join(f"  • {m}" for m in msgs) if msgs else "—"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    query = _build_query(args)

    try:
        col = get_collection()
        docs = list(col.find(query, sort=[("scraped_at", -1)]))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]MongoDB error:[/red] {exc}")
        sys.exit(1)

    if not docs:
        console.print("[green]No degraded or failed runs found.[/green]")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    table = Table(
        title=f"Degraded/Failed Runs ({len(docs)} docs)",
        show_lines=args.detail,
    )
    table.add_column("Ticker", style="bold cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Scraped At (UTC)", no_wrap=True)
    table.add_column("Findings (high/medium)", overflow="fold")
    table.add_column("Identity Warnings", overflow="fold")

    for doc in docs:
        ticker = doc.get("ticker", "?")
        status = doc.get("status", "?")
        scraped_at = doc.get("scraped_at")
        scraped_str = scraped_at.strftime("%Y-%m-%d %H:%M") if scraped_at else "?"

        findings: list[dict] = doc.get("findings") or []
        iw: list[str] = doc.get("identity_warnings") or []

        status_style = "red" if status == "failed" else "yellow"
        status_cell = f"[{status_style}]{status}[/{status_style}]"

        if args.detail:
            finding_cell = _format_messages(findings)
        else:
            finding_cell = _format_finding_types(_finding_type_summary(findings))

        iw_cell = ("  ".join(iw[:3]) + ("…" if len(iw) > 3 else "")) if iw else "—"

        table.add_row(ticker, status_cell, scraped_str, finding_cell, iw_cell)

    console.print(table)

    # ── Aggregate finding-type counts across all docs ────────────────────────
    aggregate: dict[str, int] = defaultdict(int)
    ticker_finding_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for doc in docs:
        ticker = doc.get("ticker", "?")
        for f in doc.get("findings") or []:
            if isinstance(f, dict) and f.get("severity") in ("high", "medium"):
                ftype = f.get("type", "unknown")
                aggregate[ftype] += 1
                ticker_finding_map[ticker][ftype] += 1

    if aggregate:
        console.print()
        agg_table = Table(title="Finding Type Totals (all docs above)")
        agg_table.add_column("Finding Type", style="bold")
        agg_table.add_column("Occurrences", justify="right")
        agg_table.add_column("Tickers Affected")
        for ftype, count in sorted(aggregate.items(), key=lambda x: -x[1]):
            affected = ", ".join(
                t for t, m in ticker_finding_map.items() if ftype in m
            )
            agg_table.add_row(ftype, str(count), affected)
        console.print(agg_table)

    # ── Hint file suggestions ─────────────────────────────────────────────────
    # Tickers with 2+ degraded docs get flagged as candidates for a hint file.
    ticker_counts: dict[str, int] = defaultdict(int)
    for doc in docs:
        ticker_counts[doc.get("ticker", "?")] += 1

    repeat_offenders = [t for t, n in ticker_counts.items() if n >= 2]
    if repeat_offenders:
        console.print()
        console.print(
            "[bold yellow]Hint file candidates[/bold yellow] "
            "(≥2 degraded/failed runs — consider creating "
            "[cyan]data/company_hints/{TICKER}.md[/cyan]):"
        )
        for t in sorted(repeat_offenders):
            console.print(f"  • [cyan]{t}[/cyan] ({ticker_counts[t]} runs)")


if __name__ == "__main__":
    main()
