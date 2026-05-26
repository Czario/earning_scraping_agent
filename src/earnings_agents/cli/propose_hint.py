"""``uv run earnings-propose-hint`` — draft a company hint file from failures.

Queries MongoDB for recent degraded/failed runs for a given ticker, assembles
the findings into a prompt, and asks the configured LLM to draft extraction
hints.  The draft is written to ``data/company_hints/_proposed/{TICKER}.md``
for a human to review and promote (move to ``data/company_hints/{TICKER}.md``).

Usage::

    uv run earnings-propose-hint --ticker AAPL
    uv run earnings-propose-hint --ticker MSFT --docs 10
    uv run earnings-propose-hint --ticker GOOGL --force   # overwrite existing draft
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from earnings_agents.llm_factory import build_llm
from earnings_agents.tools.mongodb_client import get_collection

console = Console()

_HINTS_DIR = Path(__file__).parents[3] / "data" / "company_hints"
_PROPOSED_DIR = _HINTS_DIR / "_proposed"

_DRAFT_PROMPT = """\
You are an expert financial data engineer helping to fix a recurring earnings-extraction problem.

The pipeline uses an LLM to extract financial metrics from SEC filings and IR press releases.
It failed (status="degraded" or "failed") for {ticker} ({company}) on {run_count} recent run(s).

== FINDINGS FROM FAILED RUNS ==
{findings_block}

== IDENTITY WARNINGS (balance-sheet checks that failed) ==
{identity_block}

== EXISTING HINT FILE (currently in use, may be empty) ==
{existing_hints}

== YOUR TASK ==
Write a concise set of bullet-point hints that the extraction LLM should follow
to avoid the above failures in future runs.  Each hint must be:
  • Actionable: tells the extractor what to look for, use, or avoid.
  • Specific to {ticker}: reference their actual label names when known.
  • Minimal: do not repeat what the extractor already knows (standard GAAP).

Output ONLY the bullet points in Markdown format — no preamble, no section headers,
no explanation outside of the bullets.  Begin each bullet with "- ".
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="earnings-propose-hint",
        description="Draft a company hint file from recent degraded/failed runs.",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        metavar="TICKER",
        help="Ticker symbol to generate hints for (e.g. AAPL).",
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=5,
        metavar="N",
        help="Number of recent degraded/failed docs to analyse (default: 5).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing proposed draft file.",
    )
    return parser.parse_args(argv)


def _fetch_docs(ticker: str, limit: int) -> list[dict[str, Any]]:
    col = get_collection()
    return list(
        col.find(
            {
                "ticker": ticker.upper(),
                "status": {"$in": ["degraded", "failed"]},
            },
            sort=[("scraped_at", -1)],
            limit=limit,
        )
    )


def _build_findings_block(docs: list[dict[str, Any]]) -> str:
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for doc in docs:
        scraped = doc.get("scraped_at")
        date_str = scraped.strftime("%Y-%m-%d") if scraped else "unknown date"
        for f in doc.get("findings") or []:
            if not isinstance(f, dict):
                continue
            ftype = f.get("type", "unknown")
            msg = f.get("message", "")
            severity = f.get("severity", "")
            key = (ftype, msg)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  [{severity.upper()}] {ftype}: {msg}  (run: {date_str})")
    return "\n".join(lines) if lines else "  (no structured findings recorded)"


def _build_identity_block(docs: list[dict[str, Any]]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for doc in docs:
        for w in doc.get("identity_warnings") or []:
            if isinstance(w, str) and w not in seen:
                seen.add(w)
                lines.append(f"  • {w}")
    return "\n".join(lines) if lines else "  (none)"


def _load_existing_hints(ticker: str) -> str:
    hint_file = _HINTS_DIR / f"{ticker.upper()}.md"
    if hint_file.is_file():
        content = hint_file.read_text(encoding="utf-8").strip()
        return content if content else "(empty)"
    return "(no existing hint file)"


def _write_proposed(ticker: str, content: str, force: bool) -> Path:
    _PROPOSED_DIR.mkdir(parents=True, exist_ok=True)
    out = _PROPOSED_DIR / f"{ticker.upper()}.md"
    if out.exists() and not force:
        console.print(
            f"[yellow]Draft already exists:[/yellow] {out}\n"
            "Use [bold]--force[/bold] to overwrite."
        )
        sys.exit(1)
    header = (
        f"# Proposed extraction hints for {ticker.upper()}\n"
        f"# Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"# Review and move to data/company_hints/{ticker.upper()}.md when satisfied.\n\n"
    )
    out.write_text(header + content.strip() + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    ticker = args.ticker.upper()

    console.print(f"Querying MongoDB for degraded/failed runs: [cyan]{ticker}[/cyan] …")
    try:
        docs = _fetch_docs(ticker, args.docs)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]MongoDB error:[/red] {exc}")
        sys.exit(1)

    if not docs:
        console.print(
            f"[green]No degraded or failed runs found for {ticker}.[/green]\n"
            "Nothing to propose — run the pipeline first."
        )
        return

    company = docs[0].get("company_name") or ticker
    console.print(f"Found [bold]{len(docs)}[/bold] doc(s) for [cyan]{company}[/cyan].")

    findings_block = _build_findings_block(docs)
    identity_block = _build_identity_block(docs)
    existing_hints = _load_existing_hints(ticker)

    prompt = _DRAFT_PROMPT.format(
        ticker=ticker,
        company=company,
        run_count=len(docs),
        findings_block=findings_block,
        identity_block=identity_block,
        existing_hints=existing_hints,
    )

    console.print("Calling LLM to draft hints …")
    try:
        llm = build_llm(format_json=False)
        draft: str = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]LLM error:[/red] {exc}")
        sys.exit(1)

    out_path = _write_proposed(ticker, draft, args.force)

    console.print(f"\n[green]Draft written to:[/green] {out_path}")
    console.print(
        "Review the draft, then move it to "
        f"[cyan]data/company_hints/{ticker}.md[/cyan] to activate."
    )


if __name__ == "__main__":
    main()
