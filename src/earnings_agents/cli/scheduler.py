from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from earnings_agents.config import COMPANIES
from earnings_agents.workflow import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _run_ticker(ticker: str, cfg: dict, graph) -> None:
    """Run the earnings scraping graph for a single ticker."""
    logger.info("Starting earnings scrape for %s (%s)", cfg["name"], ticker)
    initial_state = {
        "ticker": ticker,
        "company_name": cfg["name"],
        "ir_url": cfg["ir_url"],
        "discovered_file_url": None,
        "file_type": None,
        "raw_text": None,
        "metrics": None,
        "error": None,
        "status": "pending",
        "extraction_attempts": 0,
        "extraction_notes": None,
    }
    final_state = graph.invoke(initial_state)
    logger.info(
        "Completed %s — status=%s  metrics=%d",
        ticker,
        final_state.get("status"),
        len(final_state.get("metrics") or {}),
    )


def run_all_companies() -> None:
    """Run the earnings scraping graph for every configured company.

    Set SCRAPE_CONCURRENCY env var to process multiple tickers simultaneously.
    For concurrency > 1, start the Ollama server with OLLAMA_NUM_PARALLEL set
    to the same value so requests don't queue behind each other.
    """
    graph = build_graph()
    concurrency = int(os.getenv("SCRAPE_CONCURRENCY", "1"))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_ticker, ticker, cfg, graph): ticker
            for ticker, cfg in COMPANIES.items()
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Unexpected error for %s: %s", ticker, exc, exc_info=True)


def main() -> None:
    scheduler = BlockingScheduler()

    # Fire on days 15-31 of earnings months (Jan, Apr, Jul, Oct) at 09:00 UTC.
    # This window reliably covers all major earnings release dates.
    scheduler.add_job(
        run_all_companies,
        CronTrigger(month="1,4,7,10", day="15-31", hour=9, minute=0),
        id="quarterly_earnings_scrape",
        name="Quarterly Earnings Scrape",
        replace_existing=True,
    )

    logger.info(
        "Scheduler running — next execution during earnings months (Jan/Apr/Jul/Oct)"
    )
    scheduler.start()


if __name__ == "__main__":
    main()
