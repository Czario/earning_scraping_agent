"""Redis consumer: process 8-K filings published by admin_backend.

Listens to the **dedicated** ``sec:filings:8k`` queue (set via
``REDIS_QUEUE_NAME`` env var, default ``sec:filings:8k``).
admin_backend publishes 8-K messages here and 10-K/10-Q messages to a
separate queue consumed by the filings-extractor worker.  Each worker
only ever sees its own messages — no re-queue loops.

Pipeline mirrors ``uv run earnings --ticker X`` exactly:
  CLI    → ticker arg → EDGAR lookup → Exhibit 99.1 → pipeline
  Worker → Redis msg  → accession  → Exhibit 99.1 → pipeline

Progress events are published to the ``sec:worker:events`` Redis pub/sub
channel so admin_backend can stream them to the frontend in real time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from time import perf_counter
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from redis import Redis

from earnings_agents.cli.earnings import (
    _has_existing_period_data,
    _resolve_8k_skip_guard,
)
from earnings_agents.config import REDIS_URL
from earnings_agents.hooks import set_call_callback, set_detail_callback, set_node_callback
from earnings_agents.tools.edgar_client import get_latest_earnings_url
from earnings_agents.tools.redis_queue import get_redis_client, serialize_message
from earnings_agents.worker_progress import WorkerProgressPublisher, make_call_callback, make_node_callback, WorkerHeartbeat
from earnings_agents.workflow import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,  # override earnings.py which calls basicConfig(WARNING) on import
)
logger = logging.getLogger(__name__)

# Dedicated queue — admin_backend publishes 8-K messages here only.
_DEFAULT_QUEUE = "sec:filings:8k"


# ── MongoDB helper ─────────────────────────────────────────────────────────────

def _update_load_request_status(
    payload: dict[str, Any],
    status: str,
    period_of_report: str | None = None,
) -> None:
    """Update StockLoadRequest.status (and optionally sec_period_of_report) in MongoDB."""
    load_request_id = payload.get("load_request_id")
    if not load_request_id:
        return
    try:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DATABASE_NAME", "normalize_data")
        fields: dict[str, Any] = {"status": status}
        if period_of_report:
            fields["sec_period_of_report"] = period_of_report
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as mongo:
            mongo[db_name]["stock_load_requests"].update_one(
                {"_id": ObjectId(load_request_id)},
                {"$set": fields},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update load_request status → %s: %s", status, exc)


# ── Core processing — mirrors CLI's _build_initial_state + _run_company ───────

def _process_payload(graph, payload: dict[str, Any]) -> bool:
    """Process one 8-K filing message using the same pipeline as the CLI.

    Steps (identical to ``uv run earnings --ticker X``):
      1. Resolve the Exhibit 99.1 press-release URL from the accession number.
         If no Exhibit 99.1 exists this is a non-earnings 8-K — skip cleanly.
      2. Guard: skip if no prior normalize_data period data exists for the ticker.
      3. Guard: skip if this exact period is already stored in normalize_data.
      4. Run the LangGraph pipeline with status="discovered" — same starting
         state as the CLI's SEC EDGAR path.
    """
    ticker = (payload.get("ticker") or "").upper()
    cik = payload.get("cik") or ""
    accession = payload.get("accession_number") or ""
    label = payload.get("company_name") or ticker or cik or "?"
    period = payload.get("period_of_report") or ""

    # Create publisher immediately so every exit path can report status.
    redis_url = os.getenv("REDIS_URL", REDIS_URL)
    pub = WorkerProgressPublisher(redis_url, ticker or cik, payload.get("load_request_id"))

    # ── 1. Resolve Exhibit 99.1 + corrected period end date ──────────────────
    if not cik:
        pub.publish("skip", "message missing cik — cannot look up filing")
        pub.close()
        logger.warning("8-K message for %s missing cik — skipping", label)
        return True

    filing_url, supplemental_urls, period = get_latest_earnings_url(cik)

    if not filing_url:
        pub.publish("skip", "no Exhibit 99.1 on EDGAR — not an earnings release")
        pub.close()
        logger.info(
            "8-K for %s has no Exhibit 99.1 on EDGAR — not an earnings release, skipping",
            label,
        )
        return True  # graceful skip — not every 8-K is an earnings press release

    # ── 2. Guard: skip if no prior normalize_data period data for this ticker ─
    if ticker and not _has_existing_period_data(ticker):
        pub.publish("skip", f"skipped — no existing normalize_data period data for {ticker}")
        pub.close()
        logger.info("8-K skipped for %s — no existing normalize_data period data", label)
        return True

    # ── 3. Guard: skip if this fiscal period is already stored ────────────────
    # Uses fiscal_year_end_month + SEC submissions API to determine
    # (fiscal_year, quarter) and checks concept_values_quarterly / _annual.
    if ticker and cik and _resolve_8k_skip_guard(ticker, cik) is not None:
        pub.publish("skip", f"✓ already up to date — period already in normalize_data")
        pub.close()
        logger.info("8-K skipped for %s — period already stored in normalize_data", label)
        return True

    # ── 4. Run the pipeline ───────────────────────────────────────────────────
    # State mirrors CLI's _build_initial_state SEC path exactly.
    # period comes from get_latest_earnings_url (inferred, not raw RSS date).
    logger.info("Processing 8-K for %s  url=%s  period=%s", label, filing_url, period)

    # State mirrors CLI's _build_initial_state SEC path exactly — no extra keys.
    # company_cik is set so load_company_concepts skips its own CIK resolution.
    state = {
        "ticker": ticker or cik,
        "company_name": label,
        "company_cik": cik or None,
        "discovered_file_url": filing_url,
        "supplemental_file_urls": supplemental_urls,
        "sec_report_date": period or None,
        "file_type": None,
        "raw_text": None,
        "metrics": None,
        "error": None,
        "extraction_attempts": 0,
        "extraction_notes": None,
        "needs_reextract": False,
        "previous_high_finding_keys": None,
        "status": "discovered",
    }

    # ── Set up progress callbacks using the shared worker_progress module ────────
    # make_node_callback fires on both start (▶ stage) and end (step summary),
    # mirroring the CLI's rich progress output exactly.
    # make_call_callback forwards every LLM/DB/HTTP call event so the UI shows
    # the same intermediate lines the CLI prints.
    llm_call_count: list[int] = [0]
    set_node_callback(make_node_callback(pub, ticker or cik))
    set_call_callback(make_call_callback(pub, ticker or cik, llm_call_count))
    set_detail_callback(None)   # spinner text — not needed in worker

    t0 = perf_counter()
    try:
        with WorkerHeartbeat(pub, ticker or cik, interval_s=60):
            final = graph.invoke(state)
    except BaseException as exc:
        # Catches Exception (LLM errors etc.), KeyboardInterrupt, and SystemExit
        # (SIGTERM converted below) — publishes a visible failure line to the UI
        # so the user sees why the log stopped rather than an empty spinner.
        elapsed_s = perf_counter() - t0
        elapsed_str = (
            f"{elapsed_s:.1f}s"
            if elapsed_s < 60
            else f"{int(elapsed_s // 60)}m {elapsed_s % 60:.0f}s"
        )
        reason = (
            "worker stopped (SIGTERM)"
            if isinstance(exc, SystemExit)
            else "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else str(exc)[:120]
        )
        pub.publish(
            "summary",
            f"✗ {reason}  {elapsed_str}",
            kind="summary",
        )
        raise
    finally:
        set_node_callback(None)
        set_call_callback(None)
        pub.close()

    elapsed_s = perf_counter() - t0
    elapsed_str = (
        f"{elapsed_s:.1f}s"
        if elapsed_s < 60
        else f"{int(elapsed_s // 60)}m {elapsed_s % 60:.0f}s"
    )
    status = final.get("status", "")
    llm_tag = f"  ({llm_call_count[0]} LLM calls)" if llm_call_count[0] else ""

    if status == "saved":
        n = len(final.get("concept_metrics") or {})
        period_out = final.get("sec_report_date") or ""
        year = period_out[:4] if period_out else "?"
        summary = f"✓ {ticker or cik}_{year}_latest saved  ({n} concepts){llm_tag}  {elapsed_str}"
        pub.publish("summary", summary, kind="summary")
        # Build a human period label (FY2026 Q1 / FY2026 Annual) by reading
        # fiscal_year + quarter that normalize_data already stored — no
        # calculation here, just reading what was written by the pipeline.
        _period_label: str | None = None
        try:
            from earnings_agents.tools.normalize_data_client import (
                get_company_by_ticker,
                get_latest_period,
            )
            _company = get_company_by_ticker(ticker) if ticker else None
            if _company:
                _lp = get_latest_period(_company["cik"])
                if _lp:
                    _fy = _lp.get("fiscal_year")
                    _q  = _lp.get("quarter")
                    _pt = _lp.get("period_type", "annual")
                    if _pt == "quarterly" and _q and _fy:
                        _period_label = f"FY{_fy} Q{_q}"
                    elif _fy:
                        _period_label = f"FY{_fy} Annual"
        except Exception:  # noqa: BLE001
            pass
        # Store in payload so main() passes it to _update_load_request_status.
        if _period_label:
            payload["_sec_period_label"] = _period_label
        logger.info(
            "8-K saved for %s — period=%s  concepts=%d  llm_calls=%d  elapsed=%s",
            label, final.get("sec_report_date"), n, llm_call_count[0], elapsed_str,
        )
        return True

    if status in ("already_stored", "skipped"):
        logger.info("8-K skipped for %s — %s", label, status)
        return True

    logger.warning(
        "8-K pipeline ended with status=%s  error=%s  for %s",
        status, final.get("error"), label,
    )
    return False


# ── CLI argument parsing ───────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="earnings-8k-worker",
        description="Consume 8-K filing jobs from Redis and run the earnings extraction pipeline.",
    )
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", REDIS_URL))
    parser.add_argument("--queue-name", default=os.getenv("REDIS_QUEUE_NAME", _DEFAULT_QUEUE))
    parser.add_argument("--dead-letter-queue", default=os.getenv("REDIS_DEAD_LETTER_QUEUE", "sec:filings:dlq:8k"))
    parser.add_argument("--poll-timeout", type=int, default=5,
                        help="Seconds to block-wait for a Redis message.")
    parser.add_argument("--max-attempts", type=int,
                        default=int(os.getenv("REDIS_MAX_ATTEMPTS", "3")),
                        help="Retry limit before moving a job to the dead-letter queue.")
    parser.add_argument("--retry-delay", type=int,
                        default=int(os.getenv("REDIS_RETRY_DELAY_SECONDS", "5")),
                        help="Seconds to wait between retries.")
    parser.add_argument("--once", action="store_true",
                        help="Process one message then exit (useful for testing).")
    return parser.parse_args(argv)


# ── Main loop ──────────────────────────────────────────────────────────────────

def _make_client(redis_url: str, poll_timeout: int) -> Redis:
    """Create a Redis client suitable for blocking blpop.

    socket_timeout must be None (no socket-level deadline) so the blocking
    BLPOP command can wait the full poll_timeout seconds without the socket
    layer raising TimeoutError.  socket_connect_timeout is kept short so a
    bad URL fails fast at startup.
    """
    return Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=None,  # no socket-level timeout — blpop controls waiting
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    queue_name: str = args.queue_name
    dead_letter_queue: str = args.dead_letter_queue
    redis_url: str = args.redis_url

    # Convert SIGTERM (docker stop / docker-compose down) into SystemExit so it
    # propagates through finally blocks and the except BaseException handler in
    # _process_payload — this lets us publish a failure event and mark the DB
    # record as failed before the process exits.
    def _sigterm(_signum, _frame):  # noqa: ANN001
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    client = _make_client(redis_url, args.poll_timeout)
    graph = build_graph()
    logger.info("8-K worker listening on Redis queue '%s'", queue_name)

    while True:
        try:
            item = client.blpop(queue_name, timeout=args.poll_timeout)
        except Exception as exc:
            logger.warning("Redis blpop error (%s) — reconnecting in 5s", exc)
            time.sleep(5)
            try:
                client = _make_client(redis_url, args.poll_timeout)
            except Exception:
                pass
            continue

        if not item:
            if args.once:
                break
            continue

        _, raw_message = item
        try:
            payload: dict[str, Any] = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.error("Could not decode queue message: %r", raw_message)
            if args.once:
                break
            continue

        form_type = (payload.get("filing_type") or payload.get("form_type") or "").upper()

        if form_type != "8-K":
            # Dedicated queue: only 8-K messages should arrive here.
            # Log and skip anything unexpected without re-queuing.
            logger.warning("Unexpected form_type %r on 8-K queue — skipping", form_type)
            if args.once:
                break
            continue

        # ── Process the 8-K ───────────────────────────────────────────────────
        attempts = int(payload.get("attempts") or 0)
        success = False
        _update_load_request_status(payload, "processing")

        try:
            success = _process_payload(graph, payload)
        except (KeyboardInterrupt, SystemExit):
            # Worker is shutting down (Ctrl-C or docker stop) mid-job.
            # _process_payload already published the ✗ summary event; we just
            # need to persist the failed status before the process exits.
            _update_load_request_status(payload, "failed")
            logger.info(
                "Worker shutdown mid-job — marked %s as failed",
                payload.get("ticker"),
            )
            raise  # let the process exit normally
        except Exception as exc:  # noqa: BLE001
            payload["last_error"] = str(exc)
            logger.exception("Unhandled error processing 8-K job for %s", payload.get("ticker"))

        if success:
            # Write sec_period_of_report so the pipeline table shows the period.
            # _sec_period_label (e.g. "FY2026 Q1" / "FY2026 Annual") is set by
            # _process_payload after reading normalize_data; fall back to the
            # raw EDGAR date if it wasn't computable.
            _period = (
                payload.pop("_sec_period_label", None)
                or payload.get("sec_report_date")
                or payload.get("period_of_report")
            )
            _update_load_request_status(payload, "completed", period_of_report=_period)
        else:
            payload["attempts"] = attempts + 1
            payload["failed_at"] = time.time()
            if payload["attempts"] < args.max_attempts:
                logger.warning(
                    "8-K job failed; retrying attempt %d/%d for %s",
                    payload["attempts"], args.max_attempts, payload.get("ticker"),
                )
                time.sleep(max(0, args.retry_delay))
                client.rpush(queue_name, serialize_message(payload))
            else:
                logger.error(
                    "8-K job exhausted %d retries; moving to dead-letter queue '%s' for %s",
                    args.max_attempts, dead_letter_queue, payload.get("ticker"),
                )
                _update_load_request_status(payload, "failed")
                client.rpush(dead_letter_queue, serialize_message(payload))

        if args.once:
            break


if __name__ == "__main__":
    main()
