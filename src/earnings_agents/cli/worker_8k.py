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
import time
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import MongoClient
from redis import Redis

from earnings_agents.cli.earnings import (
    _format_step_line,
    _has_existing_period_data,
    _is_period_already_stored,
)
from earnings_agents.config import REDIS_URL
from earnings_agents.hooks import set_call_callback, set_detail_callback, set_node_callback
from earnings_agents.tools.edgar_client import _find_exhibit_99_in_index, normalize_cik
from earnings_agents.tools.redis_queue import get_redis_client, serialize_message
from earnings_agents.workflow import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,  # override earnings.py which calls basicConfig(WARNING) on import
)
logger = logging.getLogger(__name__)

# Dedicated queue — admin_backend publishes 8-K messages here only.
_DEFAULT_QUEUE    = "sec:filings:8k"
_EVENTS_CHANNEL   = "sec:worker:events"


# ── MongoDB helper ─────────────────────────────────────────────────────────────

def _update_load_request_status(payload: dict[str, Any], status: str) -> None:
    """Update StockLoadRequest.status in MongoDB. Best-effort — never raises."""
    load_request_id = payload.get("load_request_id")
    if not load_request_id:
        return
    try:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DATABASE_NAME", "normalize_data")
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as mongo:
            mongo[db_name]["stock_load_requests"].update_one(
                {"_id": ObjectId(load_request_id)},
                {"$set": {"status": status}},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to update load_request status → %s: %s", status, exc)


# ── Real-time progress publishing ──────────────────────────────────────────────

class _ProgressPublisher:
    """Publishes pipeline node progress to the Redis ``sec:worker:events`` pub/sub
    channel so admin_backend can stream them to the frontend in real time.

    Uses a separate Redis connection from the queue consumer so publish calls
    never block the main processing thread.  Best-effort — any error is logged
    and silently swallowed.
    """

    def __init__(self, redis_url: str, ticker: str, load_request_id: str | None) -> None:
        self._ticker = ticker
        self._load_request_id = load_request_id
        self._client: Redis | None = None
        try:
            self._client = Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=5
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ProgressPublisher: could not connect to Redis: %s", exc)

    def publish(self, node_name: str, message: str, elapsed_ms: float | None = None) -> None:
        if not self._client:
            return
        event = {
            "event":           "worker_progress",
            "ticker":          self._ticker,
            "load_request_id": self._load_request_id,
            "node":            node_name,
            "message":         message,
            "elapsed_ms":      round(elapsed_ms, 1) if elapsed_ms is not None else None,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._client.publish(_EVENTS_CHANNEL, json.dumps(event))
        except Exception as exc:  # noqa: BLE001
            logger.debug("ProgressPublisher: publish failed: %s", exc)

    def close(self) -> None:
        try:
            if self._client:
                self._client.close()
        except Exception:  # noqa: BLE001
            pass


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

    # ── 1. Resolve Exhibit 99.1 from the specific accession ──────────────────
    # The RSS payload carries the filing INDEX url; we need the actual Exhibit
    # 99.1 document — the same URL that get_latest_earnings_url() returns in
    # the CLI.  Using the specific accession from the RSS feed is more precise
    # than re-scanning EDGAR for the "latest" 8-K.
    if not cik or not accession:
        logger.warning("8-K message for %s missing cik/accession — skipping", label)
        return True  # not a processing failure

    cik_int = str(int(normalize_cik(cik)))
    acc_nodash = accession.replace("-", "")
    filing_url = _find_exhibit_99_in_index(cik_int, accession, acc_nodash)

    if not filing_url:
        # No Exhibit 99.1 means this 8-K is not an earnings press release
        # (e.g. executive changes, amendments, material agreements).
        logger.info(
            "8-K for %s (accession %s) has no Exhibit 99.1 — not an earnings release, skipping",
            label, accession,
        )
        return True  # graceful skip — not a failure

    # ── 2. Guard: skip if no prior normalize_data period data for this ticker ─
    # The load_company_concepts node needs existing concept mappings to work.
    if ticker and not _has_existing_period_data(ticker):
        logger.info("8-K skipped for %s — no existing normalize_data period data", label)
        return True

    # ── 3. Guard: skip if this exact period is already stored ─────────────────
    if ticker and period and _is_period_already_stored(ticker, period):
        logger.info("8-K skipped for %s — period %s already stored in normalize_data", label, period)
        return True

    # ── 4. Run the pipeline ───────────────────────────────────────────────────
    # Initial state mirrors CLI's _build_initial_state SEC path exactly:
    #   status="discovered" → skips discover_earnings_release node
    #   company_cik pre-set → skips CIK resolution in load_company_concepts
    logger.info("Processing 8-K for %s  accession=%s  url=%s", label, accession, filing_url)

    # State mirrors CLI's _build_initial_state SEC path exactly — no extra keys.
    # company_cik is set so load_company_concepts skips its own CIK resolution.
    state = {
        "ticker": ticker or cik,
        "company_name": label,
        "company_cik": cik or None,
        "discovered_file_url": filing_url,
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

    # ── Set up progress callbacks (same format as the CLI output) ─────────────
    redis_url = os.getenv("REDIS_URL", REDIS_URL)
    pub = _ProgressPublisher(redis_url, ticker or cik, payload.get("load_request_id"))

    def _node_cb(node_name: str, event: str, _ticker: str,
                 node_state=None, elapsed_ms: float | None = None) -> None:
        if event != "end" or node_state is None:
            return
        msg = _format_step_line(node_name, node_state)
        if msg:
            for line in msg.splitlines():
                pub.publish(node_name, line.strip(), elapsed_ms)
                logger.info("[%s]  %s", ticker or cik, line.strip())

    set_node_callback(_node_cb)
    set_detail_callback(None)   # detail (chunk ticks) not needed in worker
    set_call_callback(None)

    try:
        final = graph.invoke(state)
    finally:
        set_node_callback(None)
        set_call_callback(None)
        pub.close()

    status = final.get("status", "")

    if status == "saved":
        n = len(final.get("concept_metrics") or {})
        logger.info(
            "8-K saved for %s — period=%s  concepts=%d",
            label, final.get("sec_report_date"), n,
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
        except Exception as exc:  # noqa: BLE001
            payload["last_error"] = str(exc)
            logger.exception("Unhandled error processing 8-K job for %s", payload.get("ticker"))

        if success:
            _update_load_request_status(payload, "completed")
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
