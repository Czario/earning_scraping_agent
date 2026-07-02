"""LLM-based financial metric extraction tool.

Provides :func:`invoke_chunk_with_retry`, the core LLM call harness used by
``extract_financial_metrics_node``.  Extracted here so the retry logic,
semaphore management, and timeout handling are independently testable.

The caller is responsible for building the prompt and providing the response
parser callback (``parse_fn``).
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import nullcontext as _nullcontext
from typing import Any, Callable

from earnings_agents.config import (
    GROQ_REQUEST_TIMEOUT,
    OLLAMA_CONCURRENCY,
    LLM_PROVIDER as _LLM_PROVIDER,
)
from earnings_agents.hooks import report_detail, report_call, set_detail_callback
from earnings_agents.llm_factory import build_llm

logger = logging.getLogger(__name__)

_OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "75"))
_CHUNK_MAX_RETRIES = int(os.getenv("CHUNK_MAX_RETRIES", "1"))

# Shared semaphore that limits concurrent Ollama calls across all threads.
# Prevents timeout storms when running many tickers against a single-threaded
# local Ollama instance.
_OLLAMA_SEMAPHORE = threading.Semaphore(OLLAMA_CONCURRENCY)


def invoke_chunk_with_retry(
    prompt: str,
    chunk_num: int,
    total_chunks: int,
    ticker: str,
    parse_fn: Callable[[str], dict | None],
    *,
    llm: Any = None,
    max_retries: int = _CHUNK_MAX_RETRIES,
    detail_callback: Any = None,
    report_chunk: Callable[[int, str, int], None] | None = None,
    provider: str | None = None,
) -> "dict[str, Any] | None":
    """Invoke the LLM for one text chunk, retrying with a stricter prefix on parse failure.

    Each call creates its own LLM client instance; sharing one instance across
    threads can cause intermittent blocking under parallel load.

    Parameters
    ----------
    prompt:
        The full prompt string to send to the LLM.
    chunk_num:
        1-based index of this chunk (used for logging and progress reporting).
    total_chunks:
        Total number of chunks in this extraction pass.
    ticker:
        Ticker symbol, used in log messages.
    parse_fn:
        Callable that accepts the raw LLM response string and returns a parsed
        ``dict`` or ``None`` on failure.  Decouples the response-parsing logic
        from the retry harness.
    llm:
        Optional pre-built LLM client.  When ``None``, a new client is built
        from ``provider`` and timeout settings.  Callers can inject a mock here
        to keep unit tests simple.
    max_retries:
        Number of additional attempts after the first (default 1 → max 2 total).
    detail_callback:
        Optional progress callback registered via :func:`set_detail_callback`.
    report_chunk:
        Optional callable ``(chunk_index, status, attempt)`` for per-chunk
        progress bar updates.  ``chunk_index`` is 0-based.
    provider:
        Optional LLM provider override (e.g. ``"groq"``).  When ``None``, uses
        the configured ``LLM_PROVIDER``.  Ignored when *llm* is provided.

    Returns
    -------
    dict or None
        Parsed result dict on success, ``None`` after all attempts fail.
    """
    if detail_callback is not None:
        set_detail_callback(detail_callback)

    if llm is None:
        timeout = GROQ_REQUEST_TIMEOUT if provider == "groq" else _OLLAMA_REQUEST_TIMEOUT
        llm = build_llm(format_json=True, request_timeout=timeout, provider=provider)

    for attempt in range(max_retries + 1):
        if report_chunk is not None:
            report_chunk(chunk_num - 1, "running", attempt + 1)
        else:
            report_detail(f"chunk {chunk_num}/{total_chunks} attempt {attempt + 1}")

        prefix = (
            ""
            if attempt == 0
            else (
                "CRITICAL: Your previous response could not be parsed as JSON. "
                "Respond with ONLY a raw JSON object starting with '{'. "
                "Absolutely no markdown fences, no preamble, no explanation.\n\n"
            )
        )
        logger.info(
            "Chunk %d/%d attempt %d started for %s",
            chunk_num, total_chunks, attempt + 1, ticker,
        )
        try:
            # Throttle concurrent local Ollama calls via semaphore.
            # Skip for Groq (cloud API with its own rate limiting).
            _ctx = _OLLAMA_SEMAPHORE if provider != "groq" else _nullcontext()
            report_call(f"  [llm]  chunk {chunk_num}/{total_chunks}  attempt {attempt + 1}  → calling llm  ({provider or _LLM_PROVIDER or 'llm'})")
            with _ctx:
                response: str = llm.invoke(prefix + prompt)
            logger.debug(
                "Chunk %d/%d attempt %d raw response for %s: %r",
                chunk_num, total_chunks, attempt + 1, ticker, response[:300],
            )
            parsed = parse_fn(response)
            if parsed is not None:
                n_keys = len([k for k in parsed if not k.startswith("__")])
                report_call(f"  chunk {chunk_num}/{total_chunks}  ✓  {n_keys} keys parsed")
                if report_chunk is not None:
                    report_chunk(chunk_num - 1, "done", attempt + 1)
                return parsed
            report_call(f"  chunk {chunk_num}/{total_chunks}  ✗  unparseable response")
            logger.warning(
                "Chunk %d/%d attempt %d returned unparseable response for %s",
                chunk_num, total_chunks, attempt + 1, ticker,
            )
        except Exception as exc:  # noqa: BLE001
            report_call(f"  chunk {chunk_num}/{total_chunks}  ✗  {exc}")
            logger.warning(
                "Chunk %d/%d attempt %d failed for %s: %s",
                chunk_num, total_chunks, attempt + 1, ticker, exc,
            )

    if report_chunk is not None:
        report_chunk(chunk_num - 1, "failed", max_retries + 1)
    return None
