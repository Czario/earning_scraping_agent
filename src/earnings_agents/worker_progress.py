"""Shared real-time progress publishing for worker processes.

Any worker that runs the LangGraph earnings pipeline can use this module
to stream per-node progress events to the Redis ``sec:worker:events``
pub/sub channel, which admin_backend forwards to the frontend via SSE.

Each published event carries a ``kind`` field so the frontend can render
different message types with distinct styling:

  ``step_start``  — node begins:  "► load concepts"
  ``step_end``    — node summary: "[load concepts]  71 concepts  (annual)"
  ``call_llm``    — LLM API call: "[llm]  chunk 1/1  → calling llm  (deepseek)"
  ``call``        — DB / HTTP:    "[db]  query normalized_concepts_annual …"
  ``summary``     — final line:   "✓ saved  (5 LLM calls)"
  ``skip``        — non-fatal skip notice

Usage::

    from earnings_agents.worker_progress import (
        WorkerProgressPublisher,
        make_node_callback,
        make_call_callback,
    )
    from earnings_agents.hooks import (
        set_node_callback,
        set_call_callback,
        set_detail_callback,
    )

    pub = WorkerProgressPublisher(redis_url, ticker, load_request_id)
    llm_call_count: list[int] = [0]

    set_node_callback(make_node_callback(pub, ticker))
    set_call_callback(make_call_callback(pub, ticker, llm_call_count))
    set_detail_callback(None)
    try:
        final = graph.invoke(state)
    finally:
        set_node_callback(None)
        set_call_callback(None)
        pub.close()
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from redis import Redis

from earnings_agents.cli.earnings import _NODE_LABELS, _format_step_line

logger = logging.getLogger(__name__)

EVENTS_CHANNEL = "sec:worker:events"


class WorkerProgressPublisher:
    """Publishes pipeline node progress to Redis pub/sub so the frontend
    can display a live log identical to the CLI output.

    Uses a dedicated Redis connection separate from the queue consumer
    so publish calls never block processing.  All errors are swallowed
    and logged at WARNING level so a Redis hiccup never kills the pipeline.
    """

    def __init__(
        self,
        redis_url: str,
        ticker: str,
        load_request_id: str | None,
    ) -> None:
        self._ticker = ticker
        self._load_request_id = load_request_id
        self._client: Redis | None = None
        try:
            self._client = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("WorkerProgressPublisher: Redis connect failed: %s", exc)

    def publish(
        self,
        node_name: str,
        message: str,
        elapsed_ms: float | None = None,
        kind: str = "step_end",
    ) -> None:
        """Publish one log line to the ``sec:worker:events`` channel.

        ``kind`` controls frontend styling:
          ``step_start`` | ``step_end`` | ``call_llm`` | ``call`` | ``summary`` | ``skip``
        """
        if not self._client:
            return
        event: dict[str, Any] = {
            "event":           "worker_progress",
            "ticker":          self._ticker,
            "load_request_id": self._load_request_id,
            "node":            node_name,
            "message":         message,
            "kind":            kind,
            "elapsed_ms":      round(elapsed_ms, 1) if elapsed_ms is not None else None,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._client.publish(EVENTS_CHANNEL, json.dumps(event))
        except Exception as exc:  # noqa: BLE001
            logger.warning("WorkerProgressPublisher: publish failed: %s", exc)

    def close(self) -> None:
        try:
            if self._client:
                self._client.close()
        except Exception:  # noqa: BLE001
            pass


def make_node_callback(
    pub: WorkerProgressPublisher,
    ticker: str,
) -> Callable:
    """Return a node lifecycle callback compatible with ``set_node_callback``.

    Fires ``step_start`` on node entry (mirroring the CLI's "▶ stage" line)
    and ``step_end`` on node exit using the same ``_format_step_line`` output
    as the CLI.  Any future CLI format changes apply here automatically.

    Signature mirrors what ``with_hooks`` passes::

        callback(node_name, event, ticker, node_state=None, elapsed_ms=None)
    """

    def _callback(
        node_name: str,
        event: str,
        _ticker: str,
        node_state: dict | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        if event == "start":
            stage = _NODE_LABELS.get(node_name, node_name.replace("_node", "").replace("_", " "))
            pub.publish(node_name, f"► {stage}", kind="step_start")
            logger.info("[%s]  ► %s", ticker, stage)
            return

        if event != "end" or node_state is None:
            return

        msg = _format_step_line(node_name, node_state)
        if not msg:
            return
        lines = msg.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                # Only attach elapsed_ms to the last line (same as CLI duration display)
                ms = elapsed_ms if i == len(lines) - 1 else None
                pub.publish(node_name, stripped, ms, kind="step_end")
                logger.info("[%s]  %s", ticker, stripped)

    return _callback


def make_call_callback(
    pub: WorkerProgressPublisher,
    ticker: str,
    llm_call_count: list[int],
) -> Callable:
    """Return a callback for ``set_call_callback`` that publishes every
    LLM/DB/HTTP call event as it fires.

    ``llm_call_count`` is a single-element list used as a mutable counter so
    the caller can read the total after the pipeline finishes.

    LLM call messages (containing ``[llm]`` or ``→ calling llm``) are
    published with ``kind="call_llm"``; everything else with ``kind="call"``.
    """

    def _callback(msg: str) -> None:
        stripped = msg.strip()
        if not stripped:
            return
        is_llm = " [llm]" in stripped or (
            "llm" in stripped.lower() and "\u2192" in stripped
        )
        kind = "call_llm" if is_llm else "call"
        if is_llm and "\u2192 calling llm" in stripped:
            llm_call_count[0] += 1
        pub.publish("call", stripped, kind=kind)
        logger.info("[%s]  %s", ticker, stripped)

    return _callback


class WorkerHeartbeat:
    """Context manager that publishes a ``kind='heartbeat'`` event every
    *interval_s* seconds while the pipeline is running.

    The frontend uses these to distinguish a slow-but-alive LLM call from a
    crashed worker: as long as heartbeats arrive, ``lastEvt.timestamp`` stays
    fresh and the "Stalled" badge never fires.  Heartbeat events are filtered
    from the visible log in the UI.

    Usage::

        with WorkerHeartbeat(pub, ticker, interval_s=60):
            final = graph.invoke(state)
    """

    def __init__(
        self,
        pub: WorkerProgressPublisher,
        ticker: str,
        interval_s: int = 60,
    ) -> None:
        self._pub = pub
        self._ticker = ticker
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        elapsed = 0
        while not self._stop.wait(timeout=self._interval_s):
            elapsed += self._interval_s
            self._pub.publish(
                "heartbeat",
                f"\u2665 alive  ({elapsed}s)",
                kind="heartbeat",
            )

    def __enter__(self) -> "WorkerHeartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
