"""Pre/post node lifecycle hooks for the LangGraph earnings pipeline.

``with_hooks`` wraps any node function and fires structured log events at entry
and exit, measures wall-clock duration, and converts unhandled exceptions into a
``status: "failed"`` state so the graph can short-circuit cleanly.

Usage in ``workflow.py``::

    from earnings_agents.hooks import with_hooks

    graph.add_node("discover_earnings_release",
                   with_hooks(discover_earnings_release_node))
"""
from __future__ import annotations

import logging
import threading as _threading
import time
from functools import wraps
from typing import Callable

from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

_thread_local = _threading.local()


def set_node_callback(callback) -> None:
    """Register a per-thread callback invoked on node events.

    The callback receives ``(node_name: str, event: str, ticker: str)`` where
    *event* is ``"start"``, ``"end"``, or ``"error"``.
    Pass ``None`` to clear.
    """
    _thread_local.node_callback = callback


def set_detail_callback(callback) -> None:
    """Register a per-thread callback for sub-node progress detail.

    The callback receives a single ``detail`` string (e.g. ``"chunk 3/8"``).
    Pass ``None`` to clear.
    """
    _thread_local.detail_callback = callback


def report_detail(detail: str) -> None:
    """Fire the thread-local detail callback if one is registered."""
    cb = getattr(_thread_local, "detail_callback", None)
    if cb:
        cb(detail)


def with_hooks(
    node_fn: Callable[[EarningsAgentState], EarningsAgentState],
) -> Callable[[EarningsAgentState], EarningsAgentState]:
    """Return *node_fn* wrapped with pre/post structured log events and timing.

    On entry:  logs ``node_start`` with ticker and incoming status.
    On exit:   logs ``node_end`` with outgoing status and duration in ms.
    On error:  logs ``node_error``, sets ``status="failed"`` and ``error=<msg>``,
               and returns without re-raising so the graph's routing helpers can
               short-circuit to ``END`` as normal.
    """
    node_name = node_fn.__name__

    @wraps(node_fn)
    def _wrapper(state: EarningsAgentState) -> EarningsAgentState:
        ticker = state.get("ticker", "?")
        t0 = time.perf_counter()

        cb = getattr(_thread_local, "node_callback", None)
        if cb:
            cb(node_name, "start", ticker)

        logger.debug(
            '{"event":"node_start","node":"%s","ticker":"%s","status_in":"%s"}',
            node_name,
            ticker,
            state.get("status", "?"),
        )

        try:
            new_state = node_fn(state)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                '{"event":"node_error","node":"%s","ticker":"%s","error":"%s","duration_ms":%.1f}',
                node_name,
                ticker,
                str(exc).replace('"', "'"),
                elapsed_ms,
            )
            cb = getattr(_thread_local, "node_callback", None)
            if cb:
                cb(node_name, "error", ticker)
            return {
                **state,
                "status": "failed",
                "error": f"{node_name} raised: {exc}",
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            '{"event":"node_end","node":"%s","ticker":"%s","status_out":"%s","duration_ms":%.1f}',
            node_name,
            ticker,
            new_state.get("status", "?"),
            elapsed_ms,
        )
        cb = getattr(_thread_local, "node_callback", None)
        if cb:
            cb(node_name, "end", ticker)
        return new_state

    return _wrapper
