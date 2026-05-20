"""Reflect & Decide node — Stage 3 (Observe) + Stage 4 (Reflect & Decide).

Checks extracted metrics against deterministic Python patterns for critical
financial metrics. No LLM call — key-matching is exact, instant, and has
zero false positives from alias confusion.

If critical metrics are missing and ``extraction_attempts < MAX_EXTRACTION_ATTEMPTS``,
injects ``extraction_notes`` and resets ``status`` to ``"text_extracted"`` so
the graph loops back to ``extract_financial_metrics``.
"""
from __future__ import annotations

import logging
import re

from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

# Maximum number of extraction passes (initial + retries) before giving up.
MAX_EXTRACTION_ATTEMPTS = 3

# Each entry: (human-readable name, regex that must match ≥ 1 extracted key).
# If no key in the extracted metrics matches the pattern, that metric is flagged missing.
_CRITICAL_CHECKS: list[tuple[str, re.Pattern]] = [
    ("Total Revenue",    re.compile(r"revenue|net sales|net revenue", re.I)),
    ("Net Income",       re.compile(r"net income|net earnings|net loss", re.I)),
    ("EPS",              re.compile(r"per share|diluted|basic.*income.*per|\beps\b", re.I)),
    ("Operating Income", re.compile(
        r"operating income|operating profit|operating loss|income from operations", re.I
    )),
]


def _check_critical_metrics(metrics: dict) -> list[str]:
    """Return names of critical metrics absent from the extracted dict."""
    keys = list(metrics.keys())
    return [
        name
        for name, pattern in _CRITICAL_CHECKS
        if not any(pattern.search(k) for k in keys)
    ]


def reflect_metrics_node(state: EarningsAgentState) -> EarningsAgentState:
    """Check extracted metrics and decide whether a retry pass is needed.

    Uses deterministic Python pattern matching instead of an LLM call —
    saves ~25 s per run and eliminates false positives from alias confusion.
    """
    attempts = state.get("extraction_attempts", 0)
    metrics = state.get("metrics") or {}

    if attempts >= MAX_EXTRACTION_ATTEMPTS:
        logger.info(
            "Reflection skipped for %s — reached max attempts (%d)",
            state["ticker"],
            MAX_EXTRACTION_ATTEMPTS,
        )
        return {**state}

    missing = _check_critical_metrics(metrics)

    if not missing:
        logger.info(
            "Reflection: all critical metrics present for %s after %d attempt(s)",
            state["ticker"],
            attempts,
        )
        return {**state}

    logger.info(
        "Reflection: missing critical metrics for %s (attempt %d/%d): %s",
        state["ticker"],
        attempts,
        MAX_EXTRACTION_ATTEMPTS,
        missing,
    )
    notes = (
        "The previous extraction pass missed the following critical metrics. "
        "Search the document specifically for: " + ", ".join(missing) + ". "
        "Make sure to capture them with their exact labels."
    )
    return {**state, "extraction_notes": notes, "status": "text_extracted"}

