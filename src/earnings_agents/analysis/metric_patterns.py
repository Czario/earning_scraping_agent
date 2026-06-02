r"""Canonical compiled regex patterns for financial metric key matching.

All domain-specific patterns that identify financial metric names live here.
This centralises the pattern vocabulary so that ``findings.py``,
``validators.py``, and future callers share a single definition rather than
duplicating and potentially diverging.

Patterns are anchored (``^\s*``) where the string must appear at the start of
a metric key, and unanchored where mid-string matches are appropriate.

Usage::

    from earnings_agents.analysis.metric_patterns import REVENUE_RE, COGS_RE
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Balance-sheet identity patterns
# ---------------------------------------------------------------------------
BS_ASSETS_RE = re.compile(r"^\s*total assets\b", re.IGNORECASE)
BS_LIAB_RE = re.compile(r"^\s*total liabilities\s*$", re.IGNORECASE)
BS_EQUITY_RE = re.compile(
    r"total\s+(stockholders'?|shareholders'?)\s*equity", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Income-statement patterns
# ---------------------------------------------------------------------------
REVENUE_RE = re.compile(r"^\s*(total\s+)?revenues?\b", re.IGNORECASE)
COGS_RE = re.compile(
    r"^\s*(total\s+)?cost\s+of\s+(revenue|goods\s+sold|sales)\b", re.IGNORECASE
)
GROSS_PROFIT_RE = re.compile(r"^\s*gross\s+(profit|margin)\b", re.IGNORECASE)
OPINC_RE = re.compile(r"^\s*operating\s+income\b", re.IGNORECASE)
OPEX_TOTAL_RE = re.compile(r"^\s*total\s+operating\s+expenses\b", re.IGNORECASE)
OPEX_SUBTOTAL_RE = re.compile(r"^\s*operating\s+expenses\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Suspect-round heuristic constants and guard pattern
# ---------------------------------------------------------------------------
ROUND_UNIT: int = 100_000_000    # $100 M — minimum granularity to flag
ROUND_MIN: int = 100_000_000     # values below this are ignored (too small)
ROUND_MAX: int = 100_000_000_000  # values at or above this are ignored (mega-cap)

# Keys matching any of these concepts are never flagged as suspect-round
# (percentages, share counts, per-share values are legitimately round).
NEVER_ROUND_RE = re.compile(
    r"per share|eps|margin|ratio|rate|shares|weighted|%|percent",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# GAAP / Non-GAAP reconciliation leakage pattern
# ---------------------------------------------------------------------------
GAAP_NONGAAP_RE = re.compile(
    r"^(GAAP|Non-GAAP)\s+"         # prefixed with "GAAP " or "Non-GAAP "
    r"|non-gaap"                    # mid-key occurrence
    r"|\badjusted\b"               # "Adjusted operating income", etc.
    r"|\breconciliation\b"         # reconciliation table headers
    r"|impact of\b"                # "Total impact of non-GAAP adjustments"
    r"|\btax impact\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Composite / synonym-list key patterns
# ---------------------------------------------------------------------------
# SLASH: " / " never appears in legitimate GAAP labels.
COMPOSITE_SLASH_RE = re.compile(
    r"(?:diluted|basic|revenue|income|cost|expense|profit|earnings|shares|loss|eps)\b"
    r".{1,60}"
    r"\s+/\s+"
    r"(?:diluted|basic|revenue|income|cost|expense|profit|earnings|shares|loss|eps|net|per\s+share)",
    re.IGNORECASE,
)

# COMMA: only flag when the same financial keyword appears on both sides,
# which is the hallmark of an LLM copying a synonym bullet.
COMPOSITE_COMMA_RE = re.compile(
    r"\b(diluted|basic|revenue|income|cost|expense|profit|earnings|shares|loss|eps)\b"
    r"[^,]{1,80}"
    r",\s+"
    r"[^,]*\b\1\b",
    re.IGNORECASE,
)
