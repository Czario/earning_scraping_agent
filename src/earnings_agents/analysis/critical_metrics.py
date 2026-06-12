"""Tiered registry of income statement metrics for presence checking.

Extraction is scoped to the income statement only (balance sheet and cash-flow
items are intentionally excluded from LLM extraction). All three tiers reflect
that scope.

  TIER 1 — Core income statement lines. Must exist in every earnings release.
           Missing → re-extract (high-severity Finding).
  TIER 2 — Supporting income statement lines. Missing → informational hint;
           piggy-backs on a Tier-1 re-extract if one is triggered.
  TIER 3 — Optional / supplemental income statement lines. Tracked when
           present; never triggers re-extract.

A metric is considered "present" when at least one extracted key matches
its regex (case-insensitive). Patterns intentionally allow vendor-specific
wording (``Net sales``, ``Income from operations``, etc.).
"""
from __future__ import annotations

import re
from typing import Iterable

# Each entry: (display_name, compiled_pattern).
_R = lambda s: re.compile(s, re.IGNORECASE)  # noqa: E731

TIER1_REGISTRY: list[tuple[str, re.Pattern]] = [
    # Income statement — all must appear in every earnings release.
    # Balance sheet and cash-flow items are Tier-2: press-release summaries
    # often omit them, and their absence should never force a re-extract.
    ("Total Revenue",             _R(r"revenue|net sales|total revenue")),
    ("Gross Profit",              _R(r"gross profit|gross margin\b(?!.*%)")),
    ("Operating Income",          _R(r"operating income|operating profit|operating loss|income from operations")),
    ("Net Income",                _R(r"net income|net earnings|net loss")),
    ("Diluted EPS",               _R(r"diluted.*per share|diluted.*eps|\beps.*diluted\b|per share.*diluted")),
]

TIER2_REGISTRY: list[tuple[str, re.Pattern]] = [
    # Income statement supporting lines (balance sheet + cash flow excluded from extraction)
    ("Cost of Revenue",             _R(r"cost of (revenue|sales|goods)")),
    ("Total Operating Expenses",    _R(r"total operating expense|operating expenses")),
    ("Pre-tax Income",              _R(r"income before (income )?tax|pre-?tax (income|earnings)")),
    ("Income Tax Expense",          _R(r"provision for income tax|income tax expense|tax expense")),
    ("Basic EPS",                   _R(r"basic.*per share|basic.*eps|per share.*basic")),
    ("Weighted Avg Shares Diluted", _R(r"diluted.*(weighted|shares outstanding|shares used)|weighted.*shares.*diluted")),
    ("Weighted Avg Shares Basic",   _R(r"basic.*(weighted|shares outstanding|shares used)|weighted.*shares.*basic")),
    ("Interest Expense",            _R(r"interest expense")),
]

TIER3_REGISTRY: list[tuple[str, re.Pattern]] = [
    # Optional income statement / supplemental items
    ("R&D Expense",               _R(r"research and development|r&d expense")),
    ("Sales & Marketing",         _R(r"sales and marketing|selling.*marketing")),
    ("G&A",                       _R(r"general and administrative")),
    ("Comprehensive Income",      _R(r"comprehensive income")),
    ("Effective Tax Rate",        _R(r"effective tax rate")),
    ("Depreciation & Amortization", _R(r"depreciation (and|&) amortization")),
    ("Stock-Based Compensation",  _R(r"stock-?based compensation|share-?based compensation")),
    ("EBITDA",                    _R(r"\bebitda\b")),
    ("Dividends per Share",       _R(r"dividends? (declared )?per (common )?share|per share dividend")),
]


def _present(pattern: re.Pattern, keys: Iterable[str]) -> bool:
    return any(pattern.search(k) for k in keys)


def check_presence(metrics_keys: Iterable[str]) -> dict[str, list[str]]:
    """Classify presence of registry metrics in *metrics_keys*.

    Returns a dict::

        {
          "tier1_missing": [display_name, ...],
          "tier2_missing": [display_name, ...],
          "tier3_present": [display_name, ...],
        }

    Tier-3 returns the *present* list (not missing) — those are observations,
    not gaps.
    """
    keys = list(metrics_keys)
    return {
        "tier1_missing": [name for name, pat in TIER1_REGISTRY if not _present(pat, keys)],
        "tier2_missing": [name for name, pat in TIER2_REGISTRY if not _present(pat, keys)],
        "tier3_present": [name for name, pat in TIER3_REGISTRY if _present(pat, keys)],
    }
