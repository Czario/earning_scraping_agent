"""LLM-based HTML table classifier for earnings press releases.

Provides :func:`classify_other_tables_batch`, which asks the LLM to decide
whether unclassified ("other") tables from an earnings HTML document contain
useful primary GAAP financial data worth extracting.

Extracted from ``nodes/extract_html_text.py`` so the batch classification
logic and prompt can be tested independently of HTML parsing.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_OTHER_FILTER_PROMPT = """\
You are classifying tables from an earnings press release.

For each table below, decide whether it contains PRIMARY GAAP financial statement
data worth extracting — rows with numeric values for revenue, cost, expenses,
income, EPS, or share counts.

Mark useful=true for:
- Primary income-statement rows, condensed GAAP summaries.
- Revenue or sales breakdowns by segment, geography, product line, or business unit
  (these are GAAP supplementary disclosures — always keep them).

Mark useful=false for:
- Contact blocks (names, email addresses, phone numbers, job titles)
- Footnote annotation tables (rows beginning with (A), (B), * etc.)
- Supplementary stock-based compensation breakdowns
- Guidance / outlook tables
- Any other non-primary-statement content

Tables to classify:
{tables_block}

Return ONLY a JSON object mapping each table index (as a string) to true or false:
{{"0": true, "1": false, ...}}"""


def classify_other_tables_batch(
    candidates: list[tuple[str, str]],
    llm: Any,
) -> list[bool]:
    """Classify a batch of 'other' HTML tables in a single LLM call.

    Parameters
    ----------
    candidates:
        List of ``(table_text, context_text)`` pairs.  ``table_text`` is the
        raw table content; ``context_text`` is the text preceding the table in
        the document (heading, scale indicator, etc.).
    llm:
        An LLM client with an ``invoke(prompt) -> str`` method.

    Returns
    -------
    list[bool]
        Parallel list of keep flags — ``True`` = include, ``False`` = skip.
        Falls back to all-``True`` on any error so no table is silently dropped.
    """
    if not candidates:
        return []

    blocks: list[str] = []
    for i, (table_text, context_text) in enumerate(candidates):
        ctx = context_text[-300:].strip() or "(none)"
        blocks.append(f"Table {i}\nContext: {ctx}\nContent: {table_text[:600]}")

    prompt = _OTHER_FILTER_PROMPT.format(tables_block="\n\n".join(blocks))
    try:
        raw = llm.invoke(prompt)
        data = json.loads(raw)
        return [bool(data.get(str(i), True)) for i in range(len(candidates))]
    except Exception:  # noqa: BLE001
        return [True] * len(candidates)  # safe fallback — keep all
