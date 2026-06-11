"""LLM response parsing, per-chunk result merging, and retry-scope helpers."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from earnings_agents.extraction.chunker import _SECTION_PRIORITY, _UNKNOWN_SECTION_PRIORITY
from earnings_agents.workflow_state import EarningsAgentState

logger = logging.getLogger(__name__)

# Keys whose values are percentages, per-share amounts, or non-dollar counts --
# excluded from the sanity check AND from scale multiplication.
# "gross margin" (standalone) is treated as a percentage; "gross profit" is the dollar form.
_PCT_OR_PER_SHARE_PATTERNS = re.compile(
    r"(%|percent|\bgrowth\b|\bratio\b|\beps\b|per share|\byield\b|\brate\b|\byoy\b|\bpct\b"
    r"|\bemployee\b|\bheadcount\b|basis points|percentage points"
    r"|\bgross margin\b|\boperating margin\b|\bnet margin\b|\bprofit margin\b|\bmargin\s*%"
    # XBRL-style per-share suffixes: "Per Basic Share", "Per Diluted Share", etc.
    r"|\bper\s+(?:basic|diluted|basic\s+and\s+diluted|common)\s+share\b"
    # Operational unit counts -- physical quantities, never dollar-scaled
    r"|\bproduction\b|\bdeliveries\b|\bdelivered\b"
    r"|(?:super)?charger.{0,12}(?:station|connector)"
    r"|\bstations?\b|\bconnectors?\b"
    r"|\bdays.{0,5}supply\b|\blease count\b"
    r"|\bactive\b.{0,20}\bsubscriptions?\b|\bfsd subscriptions?\b)",
    re.IGNORECASE,
)

# EPS denominators ("Number of shares used ..." / "Shares used in computing ...") contain
# "per share" in their key but ARE table-scaled quantities (millions or thousands of shares).
# This pattern overrides _PCT_OR_PER_SHARE_PATTERNS for those keys.
_SHARE_COUNT_PATTERN = re.compile(
    r"\bnumber of shares\b|\bshares used\b|\bweighted.{0,15}average.{0,15}shares\b",
    re.IGNORECASE,
)

# Raw share-count values larger than this are assumed already at full count
# (e.g. 14_673_278_000 already expanded), so don't re-multiply.
_SHARE_COUNT_RAW_MAX = 100_000_000  # 100 M shares in report units -> already full if exceeded

# Scale multipliers keyed by the __scale__ sentinel the LLM returns.
_SCALE_MULTIPLIERS: dict[str, int] = {
    "millions": 1_000_000,
    "thousands": 1_000,
    "billions": 1_000_000_000,
}

# Raw table values larger than this are assumed to already be full USD (e.g.
# a narrative value like 82_900_000_000) and won't be re-multiplied.
_TABLE_RAW_MAX = 10_000_000  # 10 M raw -> $10T if x1M -- implausible, so skip

# The implausibility ceiling above is scale-relative: a raw cell is treated as
# "already full USD" only when multiplying it would exceed roughly $10T. That
# ceiling equals _TABLE_RAW_MAX * millions-multiplier, so for a thousands-scale
# document the raw cap is 1000x higher (a $17.5B annual revenue legitimately
# appears as 17,561,101 in thousands and MUST still be scaled).
_IMPLAUSIBLE_ABS_USD = _TABLE_RAW_MAX * 1_000_000  # ~$10T absolute ceiling

# Minimum fraction of the largest revenue-like value that a dollar field must
# have to be considered plausible (filters out residual unscaled cells).
_MIN_DOLLAR_FRACTION = 0.001   # 0.1 % of revenue

# Major financial metrics that should always represent a significant share of
# revenue.  When their post-scale value falls below _MIN_DOLLAR_FRACTION we
# attempt a x1 000 scale correction (one tier up: millions -> billions, etc.)
# before discarding.  Non-major metrics (specific investing / financing line
# items) can be legitimately tiny and are kept as-is.
_MAJOR_METRIC_RX = re.compile(
    r"\brevenue\b|\bnet sales\b"
    r"|\bgross profit\b"
    r"|\boperating income\b|\boperating profit\b|\boperating loss\b"
    r"|\bebit\b|\bebitda\b"
    r"|\bnet income\b|\bnet earnings\b|\bnet loss\b",
    re.IGNORECASE,
)

# A x1 000 rescaled value is accepted only when it stays below this multiple
# of revenue -- guards against inflating genuinely-tiny items.
_RESCALE_UPPER_MULTIPLE = 3.0

_INCOME_STATEMENT_KEY_RX = re.compile(
    r"revenue|gross\s+profit|cost\s+of\s+(revenue|sales|goods\s+sold)|"
    r"operating\s+income|net\s+income|earnings\s+per\s+share|"
    r"interest\s+(income|expense)|income\s+tax|research\s+and\s+development|"
    r"selling,\s*general\s+and\s+administrative|operating\s+expenses",
    re.I,
)
_BALANCE_SHEET_KEY_RX = re.compile(
    r"total\s+assets|assets\b|liabilit|equity|shareholders'\s+equity|"
    r"stockholders'\s+equity|inventory|accounts\s+receivable|accounts\s+payable",
    re.I,
)
_CASH_FLOW_KEY_RX = re.compile(
    r"cash\s+flow|net\s+cash|operating\s+activities|investing\s+activities|"
    r"financing\s+activities|capital\s+expenditures|depreciation",
    re.I,
)


def _infer_section_for_metric_key(metric_key: str) -> str | None:
    """Infer likely statement section for a metric key.

    Used only for retry-scoping hints from analysis findings.
    """
    if _INCOME_STATEMENT_KEY_RX.search(metric_key):
        return "income_statement"
    if _BALANCE_SHEET_KEY_RX.search(metric_key):
        return "balance_sheet"
    if _CASH_FLOW_KEY_RX.search(metric_key):
        return "cash_flow"
    return None


def _find_flagged_chunk_indices(
    findings: list[dict],
    chunk_metric_sources: dict[str, list[int]],
) -> set[int]:
    """Return absolute chunk indices that produced high-severity flagged metrics.

    Looks up each flagged metric key in *chunk_metric_sources* (metric key ->
    list of absolute chunk indices from the previous pass) to find the exact
    chunks that contributed wrong values.  Only high/critical findings are
    considered -- medium/low issues do not warrant a targeted re-extract.

    Returns an empty set when no specific chunks can be identified (e.g., all
    flagged metrics are *missing* rather than wrong -- we don't know which chunk
    should have contained them; the caller falls back to section-level scoping
    for those).
    """
    if not chunk_metric_sources:
        return set()
    flagged: set[int] = set()
    for f in findings:
        if not isinstance(f, dict):
            continue
        if str(f.get("severity", "")).lower() not in ("high", "critical", "error"):
            continue
        for k in (f.get("keys") or []):
            if isinstance(k, str) and k in chunk_metric_sources:
                flagged.update(chunk_metric_sources[k])
    return flagged


def _infer_retry_sections(state: EarningsAgentState) -> set[str]:
    """Infer which statement sections should be re-extracted on retry passes.

    Returns an empty set when no scoped retry can be inferred, signalling that
    the caller should process all chunks.
    """
    sections: set[str] = set()
    findings = state.get("findings") or []
    for f in findings:
        if not isinstance(f, dict):
            continue
        if str(f.get("severity", "")).lower() != "high":
            continue

        finding_type = str(f.get("type", "")).lower()
        text_blob = " ".join(
            s
            for s in [
                str(f.get("message", "")),
                str(f.get("suggested_action", "")),
            ]
            if s
        ).lower()

        if finding_type == "identity_violation":
            if "balance" in text_blob:
                sections.add("balance_sheet")
            else:
                sections.add("income_statement")

        if "income statement" in text_blob:
            sections.add("income_statement")
        if "balance sheet" in text_blob:
            sections.add("balance_sheet")
        if "cash flow" in text_blob:
            sections.add("cash_flow")

        for k in f.get("keys") or []:
            if not isinstance(k, str):
                continue
            inferred = _infer_section_for_metric_key(k)
            if inferred:
                sections.add(inferred)

    # Fall back to extraction_notes when findings are absent or coarse.
    notes = str(state.get("extraction_notes") or "").lower()
    if "income statement" in notes:
        sections.add("income_statement")
    if "balance sheet" in notes:
        sections.add("balance_sheet")
    if "cash flow" in notes:
        sections.add("cash_flow")

    return sections


def _parse_llm_response(
    response: str,
    shares_multiplier: int = 1,
    prescan_dollar_multiplier: int = 0,
) -> dict[str, Any] | None:
    """Strip markdown fences, parse JSON, and apply the __scale__ multiplier.

    The LLM is asked to report raw table values plus a ``__scale__`` sentinel.
    Python applies the exact multiplication so the LLM never has to do arithmetic.

    ``shares_multiplier`` is applied to share-denominator fields (e.g. shares
    used in EPS calculation) and may differ from the dollar multiplier when the
    document explicitly states a different scale for share counts (e.g. Apple
    reports dollars in millions but share counts in thousands).

    ``prescan_dollar_multiplier``: when > 0, overrides the LLM's ``__scale__``
    for dollar fields.  This prevents hallucinated scale labels (e.g. "millions"
    when the header says "thousands") from corrupting the merge.
    """
    cleaned = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    brace = cleaned.find("{")
    if brace > 0:
        cleaned = cleaned[brace:]
    end_brace = cleaned.rfind("}")
    if end_brace >= 0:
        cleaned = cleaned[: end_brace + 1]
    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    # Extract __scale__ -- always pop it to keep the dict clean.
    scale_str = str(parsed.pop("__scale__", "as-is")).lower()
    # If the prescan detected the document scale from the header (e.g. "(In thousands)"),
    # trust that over the LLM's returned label to prevent hallucinated-scale corruption.
    if prescan_dollar_multiplier > 1:
        multiplier = prescan_dollar_multiplier
    else:
        multiplier = _SCALE_MULTIPLIERS.get(scale_str, 1)
    # Scale-relative raw cap: a cell is "already full USD" only if multiplying
    # it would breach the ~$10T absolute ceiling. For thousands scale this cap
    # is 1000x higher than the millions case, so multi-billion annual figures
    # (e.g. 17,561,101 thousands = $17.5B) are still scaled instead of skipped.
    table_raw_max = _IMPLAUSIBLE_ABS_USD // multiplier if multiplier > 1 else _TABLE_RAW_MAX
    if multiplier > 1 or shares_multiplier > 1:
        for k, v in list(parsed.items()):
            if v is None or not isinstance(v, (int, float)):
                continue
            is_share_count = bool(_SHARE_COUNT_PATTERN.search(k))
            if is_share_count and shares_multiplier > 1:
                # Share-denominator fields use the shares multiplier.
                # Skip the _TABLE_RAW_MAX guard -- share counts are often > 10 M
                # (e.g. Apple reports ~14.7 M thousands = 14.7 B shares).
                if abs(v) < _SHARE_COUNT_RAW_MAX:
                    parsed[k] = v * shares_multiplier
            elif (
                not is_share_count
                and not _PCT_OR_PER_SHARE_PATTERNS.search(k)
                and multiplier > 1
                and abs(v) < table_raw_max   # skip values already at full USD scale
            ):
                # Ambiguous 'gross margin' label: percentage (<= 100) vs dollar amount (> 100).
                # Skip scaling when the value is clearly already a percentage.
                if "gross margin" in k.lower() and abs(v) <= 100:
                    continue
                parsed[k] = v * multiplier

    return parsed


def _merge_metrics(
    results: list[dict[str, Any]],
    source_text: str = "",
    target_year: int | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Merge per-chunk extraction dicts into one dict of all discovered metrics.

    Strategy per key:
    - Numeric values: taken from the **highest-authority statement section**
      that reported the key, never averaged across sections.  A financial line
      item belongs to exactly one primary GAAP statement, so a value that
      appears in both the income-statement chunk and a supplementary
      ("other"/segment) chunk must come from the income statement -- averaging
      the two corrupts it (the historical "median of two = phantom average"
      bug).  *sections* is a list parallel to *results* giving each chunk's
      section key (``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``,
      ``"other"`` or ``"unknown"``); when absent (char-split / PDF), all chunks
      share equal authority and the prior median-of-values behaviour applies.
      Within the single winning section, the median is used as a defensive
      tie-breaker if that section was split across multiple chunks.
    - When *target_year* is supplied (from the SEC submissions API
      ``reportDate``), chunks whose ``__period__`` field embeds a *different*
      year are treated as stale (prior-year comparison column) and excluded
      from numeric merging; a stale value is still a last-resort fallback for
      keys absent from every on-target chunk.  Chunks with no ``__period__``
      are treated as on-target so they are never spuriously discarded.
    - String values: longest non-null wins (most descriptive period label,
      narrative text, etc.).
    - Dollar-amount fields that are implausibly small relative to the largest
      revenue-like value are discarded (unscaled table cells) after merging.
    """
    # Per-result section authority (lower = higher authority). Parallel to
    # *results*; defaults to equal/unknown authority when sections unavailable.
    if sections is not None and len(sections) == len(results):
        priorities = [
            _SECTION_PRIORITY.get(s, _UNKNOWN_SECTION_PRIORITY) for s in sections
        ]
    else:
        priorities = [_UNKNOWN_SECTION_PRIORITY] * len(results)
    paired: list[tuple[dict[str, Any], int]] = list(zip(results, priorities))

    # -- Period-year partitioning ----------------------------------------------
    # Split chunks into "on-target" (period year matches target_year, or unknown)
    # and "stale" (period year is a different year, e.g. prior-year column).
    def _chunk_period_year(chunk: dict[str, Any]) -> int | None:
        period = chunk.get("__period__")
        if not isinstance(period, str):
            return None
        m = re.search(r"\b(20\d{2}|19\d{2})\b", period)
        return int(m.group(1)) if m else None

    if target_year is not None:
        on_target: list[tuple[dict[str, Any], int]] = []
        stale: list[tuple[dict[str, Any], int]] = []
        for chunk, prio in paired:
            cy = _chunk_period_year(chunk)
            if cy is None or cy == target_year:
                on_target.append((chunk, prio))
            else:
                stale.append((chunk, prio))
        if stale:
            logger.info(
                "_merge_metrics: %d/%d chunk(s) declared stale period year "
                "(target=%d) -- excluded from numeric merge, kept as fallback",
                len(stale), len(results), target_year,
            )
    else:
        on_target = list(paired)
        stale = []

    # -- Pass 1: collect non-null values per key -------------------------------
    # Numeric values are collected as (value, section_priority) pairs so the
    # merge step can prefer the highest-authority section.
    def _collect(
        chunks: list[tuple[dict[str, Any], int]],
    ) -> tuple[dict[str, list[tuple[float, int]]], dict[str, list[str]]]:
        num: dict[str, list[tuple[float, int]]] = {}
        strs: dict[str, list[str]] = {}
        for result, prio in chunks:
            for key, value in result.items():
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    num.setdefault(key, []).append((float(value), prio))
                elif isinstance(value, str):
                    strs.setdefault(key, []).append(value)
        return num, strs

    num_on, str_on = _collect(on_target)
    num_stale, str_stale = _collect(stale)

    # Build final candidate sets: on-target wins; stale is fallback only.
    numeric_candidates: dict[str, list[tuple[float, int]]] = {}
    for key in set(num_on) | set(num_stale):
        if key in num_on:
            numeric_candidates[key] = num_on[key]
        else:
            logger.debug(
                "_merge_metrics: %r only in stale chunk(s) -- including as fallback",
                key,
            )
            numeric_candidates[key] = num_stale[key]

    string_candidates: dict[str, list[str]] = {}
    for key in set(str_on) | set(str_stale):
        string_candidates[key] = str_on.get(key) or str_stale.get(key, [])

    merged: dict[str, Any] = {}

    # Numeric: take the value from the highest-authority section that reported
    # the key (income statement > balance sheet > cash flow > other > unknown).
    # Values from lower-authority sections are NEVER averaged in -- a line item
    # has one true value in one statement. The median is only a tie-breaker
    # within the single winning section (defends against a split table).
    for key, pairs in numeric_candidates.items():
        best_prio = min(p for _, p in pairs)
        values = [v for v, p in pairs if p == best_prio]
        n = len(values)
        if n == 1:
            merged[key] = values[0]
        else:
            sorted_vals = sorted(values)
            lo, hi = sorted_vals[0], sorted_vals[-1]
            denom = max(abs(lo), abs(hi))
            # When values differ by >=5x the smallest, the low value(s) are
            # almost certainly footnote / amortization-breakdown artifacts
            # that share a key label with the primary P&L line (e.g. NVDA's
            # "(A) Acquisition-related costs in Cost of revenue: $47M" vs
            # the real "Cost of revenue: $20,458M").  Take the maximum.
            if abs(lo) > 0 and abs(hi) / abs(lo) >= 5.0:
                logger.warning(
                    "Metric %r: section authority %d -- values differ by %.1fx "
                    "%s; taking max %.6g (smaller value(s) likely footnote artifacts)",
                    key, best_prio, abs(hi) / abs(lo), sorted_vals, hi,
                )
                merged[key] = hi
            else:
                median = (
                    sorted_vals[n // 2]
                    if n % 2
                    else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
                )
                if denom > 0 and (hi - lo) / denom > 0.10:
                    logger.warning(
                        "Metric %r: section authority %d reported diverging values "
                        "%s; using median %.6g",
                        key, best_prio, sorted_vals, median,
                    )
                merged[key] = median

    # String: longest non-null wins (most descriptive wins, e.g. full period name).
    # Never clobbers a numeric result for the same key.
    # Exception: __period__ prefers the candidate whose embedded year is the
    # highest (most recent) -- "Three Months Ended April 27, 2025" is the
    # prior-year comparison period and is longer than "First Quarter Fiscal
    # 2027" but contains a stale year.
    def _period_year(s: str) -> int:
        m = re.search(r"\b(20\d{2}|19\d{2})\b", s)
        return int(m.group(1)) if m else 0

    for key, values in string_candidates.items():
        if key not in merged:
            if key == "__period__":
                merged[key] = max(values, key=_period_year)
            else:
                merged[key] = max(values, key=len)

    # Sanity check: find the largest "revenue"-labelled value as reference.
    revenue_ref = max(
        (v for k, v in merged.items()
         if v is not None and isinstance(v, (int, float))
         and "revenue" in k.lower()
         and not _PCT_OR_PER_SHARE_PATTERNS.search(k)),
        default=None,
    )

    if revenue_ref:
        threshold = revenue_ref * _MIN_DOLLAR_FRACTION
        rescale_upper = revenue_ref * _RESCALE_UPPER_MULTIPLE
        for key in list(merged.keys()):
            val = merged[key]
            if (
                val is None
                or not isinstance(val, (int, float))
                or _PCT_OR_PER_SHARE_PATTERNS.search(key)
                or abs(val) >= threshold
            ):
                continue  # value is fine as-is

            # Value is below the plausibility threshold.
            if _MAJOR_METRIC_RX.search(key):
                # Major metrics must be large.  Try a x1 000 scale correction
                # (one tier up, e.g. the LLM returned 74.9 in a billions table
                # instead of 74 900 in a millions table).
                rescaled = val * 1_000
                if threshold <= abs(rescaled) <= rescale_upper:
                    logger.debug(
                        "Scale-correcting major metric %r: %s -> %s",
                        key, val, rescaled,
                    )
                    merged[key] = rescaled
                else:
                    logger.warning(
                        "Discarding implausible major metric %r=%s "
                        "(< %.1f%% of revenue ref %s; x1000 rescale also fails)",
                        key, val, _MIN_DOLLAR_FRACTION * 100, revenue_ref,
                    )
                    merged[key] = None
            # Non-major metrics below threshold are kept as-is -- they can be
            # legitimately small (e.g. $26 M investing item for an $80 B company).

    # -- Case-duplicate dedup (stale-chunk fallback eviction) ------------------
    # When target_year filtering was active, some keys in *merged* came
    # exclusively from stale (prior-year) chunks as a last-resort fallback.
    # If a case-duplicate of that key also exists from on-target chunks, the
    # stale-only variant is wrong and must be dropped so it never reaches
    # concept mapping in normalize_data.
    #
    # Example: "Cost of revenue" = 20.458B (on-target) and
    #          "Cost of Revenue" = 48B     (stale fallback only)
    # -> drop "Cost of Revenue"; keep "Cost of revenue".
    #
    # Rule: for each group of keys that share the same lowercased+collapsed
    # form, if at least one came from on-target chunks AND at least one came
    # only from stale chunks, remove the stale-only keys.
    # When ALL keys in a group are stale-only (the metric genuinely only
    # appears in the comparison column), they are all retained as-is.
    if stale:
        def _nk(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().lower()

        # Which numeric keys came from on-target chunks?
        on_target_keys: set[str] = set(num_on)

        # Build groups: normalized_form -> [actual_key, ...]
        norm_groups: dict[str, list[str]] = {}
        for k in list(merged):
            norm_groups.setdefault(_nk(k), []).append(k)

        for group_keys in norm_groups.values():
            if len(group_keys) <= 1:
                continue
            ot = [k for k in group_keys if k in on_target_keys]
            stale_only = [k for k in group_keys if k not in on_target_keys]
            if ot and stale_only:
                for k in stale_only:
                    logger.debug(
                        "_merge_metrics: evicting stale-only case-duplicate %r "
                        "(on-target variant %r = %s retained)",
                        k, ot[0], merged.get(ot[0]),
                    )
                    merged.pop(k, None)

    # Drop keys where the final value is None to keep the stored document clean.
    # Note: duplicate / synonym folding is handled downstream by the
    # constrained LLM cleanup_metrics_node, not here. Keys are preserved
    # exactly as the LLM extracted them (matching company wording).
    cleaned = {k: v for k, v in merged.items() if v is not None}

    # Merge per-metric source snippets (the "show me" verification evidence).
    # Each chunk may return a ``__sources__`` dict mapping a metric label to the
    # verbatim text it read the value from. Prefer the snippet from the
    # highest-authority chunk (lowest priority number); on-target chunks are
    # iterated before stale ones so an equal-authority on-target snippet wins.
    source_snippets: dict[str, str] = {}
    best_snip_prio: dict[str, int] = {}
    for result, prio in (on_target + stale):
        raw_sources = result.get("__sources__")
        if not isinstance(raw_sources, dict):
            continue
        for label, snip in raw_sources.items():
            if not isinstance(snip, str) or not snip.strip():
                continue
            if label not in best_snip_prio or prio < best_snip_prio[label]:
                source_snippets[label] = snip
                best_snip_prio[label] = prio
    if source_snippets:
        cleaned["__sources__"] = source_snippets

    return cleaned



def _target_year_from_report_date(report_date_str: str | None) -> int | None:
    """Extract the calendar year from a ``sec_report_date`` ``'YYYY-MM-DD'`` string.

    Returns ``None`` when the string is absent or unparseable, which disables
    the period-year partition in ``_merge_metrics`` (all chunks treated equally).
    """
    if not report_date_str:
        return None
    try:
        from datetime import date as _d
        return _d.fromisoformat(report_date_str).year
    except ValueError:
        return None
